import torch
import numpy as np
import structlog
from typing import List, Union, Optional
import tvm
from ..engine import (
    SamplingType,
    SamplingParams,
)

LOG = structlog.stdlib.get_logger(__name__)


def _apply_top_p_top_k(logits, top_ps_t, top_ks_t):
    dev = logits.device
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > top_ps_t.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=dev)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= top_ks_t.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(
        logits_sort,
        dim=-1,
        index=torch.argsort(logits_idx, dim=-1),
    )
    return logits


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
):
    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        # This allows us to do sampling with replacement by creating
        # num_samples copies of each row in the tensor, and then
        # batch sampling the resulting tensor.
        probs = (
            probs[:, None, :]
            .expand(probs.shape[0], num_samples, probs.shape[1])
            .contiguous()
            .view(-1, probs.shape[1])
        )
    q = torch.empty_like(probs).exponential_(1)
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)


def get_tensors_for_sampling(sampling_params, dtype, dev, vocab_size):
    mask_random_t = torch.tensor(
        [p.sampling_type == SamplingType.RANDOM for p in sampling_params],
        dtype=torch.bool,
    )
    mask_greedy_t = torch.logical_not(
        mask_random_t,
    )
    assert not mask_random_t.is_cuda
    assert not mask_greedy_t.is_cuda

    temperatures = []
    top_ps = []
    top_ks = []
    do_top_p = False
    do_top_k = False
    has_random = False
    has_greedy = False
    frequency_penalties = []
    presence_penalties = []
    rep_penalties = []
    logit_bias_indices = []
    logit_bias_values = []
    past_output_tokens = []
    for param in sampling_params:
        # Prepare temperature
        temperatures.append(param.temperature)
        if param.sampling_type == SamplingType.RANDOM:
            has_random |= True
            top_ps.append(param.top_p)
            top_ks.append(param.top_k if param.top_k != -1 else vocab_size)
            do_top_p |= top_ps[-1] < 1.0
            do_top_k |= top_ks[-1] != vocab_size
        else:
            has_greedy |= True

        frequency_penalties.append(param.frequency_penalty)
        presence_penalties.append(param.presence_penalty)
        assert param.repetition_penalty != 0
        rep_penalties.append(param.repetition_penalty)

        past_output_tokens.append(param.output_tokens)
        logit_bias_indices.append(param.logit_bias_index)
        logit_bias_values.append(param.logit_bias_value)

    temp_t = torch.tensor(
        temperatures,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    top_ps_t = torch.tensor(
        top_ps,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    top_ks_t = torch.tensor(
        top_ks,
        dtype=torch.int,
        device="cpu",
        pin_memory=True,
    )
    apply_top_p_top_k = do_top_p | do_top_k

    frequency_penalties_t = torch.tensor(
        frequency_penalties,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    presence_penalties_t = torch.tensor(
        frequency_penalties,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    repetition_penalties_t = torch.tensor(
        rep_penalties,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    past_output_tokens_t = torch.tensor(
        past_output_tokens,
        dtype=torch.long,
        device="cpu",
        pin_memory=True,
    )
    apply_penalty = True
    apply_bias = True
    logit_bias_indices_t = torch.tensor(
        logit_bias_indices,
        dtype=torch.long,
        device="cpu",
        pin_memory=True,
    )
    logit_bias_values_t = torch.tensor(
        logit_bias_values,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    return (
        has_random,
        has_greedy,
        mask_random_t,
        mask_greedy_t,
        temp_t.to(device=dev, non_blocking=True),
        apply_top_p_top_k,
        top_ps_t.to(device=dev, non_blocking=True),
        top_ks_t.to(device=dev, non_blocking=True),
        apply_penalty,
        frequency_penalties_t.to(device=dev, non_blocking=True),
        presence_penalties_t.to(device=dev, non_blocking=True),
        repetition_penalties_t.to(device=dev, non_blocking=True),
        apply_bias,
        logit_bias_indices_t.to(device=dev, non_blocking=True),
        logit_bias_values_t.to(device=dev, non_blocking=True),
        past_output_tokens_t.to(device=dev, non_blocking=True),
    )


def sample(
    sequence_ids,
    logits: Union[tvm.nd.NDArray, torch.Tensor],
    sampling_tensors,
    vocab_size,
    greedy_sampling_stream,
    random_sampling_stream,
    check_safety=False,
) -> Optional[np.ndarray]:
    def _is_safe_to_sample(prob_like):
        return (
            torch.sum(torch.isnan(prob_like) | torch.isinf(prob_like) | (prob_like < 0))
            == 0
        )

    logits = torch.from_dlpack(logits)
    (
        has_random,
        has_greedy,
        mask_random_t,
        mask_greedy_t,
        temp_t,
        apply_top_p_top_k,
        top_ps_t,
        top_ks_t,
        apply_penalty,
        frequency_penalties_t,
        presence_penalties_t,
        repetition_penalties_t,
        apply_bias,
        logit_bias_indices_t,
        logit_bias_values_t,
        past_output_tokens_t,
    ) = sampling_tensors

    assert logits.is_cuda
    assert frequency_penalties_t.is_cuda
    assert presence_penalties_t.is_cuda
    assert repetition_penalties_t.is_cuda

    # Adjust logits with in-place operations
    # TODO(vvchernov): need to strictly define order of using penalties and logit bias or
    # prohibit simultaneous using of them. At the latter case it can be LogitProcessor
    if apply_penalty:
        repetition_penalties_t = repetition_penalties_t[:, None].repeat(1, vocab_size)
        logits = torch.where(
            logits > 0, logits / repetition_penalties_t, logits * repetition_penalties_t
        )
        bs = len(sequence_ids)
        bin_counts = torch.zeros(
            (bs, vocab_size + 1), dtype=torch.long, device=logits.device
        )

        bin_counts.scatter_add_(
            1, past_output_tokens_t, torch.ones_like(past_output_tokens_t)
        )
        bin_counts = bin_counts[:, :vocab_size]
        mask = bin_counts > 0

        logits -= frequency_penalties_t.unsqueeze_(dim=1) * bin_counts
        logits -= presence_penalties_t.unsqueeze_(dim=1) * mask

    if apply_bias:
        logits.scatter_add_(
            1, logit_bias_indices_t, torch.ones_like(logit_bias_values_t)
        )
    res_greedy, res_random = np.array([]), np.array([])

    with torch.cuda.stream(greedy_sampling_stream):
        if has_greedy:
            logits_greedy = logits[mask_greedy_t]
            res_greedy = torch.argmax(logits_greedy, -1)
            res_greedy = res_greedy.to(torch.device("cpu"), non_blocking=True).numpy()

    with torch.cuda.stream(random_sampling_stream):
        if has_random:
            logits_random = logits[mask_random_t]
            # Further adjust logits with the factors related to random sampling
            logits_random.div_(temp_t[mask_random_t].unsqueeze(dim=1))
            if apply_top_p_top_k:
                logits = _apply_top_p_top_k(logits_random, top_ps_t, top_ks_t)

            probs = torch.softmax(logits_random, dim=-1)

            if check_safety and not _is_safe_to_sample(probs):
                return None

            res_random = _multinomial(probs, 1)[:, 0]
            res_random = res_random.to(torch.device("cpu"), non_blocking=True).numpy()

    torch.cuda.current_stream().wait_stream(greedy_sampling_stream)
    torch.cuda.current_stream().wait_stream(random_sampling_stream)

    # Prepare output
    sequence_ids = np.array(sequence_ids)
    sequence_ids_greedy = sequence_ids[mask_greedy_t.cpu().numpy()].tolist()
    sequence_ids_random = sequence_ids[mask_random_t.cpu().numpy()].tolist()
    new_sequence_ids = [
        *sequence_ids_greedy,
        *sequence_ids_random,
    ]
    next_tokens = [*res_greedy, *res_random]
    return zip(new_sequence_ids, next_tokens)
