import torch
import numpy as np
import structlog
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ..engine import (
    SamplingParams,
    SamplingType,
    SAMPLING_EPS,
    LOGPROB_TOP_K_MAX,
    RawLogprobsInfo,
)

LOG = structlog.stdlib.get_logger(__name__)


def _apply_top_p_top_k(
    logits: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor
):
    # TODO(@team): Check the ordering. We currently apply top-p -> top-k.
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= top_ks.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort, dim=-1, index=torch.argsort(logits_idx, dim=-1))
    return logits


@dataclass
class SamplingTensors:
    """
    Sampling states prepared for sampling computations (`adjust_logits()` and `sample()`).
    We keep mask tensors in CPU since putting them on GPU showed small performance regression.
    Args:
        mask_random: torch.Tensor
            Mask for requests with random sampling.
            shape: (batch_size, )
        mask_greedy: torch.Tensor
            Mask for requests with greedy sampling.
            shape: (batch_size, )
        mask_top_logprob: torch.Tensor
            Mask for requests with top_logprob.
            shape: (LOGPROB_TOP_K_MAX) + 1, batch_size,)
        mask_prompt: torch.Tensor
            Mask for request with repetition penalty (prompt part)
            shape: (batch_size, vocab_size)
        temperatures: torch.Tensor
            Tensor for temperature values
            shape: (batch_size, )
        top_ps: torch.Tensor
            Tensor for top-p values
            shape: (batch_size, )
        top_ks: torch.Tensor
            Tensor for top-k values
            shape: (batch_size, )
        frequency_penalties: torch.Tensor
            Tensor for frequency penalty values
            shape: (batch_size, )
        presence_penalties: torch.Tensor
            Tensor for presence penalty values
            shape: (batch_size, )
        repetition_penalties: torch.Tensor
            Tensor for repetition penalty values
            shape: (batch_size, )
        logit_bias_indices: torch.Tensor
            Tensor for indices of logit bias
            shape: (num_logit_bias_pairs, )
        logit_bias_values: torch.Tensor
            Tensor for values of logit bias
            shape: (num_logit_bias_pairs, )
        past_output_tokens: torch.Tensor
            Tensor for generated tokens
            shape: (batch_size, max_num_gen_tokens,)
    """

    mask_random: torch.Tensor
    mask_greedy: torch.Tensor
    mask_top_logprob: torch.Tensor
    mask_prompt: torch.Tensor
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    logit_bias_indices: torch.Tensor
    logit_bias_values: torch.Tensor
    past_output_tokens: torch.Tensor

    @classmethod
    def from_lists(
        cls,
        dtype,
        dev,
        list_mask_random: List[bool],
        list_mask_top_logprob: List[List[bool]],
        list_mask_prompt: List[torch.Tensor],
        list_temperatures: List[float],
        list_top_ps: List[float],
        list_top_ks: List[int],
        list_frequency_penalties: List[float],
        list_presence_penalties: List[float],
        list_repetition_penalties: List[float],
        list_logit_bias_indices: List[List[int]],
        list_logit_bias_values: List[List[float]],
        list_past_output_tokens: List[List[int]],
    ):
        # NOTE: Keep `mask_random` and `mask_greedy` tensors in CPU.
        #       Moving them to gpu showed a small performance regression.
        mask_random = torch.tensor(
            list_mask_random,
            dtype=torch.bool,
            device="cpu",
        )
        mask_greedy = torch.logical_not(
            mask_random,
        )
        # `mask_top_logprob` will be on cpu
        mask_top_logprob = torch.from_numpy(list_mask_top_logprob)
        mask_prompt = torch.stack(list_mask_prompt)
        temp = torch.tensor(
            list_temperatures,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        top_ps = torch.tensor(
            list_top_ps,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        top_ks = torch.tensor(
            list_top_ks,
            dtype=torch.int,
            device="cpu",
            pin_memory=True,
        )
        frequency_penalties = torch.tensor(
            list_frequency_penalties,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        presence_penalties = torch.tensor(
            list_presence_penalties,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        repetition_penalties = torch.tensor(
            list_repetition_penalties,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        logit_bias_indices = torch.tensor(
            list_logit_bias_indices,
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        # Convert 1-based index to 0-based
        logit_bias_indices -= 1
        logit_bias_values = torch.tensor(
            list_logit_bias_values,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        past_output_tokens = torch.tensor(
            list_past_output_tokens,
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )

        return cls(
            mask_random,
            mask_greedy,
            mask_top_logprob,
            mask_prompt,
            temp.to(device=dev, non_blocking=True),
            top_ps.to(device=dev, non_blocking=True),
            top_ks.to(device=dev, non_blocking=True),
            frequency_penalties.to(device=dev, non_blocking=True),
            presence_penalties.to(device=dev, non_blocking=True),
            repetition_penalties.to(device=dev, non_blocking=True),
            logit_bias_indices.to(device=dev, non_blocking=True),
            logit_bias_values.to(device=dev, non_blocking=True),
            past_output_tokens.to(device=dev, non_blocking=True),
        )


@dataclass
class SamplingState:
    """
    Sampling states prepared for sampling computations (`adjust_logits()` and `sample()`).

    Args:
        has_random: bool
            True if the current batch contains a request (or requests)
            with random sampling.
        has_greedy: bool
            True if the current batch contains a request (or requests)
            with greedy sampling.
        apply_top_p_top_k: bool
            True if the current batch contains a request (or requests)
            with top-p or top-k.
        apply_penalty: bool
            True if the current batch contains a request (or requests)
            with at least one of the repetition/frequency/presence penalties.
        apply_bias: bool
            True if the current batch contains a request (or requests)
            with logit bias
        has_logprob: bool
            True if the current batch contains a request (or requests)
            with logprob
        logprob_batch_indices: List[int]
            A list of indices of the requests with logprob inside the batch
        sampling_tensors: SamplingTensors
            A set of torch tensors that contains masks and parameter
            values for sampling computation
        sampling_params: List[SamplingParams]
            A list of SamplingParams from the user request
    """

    has_random: bool
    has_greedy: bool
    apply_top_p_top_k: bool
    apply_penalty: bool
    apply_bias: bool
    has_logprob: bool
    logprob_batch_indices: List[int]
    sampling_tensors: SamplingTensors
    sampling_params: List[SamplingParams]

    @classmethod
    def from_sampling_params(
        cls,
        sampling_params: List[SamplingParams],
        list_past_output_tokens: List[List[int]],
        dtype: torch.dtype,
        dev: str,
        vocab_size: int,
    ):
        list_mask_random = []
        list_mask_prompt = []
        list_temperatures = []
        list_top_ps = []
        list_top_ks = []
        do_top_p = False
        do_top_k = False
        apply_penalty = False
        apply_bias = False
        list_frequency_penalties = []
        list_presence_penalties = []
        list_repetition_penalties = []
        list_logit_bias_indices = []
        list_logit_bias_values = []

        idx_random = -1
        idx_greedy = -1
        batch_size = len(sampling_params)
        # index 0 is for non-logprob requests
        has_logprob = False
        logprob_batch_indices = []
        list_mask_top_logprob = np.full(
            ((LOGPROB_TOP_K_MAX) + 1, batch_size), False, dtype=bool
        )
        logit_bias_maxlen = 0
        for batch_idx, param in enumerate(sampling_params):
            # Prepare temperature
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            list_temperatures.append(
                param.temperature if param.temperature >= SAMPLING_EPS else 1.0
            )

            if param.sampling_type == SamplingType.RANDOM:
                list_mask_random.append(True)
                idx_random += 1
                list_top_ps.append(param.top_p)
                list_top_ks.append(param.top_k if param.top_k != -1 else vocab_size)
                do_top_p |= list_top_ps[-1] < 1.0 - SAMPLING_EPS
                do_top_k |= list_top_ks[-1] != vocab_size
            else:
                list_mask_random.append(False)
                idx_greedy += 1

            if param.logprobs:
                logprob_batch_indices.append(batch_idx)
                # param.top_logprobs is zero if logprob is not used
                list_mask_top_logprob[param.top_logprobs][batch_idx] = param.logprobs
                has_logprob |= True

            apply_penalty |= (
                abs(param.presence_penalty) >= SAMPLING_EPS
                or abs(param.frequency_penalty) >= SAMPLING_EPS
                or abs(param.repetition_penalty - 1.0) >= SAMPLING_EPS
            )
            list_frequency_penalties.append(param.frequency_penalty)
            list_presence_penalties.append(param.presence_penalty)
            list_repetition_penalties.append(param.repetition_penalty)
            list_mask_prompt.append(param.mask_prompt)

            if param.logit_bias_index:
                assert param.logit_bias_value
                apply_bias |= True
                logit_bias_maxlen = max(logit_bias_maxlen, len(param.logit_bias_index))
                list_logit_bias_indices.append(param.logit_bias_index)
                list_logit_bias_values.append(param.logit_bias_value)
            else:
                list_logit_bias_indices.append([])

        num_random_samples = idx_random + 1
        num_greedy_samples = idx_greedy + 1

        has_random = num_random_samples > 0
        has_greedy = num_greedy_samples > 0
        apply_top_p_top_k = do_top_p | do_top_k

        if apply_bias:
            # Match the length of each request by padding
            for ii in range(batch_size):
                logit_bias_values = list_logit_bias_values[ii]
                num_padding = logit_bias_maxlen - len(logit_bias_values)
                # arbitrary index
                list_logit_bias_indices[ii] += [1] * num_padding
                list_logit_bias_values[ii] += [0] * num_padding

        max_num_past_tokens = 0
        for past_output_tokens in list_past_output_tokens:
            max_num_past_tokens = max(max_num_past_tokens, len(past_output_tokens))

        for i in range(batch_size):
            num = len(list_past_output_tokens[i])
            list_past_output_tokens[i] = list_past_output_tokens[i] + [vocab_size] * (
                max_num_past_tokens - num
            )

        sampling_tensors = SamplingTensors.from_lists(
            dtype,
            dev,
            list_mask_random,
            list_mask_top_logprob,
            list_mask_prompt,
            list_temperatures,
            list_top_ps,
            list_top_ks,
            list_frequency_penalties,
            list_presence_penalties,
            list_repetition_penalties,
            list_logit_bias_indices,
            list_logit_bias_values,
            list_past_output_tokens,
        )

        return cls(
            has_random,
            has_greedy,
            apply_top_p_top_k,
            apply_penalty,
            apply_bias,
            has_logprob,
            logprob_batch_indices,
            sampling_tensors,
            sampling_params,
        )


def get_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def adjust_logits(
        logits: torch.Tensor,
        sampling_state: SamplingState,
        vocab_size: int):
    batch_size = logits.shape[0]
    (
        apply_top_p_top_k,
        apply_penalty,
        apply_bias,
        sampling_tensors,
    ) = (
        sampling_state.apply_top_p_top_k,
        sampling_state.apply_penalty,
        sampling_state.apply_bias,
        sampling_state.sampling_tensors,
    )
    (
        prompt_mask,
        temp_t,
        top_ps_t,
        top_ks_t,
        frequency_penalties_t,
        repetition_penalties_t,
        presence_penalties_t,
        past_output_tokens_t,
        logit_bias_indices_t,
        logit_bias_values_t,
    ) = (
        sampling_tensors.mask_prompt,
        sampling_tensors.temperatures,
        sampling_tensors.top_ps,
        sampling_tensors.top_ks,
        sampling_tensors.frequency_penalties,
        sampling_tensors.repetition_penalties,
        sampling_tensors.presence_penalties,
        sampling_tensors.past_output_tokens,
        sampling_tensors.logit_bias_indices,
        sampling_tensors.logit_bias_values,
    )

    # TODO(vvchernov): make sure we are applying various sampling params
    # (e.g., repetition penalty, frequency/presence penalty, logit bias, temperature...)
    # in the right order.
    if apply_penalty:
        bin_counts, output_mask = get_bin_counts_and_mask(
            past_output_tokens_t,
            vocab_size,
            batch_size,
        )

        # It was checked that vLLM and HF approaches for repetition penalty are the same
        # For calculation of it their combination is used (see references below)
        # Calculate repetition penalty use vLLM approach
        # https://github.com/vllm-project/vllm/blob/0580aab02ffe60fee50bddc80b787828eb233c44/vllm/model_executor/layers/sampler.py#L177
        # and RepetitionPenaltyLogitsProcessor approach from HF TGI API
        # https://github.com/huggingface/transformers/blob/de11e654c962d5b23eb53a4387cd637b01987491/src/transformers/generation/logits_process.py#L332C1-L339C22
        # where score is logits
        # https://github.com/huggingface/transformers/blob/de11e654c962d5b23eb53a4387cd637b01987491/src/transformers/generation/logits_process.py#L76C1-L78C92
        repetition_penalties_t = repetition_penalties_t[:, None].repeat(1, vocab_size)
        prompt_mask = prompt_mask.to(repetition_penalties_t.device)
        repetition_penalties_t[~(prompt_mask | output_mask)] = 1.0
        logits = torch.where(
            logits > 0, logits / repetition_penalties_t, logits * repetition_penalties_t
        )

        # Calculate frequency and presence penalties
        logits -= frequency_penalties_t.unsqueeze_(dim=1) * bin_counts
        logits -= presence_penalties_t.unsqueeze_(dim=1) * output_mask

    # Adjust temperature
    logits.div_(temp_t.unsqueeze(dim=1))
    if apply_top_p_top_k:
        logits = _apply_top_p_top_k(logits, top_ps_t, top_ks_t)

    if apply_bias:
        # logits.scatter_add_ performs the following computation:
        #   logit[i][index[i][j]] += src[i][j]
        #     where 0<=i<src.shape[0] && 0<=j<src.shape[1]
        logits.scatter_add_(dim=1, index=logit_bias_indices_t, src=logit_bias_values_t)
    return logits


@dataclass
class SamplingOutput:
    next_tokens: list[int]
    logprob_infos: list[Optional[RawLogprobsInfo]]


def sample(
    logits: torch.Tensor,
    sampling_state: SamplingState,
    check_safety: bool = False,
) -> SamplingOutput:
    def _is_safe_to_sample(prob_like):
        return (
            torch.sum(torch.isnan(prob_like) | torch.isinf(prob_like) | (prob_like < 0))
            == 0
        )

    res_greedy, res_random = None, None
    sampling_tensors = sampling_state.sampling_tensors

    batch_size = logits.shape[0]
    mask_greedy_t, mask_random_t = (
        sampling_tensors.mask_greedy,
        sampling_tensors.mask_random,
    )

    next_tokens = np.empty((batch_size,), dtype=np.int64)
    if sampling_state.has_greedy:
        res_greedy = torch.argmax(logits[mask_greedy_t], -1)
        np_mask_greedy = mask_greedy_t.cpu().numpy()
        next_tokens[np_mask_greedy] = res_greedy.cpu().numpy()

    probs_random = None
    if sampling_state.has_random:
        probs_random = torch.softmax(logits[mask_random_t], dim=-1)
        if check_safety and not _is_safe_to_sample(probs_random):
            return None
        res_random = torch.multinomial(probs_random, 1, True)[:, 0]
        np_mask_random = mask_random_t.cpu().numpy()
        next_tokens[np_mask_random] = res_random.cpu().numpy()

    logprob_infos: List[Optional[RawLogprobsInfo]] = [None] * batch_size
    if sampling_state.has_logprob:
        # If everything is random sampling, save one extra softmax
        if not sampling_state.has_greedy:
            assert probs_random is not None
            logprobs = torch.log(probs_random)
        else:
            logprobs = torch.log_softmax(logits, dim=-1)

        # Redudandant but vectorized
        extended_logprobs = logprobs.repeat((LOGPROB_TOP_K_MAX + 1, 1, 1))
        all_top_logprobs, all_top_tokens = torch.topk(
            extended_logprobs, k=LOGPROB_TOP_K_MAX, dim=-1, largest=True, sorted=True
        )
        mask = sampling_state.sampling_tensors.mask_top_logprob
        top_tokens = all_top_tokens[mask]
        top_logprobs = all_top_logprobs[mask]
        for idx, batch_idx in enumerate(sampling_state.logprob_batch_indices):
            next_token = next_tokens[batch_idx]
            assert sampling_state.sampling_params[batch_idx].logprobs
            top_k = sampling_state.sampling_params[batch_idx].top_logprobs
            logprob_infos[batch_idx] = RawLogprobsInfo(
                current_token_id=next_token,
                current_logprob=logprobs[batch_idx][next_token],
                top_token_ids=top_tokens[idx][:top_k],
                top_logprobs=top_logprobs[idx][:top_k],
            )

    return SamplingOutput(next_tokens, logprob_infos)
