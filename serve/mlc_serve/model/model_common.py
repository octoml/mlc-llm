from typing import List, Optional, Tuple, Union

import structlog
import numpy as np
import torch
import tvm

from .paged_cache_manager import CacheManager
from ..engine import (
    SamplingType,
    SamplingParams,
    get_prompt_sequence_id,
    LOGPROB_TOP_K_MAX,
    PROMPT_SEQEUNCE_INDEX,
    RawLogprobsInfo,
    RawLogprobsInfos,
    SequenceId,
)
from ..engine.model_module import (
    PrefillRequest,
    EvalMultiQueryRequest,
    RequestType,
    RequestsType,
    TextGenerationResult,
)


LOG = structlog.stdlib.get_logger(__name__)


def get_gpu_memory(gpu: int = 0) -> int:
    return torch.cuda.get_device_properties(gpu).total_memory


def get_num_cache_blocks(
    model,
    seq_lens,
    num_layers,
    num_kv_heads,
    head_size,
    gpu_memory_utilization=0.9,  # the default used by vllm
):
    used_memory_bytes = model.profile_memory_usage(seq_lens)
    cache_block_size = CacheManager.get_cache_block_size(
        num_layers, num_kv_heads, head_size
    )
    total_vram = get_gpu_memory()
    return int(
        (total_vram * gpu_memory_utilization - used_memory_bytes) // cache_block_size
    )


def get_logprob_infos(
    i: int,
    logprob_infos: Optional[RawLogprobsInfos],
) -> Optional[RawLogprobsInfos]:
    if logprob_infos is None or logprob_infos[i] is None:
        return None
    return [logprob_infos[i]]


def get_raw_logprob_info(
    logits,
    token_id,
    top_logprobs_num,
) -> RawLogprobsInfo:
    logprobs = torch.log_softmax(logits, dim=-1)
    res_logprob = logprobs[token_id]

    if top_logprobs_num == 0:
        top_logprobs = None
        top_tokens = None
    else:
        assert top_logprobs_num <= LOGPROB_TOP_K_MAX, "Invalid input top_logprobs"
        top_logprobs, top_tokens = torch.topk(
            logprobs, k=top_logprobs_num, dim=-1, largest=True, sorted=True
        )
        top_tokens = top_tokens.cpu().numpy()
        top_logprobs = top_logprobs.cpu().numpy()

    # Set to raw logprob info
    return RawLogprobsInfo(
        current_token_id=token_id,
        current_logprob=res_logprob,
        top_token_ids=top_tokens,
        top_logprobs=top_logprobs,
    )


def get_logprob_indices(
    sampling_params: List[SamplingParams],
    num_seq: int,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    lgp_inds_greedy: List[Tuple[int, int, int]] = []
    lgp_inds_random: List[Tuple[int, int, int]] = []

    g_ind = 0
    r_ind = 0
    for i in range(num_seq):
        sampling_param = sampling_params[i]
        if sampling_param.sampling_type == SamplingType.RANDOM:
            if sampling_param.logprobs:
                lgp_inds_random.append((i, r_ind, sampling_param.top_logprobs))
            r_ind = r_ind + 1
        else:
            if sampling_param.logprobs:
                lgp_inds_greedy.append((i, g_ind, sampling_param.top_logprobs))
            g_ind = g_ind + 1

    return lgp_inds_greedy, lgp_inds_random


def get_raw_logprob_infos(
    logprob_infos: RawLogprobsInfos,
    indices: List[Tuple[int, int, int]],
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> RawLogprobsInfos:
    for i, ind, top_logprobs in indices:
        logprob_infos[i] = get_raw_logprob_info(
            logits[ind],
            token_ids[ind],
            top_logprobs,
        )

    return logprob_infos


def check_logprob_infos(
    logprob_infos: RawLogprobsInfos,
) -> Optional[RawLogprobsInfos]:
    check = False
    for info in logprob_infos:
        if info is not None:
            check = True
            break
    if check:
        return logprob_infos
    return None


def _apply_top_p_top_k(logits, top_ps, top_ks):
    p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort, dim=-1, index=torch.argsort(logits_idx, dim=-1))
    return logits


def sample(
    logits: Union[tvm.nd.NDArray, torch.Tensor],
    sampling_params: List[SamplingParams],
    vocab_size: int,
    check_safety=False,
) -> Optional[Tuple[np.ndarray, Optional[RawLogprobsInfos]]]:
    def _is_safe_to_sample(prob_like):
        return (
            torch.sum(torch.isnan(prob_like) | torch.isinf(prob_like) | (prob_like < 0))
            == 0
        )

    torch.cuda.nvtx.range_push(f"sample {logits.shape}")
    logits = torch.from_dlpack(logits)
    num_seq = len(sampling_params)

    mask_random_cpu = torch.tensor(
        [p.sampling_type == SamplingType.RANDOM for p in sampling_params],
        dtype=torch.bool,
    )
    mask_greedy_cpu = torch.logical_not(mask_random_cpu)
    if logits.device == torch.device("cpu"):
        mask_random_dvc = mask_random_cpu
        mask_greedy_dvc = mask_greedy_cpu
    else:  # gpu
        mask_random_dvc = mask_random_cpu.to(logits.device)
        mask_greedy_dvc = mask_greedy_cpu.to(logits.device)

    logits_greedy = logits[mask_greedy_dvc]

    logprob_infos: RawLogprobsInfos = [None] * num_seq
    lgp_inds_greedy, lgp_inds_random = get_logprob_indices(
        sampling_params,
        num_seq,
    )

    if logits_greedy.shape[0] > 0:
        res_greedy = torch.argmax(logits_greedy, -1).cpu().numpy()

        logprob_infos = get_raw_logprob_infos(
            logprob_infos,
            lgp_inds_greedy,
            logits_greedy,
            res_greedy,
        )

        # Case when there's only greedy sampling
        if logits_greedy.shape[0] == num_seq:
            torch.cuda.nvtx.range_pop()
            return res_greedy, check_logprob_infos(logprob_infos)

    temperatures = []
    top_ps = []
    top_ks = []
    divide_by_temperature = False
    do_top_p = False
    do_top_k = False

    for i in range(num_seq):
        param = sampling_params[i]
        freq = param.appeared_tokens_freq

        if param.sampling_type == SamplingType.RANDOM:
            temperatures.append(param.temperature)
            top_ps.append(param.top_p)
            top_ks.append(param.top_k if param.top_k != -1 else vocab_size)

            divide_by_temperature |= temperatures[-1] != 1.0
            do_top_p |= top_ps[-1] < 1.0
            do_top_k |= top_ks[-1] != vocab_size

            # TODO(vvchernov): need to strictly define order of using penalties and logit bias or
            # prohibit simultaneous using of them. At the latter case it can be LogitProcessor
            if (
                not param.presence_penalty == 0.0 or not param.frequency_penalty == 0
            ) and bool(freq):
                index = torch.from_numpy(np.array(list(freq.keys()))).to(
                    device=logits.device
                )
                src = (
                    torch.from_numpy(np.array(list(freq.values())))
                    .type_as(logits)
                    .to(device=logits.device)
                )
                logits[i][index] -= (
                    src * param.frequency_penalty + param.presence_penalty
                )

            if not param.repetition_penalty == 1.0 and bool(freq):
                index = torch.from_numpy(np.array(list(freq.keys()))).to(
                    device=logits.device
                )
                logits[i][index] /= param.repetition_penalty

            if param.logit_bias:
                logits[i][param.logit_bias_index] += (
                    torch.Tensor(param.logit_bias_value)
                    .type_as(logits)
                    .to(device=logits.device)
                )

    logits_random = logits[mask_random_dvc]

    if divide_by_temperature:
        t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
        logits_random.div_(t.unsqueeze(dim=1))

    if do_top_p or do_top_k:
        logits_random = _apply_top_p_top_k(logits_random, top_ps, top_ks)

    probs = torch.softmax(logits_random, dim=-1)

    if check_safety and not _is_safe_to_sample(probs):
        torch.cuda.nvtx.range_pop()
        return None

    res_random = torch.multinomial(probs, 1, True)[:, 0].cpu().numpy()

    logprob_infos = get_raw_logprob_infos(
        logprob_infos,
        lgp_inds_random,
        logits_random,
        res_random,
    )

    # Case when there's only random sampling
    if logits_random.shape[0] == num_seq:
        torch.cuda.nvtx.range_pop()
        return res_random, check_logprob_infos(logprob_infos)

    res = np.empty((num_seq,), dtype=np.int32)
    res[mask_random_cpu] = res_random

    if logits_greedy.shape[0] > 0:
        res[mask_greedy_cpu] = res_greedy

    torch.cuda.nvtx.range_pop()
    return res, check_logprob_infos(logprob_infos)


def update_tokens_frequency(
    request: RequestType,
    new_token: int
):
    if not new_token in request.sampling_params.appeared_tokens_freq:
        request.sampling_params.appeared_tokens_freq[new_token] = 0
    request.sampling_params.appeared_tokens_freq[new_token] += 1


def append_text_gen_res(
    outputs: List[TextGenerationResult],
    request: RequestType,
    new_token: List[int],
    sequence_id: SequenceId,
    logprob_info: Optional[RawLogprobsInfos],
    err_msg: Optional[str]=None,
) -> List[TextGenerationResult]:
    if sequence_id.sequence_index == PROMPT_SEQEUNCE_INDEX:
        assert isinstance(request, PrefillRequest)
        for seq_id in range(request.num_sequence):  # type: ignore
            outputs.append(
                TextGenerationResult(
                    sequence_id=SequenceId(sequence_id.request_id, seq_id),
                    generated_tokens=new_token,
                    error=err_msg,
                    logprob_info=logprob_info,
                )
            )
    else:
        outputs.append(
            TextGenerationResult(
                sequence_id=sequence_id,
                generated_tokens=new_token,
                error=err_msg,
                logprob_info=logprob_info,
            )
        )
    return outputs


def sample_from_logits(
    logits: Union[tvm.nd.NDArray, torch.Tensor],
    sequence_ids: List[SequenceId],
    requests: RequestsType,
    vocab_size,
) -> List[TextGenerationResult]:
    assert logits.shape[0] == len(requests)

    sampling_params = [req.sampling_params for req in requests]
    outputs: List[TextGenerationResult] = []

    try:
        next_tokens, logprob_infos = sample(logits, sampling_params, vocab_size)
        assert next_tokens is not None
        for i, (sequence_id, new_token) in enumerate(zip(sequence_ids, next_tokens)):
            update_tokens_frequency(requests[i], new_token)
            outputs = append_text_gen_res(
                outputs,
                requests[i],
                [new_token],
                sequence_id,
                get_logprob_infos(i, logprob_infos),
            )

        return outputs
    except RuntimeError:
        # Fallback to per-token sampling in case some logits values are corrupted.
        err_msg = (
            "Error from sampling: probability tensor contains either `inf`, `nan`"
            " or element < 0"
        )

        for i, (sequence_id, logits_per_token, sampling_param) in enumerate(
            zip(sequence_ids, torch.from_dlpack(logits), sampling_params)
        ):
            maybe_new_token, logprob_infos = sample(
                torch.unsqueeze(logits_per_token, 0),
                [sampling_param],
                vocab_size,
                check_safety=True,
            )

            if maybe_new_token is not None:
                new_token = maybe_new_token[0]
                update_tokens_frequency(requests[i], new_token)
                outputs = append_text_gen_res(
                    outputs,
                    requests[i],
                    [new_token],
                    sequence_id,
                    get_logprob_infos(0, logprob_infos),
                )
            else:
                outputs = append_text_gen_res(
                    outputs,
                    requests[i],
                    [],  # new_token
                    sequence_id,
                    get_logprob_infos(0, logprob_infos),
                    err_msg,
                )

        return outputs


def prepare_inputs(
    sequence_ids,
    all_token_ids,
    prompt_lens,
    all_slot_mappings,
    all_decode_block_tables,
    sliding_window,
    is_prefill,
):
    block_tables = []
    seq_lens = []
    input_ids = []
    slot_mapping = []
    positions = []
    max_num_blocks_per_seq = 0
    indices_within_window = []
    start_idx = 0

    for i, (sequence_id, token_ids) in enumerate(zip(sequence_ids, all_token_ids)):
        if is_prefill:
            input_ids += token_ids
            prompt_len = len(token_ids)
            seq_lens.append(prompt_len)
            positions += range(prompt_len)
            slot_mapping += all_slot_mappings[sequence_id]

            if sliding_window:
                indices_within_window += range(
                    start_idx + max(0, prompt_len - sliding_window),
                    start_idx + prompt_len,
                )
                start_idx += prompt_len

        else:
            input_ids.append(token_ids[-1])
            seq_len = prompt_lens[i] + len(token_ids)
            positions.append(seq_len - 1)
            block_table = all_decode_block_tables[sequence_id]
            max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
            block_tables.append(block_table.get_blocks())
            slot_mapping.append(all_slot_mappings[sequence_id][-1])

            if sliding_window:
                seq_lens.append(min(seq_len, sliding_window))
            else:
                seq_lens.append(seq_len)

    def to_torch(arr, torch_dtype):
        return torch.tensor(arr, dtype=torch_dtype, device="cuda")

    input_ids = to_torch(input_ids, torch.int)
    positions = to_torch(positions, torch.int)
    seq_lens = to_torch(seq_lens, torch.int)
    slot_mapping = to_torch(slot_mapping, torch.int)

    if is_prefill and sliding_window:
        indices_within_window = to_torch(indices_within_window, torch.int)
    else:
        indices_within_window = None

    if not is_prefill:

        def _pad_to_max(x: List[int], max_len: int) -> List[int]:
            return x + [0] * (max_len - len(x))

        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in block_tables
        ]
        block_tables = to_torch(padded_block_tables, torch.int)
    else:
        block_tables = None

    return (
        input_ids,
        positions,
        seq_lens,
        slot_mapping,
        indices_within_window,
        block_tables,
    )


def prepare_multi_query_decode_inputs(
    requests: List[EvalMultiQueryRequest],
    all_slot_mappings,
    sliding_window,
    dev,
):
    seq_lens = []
    query_lens = []
    input_ids = []
    slot_mapping = []
    past_slot_mapping = []
    positions = []
    permute_map = []

    query_offset = sum([request.num_past_tokens for request in requests])
    past_offset = 0

    for request in requests:
        num_queries = request.queries.num_tokens
        query_lens.append(num_queries)
        input_ids += request.queries.token_ids
        positions += [request.num_past_tokens + i for i in range(num_queries)]

        prompt_seq_id = get_prompt_sequence_id(request.sequence_id.request_id)
        prompt_slot_mappings = all_slot_mappings[prompt_seq_id]

        if sliding_window and request.num_past_tokens + num_queries >= sliding_window:
            seq_lens.append(sliding_window)
            prompt_and_decode_slot_mappings = (
                prompt_slot_mappings + all_slot_mappings[request.sequence_id]
            )
            past_slot_mapping += prompt_and_decode_slot_mappings[
                request.num_past_tokens
                - (sliding_window - num_queries) : request.num_past_tokens
            ]
            slot_mapping += prompt_and_decode_slot_mappings[
                request.num_past_tokens : request.num_past_tokens + num_queries
            ]
        else:
            seq_lens.append(request.num_past_tokens + num_queries)

            if request.num_past_tokens < len(prompt_slot_mappings):
                raise RuntimeError(
                    "For EvalMultiQueryRequest, the number of past tokens"
                    "smaller than the prompt length is not supported for now."
                )
            elif request.num_past_tokens == len(prompt_slot_mappings):
                # The case for restoring an evicted parallel-sampling request
                past_slot_mapping += prompt_slot_mappings
                slot_mapping += all_slot_mappings[request.sequence_id][:num_queries]
            else:
                query_begin_offset = request.num_past_tokens - len(prompt_slot_mappings)
                past_slot_mapping += (
                    prompt_slot_mappings
                    + all_slot_mappings[request.sequence_id][:query_begin_offset]
                )
                slot_mapping += all_slot_mappings[request.sequence_id][
                    query_begin_offset : query_begin_offset + num_queries
                ]

        permute_map += list(
            range(past_offset, past_offset + request.num_past_tokens)
        ) + list(range(query_offset, query_offset + num_queries))

        query_offset += num_queries
        past_offset += request.num_past_tokens

    input_ids = tvm.nd.array(np.array(input_ids, dtype="int32"), dev)
    positions = tvm.nd.array(np.array(positions, dtype="int32"), dev)
    seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), dev)
    slot_mapping = tvm.nd.array(np.array(slot_mapping, dtype="int32"), dev)

    query_lens = tvm.nd.array(np.array(query_lens, dtype="int32"), dev)
    # TODO(masahi): These inputs need to be replaced by block_table when a proper attention kernel
    # becomes available.
    past_slot_mapping = tvm.nd.array(np.array(past_slot_mapping, dtype="int32"), dev)
    permute_map = tvm.nd.array(np.array(permute_map, dtype="int32"), dev)

    return (
        input_ids,
        positions,
        seq_lens,
        slot_mapping,
        query_lens,
        past_slot_mapping,
        permute_map,
    )
