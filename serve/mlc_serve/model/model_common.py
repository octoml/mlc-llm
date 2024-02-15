from typing import List, Optional, Union, Sequence

import structlog
import numpy as np
import torch
import tvm

from .paged_cache_manager import CacheManager
from ..engine import (
    get_prompt_sequence_id,
    PROMPT_SEQEUNCE_INDEX,
    SequenceId,
    RawLogprobsInfo,
)
from ..engine.model_module import (
    PrefillRequest,
    DecodeRequest,
    EvalMultiQueryRequest,
    RequestType,
    TextGenerationResult,
)
from .sampler import sample, adjust_logits, SamplingState, SamplingOutput


LOG = structlog.stdlib.get_logger(__name__)


def get_gpu_memory(gpu: int = 0) -> int:
    return torch.cuda.get_device_properties(gpu).total_memory


def get_num_cache_blocks(
    model,
    block_size,
    seq_lens,
    num_layers,
    num_kv_heads,
    head_size,
    gpu_memory_utilization=0.9,  # the default used by vllm
):
    used_memory_bytes = model.profile_memory_usage(seq_lens)
    cache_block_size = CacheManager.get_cache_block_size(
        block_size, num_layers, num_kv_heads, head_size
    )
    total_vram = get_gpu_memory()
    return int(
        (total_vram * gpu_memory_utilization - used_memory_bytes) // cache_block_size
    )


def prepare_textgen_result(
    request: RequestType,
    new_token: List[int],
    sequence_id: SequenceId,
    logprob_info: Optional[List[Optional[RawLogprobsInfo]]],
    err_msg: Optional[str] = None,
) -> List[TextGenerationResult]:
    outputs = []
    if sequence_id.sequence_index == PROMPT_SEQEUNCE_INDEX:
        assert isinstance(request, PrefillRequest)
        for seq_id in range(request.num_sequence):
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
    requests: Sequence[RequestType],
    sampling_state: SamplingState,
    vocab_size: int,
    copy_stream: torch.cuda.Stream,
    torch_dtype: torch.dtype,
    torch_dev: str,
    past_decode_tokens: List[List[int]],
) -> List[TextGenerationResult]:
    batch_size = logits.shape[0]
    assert batch_size == len(requests)
    # Convert to torch tensors if logits are in tvm ndarray
    if isinstance(logits, tvm.nd.NDArray):
        logits = torch.from_dlpack(logits)

    # synchronization point for sampling tensors
    # wait until all the tensors are loaded on GPU
    torch.cuda.current_stream().wait_stream(copy_stream)

    # Logit processing for constraint sampling e.g., JSON Mode
    for i, (sequence_id, request) in enumerate(zip(sequence_ids, requests)):
        if request.sampling_params.logits_processor is not None:
            cs_input_ids = (
                request.token_ids if isinstance(request, DecodeRequest) else []
            )
            logits[i] = request.sampling_params.logits_processor(
                sequence_id, cs_input_ids, logits[i]
            )

    logits = adjust_logits(logits, sampling_state, vocab_size)
    outputs: List[TextGenerationResult] = []

    try:
        sampling_output: Optional[SamplingOutput] = sample(
            logits,
            sampling_state,
        )

        for i, (new_token, logprob_info) in enumerate(
            zip(sampling_output.next_tokens, sampling_output.logprob_infos)
        ):
            sequence_id = sequence_ids[i]
            request = requests[i]
            outputs.extend(
                prepare_textgen_result(
                    request,
                    [new_token],
                    sequence_id,
                    [logprob_info] if logprob_info else None,
                )
            )
    except RuntimeError:
        # Fallback to per-token sampling in case some logits values are corrupted.
        err_msg = (
            "Error from sampling: probability tensor contains either `inf`, `nan`"
            " or element < 0"
        )
        logits = torch.from_dlpack(logits)
        for i in range(batch_size):
            sequence_id = sequence_ids[i]
            logits_per_token = logits[i]
            sampling_param = sampling_state.sampling_params[i]
            past_decode_tokens_per_request = past_decode_tokens[i]
            # NOTE: Rerun the preparation for simplicity.
            # Assume this code path is taken rarely and the recomputation overhead is
            # marginal.
            with torch.cuda.stream(copy_stream):
                new_sampling_state = SamplingState.from_sampling_params(
                    [sampling_param],
                    [past_decode_tokens_per_request],
                    torch_dtype,
                    torch_dev,
                    vocab_size,
                )
            torch.cuda.current_stream().wait_stream(copy_stream)
            maybe_sampling_output: Optional[SamplingOutput] = sample(
                torch.unsqueeze(logits_per_token, 0),
                new_sampling_state,
                check_safety=True,
            )

            new_token = maybe_sampling_output.next_tokens[0]
            logprob_info = maybe_sampling_output.logprob_infos[0]
            # Valid sample
            request = requests[i]
            if maybe_sampling_output is not None:
                outputs.extend(
                    prepare_textgen_result(
                        request,
                        [new_token],
                        sequence_id,
                        [logprob_info] if logprob_info else None,
                    )
                )
            else:
                outputs.extend(
                    prepare_textgen_result(
                        request,
                        [],
                        sequence_id,
                        None,
                        err_msg,
                    )
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
    num_decode_query_tokens=1,
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
            seq_len = prompt_lens[i] + len(token_ids)
            input_ids += token_ids[-num_decode_query_tokens:]

            for i in range(num_decode_query_tokens):
                positions.append(seq_len - (num_decode_query_tokens - i))

            slot_mapping += all_slot_mappings[sequence_id][-num_decode_query_tokens:]

            block_table = all_decode_block_tables[sequence_id]
            max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
            block_tables.append(block_table.get_blocks())

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
