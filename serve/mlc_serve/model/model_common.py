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
    used_memory_bytes,
    block_size,
    num_layers,
    num_kv_heads,
    head_size,
    gpu_memory_utilization=0.9,  # the default used by vllm
):
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
    torch_dtype: torch.dtype,
    torch_dev: str,
    past_decode_tokens: List[List[int]],
) -> List[TextGenerationResult]:
    batch_size = logits.shape[0]
    assert batch_size == len(requests)

    # Convert to torch tensors if logits are in tvm ndarray
    if isinstance(logits, tvm.nd.NDArray):
        logits = torch.from_dlpack(logits)

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
            new_sampling_state = SamplingState.from_sampling_params(
                [sampling_param],
                [past_decode_tokens_per_request],
                torch_dtype,
                torch_dev,
                vocab_size,
            )
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
    block_size,
    num_decode_query_tokens=1,
    for_vllm=False,
):
    if for_vllm:
        torch_int_dtype = torch.long
    else:
        torch_int_dtype = torch.int

    block_tables = []
    seq_lens = []
    input_ids = []
    slot_mapping = []
    positions = []
    indices_within_window = []
    start_idx = 0
    max_prompt_len = -1
    max_context_len = -1

    for i, (sequence_id, token_ids) in enumerate(zip(sequence_ids, all_token_ids)):
        if is_prefill:
            prompt_len = len(token_ids)
            seq_lens.append(prompt_len)
            max_prompt_len = max(max_prompt_len, prompt_len)

            if for_vllm:
                input_ids.append(token_ids)
                positions.append(list(range(prompt_len)))
                slot_mapping.append(all_slot_mappings[sequence_id])
            else:
                input_ids += token_ids
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

            if for_vllm:
                assert num_decode_query_tokens == 1
                input_ids.append([token_ids[-1]])
                positions.append([seq_len - 1])
                slot_mapping.append([all_slot_mappings[sequence_id][-1]])
            else:
                input_ids += token_ids[-num_decode_query_tokens:]

                for i in range(num_decode_query_tokens):
                    positions.append(seq_len - (num_decode_query_tokens - i))

                slot_mapping += all_slot_mappings[sequence_id][-num_decode_query_tokens:]

            block_table = all_decode_block_tables[sequence_id]
            block_tables.append(block_table.get_blocks())

            if sliding_window:
                seq_lens.append(min(seq_len, sliding_window))
            else:
                seq_lens.append(seq_len)

            max_context_len = max(max_context_len, seq_lens[-1])

    def _do_pad(
        x: List[List[int]],
        max_len: int,
        pad_val: int,
    ) -> List[List[int]]:
        def _pad_to_max(x: List[int], max_len: int, pad_val: int) -> List[int]:
            assert len(x) <= max_len
            return x + [pad_val] * (max_len - len(x))

        return [_pad_to_max(x_i, max_len, pad_val) for x_i in x]

    if for_vllm and is_prefill:
        input_ids = _do_pad(input_ids, max_prompt_len, 0)
        positions = _do_pad(positions, max_prompt_len, 0)
        slot_mapping = _do_pad(slot_mapping, max_prompt_len, -1)

    def to_torch(arr, torch_dtype):
        return torch.tensor(arr, dtype=torch_dtype, device="cuda")

    input_ids = to_torch(input_ids, torch_int_dtype)
    positions = to_torch(positions, torch_int_dtype)
    seq_lens = to_torch(seq_lens, torch.int)
    slot_mapping = to_torch(slot_mapping, torch_int_dtype)

    if is_prefill and sliding_window:
        indices_within_window = to_torch(indices_within_window, torch.int)
    else:
        indices_within_window = None

    if not is_prefill:
        max_block_table_len = (
            max_context_len + block_size - 1
        ) // block_size

        padded_block_tables = _do_pad(block_tables, max_block_table_len, 0)
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
