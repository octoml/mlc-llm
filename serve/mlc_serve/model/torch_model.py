import math
import os
from typing import List, Union, Tuple

import structlog
import numpy as np
import torch
import tvm

from .base import ModelArtifactConfig
from .paged_cache_manager import KVCache, CacheManager
from .model_common import (
    sample,
    prepare_inputs,
    get_num_cache_blocks,
)

from ..engine import (
    SequenceId,
    PROMPT_SEQEUNCE_INDEX,
    get_prompt_sequence_id,
    MLCServeEngineConfig,
)
from ..engine.model_module import (
    DecodeRequest,
    PrefillRequest,
    TextGenerationResult,
    TextGenerator,
)

LOG = structlog.stdlib.get_logger(__name__)


class Model:
    def __init__(
        self,
        config,
    ):
        self.vocab_size = config.vocab_size
        self.sliding_window = config.sliding_window
        self.num_shards = config.num_shards

        if self.sliding_window:
            self.block_sliding_window = self.sliding_window // CacheManager.block_size
        else:
            self.block_sliding_window = None

    def get_used_memory(self):
        return 0

    def profile_memory_usage(self, seq_lens):
        return self.get_used_memory()

    def generate(
        self,
        requests: Union[List[PrefillRequest], List[DecodeRequest]],
        cache: KVCache,
    ) -> List[TextGenerationResult]:
        if len(requests) == 0:
            return []

        is_prefill = isinstance(requests[0], PrefillRequest)

        all_token_ids = []
        sampling_params = []
        sequence_ids = []
        prompt_lens = []
        num_sequences = []

        for request in requests:
            if isinstance(request, PrefillRequest):
                sequence_ids.append(get_prompt_sequence_id(request.request_id))
                num_sequences.append(request.num_sequence)
            else:
                sequence_ids.append(request.sequence_id)
                prompt_lens.append(request.prompt_token_counts)

            all_token_ids.append(request.token_ids)
            sampling_params.append(request.sampling_params)

        (
            input_ids,
            positions,
            seq_lens,
            slot_mapping,
            indices_within_window,
            block_tables,
        ) = prepare_inputs(
            sequence_ids,
            all_token_ids,
            prompt_lens,
            cache.slot_mappings,
            cache.decode_block_tables,
            self.sliding_window,
            is_prefill,
        )

        input_shape = input_ids.shape

        if is_prefill:
            torch.cuda.nvtx.range_push(f"forward prefill {input_shape}")
        else:
            torch.cuda.nvtx.range_push(f"forward decode {input_shape}")

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        logits = None

        try:
            next_tokens = sample(logits, sampling_params, self.vocab_size)
            assert next_tokens is not None
            outputs = []
            for i, (sequence_id, new_token) in enumerate(
                zip(sequence_ids, next_tokens)
            ):
                if not new_token in requests[i].sampling_params.appeared_tokens_freq:
                    requests[i].sampling_params.appeared_tokens_freq[new_token] = 0
                requests[i].sampling_params.appeared_tokens_freq[new_token] += 1
                if sequence_id.sequence_index == PROMPT_SEQEUNCE_INDEX:
                    for seq_id in range(num_sequences[i]):
                        outputs.append(
                            TextGenerationResult(
                                sequence_id=SequenceId(sequence_id.request_id, seq_id),
                                generated_tokens=[new_token],
                                error=None,
                            )
                        )
                else:
                    outputs.append(
                        TextGenerationResult(
                            sequence_id=sequence_id,
                            generated_tokens=[new_token],
                            error=None,
                        )
                    )

            return outputs
        except RuntimeError:
            # Fallback to per-token sampling in case some logits values are corrupted.
            outputs = []
            err_msg = (
                "Error from sampling: probability tensor contains either `inf`, `nan`"
                " or element < 0"
            )

            for i, (sequence_id, logits_per_token, sampling_param) in enumerate(
                zip(sequence_ids, torch.from_dlpack(logits), sampling_params)
            ):
                maybe_new_token = sample(
                    torch.unsqueeze(logits_per_token, 0),
                    [sampling_param],
                    self.vocab_size,
                    check_safety=True,
                )

                if maybe_new_token is not None:
                    new_token = maybe_new_token[0]
                    if (
                        not new_token
                        in requests[i].sampling_params.appeared_tokens_freq
                    ):
                        requests[i].sampling_params.appeared_tokens_freq[new_token] = 0
                    requests[i].sampling_params.appeared_tokens_freq[new_token] += 1
                    if sequence_id.sequence_index == PROMPT_SEQEUNCE_INDEX:
                        for seq_id in range(num_sequences[i]):
                            outputs.append(
                                TextGenerationResult(
                                    sequence_id=SequenceId(
                                        sequence_id.request_id, seq_id
                                    ),
                                    generated_tokens=[new_token],  # type: ignore
                                    error=None,
                                )
                            )
                    else:
                        outputs.append(
                            TextGenerationResult(
                                sequence_id=sequence_id,
                                generated_tokens=[new_token],  # type: ignore
                                error=None,
                            )
                        )
                else:
                    if sequence_id.sequence_index == PROMPT_SEQEUNCE_INDEX:
                        for seq_id in range(num_sequences[i]):
                            outputs.append(
                                TextGenerationResult(
                                    sequence_id=SequenceId(
                                        sequence_id.request_id, seq_id
                                    ),
                                    generated_tokens=[],
                                    error=err_msg,
                                )
                            )
                    else:
                        outputs.append(
                            TextGenerationResult(
                                sequence_id=sequence_id,
                                generated_tokens=[],
                                error=err_msg,
                            )
                        )

            return outputs


def init_torch_model(
    model_artifact_config: ModelArtifactConfig, engine_config: MLCServeEngineConfig
) -> Tuple[TextGenerator, CacheManager]:
    model = Model(model_artifact_config)

    num_kv_heads = (
        model_artifact_config.num_key_value_heads // model_artifact_config.num_shards
    )
    head_size = (
        model_artifact_config.hidden_size // model_artifact_config.num_attention_heads
    )

    if engine_config.max_num_batched_tokens > 0:
        LOG.info("Running memory profiling.")
        num_blocks = get_num_cache_blocks(
            model,
            [engine_config.max_input_len] * engine_config.max_num_sequences,
            model_artifact_config.num_hidden_layers,
            num_kv_heads,
            head_size,
        )
    else:
        num_blocks = 500

    num_cache_slots = num_blocks * CacheManager.block_size

    if num_cache_slots <= engine_config.max_num_batched_tokens:
        raise RuntimeError(
            f"max_num_batched_tokens = {engine_config.max_num_batched_tokens} but"
            f" only {num_blocks} cache blocks can be allocated. The number of"
            f" available cache slots is {num_cache_slots}, not enough for"
            f" {engine_config.max_num_batched_tokens} tokens. Try reducing"
            " --max_input_len or --max_num_sequences."
        )

    LOG.info(f"Using {num_blocks} cache blocks.")

    cache_blocks = None

    cache_manager = CacheManager(
        cache_blocks,
        num_blocks,
        model_artifact_config.sliding_window,
    )

    LOG.info("Allocated KV cache blocks.")

    # TODO(masahi): Make mypy understand that model confirms to TextGenerator Protocol.
    return model, cache_manager  # type: ignore
