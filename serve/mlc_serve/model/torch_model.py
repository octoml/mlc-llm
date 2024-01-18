import time
import os
from typing import List, Union, Tuple, Sequence
from collections import defaultdict

import structlog
import torch

from transformers import AutoConfig

from vllm.model_executor.layers.sampler import get_logits
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.sequence import SequenceData
from vllm.model_executor import InputMetadata
from vllm.sampling_params import SamplingParams

import torch.multiprocessing as multiprocessing

import rpyc
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from concurrent.futures import ThreadPoolExecutor

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
    SamplingParams as MLCSamplingParams,
)
from ..engine.model_module import (
    DecodeRequest,
    PrefillRequest,
    TextGenerationResult,
    TextGenerator,
)

LOG = structlog.stdlib.get_logger(__name__)


def convert_sampling_params(mlc_params: MLCSamplingParams) -> SamplingParams:
    return SamplingParams(
        presence_penalty=mlc_params.presence_penalty,
        frequency_penalty=mlc_params.frequency_penalty,
        temperature=mlc_params.temperature,
        top_p=mlc_params.top_p,
        top_k=mlc_params.top_k,
    )


def init_cache_blocks(head_size, num_layers, num_heads, block_size, num_gpu_blocks):
    element_size = 2
    x = 16 // element_size

    key_block_shape = (num_heads, head_size // x, block_size, x)
    value_block_shape = (num_heads, head_size, block_size)

    gpu_cache = []
    for _ in range(num_layers):
        key_blocks = torch.empty(
            size=(num_gpu_blocks, *key_block_shape),
            dtype=torch.float16,
            device="cuda",
        )
        value_blocks = torch.empty(
            size=(num_gpu_blocks, *value_block_shape),
            dtype=torch.float16,
            device="cuda",
        )
        gpu_cache.append((key_blocks, value_blocks))
    return gpu_cache


class ModelRpcServer(rpyc.Service):
    def exposed_init_model(
        self, tp_rank, num_shards, model_path, hf_config, engine_config
    ):
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        torch.cuda.set_device(tp_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=num_shards,
            rank=tp_rank,
            # init_method=f"tcp://127.0.0.1:{self.nccl_port}",
        )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cuda())

        with torch.device("cuda"):
            torch.set_default_dtype(torch.float16)
            pt_model = LlamaForCausalLM(hf_config)
            pt_model.load_weights(model_path, None, "auto", None)

        num_kv_heads = hf_config.num_key_value_heads // num_shards
        head_size = hf_config.hidden_size // hf_config.num_attention_heads

        if engine_config.max_num_batched_tokens > 0:
            LOG.info("Running memory profiling.")
            seq_lens = (
                [engine_config.max_input_len] * engine_config.max_num_sequences,
            )
            used_memory_bytes = self.profile_memory_usage(seq_lens)

            num_blocks = get_num_cache_blocks(
                used_memory_bytes,
                hf_config.num_hidden_layers,
                num_kv_heads,
                head_size,
            )
        else:
            num_blocks = 500

        self.cache_blocks = init_cache_blocks(
            head_size,
            hf_config.num_hidden_layers,
            hf_config.num_attention_heads,
            CacheManager.block_size,
            num_blocks,
        )

        return num_blocks

    def profile_memory_usage(self, seq_lens):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        sampling_params = SamplingParams(top_p=0.99)

        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        seq_data = {}
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        for i, seq_len in enumerate(seq_lens):
            seq_groups.append(([i], sampling_params))
            prompt_tokens = [0] * seq_len
            seq_data[i] = SequenceData(prompt_tokens)

            input_tokens.extend(prompt_tokens)
            input_positions.extend(range(seq_len))
            slot_mapping.extend([0] * seq_len)

        input_ids = torch.cuda.LongTensor(input_tokens)
        positions = torch.cuda.LongTensor(input_positions)
        slot_mapping_tensor = torch.cuda.IntTensor(slot_mapping)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=seq_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=torch.cuda.IntTensor([]),
            max_context_len=0,
            block_tables=torch.cuda.IntTensor([]),
        )

        kv_caches = [(None, None)] * self.num_hidden_layers

        with torch.no_grad():
            self.pt_model.forward(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
                cache_events=None,
            )

        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated()
        print("peak memory", peak_memory / 1e9)

        torch.cuda.empty_cache()

        return peak_memory

    def exposed_generate(
        self, requests: Sequence[Union[PrefillRequest, DecodeRequest]], cache: KVCache
    ) -> List[TextGenerationResult]:
        requests = obtain(requests)
        cache = obtain(cache)

        if len(requests) == 0:
            return []

        is_prefill = isinstance(requests[0], PrefillRequest)

        all_token_ids = []
        sampling_params = []
        sequence_ids = []
        prompt_lens = []
        num_sequences = []
        seq_data = {}
        seq_group_sequence_ids = defaultdict(list)
        seq_group_sampling_params = {}

        for request in requests:
            if isinstance(request, PrefillRequest):
                sequence_ids.append(get_prompt_sequence_id(request.request_id))
                num_sequences.append(request.num_sequence)
                prompt_lens.append(len(request.token_ids))
                seq_group_sequence_ids[request.request_id].append(sequence_ids[-1])
                seq_group_sampling_params[request.request_id] = convert_sampling_params(
                    request.sampling_params
                )
            else:
                sequence_ids.append(request.sequence_id)
                prompt_lens.append(request.prompt_token_counts)
                req_id = request.sequence_id.request_id
                seq_group_sequence_ids[req_id].append(request.sequence_id)
                seq_group_sampling_params[req_id] = convert_sampling_params(
                    request.sampling_params
                )

            all_token_ids.append(request.token_ids)
            sampling_params.append(request.sampling_params)

            seq_data[sequence_ids[-1]] = SequenceData(request.token_ids)

        seq_groups: List[Tuple[List[SequenceId], SamplingParams]] = []

        for req_id, seq_ids in seq_group_sequence_ids.items():
            seq_groups.append((seq_ids, seq_group_sampling_params[req_id]))

        (
            input_ids,
            positions,
            seq_lens,
            slot_mapping,
            _,
            block_tables,
        ) = prepare_inputs(
            sequence_ids,
            all_token_ids,
            prompt_lens,
            cache.slot_mappings,
            cache.decode_block_tables,
            self.sliding_window,
            is_prefill,
            torch.long,
            align=8,
        )

        input_shape = input_ids.shape

        if block_tables is None:
            torch.cuda.nvtx.range_push(f"forward prefill {input_shape}")
            block_tables = torch.cuda.IntTensor([])
            context_lens = torch.cuda.IntTensor([])
            max_context_len = 0
        else:
            torch.cuda.nvtx.range_push(f"forward decode {input_shape}")
            context_lens = seq_lens
            max_context_len = torch.max(seq_lens)
            prompt_lens = []

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            max_context_len=max_context_len,
            block_tables=block_tables,
        )

        with torch.no_grad():
            hidden_states = self.pt_model.model(
                input_ids,
                positions,
                self.cache_blocks,
                input_metadata,
                # No need for this until parallel sampling is supported.
                cache_events=None,
            )

            if hidden_states.shape[0] != len(
                input_metadata.prompt_lens
            ) and hidden_states.shape[0] != len(input_metadata.context_lens):
                logits = get_logits(
                    self.pt_model.lm_head.weight,
                    hidden_states,
                    input_metadata,
                    self.vocab_size,
                )

            next_tokens = sample(logits, sampling_params, self.vocab_size)

            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

        outputs = []

        for i, (sequence_id, new_token) in enumerate(zip(sequence_ids, next_tokens)):
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


def _init_service(port):
    t = ThreadedServer(
        ModelRpcServer(),
        port=port,
        protocol_config={"allow_pickle": True, "sync_request_timeout": 600},
    )
    t.start()


def start_model_process(port):
    multiprocessing.set_start_method('spawn', force=True)
    proc = multiprocessing.Process(target=_init_service, args=(port,))
    proc.start()
    time.sleep(1)

    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect(
                "localhost",
                port,
                config={"allow_pickle": True, "sync_request_timeout": 600},
            )
            break
        except ConnectionRefusedError:
            time.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise RuntimeError("init rpc env error!")

    assert proc.is_alive()
    return con.root, proc


class ModelRpcClient:
    def __init__(self, num_shards, model_path, hf_config, engine_config):
        with ThreadPoolExecutor(num_shards) as executor:
            ports = [3000 + i for i in range(num_shards)]
            rets = executor.map(start_model_process, ports)
            print("started processes")

            self.model_servers = [x[0] for x in rets]
            self.procs = [x[1] for x in rets]

            def init_model(i):
                return self.model_servers[i].init_model(i, num_shards, model_path, hf_config, engine_config)

            rets = [obtain(x) for x in executor.map(init_model, range(num_shards))]

            self.num_blocks = rets[0]

            def _func(
                requests: Sequence[Union[PrefillRequest, DecodeRequest]], cache: KVCache
            ) -> List[TextGenerationResult]:
                def generate(i):
                    return self.model_servers[i].generate(requests, cache)

                res = [obtain(x) for x in executor.map(generate, range(num_shards))]
                return obtain(res[0].value)

            self.generate = _func

    def get_num_cache_blocks(self):
        return self.num_blocks


class Model:
    def __init__(
        self,
        pt_model,
        config,
    ):
        self.pt_model = pt_model
        self.vocab_size = config.vocab_size
        self.sliding_window = config.sliding_window
        self.num_shards = config.num_shards
        self.num_hidden_layers = config.num_hidden_layers

        if self.sliding_window:
            self.block_sliding_window = self.sliding_window // CacheManager.block_size
        else:
            self.block_sliding_window = None

        self.cache_blocks = None

    def profile_memory_usage(self, seq_lens):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        sampling_params = SamplingParams(top_p=0.99)

        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        seq_data = {}
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        for i, seq_len in enumerate(seq_lens):
            seq_groups.append(([i], sampling_params))
            prompt_tokens = [0] * seq_len
            seq_data[i] = SequenceData(prompt_tokens)

            input_tokens.extend(prompt_tokens)
            input_positions.extend(range(seq_len))
            slot_mapping.extend([0] * seq_len)

        input_ids = torch.cuda.LongTensor(input_tokens)
        positions = torch.cuda.LongTensor(input_positions)
        slot_mapping_tensor = torch.cuda.IntTensor(slot_mapping)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=seq_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=torch.cuda.IntTensor([]),
            max_context_len=0,
            block_tables=torch.cuda.IntTensor([]),
        )

        kv_caches = [(None, None)] * self.num_hidden_layers

        with torch.no_grad():
            self.pt_model.forward(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
                cache_events=None,
            )

        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated()
        print("peak memory", peak_memory / 1e9)

        torch.cuda.empty_cache()

        return peak_memory

    def generate(
        self,
        requests: Sequence[Union[PrefillRequest, DecodeRequest]],
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
        seq_data = {}
        seq_group_sequence_ids = defaultdict(list)
        seq_group_sampling_params = {}

        for request in requests:
            if isinstance(request, PrefillRequest):
                sequence_ids.append(get_prompt_sequence_id(request.request_id))
                num_sequences.append(request.num_sequence)
                prompt_lens.append(len(request.token_ids))
                seq_group_sequence_ids[request.request_id].append(sequence_ids[-1])
                seq_group_sampling_params[request.request_id] = convert_sampling_params(
                    request.sampling_params
                )
            else:
                sequence_ids.append(request.sequence_id)
                prompt_lens.append(request.prompt_token_counts)
                req_id = request.sequence_id.request_id
                seq_group_sequence_ids[req_id].append(request.sequence_id)
                seq_group_sampling_params[req_id] = convert_sampling_params(
                    request.sampling_params
                )

            all_token_ids.append(request.token_ids)
            sampling_params.append(request.sampling_params)

            seq_data[sequence_ids[-1]] = SequenceData(request.token_ids)

        seq_groups: List[Tuple[List[SequenceId], SamplingParams]] = []

        for req_id, seq_ids in seq_group_sequence_ids.items():
            seq_groups.append((seq_ids, seq_group_sampling_params[req_id]))

        (
            input_ids,
            positions,
            seq_lens,
            slot_mapping,
            _,
            block_tables,
        ) = prepare_inputs(
            sequence_ids,
            all_token_ids,
            prompt_lens,
            cache.slot_mappings,
            cache.decode_block_tables,
            self.sliding_window,
            is_prefill,
            torch.long,
            align=8,
        )

        input_shape = input_ids.shape

        if block_tables is None:
            torch.cuda.nvtx.range_push(f"forward prefill {input_shape}")
            block_tables = torch.cuda.IntTensor([])
            context_lens = torch.cuda.IntTensor([])
            max_context_len = 0
        else:
            torch.cuda.nvtx.range_push(f"forward decode {input_shape}")
            context_lens = seq_lens
            max_context_len = torch.max(seq_lens)
            prompt_lens = []

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            max_context_len=max_context_len,
            block_tables=block_tables,
        )

        with torch.no_grad():
            # outs = self.pt_model.forward(
            #      input_ids,
            #      positions,
            #      cache.cache_blocks,
            #      input_metadata,
            #      cache_events=None,  # TODO: what to do about this?
            #  )

            # next_tokens = []
            # for samples in outs:
            #     next_tokens.append(samples[0].output_token)

            hidden_states = self.pt_model.model(
                input_ids,
                positions,
                self.cache_blocks,
                input_metadata,
                # No need for this until parallel sampling is supported.
                cache_events=None,
            )

            if hidden_states.shape[0] != len(
                input_metadata.prompt_lens
            ) and hidden_states.shape[0] != len(input_metadata.context_lens):
                logits = get_logits(
                    self.pt_model.lm_head.weight,
                    hidden_states,
                    input_metadata,
                    self.vocab_size,
                )

            next_tokens = sample(logits, sampling_params, self.vocab_size)

            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

            print("logits.shape", logits.shape)

        print("next tokens", next_tokens)
        outputs = []

        for i, (sequence_id, new_token) in enumerate(zip(sequence_ids, next_tokens)):
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


def init_torch_model(
    model_path, engine_config: MLCServeEngineConfig
) -> Tuple[TextGenerator, CacheManager, ModelArtifactConfig]:
    hf_config = AutoConfig.from_pretrained(model_path)

    # TODO
    num_shards = 2

    num_kv_heads = hf_config.num_key_value_heads // num_shards
    head_size = hf_config.hidden_size // hf_config.num_attention_heads

    if not hasattr(hf_config, "sliding_window"):
        hf_config.sliding_window = None

    hf_config.num_shards = num_shards

    artifact_config = ModelArtifactConfig(
        model_artifact_path=model_path,
        num_shards=1,
        quantization=None,
        max_context_length=hf_config.max_position_embeddings,  # TODO,
        vocab_size=hf_config.vocab_size,
        sliding_window=hf_config.sliding_window,
        num_key_value_heads=num_kv_heads,
        num_attention_heads=hf_config.num_attention_heads,
        num_hidden_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
    )

    if num_shards > 1:
        model = ModelRpcClient(num_shards, model_path, hf_config, engine_config)
        num_blocks = model.get_num_cache_blocks()
    else:
        with torch.device("cuda"):
            torch.set_default_dtype(torch.float16)
            pt_model = LlamaForCausalLM(hf_config)
            pt_model.load_weights(model_path, None, "auto", None)

        model = Model(pt_model, hf_config)

        if engine_config.max_num_batched_tokens > 0:
            LOG.info("Running memory profiling.")
            num_blocks = get_num_cache_blocks(
                model,
                [engine_config.max_input_len] * engine_config.max_num_sequences,
                hf_config.num_hidden_layers,
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

        model.cache_blocks = init_cache_blocks(
            head_size,
            hf_config.num_hidden_layers,
            hf_config.num_attention_heads,
            CacheManager.block_size,
            num_blocks,
        )

        LOG.info("Allocated KV cache blocks.")

    cache_manager = CacheManager(
        num_blocks,
        hf_config.sliding_window,
    )

    return model, cache_manager, artifact_config
