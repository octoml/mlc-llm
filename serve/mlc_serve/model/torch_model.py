import time
import os
from typing import List, Union, Tuple, Sequence
from pathlib import Path

import structlog
import torch

from transformers import AutoConfig

from vllm.model_executor.layers.sampler import get_logits
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.qwen import QWenLMHeadModel
from vllm.model_executor.models.phi import PhiForCausalLM
from vllm.model_executor import InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel,
)

import torch.multiprocessing as multiprocessing

import rpyc
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from concurrent.futures import ThreadPoolExecutor

from .base import ModelArtifactConfig
from .paged_cache_manager import KVCacheInfo, CacheManager
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


def profile_memory_usage(pt_model, seq_lens, num_hidden_layers):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    input_tokens: List[List[int]] = []
    input_positions: List[List[int]] = []
    slot_mapping: List[List[int]] = []

    for seq_len in seq_lens:
        prompt_tokens = [0] * seq_len

        input_tokens.append(prompt_tokens)
        input_positions.append(list(range(seq_len)))
        slot_mapping.append([0] * seq_len)

    input_ids = torch.cuda.LongTensor(input_tokens)
    positions = torch.cuda.LongTensor(input_positions)
    slot_mapping_tensor = torch.cuda.LongTensor(slot_mapping)
    prompt_lens_tensor = torch.cuda.LongTensor(seq_lens)

    input_metadata = InputMetadata(
        is_prompt=True,
        slot_mapping=slot_mapping_tensor,
        prompt_lens=prompt_lens_tensor,
        max_seq_len=None,
        start_loc=None,
        max_context_len=0,
        context_lens=torch.cuda.IntTensor([]),
        block_tables=torch.cuda.IntTensor([]),
        use_cuda_graph=False,
    )

    kv_caches = [(None, None)] * num_hidden_layers

    with torch.no_grad():
        pt_model.forward(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
        )

    torch.cuda.synchronize()

    peak_memory = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()

    return peak_memory


def profile_and_init_cache(
    pt_model,
    hf_config,
    num_shards,
    max_num_batched_tokens,
):
    num_kv_heads = hf_config.num_key_value_heads // num_shards
    num_hidden_layers = hf_config.num_hidden_layers
    head_size = hf_config.hidden_size // hf_config.num_attention_heads

    if max_num_batched_tokens > 0:
        LOG.info("Running memory profiling.")
        seq_lens = [1] * max_num_batched_tokens
        used_memory_bytes = profile_memory_usage(pt_model, seq_lens, num_hidden_layers)
        num_blocks = get_num_cache_blocks(
            used_memory_bytes,
            hf_config.num_hidden_layers,
            num_kv_heads,
            head_size,
        )
    else:
        num_blocks = 500

    LOG.info(f"Using {num_blocks} cache blocks.")

    cache_blocks = init_cache_blocks(
        head_size,
        hf_config.num_hidden_layers,
        num_kv_heads,
        CacheManager.block_size,
        num_blocks,
    )

    LOG.info("Allocated KV cache blocks.")

    return cache_blocks, num_blocks


def load_model(hf_config, model_path):
    model_map = {
        "LlamaForCausalLM": LlamaForCausalLM,
        "PhiForCausalLM": PhiForCausalLM,
        "QWenLMHeadModel": QWenLMHeadModel,  # requires tiktoken package
    }

    arch = hf_config.architectures[0]

    if arch not in model_map:
        raise RuntimeError(f"Unsupported model: {arch}")

    with torch.device("cuda"):
        torch.set_default_dtype(torch.float16)
        model = model_map[arch](hf_config)
        model.load_weights(model_path, None, "auto", None)
        return model


def generate(
    requests: Sequence[Union[PrefillRequest, DecodeRequest]],
    cache_info: KVCacheInfo,
    pt_model,
    cache_blocks,
    sliding_window,
    vocab_size,
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
            prompt_lens.append(len(request.token_ids))
        else:
            sequence_ids.append(request.sequence_id)
            prompt_lens.append(request.prompt_token_counts)

        all_token_ids.append(request.token_ids)
        sampling_params.append(request.sampling_params)

    selected_token_indices: List[int] = []

    if is_prefill:
        max_prompt_len = max(prompt_lens)
        seq_start = 0

        for prompt_len in prompt_lens:
            selected_token_indices.append(seq_start + prompt_len - 1)
            seq_start += max_prompt_len

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
        cache_info.slot_mappings,
        cache_info.decode_block_tables,
        sliding_window,
        is_prefill,
        for_vllm=True,
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

    prompt_lens = torch.cuda.LongTensor(prompt_lens)

    input_metadata = InputMetadata(
        is_prompt=is_prefill,
        slot_mapping=slot_mapping,
        prompt_lens=prompt_lens,
        max_seq_len=None,
        start_loc=None,
        max_context_len=max_context_len,
        context_lens=context_lens,
        block_tables=block_tables,
        use_cuda_graph=False,
    )

    sampling_metadata = SamplingMetadata(
        seq_groups=None,
        seq_data=None,
        prompt_lens=prompt_lens,
        selected_token_indices=torch.tensor(
            selected_token_indices, dtype=torch.long, device="cuda"
        ),
        categorized_sample_indices=None,
    )

    with torch.no_grad():
        hidden_states = pt_model.model(
            input_ids,
            positions,
            cache_blocks,
            input_metadata,
        )

        logits = get_logits(
            pt_model.lm_head.weight,
            hidden_states,
            sampling_metadata,
            vocab_size,
        )

        next_tokens = sample(logits, sampling_params, vocab_size)

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


class ModelRpcServer(rpyc.Service):
    def exposed_init_model(
        self,
        tp_rank: int,
        num_shards: int,
        model_path: Path,
        hf_config: AutoConfig,
        engine_config: MLCServeEngineConfig,
    ) -> int:
        hf_config = obtain(hf_config)
        engine_config = obtain(engine_config)
        model_path = obtain(model_path)

        self.vocab_size = hf_config.vocab_size
        self.sliding_window = hf_config.sliding_window

        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        torch.cuda.set_device(tp_rank)

        os.environ["MASTER_ADDR"] = str("127.0.0.1")
        os.environ["MASTER_PORT"] = str(4000)  # TODO port

        torch.distributed.init_process_group(
            backend="nccl",
            world_size=num_shards,
            rank=tp_rank,
        )
        initialize_model_parallel(num_shards)

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cuda())

        self.pt_model = load_model(hf_config, model_path)

        self.cache_blocks, num_blocks = profile_and_init_cache(
            self.pt_model,
            hf_config,
            num_shards,
            engine_config.max_num_batched_tokens,
        )

        return num_blocks

    def exposed_generate(
        self,
        requests: Sequence[Union[PrefillRequest, DecodeRequest]],
        cache: KVCacheInfo,
    ) -> List[TextGenerationResult]:
        requests = obtain(requests)
        cache = obtain(cache)
        return generate(
            requests,
            cache,
            self.pt_model,
            self.cache_blocks,
            self.sliding_window,
            self.vocab_size,
        )


def _init_service(port):
    t = ThreadedServer(
        ModelRpcServer(),
        port=port,
        protocol_config={"allow_pickle": True, "sync_request_timeout": 600},
    )
    t.start()


def start_model_process(port):
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
    def __init__(
        self,
        model_path: Path,
        hf_config: AutoConfig,
        engine_config: MLCServeEngineConfig,
    ):
        assert engine_config.num_shards is not None

        self.num_shards = engine_config.num_shards

        with ThreadPoolExecutor(self.num_shards) as executor:
            ports = [3010 + i for i in range(self.num_shards)]  # TODO port
            rets = executor.map(start_model_process, ports)

            self.model_servers = [x[0] for x in rets]
            self.procs = [x[1] for x in rets]

            def init_model(i):
                return self.model_servers[i].init_model(
                    i, self.num_shards, model_path, hf_config, engine_config
                )

            rets = executor.map(init_model, range(self.num_shards))
            self.num_blocks = obtain(list(rets)[0])

    def generate(
        self,
        requests: Sequence[Union[PrefillRequest, DecodeRequest]],
        cache: KVCacheInfo,
    ) -> List[TextGenerationResult]:
        def _generate(i):
            return self.model_servers[i].generate(requests, cache)

        with ThreadPoolExecutor(self.num_shards) as executor:
            res = [obtain(x) for x in executor.map(_generate, range(self.num_shards))]
            return obtain(res[0])


class Model:
    def __init__(
        self,
        model_path: Path,
        hf_config: AutoConfig,
        engine_config: MLCServeEngineConfig,
    ):
        if engine_config.num_shards and engine_config.num_shards > 1:
            torch.multiprocessing.set_start_method("spawn")
            self.model_rpc = ModelRpcClient(model_path, hf_config, engine_config)
            self.num_blocks = self.model_rpc.num_blocks
            self.cache_blocks = None
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=1,
                rank=0,
                init_method="tcp://localhost:59157",  # TODO port
            )
            initialize_model_parallel(1, 1)

            self.pt_model = load_model(hf_config, model_path)
            self.cache_blocks, self.num_blocks = profile_and_init_cache(
                self.pt_model,
                hf_config,
                1,
                engine_config.max_num_batched_tokens,
            )
            self.model_rpc = None

        self.vocab_size = hf_config.vocab_size
        self.sliding_window = hf_config.sliding_window

    def generate(
        self,
        requests: Sequence[Union[PrefillRequest, DecodeRequest]],
        cache: KVCacheInfo,
    ) -> List[TextGenerationResult]:
        if self.model_rpc is None:
            return generate(
                requests,
                cache,
                self.pt_model,
                self.cache_blocks,
                self.sliding_window,
                self.vocab_size,
            )

        return self.model_rpc.generate(requests, cache)


def init_torch_model(
    model_path: Path, engine_config: MLCServeEngineConfig
) -> Tuple[TextGenerator, CacheManager, ModelArtifactConfig]:
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if not hasattr(hf_config, "num_key_value_heads") and hasattr(
        hf_config, "num_attention_heads"
    ):
        hf_config.num_key_value_heads = hf_config.num_attention_heads

    if not hasattr(hf_config, "sliding_window"):
        hf_config.sliding_window = None

    if engine_config.num_shards is None:
        raise RuntimeError("num_shards needs to be specifed for PyTorch models.")

    num_shards = engine_config.num_shards

    artifact_config = ModelArtifactConfig(
        model_artifact_path=str(model_path),
        num_shards=num_shards,
        quantization=None,
        max_context_length=hf_config.max_position_embeddings,  # TODO,
        vocab_size=hf_config.vocab_size,
        sliding_window=hf_config.sliding_window,
        num_key_value_heads=hf_config.num_key_value_heads // num_shards,
        num_attention_heads=hf_config.num_attention_heads,
        num_hidden_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
    )

    model = Model(model_path, hf_config, engine_config)

    cache_manager = CacheManager(
        model.num_blocks,
        hf_config.sliding_window,
    )

    return model, cache_manager, artifact_config
