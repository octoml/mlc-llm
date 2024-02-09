import math
import os
import json
import pathlib
from typing import List, Union, Tuple, Sequence

import structlog
import numpy as np
import torch
import tvm
from tvm import relax
from tvm.runtime import disco as di

from .base import ModelArtifactConfig
from .paged_cache_manager import KVCacheInfo, CacheManager
from .model_common import (
    sample_from_logits,
    prepare_inputs,
    prepare_multi_query_decode_inputs,
    get_num_cache_blocks,
)
from .lazy_safetensor import LazySafetensorDir

from ..engine import (
    get_prompt_sequence_id,
    MLCServeEngineConfig,
)
from ..engine.model_module import (
    DraftTokens,
    EvalMultiQueryRequest,
    PrefillRequest,
    DecodeRequest,
    TextGenerator,
    TextGenerationResult,
    RequestType,
)
from .sampler import SamplingState

LOG = structlog.stdlib.get_logger(__name__)


def load_disco_weights_through_shard_loader(sess, param_path, lib_path):
    # TODO(Lunderberg): Use a callback for get_item/set_item so these
    # only need to be defined on the transform_params path.
    worker_device = sess.get_global_func("runtime.disco.device")()
    sess.import_python_module("mlc_serve.model.lazy_safetensor")
    sess.get_global_func("mlc_serve.model.define_safetensors_get_item")(
        ".", "", worker_device
    )

    metadata_path = param_path.joinpath("ndarray-cache.json")
    module = sess.load_vm_module(lib_path.as_posix())

    loader_create = sess.get_global_func("runtime.disco.ShardLoader")
    with metadata_path.open(encoding="utf-8") as f:
        ndarray_cache_metadata = f.read()

    loader = loader_create(metadata_path.as_posix(), ndarray_cache_metadata, "", module)
    loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAllPresharded")
    params = loader_load(loader)

    return module, params


def load_disco_weights_through_transform_params(sess, orig_param_path, lib_path):
    worker_id = sess.get_global_func("runtime.disco.worker_id")()
    worker_device = sess.get_global_func("runtime.disco.device")()

    @tvm.register_func("get_item", override=True)
    def _unused_get_item():
        pass

    vm = relax.VirtualMachine(tvm.runtime.load_module(lib_path.as_posix()), tvm.cpu())
    param_names = vm["get_original_param_names"]()

    sess.import_python_module("mlc_serve.model.lazy_safetensor")
    sess.get_global_func("mlc_serve.model.define_safetensors_get_item")(
        orig_param_path.as_posix(), "\n".join(param_names), worker_device
    )

    module = sess.load_vm_module(lib_path.as_posix())
    params = module["transform_params"](worker_id)
    return module, params


def load_disco_module(
    artifact_path: Union[str, pathlib.Path],
    lib_path: Union[str, pathlib.Path],
    num_shards,
):
    artifact_path = pathlib.Path(artifact_path)
    lib_path = pathlib.Path(lib_path)

    sess = di.ProcessSession(num_workers=num_shards, entrypoint="tvm.exec.disco_worker")

    sess.init_ccl("nccl", *range(num_shards))

    param_path = artifact_path.joinpath("params")
    orig_param_path = artifact_path.joinpath("original_params")

    if param_path.exists():
        LOG.info("Loading multi-GPU pre-processed weights", path=param_path)
        module, params = load_disco_weights_through_shard_loader(
            sess, param_path, lib_path
        )
    elif orig_param_path.exists():
        LOG.info(
            "Loading multi-GPU weights through transform_params", path=orig_param_path
        )
        module, params = load_disco_weights_through_transform_params(
            sess, orig_param_path, lib_path
        )
    else:
        raise RuntimeError(
            f"The artifact path {artifact_path} contains neither a 'params' directory, "
            f"nor an 'original_params' directory."
        )

    return module, params, sess


def copy_to_worker_0(sess: di.Session, host_array):
    x_array = sess.empty(host_array.shape, host_array.dtype)
    sess.copy_to_worker_0(host_array, x_array)
    return x_array


def broadcast_from_worker_0(sess: di.Session, src, shape, dtype):
    dst = sess.empty(shape, dtype)
    sess.broadcast_from_worker0(src, dst)
    return dst


def load_tvm_weights(
    param_path: Union[str, pathlib.Path], dev: tvm.runtime.Device
) -> List[tvm.runtime.NDArray]:
    """Load a set of weights

    Loads a set of weights.  Assumes that the weights had previously
    been saved using `tvmjs.dump_ndarray_cache`.

    Parameters
    ----------
    param_path: Union[str, pathlib.Path]

        The path to the saved parameters.

    dev: tvm.runtime.Device

        The device onto which the saved parameters should be loaded.

    Returns
    -------
    params: List[tvm.runtime.NDArray]

        The loaded parameters

    """
    param_path = pathlib.Path(param_path)

    from tvm.contrib import tvmjs  # pylint: disable=import-outside-toplevel

    params, metadata = tvmjs.load_ndarray_cache(param_path.as_posix(), dev)

    try:
        params_names = [
            d["name"]
            for d in json.loads(str(vm.module.get_function("_metadata")()))["params"]
        ]
    except:
        params_names = [f"param_{i}" for i in range(metadata["ParamSize"])]

    return [params[param_name] for param_name in param_names]


def preprocess_weights(
    param_path: Union[str, pathlib.Path],
    executable_module: tvm.runtime.Module,
    dev: tvm.runtime.Device,
) -> List[tvm.runtime.NDArray]:
    """Load and preprocess a set of model weights

    Load and preprocess a set of model weights.

    Parameters
    ----------
    param_path: Union[str, pathlib.Path]

        The path to the model parameters parameters.

    executable_module: tvm.runtime.Module

        The compiled module, as returned from
        `tvm.runtime.load_module`, which contains the functions
        `transform_params` and `get_original_param_names`.

    dev: tvm.runtime.Device

        The device on which the pre-processed parameters should be
        loaded, after pre-processing is complete.

    Returns
    -------
    params: List[tvm.runtime.NDArray]

        The loaded parameters

    """
    param_path = pathlib.Path(param_path)

    safetensors = LazySafetensorDir(param_path)
    param_names = []

    @tvm.register_func("get_item", override=True)
    def get_item(i):
        safetensor = safetensors[param_names[i]]
        tvm_on_cpu = safetensor.as_tvm_array()
        return tvm.nd.array(tvm_on_cpu, dev)

    vm = relax.VirtualMachine(executable_module, dev)

    param_names.extend(vm["get_original_param_names"]())

    params = vm["transform_params"]()

    return params


def get_tvm_model(config, dev):
    LOG.info(f"Loading parameters from {config.model_artifact_path}.")

    artifact_path = pathlib.Path(config.model_artifact_path)
    lib_path = artifact_path.joinpath(config.library_name)

    if config.num_shards != 1:
        LOG.info(
            "Loading multi-GPU model, based on `num_shards` from config",
            num_shards=config.num_shards,
        )
        return load_disco_module(artifact_path, lib_path, config.num_shards)

    LOG.info(
        "Loading single-GPU model, based on `num_shards` from config",
        num_shards=config.num_shards,
    )

    ex = tvm.runtime.load_module(lib_path.as_posix())

    param_path = artifact_path.joinpath("params")
    orig_param_path = artifact_path.joinpath("original_params")
    if param_path.exists():
        LOG.info("Loading single-GPU pre-processed weights", path=param_path)

        # TODO(Lunderberg): Use a callback for get_item/set_item so these
        # only need to be defined on the transform_params path.
        @tvm.register_func("get_item", override=True)
        def _unused_get_item():
            pass

        params = load_tvm_weights(param_path, dev)
    elif orig_param_path.exists():
        LOG.info("Loading single-GPU weights through transform_params", path=param_path)
        params = preprocess_weights(orig_param_path, ex, dev)
    else:
        raise RuntimeError(
            f"The artifact path {artifact_path} contains neither a 'params' directory, "
            f"nor an 'original_params' directory."
        )

    vm = relax.VirtualMachine(ex, dev)
    return vm.module, params, None


def _prepare_inputs(
    sequence_ids,
    all_token_ids,
    prompt_lens,
    all_slot_mappings,
    all_decode_block_tables,
    sliding_window,
    is_prefill,
    num_decode_query_tokens=1,
):
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
        all_slot_mappings,
        all_decode_block_tables,
        sliding_window,
        is_prefill,
        num_decode_query_tokens,
    )

    if block_tables is not None:
        block_tables = tvm.nd.from_dlpack(block_tables)
    if indices_within_window is not None:
        indices_within_window = tvm.nd.from_dlpack(indices_within_window)

    if not is_prefill and num_decode_query_tokens > 1:
        input_ids = torch.reshape(input_ids, (-1, num_decode_query_tokens))

    return (
        tvm.nd.from_dlpack(input_ids),
        tvm.nd.from_dlpack(positions),
        tvm.nd.from_dlpack(seq_lens),
        tvm.nd.from_dlpack(slot_mapping),
        indices_within_window,
        block_tables,
    )


class Model:
    def __init__(
        self,
        config: ModelArtifactConfig,
        dev: tvm.runtime.Device,
        block_size: int,
        copy_blocks_func_name: str,
    ):
        self.mod, self.params, self.disco_session = get_tvm_model(config, dev)
        self.dev = dev
        self.vocab_size = config.vocab_size
        self.sliding_window = config.sliding_window
        self.num_shards = config.num_shards

        # TODO(@sunggg): Find a better way
        if config.model_type == "llama":
            self.torch_dtype = torch.float32
        elif config.model_type == "mistral" or config.model_type == "mixtral":
            self.torch_dtype = torch.float32
        else:
            assert 0, f"{config.model_type} is NOT supported yet"

        self._copy_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.torch_dev: str = "cuda"

        if self.sliding_window:
            self.block_sliding_window = self.sliding_window // block_size
        else:
            self.block_sliding_window = None

        if self.disco_session:
            self.copy_cache_blocks_func = self.disco_session.get_global_func(
                "tvm.contrib.vllm.copy_blocks"
            )
        else:
            self.copy_cache_blocks_func = tvm.get_global_func(
                copy_blocks_func_name,
            )

        self.cache_blocks = None

    def get_param_nbytes(self):
        """Get the total size of the parameters"""
        if self.disco_session:
            params = self.params.debug_get_from_remote(0)
        else:
            params = self.params

        return sum(
            math.prod(param.shape) * np.dtype(param.dtype).itemsize for param in params
        )

    def get_used_memory(self):
        """Get the total memory allocated by the VM"""
        if self.disco_session:
            get_used_memory_func = self.disco_session.get_global_func(
                "vm.memory_manager.get_used_memory"
            )
            # For Disco, we explicitly query the device 0.
            peak_memory = get_used_memory_func(
                tvm.device("cuda", 0)
            ).debug_get_from_remote(0)
        else:
            get_used_memory_func = tvm.get_global_func(
                "vm.memory_manager.get_used_memory"
            )
            peak_memory = get_used_memory_func(self.dev)

        return peak_memory

    def profile_memory_usage(self, seq_lens):
        input_ids = [0] * sum(seq_lens)
        positions = []

        for s in seq_lens:
            positions += range(s)

        vm_alloc_before = self.get_used_memory()

        input_ids = tvm.nd.array(np.array(input_ids, dtype="int32"), self.dev)
        positions = tvm.nd.array(np.array(positions, dtype="int32"), self.dev)
        seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), self.dev)

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)

            start_profiling_func = self.disco_session.get_global_func(
                "vm.memory_manager.start_profiling"
            )
            stop_profiling_func = self.disco_session.get_global_func(
                "vm.memory_manager.stop_profiling"
            )
        else:
            start_profiling_func = tvm.get_global_func(
                "vm.memory_manager.start_profiling"
            )
            stop_profiling_func = tvm.get_global_func(
                "vm.memory_manager.stop_profiling"
            )

        start_profiling_func()
        self.mod["evaluate"](input_ids, positions, seq_lens, self.params)
        stop_profiling_func()

        vm_alloc_after = self.get_used_memory()

        return self.get_param_nbytes() + (vm_alloc_after - vm_alloc_before)

    def generate_multi_query(
        self,
        requests: List[EvalMultiQueryRequest],
        cache: KVCacheInfo,
    ) -> List[TextGenerationResult]:
        sequence_ids = []
        last_query_offsets: List[int] = []
        sampling_params = []
        past_decode_tokens = []
        for request in requests:
            assert not isinstance(request.queries, DraftTokens)
            sequence_ids.append(request.sequence_id)
            if len(last_query_offsets) == 0:
                last_query_offsets.append(request.queries.num_tokens - 1)
            else:
                last_query_offsets.append(
                    last_query_offsets[-1] + request.queries.num_tokens
                )
            sampling_params.append(request.sampling_params)
            # Use `vocab_size` as a padding
            past_decode_tokens.append([self.vocab_size, *request.queries.token_ids])

        # Prepare sampling tensors in another stream to overlap
        # CPU<->GPU data transfer with GPU computation in forward pass.
        with torch.cuda.stream(self._copy_stream):
            sampling_state = SamplingState.from_sampling_params(
                sampling_params,
                past_decode_tokens,
                self.torch_dtype,
                self.torch_dev,
                self.vocab_size,
            )

        (
            input_ids,
            positions,
            seq_lens,
            slot_mapping,
            query_lens,
            past_slot_mapping,
            permute_map,
        ) = prepare_multi_query_decode_inputs(
            requests,
            cache.slot_mappings,
            None,
            self.dev,
        )

        torch.cuda.nvtx.range_push(f"forward multi-query decode {input_ids.shape}")

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)
            slot_mapping = copy_to_worker_0(self.disco_session, slot_mapping)
            query_lens = copy_to_worker_0(self.disco_session, query_lens)
            past_slot_mapping = copy_to_worker_0(self.disco_session, past_slot_mapping)
            permute_map = copy_to_worker_0(self.disco_session, permute_map)

        out = self.mod["evaluate_multi_query"](
            input_ids,
            positions,
            seq_lens,
            self.cache_blocks,
            slot_mapping,
            query_lens,
            past_slot_mapping,
            permute_map,
            self.params,
        )

        if self.disco_session:
            logits, _ = out.debug_get_from_remote(0)
        else:
            logits = out[0]

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        last_query_logits = torch.from_dlpack(logits)[last_query_offsets]
        return sample_from_logits(
            last_query_logits,
            sequence_ids,
            requests,
            sampling_state,
            self.vocab_size,
            self._copy_stream,
            self.torch_dtype,
            self.torch_dev,
            past_decode_tokens,
        )

    def generate(
        self,
        requests: Sequence[RequestType],
        cache: KVCacheInfo,
    ) -> List[TextGenerationResult]:
        batch_size = len(requests)
        LOG.debug(f"Generation batch size: f{batch_size}.", batch_size=batch_size)
        if batch_size == 0:
            return []

        is_prefill = isinstance(requests[0], PrefillRequest)
        is_multi_query_decode = isinstance(requests[0], EvalMultiQueryRequest)

        if is_multi_query_decode:
            return self.generate_multi_query(requests, cache)  # type: ignore

        # Prefill or decode
        all_token_ids = []
        sequence_ids = []
        prompt_lens = []
        # TODO(masahi, yelite): Update this when a new request type for speculative decoding
        # is implemented.
        num_decode_query_tokens = 1
        sampling_params = []
        past_decode_tokens = []

        for request in requests:
            if isinstance(request, PrefillRequest):
                seq_id = get_prompt_sequence_id(request.request_id)
                # Use `vocab_size` as a padding.
                # This is convenient way to filter out paddings
                # after the vectorized sampling computation
                # since logit index will be in range of [0,vocab_size)
                request_past_decode_tokens = [self.vocab_size]
            elif isinstance(request, DecodeRequest):
                seq_id = request.sequence_id
                prompt_lens.append(request.prompt_token_counts)
                # Use `vocab_size` as a padding
                # This is convenient way to filter out paddings
                # after the vectorized sampling computation
                # since logit index will be in range of [0,vocab_size)
                request_past_decode_tokens = [self.vocab_size, *request.token_ids]
            else:
                raise Exception("`EvalMultiQueryRequest` should not reach here.")

            past_decode_tokens.append(request_past_decode_tokens)
            sequence_ids.append(seq_id)

            assert not isinstance(request, EvalMultiQueryRequest)
            all_token_ids.append(request.token_ids)
            sampling_params.append(request.sampling_params)

        # Prepare sampling tensors in another stream to overlap
        # CPU<->GPU data transfer with GPU computation in forward pass.
        with torch.cuda.stream(self._copy_stream):
            sampling_state = SamplingState.from_sampling_params(
                sampling_params,
                past_decode_tokens,
                self.torch_dtype,
                self.torch_dev,
                self.vocab_size,
            )

        (
            input_ids,
            positions,
            seq_lens,
            slot_mapping,
            indices_within_window,
            block_tables,
        ) = _prepare_inputs(
            sequence_ids,
            all_token_ids,
            prompt_lens,
            cache.slot_mappings,
            cache.decode_block_tables,
            self.sliding_window,
            is_prefill,
            num_decode_query_tokens,
        )

        input_shape = input_ids.shape
        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)
            slot_mapping = copy_to_worker_0(self.disco_session, slot_mapping)

        if is_prefill:
            torch.cuda.nvtx.range_push(f"forward prefill {input_shape}")

            if self.sliding_window:
                if self.disco_session:
                    indices_within_window = copy_to_worker_0(
                        self.disco_session, indices_within_window
                    )

                out = self.mod["prefill"](
                    input_ids,
                    positions,
                    seq_lens,
                    self.cache_blocks,
                    slot_mapping,
                    indices_within_window,
                    self.params,
                )
            else:
                out = self.mod["prefill"](
                    input_ids,
                    positions,
                    seq_lens,
                    self.cache_blocks,
                    slot_mapping,
                    self.params,
                )
        else:
            torch.cuda.nvtx.range_push(f"forward decode {input_shape}")

            if self.disco_session:
                block_tables = copy_to_worker_0(self.disco_session, block_tables)

            if num_decode_query_tokens is not None and num_decode_query_tokens > 1:
                out = self.mod["decode_multi_query"](
                    input_ids,
                    positions,
                    seq_lens,
                    self.cache_blocks,
                    slot_mapping,
                    block_tables,
                    self.params,
                )
            else:
                out = self.mod["decode"](
                    input_ids,
                    positions,
                    seq_lens,
                    self.cache_blocks,
                    slot_mapping,
                    block_tables,
                    self.params,
                )

        if self.disco_session:
            logits, _ = out.debug_get_from_remote(0)
        else:
            logits = out[
                0
            ]  # Ignore returned KV cache since it is updated in-place anyway.

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        if is_prefill and cache.pending_copy_from_to:
            block_mapping = tvm.nd.array(
                np.array(cache.pending_copy_from_to, dtype="int64")
            )

            if self.disco_session:
                block_mapping = broadcast_from_worker_0(
                    self.disco_session,
                    copy_to_worker_0(self.disco_session, block_mapping),
                    block_mapping.shape,
                    "int64",
                )

            self.copy_cache_blocks_func(self.cache_blocks, block_mapping)
            cache.pending_copy_from_to = []

        if len(logits.shape) == 3:
            # TODO(masahi, yelite): Proper logic for handling multi-query logits (speculative decoding).
            return []

        return sample_from_logits(
            logits,
            sequence_ids,
            requests,
            sampling_state,
            self.vocab_size,
            self._copy_stream,
            self.torch_dtype,
            self.torch_dev,
            past_decode_tokens,
        )


def init_tvm_model(
    model_artifact_config: ModelArtifactConfig, engine_config: MLCServeEngineConfig
) -> Tuple[TextGenerator, CacheManager]:
    dev = tvm.device("cuda", 0)

    num_kv_heads = (
        model_artifact_config.num_key_value_heads // model_artifact_config.num_shards
    )
    head_size = (
        model_artifact_config.hidden_size // model_artifact_config.num_attention_heads
    )

    if model_artifact_config.paged_kv_cache_type == "flash-decoding":
        allocate_func_name = "tvm.contrib.flash_attn.allocate_kv_cache"
        copy_blocks_func_name = "tvm.contrib.flash_attn.copy_blocks"
        # This needs to match with the model definition in llama_batched_vllm.py
        if head_size <= 64:
            block_size = 256
        elif head_size <= 128:
            block_size = 128
        else:
            block_size = 64
    else:
        allocate_func_name = "tvm.contrib.vllm.allocate_kv_cache"
        copy_blocks_func_name = "tvm.contrib.vllm.copy_blocks"
        block_size = 16

    model = Model(model_artifact_config, dev, block_size, copy_blocks_func_name)

    if model_artifact_config.num_shards > 1:
        model.disco_session.sync_worker_0()

    if engine_config.max_num_batched_tokens > 0:
        LOG.info("Running memory profiling.")
        try:
            num_blocks = get_num_cache_blocks(
                model,
                block_size,
                [1] * engine_config.max_num_batched_tokens,
                model_artifact_config.num_hidden_layers,
                num_kv_heads,
                head_size,
            )
        except tvm.error.InternalError:
            raise RuntimeError(
                f"Memory profiling failed with max_num_batched_tokens = "
                "{engine_config.max_num_batched_tokens}."
            )
    else:
        num_blocks = 500

    num_cache_slots = num_blocks * block_size

    if num_cache_slots <= engine_config.max_num_batched_tokens:
        raise RuntimeError(
            f"max_num_batched_tokens = {engine_config.max_num_batched_tokens} but"
            f" only {num_blocks} cache blocks can be allocated. The number of"
            f" available cache slots is {num_cache_slots}, not enough for"
            f" {engine_config.max_num_batched_tokens} tokens. Try reducing"
            " --max_num_batched_tokens."
        )

    LOG.info(f"Using {num_blocks} cache blocks.")

    if model.disco_session:
        init_cache_func = model.disco_session.get_global_func(allocate_func_name)
    else:
        init_cache_func = tvm.get_global_func(allocate_func_name)

    try:
        model.cache_blocks = init_cache_func(
            head_size,
            model_artifact_config.num_hidden_layers,
            num_kv_heads,
            block_size,
            num_blocks,
        )
    except tvm.error.InternalError:
        raise RuntimeError(f"Failed to allocate {num_blocks} cache blocks.")

    cache_manager = CacheManager(
        num_blocks,
        block_size,
        model_artifact_config.sliding_window,
    )

    LOG.info("Allocated KV cache blocks.")

    return model, cache_manager
