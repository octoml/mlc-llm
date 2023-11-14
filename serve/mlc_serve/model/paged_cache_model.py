import json
import logging
import math
import os
import inspect
from collections import defaultdict
from typing import List, Union, Optional
from dataclasses import dataclass, fields

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer
from tvm import relax
from tvm.runtime import disco as di

from mlc_llm import utils
from mlc_llm.relax_model.llama import LlamaConfig

from ..engine import ChatMessage, RequestId, SamplingType
from ..engine.model_module import (
    DecodeRequest,
    PrefillRequest,
    SequenceId,
    TextGenerationResult,
)

logger = logging.getLogger(__name__)


class KVCache:
    def __init__(
        self, num_blocks, block_size, num_layers, num_heads, head_size, disco_session
    ):
        if disco_session:
            init_cache_func = disco_session.get_global_func(
                "tvm.contrib.vllm.allocate_kv_cache"
            )
        else:
            init_cache_func = tvm.get_global_func("tvm.contrib.vllm.allocate_kv_cache")

        self.cache = init_cache_func(
            head_size, num_layers, num_heads, block_size, num_blocks
        )

        self.block_tables = defaultdict(list)
        self.slot_mappings = defaultdict(list)
        self.block_size = block_size


class CacheManager:
    block_size: int = 16

    @staticmethod
    def get_cache_block_size(num_layers, num_heads, head_size):
        # Taken from vllm/worker/cache_engine.py
        key_cache_block = CacheManager.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = 2  # fp16
        return dtype_size * total

    def __init__(
        self,
        num_blocks,
        num_layers,
        num_heads,
        head_size,
        disco_session=None,
        sliding_window=None,
    ):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.kv_cache = KVCache(
            num_blocks, self.block_size, num_layers, num_heads, head_size, disco_session
        )
        self.allocated_tokens = dict[RequestId, int]()

        if sliding_window:
            assert sliding_window % self.kv_cache.block_size == 0
            self.block_sliding_window = sliding_window // self.kv_cache.block_size
        else:
            self.block_sliding_window = None

    def set_size(self, request_ids: List[int], target_sizes: List[int]):
        for id, size in zip(request_ids, target_sizes):
            num_needed_block = math.ceil(size / self.block_size)

            if self.block_sliding_window:
                num_needed_block = min(num_needed_block, self.block_sliding_window)

            if id in self.kv_cache.block_tables and size == 0:
                self.free_blocks.extend(self.kv_cache.block_tables[id])
                del self.kv_cache.block_tables[id]
                del self.kv_cache.slot_mappings[id]

            elif id in self.kv_cache.block_tables:
                # Decoding
                if len(self.kv_cache.block_tables[id]) < num_needed_block:
                    # Need to allocate a new block for this request
                    assert len(self.kv_cache.block_tables[id]) + 1 == num_needed_block
                    self.kv_cache.block_tables[id].append(self.free_blocks.pop())

                pos = size - 1
                block_number = self.kv_cache.block_tables[id][-1]

                if self.block_sliding_window:
                    block_number = self.kv_cache.block_tables[id][
                        (pos // self.block_size) % self.block_sliding_window
                    ]
                else:
                    block_number = self.kv_cache.block_tables[id][-1]

                block_offset = pos % self.block_size
                slot = block_number * self.block_size + block_offset
                self.kv_cache.slot_mappings[id].append(slot)

            elif id not in self.kv_cache.block_tables:
                assert (
                    len(self.free_blocks) >= num_needed_block
                ), "Not enough free blocks."

                for _ in range(num_needed_block):
                    self.kv_cache.block_tables[id].append(self.free_blocks.pop())

                for block_idx in range(math.floor(size / self.block_size)):
                    if self.block_sliding_window:
                        block_idx %= self.block_sliding_window

                    block_number = self.kv_cache.block_tables[id][block_idx]
                    slots = [
                        block_number * self.block_size + block_offset
                        for block_offset in range(self.block_size)
                    ]
                    self.kv_cache.slot_mappings[id] += slots

                for i in range(len(self.kv_cache.slot_mappings[id]), size):
                    block_idx = i // self.block_size

                    if self.block_sliding_window:
                        block_idx %= self.block_sliding_window

                    block_number = self.kv_cache.block_tables[id][block_idx]
                    block_offset = i % self.block_size
                    slot = block_number * self.block_size + block_offset
                    self.kv_cache.slot_mappings[id].append(slot)

    def get_cache(self):
        return self.kv_cache

    def allocate(self, request_id: RequestId, num_tokens: int):
        """
        Allocate cache space for request, raise error if there is no space.
        """
        self.set_size([request_id], [num_tokens])
        self.allocated_tokens[request_id] = num_tokens

    def extend(self, sequence_id: SequenceId, new_tokens: int):
        """
        Extend cache space for a sequence, raise error if there is no space.
        """
        assert sequence_id.sequence_index == 0, "multiple sequences not supported"
        request_id = sequence_id.request_id
        allocated = self.allocated_tokens[request_id]
        self.set_size([request_id], [allocated + new_tokens])
        self.allocated_tokens[request_id] += new_tokens

    def free(self, sequence_id: SequenceId):
        """
        Free cache space for a sequence or all sequences of a request.
        """
        assert sequence_id.sequence_index == 0, "multiple sequences not supported"
        request_id = sequence_id.request_id
        del self.allocated_tokens[request_id]
        self.set_size([request_id], [0])

    def get_kv_cache_size(self) -> int:
        """
        Return the size of the cache, in number of tokens.
        """
        return self.num_blocks * self.block_size

    def get_free_space(self) -> int:
        """
        Get available space of the cache.
        Return number of tokens that can be allocated for a new request.

        For paged KV cache, this ignores the remaining tokens in pages allocated
        for existing sequences, since they cannot be used for the new request.
        """
        return len(self.free_blocks) * self.block_size

    def get_max_new_tokens(self) -> int:
        """
        Get the maximum number of new tokens that can be extended for
        all sequences in the cache.

        For example, if the cache size is 16 tokens, with page size 1, and
        there are 3 sequences in the cache, each of them have 3 tokens cached,
        this method should return 2.

        It should return the result of `get_kv_cache_size` if there is
        no requests in the cache.
        """
        if not self.allocated_tokens:
            return len(self.free_blocks) * self.block_size

        free_blocks_per_request = len(self.free_blocks) // len(self.allocated_tokens)
        remaining_blocks = len(self.free_blocks) - free_blocks_per_request * len(
            self.allocated_tokens
        )
        remaining_tokens_in_last_block = [
            self.block_size - (tokens - 1) % self.block_size - 1
            for _, tokens in self.allocated_tokens.items()
        ]

        return (
            free_blocks_per_request * self.block_size
            + sorted(remaining_tokens_in_last_block)[remaining_blocks]
        )


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


def sample(logits, sampling_params, vocab_size):
    logits = torch.from_dlpack(logits)
    num_seq = len(sampling_params)

    mask_random = torch.tensor(
        [p.sampling_type == SamplingType.RANDOM for p in sampling_params],
        dtype=torch.bool,
    )
    mask_greedy = torch.logical_not(mask_random)

    logits_greedy = logits[mask_greedy]

    if logits_greedy.shape[0] > 0:
        res_greedy = torch.argmax(logits_greedy, -1).cpu().numpy()

        if logits_greedy.shape[0] == num_seq:
            return res_greedy

    temperatures = []
    top_ps = []
    top_ks = []
    divide_by_temperature = False
    do_top_p = False
    do_top_k = False

    for i in range(num_seq):
        param = sampling_params[i]

        if param.sampling_type == SamplingType.RANDOM:
            temperatures.append(param.temperature)
            top_ps.append(param.top_p)
            top_ks.append(param.top_k if param.top_k != -1 else vocab_size)

            divide_by_temperature |= temperatures[-1] != 1.0
            do_top_p |= top_ps[-1] < 1.0
            do_top_k |= top_ks[-1] != vocab_size

    logits_random = logits[mask_random]

    if divide_by_temperature:
        t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
        logits_random.div_(t.unsqueeze(dim=1))

    if do_top_p or do_top_k:
        logits = _apply_top_p_top_k(logits_random, top_ps, top_ks)

    probs = torch.softmax(logits_random, dim=-1)
    res_random = torch.multinomial(probs, 1, True).cpu().numpy()[:, 0]

    if logits_random.shape[0] == num_seq:
        return res_random

    res = np.empty((num_seq,), dtype=np.int32)
    res[mask_random] = res_random

    if logits_greedy.shape[0] > 0:
        res[mask_greedy] = res_greedy

    return res


def load_disco_module(artifact_path, lib_path, num_shards):
    sess = di.ProcessSession(num_workers=num_shards)
    devices = range(num_shards)
    sess.init_ccl("nccl", *devices)
    module = sess.load_vm_module(lib_path)

    loader_create = sess.get_global_func("runtime.disco.ShardLoader")
    metadata_path = os.path.join(artifact_path, "params", "ndarray-cache.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        ndarray_cache_metadata = f.read()

    loader = loader_create(metadata_path, ndarray_cache_metadata, "", module)
    loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAllPresharded")
    params = loader_load(loader)

    return module, params, sess


def copy_to_worker_0(sess: di.Session, host_array):
    x_array = sess.empty(host_array.shape, host_array.dtype)
    sess.copy_to_worker_0(host_array, x_array)
    return x_array


def get_tvm_model(artifact_path, model, quantization, num_shards, dev):
    model_artifact_path = os.path.join(
        artifact_path, f"{model}-{quantization}-presharded-{num_shards}gpu"
    )
    if not os.path.exists(model_artifact_path):
        model_artifact_path = os.path.join(artifact_path, f"{model}-{quantization}")

    lib_path = os.path.join(model_artifact_path, f"{model}-{quantization}-cuda.so")

    chat_file = os.path.join(f"{model_artifact_path}/params", "mlc-chat-config.json")
    if num_shards == 1:
        ex = tvm.runtime.load_module(lib_path)
        vm = relax.VirtualMachine(ex, dev)
        params = utils.load_params(model_artifact_path, dev)
        return vm.module, params, None, chat_file

    return load_disco_module(model_artifact_path, lib_path, num_shards), chat_file

@dataclass
class ConvConfig:
    r"""A dataclass that represents user-defined partial configuration for conversation template.

    This is an attribute of :class:`mlc_chat.ChatConfig`, which can then be passed in to the
    instantiation of a :class:`mlc_chat.ChatModule` instance to override the default
    setting in ``mlc-chat-config.json`` under the model folder. Note that we will
    first load the predefined template with the name specified in ``conv_template``.

    Since the configuration is partial, everything will be ``Optional``.

    Parameters
    ----------
    name : Optional[str]
        Name of the conversation.
    system : Optional[str]
        The prompt encoded before starting the chat.
    roles : Optional[List[str]]
        An array that describes the role names of the user and the model. These
        names are specific to the model being used.
    messages : Optional[List[str]]
        The chat history represented as an array of string pairs in the following
        format: ``[[role_0, msg_0], [role_1, msg_1], ...]``.
    offset : Optional[str]
        The offset used to begin the chat from the chat history. When offset
        is not ``0``, ``messages[0:offset-1]`` will be encoded.
    separator_style : Optional[int]
        Specifies whether we are in chat-bot mode (``0``) or pure LM prompt mode (``1``).
    seps : Optional[List[str]]
        An array of strings indicating the separators to be used after a user
        message and a model message respectively.
    role_msg_sep : Optional[str]
        A string indicating the separator between a role and a message.
    role_empty_sep : Optional[str]
        A string indicating the separator to append to a role when there is no message yet.
    stop_str : Optional[str]
        When the ``stop_str`` is encountered, the model will stop generating output.
    stop_tokens : Optional[List[int]]
        A list of token IDs that act as stop tokens.
    add_bos : Optional[bool]
        Determines whether a beginning-of-string (bos) token should be added
        before the input tokens.
    """

    name: Optional[str] = None
    system: Optional[str] = None
    roles: Optional[List[str]] = None
    messages: Optional[List[List[str]]] = None
    offset: Optional[str] = None
    separator_style: Optional[int] = None
    seps: Optional[List[str]] = None
    role_msg_sep: Optional[str] = None
    role_empty_sep: Optional[str] = None
    stop_str: Optional[str] = None
    stop_tokens: Optional[List[int]] = None
    add_bos: Optional[bool] = None

    def __post_init__(self):
        if self.messages is not None and self.offset is None:
            self.offset = len(self.messages)

@dataclass
class ChatConfig:
    r"""A dataclass that represents user-defined partial configuration for the
    chat config file.

    An instance of ``ChatConfig`` can be passed in to the instantiation of a
    :class:`mlc_chat.ChatModule` instance to override the default setting in
    ``mlc-chat-config.json`` under the model folder.

    Since the configuraiton is partial, everything will be ``Optional``.

    Note that we will exploit this class to also represent ``mlc-chat-config.json``
    during intermediate processing.

    Parameters
    ----------
    model_lib : Optional[str]
        The necessary model library to launch this model architecture. We recommend
        reuse model library when possible. For example, all LLaMA-7B models can
        use ``vicuna-v1-7b-{matching quantization scheme}``. So you can distribute
        LLaMA-7B weight variants and still use them in prebuilt MLC chat apps.
    local_id : Optional[str]
        Uniquely identifying the model in application. This is also used by
        command line interface app to specify which model to run.
    conv_template : Optional[str]
        The name of the conversation template that this chat uses.
    temperature : Optional[float]
        The temperature applied to logits before sampling. The default value is
        ``0.7``. A higher temperature encourages more diverse outputs, while a
        lower temperature produces more deterministic outputs.
    repetition_penalty : Optional[float]
        The repetition penalty controls the likelihood of the model generating
        repeated texts. The default value is set to ``1.0``, indicating that no
        repetition penalty is applied. Increasing the value reduces the
        likelihood of repeat text generation. However, setting a high
        ``repetition_penalty`` may result in the model generating meaningless
        texts. The ideal choice of repetition penalty may vary among models.

        For more details on how repetition penalty controls text generation, please
        check out the CTRL paper (https://arxiv.org/pdf/1909.05858.pdf).
    top_p : Optional[float]
        This parameter determines the set of tokens from which we sample during
        decoding. The default value is set to ``0.95``. At each step, we select
        tokens from the minimal set that has a cumulative probability exceeding
        the ``top_p`` parameter.

        For additional information on top-p sampling, please refer to this blog
        post: https://huggingface.co/blog/how-to-generate#top-p-nucleus-sampling.
    mean_gen_len : Optional[int]
    max_gen_len : Optional[int]
    shift_fill_factor : Optional[float]
    tokenizer_files : Optional[List[str]]
        List of tokenizer files of the model.
    conv_config : Optional[ConvConfig]
        The partial overriding configuration for conversation template. Will first
        load the predefined template with the name specified in ``conv_template``
        and then override some of the configuraitons specified in ``conv_config``.
    model_category : Optional[str]
        The category of the model's architecture (e.g. ``llama``, ``gpt_neox``, ``rwkv``).
    model_name : Optional[str]
        Name of the model (e.g. ``Llama-2-7b-chat-hf``).
    num_shards: Optional[str]
        Tensor parallel degree.
    max_window_size: Optional[str]
        Maximum kv cache window size.
    """

    model_lib: Optional[str] = None
    local_id: Optional[str] = None
    conv_template: Optional[str] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_p: Optional[float] = None
    mean_gen_len: Optional[int] = None
    max_gen_len: Optional[int] = None
    shift_fill_factor: Optional[float] = None
    tokenizer_files: Optional[List[str]] = None
    conv_config: Optional[ConvConfig] = None
    model_category: Optional[str] = None
    model_name: Optional[str] = None
    num_shards: Optional[int] = None
    max_window_size: Optional[int] = None
    vocab_size: Optional[int] = None
    sliding_window: Optional[int] = None

    @classmethod
    def _from_json(chat_config_cls, json_obj: dict):
        return chat_config_cls(
            **{
                k: v
                for k, v in json_obj.items()
                if k in inspect.signature(chat_config_cls).parameters
            }
        )

def get_chat_config(config_file_path: str, user_chat_config: Optional[ChatConfig]) -> ChatConfig:
    """Read in the config file in model path, then potentially override with user input.

    Parameters
    ----------
    config_file_path : str
        ``chat_file`` returned by ``_get_model_path()``.
    user_chat_config : Optional[ChatConfig]
        User's input, a partial ``ChatConfig`` to override the one in ``config_file_path``.

    Returns
    ------
    final_chat_config : ChatConfig
        ``ChatConfig`` corresponding to ``config_file_path``, overriden by ``user_chat_config``.
    """
    final_chat_config = None
    with open(config_file_path, mode="rt", encoding="utf-8") as f:
        json_object = json.load(f)
        final_chat_config = ChatConfig._from_json(json_object)
    if user_chat_config is not None:
        # We override using user's chat config
        for field in fields(user_chat_config):
            field_name = field.name
            field_value = getattr(user_chat_config, field_name)
            if field_value is not None:
                setattr(final_chat_config, field_name, field_value)
    return final_chat_config


def _prepare_inputs(
    sequence_ids,
    all_token_ids,
    all_slot_mappings,
    all_block_tables,
    sliding_window,
    dev,
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

    for sequence_id, token_ids in zip(sequence_ids, all_token_ids):
        if is_prefill:
            input_ids += token_ids
            prompt_len = len(token_ids)
            seq_lens.append(prompt_len)
            positions += range(prompt_len)
            slot_mapping += all_slot_mappings[sequence_id.request_id]

            if sliding_window:
                indices_within_window += range(
                    start_idx + max(0, prompt_len - sliding_window),
                    start_idx + prompt_len,
                )
                start_idx += prompt_len

        else:
            input_ids.append(token_ids[-1])
            pos = len(token_ids) - 1
            positions.append(pos)
            block_table = all_block_tables[sequence_id.request_id]
            max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
            block_tables.append(block_table)
            slot_mapping.append(all_slot_mappings[sequence_id.request_id][-1])

            if sliding_window:
                seq_lens.append(min(len(token_ids), sliding_window))
            else:
                seq_lens.append(len(token_ids))

    def to_ndarray_via_torch(arr, torch_dtype):
        return tvm.nd.from_dlpack(torch.tensor(arr, dtype=torch_dtype, device="cuda"))

    input_ids = to_ndarray_via_torch(input_ids, torch.int)
    positions = to_ndarray_via_torch(positions, torch.int)
    seq_lens = to_ndarray_via_torch(seq_lens, torch.int)
    slot_mapping = to_ndarray_via_torch(slot_mapping, torch.int)

    if is_prefill and sliding_window:
        indices_within_window = to_ndarray_via_torch(indices_within_window, torch.int)
    else:
        indices_within_window = None

    if not is_prefill:

        def _pad_to_max(x: List[int], max_len: int) -> List[int]:
            return x + [0] * (max_len - len(x))

        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in block_tables
        ]

        block_tables = to_ndarray_via_torch(padded_block_tables, torch.int)
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


class Model:
    def __init__(
        self,
        artifact_path,
        model_name,
        quant,
        num_shards,
        dev,
        sliding_window=None,
    ):
        self.mod, self.params, self.disco_session, self.config_file_path = get_tvm_model(
            artifact_path, model_name, quant, num_shards, dev
        )
        self.chat_config = get_chat_config(self.config_file_path, None)
        self.chat_config.num_shards = num_shards
        self.chat_config.sliding_window = sliding_window # this should migrate eventually into chat config
        self.dev = dev

        if self.chat_config.sliding_window:
            self.block_sliding_window = self.chat_config.sliding_window // CacheManager.block_size
        else:
            self.block_sliding_window = None

    def get_used_memory(self):
        if self.disco_session:
            params = self.params.debug_get_from_remote(0)

            get_used_memory_func = self.disco_session.get_global_func(
                "vm.memory_manager.get_used_memory"
            )
            # For Disco, we explicitly query the device 0.
            peak_memory = get_used_memory_func(
                tvm.device("cuda", 0)
            ).debug_get_from_remote(0)

            # TODO: temp hack to switch the VM allocator to eager recycling mode on all devices
            for i in range(1, self.chat_config.num_shards):
                get_used_memory_func(tvm.device("cuda", i)).debug_get_from_remote(i)
        else:
            params = self.params

            get_used_memory_func = tvm.get_global_func(
                "vm.memory_manager.get_used_memory"
            )
            peak_memory = get_used_memory_func(self.dev)

        param_bytes = sum(
            math.prod(param.shape) * np.dtype(param.dtype).itemsize for param in params
        )

        return peak_memory + param_bytes

    def profile_memory_usage(self, seq_lens):
        input_ids = [0] * sum(seq_lens)
        positions = []

        for s in seq_lens:
            positions += range(s)

        input_ids = tvm.nd.array(np.array(input_ids, dtype="int32"), self.dev)
        positions = tvm.nd.array(np.array(positions, dtype="int32"), self.dev)
        seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), self.dev)

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)

        self.mod["evaluate"](input_ids, positions, seq_lens, self.params)

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

        for request in requests:
            if isinstance(request, PrefillRequest):
                assert request.num_sequence == 1, "Multiple sequences not supported yet"
                sequence_id = SequenceId(request.request_id, 0)
                sequence_ids.append(sequence_id)
            else:
                sequence_id = request.sequence_id
                sequence_ids.append(request.sequence_id)

            all_token_ids.append(request.token_ids)
            sampling_params.append(request.sampling_params)

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
            cache.slot_mappings,
            cache.block_tables,
            self.chat_config.sliding_window,
            self.dev,
            is_prefill,
        )

        input_shape = input_ids.shape

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)
            slot_mapping = copy_to_worker_0(self.disco_session, slot_mapping)

        kv_cache = cache.cache

        if is_prefill:
            torch.cuda.nvtx.range_push(f"forward prefill {input_shape}")

            if self.chat_config.sliding_window:
                if self.disco_session:
                    indices_within_window = copy_to_worker_0(
                        self.disco_session, indices_within_window
                    )

                out = self.mod["prefill"](
                    input_ids,
                    positions,
                    seq_lens,
                    kv_cache,
                    slot_mapping,
                    indices_within_window,
                    self.params,
                )
            else:
                out = self.mod["prefill"](
                    input_ids, positions, seq_lens, kv_cache, slot_mapping, self.params
                )

            if self.disco_session:
                logits, _ = out.debug_get_from_remote(0)
            else:
                logits = out[
                    0
                ]  # Ignore returned KV cache since it is updated in-place anyway.
        else:
            torch.cuda.nvtx.range_push(f"forward decode {input_shape}")

            if self.disco_session:
                block_tables = copy_to_worker_0(self.disco_session, block_tables)

            out = self.mod["decode"](
                input_ids,
                positions,
                seq_lens,
                kv_cache,
                slot_mapping,
                block_tables,
                self.params,
            )

            if self.disco_session:
                logits, _ = out.debug_get_from_remote(0)
            else:
                logits = out[0]

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        next_tokens = sample(logits, sampling_params, self.chat_config.vocab_size)

        return [
            TextGenerationResult(
                sequence_id=sequence_id,
                generated_tokens=[new_token],
                error=None,
            )
            for sequence_id, new_token in zip(sequence_ids, next_tokens)
        ]


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


class PagedCacheModelTextGenerator:
    def __init__(self, model: Model):
        self.model = model

    def generate(
        self, requests: list[Union[PrefillRequest, DecodeRequest]], kv_cache
    ) -> list[TextGenerationResult]:
        prefill_requests = [r for r in requests if isinstance(r, PrefillRequest)]
        decode_requests = [r for r in requests if isinstance(r, DecodeRequest)]

        out = []
        if prefill_requests:
            out.extend(self.model.generate(prefill_requests, kv_cache))
        if decode_requests:
            out.extend(self.model.generate(decode_requests, kv_cache))

        return out


class Tokenizer:
    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer
        self.eos_token_id = self._tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


class ConversationTemplate:
    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer

    def apply(self, messages: list[ChatMessage]) -> str:
        return self._tokenizer.apply_chat_template(
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            tokenize=False,
            add_generation_prompt=True,
        )


class HfTokenizerModule:
    def __init__(self, model_name: str, artifact_path: str):
        model_path = os.path.join(artifact_path, "models", model_name)
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
        )
        self.tokenizer = Tokenizer(hf_tokenizer)
        self.conversation_template = ConversationTemplate(hf_tokenizer)


class PagedCacheModelModule:
    def __init__(
        self,
        model_name: str,
        artifact_path: str,
        quantization: str,
        num_shards: int,
        max_num_batched_tokens: int = 0,
        max_input_len: int = 0,
    ):
        model_path = os.path.join(artifact_path, "models", model_name)

        dev = tvm.device("cuda", 0)

        # TODO(amalyshe): HF config should be substituted by mlc-chat-config.json
        # we cannot get rid of HF config so far, since there is no number of parameters
        # in the chat config one:
        # sliding_window, hidden_size, num_attention_heads, get_num_key_value_heads(), num_hidden_layers
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as i_f:
            config = LlamaConfig(**json.load(i_f))

        model = Model(
            artifact_path,
            model_name,
            quantization,
            num_shards,
            dev,
            config.sliding_window,
        )

        if num_shards > 1:
            model.disco_session.sync_worker_0()

        num_kv_heads = config.get_num_key_value_heads() // num_shards
        head_size = config.hidden_size // config.num_attention_heads

        if max_num_batched_tokens > 0:
            assert max_input_len > 0
            assert max_num_batched_tokens % max_input_len == 0  # for simplicity

            num_seqs = max_num_batched_tokens // max_input_len
            num_blocks = get_num_cache_blocks(
                model,
                [max_input_len] * num_seqs,
                config.num_hidden_layers,
                num_kv_heads,
                head_size,
            )
        else:
            num_blocks = 500

        logger.info(f"Using {num_blocks} cache blocks.")

        cache_manager = CacheManager(
            num_blocks,
            config.num_hidden_layers,
            num_kv_heads,
            head_size,
            model.disco_session,
            config.sliding_window,
        )

        self.text_generator = PagedCacheModelTextGenerator(model)
        self.cache_manager = cache_manager

        tokenizer_module = HfTokenizerModule(model_name, artifact_path)
        self.tokenizer = tokenizer_module.tokenizer
        self.conversation_template = tokenizer_module.conversation_template
