import math
import os
import json
from collections import defaultdict
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

import torch
from transformers import LlamaTokenizer

import tvm
from tvm import relax

from mlc_llm.relax_model.llama import LlamaConfig
from mlc_llm import utils


def init_cache_blocks(head_size, num_layers, num_heads, block_size, num_gpu_blocks, dev):
    element_size = 2
    x = 16 // element_size

    key_block_shape = (num_heads, head_size // x, block_size, x)
    value_block_shape = (num_heads, head_size, block_size)

    gpu_cache = ()
    for _ in range(num_layers):
        key_blocks = tvm.nd.empty(
            (num_gpu_blocks, *key_block_shape),
            dtype="float16",
            device=dev,
        )
        value_blocks = tvm.nd.empty(
            (num_gpu_blocks, *value_block_shape),
            dtype="float16",
            device=dev,
        )
        gpu_cache += (key_blocks, value_blocks)
    return gpu_cache


class KVCache:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_size, dev):
        self.cache = init_cache_blocks(
            head_size, num_layers, num_heads, block_size, num_blocks, dev
        )
        self.block_tables = defaultdict(list)
        self.block_size = block_size


class CacheManager:
    def __init__(self, num_layers, num_heads, head_size, dev):
        # TODO: Hardcoded for now
        block_size = 16
        num_blocks = 500

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.kv_cache = KVCache(num_blocks, block_size, num_layers, num_heads, head_size, dev)

    def set_size(self, request_ids: List[int], target_sizes: List[int]):
        for id, size in zip(request_ids, target_sizes):
            num_needed_block = math.ceil(size / self.block_size)

            if id in self.kv_cache.block_tables and size == 0:
                self.free_blocks.extend(self.kv_cache.block_tables[id])
                del self.kv_cache.block_tables[id]

            elif (
                id in self.kv_cache.block_tables
                and len(self.kv_cache.block_tables[id]) < num_needed_block
            ):
                # Decoding, need to allocate a new block for this request
                assert len(self.kv_cache.block_tables[id]) + 1 == num_needed_block
                self.kv_cache.block_tables[id].append(self.free_blocks.pop())

            elif id not in self.kv_cache.block_tables:
                assert len(self.free_blocks) >= num_needed_block, "Not enough free blocks."

                for _ in range(num_needed_block):
                    self.kv_cache.block_tables[id].append(self.free_blocks.pop())

    def get(self):
        return self.kv_cache


@dataclass
class SequenceGenerationRequest:
    request_id: int
    token_ids: List[int]
    start_position: int
    # sampling_params: SamplingParams


@dataclass
class SequenceGenerationResponse:
    request_id: int
    token_id: int


def get_tvm_model(artifact_path, model, dev):
    const_params = utils.load_params(artifact_path, dev)
    ex = tvm.runtime.load_module(
        os.path.join(
            artifact_path,
            f"{model}-q4f16_ft-cuda.so",
        )
    )
    vm = relax.VirtualMachine(ex, dev)

    return vm, const_params


class Model:
    def __init__(self, artifact_path, model_name, dev):
        self.vm, self.params = get_tvm_model(artifact_path, model_name, dev)
        self.dev = dev

    def generate(
        self, requests: List[SequenceGenerationRequest], cache: KVCache, is_prompt: bool
    ) -> List[SequenceGenerationResponse]:
        block_tables = []
        seq_lens = []
        input_ids = []
        slot_mappings = []
        positions = []
        max_num_blocks_per_seq = 0
        block_size = cache.block_size

        for request in requests:
            block_table = cache.block_tables[request.request_id]
            seq_lens.append(len(request.token_ids))

            if is_prompt:
                input_ids += request.token_ids
                positions += range(seq_lens[-1])
            else:
                input_ids.append(request.token_ids[-1])
                positions.append(seq_lens[-1] - 1)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
                block_tables.append(block_table)

            for i in range(len(request.token_ids)):
                block_number = block_table[i // block_size]
                block_offset = i % block_size
                slot = block_number * block_size + block_offset
                slot_mappings.append(slot)

        input_ids = tvm.nd.array(np.array(input_ids, dtype="int32"), self.dev)
        positions = tvm.nd.array(np.array(positions, dtype="int32"), self.dev)
        seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), self.dev)
        slot_mapping = tvm.nd.array(np.array(slot_mappings, dtype="int32"), self.dev)
        kv_cache = cache.cache

        if is_prompt:
            logits, kv_cache_next = self.vm["prefill"](
                input_ids, positions, seq_lens, kv_cache, slot_mapping, self.params
            )
        else:

            def _pad_to_max(x: List[int], max_len: int) -> List[int]:
                return x + [0] * (max_len - len(x))

            padded_block_tables = [
                _pad_to_max(block_table, max_num_blocks_per_seq) for block_table in block_tables
            ]

            block_tables = tvm.nd.array(
                np.array(np.vstack(padded_block_tables), dtype="int32"), self.dev
            )

            logits, kv_cache_next = self.vm["decode"](
                input_ids, positions, seq_lens, kv_cache, slot_mapping, block_tables, self.params
            )

        cache.cache = kv_cache_next

        next_tokens = torch.argmax(torch.from_dlpack(logits), -1).cpu().numpy()

        responses = []

        for request, new_token in zip(requests, next_tokens):
            responses.append(SequenceGenerationResponse(request.request_id, new_token))

        return responses


def test():
    artifact_path = "/home/masahi/projects/dev/mlc-llm/dist/vicuna-v1-7b-q4f16_ft"
    model_path = "/home/masahi/projects/dev/mlc-llm/dist/models/vicuna-v1-7b"
    model_name = "vicuna-v1-7b"

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as i_f:
        config = LlamaConfig(json.load(i_f))

    tokenizer = LlamaTokenizer.from_pretrained(
        os.path.join(artifact_path, "params"), trust_remote_code=True
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    batched_token_ids = [tokenizer.encode(p) for p in prompts]
    request_ids = list(range(len(prompts)))
    target_sizes = []
    requests = []

    dev = tvm.device("cuda", 0)

    for token_ids, request_id in zip(batched_token_ids, request_ids):
        request_ids.append(request_id)
        target_sizes.append(len(token_ids))
        requests.append(SequenceGenerationRequest(request_id, token_ids, 0))

    cache_manager = CacheManager(
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
        dev,
    )
    cache = cache_manager.get()

    cache_manager.set_size(request_ids, target_sizes)

    model = Model(artifact_path, model_name, dev)
    out = model.generate(requests, cache, True)

    return

    num_steps = 13

    generated = ["" for _ in range(len(prompts))]

    for _ in range(num_steps):
        for i, response in enumerate(out):
            new_token = response.token_id
            requests[i].token_ids.append(new_token)
            generated[i] += tokenizer.decode(new_token)
            target_sizes[i] += 1

        cache_manager.set_size(request_ids, target_sizes)

        out = model.generate(requests, cache, False)

    for p, g in zip(prompts, generated):
        print("Prompt = '{}', generate tokens = '{}'".format(p, g))


test()
