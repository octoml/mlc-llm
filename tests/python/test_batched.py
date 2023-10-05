import argparse
import math
import os
import json
from collections import defaultdict
from typing import List
from dataclasses import dataclass

import torch
import numpy as np

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


class SamplingParams:
    def __init__(
        self,
        greedy=True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ):
        self.greedy = greedy
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


@dataclass
class SequenceGenerationRequest:
    request_id: int
    token_ids: List[int]
    start_position: int
    sampling_params: SamplingParams


@dataclass
class SequenceGenerationResponse:
    request_id: int
    token_id: int


def sample(logits, sampling_params):
    logits = torch.from_dlpack(logits)
    # TODO: Support beam search?
    do_greedy = [p.greedy for p in sampling_params]
    # TODO: Support per-type batched sampling like vllm.
    assert all(do_greedy) or all([not greedy for greedy in do_greedy])

    if all(do_greedy):
        return torch.argmax(logits, -1).cpu().numpy()

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, True).cpu().numpy()[:, 0]


def get_tvm_model(artifact_path, model, quantization, dev):
    const_params = utils.load_params(artifact_path, dev)
    ex = tvm.runtime.load_module(
        os.path.join(
            artifact_path,
            f"{model}-{quantization}-cuda.so",
        )
    )
    vm = relax.VirtualMachine(ex, dev)

    return vm, const_params


class Model:
    def __init__(self, artifact_path, model_name, quant, dev):
        self.vm, self.params = get_tvm_model(artifact_path, model_name, quant, dev)
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
        sampling_params = []

        for request in requests:
            block_table = cache.block_tables[request.request_id]
            seq_lens.append(len(request.token_ids))
            sampling_params.append(request.sampling_params)

            if is_prompt:
                input_ids += request.token_ids
                positions += range(seq_lens[-1])

                for i in range(len(request.token_ids)):
                    block_number = block_table[i // block_size]
                    block_offset = i % block_size
                    slot = block_number * block_size + block_offset
                    slot_mappings.append(slot)
            else:
                input_ids.append(request.token_ids[-1])
                pos = seq_lens[-1] - 1
                positions.append(pos)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
                block_tables.append(block_table)

                block_number = block_table[pos // block_size]
                block_offset = pos % block_size
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

        next_tokens = sample(logits, sampling_params)

        return [
            SequenceGenerationResponse(request.request_id, new_token)
            for request, new_token in zip(requests, next_tokens)
        ]


def parse_args():
    # Example
    # python build.py --model vicuna-v1-7b --quantization q4f16_ft ache=0 --max-seq-len 768 --batched
    # python tests/python/test_batched.py --local-id vicuna-v1-7b-q4f16_ft
    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--artifact-path", type=str, default="dist")
    parsed = args.parse_args()
    parsed.model, parsed.quantization = parsed.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(parsed)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, f"{parsed.model}-{parsed.quantization.name}-batched"
    )
    return parsed


def test(args):
    quantization = args.quantization.name
    artifact_path = args.artifact_path
    model_name = args.model
    model_path = f"dist/models/{model_name}"

    dev = tvm.device("cuda", 0)

    model = Model(artifact_path, model_name, quantization, dev)

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as i_f:
        config = LlamaConfig(**json.load(i_f))

    tokenizer = LlamaTokenizer.from_pretrained(
        os.path.join(artifact_path, "params"), trust_remote_code=True
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Shohei Ohtani is",
    ]

    batched_token_ids = [tokenizer.encode(p) for p in prompts]
    prompts_len = [len(ids) for ids in batched_token_ids]
    request_ids = list(range(len(prompts)))
    target_sizes = []
    requests = []

    for token_ids, request_id in zip(batched_token_ids, request_ids):
        sampling_params = SamplingParams(greedy=True, temperature=0.8, top_p=0.95)
        request_ids.append(request_id)
        target_sizes.append(len(token_ids))
        requests.append(SequenceGenerationRequest(request_id, token_ids, 0, sampling_params))

    cache_manager = CacheManager(
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
        dev,
    )
    cache = cache_manager.get()

    cache_manager.set_size(request_ids, target_sizes)

    out = model.generate(requests, cache, True)

    num_steps = 20

    for s in range(num_steps):
        for i, response in enumerate(out):
            new_token_id = response.token_id
            requests[i].token_ids.append(new_token_id)
            target_sizes[i] += 1

        cache_manager.set_size(request_ids, target_sizes)

        out = model.generate(requests, cache, False)

    output_tokens = [
        tokenizer.convert_ids_to_tokens(
            requests[i].token_ids[prompts_len[i] :], skip_special_tokens=True
        )
        for i in range(len(requests))
    ]

    generated = [tokenizer.convert_tokens_to_string(tokens) for tokens in output_tokens]

    for p, g in zip(prompts, generated):
        print("Prompt = '{}', generated text = '{}'".format(p, g))


if __name__ == "__main__":
    test(parse_args())
