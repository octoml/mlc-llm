import argparse
import math
import os
import json
from collections import defaultdict
from typing import List
from dataclasses import dataclass

import numpy as np

import tvm
from tvm import relax
from tvm.runtime import disco as di

import torch
from transformers import LlamaTokenizer

from mlc_llm.relax_model.llama import LlamaConfig
from mlc_llm import utils


class KVCache:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_size, disco_session):
        if disco_session:
            init_cache_func = disco_session.get_global_func("tvm.contrib.vllm.allocate_kv_cache")
        else:
            init_cache_func = tvm.get_global_func("tvm.contrib.vllm.allocate_kv_cache")

        self.cache = init_cache_func(head_size, num_layers, num_heads, block_size, num_blocks)

        self.block_tables = defaultdict(list)
        self.block_size = block_size


class CacheManager:
    def __init__(self, num_layers, num_heads, head_size, disco_session=None):
        # TODO: Hardcoded for now
        block_size = 16
        num_blocks = 500

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.kv_cache = KVCache(
            num_blocks, block_size, num_layers, num_heads, head_size, disco_session
        )

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
    # TODO: Support beam search?
    do_greedy = [p.greedy for p in sampling_params]
    # TODO: Support per-type batched sampling like vllm.
    assert all(do_greedy) or all([not greedy for greedy in do_greedy])

    temperatures = [p.temperature for p in sampling_params]
    if any(t != 1.0 for t in temperatures):
        t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
        logits.div_(t.unsqueeze(dim=1))

    top_ps = [p.top_p for p in sampling_params]
    top_ks = [p.top_k if p.top_k != -1 else vocab_size for p in sampling_params]

    do_top_p = any(p < 1.0 for p in top_ps)
    do_top_k = any(k != vocab_size for k in top_ks)

    if do_top_p or do_top_k:
        logits = _apply_top_p_top_k(logits, top_ps, top_ks)

    if all(do_greedy):
        return torch.argmax(logits, -1).cpu().numpy()

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, True).cpu().numpy()[:, 0]


def load_params_disco(artifact_path, lib_path, num_shards):
    sess = di.ProcessSession(num_workers=num_shards)
    devices = range(num_shards)
    sess.init_ccl("nccl", *devices)
    module = sess.load_vm_module(lib_path)

    loader_create = sess.get_global_func("runtime.disco.ShardLoader")
    metadata_path = os.path.join(artifact_path, "params", "ndarray-cache.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        ndarray_cache_metadata = f.read()

    loader = loader_create(metadata_path, ndarray_cache_metadata, "", module)
    loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAll")
    params = loader_load(loader)

    return module, params, sess


def copy_to_worker_0(sess: di.Session, host_array):
    x_array = sess.empty(host_array.shape, host_array.dtype)
    sess.copy_to_worker_0(host_array, x_array)
    return x_array


def get_tvm_model(artifact_path, model, quantization, num_shards, dev):
    lib_path = os.path.join(artifact_path, f"{model}-{quantization}-cuda.so")

    if num_shards == 1:
        ex = tvm.runtime.load_module(lib_path)
        vm = relax.VirtualMachine(ex, dev)
        params = utils.load_params(artifact_path, dev)
        return vm.module, params, None

    return load_params_disco(artifact_path, lib_path, num_shards)


class Model:
    def __init__(self, artifact_path, model_name, quant, vocab_size, num_shards, dev):
        self.mod, self.params, self.disco_session = get_tvm_model(
            artifact_path, model_name, quant, num_shards, dev
        )
        self.dev = dev
        self.vocab_size = vocab_size

    def get_used_memory(self):
        if self.disco_session:
            params = self.params.debug_get_from_remote(0)

            get_used_memory_func = self.disco_session.get_global_func(
                "vm.memory_manager.get_used_memory"
            )
            # For Disco, we explicitly query the device 0.
            peak_memory = get_used_memory_func(tvm.device("cuda", 0)).debug_get_from_remote(0)

        else:
            params = self.params

            get_used_memory_func = tvm.get_global_func("vm.memory_manager.get_used_memory")
            peak_memory = get_used_memory_func(self.dev)

        param_bytes = 0

        for param in params:
            param_bytes += param.numpy().nbytes

        return peak_memory + param_bytes

    def generate(
        self, requests: List[SequenceGenerationRequest], cache: KVCache, is_prompt: bool
    ) -> List[SequenceGenerationResponse]:
        block_tables = []
        seq_lens = []
        input_ids = []
        slot_mapping = []
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
                    slot_mapping.append(slot)
            else:
                input_ids.append(request.token_ids[-1])
                pos = seq_lens[-1] - 1
                positions.append(pos)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq, len(block_table))
                block_tables.append(block_table)

                block_number = block_table[pos // block_size]
                block_offset = pos % block_size
                slot = block_number * block_size + block_offset
                slot_mapping.append(slot)

        input_ids = tvm.nd.array(np.array(input_ids, dtype="int32"), self.dev)
        positions = tvm.nd.array(np.array(positions, dtype="int32"), self.dev)
        seq_lens = tvm.nd.array(np.array(seq_lens, dtype="int32"), self.dev)
        slot_mapping = tvm.nd.array(np.array(slot_mapping, dtype="int32"), self.dev)

        if self.disco_session:
            input_ids = copy_to_worker_0(self.disco_session, input_ids)
            positions = copy_to_worker_0(self.disco_session, positions)
            seq_lens = copy_to_worker_0(self.disco_session, seq_lens)
            slot_mapping = copy_to_worker_0(self.disco_session, slot_mapping)

        kv_cache = cache.cache

        if is_prompt:
            out = self.mod["prefill"](
                input_ids, positions, seq_lens, kv_cache, slot_mapping, self.params
            )

            if self.disco_session:
                logits, _ = out.debug_get_from_remote(0)
            else:
                logits = out[0]  # Ignore returned KV cache since it is updated in-place anyway.
        else:

            def _pad_to_max(x: List[int], max_len: int) -> List[int]:
                return x + [0] * (max_len - len(x))

            padded_block_tables = [
                _pad_to_max(block_table, max_num_blocks_per_seq) for block_table in block_tables
            ]

            block_tables_np = np.vstack(padded_block_tables).astype("int32")
            block_tables = tvm.nd.array(np.array(block_tables_np, dtype="int32"), self.dev)

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

        next_tokens = sample(logits, sampling_params, self.vocab_size)

        return [
            SequenceGenerationResponse(request.request_id, new_token)
            for request, new_token in zip(requests, next_tokens)
        ]


def parse_args():
    # Example
    # python build.py --model vicuna-v1-7b --quantization q4f16_ft --use-cache=0 --max-seq-len 768 --batched
    # python tests/python/test_batched.py --local-id vicuna-v1-7b-q4f16_ft
    #
    # For Disco:
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768  --batched --build-model-only --num-shards 2
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768  --batched --convert-weight-only
    # /opt/bin/cuda-reserve.py  --num-gpus 2 python tests/python/test_batched.py --local-id vicuna-v1-7b-q0f16 --num-shards 2
    args = argparse.ArgumentParser()
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--num-shards", type=int, default=1)
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

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as i_f:
        config = LlamaConfig(**json.load(i_f))

    model = Model(artifact_path, model_name, quantization, config.vocab_size, args.num_shards, dev)

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
    prompts_len = [len(ids) for ids in batched_token_ids]
    request_ids = list(range(len(prompts)))
    target_sizes = []
    requests = []

    for token_ids, request_id in zip(batched_token_ids, request_ids):
        # sampling_params = SamplingParams(greedy=False, temperature=0.8, top_p=0.95)
        sampling_params = SamplingParams(greedy=True)
        request_ids.append(request_id)
        target_sizes.append(len(token_ids))
        requests.append(SequenceGenerationRequest(request_id, token_ids, 0, sampling_params))

    num_kv_heads = config.get_num_key_value_heads() // args.num_shards

    cache_manager = CacheManager(
        config.num_hidden_layers,
        num_kv_heads,
        config.hidden_size // config.num_attention_heads,
        model.disco_session,
    )
    cache = cache_manager.get()

    cache_manager.set_size(request_ids, target_sizes)

    out = model.generate(requests, cache, True)
    print(model.get_used_memory() / 1e9)
    return

    num_steps = 20

    for _ in range(num_steps):
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
