import math
import os
import json
from collections import defaultdict
from typing import List, Optional, Tuple
from dataclasses import dataclass

from transformers import AutoConfig, LlamaTokenizer

import tvm
from mlc_llm.relax_model.llama import LlamaConfig


def init_cache_blocks(head_size, num_layers, num_heads, block_size, num_gpu_blocks):
    element_size = 2
    x = 16 // element_size

    key_block_shape = (num_heads, head_size // x, block_size, x)
    value_block_shape = (num_heads, head_size, block_size)

    gpu_cache = ()
    for _ in range(num_layers):
        key_blocks = tvm.nd.empty(
            (num_gpu_blocks, *key_block_shape),
            dtype="float16",
            device="cuda",
        )
        value_blocks = tvm.nd.empty(
            (num_gpu_blocks, *value_block_shape),
            dtype="float16",
            device="cuda",
        )
        gpu_cache += (key_blocks, value_blocks)
    return gpu_cache


class KVCache:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_size):
        self.cache = init_cache_blocks(
            head_size, num_layers, num_heads, block_size, num_blocks
        )
        self.block_tables = defaultdict(list)
        self.block_size = block_size


class CacheManager:
    def __init__(self, num_layers, num_heads, head_size):
        # TODO: Hardcoded for now
        block_size = 16
        num_blocks = 500

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.kv_cache = KVCache(
            num_blocks, block_size, num_layers, num_heads, head_size
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
                assert (
                    len(self.free_blocks) >= num_needed_block
                ), "Not enough free blocks."

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


class Model:
    def __init__(self):
        self.pt_model = get_model()

    def generate(
        self, requests: List[SequenceGenerationRequest], cache: KVCache, is_prompt: bool
    ) -> List[SequenceGenerationResponse]:
        seq_groups = []
        request_ids = []

        for request in requests:
            request_id = request.request_id
            seq = Sequence(request_id, "", request.token_ids, cache.block_size)
            seq_groups.append(
                SequenceGroup(request_id, [seq], request.sampling_params, time.time())
            )
            request_ids.append(request_id)

        seq_group_metadata_list = get_seq_group_metadata(
            seq_groups, cache.block_tables, is_prompt
        )

        input_ids, positions, input_metadata = prepare_inputs(
            seq_group_metadata_list, cache.block_size
        )

        with torch.no_grad():
            out = self.pt_model.forward(
                input_ids, positions, cache.cache, input_metadata
            )

        responses = []

        for request_id, samples in zip(request_ids, out):
            new_token = samples[0].output_token
            responses.append(SequenceGenerationResponse(request_id, new_token))

        return responses


def test():
    artifact_path = ""
    model_path = ""

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

    model = Model()

    cache_manager = CacheManager(config.num_hidden_layers,
                                 config.num_attention_heads,
                                 config.hidden_size // config.num_attention_heads)
    cache = cache_manager.get()

    batched_token_ids = [tokenizer.encode(p) for p in prompts]
    request_ids = list(range(len(prompts)))
    target_sizes = []
    requests = []

    for token_ids, request_id in zip(batched_token_ids, request_ids):
        request_ids.append(request_id)
        target_sizes.append(len(token_ids))
        requests.append(
            SequenceGenerationRequest(request_id, token_ids, 0)
        )

    cache_manager.set_size(request_ids, target_sizes)

    out = model.generate(requests, cache, True)

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
