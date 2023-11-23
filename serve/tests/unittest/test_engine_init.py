import torch

import argparse
import json
import random
import os
import pytest

from mlc_llm import utils
from mlc_serve.engine import (
    Request,
    ChatMessage,
    DebugOptions,
    SamplingParams,
    StoppingCriteria,
    get_engine_config
)
from mlc_serve.model.paged_cache_model import PagedCacheModelModule


def test_insufficient_cache_block_fail(artifact_path):
    model_artifact_path = os.path.join(artifact_path, "codellama-13b-instruct-hf-q0f16")

    if not os.path.exists(os.path.join(model_artifact_path)):
        return

    def try_init(max_num_seqs):
        engine_config = get_engine_config({
            "use_staging_engine": False,
            "max_num_sequences": max_num_seqs,
            "max_input_len": 16384,
            "min_decode_steps": 12,
            "max_decode_steps": 16,
            "prompt_allocate_ratio": 2.0
        })

        PagedCacheModelModule(
            model_artifact_path = model_artifact_path,
            engine_config = engine_config,
        )

    with pytest.raises(RuntimeError) as e_info:
        # This test assumes that 80GB VRAM is available.
        try_init(2)

    assert "Try reducing" in str(e_info.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", type=str, default="dist")
    args = parser.parse_args()

    test_insufficient_cache_block_fail(args.artifact_path)
