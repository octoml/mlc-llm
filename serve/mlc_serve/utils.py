"""Utility functions for mlc-serve"""

from pathlib import Path
import os
import torch
import random
import argparse

from mlc_serve.engine import get_engine_config, InferenceEngine
from mlc_serve.logging_utils import configure_logging
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.base import (
    get_model_artifact_config,
    ModelArtifactConfig,
    get_hf_config,
)
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule


def get_default_mlc_serve_argparser(description="", allow_override=False):
    if allow_override:
        parser = argparse.ArgumentParser(
            description=description, conflict_handler="resolve"
        )
    else:
        parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="dist")
    parser.add_argument("--use-sync-engine", action="store_true")
    parser.add_argument("--num-sequences-to-sample", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seq", type=int, default=2048)
    parser.add_argument("--min-decode-steps", type=int, default=32)
    parser.add_argument("--max-decode-steps", type=int, default=56)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--debug-logging", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)  # Needed for PT models
    return parser


def postproc_mlc_serve_args(args):
    log_level = "DEBUG" if args.debug_logging else "INFO"
    configure_logging(enable_json_logs=False, log_level=log_level)
    args.model_artifact_path = Path(os.path.join(args.artifact_path, args.local_id))
    if not os.path.exists(args.model_artifact_path):
        raise Exception(f"Invalid local id: {args.local_id}")

    args.use_staging_engine = not args.use_sync_engine

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)


def create_mlc_engine(args: argparse.Namespace, start_engine=True) -> InferenceEngine:
    model_type = "tvm"
    num_shards = None

    if not os.path.exists(args.model_artifact_path.joinpath("build_config.json")):
        model_type = "torch"
        num_shards = args.num_shards

        assert (
            num_shards is not None
        ), "--num-shards needs to be provided for PT models."

        if num_shards > 1:
            import torch

            torch.multiprocessing.set_start_method("spawn")

    engine_config = get_engine_config(
        {
            "use_staging_engine": args.use_staging_engine,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seq": args.max_num_seq,
            "min_decode_steps": args.min_decode_steps,
            "max_decode_steps": args.max_decode_steps,
            "model_type": model_type,
            "num_shards": num_shards,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
    )

    if model_type == "tvm":
        model_artifact_config = get_model_artifact_config(args.model_artifact_path)
    else:
        hf_config = get_hf_config(args.model_artifact_path)

        model_artifact_config = ModelArtifactConfig(
            model_artifact_path=str(args.model_artifact_path),
            num_shards=num_shards,
            quantization=None,
            max_context_length=hf_config.max_position_embeddings,
            vocab_size=hf_config.vocab_size,
            sliding_window=hf_config.sliding_window,
            num_key_value_heads=hf_config.num_key_value_heads // num_shards,
            num_attention_heads=hf_config.num_attention_heads,
            num_hidden_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
        )

    engine: InferenceEngine

    if args.use_staging_engine:
        if model_type == "tvm":
            tokenizer_path = args.model_artifact_path.joinpath("model")
        else:
            tokenizer_path = args.model_artifact_path

        engine = StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(tokenizer_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": args.model_artifact_path,
                "engine_config": engine_config,
                "model_artifact_config": model_artifact_config,
            },
        )

        if start_engine:
            engine.start()
    else:
        engine = SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path=args.model_artifact_path,
                engine_config=engine_config,
                model_artifact_config=model_artifact_config,
            )
        )

    return engine
