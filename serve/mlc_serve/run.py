import argparse
import logging
import logging.config
import os
import uvicorn
#from mlc_llm import utils

from .api import create_app
from .engine import AsyncEngineConnector, MLCServeEngineConfig
from .engine.staging_engine import StagingInferenceEngine
from .engine.sync_engine import SynchronousInferenceEngine
from .model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule


def parse_args():
    # Example
    # python build.py --model vicuna-v1-7b --quantization q4f16_ft --use-cache=0 --max-seq-len 768 --batched
    # python tests/python/test_batched.py --local-id vicuna-v1-7b-q4f16_ft
    #
    # For Disco:
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768  --batched --build-model-only --num-shards 2
    # python build.py --model vicuna-v1-7b --quantization q0f16 --use-cache=0 --max-seq-len 768  --batched --convert-weight-only
    # /opt/bin/cuda-reserve.py  --num-gpus 2 python -m mlc_serve --local-id vicuna-v1-7b-q0f16 --num-shards 2
    #
    # Profile the gpu memory usage, and use the maximum number of cache blocks possible:
    # /opt/bin/cuda-reserve.py  --num-gpus 2 python -m mlc_serve --local-id vicuna-v1-7b-q0f16 --num-shards 2 --max-num-batched-tokens 2560 --max-input-len 256

    args = argparse.ArgumentParser()
    args.add_argument("--host", type=str, default="127.0.0.1")
    args.add_argument("--port", type=int, default=8000)
    args.add_argument("--local-id", type=str, required=True)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--use-staging-engine", action="store_true")
    args.add_argument("--max-batched-tokens", type=int, default=-1)
    args.add_argument("--max-input-len", type=int, default=-1)
    args.add_argument("--min-decode-steps", type=int, default=12)
    args.add_argument("--max-decode-steps", type=int, default=16)
    args.add_argument("--prompt-allocate-ratio", type=float, default=2.0)
    args.add_argument("--debug-logging", action="store_true")
    parsed = args.parse_args()
    return parsed


def setup_logging(args):
    level = "INFO"
    if args.debug_logging:
        level = "DEBUG"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "level": level,  # Set the handler's log level to DEBUG
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": level,  # Set the logger's log level to DEBUG
        },
        "mlc_serve.engine.sync_engine": {"level": level},
        "mlc_serve.engine.staging_engine": {"level": level},
        "mlc_serve.engine.staging_engine_worker": {"level": level},
    }
    logging.config.dictConfig(logging_config)


def create_engine(
    args: argparse.Namespace,
):
    # `model_artifact_path` has the following structure
    #  |- compiled artifact (.so)
    #  |- `build_config.json`: stores compile-time info, such as `num_shards` and `quantization`. 
    #  |- params/ : stores weights in mlc format and `ndarray-cache.json`. 
    #  |            `ndarray-cache.json` is especially important for Disco.
    #  |- model/ : stores info from hf model cards such as max context length and tokenizer
    model_artifact_path = os.path.join(args.artifact_path, args.local_id)
    if not os.path.exists(model_artifact_path):
        raise Exception(f"Invalid local id: {args.local_id}")

    # Set the engine config
    engine_config = MLCServeEngineConfig._from_json({
        "use_staging_engine": args.use_staging_engine,
        "max_batched_tokens": args.max_batched_tokens, 
        "max_input_len": args.max_input_len,    
        "min_decode_steps": args.min_decode_steps,
        "max_decode_steps": args.max_decode_steps,
        "prompt_allocate_ratio": args.prompt_allocate_ratio
    })
  
    if args.use_staging_engine:
        return StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(model_artifact_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": model_artifact_path,
                "engine_config": engine_config,
            },
        )
    else:
        return SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path = model_artifact_path,
                engine_config = engine_config,
            )
        )


def run_server():
    args = parse_args()
    setup_logging(args)

    engine = create_engine(args)
    connector = AsyncEngineConnector(engine)
    app = create_app(connector)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    run_server()
