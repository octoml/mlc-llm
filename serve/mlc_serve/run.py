import argparse
import tempfile
import os
import uvicorn

from .api import create_app
from .engine import AsyncEngineConnector, get_engine_config
from .engine.staging_engine import StagingInferenceEngine
from .engine.sync_engine import SynchronousInferenceEngine
from .model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule
from .utils import get_default_mlc_serve_argparser, postproc_mlc_serve_args


def parse_args():
    parser = get_default_mlc_serve_argparser(description="Launch mlc-serve")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    postproc_mlc_serve_args(args)
    return args


def create_engine(
    args: argparse.Namespace,
):
    """
    `model_artifact_path` has the following structure
    |- compiled artifact (.so)
    |- `build_config.json`: stores compile-time info, such as `num_shards` and `quantization`.
    |- params/ : stores weights in mlc format and `ndarray-cache.json`.
    |            `ndarray-cache.json` is especially important for Disco.
    |- model/ : stores info from hf model cards such as max context length and tokenizer
    """
    # Set the engine config
    engine_config = get_engine_config(
        {
            "use_staging_engine": args.use_staging_engine,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "min_decode_steps": args.min_decode_steps,
            "max_decode_steps": args.max_decode_steps,
        }
    )

    if args.use_staging_engine:
        return StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(args.model_artifact_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": args.model_artifact_path,
                "engine_config": engine_config,
            },
        )
    else:
        return SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path=args.model_artifact_path,
                engine_config=engine_config,
            )
        )


def run_server():
    args = parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = temp_dir

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
