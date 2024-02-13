import tempfile
import os
import uvicorn

from .api import create_app
from .engine import AsyncEngineConnector
from .utils import get_default_mlc_serve_argparser, postproc_mlc_serve_args, create_mlc_engine


def parse_args():
    parser = get_default_mlc_serve_argparser(description="Launch mlc-serve")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    postproc_mlc_serve_args(args)
    return args


def run_server():
    args = parse_args()

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = temp_dir

        engine = create_mlc_engine(args, start_engine=False)
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
