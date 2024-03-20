import argparse

from mlc_serve.engine import (
    ChatMessage,
    DebugOptions,
    Request,
    SamplingParams,
    StoppingCriteria,
)
from mlc_serve.utils import (
    create_mlc_engine,
    get_default_mlc_serve_argparser,
    postproc_mlc_serve_args,
)


def _test_bad_json_schema(args: argparse.Namespace):
    engine = create_mlc_engine(args)

    sampling_params = SamplingParams(
        temperature=0.0,
        vocab_size=engine.model_artifact_config.vocab_size,
    )

    sampling_params_json = SamplingParams(
        temperature=0.0,
        vocab_size=engine.model_artifact_config.vocab_size,
        json_schema={
            "schema": {
                "$defs": {
                    "Person": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                    "Family": {
                        "type": "object",
                        "properties": {
                            "last_name": {"type": "string"},
                            "members": {
                                "type": "array",
                                "items": {"$ref": "#/$defs/Person"},
                            },
                        },
                    },
                },
            }
        },
    )
    prompt = "Hello, my name is"

    # The first request should fail, but the engine should be able to continue
    # processing the second request.
    engine.add(
        [
            Request(
                request_id="0",
                messages=[ChatMessage(role="user", content=prompt)],
                sampling_params=sampling_params_json,
                stopping_criteria=StoppingCriteria(max_tokens=20, stop_sequences=None),
                debug_options=DebugOptions(prompt=prompt),
            )
        ]
    )
    engine.add(
        [
            Request(
                request_id="1",
                messages=[ChatMessage(role="user", content=prompt)],
                sampling_params=sampling_params,
                stopping_criteria=StoppingCriteria(max_tokens=20, stop_sequences=None),
                debug_options=DebugOptions(prompt=prompt),
            )
        ]
    )

    results = []

    while engine.has_pending_requests():
        results.append(engine.step())

    if args.use_staging_engine:
        engine.stop()

    # The second request should succeed
    assert len(results) > 1
    # The first request fails with an empty output and an error message
    assert len(results[0].outputs[0].sequences) == 0 and results[0].outputs[0].error is not None


if __name__ == "__main__":
    parser = get_default_mlc_serve_argparser("test engine")
    args = parser.parse_args()
    postproc_mlc_serve_args(args)

    _test_bad_json_schema(args)
