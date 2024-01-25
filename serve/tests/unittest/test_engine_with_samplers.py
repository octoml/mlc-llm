import json
from mlc_serve.engine import (
    Request,
    ChatMessage,
    DebugOptions,
    SamplingParams,
    StoppingCriteria,
    FinishReason,
    get_engine_config,
)
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.base import get_model_artifact_config
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule
from mlc_serve.utils import get_default_mlc_serve_argparser, postproc_mlc_serve_args
from pydantic import BaseModel
from typing import List


def create_engine(
    model_artifact_path,
    use_staging_engine,
    max_num_batched_tokens,
):
    engine_config = get_engine_config(
        {
            "use_staging_engine": use_staging_engine,
            "max_num_batched_tokens": max_num_batched_tokens,
            # Use defaults for "min_decode_steps", "max_decode_steps"
        }
    )

    if use_staging_engine:
        engine = StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(model_artifact_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": model_artifact_path,
                "engine_config": engine_config,
            },
        )
        engine.start()
    else:
        engine = SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path=model_artifact_path,
                engine_config=engine_config,
            )
        )
    return engine


def create_request(
    idx, prompt, temp, freq_pen, pre_pen, max_tokens, stop, ignore_eos, logit_bias=None, json_schema=None
):
    return Request(
        request_id=str(idx),
        messages=[ChatMessage(role="user", content=prompt)],
        sampling_params=SamplingParams(
            temperature=temp,
            frequency_penalty=freq_pen,
            presence_penalty=pre_pen,
            logit_bias=logit_bias,
            json_schema=json_schema
        ),
        stopping_criteria=StoppingCriteria(max_tokens=max_tokens, stop_sequences=stop),
        debug_options=DebugOptions(ignore_eos=ignore_eos),
    )


def _test_max_tokens(
    model_artifact_path,
    use_staging_engine,
    max_num_batched_tokens=2048,
    num_requests=5,
    ignore_eos=False,
):
    prompt = "Write a merge sort program in Python."
    engine = create_engine(
        model_artifact_path,
        use_staging_engine,
        max_num_batched_tokens,
    )

    requests = [
        create_request(
            idx=str(n - 1),
            prompt=prompt,
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=n,
            stop=None,
            ignore_eos=ignore_eos,
        )
        for n in range(1, num_requests)
    ]
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]

            if seq.is_finished:
                assert (
                    seq.num_generated_tokens
                    == requests[int(res.request_id)].stopping_criteria.max_tokens
                )
                assert seq.finish_reason == FinishReason.Length
            else:
                generated[int(res.request_id)] += seq.delta

    if use_staging_engine:
        engine.stop()


def _test_max_context_length(
    model_artifact_path,
    use_staging_engine,
    max_num_sequences=4,
    num_requests=5,
    ignore_eos=False,
):
    model_artifact_config = get_model_artifact_config(model_artifact_path)
    max_context_length = model_artifact_config.max_context_length

    engine = create_engine(
        model_artifact_path,
        use_staging_engine,
        max_num_batched_tokens=max_context_length * max_num_sequences,
    )
    prompt = "hi " * (max_context_length - 15)

    requests = [
        create_request(
            idx=str(n),
            prompt=prompt,
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=None,
            stop=None,
            ignore_eos=ignore_eos,
        )
        for n in range(num_requests)
    ]
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]

            if seq.is_finished:
                assert seq.finish_reason == FinishReason.Length, seq.finish_reason
            else:
                generated[int(res.request_id)] += seq.delta

    if use_staging_engine:
        engine.stop()


def _test_ignore_eos(
    model_artifact_path,
    use_staging_engine,
    max_num_batched_tokens=2048,
    num_requests=5,
):
    prompt = "hi"
    engine = create_engine(
        model_artifact_path,
        use_staging_engine,
        max_num_batched_tokens,
    )
    s = 113
    requests = [
        create_request(
            idx=str(n - s),
            prompt=prompt,
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=n,
            stop=None,
            ignore_eos=True,
        )
        for n in range(s, s + num_requests)
    ]
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]

            if seq.is_finished:
                assert (
                    seq.num_generated_tokens
                    == requests[int(res.request_id)].stopping_criteria.max_tokens
                )
                assert seq.finish_reason == FinishReason.Length
            else:
                generated[int(res.request_id)] += seq.delta

    if use_staging_engine:
        engine.stop()


def _test_stop(
    model_artifact_path,
    use_staging_engine,
    max_num_batched_tokens=2048,
    num_requests=5,
):
    prompt = "Write a merge sort program in Python."
    engine = create_engine(
        model_artifact_path,
        use_staging_engine,
        max_num_batched_tokens,
    )
    requests = []
    for n, stop in enumerate(["\n", ["\n"], "\n\n", "!", ["n", "!"]]):
        requests.append(
            create_request(
                idx=str(n),
                prompt=prompt,
                temp=0,
                freq_pen=0,
                pre_pen=0,
                max_tokens=300,
                stop=stop,
                ignore_eos=False,
            )
        )
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]
            req_id = int(res.request_id)

            if seq.delta:
                generated[int(res.request_id)] += seq.delta

            if seq.is_finished:
                assert (
                    seq.finish_reason == FinishReason.Stop
                ), f"{seq.finish_reason.name}"
                gen_txt = generated[req_id]

                # stop token should appear only once in the gen text.
                found = sum(
                    [
                        gen_txt.count(str_stop)
                        for str_stop in requests[
                            req_id
                        ].stopping_criteria.stop_sequences
                    ]
                )
                assert found == 1, f"{gen_txt!r}, matches: {found}"

    if use_staging_engine:
        engine.stop()


def _test_penalty(
    model_artifact_path,
    use_staging_engine,
    max_num_batched_tokens=2048,
    num_requests=5,
    ignore_eos=False,
):
    prompt = "Write a merge sort program in Python."
    engine = create_engine(
        model_artifact_path,
        use_staging_engine,
        max_num_batched_tokens,
    )

    random_requests = [
        create_request(
            idx=str(n - 1),
            prompt=prompt,
            temp=0.5,
            freq_pen=0.5,
            pre_pen=-0.5,
            max_tokens=n,
            stop=None,
            ignore_eos=ignore_eos,
            logit_bias={123: -100, 456: 100},
        )
        for n in range(1, num_requests)
    ]
    greedy_requests = [
        create_request(
            idx=str(n - 1),
            prompt=prompt,
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=n,
            stop=None,
            ignore_eos=ignore_eos,
        )
        for n in range(num_requests, num_requests << 1)
    ]
    requests = random_requests + greedy_requests
    engine.add(requests)

    generated = ["" for _ in range(num_requests << 1)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]

            if seq.is_finished:
                assert (
                    seq.num_generated_tokens
                    == requests[int(res.request_id)].stopping_criteria.max_tokens
                )
                assert seq.finish_reason == FinishReason.Length
            else:
                generated[int(res.request_id)] += seq.delta

    if use_staging_engine:
        engine.stop()

# These three models are used in _test_json_mode
class France(BaseModel):
    capital: str


class Snow(BaseModel):
    color: str


class SnowList(BaseModel):
    snow: List[Snow]


def _test_json_mode(
    model_artifact_path,
    use_staging_engine,
    max_num_batched_tokens=2048,
    ignore_eos=False,
):
    engine = create_engine(
        model_artifact_path,
        use_staging_engine,
        max_num_batched_tokens,
    )

    requests = [

        # test France schema
        create_request(
            idx=str(0),
            prompt="what is the capital of France?",
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=30,
            stop=None,
            ignore_eos=ignore_eos,
            json_schema=France.model_json_schema()
        ),

        # test with no JSON schema
        create_request(
            idx=str(1),
            prompt="Hello",
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=30,
            stop=None,
            ignore_eos=ignore_eos
        ),

        # test Snow schema
        create_request(
            idx=str(2),
            prompt="what is the color of the snow?",
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=30,
            stop=None,
            ignore_eos=ignore_eos,
            json_schema=Snow.model_json_schema()
        ),

        # test SnowList schema (nested structure)
        create_request(
            idx=str(3),
            prompt="Quick Facts About Snow | National Snow and Ice Data Center When light reflects off it, snow appears white. The many sides of a snowflake scatter light, diffusing the color spectrum in many directions. Snow can look dark when dust, or pollution, cover it. Fresh-water algae that loves snow can turn it into other colors like orange, blue, or watermelon pink. List the colors of snow.",
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=256,
            stop=None,
            ignore_eos=ignore_eos,
            json_schema=SnowList.model_json_schema()
        )
 
    ]
    num_requests= len(requests)
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]

            if not seq.is_finished:
                generated[int(res.request_id)] += seq.delta
    if use_staging_engine:
        engine.stop()

    for i, out_text in enumerate(generated):
        if i == 0:
            France.model_validate(json.loads(out_text))
        elif i == 1:
            assert isinstance(out_text, str)
        elif i == 2:
            Snow.model_validate(json.loads(out_text))
        else:
            SnowList.model_validate(json.loads(out_text))


if __name__ == "__main__":
    parser = get_default_mlc_serve_argparser("test engine with samplers")
    args = parser.parse_args()
    postproc_mlc_serve_args(args)

    _test_max_tokens(args.model_artifact_path, use_staging_engine=True)
    _test_max_tokens(args.model_artifact_path, use_staging_engine=False)
    _test_ignore_eos(args.model_artifact_path, use_staging_engine=True)
    _test_ignore_eos(args.model_artifact_path, use_staging_engine=False)
    _test_stop(args.model_artifact_path, use_staging_engine=False)
    _test_stop(args.model_artifact_path, use_staging_engine=True)
    # These tests are broken since we are now imposing no length limit
    # if max_tokens = None. The tests do not finish in a reasonable time.
    # _test_max_context_length(model_artifact_path, use_staging_engine=True)
    # _test_max_context_length(model_artifact_path, use_staging_engine=False)
    _test_penalty(args.model_artifact_path, use_staging_engine=True)
    _test_penalty(args.model_artifact_path, use_staging_engine=False)
    _test_json_mode(args.model_artifact_path, use_staging_engine=True)
    _test_json_mode(args.model_artifact_path, use_staging_engine=False)
