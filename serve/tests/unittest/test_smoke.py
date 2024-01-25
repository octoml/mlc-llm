from mlc_serve.engine import (
    Request,
    DebugOptions,
    SamplingParams,
    StoppingCriteria,
    get_engine_config,
    ChatMessage
)
from pydantic import BaseModel
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule
from mlc_serve.utils import get_default_mlc_serve_argparser, postproc_mlc_serve_args
from typing import List
import json

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

class France(BaseModel):
    capital: str

class Snow(BaseModel):
    color: str

class SnowList(BaseModel):
    snow: List[Snow]

def test_smoke(
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
        if i == 1:
            print(f"{i}th text output: {out_text}")
        else:
            print(f"{i}th JSON output: {json.loads(out_text)}")


if __name__ == "__main__":
    parser = get_default_mlc_serve_argparser("test engine with samplers")
    args = parser.parse_args()
    postproc_mlc_serve_args(args)

    test_smoke(args.model_artifact_path, use_staging_engine=True)
