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
    if int(idx)==1:
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
    else:
        return Request(
            request_id=str(idx),
            messages=None,
            sampling_params=SamplingParams(
                temperature=temp,
                frequency_penalty=freq_pen,
                presence_penalty=pre_pen,
                logit_bias=logit_bias,
                json_schema=json_schema
            ),
            stopping_criteria=StoppingCriteria(max_tokens=max_tokens, stop_sequences=stop),
            debug_options=DebugOptions(ignore_eos=ignore_eos, prompt=prompt),
        )

class France(BaseModel):
    capital: str
class Snow(BaseModel):
    color: str

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
        create_request(
            idx=str(3),
            prompt="what is the capital of France?",
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=30,
            stop=None,
            ignore_eos=ignore_eos,
            json_schema=France.model_json_schema()
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

            if seq.is_finished:
                print(f"finish reason: {seq.finish_reason}")
                # assert (
                #     seq.num_generated_tokens
                #     == requests[int(res.request_id)].stopping_criteria.max_tokens
                # )
                # assert seq.finish_reason == FinishReason.Length
            else:
                generated[int(res.request_id)] += seq.delta
    if use_staging_engine:
        engine.stop()

    for i, out_text in enumerate(generated):
        if i == 1:
            print(f"{i}th output: {out_text}")
        else:
            print(f"{i}th output: {json.loads(out_text)}")
    


if __name__ == "__main__":
    parser = get_default_mlc_serve_argparser("test engine with samplers")
    args = parser.parse_args()
    postproc_mlc_serve_args(args)

    test_smoke(args.model_artifact_path, use_staging_engine=True)
