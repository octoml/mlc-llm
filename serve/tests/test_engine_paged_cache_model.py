import argparse
import json

from mlc_llm import utils
from mlc_serve.engine import (
    Request,
    ChatMessage,
    DebugOptions,
    SamplingParams,
    StoppingCriteria,
)
from mlc_serve.engine.local import LocalProcessInferenceEngine
from mlc_serve.model.paged_cache_model import PagedCacheModelModule


def test(args: argparse.Namespace):
    model_module = PagedCacheModelModule(
        args.model,
        args.artifact_path,
        args.quantization.name,
        args.num_shards,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_input_len=args.max_input_len,
    )

    engine = LocalProcessInferenceEngine(
        model_module,
        max_batched_tokens=args.max_num_batched_tokens,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
    )

    if args.long_prompt:
        with open("serve/tests/data/long_prompts.json", "r") as f:
            prompts = json.load(f)["prompts"]
            prompts = [prompts[0], prompts[2], prompts[3]]
    else:
        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

    for i, prompt in enumerate(prompts):
        engine.add(
            [
                Request(
                    request_id=str(i),
                    messages=[ChatMessage(role="user", content=prompt)],
                    sampling_params=sampling_params,
                    stopping_criteria=StoppingCriteria(max_tokens=args.max_output_len),
                    debug_options=DebugOptions(prompt=prompt),
                )
            ]
        )

    generated = ["" for _ in range(len(prompts))]

    while engine._has_request_to_process():
        results = engine.step()
        for res in results.outputs:
            seq = res.sequences[0]
            if not seq.is_finished:
                generated[int(res.request_id)] += seq.delta

    if args.long_prompt:
        for g in generated:
            print(f"Generated text = '{g}'")
    else:
        for p, g in zip(prompts, generated):
            print(f"Prompt = '{p}', generated text = '{g}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--local-id", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="dist")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=-1)
    parser.add_argument("--max-input-len", type=int, default=-1)
    parser.add_argument("--max-output-len", type=int, default=100)
    parser.add_argument("--long-prompt", action="store_true")
    args = parser.parse_args()

    args.model, args.quantization = args.local_id.rsplit("-", 1)
    utils.argparse_postproc_common(args)

    test(args)
