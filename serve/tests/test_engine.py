import argparse
import json
import random

from mlc_serve.engine import (
    Request,
    ChatMessage,
    DebugOptions,
    SamplingParams,
    StoppingCriteria,
)
from mlc_serve.utils import (
    get_default_mlc_serve_argparser,
    postproc_mlc_serve_args,
    create_mlc_engine,
)


def _test(args: argparse.Namespace):
    engine = create_mlc_engine(args)

    sampling_params_greedy = SamplingParams(
        temperature=0.0,
        vocab_size=engine.model_artifact_config.vocab_size,
    )
    sampling_params_random = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        vocab_size=engine.model_artifact_config.vocab_size,
    )

    num_sequences = args.num_sequences_to_sample

    if num_sequences > 1:
        sampling_params_choices = [sampling_params_random]
    elif args.use_random_sampling:
        # This tests different sampling types in the same batch
        sampling_params_choices = [sampling_params_random, sampling_params_greedy]
    else:
        sampling_params_choices = [sampling_params_greedy]

    if args.long_prompt:
        with open("serve/tests/data/long_prompts.json", "r") as f:
            prompts = json.load(f)["prompts"]
    else:
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "The president of the United States is a powerful man. But he can also be",
            "The future of AI is full of promise. But we need to carefully",
        ]

    for i, prompt in enumerate(prompts):
        sampling_param = random.choice(sampling_params_choices)
        engine.add(
            [
                Request(
                    request_id=str(i),
                    messages=[ChatMessage(role="user", content=prompt)],
                    sampling_params=sampling_param,
                    stopping_criteria=StoppingCriteria(
                        max_tokens=args.max_output_len, stop_sequences=None
                    ),
                    debug_options=DebugOptions(prompt=prompt),
                    num_sequences=num_sequences,
                )
            ]
        )

    generated = [["" for _ in range(num_sequences)] for _ in range(len(prompts))]

    any_finished = set()

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            if any(seq.is_finished for seq in res.sequences):
                any_finished.add(res.request_id)

            if res.request_id not in any_finished:
                # If all sequences are still running, we should always get num_sequences samples back.
                assert len(res.sequences) == num_sequences, res

            for i, seq in enumerate(res.sequences):
                if seq.delta:
                    generated[int(res.request_id)][i] += seq.delta

    if args.long_prompt:
        for g in generated:
            for i, seq in enumerate(g):
                print(f"Generated {i}-th sample = '{seq}'")
                print("")
            print("")
    else:
        for p, g in zip(prompts, generated):
            print(f"Prompt = '{p}'")
            for i, seq in enumerate(g):
                print(f"Generated {i}-th sample = '{seq}'")
            print("")

    if args.use_staging_engine:
        engine.stop()


if __name__ == "__main__":
    parser = get_default_mlc_serve_argparser("test engine")
    parser.add_argument("--long-prompt", action="store_true")
    parser.add_argument("--use-random-sampling", action="store_true")
    parser.add_argument("--max-output-len", type=int, default=20)
    args = parser.parse_args()
    postproc_mlc_serve_args(args)

    if args.long_prompt:
        args.max_num_batched_tokens = 50000

    if args.num_sequences_to_sample > 1:
        args.use_random_sampling = True

    _test(args)
