"""Utils for benchmark scripts"""


def add_sampling_flags(parser):
    parser.add_argument(
        "--apply-penalties",
        action="store_true",
        help="Apply presence/repetiton/frequency penalties.",
    )
    parser.add_argument(
        "--apply-logit-bias",
        action="store_true",
        help="Apply logit bias.",
    )
    parser.add_argument(
        "--apply-top-p-top-k",
        action="store_true",
        help="Apply top-p and top-k.",
    )
    parser.add_argument(
        "--apply-all-sampling-params",
        action="store_true",
        help="Apply all penalties, logit bias, top-p and top-k.",
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        default=False,
        help="Switch on logprobs output"
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=5,
        help="Number of top logprobs to output, limited by 5. Works only with logprobs true."
    )


def postproc_sampling_args(args):
    args.sampling_setting = {
        "ignore_eos": True,
        "logit_bias": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "logprobs": False,
        "top_logprobs": 5,
    }

    if args.apply_all_sampling_params:
        args.apply_penalties = True
        args.apply_logit_bias = True
        args.apply_top_p_top_k = True

    if args.apply_penalties:
        args.sampling_setting["presence_penalty"] = 0.7
        args.sampling_setting["frequency_penalty"] = 0.7
        args.sampling_setting["repetition_penalty"] = 0.7

    if args.apply_logit_bias:
        args.sampling_setting["logit_bias"] = {1: -1, 3: 1, 2: 2}

    if args.apply_top_p_top_k:
        args.sampling_setting["top_k"] = 2
        args.sampling_setting["top_p"] = 0.7

    if args.logprobs:
        args.sampling_setting["logprobs"] = True
        args.sampling_setting["top_logprobs"] = args.top_logprobs
