"""Utils for benchmark scripts"""
from pydantic import BaseModel

class Output(BaseModel):
    answer: str


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
        "--apply-json-mode",
        action="store_true",
        help="Apply json mode.",
    )
    parser.add_argument(
        "--apply-all-sampling-params",
        action="store_true",
        help="Apply all penalties, logit bias, top-p and top-k.",
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
        "json_schema": None
    }

    if args.apply_all_sampling_params:
        args.apply_penalties = True
        args.apply_logit_bias = True
        args.apply_top_p_top_k = True
        args.apply_json_mode = True

    if args.apply_penalties:
        args.sampling_setting["presence_penalty"] = 0.7
        args.sampling_setting["frequency_penalty"] = 0.7
        args.sampling_setting["repetition_penalty"] = 0.7

    if args.apply_logit_bias:
        args.sampling_setting["logit_bias"] = {1: -1, 3: 1, 2: 2}

    if args.apply_top_p_top_k:
        args.sampling_setting["top_k"] = 2
        args.sampling_setting["top_p"] = 0.7

    if args.apply_json_mode:
        args.sampling_setting["json_schema"] = Output.model_json_schema()
