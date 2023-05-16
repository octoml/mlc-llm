import torch
import argparse, os, time
import tvm
from mlc_llm.conversation import SeparatorStyle, conv_templates, compute_skip_echo_len
from mlc_llm import utils
from utils import get_tokenizer, get_pytorch_model, get_tvm_model, sample_top_p
import numpy as np
from transformers import AutoModelForCausalLM


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# NOTE: "all" is currently not supported due to GPU OOM issue in creating multiple model wrappers.
BENCHMARK_MODES = ["tvm", "torch-eager", "torch-inductor"]


def _parse_args():
    args = argparse.ArgumentParser()
    utils.argparse_add_common(args)
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--max-gen-len", type=int, default=2048)
    args.add_argument(
        "--benchmark-mode",
        type=str,
        choices=BENCHMARK_MODES,
        default=BENCHMARK_MODES[0],
    )
    args.add_argument("--num-warm-up", type=int, default=5)
    args.add_argument("--num-measurements", type=int, default=10)
    args.add_argument("--num-input-tokens", type=int, default=32)
    args.add_argument("--num-output-tokens", type=int, default=32)

    parsed = args.parse_args()

    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path,
        parsed.model,
        parsed.dtype,
    )
    utils.argparse_postproc_common(parsed)
    return parsed


class ModelWrapper:
    def __init__(self, tokenizer, max_gen_len, conv_template):
        self.name = None
        self.tokenizer = tokenizer
        self.max_gen_len = max_gen_len
        self.conv_template = conv_template

    def sync(self):
        assert 0, "Not intended to call directly"

    def benchmark_core(self, num_input_tokens, num_output_tokens):
        assert 0, "Not intended to call directly"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
    ):
        assert 0, "Not intended to call directly"


# TODO: Reset kv cache
class TvmModelWrapper(ModelWrapper):
    def __init__(
        self,
        tokenizer,
        max_gen_len,
        conv_template,
        artifact_path,
        model_name,
        dtype,
        tvm_device,
        torch_device=("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(tokenizer, max_gen_len, conv_template)

        self.name = "tvm_model_wrapper"
        self.artifact_path = artifact_path
        self.model_name = model_name
        self.dtype = dtype
        tvm_ex = tvm.runtime.load_module(
            f"{artifact_path}/{model_name}_{tvm_device}_{dtype}.so"
        )
        self.tvm_device = tvm.device(tvm_device)
        self.torch_device = torch.device(torch_device)

        self.const_params = utils.load_params(artifact_path, self.tvm_device)
        self.vm = tvm.relax.VirtualMachine(tvm_ex, self.tvm_device)
        self.prep_model()

    def sync(self):
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize()

        if str(self.tvm_device) != "cpu":
            self.tvm_device.sync()

    def prep_model(self):
        self.model = get_tvm_model(self.const_params, self.vm)

    def benchmark_core(self, num_input_tokens, num_output_tokens):
        total_len = num_input_tokens + num_output_tokens
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(self.torch_device)
        )

        start_pos = num_input_tokens
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                # TODO: switch to the below when Eric's PR is merged.
                #    tok = tvm.nd.from_dlpack(tokens[:, :cur_pos])
                tok = tvm.nd.array(tokens[:, :cur_pos].numpy(), self.tvm_device)
                logits = self.model(tok)
            else:
                # TODO: switch to the below when Eric's PR is merged.
                #    tok = tvm.nd.from_dlpack(tokens[:, cur_pos - 1 : cur_pos])
                tok = tvm.nd.array(
                    tokens[:, cur_pos - 1 : cur_pos].numpy(), self.tvm_device
                )
                logits = self.model(tok)

            # NOTE:
            # There are three methods to work with torch ops.
            # Method 1: tvm GPU -> torch GPU
            #    logits = torch.from_dlpack(logits)
            # Method 2: tvm GPU-> tvm CPU -> torch CPU
            #    logits = tvm.nd.array(logits.numpy(), tvm.cpu())
            #    logits = torch.from_dlpack(logits)
            # Method 3: tvm GPU -> numpy CPU -> torch CPU
            #    logits = torch.from_numpy(logits.numpy())
            #
            # For logits of [1,1,32k] = 128KB (our case), conversion on GPU (Method1) takes ~1sec,
            # which could be non-neglibile for short response.
            # However, conversion on CPU (Method 2&3) is very cheap even with the data copy from GPU to CPU.
            # Rather than addressing this issue, we choose Method3 by default for now.

            if str(self.torch_device) == "cpu":
                logits = torch.from_numpy(logits.numpy())
            else:
                logits = torch.from_dlpack(logits)

            logits = logits[:, -1, :]
            # Use greedy
            next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token

    def generate(
        self,
        prompt_tokens,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
    ):
        prompt_len = len(prompt_tokens)
        total_len = self.max_gen_len + prompt_len
        # TODO: Replace Torch ops to TVM
        # Issue 1: `tokens` requires in-place update.
        # Issue 2: p-sampling use random generator which might have side effect.
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(self.torch_device)
        )

        tokens[0, :prompt_len] = (
            torch.tensor(prompt_tokens).to(torch.int32).to(self.torch_device)
        )

        start_pos = prompt_len
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                # TODO: switch to the below when Eric's PR is merged.
                #    tok = tvm.nd.from_dlpack(tokens[:, :cur_pos])
                tok = tvm.nd.array(tokens[:, :cur_pos].numpy(), self.tvm_device)
                logits = self.model(tok)
            else:
                # TODO: switch to the below when Eric's PR is merged.
                #    tok = tvm.nd.from_dlpack(tokens[:, cur_pos - 1 : cur_pos])
                tok = tvm.nd.array(
                    tokens[:, cur_pos - 1 : cur_pos].numpy(), self.tvm_device
                )
                logits = self.model(tok)

            # NOTE:
            # There are three methods to work with torch ops.
            # Method 1: tvm GPU -> torch GPU
            #    logits = torch.from_dlpack(logits)
            # Method 2: tvm GPU-> tvm CPU -> torch CPU
            #    logits = tvm.nd.array(logits.numpy(), tvm.cpu())
            #    logits = torch.from_dlpack(logits)
            # Method 3: tvm GPU -> numpy CPU -> torch CPU
            #    logits = torch.from_numpy(logits.numpy())
            #
            # For logits of [1,1,32k] = 128KB (our case), conversion on GPU (Method1) takes ~1sec,
            # which could be non-neglibile for short response.
            # However, conversion on CPU (Method 2&3) is very cheap even with the data copy from GPU to CPU.
            # Rather than addressing this issue, we choose Method3 by default for now.

            if str(self.torch_device) == "cpu":
                logits = torch.from_numpy(logits.numpy())
            else:
                logits = torch.from_dlpack(logits)

            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax((logits / temperature).to(torch.float32), dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            tokens[:, cur_pos] = next_token

            stopped = next_token[0] == self.tokenizer.eos_token_id

            i = cur_pos - start_pos
            if i % stream_interval == 0 or i == self.max_gen_len - 1 or stopped:
                output = tokens[0, : cur_pos + 1]
                output = self.tokenizer.decode(output, skip_special_tokens=True)
                pos = output.rfind(stop_str, prompt_len)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output

            if stopped:
                break


class TorchModelWrapper(ModelWrapper):
    def __init__(
        self,
        tokenizer,
        max_gen_len,
        conv_template,
        model_path,
        dtype,
        torch_mode,
        torch_device=("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(tokenizer, max_gen_len, conv_template)
        self.name = f"torch_{torch_mode}_model_wrapper"
        self.torch_device = torch.device(torch_device)
        self.model_path = model_path
        self.dtype = dtype
        self.torch_mode = torch_mode

        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        if dtype == "float16":
            model = model.to(torch.float16)
        self._model = model.to(torch_device)
        self.prep_model()

    def prep_model(self):
        model = get_pytorch_model(self._model)
        self.model = torch.compile(model, backend=self.torch_mode, dynamic=True)

    def sync(self):
        if self.torch_device.type == "cuda":
            torch.cuda.synchronize()

    def benchmark_core(self, num_input_tokens, num_output_tokens):
        total_len = num_input_tokens + num_output_tokens
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(self.torch_device)
        )

        for cur_pos in range(num_input_tokens, total_len):
            logits = self.model(tokens[:, :cur_pos])
            logits = logits[:, -1, :]
            # Use greedy
            next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token

    def generate(
        self,
        prompt_tokens,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
    ):
        prompt_len = len(prompt_tokens)
        total_len = self.max_gen_len + prompt_len
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(self.torch_device)
        )
        tokens[0, :prompt_len] = torch.tensor(prompt_tokens).to(torch.int32)
        start_pos = prompt_len

        for cur_pos in range(start_pos, total_len):
            logits = self.model(tokens[:, :cur_pos])
            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax((logits / temperature).to(torch.float32), dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token

            stopped = next_token[0] == self.tokenizer.eos_token_id

            i = cur_pos - start_pos
            if i % stream_interval == 0 or i == self.max_gen_len - 1 or stopped:
                output = self.tokenizer.decode(
                    tokens[0, : cur_pos + 1], skip_special_tokens=True
                )
                pos = output.rfind(stop_str, prompt_len)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output
            if stopped:
                break


# Benchmark single-round conv
def benchmark_e2e_chat(model_wrapper, prompt, enable_print=False):
    conv = conv_templates[model_wrapper.conv_template].copy()
    keep_first_token = True

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    model_wrapper.sync()
    t0 = time.time()
    prompt_tokens = model_wrapper.tokenizer.encode(prompt)
    if not keep_first_token:
        prompt_tokens = prompt_tokens[1:]

    pre = 0
    skip_echo_len = compute_skip_echo_len(model_wrapper.conv_template, conv, prompt)
    outputs = None
    for outputs in model_wrapper.generate(
        prompt_tokens,
        temperature=0,  # Use greedy to make it deterministic for benchmarking
        stop_str=conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        # stream_interval=2048,  # To output at once
    ):
        outputs = outputs[skip_echo_len:].strip()
        now = len(outputs)
        if now - 1 > pre:
            if enable_print:
                print(
                    Colors.OKBLUE + " ".join(outputs[pre : now - 1]) + Colors.ENDC,
                    end=" ",
                    flush=True,
                )
            pre = now - 1
    if enable_print:
        print(
            Colors.OKBLUE + " ".join(outputs[pre:]) + Colors.ENDC,
            flush=True,
        )

    model_wrapper.sync()
    t1 = time.time()
    assert now == len(outputs)
    elapsed = t1 - t0
    num_input_prompt_tokens, num_decoded_output_tokens = len(prompt_tokens), len(
        outputs
    )

    return num_input_prompt_tokens, num_decoded_output_tokens, elapsed


def benchmark_core_chat(model_wrapper, num_input_tokens, num_output_tokens):
    # Clear kv cache and prepare the model
    model_wrapper.prep_model()
    model_wrapper.sync()
    t0 = time.time()
    model_wrapper.benchmark_core(num_input_tokens, num_output_tokens)
    model_wrapper.sync()
    t1 = time.time()
    return t1 - t0


def benchmark(
    model_wrapper,
    num_input_tokens,
    num_output_tokens,
    num_warm_up,
    num_measurement,
    percentiles=[50, 90, 99],
):
    # Warm-up
    for _ in range(num_warm_up):
        benchmark_core_chat(model_wrapper, num_input_tokens, num_output_tokens)

    # Actual measurement
    elapsed = []
    for _ in range(num_measurement):
        elapsed.append(
            benchmark_core_chat(model_wrapper, num_input_tokens, num_output_tokens)
        )
    elapsed = [np.percentile(elapsed, p) for p in percentiles]
    tok_per_sec = [num_output_tokens / e for e in elapsed]
    return elapsed, tok_per_sec


def get_model_wrapper(mode, tokenizer, ARGS):
    if mode == "tvm":
        return TvmModelWrapper(
            tokenizer,
            ARGS.max_gen_len,
            ARGS.conv_template,
            ARGS.artifact_path,
            ARGS.model,
            ARGS.dtype,
            tvm_device=ARGS.device_name,
            torch_device="cpu",
        )
    elif mode.startswith("torch-"):
        return TorchModelWrapper(
            tokenizer,
            ARGS.max_gen_len,
            ARGS.conv_template,
            ARGS.model_path,
            ARGS.dtype,
            torch_device="cuda",
            torch_mode=mode[6:],
        )


if __name__ == "__main__":
    ARGS = _parse_args()
    # Torch setup
    torch.set_float32_matmul_precision("high")

    # Tokenizer setup
    tokenizer = get_tokenizer(ARGS.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Example of e2e chat. Disable if you want to try
    # prompt = """Repeat the following paragraph exactly: Carlos Alcaraz is a professional tennis player from Spain. He was born on April 12, 1997, and has been playing tennis since he was a child. Alcaraz is known for his aggressive playing style and his ability to come back from difficult situations. He has won several junior tournaments and has also played on the ATP Tour. Alcaraz He has a career high ranking in men’s singles by the Association of Tennis Professionals of world No. 1 achieved on 12 September 2022. He is currently ranked world No. 2 by the ATP. He is known for his strong serve and powerful groundstrokes. He is also known for his positive attitude and his determination to succeed on the court."""
    # benchmark_e2e_chat(model_wrapper, prompt, True)

    model_wrapper = get_model_wrapper(ARGS.benchmark_mode, tokenizer, ARGS)
    percentiles = [50, 90, 99]

    # TODO: Extend to the list of repr input pairs
    pairs = [(ARGS.num_input_tokens, ARGS.num_output_tokens)]
    for num_input_tokens, num_output_tokens in pairs:
        elapsed, tok_per_sec = benchmark(
            model_wrapper,
            num_input_tokens,
            num_output_tokens,
            ARGS.num_warm_up,
            ARGS.num_measurements,
            percentiles=percentiles,
        )

        print("|{:^15}|{:^12}|{:^12}|".format("mode", "seqlen", "genlen"), end="")
        for p in percentiles:
            print("{:^12}|{:^12}|".format(f"p{p}: sec", f"p{p}: tok/s"), end="")
        print("")
        print(
            "|{:^15}|{:^12}|{:^12}|".format(
                ARGS.benchmark_mode,
                num_input_tokens,
                num_output_tokens,
            ),
            end="",
        )
        for i in range(len(percentiles)):
            print("{:^12.3f}|{:^12.3f}|".format(elapsed[i], tok_per_sec[i]), end="")
        print("")
    del model_wrapper
