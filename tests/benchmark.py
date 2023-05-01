import torch
from mlc_llm import utils
import argparse, os, time
from transformers import AutoTokenizer, AutoModelForCausalLM
import tvm
from tvm import relax
from mlc_llm.conversation import SeparatorStyle, conv_templates
from utils import get_tokenizer, get_pytorch_model, get_tvm_model, sample_top_p

torch_device = None


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--model", type=str, default="vicuna-v1-7b")
    args.add_argument("--max-gen-len", type=int, default=2048)
    args.add_argument("--run-torch-model", action="store_true", default=False)
    args.add_argument(
        "--dtype", type=str, choices=["float32", "float16", "int4"], default="float16"
    )
    parsed = args.parse_args()
    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(
        parsed.artifact_path, parsed.model, parsed.dtype
    )

    if parsed.device_name == "auto":
        if tvm.cuda().exist:
            parsed.device_name = "cuda"
        elif tvm.metal().exist:
            parsed.device_name = "metal"
        else:
            raise ValueError("Cannot auto deduce device-name, please set it")
    return parsed


class ModelWrapper:
    def __init__(self, tokenizer, max_gen_len):
        self.tokenizer = tokenizer
        self.max_gen_len = max_gen_len

    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
        add_bos=True,
    ):
        assert 0, "Need to implement"


class TvmModelWrapper(ModelWrapper):
    def __init__(
        self, tokenizer, max_gen_len, artifact_path, model, device_name, dtype
    ):
        super().__init__(tokenizer, max_gen_len)
        self.model = get_tvm_model(artifact_path, model, device_name, dtype)

    def generate(
        self,
        prompt_tokens,
        prompt_len,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
        add_bos=True,
    ):
        total_len = self.max_gen_len + len(prompt_tokens)
        # TODO: Replace Torch ops to TVM
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(torch_device)
        )
        tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens).to(torch_device)

        start_pos = len(prompt_tokens)
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                logits = self.model(tokens[:, :cur_pos])
            else:
                logits = self.model(tokens[:, cur_pos - 1 : cur_pos])

            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(
                    (logits / temperature).to(torch.float32), dim=-1
                ).to(torch_device)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1).to(torch_device)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token
            # the following code assumes bsz == 1
            if next_token[0] == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            i = cur_pos - start_pos
            if i % stream_interval == 0 or i == self.max_gen_len - 1 or stopped:
                # TODO: Parallelize decoding
                output = tokens[0, : cur_pos + 1]
                output = tokenizer.decode(output, skip_special_tokens=True)
                pos = output.rfind(stop_str, prompt_len)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output
            if stopped:
                break


class TorchModelWrapper(ModelWrapper):
    def __init__(self, tokenizer, max_gen_len, model_path, torch_device, dtype):
        super().__init__(tokenizer, max_gen_len)
        self.model = get_pytorch_model(model_path, torch_device, dtype)

    def generate(
        self,
        prompt_tokens,
        prompt_len,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
        add_bos=True,
    ):
        total_len = self.max_gen_len + len(prompt_tokens)
        tokens = (
            torch.full((1, total_len), self.tokenizer.pad_token_id)
            .to(torch.int32)
            .to(torch_device)
        )
        tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens).to(torch_device)
        start_pos = len(prompt_tokens)
        for cur_pos in range(start_pos, total_len):
            logits = self.model(tokens[:, :cur_pos])
            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(
                    (logits / temperature).to(torch.float32), dim=-1
                ).to(torch_device)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1).to(torch_device)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token
            # the following code assumes bsz == 1
            if next_token[0] == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            i = cur_pos - start_pos
            if i % stream_interval == 0 or i == self.max_gen_len - 1 or stopped:
                # TODO: Parallelize decoding
                output = tokens[0, : cur_pos + 1]
                output = tokenizer.decode(output, skip_special_tokens=True)
                pos = output.rfind(stop_str, prompt_len)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output
            if stopped:
                break


def chat(model_wrapper, user_inps):
    conv = conv_templates["vicuna_v1.1"].copy()
    add_bos = True

    for iid, inp in enumerate(user_inps):
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        # TODO: Torch does not work with the following function for multi-round conv.
        #       Check this w/ authors.
        # prompt = conv.get_prompt_unprocessed()
        prompt = conv.get_prompt()

        print(f"=== Input {iid+1} ===")
        print(f"{conv.roles[0]}: {inp}", flush=True)
        print(f"{conv.roles[1]}: ", end="", flush=True)

        t0 = time.time()
        prompt_tokens = model_wrapper.tokenizer.encode(prompt)
        if not add_bos:
            prompt_tokens = prompt_tokens[1:]

        prompt_len = len(prompt)
        pre = 0
        for outputs in model_wrapper.generate(
            prompt_tokens,
            prompt_len,
            temperature=0,  # Use greedy to make it deterministic for benchmarking
            stop_str=conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
            add_bos=add_bos,
        ):
            outputs = outputs[prompt_len + 1 :].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                print(" ".join(outputs[pre : now - 1]), end=" ", flush=True)
                pre = now - 1
        t1 = time.time()
        print(" ".join(outputs[pre:]), flush=True)
        print(f"   - # input token: {len(prompt_tokens)}")
        print(f"   - # output token: {len(outputs)}")
        print(f"   - process time: {(t1-t0):.3f} s")

        conv.messages[-1][-1] = " ".join(outputs)
        add_bos = False


if __name__ == "__main__":
    ARGS = _parse_args()
    tokenizer = get_tokenizer(ARGS.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if not ARGS.run_torch_model:
        torch_device = torch.device("cpu")
        model = TvmModelWrapper(
            tokenizer,
            ARGS.max_gen_len,
            ARGS.artifact_path,
            ARGS.model,
            ARGS.device_name,
            ARGS.dtype,
        )
    else:
        torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = TorchModelWrapper(
            tokenizer, ARGS.max_gen_len, ARGS.model_path, torch_device, ARGS.dtype
        )

    inputs = [
        "Hi",
        "Repeat this sentence: Sure! Amazon is a multinational technology company that operates in a variety of industries, including e-commerce, cloud computing, digital streaming, and more. The company's headquarters, also known as Amazon Headquarters or Amazon HQ, is located in Seattle, Washington. It is the main base of operations for Amazon.com, Inc., the parent company of Amazon's various subsidiaries and businesses. The headquarters is home to many of the company's executives, as well as its research and development teams, customer service teams, and other support staff.",
    ]
    chat(model, inputs)
