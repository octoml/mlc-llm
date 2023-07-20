from cohere.models import base_medium as cohere_medium
from utils import get_tokenizer
from mlc_llm import utils
import torch
from transformers import GenerationConfig, GPT2Config, AutoTokenizer, GPT2LMHeadModel
import tvm
import argparse
import os

torch_device = "cuda"
tvm_device = tvm.device("cuda")


def _parse_args():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--weight-dist", type=str, default="cohere-medium/pytorch_model.bin"
    )
    args.add_argument("--model", type=str, default="cohere-medium")
    args.add_argument("--quant", type=str, default="q0f16")
    args.add_argument("--dist", type=str, default="dist")
    args.add_argument("--mod", choices=["tvm", "torch", "compare", "torch-gpt2-xl"])
    args.add_argument("--input-text", action="store_true")
    args.add_argument("--output-dir", type=str, default="test_output")
    parsed = args.parse_args()
    return parsed


def load_torch_model(weight_dist):
    model = cohere_medium(
        dev=torch_device,
        dtype=torch.float16,
    )

    state_dict = torch.load(weight_dist, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_torch_gpt2xl_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to("cuda").to(torch.float16)

    model.eval()
    return model


def load_tvm_model(model_name, quant, dist):
    dist = f"{dist}/{model_name}-{quant}"

    tvm_ex = tvm.runtime.load_module(f"{dist}/{model_name}-{quant}-cuda.so")
    const_params = utils.load_params(dist, tvm_device)
    return tvm_ex, const_params


def get_torch_output(torch_model, token_ids, prefill_len):
    outputs = []

    num_input_tokens = prefill_len
    total_len = len(token_ids[0])
    batch_size = len(token_ids)
    tokens = torch.tensor(token_ids).to(torch.int32).to(torch_device)
    attention_mask = torch.ones((batch_size, total_len)).to(torch_device)
    position_ids = (
        torch.arange(0, total_len)
        .unsqueeze(0)
        .broadcast_to(batch_size, total_len)
        .to(torch_device)
    )
    past_key_values = None
    for cur_pos in range(num_input_tokens, total_len + 1):
        if cur_pos == num_input_tokens:
            logits, past_key_values = torch_model.forward(
                input_ids=tokens[:, :cur_pos],
                past_key_values=past_key_values,
                attention_mask=attention_mask[:, :cur_pos],
                position_ids=position_ids[:, :cur_pos],
                return_dict=False,
            )
        else:
            logits, past_key_values = torch_model.forward(
                input_ids=tokens[:, cur_pos - 1 : cur_pos],
                past_key_values=past_key_values,
                attention_mask=attention_mask[:, cur_pos - 1 : cur_pos],
                position_ids=position_ids[:, cur_pos - 1 : cur_pos],
                use_cache=True,
                return_dict=False,
            )

        logits = logits.cpu()
        logits = logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)

        outputs.append((logits, next_token))

    return outputs


def get_tvm_output(vm, const_params, token_ids, prefill_len):
    outputs = []

    vm = tvm.relax.VirtualMachine(tvm_ex, tvm_device)
    kv_cache = vm["create_kv_cache"]()
    num_input_tokens = prefill_len
    total_len = len(token_ids[0])
    tokens = torch.tensor(token_ids).to(torch.int32).to(torch_device)

    start_pos = num_input_tokens
    for cur_pos in range(start_pos, total_len + 1):
        if cur_pos == start_pos:
            tok = tvm.nd.array(tokens[:, :cur_pos].cpu().numpy(), tvm_device)
            seq_len_shape = tvm.runtime.ShapeTuple([cur_pos])
            logits, kv_cache = vm["prefill"](tok, seq_len_shape, kv_cache, const_params)
        else:
            tok = tvm.nd.array(
                tokens[:, cur_pos - 1 : cur_pos].cpu().numpy(), tvm_device
            )
            seq_len_shape = tvm.runtime.ShapeTuple([cur_pos])
            logits, kv_cache = vm["decode"](tok, seq_len_shape, kv_cache, const_params)

        logits = torch.from_numpy(logits.numpy()).to(torch.float16)

        logits = logits.cpu().squeeze(1)

        next_token = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)
        outputs.append((logits, next_token))

    return outputs


if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_text:
        tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        prompts = [
            "Hi, let's use tvm. Have fun, folks.",
            # "Hi",
            # "OctoML is the best.",
            # "Let's go and do something fun. Try Try Try",
        ]
        token_ids = []
        for p in prompts:
            encoded_prompt = tokenizer.encode(
                p, max_length=8, padding="max_length", truncation=True
            )
            token_ids.append(encoded_prompt)
    else:
        token_ids = [
            [17250, 314, 716, 12963, 5474, 9338, 14514],
            # [17250, 428, 318, 474, 320, 1820, 1234],
        ]
    prefill_len = 4

    if args.mod == "torch":
        torch_model = load_torch_model(args.weight_dist)
        torch_output = get_torch_output(torch_model, token_ids, prefill_len=prefill_len)

        torch.save(torch_output, f"{args.output_dir}/torch_output.pkl")
    if args.mod == "torch-gpt2-xl":
        torch_model = load_torch_gpt2xl_model()
        torch_output = get_torch_output(torch_model, token_ids, prefill_len=prefill_len)

        torch.save(torch_output, f"{args.output_dir}/torch_output.pkl")
    elif args.mod == "tvm":
        tvm_ex, const_params = load_tvm_model(args.model, args.quant, args.dist)
        tvm_output = get_tvm_output(
            tvm_ex, const_params, token_ids, prefill_len=prefill_len
        )

        torch.save(tvm_output, f"{args.output_dir}/tvm_output.pkl")
    else:
        torch_output = torch.load(f"{args.output_dir}/torch_output.pkl")
        tvm_output = torch.load(f"{args.output_dir}/tvm_output.pkl")
        atol_val = 0.5
        for i, (torch_o, tvm_o) in enumerate(zip(torch_output, tvm_output)):
            torch_logits, torch_tokens = torch_o
            tvm_logits, tvm_tokens = tvm_o
            print(f"Compare {i}th logits")
            print("next token prediction same?:", torch_tokens == tvm_tokens)
            print(
                f"torch.allclose atol={atol_val}:",
                torch.allclose(torch_logits, tvm_logits, atol=atol_val),
            )
            print(
                f"max(abs(torch_logits - tvm_logits)):",
                torch.max(torch.abs(torch_logits - tvm_logits)).item(),
            )
            print("========")
