import torch
import tvm
from mlc_llm import utils
from tvm import relax
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def get_tvm_model(artifact_path, model, device, device_name, dtype):
    const_params = utils.load_params(artifact_path, device)
    ex = tvm.runtime.load_module(f"{artifact_path}/{model}_{device_name}_{dtype}.so")
    vm = relax.VirtualMachine(ex, device)

    class Model:
        def __init__(self) -> None:
            self.tot_seq_len = 0
            self.kv_cache = vm["create_kv_cache"]()

        def forward(self, inputs: tvm.nd.array) -> tvm.nd.array:
            if inputs.device != device_name:
                inputs = tvm.nd.array(inputs.numpy(), device)

            self.tot_seq_len += inputs.shape[1]
            seq_len_shape = tvm.runtime.ShapeTuple([self.tot_seq_len])
            if inputs.shape[1] > 1:
                logits, kv_cache = vm["encoding"](
                    inputs, seq_len_shape, self.kv_cache, const_params
                )
            else:
                logits, kv_cache = vm["decoding"](
                    inputs, seq_len_shape, self.kv_cache, const_params
                )
            self.kv_cache = kv_cache
            return logits

    model = Model()
    return model.forward


def get_pytorch_model(model_path, torch_device, dtype, use_cache=True):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    if dtype == "float16":
        model = model.to(torch.float16)
    model = model.to(torch_device)

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(inputs, use_cache=use_cache).logits

    return forward


def get_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)
