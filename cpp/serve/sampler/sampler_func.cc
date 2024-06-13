

// #include <chrono>
// #include <iostream>
// #include <map>
// #include <memory>
// #include <optional>
// #include <queue>
// #include <string>
// #include <unordered_set>
// #include <vector>

#include "sampler_func.h"
#include <tvm/runtime/registry.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

#pragma message("YYYYYYYYYYYYYYYYYYYYYYY")

TVM_REGISTER_OBJECT_TYPE(SamplerNode);
TVM_REGISTER_OBJECT_TYPE(GPUSamplerNode);

GPUSampler::GPUSampler(int vocab_size, DLDevice device, FunctionTable& ft) {
  ObjectPtr<GPUSamplerNode> n = make_object<GPUSamplerNode>();

  n->vocab_size = vocab_size;
  n->gpu_sampler = Sampler::CreateGPUSampler(1, vocab_size, &ft, device, {});
  n->generation_cfg.Create("{top_p: 0.7}");
  data_ = std::move(n);
  // 
}

std::vector<SampleResult> GPUSamplerNode::BatchDecode(NDArray probs)
{
    // std::vector<int> sample_indices(num_rsentries);
    // std::iota(sample_indices.begin(), sample_indices.end(), 0);
    // std::vector<SampleResult> sample_results = data_->IsInstance<GPUSamplerNode>()
    // .BatchSampleTokensWithProbBeforeTopP(
    //     probs_on_device, sample_indices, request_ids, generation_cfg, rngs);
  return std::vector<SampleResult>();
}

TVM_REGISTER_GLOBAL("mlc.serve.GPUSamplerBatchDecode").set_body_typed([](GPUSampler sampler, NDArray samples) {
  int num_rsentries = samples->shape[0]; // to check
  std::cout << "HERE!\n";
  std::vector<RandomGenerator*> rngs;
  Array<String> request_ids;
  Array<GenerationConfig> generation_cfg;
  rngs.reserve(num_rsentries);
  generation_cfg.reserve(num_rsentries);
  std::vector<int> sample_indices(num_rsentries);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);
  // RandomGenerator rng;
  for (size_t i = 0; i < num_rsentries; ++i) {
     rngs.push_back(&sampler->rng);
     generation_cfg.push_back(sampler->generation_cfg);
  }
  auto res = sampler->gpu_sampler->BatchSampleTokensWithProbBeforeTopP(
        samples, sample_indices, request_ids, generation_cfg, rngs);
});


TVM_REGISTER_GLOBAL("mlc.serve.GPUSampler").set_body_typed([](int vocab_size, DLDevice device) {
  // /home/sshtin/dev/ollm/deps/mlc-llm/cpp/serve/function_table.h
  std::string reload_lib_path = "/home/sshtin/dev/ollm/mlc-serve/dist/Mistral-7B-Instruct-v0.2-q0f16-vllm-1gpu/Mistral-7B-Instruct-v0.2-q0f16-vllm-1gpu-allreduce_AUTO.so";
  auto executable = tvm::runtime::Module::LoadFromFile(reload_lib_path);
  auto fload_exec = executable->GetFunction("vm_load_executable");

  FunctionTable ft;
  tvm::runtime::Module local_vm = fload_exec();
  // local_gpu_device = device;
  ft.gpu_multinomial_from_uniform_func_ = local_vm->GetFunction("multinomial_from_uniform", true);
  ft.gpu_argsort_probs_func_ = local_vm->GetFunction("argsort_probs", true);
  ft.gpu_sample_with_top_p_func_ = local_vm->GetFunction("sample_with_top_p", true);
  ft.gpu_sampler_take_probs_func_ = local_vm->GetFunction("sampler_take_probs", true);
  ft.gpu_verify_draft_tokens_func_ = local_vm->GetFunction("sampler_verify_draft_tokens", true);
  ft.gpu_renormalize_by_top_p_func_ = local_vm->GetFunction("renormalize_by_top_p", true);
  // std::cout << "device " << device << "\n";
  // std::cout << "ft.gpu_multinomial_from_uniform_func_ " << ft.gpu_multinomial_from_uniform_func_.defined() << "\n";
  // std::cout << " ft.gpu_argsort_probs_func_ " <<  ft.gpu_argsort_probs_func_.defined() << "\n";
  return GPUSampler(vocab_size, device, ft);
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
