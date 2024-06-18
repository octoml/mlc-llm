

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
#include <tvm/runtime/memory/memory_manager.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

#pragma message("YYYYYYYYYYYYYYYYYYYYYYY")

TVM_REGISTER_OBJECT_TYPE(SamplerNode);
TVM_REGISTER_OBJECT_TYPE(GPUSamplerNode);

GPUSamplerTest::GPUSamplerTest(int vocab_size, DLDevice device, FunctionTable& ft) {
  ObjectPtr<GPUSamplerNode> n = make_object<GPUSamplerNode>();
  n->vocab_size = vocab_size;
  n->gpu_sampler = Sampler::CreateGPUSampler(64, vocab_size, &ft, device, {});
  data_ = std::move(n);
  // 
}

// std::vector<SampleResult> GPUSamplerNode::BatchDecode(NDArray probs)
// {
//     // std::vector<int> sample_indices(num_rsentries);
//     // std::iota(sample_indices.begin(), sample_indices.end(), 0);
//     // std::vector<SampleResult> sample_results = data_->IsInstance<GPUSamplerNode>()
//     // .BatchSampleTokensWithProbBeforeTopP(
//     //     probs_on_device, sample_indices, request_ids, generation_cfg, rngs);
//   return std::vector<SampleResult>();
// }

TVM_REGISTER_GLOBAL("mlc.serve.GPUSamplerBatchDecode").set_body_typed([](GPUSamplerTest sampler, NDArray samples) {
  int num_rsentries = samples->shape[0]; // to check
  static const String conf_string = "{\"top_p\": 5, \"temperature\": 0.7, \"frequency_penalty\": 0.0, \"presence_penalty\": 0.0}";
  static GenerationConfig generation_config(conf_string);
  std::vector<RandomGenerator*> rngs;
  Array<String> request_ids;
  Array<GenerationConfig> generation_cfg;
  rngs.reserve(num_rsentries);
  generation_cfg.reserve(num_rsentries);
  request_ids.reserve(num_rsentries);
  std::vector<int> sample_indices(num_rsentries);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);
  // RandomGenerator rng;
  for (size_t i = 0; i < num_rsentries; ++i) {
     rngs.push_back(const_cast<RandomGenerator*>(&sampler->rng));
     generation_cfg.push_back(generation_config);
     request_ids.push_back(std::to_string(i));
  }
  auto res = sampler->gpu_sampler->BatchSampleTokensWithProbBeforeTopP(
        samples, sample_indices, request_ids, generation_cfg, rngs);
});

class GPUSamplerInstance
{
public:
  static GPUSamplerInstance& GPUInstance()
  {
    static GPUSamplerInstance singleton;
    return singleton;
  }
  FunctionTable& getTable(DLDevice device) {
    if (!initialized_) {
      auto executable = tvm::runtime::Module::LoadFromFile(reload_lib_path);
      device_ = device;
      auto fload_exec = executable->GetFunction("vm_load_executable");
      local_vm_ = fload_exec();
      local_vm_->GetFunction("vm_initialization")(
        static_cast<int>(device.device_type), device.device_id,
        static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
        static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled));
      ft_.gpu_multinomial_from_uniform_func_ = local_vm_->GetFunction("multinomial_from_uniform", true);
      ft_.gpu_argsort_probs_func_ = local_vm_->GetFunction("argsort_probs", true);
      ft_.gpu_sample_with_top_p_func_ = local_vm_->GetFunction("sample_with_top_p", true);
      ft_.gpu_sampler_take_probs_func_ = local_vm_->GetFunction("sampler_take_probs", true);
      ft_.gpu_verify_draft_tokens_func_ = local_vm_->GetFunction("sampler_verify_draft_tokens", true);
      ft_.gpu_renormalize_by_top_p_func_ = local_vm_->GetFunction("renormalize_by_top_p", true);
      initialized_ = true;
    }
    return ft_;
  }
private:
  GPUSamplerInstance() {
  }
  ~GPUSamplerInstance() {
  }
  GPUSamplerInstance(const GPUSamplerInstance&);
  GPUSamplerInstance& operator=(const GPUSamplerInstance&);
  FunctionTable ft_;
  tvm::runtime::Module local_vm_{nullptr};
  DLDevice device_;
  bool initialized_{false};
  const std::string reload_lib_path = "/home/sshtin/dev/ollm/mlc-serve/dist/Mistral-7B-Instruct-v0.2-q0f16-vllm-1gpu/Mistral-7B-Instruct-v0.2-q0f16-vllm-1gpu-allreduce_AUTO.so";
};

TVM_REGISTER_GLOBAL("mlc.serve.GPUSampler").set_body_typed([](int vocab_size, DLDevice device) {
  
  // /home/sshtin/dev/ollm/deps/mlc-llm/cpp/serve/function_table.h
  std::cout << "device " << device << "\n";
  // local_gpu_device = device;
  return GPUSamplerTest(vocab_size, device, GPUSamplerInstance::GPUInstance().getTable(device));
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
