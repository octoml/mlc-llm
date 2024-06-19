#include "sampler_func.h"
#include <tvm/runtime/registry.h>
#include <tvm/runtime/memory/memory_manager.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;


TVM_REGISTER_OBJECT_TYPE(SamplerNode);
TVM_REGISTER_OBJECT_TYPE(GPUSamplerNode);


GPUSamplerTest::GPUSamplerTest(int vocab_size, DLDevice device, FunctionTable& ft) {
  ObjectPtr<GPUSamplerNode> n = make_object<GPUSamplerNode>();
  n->vocab_size = vocab_size;
  n->gpu_sampler = Sampler::CreateGPUSampler(64, vocab_size, &ft, device, {});
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("mlc.serve.GPUSamplerBatchDecode").set_body([](TVMArgs args, TVMRetValue* ret) {
  GPUSamplerTest sampler = args[0];
  NDArray samples = args[1];
  int num_rsentries = samples->shape[0]; // to check
  static const String conf_string = "{\"top_p\": 5, \"temperature\": 0.7, \"frequency_penalty\": 0.7, \"presence_penalty\": 0.7}";
  static GenerationConfig generation_config(conf_string);
  Array<String> request_ids;
  Array<GenerationConfig> generation_cfg;
  generation_cfg.reserve(num_rsentries);
  request_ids.reserve(num_rsentries);
  std::vector<int> sample_indices(num_rsentries);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);
  for (size_t i = 0; i < num_rsentries; ++i) {
     generation_cfg.push_back(generation_config);
     request_ids.push_back(std::to_string(i));
  }
  auto result = sampler->gpu_sampler->BatchRenormalizeProbsByTopP(samples, sample_indices, request_ids, generation_cfg);
  *ret = result;
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
      ft_.apply_logit_bias_func_ = local_vm_->GetFunction("apply_logit_bias_inplace", true);
      ft_.apply_penalty_func_ = local_vm_->GetFunction("apply_penalty_inplace", true);
      ft_.apply_bitmask_func_ = local_vm_->GetFunction("apply_bitmask_inplace", true);
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
  const std::string reload_lib_path = "/home/sshtin/dev/ollm/mlc-serve/dist/Mistral-7B-Instruct-v0.2-q0f16-vllm-1gpu/Mistral-7B-Instruct-v0_2-q0f16-vllm-1gpu-allreduce_AUTO.so";
};

TVM_REGISTER_GLOBAL("mlc.serve.GPUSampler").set_body_typed([](int vocab_size, DLDevice device) {
  return GPUSamplerTest(vocab_size, device, GPUSamplerInstance::GPUInstance().getTable(device));
});

TVM_REGISTER_OBJECT_TYPE(LogitsProcessorNode);
TVM_REGISTER_OBJECT_TYPE(GPULogitsProcessorNode);

GPULogitsProcessorTest::GPULogitsProcessorTest(int max_num_token, int vocab_size, DLDevice device) {
  ObjectPtr<GPULogitsProcessorNode> n = make_object<GPULogitsProcessorNode>();
  n->vocab_size_ = vocab_size;
  n->max_num_token_ = max_num_token;
  n->logits_processor_ = LogitProcessor(max_num_token, vocab_size, &GPUSamplerInstance::GPUInstance().getTable(device), device, {});
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("mlc.serve.ComputeProbsFromLogits").set_body([](TVMArgs args, TVMRetValue* ret) {
  GPULogitsProcessorTest unit = args[0];
  NDArray logits_for_sample = args[1];
  std::cout << "Here!\n";
  int num_rsentries = logits_for_sample->shape[0]; // to check

  static const String conf_string = "{\"top_p\": 5, \"temperature\": 0.7, \"frequency_penalty\": 0.7, \"presence_penalty\": 0.7}";
  static GenerationConfig generation_config(conf_string);
  // explicit Request(String id, Array<Data> inputs, GenerationConfig generation_cfg);
// class RequestModelState : public ObjectRef {
//  public:
//   explicit RequestModelState(
//       Request request, int model_id, int64_t internal_id, Array<Data> inputs,
//       const std::optional<std::shared_ptr<GrammarStateInitContext>>& grammar_state_init_ctx);

  // Array<RequestModelState> mstates;
  // mstates.reserve(num_rsentries);
  // for (size_t i = 0; i < num_rsentries; ++i) {

  // }
  // unit->logit_processor_->InplaceUpdateLogits(logits_for_sample, generation_cfg, mstates_for_logitproc,
  //                                         request_ids);


  //   NDArray probs_on_device =
  //       logit_processor_->ComputeProbsFromLogits(logits_for_sample, generation_cfg, request_ids);

    // - Sample tokens.
  // GPUSamplerTest sampler = args[0];
  // NDArray samples = args[1];
  // int num_rsentries = samples->shape[0]; // to check
  // static const String conf_string = "{\"top_p\": 5, \"temperature\": 0.7, \"frequency_penalty\": 0.7, \"presence_penalty\": 0.7}";
  // static GenerationConfig generation_config(conf_string);
  // Array<String> request_ids;
  // Array<GenerationConfig> generation_cfg;
  // generation_cfg.reserve(num_rsentries);
  // request_ids.reserve(num_rsentries);
  // std::vector<int> sample_indices(num_rsentries);
  // std::iota(sample_indices.begin(), sample_indices.end(), 0);
  // for (size_t i = 0; i < num_rsentries; ++i) {
  //    generation_cfg.push_back(generation_config);
  //    request_ids.push_back(std::to_string(i));
  // }
  // auto result = sampler->gpu_sampler->BatchRenormalizeProbsByTopP(samples, sample_indices, request_ids, generation_cfg);
  *ret = logits_for_sample;
});

TVM_REGISTER_GLOBAL("mlc.serve.GPULogitsProcessor").set_body_typed([](int max_num_token, int vocab_size, DLDevice device) {
  return GPULogitsProcessorTest(max_num_token, vocab_size, device);
});

}  // namespace serve
}  // namespace llm
}  // namespace mlc
