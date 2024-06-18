#include <tvm/runtime/container/array.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include "sampler.h"
#include "../logit_processor.h"
#include "../function_table.h"
#include "../data.h"
#include "../config.h"
#include "../../random.h"
namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;


/*! \brief The base interface class for Sampler. */
class SamplerNode : public Object {
 public:
  static constexpr const char* _type_key = "mlc.serve.Sampler";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;

  TVM_DECLARE_BASE_OBJECT_INFO(SamplerNode, Object);
};

/*! \brief The interface class for GPU sampler. */
class GPUSamplerNode : public SamplerNode {
 public:
  int vocab_size{0};
  Sampler gpu_sampler;
  static constexpr const char* _type_key = "mlc.serve.GPUSampler";
  TVM_DECLARE_BASE_OBJECT_INFO(GPUSamplerNode, SamplerNode);
};


class GPUSamplerTest : public Sampler {
 public:
  explicit GPUSamplerTest(int vocab_size, DLDevice device, FunctionTable& ft);
  TVM_DEFINE_OBJECT_REF_METHODS(GPUSamplerTest, Sampler, GPUSamplerNode);
};

/*! \brief The base interface class for logits processor. */
class LogitsProcessorNode : public Object {
 public:
  static constexpr const char* _type_key = "mlc.serve.LogitsProcessor";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;

  TVM_DECLARE_BASE_OBJECT_INFO(LogitsProcessorNode, Object);
};

/*! \brief The GPU logits processor class. */
class GPULogitsProcessorNode : public LogitsProcessorNode {
 public:
  LogitProcessor logits_processor_;
  int vocab_size_{0};
  int max_num_token_{0};
  static constexpr const char* _type_key = "mlc.serve.GPULogitsProcessor";
  TVM_DECLARE_BASE_OBJECT_INFO(GPULogitsProcessorNode, LogitsProcessorNode);
};

class GPULogitsProcessorTest : public LogitProcessor {
 public:
  explicit GPULogitsProcessorTest(int max_num_token, int vocab_size, DLDevice device);
  TVM_DEFINE_OBJECT_REF_METHODS(GPULogitsProcessorTest, LogitProcessor, LogitsProcessorNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc
