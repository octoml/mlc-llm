#include <tvm/runtime/container/array.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include "sampler.h"
#include "../function_table.h"
#include "../data.h"
#include "../config.h"
#include "../../random.h"
namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;


/*! \brief The class of text data, containing a text string. */
class SamplerNode : public Object {
 public:
  // virtual std::vector<SampleResult> BatchDecode(NDArray probs) = 0;
  static constexpr const char* _type_key = "mlc.serve.Sampler";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;

  TVM_DECLARE_BASE_OBJECT_INFO(SamplerNode, Object);
};

// class SamplerBase : public ObjectRef {
//   public:
//   // explicit Sampler(int vc_size);
//   TVM_DEFINE_OBJECT_REF_METHODS(Sampler, ObjectRef, SamplerNode);
// };
/*! \brief The class of text data, containing a text string. */
class GPUSamplerNode : public SamplerNode {
 public:
  /*! \brief The text string. */
  int vocab_size{0};
  Sampler gpu_sampler;
  RandomGenerator rng;

  // std::vector<SampleResult> BatchDecode(NDArray probs) final;
  static constexpr const char* _type_key = "mlc.serve.GPUSampler";
  TVM_DECLARE_BASE_OBJECT_INFO(GPUSamplerNode, SamplerNode);
};

class GPUSamplerTest : public Sampler {
 public:
  explicit GPUSamplerTest(int vocab_size, DLDevice device, FunctionTable& ft);
  TVM_DEFINE_OBJECT_REF_METHODS(GPUSamplerTest, Sampler, GPUSamplerNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc
