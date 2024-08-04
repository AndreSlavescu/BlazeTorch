#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
using namespace torch::jit;

bool find_tensor(Value *input, at::Tensor *tensor, std::unordered_map<Value*, torch::Tensor> value_to_tensor);
void blazetorch_compile(Graph &graph, std::vector<torch::Tensor> &tensors);
