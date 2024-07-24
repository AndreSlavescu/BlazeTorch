#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

void FuseAddDiv(std::shared_ptr<torch::jit::Graph> graph);
