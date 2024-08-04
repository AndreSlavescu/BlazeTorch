
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/fuse_relu.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <ATen/ATen.h>
#include "fuse_candidates.h"
#include "fusion_passes/fusion_adddiv.h"
#include "fusion_passes/fusion_addmul.h"
#include "compiler.h"
#include "blazetorch_compiler.h"

namespace py = pybind11;
using namespace torch;
using namespace torch::jit;

static bool opt_enabled = false;

void register_opt()
{
    torch::jit::RegisterPass pass([](std::shared_ptr<Graph> &g)
    {
        if (opt_enabled) 
        {
            FuseLinear(g);
            BatchMM(g);
            FuseAddRelu(g);
            FuseAddDiv(g);
            FuseAddMul(g);
            FuseSupportedOps(g);
        } 
    });

    torch::jit::RegisterOperators op({torch::jit::Operator(
        getBlazetorchSymbol(),
        [](const torch::jit::Node *node) -> torch::jit::Operation 
        {
            auto compiler = std::make_shared<blazeTorchCompiler>(node);
            return [compiler](torch::jit::Stack& stack) {
                compiler->run(&stack); return 0;
            }; 
        },
        AliasAnalysisKind::PURE_FUNCTION)}
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    register_opt();
    m.def("enable", [](){ opt_enabled = true; });
    m.def("disable", [](){ opt_enabled = false; });
    m.def("blazetorch_compile", &blazetorch_compile, "blazetorch_compile");
}
