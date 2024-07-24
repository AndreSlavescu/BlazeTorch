#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include "fusion_addmul.h"

using namespace torch::jit;

void fuseAddMulImpl(std::shared_ptr<torch::jit::Graph> graph)
{
    SubgraphRewriter rewriter;
    std::string add_mul_0 = R"(
    graph(%a, %b, %c, %alpha):
        %add_res = aten::add(%a, %b, %alpha)
        %res = aten::mul(%add_res, %c)
        return (%res))";
    std::string add_mul_1 = R"(
    graph(%a, %b, %c, %alpha):
        %add_res = aten::add(%a, %b, %alpha)
        %res = aten::mul_(%add_res, %c)
        return (%res))";
    std::string add_mul_fused = R"(
    graph(%a, %b, %c, %alpha):
        %res = aten::addcmul(%a, %b, %c, %alpha)
        return (%res))";
    std::string add_mul_inplace = R"(
    graph(%a, %b, %c, %alpha):
        %add_res = aten::add_(%a, %b, %alpha)
        %res = aten::mul_(%add_res, %c)
        return (%res))";
    std::string add_mul_inplace_fused = R"(
    graph(%a, %b, %c, %alpha):
        %res = aten::addcmul_(%a, %b, %c, %alpha)
        return (%res))";
    rewriter.RegisterRewritePattern(add_mul_0, add_mul_fused);
    rewriter.RegisterRewritePattern(add_mul_1, add_mul_fused);
    rewriter.RegisterRewritePattern(add_mul_inplace, add_mul_inplace_fused);
}

void FuseAddMul(std::shared_ptr<torch::jit::Graph> graph)
{
    fuseAddMulImpl(graph);
}
