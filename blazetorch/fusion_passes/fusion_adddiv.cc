#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include "fusion_adddiv.h"

using namespace torch::jit;

void fuseAddDivImpl(std::shared_ptr<torch::jit::Graph> graph)
{
    SubgraphRewriter rewriter;
    std::string add_div_0 = R"(
    graph(%a, %b, %c, %alpha):
        %add_res = aten::add(%a, %b, %alpha)
        %res = aten::div(%add_res, %c)
        return (%res))";
    std::string add_div_1 = R"(
    graph(%a, %b, %c, %alpha):
        %add_res = aten::add(%a, %b, %alpha)
        %res = aten::div_(%add_res, %c)
        return (%res))";
    std::string add_div_fused = R"(
    graph(%a, %b, %c, %alpha):
        %res = aten::addcdiv(%a, %b, %c, %alpha)
        return (%res))";
    std::string add_div_inplace = R"(
    graph(%a, %b, %c, %alpha):
        %add_res = aten::add_(%a, %b, %alpha)
        %res = aten::div_(%add_res, %c)
        return (%res))";
    std::string add_div_inplace_fused = R"(
    graph(%a, %b, %c, %alpha):
        %res = aten::addcdiv_(%a, %b, %c, %alpha)
        return (%res))";
    rewriter.RegisterRewritePattern(add_div_0, add_div_fused);
    rewriter.RegisterRewritePattern(add_div_1, add_div_fused);
    rewriter.RegisterRewritePattern(add_div_inplace, add_div_inplace_fused);
    rewriter.runOnGraph(graph);
}

void FuseAddDiv(std::shared_ptr<torch::jit::Graph> graph)
{
    fuseAddDivImpl(graph);
}
