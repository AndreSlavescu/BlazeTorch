// based on: https://github.com/pytorch/tvm
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include "trace_utils.h"
#include "fuse_candidates.h"

using namespace torch::jit;

value_list sortReverseTopological(ArrayRef<Value*> inputs, Block *block)
{
    value_list result;
    for (auto input : inputs)
    {
        if (input->node()->owningBlock() == block)
        {
            result.push_back(input);
        }
    }
    std::sort(
        result.begin(), 
        result.end(), 
        [&](Value *a, Value *b)
        { 
            return a->node()->isAfter(b->node()); 
        }
    );
    return result;
}

bool supported(const torch::jit::Node *node)
{
    switch (node->kind())
    {
    case aten::matmul:
    case aten::div:
        return true;
    default:
        return false;
    }
    return false;
}

bool supported(Block *block)
{
    for (Node *node : block->nodes())
    {
        if (!supported(node))
        {
            return false;
        }
    }
    return true;
}

c10::optional<Node*> tryMerge(
    Node *consumer,
    Node *producer,
    AliasDb &aliasDb
) {
    CHECKEQ(supported(producer));
    CHECKEQ((supported(consumer) || consumer->kind() == getBlazetorchSymbol()));
    CHECKEQ(aliasDb.moveAfterTopologicallyValid(consumer, producer));

    if (!(aliasDb.isMutable(consumer) && aliasDb.isMutable(producer)))
    {
        if (aliasDb.isMutable(consumer))
        {
            CHECKEQ(!aliasDb.hasInputWriters(producer));
        }
        else if (aliasDb.isMutable(producer))
        {
            CHECKEQ(!aliasDb.hasOutputWriters(consumer));
        }
    }
    if (!consumer->hasAttribute(attr::Subgraph) &&
        consumer->kind() != getBlazetorchSymbol())
    {
        consumer = SubgraphUtils::createSingletonSubgraph(consumer, getBlazetorchSymbol());
    }
    if (producer->kind() == prim::Constant)
    {
        auto &subgraph = consumer->g(attr::Subgraph);
        Node *in_const = subgraph->createClone(producer, [](Value *) -> Value *
                                               { throw std::runtime_error("unexpected input"); });
        subgraph->insertNode(in_const);
    }
    else
    {
        SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
    }

    return consumer;
}

std::pair<graph_node_list::iterator, bool> scanNode(
    Node *consumer,
    AliasDb &aliasDb,
    Block *block
) {
    auto inputs = sortReverseTopological(consumer->inputs(), block);
    for (auto input : inputs)
    {
        if (auto group = tryMerge(consumer, input->node(), aliasDb))
        {
            return {group.value()->reverseIterator(), true};
        }
    }
    return {++consumer->reverseIterator(), false};
}

void FuseSupportedOps(std::shared_ptr<Graph> graph)
{
    AliasDb aliasDb(graph);
    auto block = graph->block();

    bool any_changed{true};
    while (any_changed)
    {
        any_changed = false;
        for (auto it = block->nodes().rbegin(); it != block->nodes().rend();)
        {
            bool changed;
            std::tie(it, changed) = scanNode(*it, aliasDb, block);
            any_changed |= changed;
        }
    }
    EliminateCommonSubexpression(graph);
    EliminateDeadCode(graph);
}

const torch::jit::Symbol &getBlazetorchSymbol()
{
    static torch::jit::Symbol sym = torch::jit::Symbol::fromQualString("opt::CompilationGroup");
    return sym;
}
