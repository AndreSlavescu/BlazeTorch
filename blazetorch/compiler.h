#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <ATen/ATen.h>
#include "fuse_candidates.h"

using namespace torch::jit;

struct Compiled_info
{
    uint64_t out_size;              
    std::vector<int64_t> out_shape;
    at::Tensor weight;
    Node* in_node;
};

class blazeTorchCompiler
{
public:
    blazeTorchCompiler(const torch::jit::Node *node);
    ~blazeTorchCompiler();
    int getGraphSize() const;
    void setGraphSize(int g_size);
    void run(torch::jit::Stack *stack);
    void processGraph(std::shared_ptr<Compiled_info> cinfo, const std::unordered_map<Value*, IValue>& value_to_ivalue);


private:
    std::shared_ptr<torch::jit::Graph> subgraph_;                                 
    std::unordered_map<torch::jit::CompleteArgumentSpec, Compiled_info*> cache_; 
    int graph_size;
};
