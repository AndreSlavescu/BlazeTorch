

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
    Node *in_node;
};

class optCompiler
{
public:
    optCompiler(const torch::jit::Node *node);
    ~optCompiler();
    int getGraphSize() const;
    void setGraphSize(int g_size);
    void run(torch::jit::Stack *stack);

private:
    std::shared_ptr<torch::jit::Graph> subgraph_;                                 
    std::unordered_map<torch::jit::CompleteArgumentSpec, Compiled_info*> cache_; 
    int graph_size;
};
