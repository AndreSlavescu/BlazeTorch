#include <iostream>
#include <stack>
#include <numeric>
#include <c10/util/C++17.h>
#include "process_graph_ops.h"

void process_matmul(
    std::shared_ptr<Compiled_info> cinfo, 
    Node* node, 
    const std::unordered_map<Value*, 
    IValue>& value_to_ivalue, 
    int& out_dim, 
    int& w_size
)
{
    assert(value_to_ivalue.find(node->input(1)) != value_to_ivalue.end());
    cinfo->weight = get_tensor(&value_to_ivalue.at(node->input(1)));
    out_dim = cinfo->weight.sizes()[cinfo->weight.dim() - 1];
    w_size = 1;
    for (auto d : cinfo->weight.sizes()) {
        w_size *= d;
    }
    cinfo->in_node = node;
    if (tdebug) {
        std::cout << "Weights:" << cinfo->weight.sizes() << std::endl;
    }
}

void process_div(std::shared_ptr<Compiled_info> cinfo, Node* node, int w_size)
{
    float div_const = get_const_double(node->input(1));
    assert(node->inputs()[0]->unique() == cinfo->in_node->outputs()[0]->unique());
    if (tdebug) {
        std::cout << "div_const: " << div_const << std::endl;
    }
    auto weight_accessor = cinfo->weight.accessor<float, 1>();
    for (int x = 0; x < w_size; x++) {
        weight_accessor[x] /= div_const;
    }
}