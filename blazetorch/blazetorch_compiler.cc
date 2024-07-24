#include <pybind11/pybind11.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include "blazetorch_compiler.h"

namespace py = pybind11;
using namespace torch::jit;

bool find_tensor(Value *input, at::Tensor *tensor, std::unordered_map<Value*, torch::Tensor> value_to_tensor)
{
    if (value_to_tensor.find(input) != value_to_tensor.end())
    {
        *tensor = value_to_tensor[input];
        return true;
    }
    else
    {
        Graph *graph = input->owningGraph();
        for (auto node : graph->nodes())
        {
            if (node->kind() == prim::Constant && input == node->outputs()[0])
            {

                auto ivalue = toIValue(node->outputs()[0]);
                if (ivalue->isTensor())
                {
                    *tensor = ivalue->toTensor();
                    return true;
                }
            }
        }
    }
    return false;
}

void blazetorch_compile(Graph &graph, std::vector<torch::Tensor> &tensors)
{
    std::unordered_map<Value *, torch::Tensor> value_to_tensor;
    for (size_t i = 1; i < graph.inputs().size(); ++i) // weights, bias may come from input
    {
        auto value_input = graph.inputs()[i];
        value_to_tensor[value_input] = tensors[i - 1];
    }
    int g_size = 0;
    for (auto node : graph.nodes())
        g_size++;

    bool ret;
    for (auto node : graph.nodes())
    {
        int num_input = 1;
        if (node->kind() != prim::Constant && node->kind() != prim::ListConstruct)
        {
            if (node->kind() == aten::linear)
            {
                at::Tensor kk, bb;
                ret = find_tensor(node->input(1), &kk, value_to_tensor);
                assert(ret);
                assert(kk.has_storage());
                float *weight = (float *)kk.data_ptr();
                ret = find_tensor(node->input(2), &bb, value_to_tensor);
                float *bias = NULL;
                if (ret)
                {
                    bias = (float *)bb.data_ptr();
                }
                int i = 0, o = 0;
                o = kk.sizes()[0];
                i = kk.sizes()[1];
                std::cout << "Compiled Linear " << std::endl;
            }
        }
    }
    return;
}


