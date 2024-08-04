#include <pybind11/pybind11.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include "blazetorch_compiler.h"

namespace py = pybind11;

namespace blazetorch {

namespace {

bool find_tensor(torch::jit::Value* input, std::shared_ptr<at::Tensor>& tensor, const std::unordered_map<torch::jit::Value*, torch::Tensor>& value_to_tensor)
{
    auto it = value_to_tensor.find(input);
    if (it != value_to_tensor.end())
    {
        tensor = std::make_shared<at::Tensor>(it->second);
        return true;
    }
    
    torch::jit::Graph* graph = input->owningGraph();
    for (const auto& node : graph->nodes())
    {
        if (node->kind() == torch::jit::prim::Constant && input == node->outputs()[0])
        {
            auto ivalue = torch::jit::toIValue(node->outputs()[0]);
            if (ivalue->isTensor())
            {
                tensor = std::make_shared<at::Tensor>(ivalue->toTensor());
                return true;
            }
        }
    }
    
    return false;
}

} // anonymous namespace

void blazetorch_compile(torch::jit::Graph &graph, const std::vector<torch::Tensor> &tensors)
{
    std::unordered_map<torch::jit::Value*, torch::Tensor> value_to_tensor;
    for (size_t i = 1; i < graph.inputs().size(); ++i)
    {
        value_to_tensor[graph.inputs()[i]] = tensors[i - 1];
    }

    for (const auto &node : graph.nodes())
    {
        if (node->kind() == torch::jit::aten::linear)
        {
            auto weight = std::make_shared<at::Tensor>(); 
            auto bias = std::make_shared<at::Tensor>();   
            if (!find_tensor(node->input(1), weight, value_to_tensor))
            {
                throw std::runtime_error("Failed to find weight tensor for linear layer");
            }
            assert(weight->has_storage());

            std::unique_ptr<float[]> weight_data(weight->data_ptr<float>());
            std::unique_ptr<float[]> bias_data;

            if (find_tensor(node->input(2), bias, value_to_tensor))
            {
                bias_data.reset(bias->data_ptr<float>());
            }

            int64_t output_features = weight->size(0);
            int64_t input_features = weight->size(1);

            std::cout << "Compiled Linear layer: " << input_features << " -> " << output_features << std::endl;
        }
    }
}

} // namespace blazetorch
