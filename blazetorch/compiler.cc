#include <iostream>
#include <stack>
#include <numeric>
#include <c10/util/C++17.h>
#include <ATen/ATen.h>
#include "compiler.h"

using namespace torch::jit;
bool tdebug = false;

static void print_node(Node *node)
{
    std::cout << "Node: " << node->kind().toDisplayString() << std::endl;
    std::cout << "  Inputs (" << node->inputs().size() << "):" << std::endl;
    for (size_t i = 0; i < node->inputs().size(); i++)
    {
        std::cout << "    " << i << ": " << node->inputs()[i]->unique() << std::endl;
    }
    std::cout << "  Outputs (" << node->outputs().size() << "):" << std::endl;
    for (size_t i = 0; i < node->outputs().size(); i++)
    {
        std::cout << "    " << i << ": " << node->outputs()[i]->unique() << std::endl;
    }
    std::cout << std::endl;
}

blazeTorchCompiler::blazeTorchCompiler(const torch::jit::Node *node) 
    : subgraph_(node->g(torch::jit::attr::Subgraph))
{
    int g_size = 0;
    for (auto node : subgraph_->nodes()) 
    {
        g_size++;
    }
    setGraphSize(g_size);
}

blazeTorchCompiler::~blazeTorchCompiler()
{
    for (auto cache_entry : cache_)
        delete cache_entry.second;
}

int blazeTorchCompiler::getGraphSize() const {
    return graph_size;
}

void blazeTorchCompiler::setGraphSize(int g_size) {
    graph_size = g_size;
}

void blazeTorchCompiler::run(torch::jit::Stack *stack)
{
    const at::ArrayRef<Value*> &graph_inputs = subgraph_->inputs();
    const auto num_inputs = graph_inputs.size();
    at::ArrayRef<IValue> inputs = last(stack, num_inputs);

    std::unordered_map<Value*, IValue> value_to_ivalue;
    for (size_t i = 0; i < inputs.size(); ++i) {
        value_to_ivalue[subgraph_->inputs()[i]] = inputs[i];
    }

    CompleteArgumentSpec spec{false, ArrayRef<IValue>(inputs)};

    if (tdebug) {
        for (auto node : subgraph_->nodes()) {
            print_node(node);
        }
    }

    std::shared_ptr<Compiled_info> cinfo;
    if (cache_.find(spec) == cache_.end()) {
        cinfo = std::make_shared<Compiled_info>();
        processGraph(cinfo, value_to_ivalue);
        cache_[spec] = cinfo.get(); 
    } else {
        cinfo = std::shared_ptr<Compiled_info>(cache_[spec], [](Compiled_info*){});  
    }

    assert(cinfo);
    assert(value_to_ivalue.find(cinfo->in_node->input(0)) != value_to_ivalue.end());
    at::Tensor in_data = get_tensor(&value_to_ivalue[cinfo->in_node->input(0)]);
    auto out_tensor = at::matmul(in_data, cinfo->weight);

    if (tdebug) {
        // input tensor
        for (const auto& input : inputs) {
            if (input.isTensor()) {
                const auto& tensor = input.toTensor();
                std::cout << "Input Tensor: [" 
                          << tensor.sizes() << ", " 
                          << tensor.dtype() << ", " 
                          << tensor.device() << "]" 
                          << std::endl;
            }
        }

        // output tensor
        for (const auto& output : subgraph_->outputs()) {
            if (output->type()->isSubtypeOf(TensorType::get())) {
                auto tensor_type = output->type()->expect<TensorType>();
                std::cout << "Output Tensor: [";
                if (tensor_type->sizes().isComplete()) {
                    std::cout << tensor_type->sizes().concrete_sizes().value();
                } else {
                    std::cout << "unknown_size";
                }
                std::cout << ", " << tensor_type->scalarType().value_or(at::ScalarType::Undefined);
                if (tensor_type->device().has_value()) {
                    std::cout << ", " << tensor_type->device().value();
                } else {
                    std::cout << ", unknown_device";
                }
                std::cout << "]" << std::endl;
            }
        }
    }

    drop(stack, num_inputs); 
    for (auto &output : subgraph_->outputs()) {
        auto var = torch::autograd::make_variable(out_tensor);
        stack->push_back(IValue(var));
    }
}

void blazeTorchCompiler::processGraph(std::shared_ptr<Compiled_info> cinfo, const std::unordered_map<Value*, IValue>& value_to_ivalue)
{
    Node* in_node = nullptr;
    int out_dim = 1;
    int w_size = 1;

    for (auto node : subgraph_->nodes()) {
        if (supported(node) && node->kind() != prim::Constant && node->kind() != prim::ListConstruct) {
            if (!in_node) {
                in_node = node;  
            }
            if (node->kind() == aten::matmul) {
                process_matmul(cinfo, node, value_to_ivalue, out_dim, w_size, tdebug);
            } else if (node->kind() == aten::div) {
                process_div(cinfo, node, w_size, tdebug);
            }
        }
    }

    assert(in_node);
    assert(value_to_ivalue.find(in_node->input(0)) != value_to_ivalue.end());
    auto& ivalue = value_to_ivalue.at(in_node->input(0));
    at::Tensor in_data = get_tensor(const_cast<IValue*>(&ivalue));
    
    std::vector<int64_t> osizes;
    cinfo->out_size = 1;
    for (auto dim : in_data.sizes()) {
        cinfo->out_size *= static_cast<int64_t>(dim);
        osizes.push_back(static_cast<int64_t>(dim));
    }
    osizes.back() = out_dim;

    if (tdebug) {
        std::cout << "Inputs: [" << in_data.sizes() << ", " << in_data.dtype() << ", " << in_data.device() << "]" << std::endl;
    }

    cinfo->out_shape = std::move(osizes);
    cinfo->in_node = in_node;
}