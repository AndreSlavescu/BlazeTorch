#include "compiler.h"
#include <iostream>
#include <stack>
#include <numeric>
#include <c10/util/C++17.h>
#include "process_graph_ops.h"
#include <ATen/ATen.h>
#include <ATen/TensorAccessor.h>

c10::List<int64_t> get_const_intlist(Value *input)
{
    auto nn = std::unique_ptr<Node>(input->node());
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isIntList());
    return ivalue->toIntList();
}

int64_t get_const_int(Value *input)
{
    auto nn = std::unique_ptr<Node>(input->node());
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isInt());
    return ivalue->toInt();
}

double get_const_double(Value *input)
{
    auto nn = std::unique_ptr<Node>(input->node());
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isDouble());
    return ivalue->toDouble();
}

bool get_const_bool(Value *input)
{
    auto nn = std::unique_ptr<Node>(input->node());
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isBool());
    return ivalue->toBool();
}

c10::intrusive_ptr<c10::ivalue::ConstantString> get_const_string(Value *input)
{
    auto nn = std::unique_ptr<Node>(input->node());
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isString());
    return ivalue->toString(); 
}

at::Tensor get_tensor(IValue *input)
{
    assert(input->isTensor());
    return input->toTensor();
}

c10::List<at::Tensor> get_listtensor(IValue *input)
{
    assert(input->isTensorList());
    return input->toTensorList();
}

void process_matmul(
    std::shared_ptr<Compiled_info> cinfo, 
    Node* node, 
    const std::unordered_map<Value*, IValue>& value_to_ivalue, 
    int& out_dim, 
    int& w_size,
    bool debug
)
{
    assert(value_to_ivalue.find(node->input(1)) != value_to_ivalue.end());
    auto& ovalue = value_to_ivalue.at(node->input(1));
    cinfo->weight = get_tensor(const_cast<IValue*>(&ovalue));
    out_dim = cinfo->weight.sizes()[cinfo->weight.dim() - 1];
    w_size = 1;
    for (auto d : cinfo->weight.sizes()) {
        w_size *= d;
    }
    cinfo->in_node = node;
    if (debug) {
        std::cout << "Weights:" << cinfo->weight.sizes() << std::endl;
    }
}

void process_div(std::shared_ptr<Compiled_info> cinfo, Node* node, int w_size, bool debug)
{
    float div_const = get_const_double(node->input(1));
    assert(node->inputs()[0]->unique() == cinfo->in_node->outputs()[0]->unique());
    if (debug) {
        std::cout << "div_const: " << div_const << std::endl;
    }
    auto weight_accessor = cinfo->weight.accessor<float, 1>();
    for (int x = 0; x < w_size; x++) {
        weight_accessor[x] /= div_const;
    }
}