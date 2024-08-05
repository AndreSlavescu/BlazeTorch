#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <ATen/ATen.h>

using namespace torch::jit;

struct Compiled_info;

c10::List<int64_t> get_const_intlist(Value *input);
int64_t get_const_int(Value *input);
double get_const_double(Value *input);
bool get_const_bool(Value *input);
c10::intrusive_ptr<c10::ivalue::ConstantString> get_const_string(Value *input);
at::Tensor get_tensor(IValue *input);
c10::List<at::Tensor> get_listtensor(IValue *input);
void process_matmul(
    std::shared_ptr<Compiled_info> cinfo, 
    Node* node, 
    const std::unordered_map<Value*, IValue>& value_to_ivalue, 
    int& out_dim, 
    int& w_size,
    bool debug
);
void process_div(std::shared_ptr<Compiled_info> cinfo, Node* node, int w_size, bool debug);
