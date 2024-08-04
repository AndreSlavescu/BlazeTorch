#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <ATen/ATen.h>
#include "fuse_candidates.h"

using namespace torch::jit;

struct Compiled_info;

void process_matmul(
    std::shared_ptr<Compiled_info> cinfo, 
    Node* node, 
    const std::unordered_map<Value*, IValue>& value_to_ivalue, 
    int& out_dim, 
    int& w_size
);
void process_div(std::shared_ptr<Compiled_info> cinfo, Node* node, int w_size);
