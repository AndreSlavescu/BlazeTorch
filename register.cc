#include <torch/extension.h>
#include <pybind11/pybind11.h>

torch::Tensor custom_add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &custom_add, "A function that adds two tensors");
}
