#include <torch/extension.h>

// Forward declarations
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor matrix_add(torch::Tensor a, torch::Tensor b);

// PyTorch bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "CUDA vector addition");
    m.def("matrix_add", &matrix_add, "CUDA matrix addition");
}
