#include <torch/extension.h>

// Forward declarations
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
torch::Tensor matrix_add(torch::Tensor a, torch::Tensor b);
torch::Tensor matrix_sub(torch::Tensor a, torch::Tensor b);
torch::Tensor gray_scale(torch::Tensor image);
torch::Tensor seeded_dropout(torch::Tensor x, float p, int seed);
torch::Tensor matrix_transpose(torch::Tensor input);
torch::Tensor softmax(torch::Tensor input);
torch::Tensor silu(torch::Tensor input);
// PyTorch bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "CUDA vector addition");
    m.def("matrix_add", &matrix_add, "CUDA matrix addition");
    m.def("matrix_sub", &matrix_sub, "CUDA matrix subtraction");
    m.def("gray_scale", &gray_scale, "CUDA grayscale conversion");
    m.def("seeded_dropout", &seeded_dropout, "CUDA seeded dropout");
    m.def("matrix_transpose", &matrix_transpose, "CUDA matrix transpose");
    m.def("softmax", &softmax, "CUDA softmax");
    m.def("silu", &silu, "CUDA silu");
}
