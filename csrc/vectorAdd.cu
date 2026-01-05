#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA kernel for vector addition
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// PyTorch wrapper function
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    // Check inputs
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensors must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensors must be float32");

    // Create output tensor
    auto c = torch::empty_like(a);

    // Get tensor data pointers
    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();
    float* c_data = c.data_ptr<float>();

    // Launch kernel
    int n = a.numel();
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add_kernel<<<num_blocks, threads_per_block>>>(a_data, b_data, c_data, n);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return c;
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "CUDA vector addition");
}
