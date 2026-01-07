#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void matrix_add_kernel(const float* a, const float* b, float* c, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int idx = row * N + col;
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor matrix_add(torch::Tensor a, torch::Tensor b) {
    // Check inputs
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensors must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensors must be float32");

    // Create output tensora
    auto c = torch::empty_like(a);

    // Get tensor data pointers
    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();
    float* c_data = c.data_ptr<float>();

    // Get tensor shapes
    int M = a.size(0);
    int N = a.size(1);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matrix_add_kernel<<<grid, block>>>(a_data, b_data, c_data, M, N);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return c;
}
