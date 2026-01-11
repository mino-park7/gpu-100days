#include <ATen/Dispatch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void matrix_sub_kernel(const T* a, const T* b, T* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] - b[idx];
    }
}

torch::Tensor matrix_sub(torch::Tensor a, torch::Tensor b) {
    // Check inputs
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(a.dtype() == b.dtype(), "Input tensors must have the same dtype");

    // Create output tensor
    auto c = torch::empty_like(a);
    int N = a.numel();

    // Launch kernel configuration
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "matrix_sub", [&] {
            const scalar_t* a_data = a.data_ptr<scalar_t>();
            const scalar_t* b_data = b.data_ptr<scalar_t>();
            scalar_t* c_data = c.data_ptr<scalar_t>();
            matrix_sub_kernel<scalar_t><<<grid, block>>>(a_data, b_data, c_data, N);
        });

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return c;
}
