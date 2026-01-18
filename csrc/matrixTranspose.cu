#include <ATen/Dispatch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void matrix_transpose_kernel(const T* input, T* output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int idx = row * N + col;
        int new_index = col * M + row;

        output[new_index] = input[idx];
    }
}

torch::Tensor matrix_transpose(torch::Tensor input) {
    // Check inputs
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be a 2D tensor");

    // Create output tensor

    int M = input.size(0);
    int N = input.size(1);
    auto output = torch::empty({N, M}, input.options());

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "matrix_transpose",
        [&] {
            const scalar_t* input_data = input.data_ptr<scalar_t>();
            scalar_t* output_data = output.data_ptr<scalar_t>();
            matrix_transpose_kernel<scalar_t><<<grid, block>>>(input_data, output_data, M, N);
        });

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
