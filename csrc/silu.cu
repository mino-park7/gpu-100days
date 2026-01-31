#include <ATen/Dispatch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void silu_kernel(const T* input, T* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = static_cast<float>(input[idx]);
        float y = x / (1.0f + expf(-x));
        output[idx] = static_cast<T>(y);
    }
}

torch::Tensor silu(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat16 || input.dtype() == torch::kFloat32 ||
                    input.dtype() == torch::kBFloat16,
                "Input tensor must be float16, float32, or bfloat16");

    auto output = torch::empty_like(input);

    int n = input.numel();
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "silu", [&] {
            const scalar_t* input_data = input.data_ptr<scalar_t>();
            scalar_t* output_data = output.data_ptr<scalar_t>();
            silu_kernel<scalar_t><<<num_blocks, threads_per_block>>>(input_data, output_data, n);
        });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    return output;
}
