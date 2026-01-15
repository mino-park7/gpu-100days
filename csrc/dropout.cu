#include <ATen/Dispatch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/extension.h>

__global__ void setup_rand(curandState* state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

template <typename T>
__global__ void seeded_dropout_kernel(const T* x_ptr, T* output_ptr, int n_elements, float p,
                                      int seed, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        curandState localState = state[idx];

        float random_num = curand_uniform(&localState);

        T x = x_ptr[idx];
        T output;
        if (random_num > p) {
            output = x / (1 - p);
        } else {
            output = 0.0;
        }

        output_ptr[idx] = output;
    }
}

torch::Tensor seeded_dropout(torch::Tensor x, float p, int seed) {
    // Check inputs
    TORCH_CHECK(x.device().is_cuda(), "Input tensor x must be on CUDA");

    // Create output tensor
    auto output = torch::empty_like(x);
    int N = x.numel();

    // Launch kernel configuration
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "seeded_dropout", [&] {
            const scalar_t* x_data = x.data_ptr<scalar_t>();
            scalar_t* output_data = output.data_ptr<scalar_t>();

            // Allocate device memory for curandState
            curandState* state;
            cudaError_t malloc_err = cudaMalloc(&state, N * sizeof(curandState));
            if (malloc_err != cudaSuccess) {
                TORCH_CHECK(false,
                            "Failed to allocate device memory: ", cudaGetErrorString(malloc_err));
            }

            setup_rand<<<grid, block>>>(state, seed);
            cudaDeviceSynchronize();

            seeded_dropout_kernel<scalar_t>
                <<<grid, block>>>(x_data, output_data, N, p, seed, state);
            cudaDeviceSynchronize();

            // Free device memory
            cudaFree(state);

            // Check for CUDA errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
            }
        });

    return output;
}
