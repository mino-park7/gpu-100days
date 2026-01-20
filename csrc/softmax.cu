#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int WARP_SIZE = 32;

// Warp-level reduction for max
__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 최적화된 softmax kernel - 각 warp가 하나의 row를 처리
__global__ void softmax_kernel_optimized(const float* input, float* output, int n_rows,
                                         int n_cols) {
    // 각 warp가 하나의 row를 처리
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= n_rows) return;

    const float* row_input = input + warp_id * n_cols;
    float* row_output = output + warp_id * n_cols;

    // Step 1: Warp 내에서 병렬로 최대값 찾기
    float thread_max = -INFINITY;
    for (int i = lane_id; i < n_cols; i += WARP_SIZE) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    float max_val = warpReduceMax(thread_max);
    // Warp 내 모든 스레드가 같은 max_val을 가지도록 broadcast
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Step 2: exp(x - max)를 계산하면서 합 구하기
    float thread_sum = 0.0f;
    for (int i = lane_id; i < n_cols; i += WARP_SIZE) {
        float exp_val = expf(row_input[i] - max_val);
        row_output[i] = exp_val;
        thread_sum += exp_val;
    }
    float sum = warpReduceSum(thread_sum);
    // Warp 내 모든 스레드가 같은 sum을 가지도록 broadcast
    sum = __shfl_sync(0xffffffff, sum, 0);

    // Step 3: 정규화 (각 값을 합으로 나누기)
    for (int i = lane_id; i < n_cols; i += WARP_SIZE) {
        row_output[i] /= sum;
    }
}

// 간단한 버전 (작은 데이터용) - 각 스레드가 하나의 row 처리
__global__ void softmax_kernel_simple(const float* input, float* output, int n_rows, int n_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_rows) return;

    const float* row_input = input + row * n_cols;
    float* row_output = output + row * n_cols;

    // Step 1: 최대값 찾기
    float max_val = row_input[0];
    for (int i = 1; i < n_cols; i++) {
        max_val = fmaxf(max_val, row_input[i]);
    }

    // Step 2: exp(x - max)를 계산하면서 합 구하기
    float sum = 0.0f;
    for (int i = 0; i < n_cols; i++) {
        float exp_val = expf(row_input[i] - max_val);
        row_output[i] = exp_val;
        sum += exp_val;
    }

    // Step 3: 정규화
    for (int i = 0; i < n_cols; i++) {
        row_output[i] /= sum;
    }
}

// PyTorch 바인딩을 위한 wrapper 함수
torch::Tensor softmax(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);

    int n_rows = input.size(0);
    int n_cols = input.size(1);

    // n_cols가 작으면 simple 버전, 크면 warp-optimized 버전 사용
    if (n_cols <= 128) {
        // Simple version: 각 스레드가 하나의 row 처리
        const int threads = 256;
        const int blocks = (n_rows + threads - 1) / threads;

        softmax_kernel_simple<<<blocks, threads>>>(input.data_ptr<float>(),
                                                   output.data_ptr<float>(), n_rows, n_cols);
    } else {
        // Optimized version: 각 warp가 하나의 row 처리
        const int threads = 256;  // 8 warps per block
        const int warps_per_block = threads / WARP_SIZE;
        const int blocks = (n_rows + warps_per_block - 1) / warps_per_block;

        softmax_kernel_optimized<<<blocks, threads>>>(input.data_ptr<float>(),
                                                      output.data_ptr<float>(), n_rows, n_cols);
    }

    return output;
}
