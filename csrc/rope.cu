#include <ATen/Dispatch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void rope_kernel(const T* q, const T* k, const T* cos, const T* sin, T* q_rot, T* k_rot,
                            int stride_qb, int stride_qs, int stride_qh, int stride_qd,
                            int stride_kb, int stride_ks, int stride_kh, int stride_kd,
                            int stride_cos_s, int stride_cos_d, int stride_sin_s,
                            int stride_sin_d) {

    int batch_id = blockIdx.x;
    int seq_len_id = blockIdx.y;
    int head_id = blockIdx.z;
    int dim_idx_half = threadIdx.x;

    int q_base_idx = batch_id * stride_qb + seq_len_id * stride_qs + head_id * stride_qh;
    int k_base_idx = batch_id * stride_kb + seq_len_id * stride_ks + head_id * stride_kh;
    int cos_idx = seq_len_id * stride_cos_s + dim_idx_half * stride_cos_d;
    int sin_idx = seq_len_id * stride_sin_s + dim_idx_half * stride_sin_d;
    int q_real_idx = q_base_idx + dim_idx_half * 2 * stride_qd;
    int q_imag_idx = q_base_idx + dim_idx_half * 2 * stride_qd + 1;
    int k_real_idx = k_base_idx + dim_idx_half * 2 * stride_kd;
    int k_imag_idx = k_base_idx + dim_idx_half * 2 * stride_kd + 1;

    float q_real = static_cast<float>(q[q_real_idx]);
    float q_imag = static_cast<float>(q[q_imag_idx]);
    float k_real = static_cast<float>(k[k_real_idx]);
    float k_imag = static_cast<float>(k[k_imag_idx]);

    float cos_val = cos[cos_idx];
    float sin_val = sin[sin_idx];

    T q_rot_real = static_cast<T>(q_real * cos_val - q_imag * sin_val);
    T q_rot_imag = static_cast<T>(q_real * sin_val + q_imag * cos_val);
    T k_rot_real = static_cast<T>(k_real * cos_val - k_imag * sin_val);
    T k_rot_imag = static_cast<T>(k_real * sin_val + k_imag * cos_val);

    q_rot[q_real_idx] = q_rot_real;
    q_rot[q_imag_idx] = q_rot_imag;
    k_rot[k_real_idx] = k_rot_real;
    k_rot[k_imag_idx] = k_rot_imag;
}

std::tuple<torch::Tensor, torch::Tensor> rope(torch::Tensor q, torch::Tensor k, torch::Tensor cos,
                                              torch::Tensor sin) {
    TORCH_CHECK(q.device().is_cuda(), "q must be on CUDA");
    TORCH_CHECK(k.device().is_cuda(), "k must be on CUDA");
    TORCH_CHECK(cos.device().is_cuda(), "cos must be on CUDA");
    TORCH_CHECK(sin.device().is_cuda(), "sin must be on CUDA");
    TORCH_CHECK(q.dim() == 4, "q must be 4D tensor (batch_size, seq_len, n_heads, head_dim)");
    TORCH_CHECK(k.dim() == 4, "k must be 4D tensor (batch_size, seq_len, n_heads, head_dim)");
    TORCH_CHECK(cos.dim() == 2, "cos must be 2D tensor (seq_len, head_dim // 2)");
    TORCH_CHECK(sin.dim() == 2, "sin must be 2D tensor (seq_len, head_dim // 2)");
    TORCH_CHECK(q.size(0) == k.size(0), "batch_size must be the same");
    TORCH_CHECK(q.size(1) == k.size(1), "seq_len must be the same");
    TORCH_CHECK(q.size(2) == k.size(2), "n_heads must be the same");
    TORCH_CHECK(q.size(3) == k.size(3), "head_dim must be the same");
    TORCH_CHECK(q.size(3) % 2 == 0, "head_dim must be even for RoPE");
    TORCH_CHECK(cos.size(0) == q.size(1), "cos seq_len must match q seq_len");
    TORCH_CHECK(sin.size(0) == q.size(1), "sin seq_len must match q seq_len");
    TORCH_CHECK(cos.size(1) == q.size(3) / 2, "cos shape must be (seq_len, head_dim // 2)");
    TORCH_CHECK(sin.size(1) == q.size(3) / 2, "sin shape must be (seq_len, head_dim // 2)");

    auto q_rot = torch::empty_like(q);
    auto k_rot = torch::empty_like(k);

    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int n_heads = q.size(2);
    int head_dim = q.size(3);

    int threads_per_block = head_dim / 2;
    dim3 grid(batch_size, seq_len, n_heads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "rope", [&] {
            const scalar_t* q_data = q.data_ptr<scalar_t>();
            const scalar_t* k_data = k.data_ptr<scalar_t>();
            const scalar_t* cos_data = cos.data_ptr<scalar_t>();
            const scalar_t* sin_data = sin.data_ptr<scalar_t>();
            scalar_t* q_rot_data = q_rot.data_ptr<scalar_t>();
            scalar_t* k_rot_data = k_rot.data_ptr<scalar_t>();
            rope_kernel<scalar_t><<<grid, threads_per_block>>>(
                q_data, k_data, cos_data, sin_data, q_rot_data, k_rot_data, q.stride(0),
                q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2),
                k.stride(3), cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1));
        });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return std::make_tuple(q_rot, k_rot);
}
