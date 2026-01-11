#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void gray_scale_kernel(const uint8_t* image, uint8_t* out, int height, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int index = row * width + col;
        float R = image[0 * width * height + index];
        float G = image[1 * width * height + index];
        float B = image[2 * width * height + index];
        float out_data = 0.299 * R + 0.587 * G + 0.114 * B;
        out[index] = static_cast<uint8_t>(out_data);
    }
}

torch::Tensor gray_scale(torch::Tensor image) {
    TORCH_CHECK(image.device().is_cuda(), "Image must be on CUDA");
    TORCH_CHECK(image.dtype() == torch::kUInt8, "Image must be uint8");
    TORCH_CHECK(image.dim() == 3, "Image must have 3 dimensions");

    auto shape = image.sizes();
    int channel = shape[0];
    int height = shape[1];
    int width = shape[2];

    auto device = image.device();
    auto dtype = image.dtype();
    auto tensor_options = torch::TensorOptions().device(device).dtype(dtype);
    auto out = torch::empty({height, width}, tensor_options);

    dim3 block(16, 16);
    dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y);
    gray_scale_kernel<<<grid, block>>>(image.data_ptr<uint8_t>(), out.data_ptr<uint8_t>(), height,
                                       width);

    return out;
}
