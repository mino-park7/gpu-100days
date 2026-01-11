import pytest
import torch
from conftest import benchmark_kernel_vs_pytorch, compare_kernel_with_pytorch, ensure_cuda_device

from gpu_100days import matrix_sub, matrix_sub_triton

# Test cases: (m, n, description)
MATRIX_SUB_TEST_CASES = [
    # Small sizes
    (1, 1, "small_1x1_float32", torch.float32),
    (1, 10, "small_1x10_float32", torch.float32),
    (10, 1, "small_10x1_float32", torch.float32),
    (5, 5, "small_5x5_float32", torch.float32),
    # Square matrices
    (16, 16, "square_16x16_float32", torch.float32),
    (32, 32, "square_32x32_float32", torch.float32),
    (64, 64, "square_64x64_float32", torch.float32),
    (128, 128, "square_128x128_float32", torch.float32),
    # Rectangular matrices
    (10, 100, "rect_10x100_float32", torch.float32),
    (100, 10, "rect_100x10_float32", torch.float32),
    (50, 200, "rect_50x200_float32", torch.float32),
    (200, 50, "rect_200x50_float32", torch.float32),
    # Odd sizes
    (17, 17, "odd_17x17_float32", torch.float32),
    (33, 33, "odd_33x33_float32", torch.float32),
    (99, 99, "odd_99x99_float32", torch.float32),
    (101, 101, "odd_101x101_float32", torch.float32),
    # Large sizes
    (500, 500, "large_500x500_float32", torch.float32),
    (1000, 1000, "large_1000x1000_float32", torch.float32),
    # Asymmetric
    (1, 1000, "asymmetric_1x1000_float32", torch.float32),
    (1000, 1, "asymmetric_1000x1_float32", torch.float32),
    # Different dtypes
    (1, 1, "small_1x1_float16", torch.float16),
    (1, 10, "small_1x10_float16", torch.float16),
    (10, 1, "small_10x1_float16", torch.float16),
    (5, 5, "small_5x5_float16", torch.float16),
    # Square matrices
    (16, 16, "square_16x16_float16", torch.float16),
    (32, 32, "square_32x32_float16", torch.float16),
    (64, 64, "square_64x64_float16", torch.float16),
    (128, 128, "square_128x128_float16", torch.float16),
    # Rectangular matrices
    (10, 100, "rect_10x100_float16", torch.float16),
    (100, 10, "rect_100x10_float16", torch.float16),
    (50, 200, "rect_50x200_float16", torch.float16),
    (200, 50, "rect_200x50_float16", torch.float16),
    # Odd sizes
    (17, 17, "odd_17x17_float16", torch.float16),
    (33, 33, "odd_33x33_float16", torch.float16),
    (99, 99, "odd_99x99_float16", torch.float16),
    (101, 101, "odd_101x101_float16", torch.float16),
    # Large sizes
    (500, 500, "large_500x500_float16", torch.float16),
    (1000, 1000, "large_1000x1000_float16", torch.float16),
    # Asymmetric
    (1, 1000, "asymmetric_1x1000_float16", torch.float16),
    (1000, 1, "asymmetric_1000x1_float16", torch.float16),
    # Different dtypes
    (1, 1, "small_1x1_bfloat16", torch.bfloat16),
    (1, 10, "small_1x10_bfloat16", torch.bfloat16),
    (10, 1, "small_10x1_bfloat16", torch.bfloat16),
    (5, 5, "small_5x5_bfloat16", torch.bfloat16),
    # Square matrices
    (16, 16, "square_16x16_bfloat16", torch.bfloat16),
    (32, 32, "square_32x32_bfloat16", torch.bfloat16),
    (64, 64, "square_64x64_bfloat16", torch.bfloat16),
    (128, 128, "square_128x128_bfloat16", torch.bfloat16),
    # Rectangular matrices
    (10, 100, "rect_10x100_bfloat16", torch.bfloat16),
    (100, 10, "rect_100x10_bfloat16", torch.bfloat16),
    (50, 200, "rect_50x200_bfloat16", torch.bfloat16),
    (200, 50, "rect_200x50_bfloat16", torch.bfloat16),
    # Odd sizes
    (17, 17, "odd_17x17_bfloat16", torch.bfloat16),
    (33, 33, "odd_33x33_bfloat16", torch.bfloat16),
    (99, 99, "odd_99x99_bfloat16", torch.bfloat16),
    (101, 101, "odd_101x101_bfloat16", torch.bfloat16),
    # Large sizes
    (500, 500, "large_500x500_bfloat16", torch.bfloat16),
    (1000, 1000, "large_1000x1000_bfloat16", torch.bfloat16),
    # Asymmetric
    (1, 1000, "asymmetric_1x1000_bfloat16", torch.bfloat16),
    (1000, 1, "asymmetric_1000x1_bfloat16", torch.bfloat16),
]


@pytest.mark.parametrize("m,n,description,dtype", MATRIX_SUB_TEST_CASES)
def test_matrix_sub(m, n, description, dtype):
    device = ensure_cuda_device()

    print(f"Testing matrix_sub with shape ({m}, {n}) ({description}) ({dtype})...")
    a = torch.randn(m, n, device=device, dtype=dtype)
    b = torch.randn(m, n, device=device, dtype=dtype)

    # Compare kernel with PyTorch
    compare_kernel_with_pytorch(matrix_sub, lambda x, y: x - y, a, b, rtol=1e-5)

    # Benchmark kernel vs PyTorch
    benchmark_kernel_vs_pytorch(matrix_sub, lambda x, y: x - y, a, b)


@pytest.mark.parametrize("m,n,description,dtype", MATRIX_SUB_TEST_CASES)
def test_matrix_sub_triton(m, n, description, dtype):
    device = ensure_cuda_device()

    print(f"Testing matrix_sub_triton with shape ({m}, {n}) ({description}) ({dtype})...")
    a = torch.randn(m, n, device=device, dtype=dtype)
    b = torch.randn(m, n, device=device, dtype=dtype)

    # Compare kernel with PyTorch
    compare_kernel_with_pytorch(matrix_sub_triton, lambda x, y: x - y, a, b, rtol=1e-5)

    # Benchmark kernel vs PyTorch
    benchmark_kernel_vs_pytorch(matrix_sub_triton, lambda x, y: x - y, a, b)
