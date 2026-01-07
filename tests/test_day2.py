import pytest
import torch
from conftest import benchmark_kernel_vs_pytorch, compare_kernel_with_pytorch, ensure_cuda_device

from gpu_100days import matrix_add, matrix_add_triton

# Test cases: (m, n, description)
MATRIX_ADD_TEST_CASES = [
    # Small sizes
    (1, 1, "small_1x1"),
    (1, 10, "small_1x10"),
    (10, 1, "small_10x1"),
    (5, 5, "small_5x5"),
    # Square matrices
    (16, 16, "square_16x16"),
    (32, 32, "square_32x32"),
    (64, 64, "square_64x64"),
    (128, 128, "square_128x128"),
    # Rectangular matrices
    (10, 100, "rect_10x100"),
    (100, 10, "rect_100x10"),
    (50, 200, "rect_50x200"),
    (200, 50, "rect_200x50"),
    # Odd sizes
    (17, 17, "odd_17x17"),
    (33, 33, "odd_33x33"),
    (99, 99, "odd_99x99"),
    (101, 101, "odd_101x101"),
    # Large sizes
    (500, 500, "large_500x500"),
    (1000, 1000, "large_1000x1000"),
    # Asymmetric
    (1, 1000, "asymmetric_1x1000"),
    (1000, 1, "asymmetric_1000x1"),
]


@pytest.mark.parametrize("m,n,description", MATRIX_ADD_TEST_CASES)
def test_matrix_add(m, n, description):
    device = ensure_cuda_device()

    print(f"Testing matrix_add with shape ({m}, {n}) ({description})...")
    a = torch.randn(m, n, device=device, dtype=torch.float32)
    b = torch.randn(m, n, device=device, dtype=torch.float32)

    # Compare kernel with PyTorch
    compare_kernel_with_pytorch(matrix_add, lambda x, y: x + y, a, b, rtol=1e-5)

    # Benchmark kernel vs PyTorch
    benchmark_kernel_vs_pytorch(matrix_add, lambda x, y: x + y, a, b)


@pytest.mark.parametrize("m,n,description", MATRIX_ADD_TEST_CASES)
def test_matrix_add_triton(m, n, description):
    device = ensure_cuda_device()

    print(f"Testing matrix_add_triton with shape ({m}, {n}) ({description})...")
    a = torch.randn(m, n, device=device, dtype=torch.float32)
    b = torch.randn(m, n, device=device, dtype=torch.float32)

    # Compare kernel with PyTorch
    compare_kernel_with_pytorch(matrix_add_triton, lambda x, y: x + y, a, b, rtol=1e-5)

    # Benchmark kernel vs PyTorch
    benchmark_kernel_vs_pytorch(matrix_add_triton, lambda x, y: x + y, a, b)
