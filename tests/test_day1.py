import pytest
import torch
from conftest import benchmark_kernel_vs_pytorch, compare_kernel_with_pytorch, ensure_cuda_device

from gpu_100days import vector_add

# Test cases: (size, description)
VECTOR_ADD_TEST_CASES = [
    # Small sizes
    (1, "single_element"),
    (10, "small_10"),
    (100, "small_100"),
    # Medium sizes
    (1000, "medium_1000"),
    (10000, "medium_10000"),
    # Large sizes
    (100000, "large_100000"),
    (1000000, "large_1000000"),
    # Odd sizes
    (999, "odd_999"),
    (10001, "odd_10001"),
    # Powers of 2
    (256, "power2_256"),
    (1024, "power2_1024"),
    (4096, "power2_4096"),
]


@pytest.mark.parametrize("n,description", VECTOR_ADD_TEST_CASES)
def test_vector_add(n, description):
    device = ensure_cuda_device()

    print(f"Testing vector_add with size {n} ({description})...")
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)

    # Compare kernel with PyTorch
    compare_kernel_with_pytorch(vector_add, lambda x, y: x + y, a, b, rtol=1e-5)

    # Benchmark kernel vs PyTorch
    benchmark_kernel_vs_pytorch(vector_add, lambda x, y: x + y, a, b)
