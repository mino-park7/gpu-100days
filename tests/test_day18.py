import pytest
import torch
import torch.testing
from conftest import benchmark_kernel_vs_pytorch

from gpu_100days import clip_triton

# Global tolerance settings
ATOL = 1e-5
RTOL = 1e-5

# Set random seed once globally
torch.manual_seed(42)


def clip_pytorch(input: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """
    Reference implementation using PyTorch to clip values.

    Args:
        input: Input tensor
        lo: Lower bound
        hi: Upper bound

    Returns:
        Clipped tensor
    """
    return torch.clamp(input, min=lo, max=hi)


@pytest.mark.parametrize(
    ("input_list", "lo", "hi", "expected"),
    [
        ([1.5, -2.0, 3.0, 4.5], 0.0, 3.5, [1.5, 0.0, 3.0, 3.5]),
        ([-1.0, 2.0, 5.0], -0.5, 2.5, [-0.5, 2.0, 2.5]),
    ],
)
def test_clip_examples(input_list, lo, hi, expected):
    """Test clipping with provided examples."""
    if clip_triton is None:
        pytest.skip("solve function not implemented yet")

    input_tensor = torch.tensor(input_list, device="cuda", dtype=torch.float32)
    result = clip_triton(input_tensor, lo, hi)

    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result, expected_tensor, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("n", [1, 10, 100, 1000, 10000, 100000])
def test_clip_various_sizes(n):
    """Test clipping with various array sizes."""
    if clip_triton is None:
        pytest.skip("solve function not implemented yet")

    lo = -1.0
    hi = 1.0
    input_tensor = torch.randn(n, device="cuda", dtype=torch.float32) * 10

    result = clip_triton(input_tensor, lo, hi)
    expected = clip_pytorch(input_tensor, lo, hi)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("n", [10000])
@pytest.mark.parametrize(
    ("lo", "hi"),
    [
        (-10.0, 10.0),
        (-1.0, 1.0),
        (0.0, 1.0),
        (-100.0, 100.0),
        (-0.5, 0.5),
    ],
)
def test_clip_various_ranges(n, lo, hi):
    """Test clipping with various value ranges."""
    if clip_triton is None:
        pytest.skip("solve function not implemented yet")

    input_tensor = torch.randn(n, device="cuda", dtype=torch.float32) * 100

    result = clip_triton(input_tensor, lo, hi)
    expected = clip_pytorch(input_tensor, lo, hi)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


def test_clip_large_array():
    """Test clipping with large array (near constraint limit)."""
    if clip_triton is None:
        pytest.skip("solve function not implemented yet")

    n = 100000  # Maximum constraint size
    lo = -50.0
    hi = 50.0
    input_tensor = torch.randn(n, device="cuda", dtype=torch.float32) * 1000

    result = clip_triton(input_tensor, lo, hi)
    expected = clip_pytorch(input_tensor, lo, hi)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


def test_clip_against_pytorch():
    """Test clipping implementation against PyTorch reference."""
    if clip_triton is None:
        pytest.skip("solve function not implemented yet")

    n = 10000
    lo = -2.5
    hi = 2.5
    input_tensor = torch.randn(n, device="cuda", dtype=torch.float32) * 10

    result = clip_triton(input_tensor, lo, hi)
    expected = clip_pytorch(input_tensor, lo, hi)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Benchmark
    benchmark_kernel_vs_pytorch(
        lambda inp, low, high: clip_triton(inp, low, high),
        lambda inp, low, high: clip_pytorch(inp, low, high),
        input_tensor,
        lo,
        hi,
    )


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_clip_benchmark(n):
    """Benchmark clipping with various sizes."""
    if clip_triton is None:
        pytest.skip("solve function not implemented yet")

    lo = -1.0
    hi = 1.0
    input_tensor = torch.randn(n, device="cuda", dtype=torch.float32) * 10

    result = clip_triton(input_tensor, lo, hi)
    expected = clip_pytorch(input_tensor, lo, hi)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    benchmark_kernel_vs_pytorch(
        lambda inp, low, high: clip_triton(inp, low, high),
        lambda inp, low, high: clip_pytorch(inp, low, high),
        input_tensor,
        lo,
        hi,
    )


def test_clip_edge_cases():
    """Test edge cases for clipping."""
    if clip_triton is None:
        pytest.skip("solve function not implemented yet")

    # Test with single element
    input_tensor = torch.tensor([5.0], device="cuda", dtype=torch.float32)
    result = clip_triton(input_tensor, 0.0, 1.0)
    expected = clip_pytorch(input_tensor, 0.0, 1.0)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Test with all values within range (no clipping needed)
    input_tensor = torch.tensor([0.5, 0.3, 0.7], device="cuda", dtype=torch.float32)
    result = clip_triton(input_tensor, 0.0, 1.0)
    expected = clip_pytorch(input_tensor, 0.0, 1.0)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Test with all values below lo
    input_tensor = torch.tensor([-10.0, -20.0, -30.0], device="cuda", dtype=torch.float32)
    result = clip_triton(input_tensor, -5.0, 5.0)
    expected = clip_pytorch(input_tensor, -5.0, 5.0)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Test with all values above hi
    input_tensor = torch.tensor([10.0, 20.0, 30.0], device="cuda", dtype=torch.float32)
    result = clip_triton(input_tensor, -5.0, 5.0)
    expected = clip_pytorch(input_tensor, -5.0, 5.0)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Test with values at constraint boundaries
    input_tensor = torch.tensor([-1e6, 1e6], device="cuda", dtype=torch.float32)
    result = clip_triton(input_tensor, -100.0, 100.0)
    expected = clip_pytorch(input_tensor, -100.0, 100.0)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Test with lo == hi (all values should become lo/hi)
    input_tensor = torch.tensor([-10.0, 0.0, 10.0], device="cuda", dtype=torch.float32)
    result = clip_triton(input_tensor, 5.0, 5.0)
    expected = clip_pytorch(input_tensor, 5.0, 5.0)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Test with negative range
    input_tensor = torch.tensor([-5.0, 0.0, 5.0], device="cuda", dtype=torch.float32)
    result = clip_triton(input_tensor, -2.0, -1.0)
    expected = clip_pytorch(input_tensor, -2.0, -1.0)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


def test_clip_constraint_boundaries():
    """Test clipping with values at constraint boundaries."""
    if clip_triton is None:
        pytest.skip("solve function not implemented yet")

    # Test with minimum size (N = 1)
    input_tensor = torch.tensor([1e6], device="cuda", dtype=torch.float32)
    result = clip_triton(input_tensor, -1e6, 1e6)
    expected = clip_pytorch(input_tensor, -1e6, 1e6)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Test with maximum size (N = 100000)
    n = 100000
    input_tensor = torch.randn(n, device="cuda", dtype=torch.float32) * 1e6
    result = clip_triton(input_tensor, -1e6, 1e6)
    expected = clip_pytorch(input_tensor, -1e6, 1e6)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
