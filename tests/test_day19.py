import pytest
import torch
import torch.nn.functional as F
import torch.testing
from conftest import benchmark_kernel_vs_pytorch

from gpu_100days import convolution_2d_triton

# Global tolerance settings
ATOL = 1e-0
RTOL = 1e-0

# Set random seed once globally
torch.manual_seed(42)


def convolution_2d_pytorch(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using PyTorch for 2D convolution with valid padding.
    Note: PyTorch conv2d performs cross-correlation, which matches the problem definition.

    Args:
        input: Input tensor of shape (input_rows, input_cols)
        kernel: Kernel tensor of shape (kernel_rows, kernel_cols)

    Returns:
        Convolved output tensor of shape (output_rows, output_cols)
    """
    # Add batch and channel dimensions for PyTorch conv2d
    # input: (input_rows, input_cols) -> (1, 1, input_rows, input_cols)
    # kernel: (kernel_rows, kernel_cols) -> (1, 1, kernel_rows, kernel_cols)
    input_4d = input.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, Kh, Kw)

    # Apply convolution with padding=0 (valid padding)
    # PyTorch conv2d performs cross-correlation, which matches the problem definition
    output_4d = F.conv2d(input_4d, kernel_4d, padding=0)

    # Remove batch and channel dimensions
    return output_4d.squeeze(0).squeeze(0)  # (output_rows, output_cols)


@pytest.mark.parametrize(
    (
        "input_data",
        "kernel_data",
        "input_rows",
        "input_cols",
        "kernel_rows",
        "kernel_cols",
        "expected",
    ),
    [
        # Example 1: 3x3 input, 2x2 kernel
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [1.0, 0.0, 0.0, 1.0],
            3,
            3,
            2,
            2,
            [6.0, 8.0, 12.0, 14.0],
        ),
    ],
)
def test_convolution_2d_examples(
    input_data, kernel_data, input_rows, input_cols, kernel_rows, kernel_cols, expected
):
    """Test 2D convolution with provided examples."""
    if convolution_2d_triton is None:
        pytest.skip("convolution_2d_triton function not implemented yet")

    # Reshape 1D arrays to 2D matrices
    input_2d = torch.tensor(input_data, device="cuda", dtype=torch.float32).reshape(
        input_rows, input_cols
    )
    kernel_2d = torch.tensor(kernel_data, device="cuda", dtype=torch.float32).reshape(
        kernel_rows, kernel_cols
    )

    result = convolution_2d_triton(input_2d, kernel_2d)

    # Reshape expected to 2D
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    expected_2d = torch.tensor(expected, device="cuda", dtype=torch.float32).reshape(
        output_rows, output_cols
    )

    torch.testing.assert_close(result, expected_2d, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    ("input_rows", "input_cols", "kernel_rows", "kernel_cols"),
    [
        (3, 3, 2, 2),
        (4, 4, 1, 3),
        (5, 5, 3, 3),
        (10, 10, 3, 3),
        (32, 32, 5, 5),
        (100, 100, 7, 7),
        (256, 256, 3, 3),
    ],
)
def test_convolution_2d_various_sizes(input_rows, input_cols, kernel_rows, kernel_cols):
    """Test 2D convolution with various input and kernel sizes."""
    if convolution_2d_triton is None:
        pytest.skip("convolution_2d_triton function not implemented yet")

    input_tensor = torch.randn(input_rows, input_cols, device="cuda", dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_rows, kernel_cols, device="cuda", dtype=torch.float32)

    result = convolution_2d_triton(input_tensor, kernel_tensor)
    expected = convolution_2d_pytorch(input_tensor, kernel_tensor)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    ("input_rows", "input_cols", "kernel_rows", "kernel_cols"),
    [
        (1, 1, 1, 1),
        (2, 2, 1, 1),
        (2, 2, 2, 2),
        (10, 1, 3, 1),
        (1, 10, 1, 3),
        (31, 31, 31, 31),
    ],
)
def test_convolution_2d_edge_cases(input_rows, input_cols, kernel_rows, kernel_cols):
    """Test 2D convolution with edge cases."""
    if convolution_2d_triton is None:
        pytest.skip("convolution_2d_triton function not implemented yet")

    input_tensor = torch.randn(input_rows, input_cols, device="cuda", dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_rows, kernel_cols, device="cuda", dtype=torch.float32)

    result = convolution_2d_triton(input_tensor, kernel_tensor)
    expected = convolution_2d_pytorch(input_tensor, kernel_tensor)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


def test_convolution_2d_against_pytorch():
    """Test 2D convolution implementation against PyTorch reference."""
    if convolution_2d_triton is None:
        pytest.skip("convolution_2d_triton function not implemented yet")

    input_rows, input_cols = 64, 64
    kernel_rows, kernel_cols = 5, 5

    input_tensor = torch.randn(input_rows, input_cols, device="cuda", dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_rows, kernel_cols, device="cuda", dtype=torch.float32)

    result = convolution_2d_triton(input_tensor, kernel_tensor)
    expected = convolution_2d_pytorch(input_tensor, kernel_tensor)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Benchmark
    benchmark_kernel_vs_pytorch(
        lambda inp, ker: convolution_2d_triton(inp, ker),
        lambda inp, ker: convolution_2d_pytorch(inp, ker),
        input_tensor,
        kernel_tensor,
    )


@pytest.mark.parametrize(
    ("input_rows", "input_cols", "kernel_rows", "kernel_cols"),
    [
        (32, 32, 3, 3),
        (64, 64, 5, 5),
        (128, 128, 7, 7),
        (256, 256, 3, 3),
    ],
)
def test_convolution_2d_benchmark(input_rows, input_cols, kernel_rows, kernel_cols):
    """Benchmark 2D convolution with various sizes."""
    if convolution_2d_triton is None:
        pytest.skip("convolution_2d_triton function not implemented yet")

    input_tensor = torch.randn(input_rows, input_cols, device="cuda", dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_rows, kernel_cols, device="cuda", dtype=torch.float32)

    result = convolution_2d_triton(input_tensor, kernel_tensor)
    expected = convolution_2d_pytorch(input_tensor, kernel_tensor)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    benchmark_kernel_vs_pytorch(
        lambda inp, ker: convolution_2d_triton(inp, ker),
        lambda inp, ker: convolution_2d_pytorch(inp, ker),
        input_tensor,
        kernel_tensor,
    )


def test_convolution_2d_constraint_boundaries():
    """Test 2D convolution with values at constraint boundaries."""
    if convolution_2d_triton is None:
        pytest.skip("convolution_2d_triton function not implemented yet")

    # Test with minimum sizes
    input_tensor = torch.randn(1, 1, device="cuda", dtype=torch.float32)
    kernel_tensor = torch.randn(1, 1, device="cuda", dtype=torch.float32)
    result = convolution_2d_triton(input_tensor, kernel_tensor)
    expected = convolution_2d_pytorch(input_tensor, kernel_tensor)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Test with maximum input size (3072)
    input_tensor = torch.randn(3072, 3072, device="cuda", dtype=torch.float32)
    kernel_tensor = torch.randn(5, 5, device="cuda", dtype=torch.float32)
    result = convolution_2d_triton(input_tensor, kernel_tensor)
    expected = convolution_2d_pytorch(input_tensor, kernel_tensor)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Test with maximum kernel size (31)
    input_tensor = torch.randn(100, 100, device="cuda", dtype=torch.float32)
    kernel_tensor = torch.randn(31, 31, device="cuda", dtype=torch.float32)
    result = convolution_2d_triton(input_tensor, kernel_tensor)
    expected = convolution_2d_pytorch(input_tensor, kernel_tensor)
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    ("input_rows", "input_cols", "kernel_rows", "kernel_cols"),
    [
        (10, 20, 3, 5),
        (20, 10, 5, 3),
        (50, 100, 7, 11),
        (100, 50, 11, 7),
    ],
)
def test_convolution_2d_rectangular(input_rows, input_cols, kernel_rows, kernel_cols):
    """Test 2D convolution with rectangular inputs and kernels."""
    if convolution_2d_triton is None:
        pytest.skip("convolution_2d_triton function not implemented yet")

    input_tensor = torch.randn(input_rows, input_cols, device="cuda", dtype=torch.float32)
    kernel_tensor = torch.randn(kernel_rows, kernel_cols, device="cuda", dtype=torch.float32)

    result = convolution_2d_triton(input_tensor, kernel_tensor)
    expected = convolution_2d_pytorch(input_tensor, kernel_tensor)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
