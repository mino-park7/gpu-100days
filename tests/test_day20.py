import pytest
import torch
import torch.testing
from conftest import benchmark_kernel_vs_pytorch

from gpu_100days import int8_quantized_matmul_triton

# Global tolerance settings
ATOL = 1e-0
RTOL = 1e-0

# Set random seed once globally
torch.manual_seed(42)


def quantized_matmul_reference(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
) -> torch.Tensor:
    """
    Reference implementation following the exact formula:
    C = round((scale_A * scale_B / scale_C) * (A - z_A) @ (B - z_B)) + z_C

    Args:
        A: Input matrix A of shape (M, K) as int8
        B: Input matrix B of shape (K, N) as int8
        scale_A: Quantization scale for A
        scale_B: Quantization scale for B
        scale_C: Quantization scale for output C
        zero_point_A: Zero point for A
        zero_point_B: Zero point for B
        zero_point_C: Zero point for output C

    Returns:
        Output matrix C of shape (M, N) as int8
    """
    # Subtract zero points and convert to int32 for accumulation
    A_sub = A.to(torch.int32) - zero_point_A
    B_sub = B.to(torch.int32) - zero_point_B

    # Matrix multiplication: PyTorch doesn't support int32 matmul on CUDA,
    # so we convert to float32 for the operation, then convert back to int32
    # to simulate int32 accumulation
    A_sub_float = A_sub.to(torch.float32)
    B_sub_float = B_sub.to(torch.float32)
    C_float_accum = torch.matmul(A_sub_float, B_sub_float)
    # Round to nearest integer to simulate int32 accumulation
    C_int32 = torch.round(C_float_accum).to(torch.int32)

    # Scale and quantize
    scale = scale_A * scale_B / scale_C
    C_float = C_int32.to(torch.float32) * scale
    C_quant = torch.round(C_float) + zero_point_C
    C_quant = torch.clamp(C_quant, -128, 127).to(torch.int8)

    return C_quant


@pytest.mark.parametrize(
    ("M", "N", "K"),
    [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1, 100, 50),
        (100, 1, 50),
        (50, 100, 1),
        (128, 256, 512),
        (512, 256, 128),
    ],
)
def test_quantized_matmul_various_sizes(M, N, K):
    """Test quantized matrix multiplication with various sizes."""
    if int8_quantized_matmul_triton is None:
        pytest.skip("int8_quantized_matmul_triton function not implemented yet")

    # Generate random int8 matrices
    A = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)

    # Random quantization parameters
    scale_A = torch.rand(1).item() * 0.1 + 0.01
    scale_B = torch.rand(1).item() * 0.1 + 0.01
    scale_C = torch.rand(1).item() * 0.1 + 0.01
    zero_point_A: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_B: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_C: int = int(torch.randint(-128, 128, (1,)).item())

    # Call int8_quantized_matmul_triton function
    result = int8_quantized_matmul_triton(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    # Verify shape and dtype
    assert result.shape == (M, N), f"Expected shape ({M}, {N}), got {result.shape}"
    assert result.dtype == torch.int8, f"Expected dtype int8, got {result.dtype}"

    # Verify values are in range
    assert result.min() >= -128 and result.max() <= 127, "Result values out of int8 range"

    # Compare with reference implementation
    expected = quantized_matmul_reference(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    ("M", "N", "K"),
    [
        (1, 1, 1),
        (2, 2, 1),
        (1, 2, 2),
        (2, 1, 2),
        (10, 1, 5),
        (1, 10, 5),
        (100, 1, 50),
        (1, 100, 50),
    ],
)
def test_quantized_matmul_edge_cases(M, N, K):
    """Test quantized matrix multiplication with edge cases."""
    if int8_quantized_matmul_triton is None:
        pytest.skip("int8_quantized_matmul_triton function not implemented yet")

    # Generate random int8 matrices
    A = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)

    # Random quantization parameters
    scale_A = torch.rand(1).item() * 0.1 + 0.01
    scale_B = torch.rand(1).item() * 0.1 + 0.01
    scale_C = torch.rand(1).item() * 0.1 + 0.01
    zero_point_A: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_B: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_C: int = int(torch.randint(-128, 128, (1,)).item())

    # Call int8_quantized_matmul_triton function
    result = int8_quantized_matmul_triton(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    # Verify shape and dtype
    assert result.shape == (M, N), f"Expected shape ({M}, {N}), got {result.shape}"
    assert result.dtype == torch.int8, f"Expected dtype int8, got {result.dtype}"

    # Compare with reference implementation
    expected = quantized_matmul_reference(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


def test_quantized_matmul_against_reference():
    """Test quantized matrix multiplication implementation against reference."""
    if int8_quantized_matmul_triton is None:
        pytest.skip("int8_quantized_matmul_triton function not implemented yet")

    M, N, K = 64, 64, 64

    # Generate random int8 matrices
    A = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)

    # Random quantization parameters
    scale_A = torch.rand(1).item() * 0.1 + 0.01
    scale_B = torch.rand(1).item() * 0.1 + 0.01
    scale_C = torch.rand(1).item() * 0.1 + 0.01
    zero_point_A: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_B: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_C: int = int(torch.randint(-128, 128, (1,)).item())

    # Call int8_quantized_matmul_triton function
    result = int8_quantized_matmul_triton(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    # Compare with reference implementation
    expected = quantized_matmul_reference(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Benchmark
    benchmark_kernel_vs_pytorch(
        lambda: int8_quantized_matmul_triton(
            A,
            B,
            scale_A,
            scale_B,
            scale_C,
            zero_point_A,
            zero_point_B,
            zero_point_C,
        ),
        lambda: quantized_matmul_reference(
            A,
            B,
            scale_A,
            scale_B,
            scale_C,
            zero_point_A,
            zero_point_B,
            zero_point_C,
        ),
    )


def test_quantized_matmul_constraint_boundaries():
    """Test quantized matrix multiplication with values at constraint boundaries."""
    if int8_quantized_matmul_triton is None:
        pytest.skip("int8_quantized_matmul_triton function not implemented yet")

    # Test with minimum sizes
    A = torch.randint(-128, 128, (1, 1), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 128, (1, 1), device="cuda", dtype=torch.int8)
    result = int8_quantized_matmul_triton(A, B, 0.01, 0.01, 0.01, 0, 0, 0)
    assert result.shape == (1, 1)
    assert result.dtype == torch.int8

    # Test with maximum sizes
    M, N, K = 4096, 4096, 4096
    A = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)
    result = int8_quantized_matmul_triton(A, B, 0.01, 0.01, 0.01, 0, 0, 0)
    assert result.shape == (M, N)
    assert result.dtype == torch.int8

    # Test with zero points at boundaries
    A = torch.randint(-128, 128, (10, 10), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 128, (10, 10), device="cuda", dtype=torch.int8)
    result = int8_quantized_matmul_triton(A, B, 0.01, 0.01, 0.01, -128, -128, -128)
    assert result.shape == (10, 10)
    assert result.dtype == torch.int8

    result = int8_quantized_matmul_triton(A, B, 0.01, 0.01, 0.01, 127, 127, 127)
    assert result.shape == (10, 10)
    assert result.dtype == torch.int8


@pytest.mark.parametrize(
    ("M", "N", "K"),
    [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ],
)
def test_quantized_matmul_benchmark(M, N, K):
    """Benchmark quantized matrix multiplication with various sizes."""
    if int8_quantized_matmul_triton is None:
        pytest.skip("int8_quantized_matmul_triton function not implemented yet")

    # Generate random int8 matrices
    A = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)

    # Random quantization parameters
    scale_A = torch.rand(1).item() * 0.1 + 0.01
    scale_B = torch.rand(1).item() * 0.1 + 0.01
    scale_C = torch.rand(1).item() * 0.1 + 0.01
    zero_point_A: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_B: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_C: int = int(torch.randint(-128, 128, (1,)).item())

    # Call int8_quantized_matmul_triton function
    result = int8_quantized_matmul_triton(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    # Compare with reference implementation
    expected = quantized_matmul_reference(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    benchmark_kernel_vs_pytorch(
        lambda: int8_quantized_matmul_triton(
            A,
            B,
            scale_A,
            scale_B,
            scale_C,
            zero_point_A,
            zero_point_B,
            zero_point_C,
        ),
        lambda: quantized_matmul_reference(
            A,
            B,
            scale_A,
            scale_B,
            scale_C,
            zero_point_A,
            zero_point_B,
            zero_point_C,
        ),
    )


def test_quantized_matmul_performance_size():
    """Test quantized matrix multiplication with performance measurement size."""
    if int8_quantized_matmul_triton is None:
        pytest.skip("int8_quantized_matmul_triton function not implemented yet")

    # Performance measurement size: K = 2,048, M = 8,192, N = 4,096
    M, N, K = 8192, 4096, 2048

    # Generate random int8 matrices
    A = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)

    # Random quantization parameters
    scale_A = torch.rand(1).item() * 0.1 + 0.01
    scale_B = torch.rand(1).item() * 0.1 + 0.01
    scale_C = torch.rand(1).item() * 0.1 + 0.01
    zero_point_A: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_B: int = int(torch.randint(-128, 128, (1,)).item())
    zero_point_C: int = int(torch.randint(-128, 128, (1,)).item())

    # Call int8_quantized_matmul_triton function
    result = int8_quantized_matmul_triton(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    # Verify shape and dtype
    assert result.shape == (M, N), f"Expected shape ({M}, {N}), got {result.shape}"
    assert result.dtype == torch.int8, f"Expected dtype int8, got {result.dtype}"

    # Compare with reference implementation
    expected = quantized_matmul_reference(
        A,
        B,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
    )

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    # Benchmark
    benchmark_kernel_vs_pytorch(
        lambda: int8_quantized_matmul_triton(
            A,
            B,
            scale_A,
            scale_B,
            scale_C,
            zero_point_A,
            zero_point_B,
            zero_point_C,
        ),
        lambda: quantized_matmul_reference(
            A,
            B,
            scale_A,
            scale_B,
            scale_C,
            zero_point_A,
            zero_point_B,
            zero_point_C,
        ),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
