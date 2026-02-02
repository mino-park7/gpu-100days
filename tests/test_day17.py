import pytest
import torch
import torch.testing
from conftest import benchmark_kernel_vs_pytorch

from gpu_100days import mse_triton

# Global tolerance settings
ATOL = 0.0
RTOL = 1e-2

# Set random seed once globally
torch.manual_seed(42)


def mse_pytorch(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using PyTorch to calculate Mean Squared Error.

    Args:
        predictions: Predicted values tensor
        targets: Target values tensor

    Returns:
        Mean Squared Error (scalar)
    """
    return ((predictions - targets) ** 2).mean()


@pytest.mark.parametrize(
    ("predictions", "targets", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5], 0.25),
        ([10.0, 20.0, 30.0], [12.0, 18.0, 33.0], 5.666666666666667),
    ],
)
def test_mse_examples(predictions, targets, expected):
    """Test MSE with provided examples."""
    if mse_triton is None:
        pytest.skip("solve function not implemented yet")

    predictions_tensor = torch.tensor(predictions, device="cuda", dtype=torch.float32)
    targets_tensor = torch.tensor(targets, device="cuda", dtype=torch.float32)

    result = mse_triton(predictions_tensor, targets_tensor)

    # Result can be either float or tensor
    if isinstance(result, torch.Tensor):
        result = result.item()

    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, expected_tensor, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("n", [1, 10, 100, 1000, 10000, 100000, 1000000])
def test_mse_various_sizes(n):
    """Test MSE with various array sizes."""
    if mse_triton is None:
        pytest.skip("solve function not implemented yet")

    predictions = torch.randn(n, device="cuda", dtype=torch.float32) * 1000
    targets = torch.randn(n, device="cuda", dtype=torch.float32) * 1000

    result = mse_triton(predictions, targets)

    expected = mse_pytorch(predictions, targets)

    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("n", [10000])
@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 100.0, 1000.0])
def test_mse_various_scales(n, scale):
    """Test MSE with various value scales."""
    if mse_triton is None:
        pytest.skip("solve function not implemented yet")

    predictions = torch.randn(n, device="cuda", dtype=torch.float32) * scale
    targets = torch.randn(n, device="cuda", dtype=torch.float32) * scale

    result = mse_triton(predictions, targets)
    if isinstance(result, torch.Tensor):
        result = result.item()

    expected = mse_pytorch(predictions, targets).item()

    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, expected_tensor, atol=ATOL, rtol=RTOL)


def test_mse_large_array():
    """Test MSE with large array (near constraint limit)."""
    if mse_triton is None:
        pytest.skip("solve function not implemented yet")

    n = 10_000_000  # Large but manageable size
    predictions = torch.randn(n, device="cuda", dtype=torch.float32) * 1000
    targets = torch.randn(n, device="cuda", dtype=torch.float32) * 1000

    result = mse_triton(predictions, targets)
    if isinstance(result, torch.Tensor):
        result = result.item()

    expected = mse_pytorch(predictions, targets).item()

    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, expected_tensor, atol=1e-4, rtol=RTOL)


def test_mse_against_pytorch():
    """Test MSE implementation against PyTorch reference."""
    if mse_triton is None:
        pytest.skip("solve function not implemented yet")

    n = 10000
    predictions = torch.randn(n, device="cuda", dtype=torch.float32) * 1000
    targets = torch.randn(n, device="cuda", dtype=torch.float32) * 1000

    result = mse_triton(predictions, targets)
    if isinstance(result, torch.Tensor):
        result = result.item()

    expected = mse_pytorch(predictions, targets).item()

    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, expected_tensor, atol=ATOL, rtol=RTOL)

    # Benchmark
    benchmark_kernel_vs_pytorch(
        lambda pred, targ: mse_triton(pred, targ),
        lambda pred, targ: mse_pytorch(pred, targ),
        predictions,
        targets,
    )


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_mse_benchmark(n):
    """Benchmark MSE with various sizes."""
    if mse_triton is None:
        pytest.skip("solve function not implemented yet")

    predictions = torch.randn(n, device="cuda", dtype=torch.float32) * 1000
    targets = torch.randn(n, device="cuda", dtype=torch.float32) * 1000

    result = mse_triton(predictions, targets)
    if isinstance(result, torch.Tensor):
        result = result.item()

    expected = mse_pytorch(predictions, targets).item()
    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, expected_tensor, atol=ATOL, rtol=RTOL)

    benchmark_kernel_vs_pytorch(
        lambda pred, targ: mse_triton(pred, targ),
        lambda pred, targ: mse_pytorch(pred, targ),
        predictions,
        targets,
    )


def test_mse_edge_cases():
    """Test edge cases for MSE."""
    if mse_triton is None:
        pytest.skip("solve function not implemented yet")

    # Test with single element
    predictions = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    targets = torch.tensor([2.0], device="cuda", dtype=torch.float32)
    result = mse_triton(predictions, targets)
    if isinstance(result, torch.Tensor):
        result = result.item()
    expected = mse_pytorch(predictions, targets).item()
    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, expected_tensor, atol=ATOL, rtol=RTOL)

    # Test with identical values (MSE should be 0)
    predictions = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    targets = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    result = mse_triton(predictions, targets)
    if isinstance(result, torch.Tensor):
        result = result.item()
    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    zero_tensor = torch.tensor(0.0, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, zero_tensor, atol=ATOL, rtol=RTOL)

    # Test with large differences
    predictions = torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32)
    targets = torch.tensor([100.0, 200.0, 300.0], device="cuda", dtype=torch.float32)
    result = mse_triton(predictions, targets)
    if isinstance(result, torch.Tensor):
        result = result.item()
    expected = mse_pytorch(predictions, targets).item()
    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, expected_tensor, atol=ATOL, rtol=RTOL)

    # Test with negative values
    predictions = torch.tensor([-10.0, -20.0, -30.0], device="cuda", dtype=torch.float32)
    targets = torch.tensor([-12.0, -18.0, -33.0], device="cuda", dtype=torch.float32)
    result = mse_triton(predictions, targets)
    if isinstance(result, torch.Tensor):
        result = result.item()
    expected = mse_pytorch(predictions, targets).item()
    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, expected_tensor, atol=ATOL, rtol=RTOL)

    # Test with values at constraint boundaries
    predictions = torch.tensor([1000.0, -1000.0], device="cuda", dtype=torch.float32)
    targets = torch.tensor([-1000.0, 1000.0], device="cuda", dtype=torch.float32)
    result = mse_triton(predictions, targets)
    if isinstance(result, torch.Tensor):
        result = result.item()
    expected = mse_pytorch(predictions, targets).item()
    result_tensor = torch.tensor(result, device="cuda", dtype=torch.float32)
    expected_tensor = torch.tensor(expected, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(result_tensor, expected_tensor, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
