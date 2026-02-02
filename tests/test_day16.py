import pytest
import torch
from conftest import benchmark_kernel_vs_pytorch

from gpu_100days import count_equal_triton


def count_elements_pytorch(input: torch.Tensor, k: int) -> torch.Tensor:
    """
    Reference implementation using PyTorch to count elements equal to k.

    Args:
        input: Input tensor of 32-bit integers
        k: Integer value to count

    Returns:
        Number of elements equal to k
    """
    return (input == k).sum()


@pytest.mark.parametrize(
    ("input_list", "k", "expected"),
    [
        ([1, 2, 3, 4, 1], 1, 2),
        ([5, 10, 5, 2], 11, 0),
        ([1], 1, 1),
        ([1], 2, 0),
        ([1, 1, 1, 1], 1, 4),
        ([1, 2, 3, 4, 5], 3, 1),
    ],
)
def test_count_elements_examples(input_list, k, expected):
    """Test count elements with provided examples."""
    if count_equal_triton is None:
        pytest.skip("solve function not implemented yet")

    input_tensor = torch.tensor(input_list, device="cuda", dtype=torch.int32)
    result = count_equal_triton(input_tensor, k)

    # Result can be either int or tensor
    if isinstance(result, torch.Tensor):
        result = result.item()

    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize("n", [1, 10, 100, 1000, 10000, 100000])
def test_count_elements_various_sizes(n):
    """Test count elements with various array sizes."""
    if count_equal_triton is None:
        pytest.skip("solve function not implemented yet")

    # Create array with some known values
    input_list = [i % 10 for i in range(n)]
    k = 5  # Count occurrences of 5

    input_tensor = torch.tensor(input_list, device="cuda", dtype=torch.int32)
    result = count_equal_triton(input_tensor, k)

    # Result can be either int or tensor
    if isinstance(result, torch.Tensor):
        result = result.item()

    # Expected count: every 10th element starting from index 5
    expected = (n + 4) // 10  # Count of 5s in pattern

    assert result == expected, f"Expected {expected}, got {result} for n={n}"


@pytest.mark.parametrize("n", [10000])
@pytest.mark.parametrize("k", [1, 5, 10, 50, 100])
def test_count_elements_various_k(n, k):
    """Test count elements with various k values."""
    if count_equal_triton is None:
        pytest.skip("solve function not implemented yet")

    # Create array with values 1 to 100
    input_list = [(i % 100) + 1 for i in range(n)]
    input_tensor = torch.tensor(input_list, device="cuda", dtype=torch.int32)

    result = count_equal_triton(input_tensor, k)
    if isinstance(result, torch.Tensor):
        result = result.item()

    # Expected: k appears n/100 times (approximately, depending on n)
    expected = n // 100

    assert result == expected, f"Expected {expected}, got {result} for k={k}"


def test_count_elements_large_array():
    """Test count elements with large array (near constraint limit)."""
    if count_equal_triton is None:
        pytest.skip("solve function not implemented yet")

    n = 10_000_000  # Large but manageable size
    k = 42

    # Create array with mostly zeros and some k values
    input_list = [k if i % 1000 == 0 else 0 for i in range(n)]
    input_tensor = torch.tensor(input_list, device="cuda", dtype=torch.int32)

    result = count_equal_triton(input_tensor, k)
    if isinstance(result, torch.Tensor):
        result = result.item()

    expected = n // 1000  # k appears every 1000th element
    assert result == expected, f"Expected {expected}, got {result}"


def test_count_elements_against_pytorch():
    """Test count elements implementation against PyTorch reference."""
    if count_equal_triton is None:
        pytest.skip("solve function not implemented yet")

    n = 10000
    k = 7

    # Create random array with values in range [1, 100]
    torch.manual_seed(42)
    input_list = torch.randint(1, 101, (n,), device="cuda", dtype=torch.int32)

    result = count_equal_triton(input_list, k)
    if isinstance(result, torch.Tensor):
        result = result.item()

    expected = count_elements_pytorch(input_list, k)

    assert result == expected, f"Expected {expected}, got {result}"

    # Benchmark
    benchmark_kernel_vs_pytorch(
        lambda inp, k_val: count_equal_triton(inp, k_val),
        lambda inp, k_val: count_elements_pytorch(inp, k_val),
        input_list,
        k,
    )


@pytest.mark.parametrize("n", [100, 1000, 10000, 100000])
def test_count_elements_benchmark(n):
    """Benchmark count elements with various sizes."""
    if count_equal_triton is None:
        pytest.skip("solve function not implemented yet")

    k = 50
    input_list = torch.randint(1, 101, (n,), device="cuda", dtype=torch.int32)

    result = count_equal_triton(input_list, k)
    if isinstance(result, torch.Tensor):
        result = result.item()

    expected = count_elements_pytorch(input_list, k)
    assert result == expected

    benchmark_kernel_vs_pytorch(
        lambda inp, k_val: count_equal_triton(inp, k_val),
        lambda inp, k_val: count_elements_pytorch(inp, k_val),
        input_list,
        k,
    )


def test_count_elements_edge_cases():
    """Test edge cases for count elements."""
    if count_equal_triton is None:
        pytest.skip("solve function not implemented yet")

    # Test with single element
    input_tensor = torch.tensor([42], device="cuda", dtype=torch.int32)
    result = count_equal_triton(input_tensor, 42)
    if isinstance(result, torch.Tensor):
        result = result.item()
    assert result == 1

    result = count_equal_triton(input_tensor, 1)
    if isinstance(result, torch.Tensor):
        result = result.item()
    assert result == 0

    # Test with all same values
    input_tensor = torch.tensor([5] * 100, device="cuda", dtype=torch.int32)
    result = count_equal_triton(input_tensor, 5)
    if isinstance(result, torch.Tensor):
        result = result.item()
    assert result == 100

    # Test with no matches
    input_tensor = torch.tensor([1, 2, 3, 4, 5], device="cuda", dtype=torch.int32)
    result = count_equal_triton(input_tensor, 10)
    if isinstance(result, torch.Tensor):
        result = result.item()
    assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
