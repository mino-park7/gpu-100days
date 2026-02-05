import time

import pytest
import torch


def ensure_cuda_device():
    """
    Ensure CUDA device is available and return it.
    Skips the test if CUDA is not available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA is not available. This test requires CUDA.")
    return device


def compare_kernel_with_pytorch(kernel_func, pytorch_func, *args, rtol=1e-5):
    """
    Compare results from kernel function and PyTorch function.

    Args:
        kernel_func: The kernel function to test
        pytorch_func: The PyTorch reference function
        *args: Arguments to pass to both functions
        rtol: Relative tolerance for comparison (default: 1e-5)

    Raises:
        AssertionError: If results do not match within tolerance
    """
    # Run kernel function
    kernel_result = kernel_func(*args)

    # Run PyTorch reference function
    pytorch_result = pytorch_func(*args)

    # Compare results
    assert torch.allclose(kernel_result, pytorch_result, rtol=rtol), (
        f"Results do not match. Kernel shape: {kernel_result.shape}, PyTorch shape: {pytorch_result.shape}"
    )


def _get_tensor_numel(*args):
    """
    Get the total number of elements from tensor arguments.
    """
    numel = 0
    for arg in args:
        if isinstance(arg, torch.Tensor):
            numel = max(numel, arg.numel())
    return numel


def _get_iterations(numel, iterations=None):
    """
    Determine number of iterations based on tensor size.

    Args:
        numel: Number of elements in the tensor
        iterations: If provided, use this value. Otherwise, auto-determine.

    Returns:
        Number of iterations to use for benchmarking
    """
    if iterations is not None:
        return iterations

    # Adaptive iterations based on size
    mid_size = 10_000
    large_size = 1_000_000
    if numel < mid_size:
        return 1000
    elif numel < large_size:
        return 100
    else:
        return 50


def benchmark_kernel_vs_pytorch(kernel_func, pytorch_func, *args, warmup=10, iterations=None):
    """
    Benchmark kernel function vs PyTorch function and print results.
    This function only prints results and does not cause test failures.

    Args:
        kernel_func: The kernel function to benchmark
        pytorch_func: The PyTorch reference function
        *args: Arguments to pass to both functions
        warmup: Number of warmup iterations (default: 10)
        iterations: Number of benchmark iterations. If None, auto-determined based on input size.
    """
    numel = _get_tensor_numel(*args)
    iterations = _get_iterations(numel, iterations)

    # Warmup
    for _ in range(warmup):
        _ = kernel_func(*args)
        _ = pytorch_func(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark kernel
    start = time.perf_counter()
    for _ in range(iterations):
        _ = kernel_func(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    kernel_time = (time.perf_counter() - start) / iterations

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(iterations):
        _ = pytorch_func(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / iterations

    # Calculate speedup
    speedup = pytorch_time / kernel_time if kernel_time > 0 else float("inf")

    # Print results
    print(f"\n  Benchmark results ({iterations} iterations):")
    print(f"    Kernel:   {kernel_time * 1000:.4f} ms")
    print(f"    PyTorch:  {pytorch_time * 1000:.4f} ms")
    print(f"    Speedup:  {speedup:.2f}x")
    if speedup > 1.0:
        print(f"    Status:   Kernel is {speedup:.2f}x faster")
    elif speedup < 1.0:
        print(f"    Status:   PyTorch is {1 / speedup:.2f}x faster")
    else:
        print("    Status:   Similar performance")
