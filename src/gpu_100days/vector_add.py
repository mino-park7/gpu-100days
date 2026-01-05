"""PyTorch CUDA extension wrapper for vector addition."""

import sys

import torch

try:
    import cuda_ops
except ImportError as err:
    raise ImportError(
        "cuda_ops extension not found. Please build the extension first by running:\n"
        "  pip install -e ."
    ) from err


class CUDAExtensionError(ValueError):
    """Error raised when CUDA extension requirements are not met."""

    pass


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors using CUDA kernel.

    Args:
        a: First input tensor (must be on CUDA, float32)
        b: Second input tensor (must be on CUDA, float32)

    Returns:
        Sum of a and b

    Example:
        >>> a = torch.randn(1000, device='cuda')
        >>> b = torch.randn(1000, device='cuda')
        >>> c = vector_add(a, b)
        >>> torch.allclose(c, a + b)
        True
    """
    if not a.is_cuda or not b.is_cuda:
        raise CUDAExtensionError("Both tensors must be on CUDA device")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise CUDAExtensionError("Both tensors must be float32")
    if a.shape != b.shape:
        raise CUDAExtensionError("Both tensors must have the same shape")

    return cuda_ops.vector_add(a, b)


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA is not available. This example requires CUDA.")
        sys.exit(1)

    print("Testing vector_add...")
    n = 10000
    a = torch.randn(n, device=device, dtype=torch.float32)
    b = torch.randn(n, device=device, dtype=torch.float32)

    # CUDA kernel version
    c_cuda = vector_add(a, b)

    # PyTorch version for comparison
    c_pytorch = a + b

    # Check if results match
    if torch.allclose(c_cuda, c_pytorch, rtol=1e-5):
        print("✓ Results match!")
        print(f"  CUDA result: {c_cuda[:5]}")
        print(f"  PyTorch result: {c_pytorch[:5]}")
    else:
        print("✗ Results do not match!")
        print(f"  CUDA result: {c_cuda[:5]}")
        print(f"  PyTorch result: {c_pytorch[:5]}")
        print(f"  Max difference: {(c_cuda - c_pytorch).abs().max().item()}")
