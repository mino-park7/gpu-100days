"""PyTorch CUDA extension wrapper for vector addition."""

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


def matrix_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two matrices using CUDA kernel.

    Args:
        a: First input tensor (must be on CUDA, float32)
        b: Second input tensor (must be on CUDA, float32)

    Returns:
        Sum of a and b
    """
    if not a.is_cuda or not b.is_cuda:
        raise CUDAExtensionError("Both tensors must be on CUDA device")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise CUDAExtensionError("Both tensors must be float32")
    if a.shape != b.shape:
        raise CUDAExtensionError("Both tensors must have the same shape")

    return cuda_ops.matrix_add(a, b)


def matrix_sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Subtract two matrices using CUDA kernel.

    Args:
        a: First input tensor (must be on CUDA, float32)
        b: Second input tensor (must be on CUDA, float32)

    Returns:
        Difference of a and b
    """
    if not a.is_cuda or not b.is_cuda:
        raise CUDAExtensionError("Both tensors must be on CUDA device")

    if not a.is_cuda or not b.is_cuda:
        raise CUDAExtensionError("Both tensors must be on CUDA device")
    if a.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise CUDAExtensionError("Both tensors must be float32, float16, or bfloat16")
    if a.shape != b.shape:
        raise CUDAExtensionError("Both tensors must have the same shape")

    return cuda_ops.matrix_sub(a, b)
