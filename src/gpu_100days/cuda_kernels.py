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


def gray_scale(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image to grayscale using CUDA kernel.

    Args:
        image: Input image tensor (must be on CUDA, uint8)

    Returns:
        Grayscale image tensor
    """
    rgb_channels = 3
    if not image.is_cuda:
        raise CUDAExtensionError("Image must be on CUDA device")
    if image.dtype != torch.uint8:
        raise CUDAExtensionError("Image must be uint8")
    if image.shape[0] != rgb_channels:
        raise CUDAExtensionError("Image must have 3 channels")

    if not image.is_contiguous():
        image = image.contiguous()

    return cuda_ops.gray_scale(image).unsqueeze(0)


def seeded_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """
    Apply seeded dropout to a tensor using CUDA kernel.

    Args:
        x: Input tensor (must be on CUDA)
        p: Dropout probability
        seed: Random seed
    """
    if not x.is_cuda:
        raise CUDAExtensionError("Input tensor must be on CUDA device")

    if x.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise CUDAExtensionError("Input tensor must be float32, float16, or bfloat16")

    if not x.is_contiguous():
        x = x.contiguous()

    return cuda_ops.seeded_dropout(x, p, seed)
