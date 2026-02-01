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


def matrix_transpose(input: torch.Tensor) -> torch.Tensor:
    """
    Transpose a matrix using CUDA kernel.

    Args:
        input: Input tensor (must be on CUDA)

    Returns:
        Transposed tensor
    """
    possible_dim = 2
    if not input.is_cuda:
        raise CUDAExtensionError("Input tensor must be on CUDA device")
    if input.dim() != possible_dim:
        raise CUDAExtensionError(f"Input tensor must be a {possible_dim}D tensor")
    if not input.is_contiguous():
        input = input.contiguous()
    return cuda_ops.matrix_transpose(input)


def softmax(input: torch.Tensor) -> torch.Tensor:
    """
    Apply softmax to a tensor using CUDA kernel.

    Args:
        input: Input tensor (must be on CUDA)

    Returns:
        Softmax tensor
    """
    possible_dim = 2
    if not input.is_cuda:
        raise CUDAExtensionError("Input tensor must be on CUDA device")
    if input.dim() != possible_dim:
        raise CUDAExtensionError(f"Input tensor must be a {possible_dim}D tensor")
    if not input.is_contiguous():
        input = input.contiguous()
    if input.dtype != torch.float32:
        raise CUDAExtensionError("Input tensor must be float32")
    return cuda_ops.softmax(input)


def silu(input: torch.Tensor) -> torch.Tensor:
    """
    Apply silu to a tensor using CUDA kernel.

    Args:
        input: Input tensor (must be on CUDA)
    """
    if not input.is_cuda:
        raise CUDAExtensionError("Input tensor must be on CUDA device")
    if input.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise CUDAExtensionError("Input tensor must be float16, float32, or bfloat16")
    if not input.is_contiguous():
        input = input.contiguous()
    return cuda_ops.silu(input)


def rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors using CUDA kernel.

    Args:
        q: Query tensor (must be on CUDA)
        k: Key tensor (must be on CUDA)
        cos: Cosine values (must be on CUDA)
        sin: Sine values (must be on CUDA)
    """
    if not q.is_cuda or not k.is_cuda or not cos.is_cuda or not sin.is_cuda:
        raise CUDAExtensionError("All tensors must be on CUDA device")
    if q.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise CUDAExtensionError("All tensors must be float16, float32, or bfloat16")
    if q.shape != k.shape:
        raise CUDAExtensionError("Query and key tensors must have the same shape")
    if cos.shape != sin.shape:
        raise CUDAExtensionError("Cosine and sine tensors must have the same shape")
    if q.shape[-1] % 2 != 0:
        raise CUDAExtensionError("head_dim must be even for RoPE")
    if cos.shape[-1] != q.shape[-1] // 2:
        raise CUDAExtensionError("cos shape must be (seq_len, head_dim // 2)")
    if sin.shape[-1] != q.shape[-1] // 2:
        raise CUDAExtensionError("sin shape must be (seq_len, head_dim // 2)")
    if q.shape[0] != k.shape[0]:
        raise CUDAExtensionError("Batch size must be the same")
    if q.shape[1] != k.shape[1]:
        raise CUDAExtensionError("Sequence length must be the same")
    if q.shape[2] != k.shape[2]:
        raise CUDAExtensionError("Number of heads must be the same")
    if q.shape[3] != k.shape[3]:
        raise CUDAExtensionError("Head dimension must be the same")
    return cuda_ops.rope(q, k, cos, sin)
