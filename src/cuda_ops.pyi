"""Type stubs for cuda_ops C++ extension module."""

import torch

def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors using CUDA kernel.

    Args:
        a: First input tensor (must be on CUDA, float32)
        b: Second input tensor (must be on CUDA, float32)

    Returns:
        Sum of a and b
    """
    ...

def matrix_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two matrices using CUDA kernel.

    Args:
        a: First input tensor (must be on CUDA, float32)
        b: Second input tensor (must be on CUDA, float32)

    Returns:
        Sum of a and b
    """
    ...

def matrix_sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Subtract two matrices using CUDA kernel.

    Args:
        a: First input tensor (must be on CUDA, float32)
        b: Second input tensor (must be on CUDA, float32)

    Returns:
        Difference of a and b
    """
    ...

def gray_scale(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image to grayscale using CUDA kernel.

    Args:
        image: Input image tensor (must be on CUDA, uint8)

    Returns:
        Grayscale image tensor
    """
    ...

def seeded_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """
    Apply seeded dropout to a tensor using CUDA kernel.

    Args:
        x: Input tensor (must be on CUDA)
        p: Dropout probability
        seed: Random seed
    ...

    Returns:
        Dropout tensor
    """

def matrix_transpose(input: torch.Tensor) -> torch.Tensor:
    """
    Transpose a matrix using CUDA kernel.

    Args:
        input: Input tensor (must be on CUDA)

    Returns:
        Transposed tensor
    """
    ...

def softmax(input: torch.Tensor) -> torch.Tensor:
    """
    Apply softmax to a tensor using CUDA kernel.

    Args:
        input: Input tensor (must be on CUDA)

    Returns:
        Softmax tensor
    """
    ...

def silu(input: torch.Tensor) -> torch.Tensor:
    """
    Apply silu to a tensor using CUDA kernel.

    Args:
        input: Input tensor (must be on CUDA)

    Returns:
        Silu tensor
    """
    ...
