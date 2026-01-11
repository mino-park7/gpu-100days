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
