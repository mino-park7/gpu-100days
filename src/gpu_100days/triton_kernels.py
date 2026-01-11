import torch
import triton
import triton.language as tl


class TritonExtensionError(ValueError):
    """Error raised when Triton extension requirements are not met."""

    pass


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 16}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def _triton_matrix_add_kernel(a_ptr, b_ptr, out, M: int, N: int, BLOCK_SIZE: tl.constexpr) -> None:
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    row_start = pid_x * BLOCK_SIZE
    col_start = pid_y * BLOCK_SIZE

    row_indices = row_start + tl.arange(0, BLOCK_SIZE)
    col_indices = col_start + tl.arange(0, BLOCK_SIZE)

    row_indices = row_indices[:, None]
    col_indices = col_indices[None, :]

    row_mask = row_indices < M
    col_mask = col_indices < N
    valid_mask = row_mask & col_mask

    flat_indices = row_indices * N + col_indices

    A = tl.load(a_ptr + flat_indices, mask=valid_mask, other=0.0)
    B = tl.load(b_ptr + flat_indices, mask=valid_mask, other=0.0)

    out_data = A + B

    tl.store(out + flat_indices, out_data, mask=valid_mask)


def matrix_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two matrices using Triton kernel.

    Args:
        a: First input tensor (must be on CUDA, float32)
        b: Second input tensor (must be on CUDA, float32)
    """
    if not a.is_cuda or not b.is_cuda:
        raise TritonExtensionError("Both tensors must be on CUDA device")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TritonExtensionError("Both tensors must be float32")

    M, N = a.shape
    out = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    _triton_matrix_add_kernel[grid](a, b, out, M, N)

    return out


@triton.jit
def _triton_matrix_sub_kernel(a_ptr, b_ptr, out, N, BLOCK_SIZE: tl.constexpr) -> None:
    pid = tl.program_id(0)

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    A = tl.load(a_ptr + idx, mask=mask, other=0.0)
    B = tl.load(b_ptr + idx, mask=mask, other=0.0)

    out_data = A - B

    tl.store(out + idx, out_data, mask=mask)


def matrix_sub_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Subtract two matrices using Triton kernel.

    Args:
        a: First input tensor (must be on CUDA, float32)
        b: Second input tensor (must be on CUDA, float32)
    """
    if not a.is_cuda or not b.is_cuda:
        raise TritonExtensionError("Both tensors must be on CUDA device")
    if a.dtype != b.dtype:
        raise TritonExtensionError("Both tensors must have the same dtype")
    if a.shape != b.shape:
        raise TritonExtensionError("Both tensors must have the same shape")

    N = a.numel()
    out = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    _triton_matrix_sub_kernel[grid](a, b, out, N, BLOCK_SIZE=tl.constexpr(256))

    return out
