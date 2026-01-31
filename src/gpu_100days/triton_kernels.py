from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    pass


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


@triton.autotune(
    configs=[
        triton.Config({"bs0": 16, "bs1": 16}, num_warps=4),
        triton.Config({"bs0": 32, "bs1": 16}, num_warps=4),
        triton.Config({"bs0": 16, "bs1": 32}, num_warps=4),
        triton.Config({"bs0": 8, "bs1": 32}, num_warps=4),
        triton.Config({"bs0": 32, "bs1": 8}, num_warps=4),
    ],
    key=["height", "width"],
)
@triton.jit
def _triton_grey_scale_kernel(
    image_ptr, out_ptr, height: int, width: int, bs0: tl.constexpr, bs1: tl.constexpr
) -> None:
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    row_indices = pid_x * bs0 + tl.arange(0, bs0)
    col_indices = pid_y * bs1 + tl.arange(0, bs1)
    row_mask = row_indices < height
    col_mask = col_indices < width

    valid_mask = row_mask[:, None] & col_mask[None, :]

    flat_indices = row_indices[:, None] * width + col_indices[None, :]

    # Load uint8 values and convert to float32 for computation
    R = tl.load(image_ptr + 0 * width * height + flat_indices, mask=valid_mask).to(tl.float32)
    G = tl.load(image_ptr + 1 * width * height + flat_indices, mask=valid_mask).to(tl.float32)
    B = tl.load(image_ptr + 2 * width * height + flat_indices, mask=valid_mask).to(tl.float32)

    # Compute grayscale value
    out_data = 0.299 * R + 0.587 * G + 0.114 * B

    # Convert back to uint8 and store
    out_data = out_data.to(tl.uint8)

    tl.store(out_ptr + flat_indices, out_data, mask=valid_mask)


def grey_scale_triton(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image to greyscale using Triton kernel.

    Args:
        image: Input image tensor (must be on CUDA, uint8)
    """
    rgb_channels = 3
    if not image.is_cuda:
        raise TritonExtensionError("Image must be on CUDA device")
    if image.dtype != torch.uint8:
        raise TritonExtensionError("Image must be uint8")
    if image.shape[0] != rgb_channels:
        raise TritonExtensionError("Image must have 3 channels")

    if not image.is_contiguous():
        image = image.contiguous()

    _channel, height, width = image.shape

    out = torch.empty((height, width), dtype=image.dtype, device=image.device)

    grid = lambda meta: (triton.cdiv(height, meta["bs0"]), triton.cdiv(width, meta["bs1"]))
    _triton_grey_scale_kernel[grid](image, out, height, width)

    out = out.unsqueeze(0)

    return out.view(1, height, width)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _triton_matrix_multiply_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: int,
    N: int,
    K: int,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
) -> None:
    """Kernel for computing matmul C = A x B
    A has shape (M, K),B has shape (K,N) and C has shape (M,N)
    """

    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matrix_multiply_triton(a: torch.Tensor, b: torch.Tensor, activation="") -> torch.Tensor:
    """
    Matrix multiplication using Triton kernel.

    Args:
        a: First input tensor (must be on CUDA, float32)
        b: Second input tensor (must be on CUDA, float32)
        activation: Activation function to use ("leaky_relu" or "none")
    """
    if not a.is_cuda or not b.is_cuda:
        raise TritonExtensionError("Both tensors must be on CUDA device")
    if a.shape[1] != b.shape[0]:
        raise TritonExtensionError("Invalid tensor shapes for matrix multiplication")
    if a.dtype != b.dtype:
        raise TritonExtensionError("Both tensors must have the same dtype")

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    _triton_matrix_multiply_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
    )

    return c


@triton.jit
def leaky_relu(x: tl.tensor):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def _seeded_dropout(
    x_ptr, output_ptr, n_elements: int, p: float, seed: int, BLOCK_SIZE: tl.constexpr
) -> None:
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    random = tl.rand(seed, offsets)

    x_keep = random > p

    output = tl.where(x_keep, x / (1 - p), 0.0)

    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout_triton(x: torch.Tensor, p: float, seed: int = 42) -> torch.Tensor:
    assert x.is_contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=tl.constexpr(256))

    return output


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr) -> None:
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    out = x + y

    tl.store(out_ptr + offsets, out, mask=mask)


def add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or not y.is_cuda:
        raise TritonExtensionError("Both tensors must be on CUDA device")
    if x.dtype != y.dtype:
        raise TritonExtensionError("Both tensors must have the same dtype")
    if x.shape != y.shape:
        raise TritonExtensionError("Both tensors must have the same shape")

    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=tl.constexpr(256))
    return out


@triton.jit
def _matrix_transpose_kernel(
    input_ptr, output_ptr, M: int, N: int, BLOCK_SIZE: tl.constexpr
) -> None:
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    row_start = pid_x * BLOCK_SIZE
    col_start = pid_y * BLOCK_SIZE

    row_indices = row_start + tl.arange(0, BLOCK_SIZE)
    col_indices = col_start + tl.arange(0, BLOCK_SIZE)

    input_row_indices = row_indices[:, None]
    input_col_indices = col_indices[None, :]

    input_row_mask = input_row_indices < M
    input_col_mask = input_col_indices < N
    valid_input_mask = input_row_mask & input_col_mask

    input_flat_indices = input_row_indices * N + input_col_indices
    input_A = tl.load(input_ptr + input_flat_indices, mask=valid_input_mask, other=0.0)

    output_row_indices = col_indices[None, :]
    output_col_indices = row_indices[:, None]
    output_row_mask = output_row_indices < N
    output_col_mask = output_col_indices < M
    valid_output_mask = output_row_mask & output_col_mask

    output_flat_indices = output_row_indices * M + output_col_indices
    tl.store(output_ptr + output_flat_indices, input_A, mask=valid_output_mask)


def matrix_transpose_triton(input: torch.Tensor) -> torch.Tensor:
    if not input.is_cuda:
        raise TritonExtensionError("Input tensor must be on CUDA device")
    possible_dim = 2
    if input.dim() != possible_dim:
        raise TritonExtensionError(f"Input tensor must be a {possible_dim}D tensor")
    if not input.is_contiguous():
        input = input.contiguous()

    M, N = input.shape
    out = torch.empty((N, M), dtype=input.dtype, device=input.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    _matrix_transpose_kernel[grid](input, out, M, N, BLOCK_SIZE=tl.constexpr(16))
    return out


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=8, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
) -> None:
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

        row_minus_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    y = torch.empty_like(x)

    _softmax_kernel[(n_rows,)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )

    return y


@triton.jit
def _layer_norm_fwd_fused(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    stride,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_elements, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x_row = tl.load(x_ptr + cols, mask=cols < n_elements, other=0.0).to(tl.float32)
        _mean += x_row
    mean = tl.sum(_mean, axis=0) / n_elements

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_elements, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < n_elements, other=0.0).to(tl.float32)
        x = tl.where(cols < n_elements, x - mean, 0.0)
        _var += x * x

    var = tl.sum(_var, axis=0) / n_elements

    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    for off in range(0, n_elements, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_elements
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(y_ptr + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(
    DX,
    DY,
    DW,
    DB,
    X,
    W,
    Mean,
    Rstd,
    Lock,
    stride,
    N,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    X += row * stride
    DY += row * stride
    DX += row * stride

    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M

    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols

    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - c1 * xhat - c2) * rstd

    tl.store(DX + cols, dx, mask=mask)

    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)

    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass

    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)

    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


class LayerNormFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        normalized_shape: tuple[int, ...],
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        y = torch.empty_like(x)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim > >= 64KB.")

        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        _layer_norm_fwd_fused[(M,)](
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,
            x_arg.stride(0),
            N,
            eps,
            BLOCK_SIZE=tl.constexpr(BLOCK_SIZE),
            num_warps=num_warps,  # type: ignore
            num_ctas=1,  # type: ignore
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.eps = eps
        ctx.num_warps = num_warps

        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):  # type: ignore
        x, w, _b, m, v = ctx.saved_tensors
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256

        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        db = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M,)](
            dx,
            dy,
            _dw,
            _db,
            x,
            w,
            m,
            v,
            locks,
            x_arg.stride(0),
            N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,  # type: ignore
            num_warps=ctx.num_warps,  # type: ignore
        )

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        _layer_norm_bwd_dwdb[grid](
            _dw,
            _db,
            dw,
            db,
            min(GROUP_SIZE_M, M),
            N,
            BLOCK_SIZE_M=tl.constexpr(32),
            BLOCK_SIZE_N=tl.constexpr(128),
            num_ctas=1,  # type: ignore
        )
        return dx, None, dw, db, None


layer_norm_fused = LayerNormFused.apply


@triton.jit
def _silu_kernel(input_ptr, output_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr) -> None:
    start = tl.program_id(0) * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_orig = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x = x_orig.to(tl.float32)
    y = (x * tl.sigmoid(x)).to(x_orig.dtype)
    tl.store(output_ptr + offsets, y, mask=mask)


def silu_triton(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise TritonExtensionError("Tensor must be on CUDA device")
    if x.dtype not in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        raise TritonExtensionError("Tensor must be float16, float32, or bfloat16")

    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    n_elements = x.numel()
    _silu_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=tl.constexpr(16),
    )
    return out
