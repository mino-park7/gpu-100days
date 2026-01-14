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

    channel, height, width = image.shape

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
