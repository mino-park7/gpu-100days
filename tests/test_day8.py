import pytest
import torch
from conftest import benchmark_kernel_vs_pytorch

from gpu_100days import matrix_transpose, matrix_transpose_triton


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 1), torch.float32),
        ((1, 10), torch.float32),
        ((10, 1), torch.float32),
        ((5, 5), torch.float32),
        ((16, 16), torch.float32),
        ((32, 32), torch.float32),
        ((64, 64), torch.float32),
        ((128, 128), torch.float32),
        ((500, 500), torch.float32),
        ((1000, 1000), torch.float32),
        ((1, 1000), torch.float32),
        ((1000, 1), torch.float32),
        ((1, 1000), torch.float16),
        ((1000, 1), torch.float16),
        ((1, 1000), torch.float16),
        ((1000, 1), torch.float16),
        ((500, 500), torch.float16),
        ((1000, 1000), torch.float16),
        ((1, 1000), torch.float16),
        ((1000, 1), torch.float16),
        ((1, 1000), torch.bfloat16),
        ((1000, 1), torch.bfloat16),
        ((1, 1000), torch.bfloat16),
        ((1000, 1), torch.bfloat16),
        ((500, 500), torch.bfloat16),
        ((1000, 1000), torch.bfloat16),
        ((1, 1000), torch.bfloat16),
        ((1000, 1), torch.bfloat16),
    ],
)
def test_matrix_transpose(shape, dtype):
    input = torch.randn(shape, device="cuda", dtype=dtype)
    output = matrix_transpose(input)
    assert output.shape == (shape[1], shape[0])
    assert torch.allclose(output, input.t().contiguous())
    benchmark_kernel_vs_pytorch(matrix_transpose, lambda x: x.t().contiguous(), input)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 1), torch.float32),
        ((1, 10), torch.float32),
        ((10, 1), torch.float32),
        ((5, 5), torch.float32),
        ((16, 16), torch.float32),
        ((32, 32), torch.float32),
        ((64, 64), torch.float32),
        ((128, 128), torch.float32),
        ((500, 500), torch.float32),
        ((1000, 1000), torch.float32),
        ((1, 1000), torch.float32),
        ((1000, 1), torch.float32),
        ((1, 1000), torch.float16),
        ((1000, 1), torch.float16),
        ((1, 1000), torch.float16),
        ((1000, 1), torch.float16),
        ((500, 500), torch.float16),
        ((1000, 1000), torch.float16),
        ((1, 1000), torch.float16),
        ((1000, 1), torch.float16),
        ((1, 1000), torch.bfloat16),
        ((1000, 1), torch.bfloat16),
        ((1, 1000), torch.bfloat16),
        ((1000, 1), torch.bfloat16),
        ((500, 500), torch.bfloat16),
        ((1000, 1000), torch.bfloat16),
        ((1, 1000), torch.bfloat16),
        ((1000, 1), torch.bfloat16),
    ],
)
def test_matrix_transpose_triton(shape, dtype):
    input = torch.randn(shape, device="cuda", dtype=dtype)
    output = matrix_transpose_triton(input)
    assert output.shape == (shape[1], shape[0])
    assert output.dtype == dtype
    assert torch.allclose(output, input.t().contiguous())
    benchmark_kernel_vs_pytorch(matrix_transpose_triton, lambda x: x.t().contiguous(), input)
