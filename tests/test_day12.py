import pytest
import torch
from conftest import benchmark_kernel_vs_pytorch
from torch.nn.functional import silu as torch_silu

from gpu_100days import silu_triton

torch.manual_seed(20)


@pytest.mark.parametrize(
    ["shape", "dtype"],
    [
        ((1, 1), torch.float16),
        ((1, 10), torch.float16),
        ((10, 1), torch.float16),
        ((5, 5), torch.float16),
        ((16, 16), torch.float16),
        ((32, 32), torch.float16),
        ((64, 64), torch.float16),
        ((128, 128), torch.float16),
        ((500, 500), torch.float16),
        ((1000, 1000), torch.float16),
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
        ((1, 1), torch.bfloat16),
        ((1, 10), torch.bfloat16),
        ((10, 1), torch.bfloat16),
        ((5, 5), torch.bfloat16),
        ((16, 16), torch.bfloat16),
        ((32, 32), torch.bfloat16),
        ((64, 64), torch.bfloat16),
        ((128, 128), torch.bfloat16),
        ((500, 500), torch.bfloat16),
        ((1000, 1000), torch.bfloat16),
    ],
)
def test_silu_triton(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    output = silu_triton(x)
    output_torch = torch_silu(x)
    assert output.shape == shape
    assert output.dtype == output_torch.dtype
    assert output.device.type == "cuda"
    assert output.is_contiguous()
    assert output.is_cuda
    torch.testing.assert_close(output, output_torch, atol=1e-3, rtol=0)

    benchmark_kernel_vs_pytorch(silu_triton, torch_silu, x)
