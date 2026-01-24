import pytest
import torch
from torch.nn.functional import layer_norm as torch_layer_norm

from gpu_100days import layer_norm_fused

torch.manual_seed(20)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((16, 1024), torch.float32),
        ((4, 16, 1024), torch.float32),
        ((16, 1024), torch.float16),
        ((4, 16, 1024), torch.float16),
        ((16, 1024), torch.bfloat16),
        ((4, 16, 1024), torch.bfloat16),
    ],
)
def test_layer_norm_fused_fwd(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    normalized_shape = (shape[-1],)
    weight = torch.randn(normalized_shape, device="cuda", dtype=dtype)
    bias = torch.randn(normalized_shape, device="cuda", dtype=dtype)
    eps = 1e-5
    y = layer_norm_fused(x, normalized_shape, weight, bias, eps)
    y_torch = torch_layer_norm(x, normalized_shape, weight, bias, eps)
    assert y.shape == y_torch.shape
    torch.testing.assert_close(y, y_torch, atol=3e-2, rtol=0)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((16, 1024), torch.float32),
        ((4, 16, 1024), torch.float32),
        ((16, 1024), torch.float16),
        ((4, 16, 1024), torch.float16),
        ((16, 1024), torch.bfloat16),
        ((4, 16, 1024), torch.bfloat16),
    ],
)
def test_layer_norm_fused_bwd(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    normalized_shape = (shape[-1],)
    weight = torch.rand(normalized_shape, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(normalized_shape, device="cuda", dtype=dtype, requires_grad=True)
    eps = 1e-5
    dy = 0.1 * torch.randn_like(x, dtype=dtype)
    x.requires_grad_(True)
    y_tri = layer_norm_fused(x, normalized_shape, weight, bias, eps)
    y_ref = torch_layer_norm(x, normalized_shape, weight, bias, eps)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = (_.grad.clone() for _ in [x, weight, bias])  # type: ignore[union-attr]
    x.grad, weight.grad, bias.grad = None, None, None
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = (_.grad.clone() for _ in [x, weight, bias])  # type: ignore[union-attr]

    torch.testing.assert_close(y_tri, y_ref, atol=3e-2, rtol=0)
    torch.testing.assert_close(dx_tri, dx_ref, atol=3e-2, rtol=0)
    torch.testing.assert_close(dw_tri, dw_ref, atol=3e-2, rtol=0)
    torch.testing.assert_close(db_tri, db_ref, atol=3e-2, rtol=0)
