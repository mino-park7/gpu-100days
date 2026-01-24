import pytest
import torch
from torch.nn.functional import layer_norm as torch_layer_norm

from gpu_100days import layer_norm_fused


@pytest.mark.parametrize("shape", [(1, 1024), (1, 1024, 16), (1, 1024, 16, 16)])
def test_layer_norm_fused(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    normalized_shape = (shape[-1],)
    weight = torch.randn(normalized_shape, device="cuda", dtype=torch.float32)
    bias = torch.randn(normalized_shape, device="cuda", dtype=torch.float32)
    eps = 1e-5
    y = layer_norm_fused(x, normalized_shape, weight, bias, eps)
    y_torch = torch_layer_norm(x, normalized_shape, weight, bias, eps)
    assert y.shape == y_torch.shape
    assert torch.allclose(y, y_torch, atol=1e-2, rtol=0)
