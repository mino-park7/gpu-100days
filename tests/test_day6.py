import pytest
import torch

from gpu_100days import seeded_dropout, seeded_dropout_triton


@pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_seeded_dropout_triton(p):
    x = torch.randn(1000, device="cuda", dtype=torch.float32)
    output = seeded_dropout_triton(x, p, seed=1000)
    output_2 = seeded_dropout_triton(x, p, seed=1000)
    output_3 = seeded_dropout_triton(x, p, seed=1000)
    assert output.shape == x.shape
    assert torch.allclose(output, output_2)
    assert torch.allclose(output, output_3)


@pytest.mark.parametrize("p", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_seeded_dropout(p):
    x = torch.randn(1000, device="cuda", dtype=torch.float32)
    output = seeded_dropout(x, p, seed=1000)
    output_2 = seeded_dropout(x, p, seed=1000)
    output_3 = seeded_dropout(x, p, seed=1000)
    assert output.shape == x.shape
    assert torch.allclose(output, output_2)
    assert torch.allclose(output, output_3)
