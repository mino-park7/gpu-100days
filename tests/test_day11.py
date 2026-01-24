import pytest
import torch

from gpu_100days import flash_attn_func

torch.manual_seed(20)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(("seqlen_q", "seqlen_k"), [(128, 128), (128, 256), (1, 128), (256, 256)])
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_func(batch_size, seqlen_q, seqlen_k, head_dim, causal, dtype):
    n_heads = 8
    q = torch.randn(batch_size, seqlen_q, n_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, n_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, n_heads, head_dim, device="cuda", dtype=dtype)
    o_triton = flash_attn_func(q, k, v, None, causal, None)
    assert o_triton.shape == (batch_size, seqlen_q, n_heads, head_dim)
    assert o_triton.dtype == dtype
    assert o_triton.device.type == "cuda"
    assert o_triton.is_contiguous()
    assert o_triton.is_cuda

    q_transposed = q.transpose(1, 2).contiguous()
    k_transposed = k.transpose(1, 2).contiguous()
    v_transposed = v.transpose(1, 2).contiguous()
    o_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q_transposed, k_transposed, v_transposed, attn_mask=None, dropout_p=0.0, is_causal=causal
    )
    o_sdpa = o_sdpa.transpose(1, 2).contiguous()
    assert o_sdpa.shape == (batch_size, seqlen_q, n_heads, head_dim)
    assert o_sdpa.dtype == dtype
    assert o_sdpa.device.type == "cuda"
    assert o_sdpa.is_contiguous()
    assert o_sdpa.is_cuda

    # Flash Attention은 블록 단위 처리로 인해 수치적 오차가 누적될 수 있음
    # head_dim이 클수록, 시퀀스가 길수록 오차가 커질 수 있음
    # fp16/bfloat16의 낮은 정밀도도 고려해야 함
    atol = 2.5 if head_dim == 128 else 2.0
    torch.testing.assert_close(o_triton, o_sdpa, atol=atol, rtol=0)
