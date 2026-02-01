import pytest
import torch
from conftest import benchmark_kernel_vs_pytorch

from gpu_100days import rope

torch.manual_seed(20)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply rotary position embedding to query and key tensors.
    Reference implementation using PyTorch operations.

    Args:
        q: Query tensor of shape (batch_size, seq_len, n_heads, head_dim)
        k: Key tensor of shape (batch_size, seq_len, n_heads, head_dim)
        cos: Cosine values of shape (seq_len, head_dim // 2)
        sin: Sine values of shape (seq_len, head_dim // 2)
        position_ids: Optional position indices of shape (batch_size, seq_len)

    Returns:
        q_rot: Rotated query tensor
        k_rot: Rotated key tensor
    """
    # Reshape cos and sin to match the tensor dimensions
    # cos/sin: (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Split head_dim into pairs for rotation
    # q: (batch_size, seq_len, n_heads, head_dim)
    # Split into (batch_size, seq_len, n_heads, head_dim // 2, 2)
    head_dim = q.shape[-1]
    q_reshaped = q.view(*q.shape[:-1], head_dim // 2, 2)
    k_reshaped = k.view(*k.shape[:-1], head_dim // 2, 2)

    # Extract real and imaginary parts
    q_real = q_reshaped[..., 0]
    q_imag = q_reshaped[..., 1]
    k_real = k_reshaped[..., 0]
    k_imag = k_reshaped[..., 1]

    # Apply rotation: [cos -sin; sin cos] @ [real; imag]
    q_rot_real = q_real * cos - q_imag * sin
    q_rot_imag = q_real * sin + q_imag * cos
    k_rot_real = k_real * cos - k_imag * sin
    k_rot_imag = k_real * sin + k_imag * cos

    # Combine back
    q_rot = torch.stack([q_rot_real, q_rot_imag], dim=-1).flatten(-2)
    k_rot = torch.stack([k_rot_real, k_rot_imag], dim=-1).flatten(-2)

    return q_rot, k_rot


def generate_rope_freqs(seq_len, head_dim, theta=10000.0, device="cuda", dtype=torch.float32):
    """
    Generate rotary position embedding frequencies.

    Args:
        seq_len: Sequence length
        head_dim: Head dimension (must be even)
        theta: Base frequency (default: 10000.0)
        device: Device to create tensors on
        dtype: Data type

    Returns:
        cos: Cosine values of shape (seq_len, head_dim // 2)
        sin: Sine values of shape (seq_len, head_dim // 2)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Create position indices
    positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # (seq_len, 1)

    # Create frequency indices
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    freqs = positions * freqs.unsqueeze(0)  # (seq_len, head_dim // 2)

    # Compute cos and sin
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 256, 512])
@pytest.mark.parametrize("n_heads", [8, 16])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rope_triton(batch_size, seq_len, n_heads, head_dim, dtype):
    """Test RoPE Triton implementation against reference PyTorch implementation."""
    # Generate input tensors
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device="cuda", dtype=dtype)

    # Generate RoPE frequencies
    cos, sin = generate_rope_freqs(seq_len, head_dim, device="cuda", dtype=dtype)

    # Apply RoPE using Triton
    q_rot_triton, k_rot_triton = rope(q, k, cos, sin)

    # Apply RoPE using reference implementation
    q_rot_ref, k_rot_ref = apply_rotary_pos_emb(q, k, cos, sin)

    # Check shapes
    assert q_rot_triton.shape == q.shape
    assert k_rot_triton.shape == k.shape
    assert q_rot_triton.dtype == dtype
    assert k_rot_triton.dtype == dtype
    assert q_rot_triton.device.type == "cuda"
    assert k_rot_triton.device.type == "cuda"
    assert q_rot_triton.is_contiguous()
    assert k_rot_triton.is_contiguous()

    # Compare results
    # RoPE can have numerical differences, especially with lower precision
    atol = 1e-1 if dtype in [torch.float16, torch.bfloat16] else 1e-5

    torch.testing.assert_close(q_rot_triton, q_rot_ref, atol=atol, rtol=0)
    torch.testing.assert_close(k_rot_triton, k_rot_ref, atol=atol, rtol=0)

    # Benchmark
    benchmark_kernel_vs_pytorch(
        lambda q, k, cos, sin: rope(q, k, cos, sin)[0],
        lambda q, k, cos, sin: apply_rotary_pos_emb(q, k, cos, sin)[0],
        q,
        k,
        cos,
        sin,
    )
