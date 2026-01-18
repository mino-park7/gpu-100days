import pytest
import torch
import triton
from conftest import benchmark_kernel_vs_pytorch

from gpu_100days import add_triton


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
        ((1, 1000), torch.float16),
        ((1, 1000), torch.bfloat16),
        ((1000, 1), torch.bfloat16),
        ((1, 1000), torch.bfloat16),
        ((1000, 1), torch.bfloat16),
        ((500, 500), torch.bfloat16),
        ((1000, 1000), torch.bfloat16),
        ((1, 1000), torch.bfloat16),
        ((1000, 1), torch.bfloat16),
        ((1, 1000), torch.bfloat16),
    ],
)
def test_add_triton(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = torch.randn(shape, device="cuda", dtype=dtype)
    output = add_triton(x, y)
    assert output.shape == shape
    assert output.dtype == dtype
    assert torch.allclose(output, x + y)
    benchmark_kernel_vs_pytorch(add_triton, lambda x, y: x + y, x, y)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_triton(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    from pathlib import Path

    pwd = Path(__file__).resolve().parent
    (pwd / "day7").mkdir(parents=True, exist_ok=True)
    benchmark.run(print_data=True, show_plots=True, save_path=pwd / "day7")
