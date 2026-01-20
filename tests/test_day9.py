import pytest
import torch
import triton
from conftest import benchmark_kernel_vs_pytorch

from gpu_100days import softmax_triton


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Naive implementation of softmax function.
    """
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    return numerator / denominator[:, None]


@pytest.mark.parametrize(
    "shape",
    [
        (1000, 1000),
        (1000, 10000),
        (10000, 1000),
        (10000, 10000),
    ],
)
def test_softmax_triton(shape):
    x = torch.randn(shape, device="cuda")
    y = softmax_triton(x)
    assert y.shape == x.shape
    assert torch.allclose(y, naive_softmax(x))
    benchmark_kernel_vs_pytorch(softmax_triton, naive_softmax, x)
    benchmark_kernel_vs_pytorch(softmax_triton, torch.nn.functional.softmax, x)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch", "naive_softmax"],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax_triton(x))
    if provider == "naive_softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    from pathlib import Path

    pwd = Path(__file__).resolve().parent
    (pwd / "day9").mkdir(parents=True, exist_ok=True)

    benchmark.run(show_plots=True, print_data=True, save_path=pwd / "day9")
