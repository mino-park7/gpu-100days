import pytest
import torch

from gpu_100days import matrix_multiply_triton


@pytest.mark.parametrize(
    "M,N,K",
    [
        (64, 64, 64),  # Small square
        (128, 128, 128),  # Medium square
        (256, 256, 256),  # Large square
        (512, 256, 512),  # Original test case
        (256, 512, 256),  # Transposed
        (128, 256, 512),  # Rectangular
        (512, 128, 256),  # Different aspect ratio
        (1024, 512, 256),  # Large M
        (256, 1024, 512),  # Large N
        (128, 128, 1024),  # Large K
    ],
)
def test_matrix_multiply_triton(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = matrix_multiply_triton(a, b)
    assert torch.allclose(c, a @ b, rtol=1e-5)


if __name__ == "__main__":
    from pathlib import Path

    import triton

    pwd = Path(__file__).resolve().parent
    (pwd / "day5").mkdir(parents=True, exist_ok=True)

    configs = []
    benchmark_config = triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["cuBLAS", "triton"],
        line_names=["cuBLAS", "triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name=("matmul-perf"),
        args={},
    )
    configs.append(benchmark_config)

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        if provider == "cuBLAS":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.matmul(a, b), quantiles=quantiles
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: matrix_multiply_triton(a, b), quantiles=quantiles
            )

        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(print_data=True, show_plots=True, save_path=pwd / "day5")
