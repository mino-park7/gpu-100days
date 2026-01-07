def main() -> None:
    print("Hello from gpu-100days!")


# Export CUDA operations
try:
    from .cuda_kernels import matrix_add, matrix_add_triton, vector_add

    __all__ = ["matrix_add", "matrix_add_triton", "vector_add"]
except ImportError:
    # Extension not built yet
    __all__ = []
