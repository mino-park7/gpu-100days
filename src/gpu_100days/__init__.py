def main() -> None:
    print("Hello from gpu-100days!")


# Export CUDA operations
try:
    from .cuda_kernels import gray_scale, matrix_add, matrix_sub, vector_add
    from .triton_kernels import grey_scale_triton, matrix_add_triton, matrix_sub_triton

    __all__ = [
        "matrix_add",
        "matrix_add_triton",
        "vector_add",
        "matrix_sub",
        "matrix_sub_triton",
        "grey_scale_triton",
        "gray_scale",
    ]
except ImportError:
    # Extension not built yet
    __all__ = []
