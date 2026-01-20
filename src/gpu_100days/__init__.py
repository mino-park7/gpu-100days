def main() -> None:
    print("Hello from gpu-100days!")


# Export CUDA operations
try:
    from .cuda_kernels import (
        gray_scale,
        matrix_add,
        matrix_sub,
        matrix_transpose,
        seeded_dropout,
        vector_add,
    )
    from .triton_kernels import (
        add_triton,
        grey_scale_triton,
        matrix_add_triton,
        matrix_multiply_triton,
        matrix_sub_triton,
        matrix_transpose_triton,
        seeded_dropout_triton,
        softmax_triton,
    )

    __all__ = [
        "matrix_add",
        "matrix_add_triton",
        "matrix_transpose",
        "matrix_transpose_triton",
        "vector_add",
        "matrix_sub",
        "matrix_sub_triton",
        "grey_scale_triton",
        "gray_scale",
        "matrix_multiply_triton",
        "seeded_dropout_triton",
        "seeded_dropout",
        "add_triton",
        "softmax_triton",
    ]
except ImportError:
    # Extension not built yet
    __all__ = []
