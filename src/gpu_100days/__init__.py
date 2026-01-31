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
        silu,
        softmax,
        vector_add,
    )
    from .flash_attention import flash_attn_func
    from .triton_kernels import (
        add_triton,
        grey_scale_triton,
        layer_norm_fused,
        matrix_add_triton,
        matrix_multiply_triton,
        matrix_sub_triton,
        matrix_transpose_triton,
        seeded_dropout_triton,
        silu_triton,
        softmax_triton,
    )

    __all__ = [
        "add_triton",
        "flash_attn_func",
        "gray_scale",
        "grey_scale_triton",
        "matrix_add",
        "matrix_add_triton",
        "matrix_multiply_triton",
        "matrix_sub",
        "matrix_sub_triton",
        "matrix_transpose",
        "matrix_transpose_triton",
        "seeded_dropout",
        "seeded_dropout_triton",
        "silu",
        "silu_triton",
        "softmax",
        "softmax_triton",
        "vector_add",
    ]
except ImportError:
    # Extension not built yet
    __all__ = []
