def main() -> None:
    print("Hello from gpu-100days!")


# Export CUDA operations
try:
    from .vector_add import vector_add

    __all__ = ["vector_add"]
except ImportError:
    # Extension not built yet
    __all__ = []
