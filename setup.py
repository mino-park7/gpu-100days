import contextlib
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths

# Get the directory where this file is located
this_dir = Path(__file__).resolve().parent

# Find all .cu files in csrc directory
csrc_dir = this_dir / "csrc"
cu_files_abs = list(csrc_dir.rglob("*.cu"))
# Convert absolute paths to relative paths (required by setuptools)
cu_files = [str(f.relative_to(this_dir)) for f in cu_files_abs]

# Get PyTorch include directories
torch_include_dirs = include_paths()

# Enable ccache for CUDA compilation
os.environ["CCACHE_COMPILERTYPE"] = "nvcc"
os.environ["CCACHE_BASEDIR"] = str(this_dir)
# Point nvcc to use ccache (CMake will pick this up)
os.environ["CUDAHOSTCXX"] = "ccache g++"

# Get PyTorch cmake path for Torch_DIR and ensure include paths are set
try:
    import torch

    # Get torch include directory directly
    torch_file = torch.__file__
    if torch_file is not None:
        torch_path = Path(torch_file).parent
        torch_include_dir = str(torch_path / "include")
        # Add torch include directory if not already in the list
        if torch_include_dir not in torch_include_dirs:
            torch_include_dirs.append(torch_include_dir)
        # Also add torch/csrc/api/include if it exists
        torch_api_include = str(torch_path / "include" / "torch" / "csrc" / "api" / "include")
        if Path(torch_api_include).exists() and torch_api_include not in torch_include_dirs:
            torch_include_dirs.append(torch_api_include)

    # Try torch.utils.cmake_prefix_path first (PyTorch 2.x)
    torch_cmake_path: str | None = None
    with contextlib.suppress(AttributeError):
        torch_cmake_path = torch.utils.cmake_prefix_path

    # Fallback to manual path construction if needed
    if torch_cmake_path is None and torch_file is not None:
        torch_path = Path(torch_file).parent
        torch_cmake_path = str(torch_path / "share" / "cmake")

    # Set Torch_DIR if path is valid
    if torch_cmake_path is not None:
        torch_dir = Path(torch_cmake_path) / "Torch"
        if (torch_dir / "TorchConfig.cmake").exists():
            # Set Torch_DIR for CMake
            os.environ["TORCH_DIR"] = str(torch_dir)
except (ImportError, AttributeError):
    # Torch not available, will be handled by CMake
    pass


# Custom BuildExtension to set CMAKE_EXPORT_COMPILE_COMMANDS
class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        # Enable compile_commands.json generation
        # This works with both CMake and ninja builds
        import multiprocessing

        super().__init__(*args, **kwargs)
        # Set environment variable for CMake
        os.environ["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"

        # Limit parallel jobs to avoid overwhelming system
        max_jobs = min(multiprocessing.cpu_count(), 8)
        os.environ["MAX_JOBS"] = str(max_jobs)

    def build_extensions(self):
        # Ensure compile_commands.json is generated
        # BuildExtension uses ninja internally, which can generate compile_commands.json
        # by setting CMAKE_EXPORT_COMPILE_COMMANDS=ON in the environment
        super().build_extensions()

        # If using CMake directly, ensure compile_commands.json is copied to project root
        build_temp = Path(self.build_temp)
        compile_commands_src = build_temp / "compile_commands.json"
        if compile_commands_src.exists():
            # Copy to project root for IDE support
            compile_commands_dst = (
                Path(self.get_finalized_command("build_py").build_lib).parent
                / "compile_commands.json"
            )
            compile_commands_dst = compile_commands_dst.resolve()
            if not compile_commands_dst.exists():
                import shutil

                shutil.copy2(compile_commands_src, compile_commands_dst)


# Define the CUDA extension
ext_modules = [
    CUDAExtension(
        name="cuda_ops",
        sources=cu_files,
        include_dirs=["csrc", *torch_include_dirs],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-lineinfo",  # Lightweight debug info instead of -g (faster compilation)
                "--ptxas-options=-O3",  # PTX assembly optimization
                # Use -gencode to compile for sm86 and above only (faster build)
                # Format: -gencode arch=compute_XX,code=sm_XX
                "-gencode",
                "arch=compute_86,code=sm_86",  # A40, A100
                # "-gencode",
                # "arch=compute_89,code=sm_89",
                # "-gencode",
                # "arch=compute_90,code=sm_90",
            ],
        },
    )
]

setup(
    name="gpu-100days",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExtension},
    zip_safe=False,
)
