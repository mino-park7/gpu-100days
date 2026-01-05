from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the directory where this file is located
this_dir = Path(__file__).resolve().parent

# Find all .cu files in csrc directory
csrc_dir = this_dir / "csrc"
cu_files_abs = list(csrc_dir.rglob("*.cu"))
# Convert absolute paths to relative paths (required by setuptools)
cu_files = [str(f.relative_to(this_dir)) for f in cu_files_abs]


# Custom BuildExtension to set CMAKE_EXPORT_COMPILE_COMMANDS
class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        # Enable compile_commands.json generation
        # This works with both CMake and ninja builds
        import os

        super().__init__(*args, **kwargs)
        # Set environment variable for CMake
        os.environ["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"

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
        include_dirs=["csrc"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                # Use -gencode to compile for multiple architectures
                # Format: -gencode arch=compute_XX,code=sm_XX
                "-gencode",
                "arch=compute_50,code=sm_50",
                "-gencode",
                "arch=compute_60,code=sm_60",
                "-gencode",
                "arch=compute_61,code=sm_61",
                "-gencode",
                "arch=compute_70,code=sm_70",
                "-gencode",
                "arch=compute_75,code=sm_75",
                "-gencode",
                "arch=compute_80,code=sm_80",
                "-gencode",
                "arch=compute_86,code=sm_86",  # A40, A100
                "-gencode",
                "arch=compute_89,code=sm_89",
                "-gencode",
                "arch=compute_90,code=sm_90",
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
