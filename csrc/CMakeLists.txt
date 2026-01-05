cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_ops)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find Python to get torch path from venv
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get PyTorch cmake prefix path from installed torch package
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake'))"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# If the above fails, try alternative method
if(NOT TORCH_CMAKE_PREFIX_PATH OR NOT EXISTS "${TORCH_CMAKE_PREFIX_PATH}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

# Set Torch_DIR to help find_package locate libtorch
if(TORCH_CMAKE_PREFIX_PATH AND EXISTS "${TORCH_CMAKE_PREFIX_PATH}")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
    set(Torch_DIR "${TORCH_CMAKE_PREFIX_PATH}/Torch")
endif()

# Find PyTorch
# find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Enable CUDA language
enable_language(CUDA)

# Set CUDA properties
# set_property(TARGET torch_library PROPERTY INTERFACE_COMPILE_OPTIONS "")
# set_property(TARGET torch_library PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")

# Set CUDA language standard for pybind11 compatibility
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture detection
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;75;80;86;89;90")
endif()

# Collect all .cu files in the current directory
file(GLOB_RECURSE CUDA_SOURCES "*.cu")

# Create the CUDA extension library
add_library(cuda_ops SHARED ${CUDA_SOURCES})

# Set CUDA language standard
set_property(TARGET cuda_ops PROPERTY CUDA_STANDARD 17)
set_property(TARGET cuda_ops PROPERTY CUDA_STANDARD_REQUIRED ON)

# Enable pybind11 support in CUDA files
set_property(TARGET cuda_ops PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Include directories
target_include_directories(cuda_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cuda_ops PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Set output directory
set_target_properties(cuda_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    CUDA_SEPARABLE_COMPILATION ON
)

# Compiler-specific options
if(MSVC)
    target_compile_options(cuda_ops PRIVATE /W4)
    target_compile_definitions(cuda_ops PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(cuda_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /W4>)
else()
    target_compile_options(cuda_ops PRIVATE -Wall -Wextra)
    target_compile_options(cuda_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall -Wextra>)
endif()

# Define TORCH_EXTENSION_NAME for pybind11
target_compile_definitions(cuda_ops PRIVATE TORCH_EXTENSION_NAME=cuda_ops)
