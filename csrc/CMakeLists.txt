cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_ops)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Prefer Python from .venv if it exists
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(VENV_PYTHON "${PROJECT_ROOT}/.venv/bin/python3")

if(EXISTS "${VENV_PYTHON}")
    get_filename_component(VENV_PYTHON_DIR "${VENV_PYTHON}" DIRECTORY)
    get_filename_component(VENV_ROOT "${VENV_PYTHON_DIR}" DIRECTORY)

    # Set Python3_ROOT_DIR to prefer venv Python
    set(Python3_ROOT_DIR "${VENV_ROOT}")
    message(STATUS "Using Python from .venv: ${VENV_PYTHON}")
else()
    message(STATUS "No .venv found, using system Python")
endif()

# Find Python to get torch path from venv
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Allow manual override of Torch_DIR
if(NOT Torch_DIR)
    # Get PyTorch cmake prefix path from installed torch package
    # Try torch.utils.cmake_prefix_path first (PyTorch 2.x)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
    )

    # If the above fails, try alternative method
    if(TORCH_CMAKE_PATH_RESULT OR NOT TORCH_CMAKE_PREFIX_PATH)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake'))"
            OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT2
        )
    endif()

    # Verify the path exists and contains TorchConfig.cmake
    if(TORCH_CMAKE_PREFIX_PATH AND EXISTS "${TORCH_CMAKE_PREFIX_PATH}")
        set(TORCH_CONFIG_FILE "${TORCH_CMAKE_PREFIX_PATH}/Torch/TorchConfig.cmake")

        if(EXISTS "${TORCH_CONFIG_FILE}")
            list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
            set(Torch_DIR "${TORCH_CMAKE_PREFIX_PATH}/Torch")
            message(STATUS "Found Torch cmake path: ${TORCH_CMAKE_PREFIX_PATH}")
            message(STATUS "Set Torch_DIR to: ${Torch_DIR}")
        else()
            message(WARNING "Torch cmake path found but TorchConfig.cmake not found at: ${TORCH_CONFIG_FILE}")
        endif()
    endif()
else()
    message(STATUS "Using manually set Torch_DIR: ${Torch_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}/..")
endif()

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# If TORCH_INCLUDE_DIRS is not set, try to get it from Python
if(NOT TORCH_INCLUDE_DIRS)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))"
        OUTPUT_VARIABLE TORCH_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(TORCH_INCLUDE_DIR AND EXISTS "${TORCH_INCLUDE_DIR}")
        set(TORCH_INCLUDE_DIRS "${TORCH_INCLUDE_DIR}")
        message(STATUS "Found Torch include directory: ${TORCH_INCLUDE_DIRS}")
    endif()
endif()

# Enable CUDA language
enable_language(CUDA)

# Set CUDA properties
# set_property(TARGET torch_library PROPERTY INTERFACE_COMPILE_OPTIONS "")
# set_property(TARGET torch_library PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set CUDA language standard for pybind11 compatibility
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture detection (sm86 only for faster build, matching setup.py)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

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
    ${Python3_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cuda_ops PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Set output directory
set_target_properties(cuda_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    CUDA_SEPARABLE_COMPILATION OFF
)

# Add optimization flags
target_compile_options(cuda_ops PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:
    $<$<CONFIG:Release>:-O3 --use_fast_math>
    >
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
