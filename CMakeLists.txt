cmake_minimum_required(VERSION 3.18)
project(MoEInfinity LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Find PyTorch via its bundled CMake config (same technique as tests/cuda/CMakeLists.txt)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "-c"
          "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE _TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _RET)
if(NOT _RET EQUAL 0)
  message(FATAL_ERROR "Could not find PyTorch cmake prefix. Install PyTorch first.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${_TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# CUTLASS: configurable via env var or -DCUTLASS_DIR=...
if(DEFINED ENV{CUTLASS_DIR})
  set(CUTLASS_DIR "$ENV{CUTLASS_DIR}" CACHE PATH "CUTLASS root")
else()
  set(CUTLASS_DIR "$ENV{HOME}/cutlass" CACHE PATH "CUTLASS root")
endif()
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_DIR}. Set CUTLASS_DIR.")
endif()

# CUDA architectures (default 80 = A100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(OpenMP REQUIRED)

option(MOE_BUILD_TESTS "Build native unit tests" OFF)

add_subdirectory(core)
add_subdirectory(extensions)

if(MOE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
