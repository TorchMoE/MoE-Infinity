cmake_minimum_required(VERSION 3.18)

find_package(GTest REQUIRED)

find_library(torch_python_LIBRARY torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib")

find_package(Python3 REQUIRED COMPONENTS Development)

add_executable(test_expert_residency
  ${CMAKE_CURRENT_SOURCE_DIR}/test_expert_residency.cpp
  ${CMAKE_SOURCE_DIR}/extensions/kernel/v4_fp4/mxfp4_dequant.cu
  ${CMAKE_SOURCE_DIR}/extensions/kernel/v4_fp4/fp8_dequant.cu
)

set_target_properties(test_expert_residency PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_include_directories(test_expert_residency PRIVATE
  ${CMAKE_SOURCE_DIR}/core
  ${CMAKE_SOURCE_DIR}/extensions
  ${CUTLASS_DIR}/include
  ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(test_expert_residency PRIVATE
  -Wl,--whole-archive $<TARGET_FILE:archer_core> -Wl,--no-whole-archive
  archer_core
  ${GTEST_LIBRARIES}
  GTest::gtest_main
  ${TORCH_LIBRARIES}
  ${torch_python_LIBRARY}
  Python3::Python
  pthread
)

add_test(NAME ExpertResidencyTests COMMAND test_expert_residency)

add_executable(test_expert_residency_variants
  ${CMAKE_CURRENT_SOURCE_DIR}/test_expert_residency_variants.cpp
  ${CMAKE_SOURCE_DIR}/extensions/kernel/v4_fp4/mxfp4_dequant.cu
  ${CMAKE_SOURCE_DIR}/extensions/kernel/v4_fp4/fp8_dequant.cu
)

set_target_properties(test_expert_residency_variants PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_include_directories(test_expert_residency_variants PRIVATE
  ${CMAKE_SOURCE_DIR}/core
  ${CMAKE_SOURCE_DIR}/extensions
  ${CUTLASS_DIR}/include
  ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(test_expert_residency_variants PRIVATE
  -Wl,--whole-archive $<TARGET_FILE:archer_core> -Wl,--no-whole-archive
  archer_core
  ${GTEST_LIBRARIES}
  GTest::gtest_main
  ${TORCH_LIBRARIES}
  ${torch_python_LIBRARY}
  Python3::Python
  pthread
)

add_test(NAME ExpertResidencyVariantTests COMMAND test_expert_residency_variants)
