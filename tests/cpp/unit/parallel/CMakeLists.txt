# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

add_executable(test_expert_timing_lifecycle test_expert_timing_lifecycle.cpp)
target_link_libraries(test_expert_timing_lifecycle PRIVATE
  archer_core GTest::gtest_main CUDA::cudart ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES})
target_include_directories(test_expert_timing_lifecycle PRIVATE
  ${PROJECT_SOURCE_DIR}/core ${TORCH_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS})
include(GoogleTest)
gtest_discover_tests(test_expert_timing_lifecycle)
