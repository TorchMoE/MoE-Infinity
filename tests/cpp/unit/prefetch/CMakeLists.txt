find_package(GTest REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH
)
if(NOT TORCH_PYTHON_LIBRARY)
  find_library(TORCH_PYTHON_LIBRARY torch_python)
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
  OUTPUT_VARIABLE ARCHER_PY_LIBDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_library(ARCHER_PYTHON_SHARED
  NAMES "python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}" python3
  PATHS "${ARCHER_PY_LIBDIR}"
  NO_DEFAULT_PATH
)

set(ARCHER_TEST_LINK_LIBS
  archer_core GTest::gtest_main pthread ${TORCH_PYTHON_LIBRARY}
)
if(ARCHER_PYTHON_SHARED)
  list(APPEND ARCHER_TEST_LINK_LIBS ${ARCHER_PYTHON_SHARED})
endif()

add_executable(test_expert_policy test_expert_policy.cpp)
target_link_libraries(test_expert_policy PRIVATE ${ARCHER_TEST_LINK_LIBS})

add_test(NAME ExpertPolicyTests COMMAND test_expert_policy)

add_executable(test_expert_residency test_expert_residency.cpp)
target_link_libraries(test_expert_residency PRIVATE ${ARCHER_TEST_LINK_LIBS})

add_test(NAME ExpertResidencyTests COMMAND test_expert_residency)
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

add_executable(test_prefetch_queue_accounting test_prefetch_queue_accounting.cpp)
target_link_libraries(test_prefetch_queue_accounting PRIVATE
  archer_core GTest::gtest_main CUDA::cudart ${TORCH_LIBRARIES}
  ${Python3_LIBRARIES})
target_include_directories(test_prefetch_queue_accounting PRIVATE
  ${PROJECT_SOURCE_DIR}/core ${TORCH_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS})
include(GoogleTest)
gtest_discover_tests(test_prefetch_queue_accounting)
