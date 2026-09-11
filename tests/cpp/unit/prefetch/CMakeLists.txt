add_executable(test_sparse_cache_resize
  test_sparse_cache_resize.cpp
  sparse_cache_resize_fixture.cpp
  # archer_core references these FP4/FP8 dequant kernels which are otherwise
  # only compiled into the separate extensions module.
  ${CMAKE_SOURCE_DIR}/extensions/kernel/v4_fp4/fp8_dequant.cu
  ${CMAKE_SOURCE_DIR}/extensions/kernel/v4_fp4/mxfp4_dequant.cu
)
set_target_properties(test_sparse_cache_resize PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
# The GTest imported targets advertise a cxx_std_* interface compile feature.
# In this CXX+CUDA project CMake does not populate CMAKE_CXX_COMPILE_FEATURES,
# so clear the propagated feature and rely on the explicit CXX_STANDARD above.
foreach(_gtest_tgt GTest::gtest GTest::gtest_main)
  if(TARGET ${_gtest_tgt})
    set_target_properties(${_gtest_tgt} PROPERTIES INTERFACE_COMPILE_FEATURES "")
  endif()
endforeach()
target_link_libraries(test_sparse_cache_resize PRIVATE archer_core GTest::gtest_main)
target_include_directories(test_sparse_cache_resize PRIVATE ${CMAKE_SOURCE_DIR}/core)
# archer_core transitively pulls in Python/pybind headers via libtorch, so the
# test executable must link the Python runtime to resolve Py* symbols.
find_package(Python3 COMPONENTS Development REQUIRED)
target_link_libraries(test_sparse_cache_resize PRIVATE Python3::Python)
add_test(NAME sparse_cache_resize COMMAND test_sparse_cache_resize)

add_executable(test_topology_lifecycle
  test_topology_lifecycle.cpp
  ${CMAKE_SOURCE_DIR}/extensions/kernel/v4_fp4/fp8_dequant.cu
  ${CMAKE_SOURCE_DIR}/extensions/kernel/v4_fp4/mxfp4_dequant.cu
)
set_target_properties(test_topology_lifecycle PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
foreach(_gtest_tgt GTest::gtest GTest::gtest_main)
  if(TARGET ${_gtest_tgt})
    set_target_properties(${_gtest_tgt} PROPERTIES INTERFACE_COMPILE_FEATURES "")
  endif()
endforeach()
target_link_libraries(test_topology_lifecycle PRIVATE archer_core GTest::gtest_main)
target_include_directories(test_topology_lifecycle PRIVATE ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(test_topology_lifecycle PRIVATE Python3::Python)
add_test(NAME topology_lifecycle COMMAND test_topology_lifecycle)
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
