set(ARCHER_CORE_CXX_SOURCES
  # utils
  ${CMAKE_SOURCE_DIR}/core/utils/logger.cpp
  ${CMAKE_SOURCE_DIR}/core/utils/cuda_utils.cpp
  # model
  ${CMAKE_SOURCE_DIR}/core/model/model_topology.cpp
  ${CMAKE_SOURCE_DIR}/core/model/moe.cpp
  # prefetch
  ${CMAKE_SOURCE_DIR}/core/prefetch/archer_prefetch_handle.cpp
  ${CMAKE_SOURCE_DIR}/core/prefetch/expert_residency.cpp
  ${CMAKE_SOURCE_DIR}/core/prefetch/task_scheduler.cpp
  ${CMAKE_SOURCE_DIR}/core/prefetch/task_thread.cpp
  # memory
  ${CMAKE_SOURCE_DIR}/core/memory/caching_allocator.cpp
  ${CMAKE_SOURCE_DIR}/core/memory/memory_pool.cpp
  ${CMAKE_SOURCE_DIR}/core/memory/pinned_memory_pool.cpp
  ${CMAKE_SOURCE_DIR}/core/memory/kv_cache_pool.cpp
  ${CMAKE_SOURCE_DIR}/core/memory/stream_pool.cpp
  ${CMAKE_SOURCE_DIR}/core/memory/event_pool.cpp
  ${CMAKE_SOURCE_DIR}/core/memory/host_caching_allocator.cpp
  ${CMAKE_SOURCE_DIR}/core/memory/device_caching_allocator.cpp
  # parallel
  ${CMAKE_SOURCE_DIR}/core/parallel/expert_dispatcher.cpp
  ${CMAKE_SOURCE_DIR}/core/parallel/expert_module.cpp
  # aio
  ${CMAKE_SOURCE_DIR}/core/aio/archer_aio_thread.cpp
  ${CMAKE_SOURCE_DIR}/core/aio/archer_prio_aio_handle.cpp
  ${CMAKE_SOURCE_DIR}/core/aio/archer_aio_utils.cpp
  ${CMAKE_SOURCE_DIR}/core/aio/archer_aio_threadpool.cpp
  ${CMAKE_SOURCE_DIR}/core/aio/archer_tensor_handle.cpp
  ${CMAKE_SOURCE_DIR}/core/aio/archer_tensor_index.cpp
  # base (.cc extension)
  ${CMAKE_SOURCE_DIR}/core/base/thread.cc
  ${CMAKE_SOURCE_DIR}/core/base/exception.cc
  ${CMAKE_SOURCE_DIR}/core/base/date.cc
  ${CMAKE_SOURCE_DIR}/core/base/process_info.cc
  ${CMAKE_SOURCE_DIR}/core/base/logging.cc
  ${CMAKE_SOURCE_DIR}/core/base/log_file.cc
  ${CMAKE_SOURCE_DIR}/core/base/timestamp.cc
  ${CMAKE_SOURCE_DIR}/core/base/file_util.cc
  ${CMAKE_SOURCE_DIR}/core/base/countdown_latch.cc
  ${CMAKE_SOURCE_DIR}/core/base/timezone.cc
  ${CMAKE_SOURCE_DIR}/core/base/log_stream.cc
  ${CMAKE_SOURCE_DIR}/core/base/thread_pool.cc
)

set(ARCHER_CORE_CUDA_SOURCES
  ${CMAKE_SOURCE_DIR}/core/model/fused_mlp.cu
  ${CMAKE_SOURCE_DIR}/extensions/kernel/fused_moe_mlp.cu
  ${CMAKE_SOURCE_DIR}/extensions/kernel/activation_kernels.cu
  ${CMAKE_SOURCE_DIR}/extensions/kernel/topk_softmax_kernels.cu
)

add_library(archer_core STATIC
  ${ARCHER_CORE_CXX_SOURCES}
  ${ARCHER_CORE_CUDA_SOURCES}
)

if(MOE_INFINITY_TESTING)
  target_compile_definitions(archer_core PUBLIC MOE_INFINITY_TESTING=1)
endif()

target_include_directories(archer_core PUBLIC
  ${CMAKE_SOURCE_DIR}/core
  ${CMAKE_SOURCE_DIR}/extensions
  ${CUTLASS_DIR}/include
  ${CUTLASS_DIR}/tools/util/include
  ${TORCH_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  /usr/local/cuda/include
)

target_compile_options(archer_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-reorder -fPIC -fopenmp -march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3 --use_fast_math
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__
  >
)

target_link_libraries(archer_core PUBLIC
  ${TORCH_LIBRARIES}
  OpenMP::OpenMP_CXX
  uuid cublas cudart cuda pthread
)

target_link_directories(archer_core PUBLIC /usr/local/cuda/lib64)
set_target_properties(archer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
