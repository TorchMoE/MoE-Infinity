add_subdirectory(test_io)

# libtorch_python.so provides Python-facing torch types (e.g. THPDeviceType)
# required for the Python extension to import correctly.
find_library(torch_python_LIBRARY torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib")

Python3_add_library(prefetch_op MODULE WITH_SOABI
  ${CMAKE_SOURCE_DIR}/core/python/py_archer_prefetch.cpp
)

# --whole-archive forces all object files from the static lib into the shared
# library, preventing "undefined symbol" errors at Python import time.
# archer_core is listed twice: once wrapped with --whole-archive (for complete
# symbol inclusion), and once as a CMake target (to propagate PUBLIC include
# dirs and transitive link libraries). The second occurrence is a linker no-op.
target_link_libraries(prefetch_op PRIVATE
  -Wl,--whole-archive $<TARGET_FILE:archer_core> -Wl,--no-whole-archive
  archer_core
  ${torch_python_LIBRARY})

# TORCH_EXTENSION_NAME is normally injected by setuptools; define it here for CMake.
target_compile_definitions(prefetch_op PRIVATE TORCH_EXTENSION_NAME=prefetch_op)
if(MOE_INFINITY_TESTING)
  target_compile_definitions(prefetch_op PRIVATE MOE_INFINITY_TESTING=1)
endif()

target_compile_options(prefetch_op PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-reorder -fPIC -fopenmp -march=native>
)

target_include_directories(prefetch_op PRIVATE
  ${CMAKE_SOURCE_DIR}/extensions
)

install(TARGETS prefetch_op
  LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/moe_infinity/ops/prefetch"
  COMPONENT Runtime
)
