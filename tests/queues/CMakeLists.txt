cmake_minimum_required(VERSION 3.10)
project(LockFreeQueueTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add GoogleTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include directories
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/../../core)

# Add test executables
add_executable(test_lockfree_queue ../test_lockfree_queue.cpp)
add_executable(test_threadsafe_queue ../test_threadsafe_queue.cpp)

# Link GoogleTest and pthread
target_link_libraries(test_lockfree_queue ${GTEST_LIBRARIES} pthread)
target_link_libraries(test_threadsafe_queue ${GTEST_LIBRARIES} pthread)
