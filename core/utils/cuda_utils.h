// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#include <cstdint>
#include <memory>
#include <sstream>
#include <iostream>

inline void throwOnCudaError(cudaError_t error, const char* file, int line,
                             const char* function, const char* call) {
  if (error != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error " << error << " at " << file << ":" << line
       << " in function " << function << ": " << cudaGetErrorString(error)
       << "\nCall: " << call;
    throw std::runtime_error(ss.str());
  }
};

inline void throwOnCutlassError(cutlass::Status status, const char* file,
                                int line, const char* function,
                                const char* call) {
  if (status != cutlass::Status::kSuccess) {
    std::stringstream ss;
    ss << "CUTLASS error " << static_cast<int>(status) << " at " << file << ":"
       << line << " in function " << function << ": "
       << cutlassGetStatusString(status) << "\nCall: " << call;
    throw std::runtime_error(ss.str());
  }
}

#define CUDA_CHECK(call) \
  throwOnCudaError(call, __FILE__, __LINE__, __FUNCTION__, #call)

#define CUTLASS_CHECK(call) \
  throwOnCutlassError(call, __FILE__, __LINE__, __FUNCTION__, #call)

/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU
 * stream
 */
struct GpuTimer {
  cudaStream_t _stream_id;
  cudaEvent_t _start;
  cudaEvent_t _stop;

  /// Constructor
  GpuTimer() : _stream_id(0) {
    CUDA_CHECK(cudaEventCreate(&_start));
    CUDA_CHECK(cudaEventCreate(&_stop));
  }

  /// Destructor
  ~GpuTimer() {
    CUDA_CHECK(cudaEventDestroy(_start));
    CUDA_CHECK(cudaEventDestroy(_stop));
  }

  /// Start the timer for a given stream (defaults to the default stream)
  void start(cudaStream_t stream_id = 0) {
    _stream_id = stream_id;
    CUDA_CHECK(cudaEventRecord(_start, _stream_id));
  }

  /// Stop the timer
  void stop() { CUDA_CHECK(cudaEventRecord(_stop, _stream_id)); }

  /// Timing-only API. Blocks the calling CPU thread until `_stop` completes.
  /// Forbidden in ExpertDispatcher and MoEMLP production paths.
  float elapsed_millis_blocking() {
    float elapsed = 0.0;
    CUDA_CHECK(cudaEventSynchronize(_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
    return elapsed;
  }
};

int GetDevice();
bool IsDevicePointer(const void* ptr);
int GetDeviceCount();
std::size_t GetTotalDeviceMemory(int device_id);
std::size_t GetFreeDeviceMemory(int device_id);

#define DEVICE_CACHE_LIMIT(gid) GetTotalDeviceMemory(gid) * 0.7
int kNumDevices();

int CudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
int CudaMemcpyAsync(void* dst, const void* src, size_t count,
                    cudaMemcpyKind kind, cudaStream_t stream = 0);
// Explicitly blocking copy for initialization/debug paths only. Never call from
// ExpertDispatcher, GPUFetchFunc, GPUExecFunc, OutputFunc, or MoEMLP::forward.
void BlockingCudaCopy(int device, void* dst, const void* src, size_t size,
                      cudaMemcpyKind kind, cudaStream_t stream);

struct CUDADeviceAllocator {
  void* operator()(std::size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
  }
};

struct CUDADeviceDeleter {
  void operator()(void* ptr) { CUDA_CHECK(cudaFree(ptr)); }
};

struct CUDAHostAllocator {
  void* operator()(std::size_t size) {
    void* ptr;
    CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    return ptr;
  }
};

struct CUDAHostDeleter {
  void operator()(void* ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }
};
