// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

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

#define CUDA_CHECK(call) \
  throwOnCudaError(call, __FILE__, __LINE__, __FUNCTION__, #call)

bool IsDevicePointer(const void* ptr);
int GetDeviceCount();
std::size_t GetTotalDeviceMemory(int device_id);
std::size_t GetFreeDeviceMemory(int device_id);

#define DEVICE_CACHE_LIMIT(gid) GetTotalDeviceMemory(gid) * 0.7
#define NUM_DEVICES GetDeviceCount()

int CudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
int CudaMemcpyAsync(void* dst, const void* src, size_t count,
                    cudaMemcpyKind kind, cudaStream_t stream = 0);
void BlockingCudaCopy(int device, void* dst, const void* src, size_t size,
                      cudaMemcpyKind kind, cudaStream_t stream);
