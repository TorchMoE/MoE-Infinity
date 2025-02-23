// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "cuda_utils.h"
#include "logger.h"

int kNumDevices = GetDeviceCount();

bool IsDevicePointer(const void* ptr) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    DLOG_ERROR("cudaPointerGetAttributes failed: ", cudaGetErrorString(err));
    return false;
  }
  return attr.type == cudaMemoryTypeDevice;
}

int GetDeviceCount() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  return device_count;
}

int GetDevice() {
  int device_id;
  cudaGetDevice(&device_id);
  return device_id;
}

std::size_t GetTotalDeviceMemory(int device_id) {
  size_t free_memory, total_memory;
  cudaSetDevice(device_id);
  cudaMemGetInfo(&free_memory, &total_memory);
  return total_memory;
}

std::size_t GetFreeDeviceMemory(int device_id) {
  size_t free_memory, total_memory;
  cudaSetDevice(device_id);
  cudaMemGetInfo(&free_memory, &total_memory);
  return free_memory;
}

int CudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  return cudaMemcpy(dst, src, count, kind);
}

int CudaMemcpyAsync(void* dst, const void* src, size_t count,
                    cudaMemcpyKind kind, cudaStream_t stream) {
  return cudaMemcpyAsync(dst, src, count, kind, stream);
}

void BlockingCudaCopy(int device, void* dst, const void* src, size_t size,
                      cudaMemcpyKind kind, cudaStream_t stream) {
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}
