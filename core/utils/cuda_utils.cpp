// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "cuda_utils.h"
#include "archer_logger.h"

bool IsDevicePointer(const void* ptr)
{
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        ARCHER_LOG_ERROR("cudaPointerGetAttributes failed: ", cudaGetErrorString(err));
        return false;
    }
    return attr.type == cudaMemoryTypeDevice;
}

int GetDeviceCount()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

std::size_t GetTotalDeviceMemory(int device_id)
{
    size_t free_memory, total_memory;
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_memory, &total_memory);
    return total_memory;
}

std::size_t GetFreeDeviceMemory(int device_id)
{
    size_t free_memory, total_memory;
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_memory, &total_memory);
    return free_memory;
}
