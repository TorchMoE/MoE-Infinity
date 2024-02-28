// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once
#include "common/pytorch.h"
#include "utils/noncopyable.h"

#include <c10/core/CPUAllocator.h>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

std::size_t GetTotalSystemMemory();

#ifndef DEVICE_MEMORY_RATIO
#define DEVICE_MEMORY_RATIO 0.8
#endif

#ifndef HOST_MEMORY_RATIO
#define HOST_MEMORY_RATIO 0.8
#endif

class HostMemoryPool : public noncopyable {
public:
    void* AllocateMemory(const std::size_t key,
                         const std::int64_t size,
                         const torch::Device& device);
    int FreeMemory(const std::size_t key,
                   void* data,
                   const std::int64_t size,
                   const torch::Device& device);
    std::int64_t GetFreeMemory();
    std::int64_t GetMemoryCapacity();

    HostMemoryPool();
    virtual ~HostMemoryPool() = default;

private:
    std::unordered_map<std::uint64_t, void*> allocated_id_;
    std::int64_t free_memory_;
    std::int64_t memory_capacity_;
    std::mutex mutex_;
};

class DeviceMemoryPool : public noncopyable {
public:
    void* AllocateMemory(const std::size_t key,
                         const std::int64_t size,
                         const torch::Device& device);
    int FreeMemory(const std::size_t key,
                   void* data,
                   const std::int64_t size,
                   const torch::Device& device);

    void SetMemoryRatio(const double ratio);
    std::int64_t GetFreeMemory(const torch::Device& device);
    std::int64_t GetMemoryCapacity(const torch::Device& device);

    DeviceMemoryPool();
    virtual ~DeviceMemoryPool() = default;

private:
    std::vector<std::unordered_map<std::uint64_t, void*>> allocated_id_;
    std::vector<std::int64_t> free_memory_;
    std::vector<std::int64_t> memory_capacity_;
    std::mutex mutex_;
};

extern std::unique_ptr<HostMemoryPool> kHostMemoryPool;
extern std::unique_ptr<DeviceMemoryPool> kDeviceMemoryPool;
