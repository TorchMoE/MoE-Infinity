// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "memory_pool.h"

#include "common/types.h"
#include "host_caching_allocator.h"
#include "memory/stream_pool.h"
#include "utils/archer_logger.h"
#include "utils/cuda_utils.h"

#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <sys/sysinfo.h>
#include <unistd.h>

std::unique_ptr<HostMemoryPool> kHostMemoryPool = std::make_unique<HostMemoryPool>();
std::unique_ptr<DeviceMemoryPool> kDeviceMemoryPool = std::make_unique<DeviceMemoryPool>();

std::size_t GetTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}

void* HostMemoryPool::AllocateMemory(const std::size_t key,
                                     const std::int64_t size,
                                     const torch::Device& device)
{
    assert(device.is_cpu());
    std::unique_lock<std::mutex> lock(mutex_);
    if (allocated_id_.find(key) != allocated_id_.end()) {
        ARCHER_LOG_ERROR("PreAllocateMemory failed, already allocated ", key);
        return allocated_id_[key];
    }
    auto allocator = c10::HostCachingAllocator::get();
    auto data_ptr = allocator->allocate(size);
    allocated_id_.insert(std::make_pair(key, data_ptr));
    return data_ptr;
}

int HostMemoryPool::FreeMemory(const std::size_t key,
                               void* data,
                               const std::int64_t size,
                               const torch::Device& device)
{
    assert(device.is_cpu());
    std::unique_lock<std::mutex> lock(mutex_);
    if (allocated_id_.find(key) == allocated_id_.end()) {
        ARCHER_LOG_ERROR("FreeMemory failed, not found ", key);
        return -1;
    }
    allocated_id_.erase(key);
    if (data != nullptr) {
        auto allocator = c10::HostCachingAllocator::get();
        allocator->free(data);
    }  // pinned_mr_->raw_deallocate(data);
    free_memory_ += size;
    return 0;
}

HostMemoryPool::HostMemoryPool()
    : free_memory_(
#ifdef TEST_LIMIT_MEMORY
          10LL * 1024 * 1024 * 1024
#else
          GetTotalSystemMemory() * HOST_MEMORY_RATIO
#endif
      )
{
    auto pinned_mr_ = c10::HostCachingAllocator::get();
    if (pinned_mr_ == nullptr) {
        ARCHER_LOG_ERROR("GetHostAllocator failed");
        exit(-1);
    }
    memory_capacity_ = free_memory_;
}

std::int64_t HostMemoryPool::GetFreeMemory()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return free_memory_;
}

std::int64_t HostMemoryPool::GetMemoryCapacity() { return memory_capacity_; }

void* DeviceMemoryPool::AllocateMemory(const std::size_t key,
                                       const std::int64_t size,
                                       const torch::Device& device)
{
    int device_id = device.index();
    std::unique_lock<std::mutex> lock(mutex_);
    if (allocated_id_[device_id].find(key) != allocated_id_[device_id].end()) {
        ARCHER_LOG_ERROR("PreAllocateMemory failed, already allocated ", key);
        return allocated_id_[device_id][key];
    }
    cudaSetDevice(device_id);
    at::Allocator* allocator = c10::cuda::CUDACachingAllocator::get();
    auto data_ptr = allocator->raw_allocate(size);
    free_memory_[device_id] -= size;
    allocated_id_[device_id].insert(std::make_pair(key, data_ptr));
    return data_ptr;
}

int DeviceMemoryPool::FreeMemory(const std::size_t key,
                                 void* data,
                                 const std::int64_t size,
                                 const torch::Device& device)
{
    int device_id = device.index();
    std::unique_lock<std::mutex> lock(mutex_);
    if (allocated_id_[device_id].find(key) == allocated_id_[device_id].end()) {
        ARCHER_LOG_ERROR("FreeMemory failed, not found ", key);
        return -1;
    }
    allocated_id_[device_id].erase(key);
    if (data != nullptr) {
        at::Allocator* allocator = c10::cuda::CUDACachingAllocator::get();
        allocator->raw_deallocate(data);
    }
    free_memory_[device_id] += size;
    return 0;
}

DeviceMemoryPool::DeviceMemoryPool()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    c10::cuda::CUDACachingAllocator::init(device_count);

    for (int i = 0; i < device_count; ++i) {
        std::unordered_map<std::uint64_t, void*> allocated_id;
        allocated_id_.emplace_back(allocated_id);
        free_memory_.emplace_back(GetTotalDeviceMemory(i));
        memory_capacity_.emplace_back(free_memory_[i]);
    }
}

std::int64_t DeviceMemoryPool::GetFreeMemory(const torch::Device& device)
{
    int device_id = device.index();
    std::lock_guard<std::mutex> lock(mutex_);
    return free_memory_[device_id];
}

std::int64_t DeviceMemoryPool::GetMemoryCapacity(const torch::Device& device)
{
    int device_id = device.index();
    return memory_capacity_[device_id];
}

void DeviceMemoryPool::SetMemoryRatio(const double ratio)
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    for (int i = 0; i < device_count; ++i) {
        free_memory_[i] = GetTotalDeviceMemory(i) * ratio;
        memory_capacity_[i] = free_memory_[i];
    }
}
