//===- c10/mobile/CPUCachingAllocator.cpp ----------------===//
//
// Part of the Pytorch Project, under the BSD 3-Clause License.
// See https://github.com/pytorch/pytorch/blob/main/LICENSE for license information.
// SPDX-License-Identifier: BSD 3-Clause

// MoE-Infinity: modified from c10::CPUCachingAllocator.
// replaced c10::alloc_cpu with cudaHostAlloc

#include "host_caching_allocator.h"
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

namespace c10 {
namespace HostCachingAllocator {

std::mutex HostCachingAllocator::mutex_;
ska::flat_hash_map<void*, size_t> HostCachingAllocator::allocation_map_;

inline void* HostCachingAllocator::allocate_and_cache(const size_t bytes)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    void* ptr;
    auto cuda_err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (cuda_err != cudaSuccess) {
        free_cached();
        cuda_err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
        if (cuda_err != cudaSuccess) { throw std::runtime_error("cudaHostAlloc failed"); }
    }

    allocation_map_[ptr] = bytes;
    return ptr;
}

void* HostCachingAllocator::allocate(const size_t bytes)
{
    std::lock_guard<std::mutex> guard(mutex_);
    const auto& it = available_map_.find(bytes);
    if (it == available_map_.end() || it->second.empty()) { return allocate_and_cache(bytes); }
    return it->second.pop_back_val();
}

void HostCachingAllocator::free(void* ptr)
{
    // NB: since we are not really freeing the memory
    // the cases such as quantization code freeing original weights
    // on mobile, will not quite work, as we likely will hold
    // onto that memory.
    // NB: We can also enable max memory cached for better memory
    // management such that free will actually free the memory if
    // we are nearing or above the watermark.
    std::lock_guard<std::mutex> guard(mutex_);
    // If this allocation was done before caching allocator was enabled
    // then free regularly
    const auto& it = allocation_map_.find(ptr);
    if (it == allocation_map_.end()) {
        // c10::free_cpu(ptr);
        cudaFreeHost(ptr);
        return;
    }
    const size_t alloc_size = it->second;
    available_map_[alloc_size].push_back(ptr);
}

void HostCachingAllocator::record_free(void* ptr)
{
    // This function captures the case when the allocated memory
    // is being freed outside the scope of this allocator.
    // At the moment only way to capture this is to have the allocator,
    // that uses this CachingAllocator as the backing allocator,
    // call this function explicitly upon freeing memory while
    // outside the scope of caching allocator.
    // If the memory is freed in some other way, then we will likely
    // have undefined behavior or page fault. But this can be
    // the case without caching allocator as well.
    std::lock_guard<std::mutex> guard(mutex_);
    const auto& it = allocation_map_.find(ptr);
    if (it != allocation_map_.end()) { allocation_map_.erase(it); }
}

void HostCachingAllocator::free_cached()
{
    for (const auto& it : available_map_) {
        for (const auto ptr : it.second) {
            //   c10::free_cpu(ptr);
            cudaFreeHost(ptr);
            // When cached memory is return to OS, it must be removed
            // from allocation_map.
            allocation_map_.erase(ptr);
        }
    }
    available_map_.clear();
}

HostCachingAllocator::~HostCachingAllocator() { free_cached(); }

HostCachingAllocator* caching_allocator = new HostCachingAllocator();

HostCachingAllocator* get()
{
    if (caching_allocator == nullptr) { caching_allocator = new HostCachingAllocator(); }
    return caching_allocator;
}

}  // namespace HostCachingAllocator

}  // namespace c10
