//===- c10/mobile/CPUCachingAllocator.cpp ----------------===//
//
// Part of the Pytorch Project, under the BSD 3-Clause License.
// See https://github.com/pytorch/pytorch/blob/main/LICENSE for license
// information. SPDX-License-Identifier: BSD 3-Clause

// MoE-Infinity: modified from c10::CPUCachingAllocator.
// replaced c10::alloc_cpu with cudaDeviceAlloc

#include "device_caching_allocator.h"
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include "utils/logger.h"

namespace c10 {
namespace DeviceCachingAllocator {

std::mutex DeviceCachingAllocator::mutex_;
ska::flat_hash_map<void*, size_t> DeviceCachingAllocator::allocation_map_;

inline void* DeviceCachingAllocator::allocate_and_cache(const size_t bytes) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* ptr;
  auto cuda_err = cudaMalloc(&ptr, bytes);
  if (cuda_err != cudaSuccess) {
    free_cached();
    cuda_err = cudaMalloc(&ptr, bytes);
    if (cuda_err != cudaSuccess) {
      DLOG_ERROR("cudaMalloc failed", bytes, cuda_err);
      throw std::runtime_error("cudaMalloc failed");
    }
  }

  allocation_map_[ptr] = bytes;
  return ptr;
}

void* DeviceCachingAllocator::allocate(const size_t bytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto& it = available_map_.find(bytes);
  if (it == available_map_.end() || it->second.empty()) {
    return allocate_and_cache(bytes);
  }
  return it->second.pop_back_val();
}

void DeviceCachingAllocator::free(void* ptr) {
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
    cudaFree(ptr);
    return;
  }
  const size_t alloc_size = it->second;
  available_map_[alloc_size].push_back(ptr);
}

void DeviceCachingAllocator::record_free(void* ptr) {
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
  if (it != allocation_map_.end()) {
    allocation_map_.erase(it);
  }
}

void DeviceCachingAllocator::free_cached() {
  for (const auto& it : available_map_) {
    for (const auto ptr : it.second) {
      //   c10::free_cpu(ptr);
      cudaFree(ptr);
      // When cached memory is return to OS, it must be removed
      // from allocation_map.
      allocation_map_.erase(ptr);
    }
  }
  available_map_.clear();
}

DeviceCachingAllocator::~DeviceCachingAllocator() { free_cached(); }

std::array<DeviceCachingAllocator*, 8> caching_allocators;

DeviceCachingAllocator* get(int device_id) {
  if (device_id < 0 || device_id >= 8) {
    throw std::runtime_error("Invalid device_id");
  }
  if (caching_allocators[device_id] == nullptr) {
    caching_allocators[device_id] = new DeviceCachingAllocator();
  }
  return caching_allocators[device_id];
}

}  // namespace DeviceCachingAllocator

}  // namespace c10
