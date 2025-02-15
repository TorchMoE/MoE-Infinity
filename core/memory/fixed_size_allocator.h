#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <memory>

#include "base/noncopyable.h"
#include "utils/logger.h"
#include "utils/cuda_utils.h"
#include "common/types.h"

template <typename T, typename Allocator = std::allocator<T>,
          typename Deleter = std::default_delete<T>>
class FixedSizeAllocator : public base::noncopyable {
 public:
  // Constructor - Allocate memory using custom allocator
  explicit FixedSizeAllocator(int num_slots, size_t slot_size)
      : num_slots_(num_slots), slot_size_(slot_size) {
    if (slot_size_ == 0) {
      DLOG_WARN << "slot size is 0 in FixedSizeAllocator";
      return;
    }

    // LCM of slot_size and 2MB to be chunk size
    chunk_size_ = 2 * 1024 * 1024;
    if (slot_size_ % chunk_size_ != 0) {
      chunk_size_ = std::lcm(slot_size_, chunk_size_);
    }

    int num_chunks = num_slots_ * slot_size_ / chunk_size_;
    for (int i = 0; i < num_chunks; ++i) {
      void* raw_ptr = allocator(chunk_size_);
      if (raw_ptr == nullptr) {
        DLOG_FATAL << "Failed to allocate memory in FixedSizeAllocator";
      }
      std::unique_ptr<T, Deleter> ptr(nullptr);
      ptr.reset(reinterpret_cast<T*>(raw_ptr));
      chunks_.push_back(std::move(ptr));
    }

    for (int i = 0; i < num_slots_; ++i) {
      int j = i * slot_size_ / chunk_size_;
      void* raw_ptr = chunks_[j].get() + (i * slot_size_ % chunk_size_);
      slot_map_[reinterpret_cast<char*>(raw_ptr)] = false;
    }

    DLOG_INFO << "FixedSizeAllocator created: num_slots=" << num_slots
              << ", slot_size=" << slot_size << ", chunk_size=" << chunk_size_;
  }

  // Access underlying pointer
  // T* get() const { return ptr.get(); }
  T* get_slot() const {
    for (auto& pair : slot_map_) {
      if (!pair.second) {
        pair.second = true;
        return reinterpret_cast<T*>(pair.first);
      }
    }
    // DLOG_WARN << "No empty slot in FixedSizeAllocator";
    return nullptr;
  }
  void release_slot(T* slot) {
    if (slot == nullptr) {
      // DLOG_WARN << "Invalid slot in FixedSizeAllocator";
      return;
    }
    if (slot_map_.find(slot) == slot_map_.end()) {
      DLOG_FATAL << "Invalid slot in FixedSizeAllocator";
    }
    slot_map_[reinterpret_cast<char*>(slot)] = false;
  }

 private:
  std::vector<std::unique_ptr<T, Deleter>> chunks_;  // The allocated memory
  Allocator allocator;                               // Custom allocator
  std::unordered_map<void*, bool> slot_map_;
  int num_slots_;
  size_t slot_size_;
  size_t chunk_size_;
};

typedef FixedSizeAllocator<void, CUDADeviceAllocator, CUDADeviceDeleter>
    CUDADeviceFixedSizeAllocator;
typedef FixedSizeAllocator<void, CUDAHostAllocator, CUDAHostDeleter>
    CUDAHostFixedSizeAllocator;
