#pragma once

#include "fixed_size_allocator.h"
#include "base/noncopyable.h"
#include "common/pytorch.h"

template <typename Allocator = std::allocator<T>,
          typename Deleter = std::default_delete<T>>
class WeightsBuffer : public base::noncopyable {
  using ThisAllocator = FixedSizeAllocator<void, Allocator, Deleter>;

 public:
  explicit WeightsBuffers(size_t num_buffers,
                          std::vector<std::vector<int64_t>> shapes,
                          int torch_dtype)
      : allocator_(nullptr) {
    size_t size = 0;
    for (const auto& shape : shapes) {
      size += torch_shape_size(shape, torch_dtype);
    }

    allocator_ = std::make_unique<ThisAllocator>(num_buffers, size);
  }

  void* get_slot(int layer_id, int expert_id) {
    uint64_t key = (static_cast<uint64_t>(layer_id) << 32) | expert_id;
    std::unique_lock<std::mutex> lock(mutex_);
    if (buffer_map_.find(key) == buffer_map_.end()) {
      void* buffer = allocator_->get_slot();
      if (buffer == nullptr) {
        // DLOG_FATAL << "No empty slot in WeightsBuffer, buffer: " << buffer;
        return nullptr;
      }
      buffer_map_[key] = buffer;
    }
    weight_in_buffer_[key] = false;
    return buffer_map_[key];
  }

  void set_slot(int layer_id, int expert_id) {
    uint64_t key = (static_cast<uint64_t>(layer_id) << 32) | expert_id;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      weight_in_buffer_[key] = true;
    }
    cv_.notify_all();
  }

  void wait_slot(int layer_id, int expert_id) {
    uint64_t key = (static_cast<uint64_t>(layer_id) << 32) | expert_id;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, key] {
      return buffer_map_.find(key) != buffer_map_.end() &&
             weight_in_buffer_[key];
    });
  }

  void release_slot((int layer_id, int expert_id)) {
    uint64_t key = (static_cast<uint64_t>(layer_id) << 32) | expert_id;
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_map_.erase(key);
    weight_in_buffer_[key] = false;
  }

 private:
  std::unordered_map<uint64_t, void*> buffer_map_;
  std::unordered_map<uint64_t, bool> weight_in_buffer_;
  std::mutex mutex_;
  std::condition_variable cv_;

  ThisAllocator allocator_;
};
