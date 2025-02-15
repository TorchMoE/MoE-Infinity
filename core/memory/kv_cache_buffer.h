#pragma once

#include "fixed_size_allocator.h"
#include "base/noncopyable.h"
#include "common/pytorch.h"

template <typename Allocator = std::allocator<T>,
          typename Deleter = std::default_delete<T>>
class KVCacheBuffer : public base::noncopyable {
  using ThisAllocator = FixedSizeAllocator<void, Allocator, Deleter>;

 public:
  explicit KVCacheBuffer(size_t context_length, std::vector<int64_t> key_shape,
                         std::vector<int64_t> value_shape, int torch_dtype)
      : key_allocator_(nullptr), value_allocator_(nullptr) {
    size_t key_size = torch_shape_size(key_shape, torch_dtype);
    size_t value_size = torch_shape_size(value_shape, torch_dtype);

    key_allocator_ = std::make_unique<ThisAllocator>(context_length, key_size);
    value_allocator_ =
        std::make_unique<ThisAllocator>(context_length, value_size);
  }

  std::tuple<void*, void*> get_slot(int layer_id, int microbatch_id) {
    uint64_t key = (static_cast<uint64_t>(layer_id) << 32) | microbatch_id;
    std::unique_lock<std::mutex> lock(mutex_);
    if (kv_buffer_map_.find(key) == kv_buffer_map_.end()) {
      void* key_ptr = key_allocator_->get_slot();
      void* value_ptr = value_allocator_->get_slot();

      if (key_ptr == nullptr || value_ptr == nullptr) {
        DLOG_FATAL << "No empty slot in KVCacheBuffer, key_ptr: " << key_ptr
                   << ", value_ptr: " << value_ptr;
        return std::make_tuple(nullptr, nullptr);
      }

      kv_buffer_map_[key] = std::make_tuple(key_ptr, value_ptr);
    }
    kv_in_buffer_[key] = false;
    return kv_buffer_map_[key];
  }

  void set_slot(int layer_id, int microbatch_id) {
    uint64_t key = (static_cast<uint64_t>(layer_id) << 32) | microbatch_id;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      kv_in_buffer_[key] = true;
    }
    cv_.notify_all();
  }

  void wait_slot(int layer_id, int microbatch_id) {
    uint64_t key = (static_cast<uint64_t>(layer_id) << 32) | microbatch_id;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, key] {
      return kv_in_buffer_.find(key) != kv_in_buffer_.end() &&
             kv_in_buffer_[key];
    });
  }

  void release_slot(int layer_id, int microbatch_id) {
    uint64_t key = (static_cast<uint64_t>(layer_id) << 32) | microbatch_id;
    std::lock_guard<std::mutex> lock(mutex_);
    kv_buffer_map_.erase(key);
    kv_in_buffer_[key] = false;
  }

 private:
  std::unordered_map<uint64_t, std::tuple<void*, void*>> kv_buffer_map_;
  std::unordered_map<uint64_t, bool> kv_in_buffer_;
  std::mutex mutex_;
  std::condition_variable cv_;

  std::unique_ptr<ThisAllocator> key_allocator_;
  std::unique_ptr<ThisAllocator> value_allocator_;
};
