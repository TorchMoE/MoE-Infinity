#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

#include "base/noncopyable.h"

template <typename T>
class ThreadSafeQueue : public base::noncopyable {
 public:
  ThreadSafeQueue() = default;

  // Disable copy constructor and assignment to avoid accidental data races.
  ThreadSafeQueue(const ThreadSafeQueue&) = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

  // Pushes an item into the queue (thread-safe).
  void Push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(std::move(item));
    cond_.notify_one();
  }

  // Pops an item from the queue (blocking).
  bool Pop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty() || stop_flag_; });

    if (stop_flag_ && queue_.empty()) {
      return false;  // Stop requested and queue is empty.
    }

    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  // Tries to pop an item without blocking. Returns false if the queue is empty.
  bool TryPop(T& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  // Returns true if the queue is empty.
  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  // Stops the queue (for graceful shutdown).
  void Stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_flag_ = true;
    cond_.notify_all();
  }

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  bool stop_flag_ = false;
};
