#pragma once

#include <atomic>
#include <memory>
#include <iostream>
#include <thread>
#include <vector>

#include "base/noncopyable.h"

template <typename T>
class LockFreeQueue : public base::noncopyable {
 public:
  LockFreeQueue() {
    Node* dummy = new Node();  // Dummy node
    head_.store(dummy);
    tail_.store(dummy);
  }

  ~LockFreeQueue() {
    while (Node* old_head = head_.load()) {
      head_.store(old_head->next);
      delete old_head;
    }
  }

  void Push(T& value) {
    std::shared_ptr<T> new_data = std::make_shared<T>(std::move(value));
    Node* new_node = new Node();
    new_node->data = new_data;

    do {
      Node* old_tail = tail_.load();
      Node* next = old_tail->next;
      if (old_tail == tail_.load()) {
        if (next == nullptr) {
          if (old_tail->next.compare_exchange_weak(next, new_node)) {
            tail_.compare_exchange_weak(old_tail, new_node);
            return;
          }
        } else {
          tail_.compare_exchange_weak(old_tail, next);
        }
      }
    } while (true);
  }

  bool Pop(T& value) {
    Node* old_head;

    do {
      old_head = head_.load();  // Read current head
      Node* next = old_head->next.load(std::memory_order_acquire);
      if (old_head == tail_.load(std::memory_order_acquire)) {
        return false;  // Queue is empty
      }

    } while (!head_.compare_exchange_weak(old_head, old_head->next,
                                          std::memory_order_release));

    value = *(old_head->next.load()->data);  // Read value
    delete old_head;                         // Free old node
    return true;
  }

  bool Empty() const { return head_.load() == tail_.load(); }

  bool Full() const {
    return false;  // Queue is unbounded
  }

 protected:
  struct Node {
    std::shared_ptr<T> data;
    std::atomic<Node*> next;

    Node() : next(nullptr) {}
  };

  std::atomic<Node*> head_;
  std::atomic<Node*> tail_;
};

template <typename T>
class LockFreeRecyclingQueue : public LockFreeQueue<T> {
 public:
  LockFreeRecyclingQueue() = default;

  void Pop(T& item) override {
    LockFreeQueue<T>::Pop(item);
    Push(item);
  }

  bool TryPop(T& item) override {
    bool success = LockFreeQueue<T>::TryPop(item);
    Push(item);
    return success;
  }
};
