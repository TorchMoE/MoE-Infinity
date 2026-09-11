// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <deque>
#include <exception>
#include <iostream>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/noncopyable.h"
#include "common/pytorch.h"
#include "model/model_topology.h"

#define SKIP_TO_NEXT_ITERATION                                \
  std::this_thread::sleep_for(std::chrono::microseconds(10)); \
  continue;

#define NUM_PRIORITY 20UL

// Priority bands, lowest value serviced first (GPUThreadFunc scans queue 0 up).
// Band 0 is also the on-demand queue: dedup sweeps skip index 0, so route-ahead
// work parked at 0 keeps on-demand's non-preemptible semantics.
constexpr std::uint32_t kOnDemandPriority = 0;
constexpr std::uint32_t kRouteAheadPriority = 1;
constexpr std::uint32_t kBackgroundPrefetchPriority = 2;

enum class RemovalReason {
  kClear,
  kReplaceCandidates,
  kFetchSweep,
  kDeduplicate,
  kObsoleteLayer,
  kExplicitCancel,
  kPopDuplicate,
  kShutdown,
};

enum class RunningOutcome {
  kCompleted,
  kEvictionFailed,
  kStateConflict,
  kAlreadyResident,
  kTransferFailed,
};

// Deterministic, GPU-free byte accounting for the speculative prefetch queue.
// Tracks each admitted task through queued -> running -> terminal transitions
// so every removal path
// (clear/replace/sweep/dedup/obsolete/cancel/pop/shutdown) and every post-pop
// outcome retires its bytes exactly once. Value type with no NodePtr
// dependency, so it is unit-testable without CUDA or the topology.
class PrefetchQueueAccounting {
 public:
  enum class State { kQueued, kRunning, kTerminal };

  bool TryAdmit(std::uint32_t tensor_id, std::int64_t bytes,
                std::int64_t max_inflight_bytes) {
    auto it = tasks_.find(tensor_id);
    if (it != tasks_.end()) return false;
    if (inflight_bytes_ + bytes > max_inflight_bytes) {
      rejected_bytes_ += bytes;
      return false;
    }
    tasks_.emplace(tensor_id, Entry{bytes, State::kQueued});
    queued_bytes_ += bytes;
    inflight_bytes_ += bytes;
    accepted_bytes_ += bytes;
    return true;
  }

  bool MarkStarted(std::uint32_t tensor_id) {
    auto it = tasks_.find(tensor_id);
    if (it == tasks_.end() || it->second.state != State::kQueued) return false;
    it->second.state = State::kRunning;
    queued_bytes_ -= it->second.bytes;
    running_bytes_ += it->second.bytes;
    return true;
  }

  std::int64_t CancelQueued(std::uint32_t tensor_id) {
    return RetireQueued(tensor_id, RemovalReason::kExplicitCancel);
  }

  std::int64_t RetireQueued(std::uint32_t tensor_id, RemovalReason reason) {
    auto it = tasks_.find(tensor_id);
    if (it == tasks_.end() || it->second.state != State::kQueued) return 0;
    std::int64_t bytes = it->second.bytes;
    queued_bytes_ -= bytes;
    inflight_bytes_ -= bytes;
    canceled_bytes_ += bytes;
    removed_by_reason_[static_cast<int>(reason)] += bytes;
    it->second.state = State::kTerminal;
    return bytes;
  }

  std::int64_t RetireRunning(std::uint32_t tensor_id, RunningOutcome outcome,
                             std::int64_t completed_bytes) {
    auto it = tasks_.find(tensor_id);
    if (it == tasks_.end() || it->second.state != State::kRunning) return 0;
    std::int64_t bytes = it->second.bytes;
    running_bytes_ -= bytes;
    inflight_bytes_ -= bytes;
    if (outcome == RunningOutcome::kCompleted) {
      completed_bytes_ += bytes;
    } else if (outcome == RunningOutcome::kAlreadyResident) {
      already_resident_bytes_ += bytes;
    } else {
      failed_bytes_ += bytes;
    }
    (void)completed_bytes;
    it->second.state = State::kTerminal;
    return bytes;
  }

  void MarkCompleted(std::uint32_t tensor_id) {
    auto it = tasks_.find(tensor_id);
    if (it == tasks_.end()) return;
    if (it->second.state == State::kRunning) {
      RetireRunning(tensor_id, RunningOutcome::kCompleted, it->second.bytes);
    } else if (it->second.state == State::kQueued) {
      std::int64_t bytes = it->second.bytes;
      queued_bytes_ -= bytes;
      inflight_bytes_ -= bytes;
      completed_bytes_ += bytes;
      it->second.state = State::kTerminal;
    }
  }

  void MarkFailed(std::uint32_t tensor_id) {
    auto it = tasks_.find(tensor_id);
    if (it == tasks_.end()) return;
    if (it->second.state == State::kRunning) {
      RetireRunning(tensor_id, RunningOutcome::kTransferFailed, 0);
    } else if (it->second.state == State::kQueued) {
      std::int64_t bytes = it->second.bytes;
      queued_bytes_ -= bytes;
      inflight_bytes_ -= bytes;
      failed_bytes_ += bytes;
      it->second.state = State::kTerminal;
    }
  }

  std::int64_t queued_bytes() const { return queued_bytes_; }
  std::int64_t running_bytes() const { return running_bytes_; }
  std::int64_t inflight_bytes() const { return inflight_bytes_; }
  std::int64_t accepted_bytes() const { return accepted_bytes_; }
  std::int64_t completed_bytes() const { return completed_bytes_; }
  std::int64_t canceled_bytes() const { return canceled_bytes_; }
  std::int64_t rejected_bytes() const { return rejected_bytes_; }
  std::int64_t failed_bytes() const { return failed_bytes_; }
  std::int64_t already_resident_bytes() const {
    return already_resident_bytes_;
  }

  std::int64_t removed_bytes(RemovalReason reason) const {
    auto it = removed_by_reason_.find(static_cast<int>(reason));
    return it == removed_by_reason_.end() ? 0 : it->second;
  }

  bool InvariantHolds() const {
    if (inflight_bytes_ != queued_bytes_ + running_bytes_) return false;
    return accepted_bytes_ == queued_bytes_ + running_bytes_ +
                                  completed_bytes_ + canceled_bytes_ +
                                  failed_bytes_ + already_resident_bytes_;
  }

 private:
  struct Entry {
    std::int64_t bytes;
    State state;
  };
  std::unordered_map<std::uint32_t, Entry> tasks_;
  std::unordered_map<int, std::int64_t> removed_by_reason_;
  std::int64_t queued_bytes_ = 0;
  std::int64_t running_bytes_ = 0;
  std::int64_t inflight_bytes_ = 0;
  std::int64_t accepted_bytes_ = 0;
  std::int64_t completed_bytes_ = 0;
  std::int64_t canceled_bytes_ = 0;
  std::int64_t rejected_bytes_ = 0;
  std::int64_t failed_bytes_ = 0;
  std::int64_t already_resident_bytes_ = 0;
};

// Runs one popped prefetch task body behind a noexcept boundary. Any std or
// unknown exception retires the task's running bytes as failed exactly once and
// never unwinds through the caller (which for a real worker is std::thread).
template <class Body>
inline void RunPrefetchTaskNoThrow(std::uint32_t tensor_id,
                                   PrefetchQueueAccounting* accounting,
                                   Body&& body) noexcept {
  try {
    std::forward<Body>(body)();
    accounting->MarkCompleted(tensor_id);
  } catch (const std::exception&) {
    accounting->MarkFailed(tensor_id);
  } catch (...) {
    accounting->MarkFailed(tensor_id);
  }
}

struct PrefetchAdmission {
  std::vector<std::uint32_t> accepted_tensor_ids;
  std::int64_t accepted_bytes = 0;
  std::int64_t rejected_bytes = 0;
  std::int64_t inflight_bytes = 0;
};

struct PrefetchSample {
  std::uint64_t generation = 0;
  std::int64_t layer_id = -1;
  std::uint32_t tensor_id = 0;
  std::int64_t bytes = 0;
  std::int64_t queue_wait_ns = 0;
  std::int64_t transfer_ns = 0;
  std::string source_device;
  std::string outcome;
};

struct Task {
  bool on_demand = false;
  NodePtr node;
  std::vector<NodePtr> remove_nodes;
  std::uint32_t priority;
  std::uint64_t request_id;
  torch::Device src_device = DISK_DEVICE;
  torch::Device dst_device = DISK_DEVICE;
  cudaStream_t stream = nullptr;

  std::uint64_t generation = 0;
  std::int64_t layer_id = -1;
  std::int64_t scheduled_bytes = 0;
  std::int64_t enqueue_ns = 0;
  std::int64_t start_ns = 0;

  bool remove_layer = false;

  std::string DebugString() {
    std::stringstream ss;
    ss << "Task: node: " << node->str() << ", on_demand: " << on_demand
       << ", priority: " << priority << "[" << src_device.str() << "->"
       << dst_device.str() << "]";
    return ss.str();
  }
};
typedef std::shared_ptr<Task> TaskPtr;

class ArcherTaskPool : public base::noncopyable {
 public:
  void StartExec(const std::uint64_t& request_id, const NodePtr& node);
  void FetchExec(const std::uint64_t& request_id, const NodePtr& node);
  void StopExec(const std::uint64_t& request_id, const NodePtr& node);
  void EnqueueTask(const TaskPtr& task);

  void ClearQueue() {
    std::lock_guard<std::mutex> lock(unified_mutex_);
    for (std::uint32_t priority = 1; priority < NUM_PRIORITY; priority++) {
      unified_queue_[priority].clear();
    }
  }

  bool RemoveCachedSparseNode(const NodePtr& node, int device_id = -1);
  bool RemoveCachedDenseNode(const NodePtr& node);
  // void RemoveCachedNode(const NodePtr& node);

  PrefetchAdmission AdmitPrefetchTasks(
      const std::vector<std::pair<NodePtr, std::int64_t>>& costed_nodes,
      std::uint32_t priority, std::uint64_t generation, std::int64_t layer_id,
      std::int64_t max_inflight_bytes);
  std::int64_t CancelQueuedPrefetch(
      std::uint64_t generation, std::int64_t layer_id,
      const std::unordered_set<std::uint32_t>& keep_tensor_ids);
  std::vector<PrefetchSample> DrainPrefetchSamples();
  std::int64_t GetInflightPrefetchBytes();

  void ReplaceCacheCandidates(const NodePtrList& candidates) {
    std::lock_guard<std::mutex> lock(unified_mutex_);
    {
      std::lock_guard<std::mutex> lock(this->candidates_mutex_);
      candidates_.clear();
      for (auto& node : candidates) {
        candidates_.insert(node);
      }
    }

    for (std::uint32_t priority = 1; priority < NUM_PRIORITY; priority++) {
      unified_queue_[priority].clear();
    }
  }

  DELETE_COPY_AND_ASSIGN(ArcherTaskPool);
  STATIC_GET_INSTANCE(ArcherTaskPool);

  ArcherTaskPool();
  ~ArcherTaskPool() {
    main_thread_stop_flag_.store(true, std::memory_order_release);
    for (auto& thread_list : exec_threads_) {
      for (auto& thread : thread_list) {
        if (thread.joinable()) thread.detach();
      }
    }
  }

 private:
  void GPUThreadFunc(int gpu_id, int thread_id);

  void SetNodeDevice(const TaskPtr& task);

  std::string DebugString(const std::vector<std::deque<TaskPtr>>& queue);

 private:
  std::vector<std::deque<TaskPtr>> unified_queue_;  // For ordered prefetch
  std::vector<std::vector<std::uint32_t>> gpu_min_priority_;
  std::unordered_map<std::uint64_t, TaskPtr> exec_queue_;
  std::mutex exec_mutex_;
  std::mutex unified_mutex_;
  std::mutex candidates_mutex_;

  std::vector<std::list<std::thread>> exec_threads_;

  std::unordered_set<NodePtr> candidates_;

  std::atomic<bool> main_thread_stop_flag_;

  std::mutex prefetch_accounting_mutex_;
  PrefetchQueueAccounting prefetch_accounting_;
  std::unordered_map<std::uint32_t, std::uint64_t> tensor_generation_;
  std::unordered_map<std::uint32_t, std::int64_t> tensor_layer_;
  std::vector<PrefetchSample> prefetch_samples_;
};

extern std::unique_ptr<ArcherTaskPool> kTaskPool;
