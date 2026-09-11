#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "model/model_topology.h"

struct FakeResizeResult {
  std::string outcome;
  std::int64_t resident_bytes;
  std::string reason;
};

struct FakeReservation {
  std::uint64_t id;
  bool ready;
};

struct FakeResizeToken {
  std::uint64_t id;
  int device_id;
  bool ready;
  std::string reason;
};

class FakeSparseCache {
 public:
  explicit FakeSparseCache(
      std::initializer_list<std::pair<std::size_t, std::int64_t>> specs,
      int device_id = 0);
  Node& node(std::size_t index);
  NodePtr node_ptr(std::size_t index);
  void ReplaceCacheCandidates(std::initializer_list<NodePtr> candidates);
  FakeResizeResult TrimSparseCache(int device_id, std::int64_t target_bytes);
  FakeReservation ReserveSparseCacheVictims(int device_id,
                                            std::int64_t target_bytes);
  void CancelSparseCacheReservation(std::uint64_t id);
  std::int64_t ResidentBytes(int device_id) const;
  bool AllNodesResident(int device_id) const;
  void PauseAfterExecSnapshot();
  bool WaitUntilExecSnapshotReleased(int timeout_ms);
  void ResumeAfterExecSnapshot();

 private:
  int device_id_;
  std::uint64_t next_reservation_id_{1};
  NodePtrList nodes_;
  std::unordered_set<Node*> protected_;
  std::unordered_set<Node*> reserved_;
  std::mutex exec_mutex_;
  std::mutex candidates_mutex_;
  std::mutex pause_mutex_;
  std::condition_variable pause_cv_;
  bool pause_after_exec_snapshot_{false};
  bool exec_snapshot_released_{false};
  bool resume_after_exec_snapshot_{false};
};

class FakeDispatcher {
 public:
  explicit FakeDispatcher(int devices);
  void EnqueueFetch(int device_id);
  void EnqueueExec(int device_id);
  void SetFetchEventComplete(int device_id, bool complete);
  void CompleteQueuesWorkersAndEvents(int device_id);
  FakeResizeToken BeginMemoryResize(int device_id, int timeout_ms);
  void EndMemoryResize(const FakeResizeToken& token);
  bool AdmissionsPaused(int device_id) const;

 public:
  struct DeviceState {
    int fetch_queue = 0;
    int exec_queue = 0;
    int active_fetch = 0;
    int active_exec = 0;
    bool fetch_event_complete = true;
    bool admissions_paused = false;
  };

 private:
  std::uint64_t next_token_id_{1};
  std::vector<DeviceState> devices_;
};
