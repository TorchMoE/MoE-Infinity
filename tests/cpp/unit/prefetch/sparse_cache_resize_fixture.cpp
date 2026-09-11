#include "sparse_cache_resize_fixture.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace {
FakeDispatcher::DeviceState& Checked(
    std::vector<FakeDispatcher::DeviceState>& devices, int device_id) {
  if (device_id < 0 || device_id >= static_cast<int>(devices.size()))
    throw std::out_of_range("device_id");
  return devices.at(device_id);
}
}  // namespace

FakeSparseCache::FakeSparseCache(
    std::initializer_list<std::pair<std::size_t, std::int64_t>> specs,
    int device_id)
    : device_id_(device_id) {
  for (const auto& [id, bytes] : specs) {
    auto node = std::make_shared<Node>();
    node->id = id;
    node->byte_size = bytes;
    node->device = torch::Device(torch::kCUDA, device_id);
    node->default_device = node->device;
    node->default_host = torch::Device(torch::kCPU);
    node->exec_state.store(NodeExecState::IDLE);
    node->pending_dispatches.store(0);
    nodes_.push_back(std::move(node));
  }
}

Node& FakeSparseCache::node(std::size_t index) { return *nodes_.at(index); }
NodePtr FakeSparseCache::node_ptr(std::size_t index) {
  return nodes_.at(index);
}

void FakeSparseCache::ReplaceCacheCandidates(
    std::initializer_list<NodePtr> candidates) {
  std::lock_guard<std::mutex> lock(candidates_mutex_);
  protected_.clear();
  for (const auto& node : candidates) protected_.insert(node.get());
}

void FakeSparseCache::PauseAfterExecSnapshot() {
  std::lock_guard<std::mutex> lock(pause_mutex_);
  pause_after_exec_snapshot_ = true;
  exec_snapshot_released_ = false;
  resume_after_exec_snapshot_ = false;
}

bool FakeSparseCache::WaitUntilExecSnapshotReleased(int timeout_ms) {
  std::unique_lock<std::mutex> lock(pause_mutex_);
  return pause_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            [&] { return exec_snapshot_released_; });
}

void FakeSparseCache::ResumeAfterExecSnapshot() {
  {
    std::lock_guard<std::mutex> lock(pause_mutex_);
    resume_after_exec_snapshot_ = true;
  }
  pause_cv_.notify_all();
}

std::int64_t FakeSparseCache::ResidentBytes(int device_id) const {
  std::int64_t bytes = 0;
  for (const auto& node : nodes_)
    if (node->device.is_cuda() && node->device.index() == device_id)
      bytes += node->byte_size;
  return bytes;
}

bool FakeSparseCache::AllNodesResident(int device_id) const {
  return std::all_of(
      nodes_.begin(), nodes_.end(), [device_id](const NodePtr& n) {
        return n->device.is_cuda() && n->device.index() == device_id;
      });
}

FakeReservation FakeSparseCache::ReserveSparseCacheVictims(
    int device_id, std::int64_t target_bytes) {
  {
    std::lock_guard<std::mutex> lock(exec_mutex_);
    // The production fixture has no execute entries; taking and releasing this
    // lock models the production exec-membership snapshot boundary.
  }
  {
    std::unique_lock<std::mutex> lock(pause_mutex_);
    if (pause_after_exec_snapshot_) {
      exec_snapshot_released_ = true;
      pause_cv_.notify_all();
      pause_cv_.wait(lock, [&] { return resume_after_exec_snapshot_; });
    }
  }
  std::unordered_set<Node*> protected_snapshot;
  {
    std::lock_guard<std::mutex> lock(candidates_mutex_);
    protected_snapshot = protected_;
  }
  reserved_.clear();
  auto remaining = ResidentBytes(device_id);
  for (const auto& node : nodes_) {
    if (remaining <= target_bytes) break;
    if (!node->device.is_cuda() || node->device.index() != device_id ||
        protected_snapshot.count(node.get()) != 0 ||
        node->pending_dispatches.load() != 0 ||
        node->exec_state.load() != NodeExecState::IDLE)
      continue;
    reserved_.insert(node.get());
    remaining -= node->byte_size;
  }
  if (remaining > target_bytes) {
    reserved_.clear();
    return {0, false};
  }
  return {next_reservation_id_++, true};
}

void FakeSparseCache::CancelSparseCacheReservation(std::uint64_t) {
  reserved_.clear();
}

FakeResizeResult FakeSparseCache::TrimSparseCache(int device_id,
                                                  std::int64_t target_bytes) {
  const auto reservation = ReserveSparseCacheVictims(device_id, target_bytes);
  if (!reservation.ready)
    return {"rejected", ResidentBytes(device_id), "pinned_or_in_flight"};
  for (const auto& node : nodes_)
    if (reserved_.count(node.get()) != 0) node->device = node->default_host;
  reserved_.clear();
  return {"committed", ResidentBytes(device_id), "committed"};
}

FakeDispatcher::FakeDispatcher(int devices) : devices_(devices) {}
void FakeDispatcher::EnqueueFetch(int d) { Checked(devices_, d).fetch_queue++; }
void FakeDispatcher::EnqueueExec(int d) { Checked(devices_, d).exec_queue++; }
void FakeDispatcher::SetFetchEventComplete(int d, bool done) {
  Checked(devices_, d).fetch_event_complete = done;
}
void FakeDispatcher::CompleteQueuesWorkersAndEvents(int d) {
  auto& state = Checked(devices_, d);
  state.fetch_queue = state.exec_queue = 0;
  state.active_fetch = state.active_exec = 0;
  state.fetch_event_complete = true;
}
FakeResizeToken FakeDispatcher::BeginMemoryResize(int d, int) {
  auto& state = Checked(devices_, d);
  state.admissions_paused = true;
  if (state.fetch_queue || state.exec_queue || state.active_fetch ||
      state.active_exec || !state.fetch_event_complete) {
    state.admissions_paused = false;
    return {0, d, false, "dispatcher_drain_timeout"};
  }
  return {next_token_id_++, d, true, "ready"};
}
void FakeDispatcher::EndMemoryResize(const FakeResizeToken& token) {
  Checked(devices_, token.device_id).admissions_paused = false;
}
bool FakeDispatcher::AdmissionsPaused(int d) const {
  return devices_.at(d).admissions_paused;
}
