// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cuda_runtime_api.h>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#ifndef NVTX_DISABLE
  #include <nvtx3/nvtx3.hpp>
#endif

#include "base/noncopyable.h"
#include "common/pytorch.h"
#include "common/types.h"
#include "memory/event_pool.h"
#include "memory/memory_pool.h"
#include "prefetch/expert_policy.h"

enum NodeState {
  NODE_STATE_NONE = 0x0,
  NODE_STATE_CACHED = 0x1,
  NODE_STATE_PREFETCHED = 0x2,
  NODE_STATE_VISITED = 0x4,
};

enum class NodeExecState : uint8_t {
  IDLE = 0,
  FETCHING = 1,
  EXECUTING = 2,
};

// extern cudaStream_t kCudaStreamH2D;

struct Node {
  std::vector<TensorID> tensor_ids;
  std::int64_t byte_size;
  std::size_t last_access_time;
  std::size_t last_prefetch_time = 0;

  std::size_t id;
  std::size_t corr_id;

  torch::Device device = DISK_DEVICE;
  torch::Device default_device =
      DEFAULT_CUDA_DEVICE;  // FIXME: should be set by scheduler
  torch::Device default_host = CPU_DEVICE;
  torch::Device initial_host = DISK_DEVICE;

  std::atomic_uint8_t state{0};  // 0 for ready, 1 for moving

  std::mutex
      mutex;  // DEPRECATED: use exec_state for dispatch/prefetch coordination
  std::atomic<NodeExecState> exec_state{NodeExecState::IDLE};
  std::atomic<bool> is_prefetching{false};
  std::atomic<int> pending_dispatches{0};
  std::condition_variable cv;

  float cache_priority = 0.0;
  std::uint64_t visit_count = 0;
  std::uint64_t incache_visit_count = 0;
  std::uint64_t unused_count = 0;
  bool is_sparse = false;
  NodeState io_state = NODE_STATE_NONE;

  bool is_overflow = false;

  ExpertPolicyMetadata policy_metadata;

  void* host_memory_ptr = nullptr;
  void* device_memory_ptr = nullptr;

 public:
  explicit Node();
  const std::string str();
  void SetDevice(const torch::Device& target_device, bool ondemand = false,
                 cudaStream_t stream = nullptr,
                 cudaEvent_t* transfer_event = nullptr);
};

typedef std::shared_ptr<Node> NodePtr;
typedef std::vector<NodePtr> NodePtrList;
typedef std::tuple<std::int64_t, NodePtrList> FilterResult;

struct NodeBody;
typedef std::shared_ptr<NodeBody> NodeBodyPtr;

struct NodeBody {
  NodePtr node;
  std::vector<NodeBodyPtr> children;
  std::vector<std::size_t> children_visit_cnt;
  std::unordered_set<std::size_t> activate_request;
  std::size_t prefetch_cnt = 0;
  std::size_t visit_cnt = 0;
  std::size_t cpu_visit_cnt = 0;
  std::size_t gpu_visit_cnt = 0;
  std::size_t hit_cnt = 0;
  std::size_t gpu_hit_cnt = 0;
  std::size_t cpu_hit_cnt = 0;
  std::size_t gpu_miss_cnt = 0;
  std::size_t cpu_miss_cnt = 0;
  bool is_sparse;
  std::deque<std::size_t> visit_time;
  explicit NodeBody(NodePtr node) : node(node), visit_cnt(0) {}

  std::string str() const noexcept {
    std::stringstream ss;
    ss << "NodeBody: " << node->str() << " visit_cnt " << visit_cnt
       << ", child visit [";
    for (auto& visit : children_visit_cnt) {
      ss << visit << ",";
    }
    ss << "]";
    return ss.str();
  }
};

struct Stage {
  bool is_sparse;
  std::vector<NodeBodyPtr> nodes;
  std::size_t visit_cnt;
  std::int64_t byte_size;
  std::deque<std::size_t> visit_time;
  std::unordered_set<std::size_t> activate_request;
  Stage() : is_sparse(false), visit_cnt(0), byte_size(0) {}
  Stage(bool is_sparse) : is_sparse(is_sparse), visit_cnt(0), byte_size(0) {}

  std::string str() const noexcept {
    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(buffer, "Stage[%ld,%ld,%d]", nodes.size(), visit_cnt, is_sparse);
    return std::string(buffer);
  }
};
typedef std::shared_ptr<Stage> StagePtr;

struct Pipeline {
  std::vector<StagePtr> stages;
  std::size_t visit_cnt = 0;

  std::string str() const noexcept {
    std::stringstream ss;
    ss << "Pipeline: " << stages.size() << " stages; visit_cnt " << visit_cnt
       << std::endl;
    return ss.str();
  }
};
typedef std::shared_ptr<Pipeline> PipelinePtr;

class ArcherTopologyHandle : public base::noncopyable {
 public:
  DELETE_COPY_AND_ASSIGN(ArcherTopologyHandle);

  ArcherTopologyHandle();
  ~ArcherTopologyHandle() = default;

  bool IsLastNode(const NodePtr& node);
  bool IsFirstNode(const NodePtr& node);

  NodePtrList GetLFUNodes(const torch::Device& device);

  NodePtrList GetDenseNodes(const NodePtr& node, const std::size_t& k);
  NodePtrList GetSparseNodes(const NodePtr& node, const std::size_t& k);
  NodePtrList GetDenseNodes();
  NodePtrList GetSparseNodes();

  std::uint64_t GetLastActivateStage(const HashID& hash_id);

  void InitializeTopology(
      const std::vector<
          std::tuple<std::string, std::vector<std::vector<TensorID>>>>&
          topology);

  void InitializeTopologyV2(
      const std::vector<
          std::tuple<std::string, bool, std::vector<std::vector<TensorID>>,
                     std::vector<std::uint64_t>>>& topology);

  std::vector<std::tuple<std::uint64_t, bool, int>> GetTopologySnapshot();

  void EnableTrace() noexcept { trace_enabled_ = true; }
  void DisableTrace() noexcept { trace_enabled_ = false; }

  std::vector<std::vector<std::size_t>> GetNodeVisitCounts();
  std::tuple<std::int64_t, std::int64_t> GetResidentAndWastedBytes();
  std::vector<std::size_t> GetChildVisitCounts();
  void SetNodeVisitCounts(const std::vector<std::size_t>& visit_counts);
  void SetChildVisitCounts(const std::vector<std::size_t>& visit_counts);

  NodePtr GetNodeFromTensorID(const TensorID& tensor_id);
  NodeBodyPtr GetNodeBodyFromCorrID(const std::uint64_t& correlation_id);

  std::tuple<std::size_t, std::size_t> GetNumLayersAndExperts();

  std::int64_t GetSparseCacheLimit(const torch::Device& device);

  std::size_t GetNumberOfStages() const noexcept {
    return pipeline_.stages.size();
  }

 private:
  struct StageSpec {
    bool is_sparse = false;
    const std::vector<std::vector<TensorID>>* tensor_groups = nullptr;
    std::vector<std::uint64_t> corr_ids;
  };
  void BuildTopologyFromSpecs(const std::vector<StageSpec>& specs);

  Pipeline pipeline_;
  std::unordered_set<HashID> visited_;
  std::unordered_map<HashID, std::uint64_t> last_active_stage_;
  std::vector<NodeBodyPtr> lfu_nodes_;
  std::unordered_map<std::size_t, std::size_t> request_time_;
  std::unordered_map<std::size_t, StagePtr> request_trace_;
  std::int64_t visit_count_ = 0;
  std::mutex mutex_;
  bool trace_enabled_ = true;

  std::unordered_map<TensorID, NodePtr> tensor_id_to_node_;
};

extern std::unique_ptr<ArcherTopologyHandle> kTopologyHandle;

#define CONTINUE_IF_NULL(node) \
  if (node == nullptr) continue;
#define BREAK_IF_NULL(node) \
  if (node == nullptr) break;

extern std::mutex kReadMutex;

void SetModuleDisk(std::vector<TensorID>& tensor_ids);
void SetModuleMemoryFromDisk(std::vector<TensorID>& tensor_ids, void* host_ptr,
                             bool on_demand = false);
void SetModuleMemoryFromDisk_Views(std::vector<TensorID>& tensor_ids,
                                   void* host_ptr);
void SetModuleCudaMemoryFromCPU(std::vector<TensorID>& tensor_ids,
                                void* device_ptr, const torch::Device& device);
void SetModuleMemoryFromCuda(std::vector<TensorID>& tensor_ids, void* host_ptr);
