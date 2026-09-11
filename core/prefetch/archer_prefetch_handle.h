// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <string>
#include <unordered_map>

#include "aio/archer_tensor_handle.h"
#include "model/model_topology.h"
#include "parallel/expert_dispatcher.h"
#include "prefetch/expert_residency.h"
#include "prefetch/task_scheduler.h"

class ArcherPrefetchHandle {
 public:
  ArcherPrefetchHandle(const std::string& prefix,
                       const double device_memory_ratio = 0.8);
  ~ArcherPrefetchHandle();

  bool IsTensorOffloaded(const std::uint32_t tensor_id);

  void AcquireTensor(std::uint64_t& request_id, torch::Tensor& buffer,
                     std::uint32_t explicit_id = UINT32_MAX);
  void ReleaseTensor(std::uint64_t& request_id, torch::Tensor& buffer,
                     std::uint32_t explicit_id = UINT32_MAX);
  void PrefetchTensors(std::uint64_t& request_id,
                       const std::vector<std::uint32_t>& buffer);
  void FetchTensors(std::uint64_t& request_id,
                    const std::vector<std::uint32_t>& buffer);

  void ReplaceCacheCandidates(const std::vector<std::uint32_t>& tensor_ids);
  void EnqueuePrefetch(const uint32_t tensor_id, int gpu_id);
  void EnqueuePrefetchTensors(const std::vector<std::uint32_t>& tensor_ids,
                              std::uint32_t priority = kRouteAheadPriority,
                              int phase = static_cast<int>(ExpertPhase::MIXED));

  PrefetchAdmission SchedulePrefetchTensors(
      const std::vector<std::uint32_t>& tensor_ids, std::uint32_t priority,
      std::uint64_t generation, std::int64_t layer_id,
      std::int64_t max_inflight_bytes);
  std::int64_t CancelPrefetchGeneration(
      std::uint64_t generation, std::int64_t layer_id,
      const std::vector<std::uint32_t>& keep_tensor_ids);
  std::vector<PrefetchSample> DrainPrefetchSamples();
  std::int64_t GetInflightPrefetchBytes();

  void OffloadTensor(torch::Tensor& tensor, const std::uint32_t tensor_id);
  void RegisterTensor(torch::Tensor& tensor, const std::uint32_t tensor_id);
  void RegisterModule(torch::nn::Module& module);
  void RegisterTensor(torch::Tensor* tensor);

  int GetNodeDefaultDevice(std::vector<std::uint32_t> tensor_ids) const;
  int GetNodeDevice(std::vector<std::uint32_t> tensor_ids) const;

  void SetTensorDevice(torch::Tensor& tensor, torch::Device device) const;

  torch::Tensor GetTrace();
  torch::Tensor GetHitRate();
  std::int64_t GetExpertOccupancyBytes();
  std::int64_t GetWastedPrefetchBytes();
  void SetTrace(const torch::Tensor& trace);
  void TraceRequest(const std::uint64_t request_id, const TensorID tensor_id);
  void SetTopology(const std::vector<
                   std::tuple<std::string, std::vector<std::vector<TensorID>>>>&
                       topology);
  void SetTopologyV2(
      const std::vector<
          std::tuple<std::string, bool, std::vector<std::vector<TensorID>>,
                     std::vector<std::uint64_t>>>& topology);
  std::vector<std::tuple<std::uint64_t, bool, int>> GetTopologySnapshot();
  NodePtr CreateDetachedNode(const std::vector<TensorID>& tensor_ids,
                             int gpu_id);
  std::uintptr_t GetResidencyManagerId() const;
  void ConfigureResidencyManager(bool manager_enabled,
                                 bool phase_policy_enabled);
  bool SetAdaptiveHbmBudgetBytes(std::int64_t bytes);
  std::size_t PrefetchExpertVariants(
      const std::vector<std::tuple<int, int, std::string, std::uint64_t>>& keys,
      std::uint32_t priority, const std::string& phase);
  void UpdateTensorMap(std::uint64_t old_ptr, std::uint64_t new_ptr);
  bool IsTensorIndexInitialized() const;
  bool IsTensorOnDevice(const torch::Tensor& tensor) const;
  bool IsTensorOnDevice(const TensorID tensor_id) const;

  std::vector<std::unordered_map<std::string, py::object>>
  GetCanonicalTensorIndexSnapshot() const;
  void BeginDerivativeOverlay(const std::string& generation,
                              std::int64_t canonical_max_tensor_id,
                              std::int64_t canonical_max_file_id);
  void RegisterDerivativeTensor(const std::string& generation,
                                std::int64_t tensor_id, std::int64_t file_id,
                                std::int64_t offset, std::int64_t size,
                                const std::vector<std::int64_t>& shape,
                                const std::string& dtype);
  void CommitDerivativeOverlay(const std::string& generation);
  void AbortDerivativeOverlay(const std::string& generation);

  void CleanUpResources();
  void ResetCache();

  std::unordered_map<std::string, std::int64_t> ResizeExpertCache(
      int device_id, std::int64_t target_bytes);
  std::int64_t GetExpertCacheLimit(int device_id);
  std::uint64_t BeginMemoryResize(int device_id, int timeout_ms);
  void EndMemoryResize(std::uint64_t token);
  void ConfigureExpertPolicy(bool enabled, int prefill_admission,
                             int decode_admission, double prefill_weight,
                             double decode_weight, int starvation_limit);
  ExpertPolicyStats GetExpertPolicyStats() const;

  // void SetNodeCachePriority(const std::uint64_t corr_id, const float
  // priority);

 private:
  void ConfigureExpertCapacityAfterTopology();

  std::string prefix_;
  std::unordered_map<std::size_t, std::unordered_set<std::uint32_t>>
      node_id_to_tensor_ids_;
  std::unordered_set<std::uint32_t> tensors_to_delete_;
  uint64_t last_layer_id_;
  NodePtr last_node_;
  bool has_cleaned_up_resources_;
  bool manager_enabled_ = false;
  bool phase_policy_enabled_ = false;

  std::unordered_map<std::uint64_t, std::unordered_set<NodePtr>>
      request_id_to_nodes_;

  std::mutex mutex_;
};
