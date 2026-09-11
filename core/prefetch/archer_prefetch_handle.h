// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include "aio/archer_tensor_handle.h"
#include "model/model_topology.h"
#include "parallel/expert_dispatcher.h"
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
                              std::uint32_t priority = kRouteAheadPriority);

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
  void UpdateTensorMap(std::uint64_t old_ptr, std::uint64_t new_ptr);
  bool IsTensorIndexInitialized() const;
  bool IsTensorOnDevice(const torch::Tensor& tensor) const;
  bool IsTensorOnDevice(const TensorID tensor_id) const;

  void CleanUpResources();
  void ResetCache();

  // void SetNodeCachePriority(const std::uint64_t corr_id, const float
  // priority);

 private:
  std::string prefix_;
  std::unordered_map<std::size_t, std::unordered_set<std::uint32_t>>
      node_id_to_tensor_ids_;
  std::unordered_set<std::uint32_t> tensors_to_delete_;
  uint64_t last_layer_id_;
  NodePtr last_node_;
  bool has_cleaned_up_resources_;

  std::unordered_map<std::uint64_t, std::unordered_set<NodePtr>>
      request_id_to_nodes_;

  std::mutex mutex_;
};
