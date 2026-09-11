// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "archer_prefetch_handle.h"

#include <algorithm>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include "aio/archer_tensor_handle.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "memory/memory_pool.h"
#include "prefetch/expert_residency.h"
#include "task_scheduler.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"

namespace {

std::optional<ExpertFormat> ParsePrefetchFormat(const std::string& value) {
  if (value == "bf16") return ExpertFormat::BF16;
  if (value == "fp8_e4m3_block128") return ExpertFormat::FP8_E4M3_BLOCK128;
  if (value == "marlin_int4_group128")
    return ExpertFormat::MARLIN_INT4_GROUP128;
  return std::nullopt;
}

}  // namespace

ArcherPrefetchHandle::ArcherPrefetchHandle(const std::string& prefix,
                                           const double device_memory_ratio)
    : prefix_(prefix), last_layer_id_(0), has_cleaned_up_resources_(false) {
  // InitLogger();
  int num_io_threads = 0;
  const char* io_threads_env = std::getenv("MOE_IO_THREADS");
  if (io_threads_env != nullptr) {
    num_io_threads = std::atoi(io_threads_env);
  }
  kTensorIndex = std::make_unique<ArcherTensorIndex>();
  kArcherTensorHandle =
      std::make_unique<ArcherTensorHandle>(prefix, num_io_threads);
  kTopologyHandle = std::make_unique<ArcherTopologyHandle>();
  kTaskPool = std::make_unique<ArcherTaskPool>();
  InitExpertResidency();
  kDeviceMemoryPool = std::make_unique<DeviceMemoryPool>();
  kHostMemoryPool = std::make_unique<HostMemoryPool>();
  kDeviceMemoryPool->SetMemoryRatio(device_memory_ratio);
  DLOG_TRACE("Free Device Memory ",
             kDeviceMemoryPool->GetFreeMemory(CUDA_DEVICE(0)));

  if (prefix_.back() != '/') {
    prefix_ += '/';
  }

  // enable peer access for kernels
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  DLOG_INFO("Device count ", device_count);

  for (int i = 0; i < device_count; i++) {
    for (int j = 0; j < device_count; j++) {
      if (i != j) {
        int can_access = 0;
        cudaDeviceCanAccessPeer(&can_access, i, j);
        if (can_access == 1) {
          cudaSetDevice(i);
          cudaError_t status = cudaDeviceEnablePeerAccess(j, 0);
          if (status == cudaErrorPeerAccessAlreadyEnabled) {
            DLOG_INFO("Peer access already enabled between device ", i, j);
            cudaGetLastError();  // clear error
          } else if (status != cudaSuccess) {
            DLOG_ERROR("Failed to enable peer access between device ", i, j);
          } else {
            DLOG_INFO("Enabled peer access between device ", i, j);
          }
        }
      }
    }
  }

  DLOG_INFO("Enabled peer access for all devices");
}

ArcherPrefetchHandle::~ArcherPrefetchHandle() {
  // served as a global manager for order of destruction
  if (!has_cleaned_up_resources_) {
    CleanUpResources();
  }
}

void ArcherPrefetchHandle::CleanUpResources() {
  kTaskPool.reset();
  kArcherTensorHandle.reset();
  kTensorIndex.reset();
  kTopologyHandle.reset();
  kDeviceMemoryPool.reset();
  kHostMemoryPool.reset();
  ResetExpertResidency();
  has_cleaned_up_resources_ = true;
}

void ArcherPrefetchHandle::ConfigureExpertPolicy(
    bool enabled, int prefill_admission, int decode_admission,
    double prefill_weight, double decode_weight, int starvation_limit) {
  PhasePolicyConfig config;
  config.enabled = enabled;
  config.prefill_admission = static_cast<AdmissionMode>(prefill_admission);
  config.decode_admission = static_cast<AdmissionMode>(decode_admission);
  config.prefill_eviction_weight = prefill_weight;
  config.decode_eviction_weight = decode_weight;
  config.starvation_limit = static_cast<std::uint32_t>(starvation_limit);
  if (kExpertResidencyManager) {
    kExpertResidencyManager->ConfigurePolicy(config);
  }
}

ExpertPolicyStats ArcherPrefetchHandle::GetExpertPolicyStats() const {
  if (!kExpertResidencyManager) return ExpertPolicyStats{};
  return kExpertResidencyManager->Snapshot();
}

void ArcherPrefetchHandle::ResetCache() {
  // Non-terminal reset (BM3 ablation): drop pending prefetch work so a prior
  // arm's route-ahead band cannot leak into the next, without tearing down the
  // engine. Unlike CleanUpResources, kTaskPool/threads stay alive and
  // has_cleaned_up_resources_ is untouched, so the next FetchTensors does not
  // deadlock. ClearQueue scans priority 1..NUM_PRIORITY, leaving on-demand (0).
  if (kTaskPool) {
    kTaskPool->ClearQueue();
  }
}

std::vector<std::unordered_map<std::string, py::object>>
ArcherPrefetchHandle::GetCanonicalTensorIndexSnapshot() const {
  return kArcherTensorHandle->GetCanonicalTensorIndexSnapshot();
}

void ArcherPrefetchHandle::BeginDerivativeOverlay(
    const std::string& generation, std::int64_t canonical_max_tensor_id,
    std::int64_t canonical_max_file_id) {
  kArcherTensorHandle->BeginDerivativeOverlay(
      generation, canonical_max_tensor_id, canonical_max_file_id);
}

void ArcherPrefetchHandle::RegisterDerivativeTensor(
    const std::string& generation, std::int64_t tensor_id, std::int64_t file_id,
    std::int64_t offset, std::int64_t size,
    const std::vector<std::int64_t>& shape, const std::string& dtype) {
  kArcherTensorHandle->RegisterDerivativeTensor(generation, tensor_id, file_id,
                                                offset, size, shape, dtype);
}

void ArcherPrefetchHandle::CommitDerivativeOverlay(
    const std::string& generation) {
  kArcherTensorHandle->CommitDerivativeOverlay(generation);
}

void ArcherPrefetchHandle::AbortDerivativeOverlay(
    const std::string& generation) {
  kArcherTensorHandle->AbortDerivativeOverlay(generation);
}

void ArcherPrefetchHandle::AcquireTensor(std::uint64_t& request_id,
                                         torch::Tensor& buffer,
                                         std::uint32_t explicit_id) {
  auto tensor_id =
      (explicit_id != UINT32_MAX)
          ? explicit_id
          : kArcherTensorHandle->GetTensorId((void*)buffer.data_ptr());
  void* old_ptr = (void*)buffer.data_ptr();
  DLOG_TRACE("Acquire tensor ", tensor_id, old_ptr);

  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
  if (node == nullptr) {
    DLOG_ERROR("AcquireTensor: no topology node for tensor_id ", tensor_id);
    return;
  }
  node->state = 1;

  // add node tensor_ids to node_id_to_tensor_ids_
  if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end() ||
      node_id_to_tensor_ids_[node->id].size() == 0) {
    node_id_to_tensor_ids_[node->id] = std::unordered_set<std::uint32_t>();
    for (auto& tensor_id : node->tensor_ids) {
      node_id_to_tensor_ids_[node->id].insert(tensor_id);
    }

    auto node_body = kTopologyHandle->GetNodeBodyFromCorrID(node->corr_id);
    if (node->device.is_cuda()) {
      node_body->gpu_hit_cnt++;
    }

    while (true) {
      auto expected = NodeExecState::IDLE;
      if (node->exec_state.compare_exchange_strong(
              expected, NodeExecState::FETCHING, std::memory_order_acq_rel)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    if (node->is_sparse) {
      bool success = kTaskPool->RemoveCachedSparseNode(node);
      if (!success) node->is_overflow = true;
    } else {
      kTaskPool->RemoveCachedDenseNode(node);
    }
    kTaskPool->StartExec(request_id, node);
    while (node->state.load() != 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  kArcherTensorHandle->SetTensor(tensor_id, buffer);
  kArcherTensorHandle->UpdateTensorMap(old_ptr, (void*)buffer.data_ptr());
}
void ArcherPrefetchHandle::ReleaseTensor(std::uint64_t& request_id,
                                         torch::Tensor& buffer,
                                         std::uint32_t explicit_id) {
  auto tensor_id =
      (explicit_id != UINT32_MAX)
          ? explicit_id
          : kArcherTensorHandle->GetTensorId((void*)buffer.data_ptr());
  void* old_ptr = (void*)buffer.data_ptr();
  DLOG_TRACE("Release tensor ", tensor_id, old_ptr);

  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
  if (node == nullptr) {
    DLOG_ERROR("ReleaseTensor: no topology node for tensor_id ", tensor_id);
    return;
  }
  // node->state = 1;

  if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end()) {
    DLOG_ERROR("Node not found in node_id_to_tensor_ids_", node->str());
    return;
  }

  /*  This needs to go after Release, default host can be changed in
   * TraceRequest Faulty case: node -> default_host = cpu, node -> default_host
   * = cuda; tensor already released
   */
  // if (node != last_node_) {
  //     // kTaskPool->Prefetch(request_id, node);
  //     TraceRequest(request_id, tensor_id);
  // }
  // TraceRequest(request_id, tensor_id);

  auto current_layer_id = node->corr_id & 0xFFFFFFFF;
  if (current_layer_id != last_layer_id_ && last_node_ != nullptr &&
      node_id_to_tensor_ids_[last_node_->id].size() != 0) {
    node_id_to_tensor_ids_[last_node_->id].clear();
    kTaskPool->StopExec(request_id,
                        last_node_);  // evict last node to cpu or disk
    last_node_->exec_state.store(NodeExecState::IDLE,
                                 std::memory_order_release);
  }
  last_layer_id_ = current_layer_id;
  last_node_ = node;

  node_id_to_tensor_ids_[node->id].erase(tensor_id);
  // DLOG_TRACE(
  //     "Node {} tensor_ids size {}", node->id,
  //     node_id_to_tensor_ids_[node->id].size());

  if (node_id_to_tensor_ids_[node->id].size() == 0) {
    kTaskPool->StopExec(request_id,
                        node);  // FIXME: change api to add request id
    // always unlock node here since, exec queue do not unlock automatically
    node->exec_state.store(NodeExecState::IDLE, std::memory_order_release);
  }

  if (kTopologyHandle->IsLastNode(node)) {
    DLOG_TRACE("Node is last, clean up", node->str());
    request_id_to_nodes_.erase(request_id);
  }

  at::TensorOptions options;
  options = options.device(torch::kCPU);
  options = options.dtype(buffer.dtype());
  auto zero_tensor = torch::zeros({1}, options);
  buffer.set_data(zero_tensor);
  kArcherTensorHandle->UpdateTensorMap(old_ptr, (void*)buffer.data_ptr());
}

void ArcherPrefetchHandle::PrefetchTensors(
    std::uint64_t& request_id, const std::vector<std::uint32_t>& buffer) {
  std::vector<NodePtr> candidates;
  for (std::uint32_t tensor_id : buffer) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    candidates.push_back(node);
  }

  if (candidates.size() == 0) {
    return;
  }
}

void ArcherPrefetchHandle::ReplaceCacheCandidates(
    const std::vector<std::uint32_t>& tensor_ids) {
  std::vector<NodePtr> candidates;
  for (std::uint32_t tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    candidates.push_back(node);
  }

  if (kExpertResidencyManager && kExpertResidencyManager->PolicyEnabled()) {
    kExpertResidencyManager->ReplaceProtectedCandidates(candidates);
  } else {
    kTaskPool->ReplaceCacheCandidates(candidates);
  }
}
void ArcherPrefetchHandle::EnqueuePrefetch(const uint32_t tensor_id,
                                           int gpu_id) {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);

  auto task = std::make_shared<Task>();
  // BM3 verdict NO-SHIP: revert the route-ahead priority band. Ordinary
  // prefetch returns to band 1 (pre-candidate), so route-ahead is no longer
  // serviced ahead of background prefetch. Two seeds proved no robust win --
  // the exposed-fetch effect flipped sign and never held throughput at once.
  task->priority = 1;
  task->node = node;
  task->on_demand = false;
  task->src_device = node->device;
  // task->dst_device = CUDA_DEVICE(gpu_id); // use default device for now
  task->dst_device = node->default_device;
  kTaskPool->EnqueueTask(task);
}

void ArcherPrefetchHandle::EnqueuePrefetchTensors(
    const std::vector<std::uint32_t>& tensor_ids, std::uint32_t priority,
    int phase) {
  TORCH_CHECK(phase >= static_cast<int>(ExpertPhase::PREFILL) &&
                  phase <= static_cast<int>(ExpertPhase::MIXED),
              "invalid expert phase: ", phase);
  for (std::uint32_t tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    auto task = std::make_shared<Task>();
    task->priority = priority;
    task->node = node;
    task->on_demand = false;
    task->phase = static_cast<ExpertPhase>(phase);
    task->src_device = node->device;
    task->dst_device = node->default_device;
    kTaskPool->EnqueueTask(task);
  }
}

PrefetchAdmission ArcherPrefetchHandle::SchedulePrefetchTensors(
    const std::vector<std::uint32_t>& tensor_ids, std::uint32_t priority,
    std::uint64_t generation, std::int64_t layer_id,
    std::int64_t max_inflight_bytes) {
  if (priority == kOnDemandPriority) {
    return PrefetchAdmission{};
  }
  std::vector<std::pair<NodePtr, std::int64_t>> costed_nodes;
  std::unordered_set<std::size_t> seen_node_ids;
  for (std::uint32_t tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (node == nullptr) continue;
    if (!seen_node_ids.insert(node->id).second) continue;
    costed_nodes.emplace_back(node, node->byte_size);
  }
  return kTaskPool->AdmitPrefetchTasks(costed_nodes, priority, generation,
                                       layer_id, max_inflight_bytes);
}

std::int64_t ArcherPrefetchHandle::CancelPrefetchGeneration(
    std::uint64_t generation, std::int64_t layer_id,
    const std::vector<std::uint32_t>& keep_tensor_ids) {
  std::unordered_set<std::uint32_t> keep_node_ids;
  for (std::uint32_t tensor_id : keep_tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (node != nullptr) {
      keep_node_ids.insert(static_cast<std::uint32_t>(node->id));
    }
  }
  return kTaskPool->CancelQueuedPrefetch(generation, layer_id, keep_node_ids);
}

std::vector<PrefetchSample> ArcherPrefetchHandle::DrainPrefetchSamples() {
  return kTaskPool->DrainPrefetchSamples();
}

std::int64_t ArcherPrefetchHandle::GetInflightPrefetchBytes() {
  return kTaskPool->GetInflightPrefetchBytes();
}

void ArcherPrefetchHandle::FetchTensors(
    std::uint64_t& request_id, const std::vector<std::uint32_t>& buffer) {
  // std::vector<NodePtr> candidates;
  for (std::uint32_t tensor_id : buffer) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    kTaskPool->FetchExec(request_id, node);
  }
}

void ArcherPrefetchHandle::OffloadTensor(torch::Tensor& tensor,
                                         const std::uint32_t tensor_id) {
  kArcherTensorHandle->StoreTensor(tensor_id, tensor);

  auto ckpt_index_path = prefix_ + std::string(ARCHER_IHDEX_NAME);

  std::unique_lock<std::mutex> lock(mutex_);
  kTensorIndex->Serialize(ckpt_index_path.c_str());
}

void ArcherPrefetchHandle::RegisterTensor(torch::Tensor& tensor,
                                          const std::uint32_t tensor_id) {
  kArcherTensorHandle->RegisterTensor(tensor_id, tensor);
}

void ArcherPrefetchHandle::RegisterModule(torch::nn::Module& module) {
  for (auto it = module.parameters().begin(); it != module.parameters().end();
       ++it) {
    auto tensor_id =
        kArcherTensorHandle->GetTensorId((void*)(*it).unsafeGetTensorImpl());
    kArcherTensorHandle->RegisterTensor(tensor_id, *it);
  }

  for (auto it = module.buffers().begin(); it != module.buffers().end(); ++it) {
    auto tensor_id =
        kArcherTensorHandle->GetTensorId((void*)(*it).unsafeGetTensorImpl());
    kArcherTensorHandle->RegisterTensor(tensor_id, *it);
  }
}

void ArcherPrefetchHandle::RegisterTensor(torch::Tensor* tensor) {
  DLOG_TRACE("Register tensor: is view ", (void*)tensor, tensor->is_view());
}

torch::Tensor ArcherPrefetchHandle::GetTrace() {
  const auto& child_visit_cnts = kTopologyHandle->GetChildVisitCounts();
  const auto num_layers_and_experts = kTopologyHandle->GetNumLayersAndExperts();
  const auto num_layers = std::get<0>(num_layers_and_experts);
  const auto num_experts = std::get<1>(num_layers_and_experts);

  std::vector<int64_t> trace_vec(child_visit_cnts.begin(),
                                 child_visit_cnts.end());
  torch::Tensor trace = torch::from_blob(trace_vec.data(),
                                         {static_cast<int64_t>(num_layers - 1),
                                          static_cast<int64_t>(num_experts),
                                          static_cast<int64_t>(num_experts)},
                                         torch::kInt64)
                            .clone();

  return trace;
}

torch::Tensor ArcherPrefetchHandle::GetHitRate() {
  const auto& node_visit_cnts = kTopologyHandle->GetNodeVisitCounts();

  // flatten vector of vectors
  std::vector<int64_t> node_visit_cnts_vec;
  for (auto& node_visit_cnt : node_visit_cnts) {
    node_visit_cnts_vec.insert(node_visit_cnts_vec.end(),
                               node_visit_cnt.begin(), node_visit_cnt.end());
  }

  torch::Tensor trace =
      torch::from_blob(node_visit_cnts_vec.data(),
                       {node_visit_cnts.size(), node_visit_cnts[0].size()},
                       torch::kInt64)
          .clone();
  return trace;
}

std::int64_t ArcherPrefetchHandle::GetExpertOccupancyBytes() {
  return std::get<0>(kTopologyHandle->GetResidentAndWastedBytes());
}

std::int64_t ArcherPrefetchHandle::GetWastedPrefetchBytes() {
  return std::get<1>(kTopologyHandle->GetResidentAndWastedBytes());
}

void ArcherPrefetchHandle::SetTrace(const torch::Tensor& trace) {
  if (trace.dim() != 3 || !trace.is_contiguous() || !trace.is_cpu()) {
    DLOG_ERROR("Trace should be a contiguous 3D tensor on CPU");
    return;
  }

  std::vector<std::size_t> child_visit_cnts(
      trace.data_ptr<int64_t>(), trace.data_ptr<int64_t>() + trace.numel());
  kTopologyHandle->SetChildVisitCounts(child_visit_cnts);
}

void ArcherPrefetchHandle::TraceRequest(const std::uint64_t request_id,
                                        const TensorID tensor_id) {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);

  auto it = request_id_to_nodes_.find(request_id);
  if (it == request_id_to_nodes_.end()) {
    request_id_to_nodes_[request_id] = std::unordered_set<NodePtr>();
  }

  auto node_it = request_id_to_nodes_[request_id].find(node);
  if (node_it != request_id_to_nodes_[request_id].end()) {
    DLOG_TRACE("Node already traced for request ", request_id, node->str());
    return;
  }

  request_id_to_nodes_[request_id].insert(node);
}

void ArcherPrefetchHandle::ConfigureExpertCapacityAfterTopology() {
  for (int gpu_id = 0; gpu_id < kNumDevices(); ++gpu_id) {
    const auto device = torch::Device(torch::kCUDA, gpu_id);
    const auto bytes = kTopologyHandle->GetSparseCacheLimit(device);
    TORCH_CHECK(kExpertResidencyManager->ConfigureCapacity(gpu_id, bytes),
                "failed to configure expert residency capacity for GPU ",
                gpu_id, " with ", bytes, " bytes");
  }
}

void ArcherPrefetchHandle::SetTopology(
    const std::vector<
        std::tuple<std::string, std::vector<std::vector<TensorID>>>>&
        topology) {
  kTopologyHandle->InitializeTopology(topology);
  ConfigureExpertResidencyCapacityFromTopology();
  ConfigureExpertCapacityAfterTopology();
}

void ArcherPrefetchHandle::SetTopologyV2(
    const std::vector<
        std::tuple<std::string, bool, std::vector<std::vector<TensorID>>,
                   std::vector<std::uint64_t>>>& topology) {
  kTopologyHandle->InitializeTopologyV2(topology);
  ConfigureExpertResidencyCapacityFromTopology();
  ConfigureExpertCapacityAfterTopology();
}

std::vector<std::tuple<std::uint64_t, bool, int>>
ArcherPrefetchHandle::GetTopologySnapshot() {
  return kTopologyHandle->GetTopologySnapshot();
}

NodePtr ArcherPrefetchHandle::CreateDetachedNode(
    const std::vector<TensorID>& tensor_ids, int gpu_id) {
  return kTopologyHandle->CreateDetachedNode(tensor_ids, gpu_id);
}

std::uintptr_t ArcherPrefetchHandle::GetResidencyManagerId() const {
  return reinterpret_cast<std::uintptr_t>(kExpertResidencyManager.get());
}

void ArcherPrefetchHandle::ConfigureResidencyManager(
    bool manager_enabled, bool phase_policy_enabled) {
  manager_enabled_ = manager_enabled;
  phase_policy_enabled_ = phase_policy_enabled;
}

bool ArcherPrefetchHandle::SetAdaptiveHbmBudgetBytes(std::int64_t bytes) {
  if (bytes <= 0 || kExpertResidencyManager == nullptr) return false;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  for (int gpu_id = 0; gpu_id < std::max(device_count, 1); ++gpu_id) {
    if (!kExpertResidencyManager->ConfigureCapacity(gpu_id, bytes))
      return false;
  }
  return true;
}

std::size_t ArcherPrefetchHandle::PrefetchExpertVariants(
    const std::vector<std::tuple<int, int, std::string, std::uint64_t>>& keys,
    std::uint32_t priority, const std::string& phase) {
  if (!manager_enabled_ || kExpertResidencyManager == nullptr ||
      kPrefetchResidencyClient == nullptr) {
    return 0;
  }
  const ExpertPhase expert_phase = phase == "prefill"  ? ExpertPhase::PREFILL
                                   : phase == "decode" ? ExpertPhase::DECODE
                                                       : ExpertPhase::MIXED;
  std::size_t admitted = 0;
  for (const auto& item : keys) {
    const auto format = ParsePrefetchFormat(std::get<2>(item));
    if (!format.has_value()) continue;
    const auto logical_key = (static_cast<std::uint64_t>(
                                  static_cast<std::uint32_t>(std::get<0>(item)))
                              << 32) |
                             static_cast<std::uint32_t>(std::get<1>(item));
    ResidencyVariantKey key{logical_key, *format, std::get<3>(item)};
    const auto variant = kExpertResidencyManager->RegisteredVariant(key);
    if (!variant.has_value()) continue;
    const int gpu_id = std::get<1>(item) % std::max(kNumDevices(), 1);
    auto transaction = kPrefetchResidencyClient->BeginAdmission(
        *variant, gpu_id, expert_phase, AdmissionMode::CACHE);
    if (transaction.outcome == AdmissionOutcome::ALREADY_RESIDENT) {
      ++admitted;
      continue;
    }
    if (!transaction.valid) continue;
    if (transaction.reserved_victim_key.has_value() &&
        !kExpertResidencyManager->EvictReserved(transaction)) {
      kExpertResidencyManager->AbortTransaction(transaction);
      continue;
    }
    try {
      cudaEvent_t event = nullptr;
      variant->node->SetDevice(CUDA_DEVICE(gpu_id), false, nullptr, &event);
      if (event != nullptr) {
        cudaEventSynchronize(event);
        kCudaEventPool->Release(event);
      }
      if (kExpertResidencyManager->CommitTransaction(transaction)) {
        kExpertResidencyManager->RecordWorkspaceUse(transaction.id, nullptr);
        kExpertResidencyManager->ReapWorkspace(gpu_id);
        ++admitted;
      }
    } catch (const std::exception&) {
      kExpertResidencyManager->AbortTransaction(transaction);
    }
  }
  (void)priority;
  return admitted;
}

bool ArcherPrefetchHandle::IsTensorOffloaded(const std::uint32_t tensor_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = kTensorIndex->find(tensor_id);
  // DLOG_TRACE("Check tensor {} {}", tensor_id, it == kTensorIndex->end());
  bool is_offloaded = it != kTensorIndex->end();
  if (is_offloaded) {
    it->second.id = tensor_id;
  }
  return is_offloaded;
}

void ArcherPrefetchHandle::SetTensorDevice(torch::Tensor& tensor,
                                           torch::Device device) const {
  void* device_ptr = nullptr;
  auto byte_size = tensor.element_size() * tensor.numel();

  DLOG_TRACE("Set tensor to device ", (void*)tensor.data_ptr(), device.str());

  // then copy to target device
  cudaSetDevice(device.index());
  cudaMalloc(&device_ptr, byte_size);

  CudaMemcpy(device_ptr, tensor.data_ptr(), byte_size,
             cudaMemcpyDeviceToDevice);

  auto new_tensor = torch::from_blob(
      device_ptr, tensor.sizes(), [](void* ptr) { cudaFree(ptr); },
      tensor.options().device(device).pinned_memory(false));
  tensor.set_data(new_tensor);
}

bool ArcherPrefetchHandle::IsTensorIndexInitialized() const {
  return kArcherTensorHandle->IsTensorIndexInitialized();
}

bool ArcherPrefetchHandle::IsTensorOnDevice(const torch::Tensor& tensor) const {
  auto tensor_id = kArcherTensorHandle->GetTensorId((void*)tensor.data_ptr());
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
  return node->device.is_cuda();
}

void ArcherPrefetchHandle::UpdateTensorMap(std::uint64_t old_data_ptr,
                                           std::uint64_t new_data_ptr) {
  kArcherTensorHandle->UpdateTensorMap((void*)old_data_ptr,
                                       (void*)new_data_ptr);
}

bool ArcherPrefetchHandle::IsTensorOnDevice(const TensorID tensor_id) const {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
  return node->device.is_cuda();
}

int ArcherPrefetchHandle::GetNodeDefaultDevice(
    std::vector<std::uint32_t> tensor_ids) const {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_ids[0]);
  // DLOG_TRACE("Get node {} default device {}", node->str(),
  return node->default_device.index();
}

int ArcherPrefetchHandle::GetNodeDevice(
    std::vector<std::uint32_t> tensor_ids) const {
  auto node = kTopologyHandle->GetNodeFromTensorID(tensor_ids[0]);
  // DLOG_TRACE("Get node {} device {}", node->str(), node->device.str());
  return node->device.index();
}

// void ArcherPrefetchHandle::SetNodeCachePriority(const std::uint32_t
// tensor_id, const float priority) {
//     auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
//     node->cache_priority = priority;
// }
