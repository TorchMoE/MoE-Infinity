// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_prefetch_handle.h"
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include "aio/archer_tensor_handle.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "memory/memory_pool.h"
#include "task_scheduler.h"

#include "utils/archer_logger.h"

ArcherPrefetchHandle::ArcherPrefetchHandle(const std::string& prefix,
                                           const double device_memory_ratio)
    : prefix_(prefix), last_layer_id_(0)
{
    InitLogger();
    kArcherTensorHandle = std::make_unique<ArcherTensorHandle>(prefix);
    kTopologyHandle = std::make_unique<ArcherTopologyHandle>();
    kTaskPool = std::make_unique<ArcherTaskPool>();
    kDeviceMemoryPool->SetMemoryRatio(device_memory_ratio);
    ARCHER_LOG_DEBUG("Free Device Memory ", kDeviceMemoryPool->GetFreeMemory(CUDA_DEVICE(0)));

    if (prefix_.back() != '/') { prefix_ += '/'; }

    // enable peer access for kernels
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    ARCHER_LOG_INFO("Device count ", device_count);

    for (int i = 0; i < device_count; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < device_count; j++) {
            if (i != j) { cudaDeviceEnablePeerAccess(j, 0); }
        }
    }

    ARCHER_LOG_INFO("Enabled peer access for all devices");
}

ArcherPrefetchHandle::~ArcherPrefetchHandle()
{
    // served as a global manager for order of destruction
    kTaskPool.reset();
    kArcherTensorHandle.reset();
}

void ArcherPrefetchHandle::AcquireTensor(std::uint64_t& request_id, torch::Tensor& buffer)
{
    auto tensor_id = kArcherTensorHandle->GetTensorId((void*)buffer.data_ptr());
    void* old_ptr = (void*)buffer.data_ptr();
    ARCHER_LOG_DEBUG("Acquire tensor ", tensor_id, old_ptr);

    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    node->state = 1;

    // add node tensor_ids to node_id_to_tensor_ids_
    if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end() ||
        node_id_to_tensor_ids_[node->id].size() == 0) {
        node_id_to_tensor_ids_[node->id] = std::unordered_set<std::uint32_t>();
        for (auto& tensor_id : node->tensor_ids) {
            node_id_to_tensor_ids_[node->id].insert(tensor_id);
        }

        auto node_body = kTopologyHandle->GetNodeBodyFromCorrID(node->corr_id);
        if (node->device.is_cuda()) { node_body->gpu_hit_cnt++; }

        // always lock node, wait for previous prefetch task to finish
        node->mutex.lock();
        std::unique_lock<std::mutex> lock(node->mutex, std::adopt_lock);

        if (node->is_sparse) {
            bool success = kTaskPool->RemoveCachedSparseNode(node);
            if (!success) node->is_overflow = true;
        } else {
            kTaskPool->RemoveCachedDenseNode(node);
        }
        kTaskPool->StartExec(request_id, node);
        node->cv.wait(lock, [node] { return node->state == 0; });
    }

    kArcherTensorHandle->SetTensor(tensor_id, buffer);
    kArcherTensorHandle->UpdateTensorMap(old_ptr, (void*)buffer.data_ptr());
}
void ArcherPrefetchHandle::ReleaseTensor(std::uint64_t& request_id, torch::Tensor& buffer)
{
    auto tensor_id = kArcherTensorHandle->GetTensorId((void*)buffer.data_ptr());
    void* old_ptr = (void*)buffer.data_ptr();
    ARCHER_LOG_DEBUG("Release tensor ", tensor_id, old_ptr);

    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    // node->state = 1;

    if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end()) {
        ARCHER_LOG_ERROR("Node not found in node_id_to_tensor_ids_", node->str());
        return;
    }

    /*  This needs to go after Release, default host can be changed in TraceRequest
     *   Faulty case: node -> default_host = cpu, node -> default_host = cuda; tensor already
     * released
     */
    // if (node != last_node_) {
    //     // kTaskPool->Prefetch(request_id, node);
    //     TraceRequest(request_id, tensor_id);
    // }
    // TraceRequest(request_id, tensor_id);

    auto current_layer_id = node->corr_id & 0xFFFFFFFF;
    if (current_layer_id != last_layer_id_ && node_id_to_tensor_ids_[last_node_->id].size() != 0) {
        node_id_to_tensor_ids_[last_node_->id].clear();
        kTaskPool->StopExec(request_id, last_node_);  // evict last node to cpu or disk
        last_node_->mutex.unlock();
    }
    last_layer_id_ = current_layer_id;
    last_node_ = node;

    node_id_to_tensor_ids_[node->id].erase(tensor_id);
    // ARCHER_LOG_DEBUG(
    //     "Node {} tensor_ids size {}", node->id, node_id_to_tensor_ids_[node->id].size());

    if (node_id_to_tensor_ids_[node->id].size() == 0) {
        kTaskPool->StopExec(request_id, node);  // FIXME: change api to add request id
        // always unlock node here since, exec queue do not unlock automatically
        node->mutex.unlock();
    }

    if (kTopologyHandle->IsLastNode(node)) {
        ARCHER_LOG_DEBUG("Node is last, clean up", node->str());
        request_id_to_nodes_.erase(request_id);
    }

    at::TensorOptions options;
    options = options.device(torch::kCPU);
    options = options.dtype(buffer.dtype());
    auto zero_tensor = torch::zeros({1}, options);
    buffer.set_data(zero_tensor);
    kArcherTensorHandle->UpdateTensorMap(old_ptr, (void*)buffer.data_ptr());
}

void ArcherPrefetchHandle::PrefetchTensors(std::uint64_t& request_id,
                                           const std::vector<std::uint32_t>& buffer)
{
    std::vector<NodePtr> candidates;
    for (std::uint32_t tensor_id : buffer) {
        auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
        candidates.push_back(node);
    }

    if (candidates.size() == 0) { return; }
}

void ArcherPrefetchHandle::ReplaceCacheCandidates(const std::vector<std::uint32_t>& tensor_ids)
{
    std::vector<NodePtr> candidates;
    for (std::uint32_t tensor_id : tensor_ids) {
        auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
        candidates.push_back(node);
    }

    kTaskPool->ReplaceCacheCandidates(candidates);
}
void ArcherPrefetchHandle::EnqueuePrefetch(const uint32_t tensor_id, int gpu_id)
{
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);

    auto task = std::make_shared<Task>();
    task->priority = 1;
    task->node = node;
    task->on_demand = false;
    task->src_device = node->device;
    // task->dst_device = CUDA_DEVICE(gpu_id); // use default device for now
    task->dst_device = node->default_device;
    kTaskPool->EnqueueTask(task);
}

void ArcherPrefetchHandle::FetchTensors(std::uint64_t& request_id,
                                        const std::vector<std::uint32_t>& buffer)
{
    // std::vector<NodePtr> candidates;
    for (std::uint32_t tensor_id : buffer) {
        auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
        kTaskPool->FetchExec(request_id, node);
    }
}

void ArcherPrefetchHandle::OffloadTensor(torch::Tensor& tensor, const std::uint32_t tensor_id)
{
    kArcherTensorHandle->StoreTensor(tensor_id, tensor);

    auto ckpt_index_path = prefix_ + std::string(ARCHER_IHDEX_NAME);

    std::unique_lock<std::mutex> lock(mutex_);
    kTensorIndex->Serialize(ckpt_index_path.c_str());
}

void ArcherPrefetchHandle::RegisterTensor(torch::Tensor& tensor, const std::uint32_t tensor_id)
{
    kArcherTensorHandle->RegisterTensor(tensor_id, tensor);
}

void ArcherPrefetchHandle::RegisterModule(torch::nn::Module& module)
{
    for (auto it = module.parameters().begin(); it != module.parameters().end(); ++it) {
        auto tensor_id = kArcherTensorHandle->GetTensorId((void*)(*it).unsafeGetTensorImpl());
        kArcherTensorHandle->RegisterTensor(tensor_id, *it);
    }

    for (auto it = module.buffers().begin(); it != module.buffers().end(); ++it) {
        auto tensor_id = kArcherTensorHandle->GetTensorId((void*)(*it).unsafeGetTensorImpl());
        kArcherTensorHandle->RegisterTensor(tensor_id, *it);
    }
}

void ArcherPrefetchHandle::RegisterTensor(torch::Tensor* tensor)
{
    ARCHER_LOG_DEBUG("Register tensor: is view ", (void*)tensor, tensor->is_view());
}

torch::Tensor ArcherPrefetchHandle::GetTrace()
{
    const auto& child_visit_cnts = kTopologyHandle->GetChildVisitCounts();
    const auto num_layers_and_experts = kTopologyHandle->GetNumLayersAndExperts();
    const auto num_layers = std::get<0>(num_layers_and_experts);
    const auto num_experts = std::get<1>(num_layers_and_experts);

    std::vector<int64_t> trace_vec(child_visit_cnts.begin(), child_visit_cnts.end());
    torch::Tensor trace = torch::from_blob(trace_vec.data(),
                                           {static_cast<int64_t>(num_layers - 1),
                                            static_cast<int64_t>(num_experts),
                                            static_cast<int64_t>(num_experts)},
                                           torch::kInt64)
                              .clone();

    return trace;
}

torch::Tensor ArcherPrefetchHandle::GetHitRate()
{
    const auto& node_visit_cnts = kTopologyHandle->GetNodeVisitCounts();

    // flatten vector of vectors
    std::vector<int64_t> node_visit_cnts_vec;
    for (auto& node_visit_cnt : node_visit_cnts) {
        node_visit_cnts_vec.insert(
            node_visit_cnts_vec.end(), node_visit_cnt.begin(), node_visit_cnt.end());
    }

    torch::Tensor trace = torch::from_blob(node_visit_cnts_vec.data(),
                                           {node_visit_cnts.size(), node_visit_cnts[0].size()},
                                           torch::kInt64)
                              .clone();
    return trace;
}

void ArcherPrefetchHandle::SetTrace(const torch::Tensor& trace)
{
    if (trace.dim() != 3 || !trace.is_contiguous() || !trace.is_cpu()) {
        ARCHER_LOG_ERROR("Trace should be a contiguous 3D tensor on CPU");
        return;
    }

    std::vector<std::size_t> child_visit_cnts(trace.data_ptr<int64_t>(),
                                              trace.data_ptr<int64_t>() + trace.numel());
    kTopologyHandle->SetChildVisitCounts(child_visit_cnts);
}

void ArcherPrefetchHandle::TraceRequest(const std::uint64_t request_id, const TensorID tensor_id)
{
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);

    auto it = request_id_to_nodes_.find(request_id);
    if (it == request_id_to_nodes_.end()) {
        request_id_to_nodes_[request_id] = std::unordered_set<NodePtr>();
    }

    auto node_it = request_id_to_nodes_[request_id].find(node);
    if (node_it != request_id_to_nodes_[request_id].end()) {
        ARCHER_LOG_DEBUG("Node already traced for request ", request_id, node->str());
        return;
    }

    request_id_to_nodes_[request_id].insert(node);
}

void ArcherPrefetchHandle::SetTopology(
    const std::vector<std::tuple<std::string, std::vector<std::vector<TensorID>>>>& topology)
{
    kTopologyHandle->InitializeTopology(topology);
}

bool ArcherPrefetchHandle::IsTensorOffloaded(const std::uint32_t tensor_id)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = kTensorIndex->find(tensor_id);
    // ARCHER_LOG_DEBUG("Check tensor {} {}", tensor_id, it == kTensorIndex->end());
    bool is_offloaded = it != kTensorIndex->end();
    if (is_offloaded) { it->second.id = tensor_id; }
    return is_offloaded;
}

void ArcherPrefetchHandle::SetTensorDevice(torch::Tensor& tensor, torch::Device device) const
{
    void* device_ptr = nullptr;
    auto byte_size = tensor.element_size() * tensor.numel();

    ARCHER_LOG_DEBUG("Set tensor to device ", (void*)tensor.data_ptr(), device.str());

    // then copy to target device
    cudaSetDevice(device.index());
    cudaMalloc(&device_ptr, byte_size);

    cudaMemcpy(device_ptr, tensor.data_ptr(), byte_size, cudaMemcpyDeviceToDevice);

    auto new_tensor = torch::from_blob(
        device_ptr,
        tensor.sizes(),
        [](void* ptr) { cudaFree(ptr); },
        tensor.options().device(device).pinned_memory(false));
    tensor.set_data(new_tensor);
}

bool ArcherPrefetchHandle::IsTensorIndexInitialized() const
{
    return kArcherTensorHandle->IsTensorIndexInitialized();
}

bool ArcherPrefetchHandle::IsTensorOnDevice(const torch::Tensor& tensor) const
{
    auto tensor_id = kArcherTensorHandle->GetTensorId((void*)tensor.data_ptr());
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    return node->device.is_cuda();
}

void ArcherPrefetchHandle::UpdateTensorMap(std::uint64_t old_data_ptr, std::uint64_t new_data_ptr)
{
    kArcherTensorHandle->UpdateTensorMap((void*)old_data_ptr, (void*)new_data_ptr);
}

bool ArcherPrefetchHandle::IsTensorOnDevice(const TensorID tensor_id) const
{
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    return node->device.is_cuda();
}

int ArcherPrefetchHandle::GetNodeDefaultDevice(std::vector<std::uint32_t> tensor_ids) const
{
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_ids[0]);
    // ARCHER_LOG_DEBUG("Get node {} default device {}", node->str(),
    return node->default_device.index();
}

int ArcherPrefetchHandle::GetNodeDevice(std::vector<std::uint32_t> tensor_ids) const
{
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_ids[0]);
    // ARCHER_LOG_DEBUG("Get node {} device {}", node->str(), node->device.str());
    return node->device.index();
}

// void ArcherPrefetchHandle::SetNodeCachePriority(const std::uint32_t tensor_id, const float priority) {
//     auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
//     node->cache_priority = priority;
// }
