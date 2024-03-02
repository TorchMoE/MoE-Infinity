// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "model_topology.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <climits>
#include <cmath>
#include <sstream>
#include "aio/archer_prio_aio_handle.h"
#include "aio/archer_tensor_handle.h"
#include "aio/archer_tensor_index.h"
#include "common/time.h"
#include "common/types.h"
#include "memory/memory_pool.h"
#include "memory/stream_pool.h"
#include "parallel/expert_dispatcher.h"
#include "prefetch/task_scheduler.h"
#include "utils/archer_logger.h"

cudaStream_t kCudaStreamH2D = NULL;
std::unique_ptr<ArcherTopologyHandle> kTopologyHandle = nullptr;

const std::string Node::str() noexcept
{
    // write same string using c style sprintf
    std::stringstream ss;
    for (auto& tensor_id : tensor_ids) { ss << tensor_id << ","; }

    char buffer[1024];
    memset(buffer, 0, 1024);
    sprintf(buffer,
            "ID[%ld,%lx] (%ldMB) STATE(%d) TENSOR[%s] DEVICE[%s;%s;%s];",
            id,
            corr_id,
            byte_size / MB,
            state.load(),
            ss.str().c_str(),
            device.str().c_str(),
            default_device.str().c_str(),
            default_host.str().c_str());

    return std::string(buffer);
}

Node::Node()
    : corr_id(0),
      byte_size(0),
      last_access_time(MCIROSECONDS_SINCE_EPOCH),
      device(DISK_DEVICE),
      default_device(DEFAULT_CUDA_DEVICE)
{
}

void Node::SetDevice(const torch::Device& target_device,
                     bool on_demand,
                     cudaStream_t stream) noexcept
{
    ARCHER_LOG_DEBUG("SetDevice: ", str(), " to ", target_device.str());
    if (device == target_device) {
        ARCHER_LOG_DEBUG("SetDevice: " + str() + " to " + target_device.str() +
                         " but device is the same");
        return;
    }

    if (device.type() == target_device.type()) {
        ARCHER_LOG_WARN("SetDevice: " + str() + " to " + target_device.str() +
                        " but device type is the same");
        return;
    }

    if (kCudaStreamH2D == NULL) {
        auto cudaError = cudaStreamCreateWithFlags(&kCudaStreamH2D, cudaStreamNonBlocking);
        if (cudaError != cudaSuccess) {
            ARCHER_LOG_ERROR("cudaStreamCreate failed: ", cudaGetErrorString(cudaError));
            exit(-1);
        }
    }

    if (target_device == DISK_DEVICE) {
        SetModuleDisk(tensor_ids);
        if (host_memory_ptr != nullptr) {
            kHostMemoryPool->FreeMemory(id, host_memory_ptr, byte_size, CPU_DEVICE);
            host_memory_ptr = nullptr;
        }
        if (device_memory_ptr != nullptr) {
            kDeviceMemoryPool->FreeMemory(id, device_memory_ptr, byte_size, device);
            device_memory_ptr = nullptr;
        }
    } else {
        // both are null, which means the node is not initialized
        if (host_memory_ptr == nullptr && device_memory_ptr == nullptr) {
            // int numa_id =
            //     default_device.index() / 4;  // TODO: 8 gpus, 2 numa nodes, so 4 gpus per numa
            host_memory_ptr = kHostMemoryPool->AllocateMemory(id, byte_size, CPU_DEVICE);
            assert(host_memory_ptr != nullptr);

            auto start_time = MCIROSECONDS_SINCE_EPOCH;
            SetModuleMemoryFromDisk(tensor_ids, host_memory_ptr, on_demand);
            auto end_time = MCIROSECONDS_SINCE_EPOCH;
            ARCHER_LOG_DEBUG("SetModuleMemoryFromDisk time:", end_time - start_time, " us");
        }

        if (target_device.is_cuda()) {
            // ARCHER_LOG_DEBUG("Allocate GPU Memory for node {}", this->id);
            device_memory_ptr = kDeviceMemoryPool->AllocateMemory(id, byte_size, target_device);
            // ARCHER_LOG_DEBUG("Allocate GPU Memory for node {} done", this->id);
            assert(device_memory_ptr != nullptr);
            assert(host_memory_ptr != nullptr);

            auto start_time = MCIROSECONDS_SINCE_EPOCH;
            if (stream == nullptr) {
                cudaMemcpy(device_memory_ptr, host_memory_ptr, byte_size, cudaMemcpyHostToDevice);
            } else {
                cudaMemcpyAsync(
                    device_memory_ptr, host_memory_ptr, byte_size, cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
            }
            SetModuleCudaMemoryFromCPU(tensor_ids, device_memory_ptr, target_device);
            auto end_time = MCIROSECONDS_SINCE_EPOCH;
            ARCHER_LOG_DEBUG("SetModuleCudaMemoryFromCPU time: {} us", end_time - start_time);
        }

        if (target_device.is_cpu() && device.is_cuda()) {
            assert(host_memory_ptr != nullptr);
            auto start_time = MCIROSECONDS_SINCE_EPOCH;
            SetModuleMemoryFromCuda(tensor_ids, host_memory_ptr);
            kDeviceMemoryPool->FreeMemory(id, device_memory_ptr, byte_size, device);
            device_memory_ptr = nullptr;
            auto end_time = MCIROSECONDS_SINCE_EPOCH;
            ARCHER_LOG_DEBUG("SetModuleMemoryFromCuda time: {} us", end_time - start_time);
        }
    }
    device = target_device;
}

ArcherTopologyHandle::ArcherTopologyHandle() {}

NodePtrList ArcherTopologyHandle::GetLFUNodes(const torch::Device& device)
{
    NodePtrList nodes;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto node_body : lfu_nodes_) {
        CONTINUE_IF_NULL(node_body);
        if (node_body->node->device == device) { nodes.push_back(node_body->node); }
    }
    return nodes;
}

NodePtrList ArcherTopologyHandle::GetDenseNodes()
{
    NodePtrList nodes;
    for (auto stage : pipeline_.stages) {
        if (stage->is_sparse) { continue; }
        for (auto node_body : stage->nodes) { nodes.push_back(node_body->node); }
    }
    return nodes;
}
NodePtrList ArcherTopologyHandle::GetSparseNodes()
{
    NodePtrList nodes;
    for (auto stage : pipeline_.stages) {
        if (!stage->is_sparse) { continue; }
        for (auto node_body : stage->nodes) { nodes.push_back(node_body->node); }
    }
    return nodes;
}

NodePtrList ArcherTopologyHandle::GetDenseNodes(const NodePtr& node, const std::size_t& k)
{
    NodePtrList nodes;

    std::size_t low_corr_id = node->corr_id & 0xFFFFFFFF;  // stage id
    std::size_t high_corr_id = node->corr_id >> 32;        // node id
    bool is_last_node = (0xFFFFFFFF == high_corr_id);
    if (is_last_node) {
        high_corr_id = 0;  // reset to 0 avoid miss use
    }

    std::lock_guard<std::mutex> lock(mutex_);

    low_corr_id++;
    std::size_t count = 0;
    while ((low_corr_id < pipeline_.stages.size()) && (count < k)) {
        // Due to MoE design, we only process layer by layer
        auto stage = pipeline_.stages[low_corr_id];
        low_corr_id++;
        if (stage->is_sparse) { continue; }

        nodes.push_back(stage->nodes[0]->node);
        count++;
    }
    return nodes;
}

NodePtrList ArcherTopologyHandle::GetSparseNodes(const NodePtr& node, const std::size_t& k)
{
    NodePtrList nodes;

    std::size_t low_corr_id = node->corr_id & 0xFFFFFFFF;  // stage id
    std::size_t high_corr_id = node->corr_id >> 32;        // node id
    bool is_last_node = (0xFFFFFFFF == high_corr_id);
    if (is_last_node) {
        high_corr_id = 0;  // reset to 0 avoid miss use
    }

    std::lock_guard<std::mutex> lock(mutex_);

    low_corr_id++;
    std::size_t count = 0;
    while ((low_corr_id < pipeline_.stages.size()) && (count < k)) {
        // Due to MoE design, we only process layer by layer
        auto stage = pipeline_.stages[low_corr_id];

        low_corr_id++;
        if (!stage->is_sparse) { continue; }

        nodes.push_back(stage->nodes[0]->node);
        count++;
    }
    return nodes;
}

std::uint64_t ArcherTopologyHandle::GetLastActivateStage(const HashID& hash_id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = last_active_stage_.find(hash_id);
    if (it == last_active_stage_.end()) { return 0; }
    return it->second;
}

std::vector<std::vector<std::size_t>> ArcherTopologyHandle::GetNodeVisitCounts()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::vector<std::size_t>> node_visit_counts;
    for (auto& stage : pipeline_.stages) {
        for (auto& node : stage->nodes) {
            node->node->io_state = NODE_STATE_NONE;
            std::vector<std::size_t> metrics{node->visit_cnt,
                                             node->gpu_visit_cnt,
                                             node->cpu_visit_cnt,
                                             node->hit_cnt,
                                             node->gpu_hit_cnt,
                                             node->cpu_hit_cnt,
                                             node->node->tensor_ids.size(),
                                             node->prefetch_cnt,
                                             node->node->unused_count,
                                             node->node->io_state,
                                             node->is_sparse};
            node_visit_counts.push_back(metrics);
        }
    }
    return node_visit_counts;
}

std::vector<std::size_t> ArcherTopologyHandle::GetChildVisitCounts()
{
    std::lock_guard<std::mutex> lock(mutex_);
    int num_layers = 0;
    int num_experts = 0;
    for (auto& stage : pipeline_.stages) {
        if (stage->is_sparse) {
            num_layers += 1;
            num_experts = stage->nodes.size();
        }
    }
    std::vector<std::size_t> child_visit_counts((num_layers - 1) * num_experts * num_experts);
    int layer_idx = 0;
    int parent_idx = 0;
    int expert_idx = 0;
    for (auto& stage : pipeline_.stages) {
        if (stage->is_sparse) {
            for (auto& node : stage->nodes) {
                if (node->children.size() > 0) {
                    for (auto& count : node->children_visit_cnt) {
                        child_visit_counts[layer_idx * num_experts * num_experts +
                                           parent_idx * num_experts + expert_idx] = count;
                        expert_idx++;
                    }
                }
                parent_idx++;
                expert_idx = 0;
            }
            layer_idx++;
            parent_idx = 0;
        }
    }

    return child_visit_counts;
}

void ArcherTopologyHandle::SetNodeVisitCounts(const std::vector<std::size_t>& visit_counts)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t num_nodes = 0;
    std::size_t num_experts = 0;
    for (auto& stage : pipeline_.stages) {
        if (stage->is_sparse) {
            num_nodes += stage->nodes.size();
            num_experts = stage->nodes.size();
        }
    }
    if (visit_counts.size() != num_nodes) {
        ARCHER_LOG_ERROR(
            "visit_counts size {} not equal to num_nodes {}", visit_counts.size(), num_nodes);
        return;
    }

    int layer_idx = 0;
    int expert_idx = 0;
    for (auto& stage : pipeline_.stages) {
        if (stage->is_sparse) {
            for (auto& node : stage->nodes) {
                node->visit_cnt = visit_counts[layer_idx * num_experts + expert_idx];
                expert_idx++;
            }
            layer_idx++;
            expert_idx = 0;
        }
    }

    DisableTrace();
}
void ArcherTopologyHandle::SetChildVisitCounts(const std::vector<std::size_t>& visit_counts)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::size_t num_layers = 0;
    std::size_t num_experts = 0;
    for (auto& stage : pipeline_.stages) {
        if (stage->is_sparse) {
            num_layers += 1;
            num_experts = stage->nodes.size();
        }
    }
    if (visit_counts.size() != (num_layers - 1) * num_experts * num_experts) {
        ARCHER_LOG_ERROR(
            "visit_counts size {} not equal to num_layers {}", visit_counts.size(), num_layers);
        return;
    }

    int layer_idx = 0;
    int parent_idx = 0;
    int expert_idx = 0;
    for (auto& stage : pipeline_.stages) {
        if (stage->is_sparse) {
            for (auto& node : stage->nodes) {
                if (node->children.size() > 0) {
                    for (auto& count : node->children_visit_cnt) {
                        count = visit_counts[layer_idx * num_experts * num_experts +
                                             parent_idx * num_experts + expert_idx];
                        expert_idx++;
                    }
                }
                parent_idx++;
                expert_idx = 0;
            }
            layer_idx++;
            parent_idx = 0;
        }
    }

    DisableTrace();
}

bool ArcherTopologyHandle::IsLastNode(const NodePtr& node)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto last_stage_ptr = pipeline_.stages.back();
    auto& nodes = last_stage_ptr->nodes;
    for (auto& n : nodes) {
        if (n->node == node) { return true; }
    }
    return false;
}
bool ArcherTopologyHandle::IsFirstNode(const NodePtr& node)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto first_stage_ptr = pipeline_.stages.front();
    auto& nodes = first_stage_ptr->nodes;
    for (auto& n : nodes) {
        if (n->node == node) { return true; }
    }
    return false;
}

void ArcherTopologyHandle::InitializeTopology(
    const std::vector<std::tuple<std::string, std::vector<std::vector<TensorID>>>>& topology)
{
    std::lock_guard<std::mutex> lock(mutex_);
    pipeline_.stages.clear();
    std::size_t node_id = 0;
    std::size_t layer_id = 0;
    std::size_t last_sparse_layer_id = UINT64_MAX;

    size_t num_sparse_layers = 0;
    size_t num_experts = 0;

    std::vector<NodePtr> all_nodes;

    for (auto& stage : topology) {
        auto& stage_tensors = std::get<1>(stage);
        auto stage_ptr = std::make_shared<Stage>(stage_tensors.size() > 1);

        std::size_t expert_id = 0;
        for (auto& tensor_ids : stage_tensors) {
            auto node_ptr = std::make_shared<Node>();
            node_ptr->tensor_ids = tensor_ids;
            int64_t byte_size = 0;
            for (auto& tensor_id : tensor_ids) {
                auto it = kTensorIndex->find(tensor_id);
                if (it != kTensorIndex->end()) {
                    std::int64_t size_aligned =
                        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
                    byte_size += size_aligned;
                } else {
                    ARCHER_LOG_ERROR("Tensor {} not found in tensor index", tensor_id);
                }
            }
            node_ptr->byte_size = byte_size;
            node_ptr->id = node_id;
            node_ptr->corr_id = (layer_id & 0xFFFFFFFF) | ((expert_id & 0xFFFFFFFF) << 32);
            node_ptr->is_sparse = stage_ptr->is_sparse;

            all_nodes.push_back(node_ptr);

            auto node_body_ptr = std::make_shared<NodeBody>(node_ptr);
            node_body_ptr->is_sparse = stage_ptr->is_sparse;

            stage_ptr->nodes.push_back(node_body_ptr);

            node_id++;
            expert_id++;
        }
        pipeline_.stages.push_back(stage_ptr);
        auto current_layer_id = layer_id;
        layer_id++;

        if (stage_ptr->is_sparse) {
            if (UINT64_MAX == last_sparse_layer_id) {
                last_sparse_layer_id = current_layer_id;
                continue;
            }
            // set node_body_ptr vectors to be the same size as the number of experts
            // all counts initialized to 0
            auto last_sparse_stage_ptr = pipeline_.stages[last_sparse_layer_id];
            for (auto& node : last_sparse_stage_ptr->nodes) {
                node->children_visit_cnt.resize(stage_ptr->nodes.size(), 0);
                node->children = stage_ptr->nodes;
            }
            last_sparse_layer_id = current_layer_id;

            num_sparse_layers++;
            num_experts = stage_ptr->nodes.size();
        }
    }

    // set last stage nodes corr_id higher 32 bits to be 0xFFFFFFFF
    auto last_stage_ptr = pipeline_.stages.back();
    for (auto& node_body : last_stage_ptr->nodes) {
        node_body->node->corr_id = (node_body->node->corr_id & 0xFFFFFFFF) | (UINT64_MAX << 32);
    }

    // output every tensor id in node
    for (auto& stage : pipeline_.stages) {
        for (auto& node : stage->nodes) {
            std::stringstream ss;
            for (auto& tensor_id : node->node->tensor_ids) { ss << tensor_id << " "; }
            // ARCHER_LOG_DEBUG("Node {} tensor ids {}", node->node->id, ss.str());
            lfu_nodes_.push_back(node);
        }
    }

    ARCHER_LOG_DEBUG("InitializeTopology pipeline_.stages.size() {}", pipeline_.stages.size());

    // Model placement
    auto num_gpu = GetDeviceCount();
    std::vector<std::int64_t> free_device_mem(num_gpu, 0);
    for (int i = 0; i < num_gpu; i++) {
        free_device_mem[i] = kDeviceMemoryPool->GetMemoryCapacity(torch::Device(torch::kCUDA, i));
    }

    auto sparse_nodes = GetSparseNodes();
    auto dense_nodes = GetDenseNodes();

    ARCHER_LOG_DEBUG("InitializeTopology num_gpu {} sparse_nodes.size() {} dense_nodes.size() {}",
                     num_gpu,
                     sparse_nodes.size(),
                     dense_nodes.size());

    int target_device_id = 0;
    int dense_gpu_idx = 0;
    int sparse_gpu_idx = 0;

    // Split evently dense nodes only
    int num_dense_nodes_per_device = std::ceil(dense_nodes.size() / num_gpu / 2);
    // int total_dense_nodes = dense_nodes.size();
    int counter = 0;
    for (auto& node_ptr : dense_nodes) {
        // split dense node evenly among GPUs
        node_ptr->default_device = torch::Device(torch::kCUDA, target_device_id);
        counter++;
        if (counter % num_dense_nodes_per_device == 0) {
            target_device_id = (target_device_id + 1) % num_gpu;
        }
    }
    dense_nodes.back()->default_device = torch::Device(torch::kCUDA, num_gpu - 1);

    // split evenly sparse nodes among GPUs
    for (auto& node_ptr : sparse_nodes) {
        node_ptr->default_device = torch::Device(torch::kCUDA, target_device_id);
        target_device_id = (target_device_id + 1) % num_gpu;
    }

    ARCHER_LOG_DEBUG("InitializeTopology pipeline_.stages.size() {}", pipeline_.stages.size());

    for (auto& node_ptr : all_nodes) {
        ARCHER_LOG_DEBUG("Node {} {} device {}",
                         node_ptr->id,
                         node_ptr->is_sparse,
                         node_ptr->default_device.str());
    }

    EnableTrace();
}

NodePtr ArcherTopologyHandle::GetNodeFromTensorID(const TensorID& tensor_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tensor_id_to_node_.find(tensor_id);
    if (it != tensor_id_to_node_.end()) {
        return it->second;
    } else {
        // search in pipeline
        for (auto& stage : pipeline_.stages) {
            for (auto& node_body : stage->nodes) {
                for (auto& id : node_body->node->tensor_ids) {
                    if (id == tensor_id) {
                        tensor_id_to_node_[tensor_id] = node_body->node;
                        return node_body->node;
                    }
                }
            }
        }
    }
    ARCHER_LOG_ERROR("Tensor {} not found in tensor id to node map", tensor_id);
    return nullptr;
}

NodeBodyPtr ArcherTopologyHandle::GetNodeBodyFromCorrID(const std::uint64_t& correlation_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::uint64_t high_corr_id = correlation_id >> 32;        // For children in the same level
    std::uint64_t low_corr_id = correlation_id & 0xFFFFFFFF;  // For model inference pipeline

    bool is_last_node = (0xFFFFFFFF == high_corr_id);
    if (is_last_node) {
        high_corr_id = 0;  // reset to 0 avoid miss use
    }

    auto stage = pipeline_.stages[low_corr_id];
    auto node_body = stage->nodes[high_corr_id];

    return node_body;
}

std::int64_t ArcherTopologyHandle::GetSparseCacheLimit(const torch::Device& device)
{
    std::int64_t dense_cache_size = 0;
    for (auto& stage : pipeline_.stages) {
        for (auto& node_body : stage->nodes) {
            if (stage->is_sparse) continue;
            if (node_body->node->device == device) {
                dense_cache_size += node_body->node->byte_size;
            }
        }
    }

    std::int64_t device_size_limit = (device.is_cuda())
                                         ? kDeviceMemoryPool->GetMemoryCapacity(device)
                                         : kHostMemoryPool->GetMemoryCapacity();
    assert(device_size_limit > dense_cache_size);
    std::int64_t sparse_cache_size = device_size_limit - dense_cache_size;

    return sparse_cache_size;
}

std::tuple<std::size_t, std::size_t> ArcherTopologyHandle::GetNumLayersAndExperts()
{
    std::lock_guard<std::mutex> lock(mutex_);
    int num_layers = 0;
    int num_experts = 0;
    for (auto& stage : pipeline_.stages) {
        if (stage->is_sparse) {
            num_layers += 1;
            num_experts = stage->nodes.size();
        }
    }
    return std::make_tuple(num_layers, num_experts);
}

// CPU, GPU -> DISK
// Moves tensors from CPU/GPU to disk.
void SetModuleDisk(std::vector<TensorID>& tensor_ids)
{
    // ARCHER_LOG_DEBUG("SetModuleDisk {} tensors", tensor_ids.size());
    for (const auto& tensor_id : tensor_ids) {
        // void* old_ptr = kTensorIndex->find(tensor_id)->second.tensor.data_ptr();
        auto it = kTensorIndex->find(tensor_id);

        at::TensorOptions options;
        options = options.device(torch::kCPU);
        options = options.dtype(it->second.tensor.dtype());
        auto tensor = torch::zeros({1}, options);
        it->second.tensor.set_data(tensor);
    }
}

std::mutex kReadMutex;

// DISK -> CPU
void SetModuleMemoryFromDisk(std::vector<TensorID>& tensor_ids, void* host_ptr, bool on_demand)
{
    std::int64_t param_size = 0;
    for (const auto& tensor_id : tensor_ids) {
        // void* old_ptr = kTensorIndex->find(tensor_id)->second.tensor.data_ptr();
        kArcherTensorHandle->ReadTensor(
            tensor_id, (void*)((char*)host_ptr + param_size), on_demand);
        auto it = kTensorIndex->find(tensor_id);
        auto options = torch::TensorOptions()
                           .dtype(it->second.options.dtype())
                           .layout(it->second.options.layout())
                           .device(torch::kCPU)
                           .requires_grad(it->second.options.requires_grad())
                           .pinned_memory(it->second.options.pinned_memory());

        ARCHER_LOG_DEBUG("SetModuleMemoryFromDisk tensor {}", it->second.DebugString());
        auto tensor_tmp = torch::from_blob((void*)((char*)host_ptr + param_size),
                                           it->second.shape,
                                           DoNothingDeleter<void>{},
                                           options);
        if (!it->second.tensor.defined()) { it->second.tensor = torch::zeros({1}, options); }
        it->second.tensor.set_data(tensor_tmp);
        std::int64_t size_aligned = (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
        param_size += size_aligned;
    }
}

// CPU -> GPU
void SetModuleCudaMemoryFromCPU(std::vector<TensorID>& tensor_ids,
                                void* device_ptr,
                                const torch::Device& device)
{
    // ARCHER_LOG_DEBUG("SetModuleCudaMemoryFromCPU {} tensors", tensor_ids.size());
    std::int64_t param_size = 0;
    for (const auto& tensor_id : tensor_ids) {
        auto it = kTensorIndex->find(tensor_id);
        ARCHER_LOG_DEBUG(
            "SetModuleCudaMemoryFromCPU tensor {} -> {}", it->second.DebugString(), device.str());
        auto tensor_options = torch::TensorOptions()
                                  .dtype(it->second.options.dtype())
                                  .layout(it->second.options.layout())
                                  .device(device)
                                  .requires_grad(it->second.options.requires_grad())
                                  .pinned_memory(false);
        it->second.tensor.set_data(torch::from_blob((char*)device_ptr + param_size,
                                                    it->second.shape,
                                                    DoNothingDeleter<void>{},
                                                    tensor_options));
        std::int64_t size_aligned = (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
        param_size += size_aligned;
    }
    // ARCHER_LOG_DEBUG("SetModuleCudaMemoryFromCPU {} tensors done", tensor_ids.size());
}

// GPU -> CPU
void SetModuleMemoryFromCuda(std::vector<TensorID>& tensor_ids, void* host_ptr)
{
    std::int64_t param_size = 0;
    for (const auto& tensor_id : tensor_ids) {
        // void* old_ptr = kTensorIndex->find(tensor_id)->second.tensor.data_ptr();

        auto it = kTensorIndex->find(tensor_id);
        ARCHER_LOG_DEBUG("SetModuleMemoryFromCuda tensor ", it->second.DebugString());
        it->second.tensor.set_data(torch::from_blob((char*)host_ptr + param_size,
                                                    it->second.shape,
                                                    DoNothingDeleter<void>{},
                                                    it->second.options));
        // kArcherTensorHandle->UpdateTensorMap(old_ptr, it->second.tensor.data_ptr());
        std::int64_t size_aligned = (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
        param_size += size_aligned;
    }
    // ARCHER_LOG_DEBUG("SetModuleMemoryFromCuda {} tensors done", tensor_ids.size());
}
