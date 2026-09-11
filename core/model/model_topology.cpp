// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "model_topology.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <climits>
#include <cmath>
#include <limits>
#include <sstream>
#include "aio/archer_prio_aio_handle.h"
#include "aio/archer_tensor_handle.h"
#include "aio/archer_tensor_index.h"
#include "common/time.h"
#include "common/types.h"
#include "memory/event_pool.h"
#include "memory/memory_pool.h"
#include "memory/stream_pool.h"
#include "parallel/expert_dispatcher.h"
#include "prefetch/task_scheduler.h"
#include "utils/logger.h"
#include "utils/tqdm.h"

#include <fcntl.h>
#include <map>
#include <sys/stat.h>
#include <unistd.h>

// cudaStream_t kCudaStreamH2D = NULL;
std::unique_ptr<ArcherTopologyHandle> kTopologyHandle = nullptr;

const std::string Node::str() {
  // write same string using c style sprintf
  std::stringstream ss;
  for (auto& tensor_id : tensor_ids) {
    ss << tensor_id << ",";
  }

  char buffer[1024];
  memset(buffer, 0, 1024);
  sprintf(buffer, "ID[%ld,%lx] (%ldMB) STATE(%d) TENSOR[%s] DEVICE[%s;%s;%s];",
          id, corr_id, byte_size / MB, state.load(), ss.str().c_str(),
          device.str().c_str(), default_device.str().c_str(),
          default_host.str().c_str());

  return std::string(buffer);
}

Node::Node()
    : corr_id(0),
      byte_size(0),
      last_access_time(MCIROSECONDS_SINCE_EPOCH),
      device(DISK_DEVICE),
      default_device(DEFAULT_CUDA_DEVICE) {}

void Node::SetDevice(const torch::Device& target_device, bool on_demand,
                     cudaStream_t stream, cudaEvent_t* transfer_event) {
  if (transfer_event != nullptr) {
    *transfer_event = nullptr;
  }
  auto sync_stream_with_event = [](cudaStream_t sync_stream) {
    cudaEvent_t sync_event = kCudaEventPool->Acquire();
    cudaEventRecord(sync_event, sync_stream);
    cudaEventSynchronize(sync_event);
    kCudaEventPool->Release(sync_event);
  };
  DLOG_TRACE("SetDevice: " + str() + " to " + target_device.str());
  if (device == target_device) {
    DLOG_TRACE("SetDevice: " + str() + " to " + target_device.str() +
               " but device is the same");
    return;
  }

  if (device.type() == target_device.type()) {
    DLOG_WARN("SetDevice: " + str() + " to " + target_device.str() +
              " but device type is the same");
    return;
  }

  // if (kCudaStreamH2D == NULL) {
  //     auto cudaError = cudaStreamCreateWithFlags(&kCudaStreamH2D,
  //     cudaStreamNonBlocking); if (cudaError != cudaSuccess) {
  //         DLOG_ERROR("cudaStreamCreate failed: {}",
  //         cudaGetErrorString(cudaError)); exit(-1);
  //     }
  // }

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
    bool from_disk =
        (host_memory_ptr == nullptr && device_memory_ptr == nullptr);

    if (from_disk && target_device.is_cuda()) {
      // Pipelined path: disk -> host -> GPU with per-tensor overlap
      host_memory_ptr =
          kHostMemoryPool->AllocateMemory(id, byte_size, CPU_DEVICE);
      assert(host_memory_ptr != nullptr);
      device_memory_ptr =
          kDeviceMemoryPool->AllocateMemory(id, byte_size, target_device);
      assert(device_memory_ptr != nullptr);

      cudaStream_t h2d_stream = stream;
      bool own_stream = false;
      if (h2d_stream == nullptr) {
        cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking);
        own_stream = true;
      }

      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      std::int64_t param_offset = 0;
      for (const auto& tensor_id : tensor_ids) {
        // Read tensor from disk into host buffer
        {
#ifndef NVTX_DISABLE
          nvtx3::scoped_range r_disk_cpu("disk_to_cpu");
#endif
          kArcherTensorHandle->ReadTensor(
              tensor_id, static_cast<char*>(host_memory_ptr) + param_offset,
              on_demand);
        }

        auto it = kTensorIndex->find(tensor_id);
        std::int64_t size_aligned =
            (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);

        // Async copy this tensor's data to GPU (overlaps with next disk read)
        {
#ifndef NVTX_DISABLE
          nvtx3::scoped_range r_h2d("cpu_to_gpu");
#endif
          CudaMemcpyAsync(static_cast<char*>(device_memory_ptr) + param_offset,
                          static_cast<char*>(host_memory_ptr) + param_offset,
                          size_aligned, cudaMemcpyHostToDevice, h2d_stream);
        }

        param_offset += size_aligned;
      }
      {
#ifndef NVTX_DISABLE
        nvtx3::scoped_range r_sync("cuda_stream_sync");
#endif
        if (transfer_event != nullptr && !own_stream) {
          *transfer_event = kCudaEventPool->Acquire();
          cudaEventRecord(*transfer_event, h2d_stream);
        } else {
          sync_stream_with_event(h2d_stream);
        }
      }
      if (own_stream) {
        cudaStreamDestroy(h2d_stream);
      }

      // Create torch tensor views on both host and device buffers
      SetModuleMemoryFromDisk_Views(tensor_ids, host_memory_ptr);
      SetModuleCudaMemoryFromCPU(tensor_ids, device_memory_ptr, target_device);
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      DLOG_TRACE("PipelinedDiskToGpu time: {} us", end_time - start_time);
    } else if (from_disk) {
      // CPU-only target: use original sequential path
      host_memory_ptr =
          kHostMemoryPool->AllocateMemory(id, byte_size, CPU_DEVICE);
      assert(host_memory_ptr != nullptr);

      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      {
#ifndef NVTX_DISABLE
        nvtx3::scoped_range r_disk_cpu("disk_to_cpu");
#endif
        SetModuleMemoryFromDisk(tensor_ids, host_memory_ptr, on_demand);
      }
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      DLOG_TRACE("SetModuleMemoryFromDisk time: {} us", end_time - start_time);
    }

    if (!from_disk && target_device.is_cuda()) {
      // Already in host memory, just copy to GPU
      device_memory_ptr =
          kDeviceMemoryPool->AllocateMemory(id, byte_size, target_device);
      assert(device_memory_ptr != nullptr);
      assert(host_memory_ptr != nullptr);

      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      if (stream == nullptr) {
        CudaMemcpy(device_memory_ptr, host_memory_ptr, byte_size,
                   cudaMemcpyHostToDevice);
      } else {
        {
#ifndef NVTX_DISABLE
          nvtx3::scoped_range r_h2d("cpu_to_gpu");
#endif
          CudaMemcpyAsync(device_memory_ptr, host_memory_ptr, byte_size,
                          cudaMemcpyHostToDevice, stream);
        }
        {
#ifndef NVTX_DISABLE
          nvtx3::scoped_range r_sync("cuda_stream_sync");
#endif
          if (transfer_event != nullptr) {
            *transfer_event = kCudaEventPool->Acquire();
            cudaEventRecord(*transfer_event, stream);
          } else {
            sync_stream_with_event(stream);
          }
        }
      }
      SetModuleCudaMemoryFromCPU(tensor_ids, device_memory_ptr, target_device);
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      DLOG_TRACE("SetModuleCudaMemoryFromCPU time: {} us",
                 end_time - start_time);
    }

    if (target_device.is_cpu() && device.is_cuda()) {
      assert(host_memory_ptr != nullptr);
      auto start_time = MCIROSECONDS_SINCE_EPOCH;
      {
#ifndef NVTX_DISABLE
        nvtx3::scoped_range r_d2h("gpu_to_cpu");
#endif
        SetModuleMemoryFromCuda(tensor_ids, host_memory_ptr);
        kDeviceMemoryPool->FreeMemory(id, device_memory_ptr, byte_size, device);
        device_memory_ptr = nullptr;
      }
      auto end_time = MCIROSECONDS_SINCE_EPOCH;
      DLOG_TRACE("SetModuleMemoryFromCuda time: {} us", end_time - start_time);
    }
  }
  device = target_device;
}

ArcherTopologyHandle::ArcherTopologyHandle() {}

NodePtrList ArcherTopologyHandle::GetLFUNodes(const torch::Device& device) {
  NodePtrList nodes;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto node_body : lfu_nodes_) {
    CONTINUE_IF_NULL(node_body);
    if (node_body->node->device == device) {
      nodes.push_back(node_body->node);
    }
  }
  return nodes;
}

NodePtrList ArcherTopologyHandle::GetDenseNodes() {
  NodePtrList nodes;
  for (auto stage : pipeline_.stages) {
    if (stage->is_sparse) {
      continue;
    }
    for (auto node_body : stage->nodes) {
      nodes.push_back(node_body->node);
    }
  }
  return nodes;
}
NodePtrList ArcherTopologyHandle::GetSparseNodes() {
  NodePtrList nodes;
  for (auto stage : pipeline_.stages) {
    if (!stage->is_sparse) {
      continue;
    }
    for (auto node_body : stage->nodes) {
      nodes.push_back(node_body->node);
    }
  }
  return nodes;
}

NodePtrList ArcherTopologyHandle::GetDenseNodes(const NodePtr& node,
                                                const std::size_t& k) {
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
    if (stage->is_sparse) {
      continue;
    }

    nodes.push_back(stage->nodes[0]->node);
    count++;
  }
  return nodes;
}

NodePtrList ArcherTopologyHandle::GetSparseNodes(const NodePtr& node,
                                                 const std::size_t& k) {
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
    if (!stage->is_sparse) {
      continue;
    }

    nodes.push_back(stage->nodes[0]->node);
    count++;
  }
  return nodes;
}

std::uint64_t ArcherTopologyHandle::GetLastActivateStage(
    const HashID& hash_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = last_active_stage_.find(hash_id);
  if (it == last_active_stage_.end()) {
    return 0;
  }
  return it->second;
}

std::vector<std::vector<std::size_t>>
ArcherTopologyHandle::GetNodeVisitCounts() {
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

std::tuple<std::int64_t, std::int64_t>
ArcherTopologyHandle::GetResidentAndWastedBytes() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::int64_t resident = 0;
  std::int64_t wasted = 0;
  for (auto& stage : pipeline_.stages) {
    for (auto& node_body : stage->nodes) {
      auto& node = node_body->node;
      if (node == nullptr) continue;
      if (node->device.is_cuda()) {
        resident += node->byte_size;
      }
      wasted += node->byte_size * static_cast<std::int64_t>(node->unused_count);
    }
  }
  return std::make_tuple(resident, wasted);
}

std::vector<std::size_t> ArcherTopologyHandle::GetChildVisitCounts() {
  std::lock_guard<std::mutex> lock(mutex_);
  int num_layers = 0;
  int num_experts = 0;
  for (auto& stage : pipeline_.stages) {
    if (stage->is_sparse) {
      num_layers += 1;
      num_experts = stage->nodes.size();
    }
  }
  std::vector<std::size_t> child_visit_counts((num_layers - 1) * num_experts *
                                              num_experts);
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

void ArcherTopologyHandle::SetNodeVisitCounts(
    const std::vector<std::size_t>& visit_counts) {
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
    DLOG_ERROR("visit_counts size {} not equal to num_nodes {}",
               visit_counts.size(), num_nodes);
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
void ArcherTopologyHandle::SetChildVisitCounts(
    const std::vector<std::size_t>& visit_counts) {
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
    DLOG_ERROR("visit_counts size {} not equal to num_layers {}",
               visit_counts.size(), num_layers);
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

bool ArcherTopologyHandle::IsLastNode(const NodePtr& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto last_stage_ptr = pipeline_.stages.back();
  auto& nodes = last_stage_ptr->nodes;
  for (auto& n : nodes) {
    if (n->node == node) {
      return true;
    }
  }
  return false;
}
bool ArcherTopologyHandle::IsFirstNode(const NodePtr& node) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto first_stage_ptr = pipeline_.stages.front();
  auto& nodes = first_stage_ptr->nodes;
  for (auto& n : nodes) {
    if (n->node == node) {
      return true;
    }
  }
  return false;
}

void ArcherTopologyHandle::BuildTopologyFromSpecs(
    const std::vector<StageSpec>& specs) {
  std::lock_guard<std::mutex> lock(mutex_);
  pipeline_.stages.clear();
  std::size_t node_id = 0;
  std::size_t last_sparse_layer_id = UINT64_MAX;

  size_t num_sparse_layers = 0;
  size_t num_experts = 0;

  std::vector<NodePtr> all_nodes;

  for (std::size_t layer_id = 0; layer_id < specs.size(); ++layer_id) {
    const auto& spec = specs[layer_id];
    const auto& stage_tensors = *spec.tensor_groups;
    auto stage_ptr = std::make_shared<Stage>(spec.is_sparse);

    for (std::size_t expert_id = 0; expert_id < stage_tensors.size();
         ++expert_id) {
      const auto& tensor_ids = stage_tensors[expert_id];
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
          DLOG_ERROR("Tensor {} not found in tensor index", tensor_id);
        }
      }
      node_ptr->byte_size = byte_size;
      node_ptr->id = node_id;
      node_ptr->corr_id = spec.corr_ids[expert_id];
      node_ptr->is_sparse = stage_ptr->is_sparse;

      all_nodes.push_back(node_ptr);

      auto node_body_ptr = std::make_shared<NodeBody>(node_ptr);
      node_body_ptr->is_sparse = stage_ptr->is_sparse;

      stage_ptr->nodes.push_back(node_body_ptr);

      node_id++;
    }
    pipeline_.stages.push_back(stage_ptr);
    auto current_layer_id = layer_id;

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

  // output every tensor id in node
  for (auto& stage : pipeline_.stages) {
    for (auto& node : stage->nodes) {
      std::stringstream ss;
      for (auto& tensor_id : node->node->tensor_ids) {
        ss << tensor_id << " ";
      }
      // DLOG_TRACE("Node {} tensor ids {}", node->node->id, ss.str());
      lfu_nodes_.push_back(node);
    }
  }

  DLOG_TRACE("InitializeTopology pipeline_.stages.size() {}",
             pipeline_.stages.size());

  // Model placement
  auto num_gpu = GetDeviceCount();
  std::vector<std::int64_t> free_device_mem(num_gpu, 0);
  for (int i = 0; i < num_gpu; i++) {
    free_device_mem[i] =
        kDeviceMemoryPool->GetMemoryCapacity(torch::Device(torch::kCUDA, i));
  }

  auto sparse_nodes = GetSparseNodes();
  auto dense_nodes = GetDenseNodes();

  DLOG_TRACE(
      "InitializeTopology num_gpu {} sparse_nodes.size() {} dense_nodes.size() "
      "{}",
      num_gpu, sparse_nodes.size(), dense_nodes.size());

  int target_device_id = 0;
  // int dense_gpu_idx = 0;
  // int sparse_gpu_idx = 0;

  // Split evently dense nodes only
  int num_dense_nodes_per_device = std::ceil(dense_nodes.size() / num_gpu / 2);
  // int total_dense_nodes = dense_nodes.size();
  int counter = 0;
  DLOG_INFO("Moving dense parameters to CPU");
  for (auto& node_ptr : tqdm::tqdm(dense_nodes)) {
    node_ptr->default_device = torch::Device(torch::kCUDA, target_device_id);
    counter++;
    if (counter % num_dense_nodes_per_device == 0) {
      target_device_id = (target_device_id + 1) % num_gpu;
    }
    node_ptr->SetDevice(CPU_DEVICE, false);
  }
  dense_nodes.back()->default_device = torch::Device(torch::kCUDA, num_gpu - 1);

  DLOG_INFO("Moving sparse parameters to CPU");
  if (!sparse_nodes.empty()) {
    auto read_partition =
        [](const std::string& filename) -> std::pair<void*, int64_t> {
      struct stat st;
      if (stat(filename.c_str(), &st) != 0) return {nullptr, 0};
      int64_t file_size = st.st_size;
      void* buf = nullptr;
      if (posix_memalign(&buf, 4096, file_size) != 0) return {nullptr, 0};
      int fd = open(filename.c_str(), O_RDONLY);
      if (fd < 0) {
        free(buf);
        return {nullptr, 0};
      }
      posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL);
      int64_t total = 0;
      while (total < file_size) {
        auto n = ::read(fd, static_cast<char*>(buf) + total,
                        std::min(file_size - total,
                                 static_cast<int64_t>(256 * 1024 * 1024)));
        if (n <= 0) break;
        total += n;
      }
      close(fd);
      return {buf, file_size};
    };

    // Fill every sparse (expert) tensor from a single sequential bulk read of
    // its own partition file. Grouping placements by file_id and scattering via
    // memcpy avoids per-tensor random ReadTensor reads, which collapse to
    // effectively-infinite latency on a cold page cache when reloading a large
    // offload store from disk.
    struct TensorPlacement {
      NodePtr node;
      int64_t param_offset;
      int64_t size;
      int64_t file_offset;
    };
    std::map<uint32_t, std::vector<TensorPlacement>> tensors_by_file;
    for (auto& node_ptr : sparse_nodes) {
      node_ptr->default_device = torch::Device(torch::kCUDA, target_device_id);
      target_device_id = (target_device_id + 1) % num_gpu;

      node_ptr->host_memory_ptr = kHostMemoryPool->AllocateMemory(
          node_ptr->id, node_ptr->byte_size, CPU_DEVICE);
      assert(node_ptr->host_memory_ptr != nullptr);

      int64_t param_offset = 0;
      for (auto& tensor_id : node_ptr->tensor_ids) {
        auto it = kTensorIndex->find(tensor_id);
        auto& meta = it->second;
        int64_t size_aligned =
            (static_cast<int64_t>(meta.size) + kAioAlignment - 1) &
            ~(kAioAlignment - 1);
        tensors_by_file[meta.file_id].push_back(
            {node_ptr, param_offset, static_cast<int64_t>(meta.size),
             static_cast<int64_t>(meta.offset)});
        param_offset += size_aligned;
      }
      node_ptr->device = CPU_DEVICE;
    }

    std::vector<uint32_t> file_ids;
    for (auto& [fid, _] : tensors_by_file) file_ids.push_back(fid);
    std::sort(file_ids.begin(), file_ids.end());

    for (size_t fi = 0; fi < file_ids.size(); fi++) {
      uint32_t fid = file_ids[fi];
      auto [buf, buf_size] =
          read_partition(kArcherTensorHandle->GetIndexFileName(fid));
      assert(buf != nullptr);

      auto& placements = tensors_by_file[fid];
      DLOG_INFO("Processing partition ", fid, " (", buf_size / (1024 * 1024),
                " MB, ", placements.size(), " tensors)");
      for (auto& p : placements) {
        memcpy(static_cast<char*>(p.node->host_memory_ptr) + p.param_offset,
               static_cast<char*>(buf) + p.file_offset,
               static_cast<size_t>(p.size));
      }
      free(buf);
    }

    for (auto& node_ptr : sparse_nodes) {
      SetModuleMemoryFromDisk_Views(node_ptr->tensor_ids,
                                    node_ptr->host_memory_ptr);
    }
  }

  DLOG_TRACE("InitializeTopology pipeline_.stages.size() {}",
             pipeline_.stages.size());

  for (auto& node_ptr : all_nodes) {
    DLOG_TRACE("Node {} {} device {}", node_ptr->id, node_ptr->is_sparse,
               node_ptr->default_device.str());
  }

  EnableTrace();
}

NodePtr ArcherTopologyHandle::CreateDetachedNode(
    const std::vector<TensorID>& tensor_ids, int gpu_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (tensor_ids.empty() || gpu_id < 0 || kTensorIndex == nullptr) {
    throw std::invalid_argument("invalid detached expert node request");
  }

  std::int64_t aligned_bytes = 0;
  for (const TensorID tensor_id : tensor_ids) {
    const auto meta = kTensorIndex->find(tensor_id);
    if (meta == kTensorIndex->end() || meta->second.size == 0 ||
        meta->second.size > static_cast<std::uint64_t>(
                                std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument("invalid detached expert tensor metadata");
    }
    const auto size = static_cast<std::int64_t>(meta->second.size);
    const auto alignment = static_cast<std::int64_t>(kAioAlignment);
    if (size > std::numeric_limits<std::int64_t>::max() - alignment + 1) {
      throw std::overflow_error("detached expert tensor size overflow");
    }
    const auto aligned = (size + alignment - 1) & ~(alignment - 1);
    if (aligned_bytes > std::numeric_limits<std::int64_t>::max() - aligned) {
      throw std::overflow_error("detached expert node size overflow");
    }
    aligned_bytes += aligned;
  }

  auto node = std::make_shared<Node>();
  node->tensor_ids = tensor_ids;
  node->byte_size = aligned_bytes;
  node->id = next_detached_node_id_++;
  node->corr_id = node->id;
  node->is_sparse = true;
  node->default_device = torch::Device(torch::kCUDA, gpu_id);
  node->default_host = CPU_DEVICE;
  return node;
}

void ArcherTopologyHandle::InitializeTopology(
    const std::vector<
        std::tuple<std::string, std::vector<std::vector<TensorID>>>>&
        topology) {
  std::vector<StageSpec> specs;
  specs.reserve(topology.size());
  for (std::size_t layer_id = 0; layer_id < topology.size(); ++layer_id) {
    const auto& stage_tensors = std::get<1>(topology[layer_id]);
    StageSpec spec;
    spec.is_sparse = stage_tensors.size() > 1;
    spec.tensor_groups = &stage_tensors;
    spec.corr_ids.reserve(stage_tensors.size());
    for (std::size_t expert_id = 0; expert_id < stage_tensors.size();
         ++expert_id) {
      spec.corr_ids.push_back((layer_id & 0xFFFFFFFF) |
                              ((expert_id & 0xFFFFFFFF) << 32));
    }
    specs.push_back(std::move(spec));
  }
  if (!specs.empty()) {
    for (auto& corr_id : specs.back().corr_ids) {
      corr_id = (corr_id & 0xFFFFFFFF) | (UINT64_MAX << 32);
    }
  }
  BuildTopologyFromSpecs(specs);
}

void ArcherTopologyHandle::InitializeTopologyV2(
    const std::vector<
        std::tuple<std::string, bool, std::vector<std::vector<TensorID>>,
                   std::vector<std::uint64_t>>>& topology) {
  std::vector<StageSpec> specs;
  specs.reserve(topology.size());
  for (const auto& stage : topology) {
    StageSpec spec;
    spec.is_sparse = std::get<1>(stage);
    spec.tensor_groups = &std::get<2>(stage);
    spec.corr_ids = std::get<3>(stage);
    if (spec.corr_ids.size() != spec.tensor_groups->size()) {
      DLOG_ERROR(
          "InitializeTopologyV2: corr_ids count {} != tensor group count {}",
          spec.corr_ids.size(), spec.tensor_groups->size());
    }
    specs.push_back(std::move(spec));
  }
  BuildTopologyFromSpecs(specs);
}

std::vector<std::tuple<std::uint64_t, bool, int>>
ArcherTopologyHandle::GetTopologySnapshot() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::tuple<std::uint64_t, bool, int>> snapshot;
  for (const auto& stage : pipeline_.stages) {
    for (const auto& node_body : stage->nodes) {
      const auto& node = node_body->node;
      snapshot.emplace_back(static_cast<std::uint64_t>(node->corr_id),
                            node->is_sparse, node->default_device.index());
    }
  }
  return snapshot;
}

NodePtr ArcherTopologyHandle::GetNodeFromTensorID(const TensorID& tensor_id) {
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
  DLOG_ERROR("Tensor {} not found in tensor id to node map", tensor_id);
  return nullptr;
}

NodeBodyPtr ArcherTopologyHandle::GetNodeBodyFromCorrID(
    const std::uint64_t& correlation_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::uint64_t high_corr_id =
      correlation_id >> 32;  // For children in the same level
  std::uint64_t low_corr_id =
      correlation_id & 0xFFFFFFFF;  // For model inference pipeline

  bool is_last_node = (0xFFFFFFFF == high_corr_id);
  if (is_last_node) {
    high_corr_id = 0;  // reset to 0 avoid miss use
  }

  auto stage = pipeline_.stages[low_corr_id];
  auto node_body = stage->nodes[high_corr_id];

  return node_body;
}

std::int64_t ArcherTopologyHandle::GetSparseCacheLimit(
    const torch::Device& device) {
  std::int64_t dense_cache_size = 0;
  for (auto& stage : pipeline_.stages) {
    for (auto& node_body : stage->nodes) {
      if (stage->is_sparse) continue;
      if (node_body->node->device == device) {
        dense_cache_size += node_body->node->byte_size;
      }
    }
  }

  std::int64_t device_size_limit =
      (device.is_cuda()) ? kDeviceMemoryPool->GetMemoryCapacity(device)
                         : kHostMemoryPool->GetMemoryCapacity();
  assert(device_size_limit > dense_cache_size);
  std::int64_t sparse_cache_size = device_size_limit - dense_cache_size;

  return sparse_cache_size;
}

std::tuple<std::size_t, std::size_t>
ArcherTopologyHandle::GetNumLayersAndExperts() {
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
void SetModuleDisk(std::vector<TensorID>& tensor_ids) {
  // DLOG_TRACE("SetModuleDisk {} tensors", tensor_ids.size());
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
void SetModuleMemoryFromDisk(std::vector<TensorID>& tensor_ids, void* host_ptr,
                             bool on_demand) {
  if (tensor_ids.empty()) return;

  // Check whether all tensors sit contiguously in the same partition file.
  // If so, read the whole region in one I/O call instead of per-tensor.
  bool contiguous = true;
  auto first_it = kTensorIndex->find(tensor_ids[0]);
  std::uint32_t file_id = first_it->second.file_id;
  std::int64_t start_offset = first_it->second.offset;
  std::int64_t expected_offset = start_offset;
  std::int64_t total_aligned = 0;

  for (const auto& tensor_id : tensor_ids) {
    auto it = kTensorIndex->find(tensor_id);
    std::int64_t sz =
        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
    if (it->second.file_id != file_id || it->second.offset != expected_offset) {
      contiguous = false;
      break;
    }
    expected_offset += sz;
    total_aligned += sz;
  }

  if (contiguous && total_aligned > 0) {
    auto filename = kArcherTensorHandle->GetIndexFileName(file_id);
    kArcherTensorHandle->ReadBulk(filename, host_ptr, on_demand, total_aligned,
                                  start_offset);
  } else {
    std::int64_t offset = 0;
    for (const auto& tensor_id : tensor_ids) {
      kArcherTensorHandle->ReadTensor(
          tensor_id, static_cast<char*>(host_ptr) + offset, on_demand);
      auto it = kTensorIndex->find(tensor_id);
      std::int64_t sz =
          (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
      offset += sz;
    }
  }

  std::int64_t param_size = 0;
  for (const auto& tensor_id : tensor_ids) {
    auto it = kTensorIndex->find(tensor_id);
    auto options = torch::TensorOptions()
                       .dtype(it->second.options.dtype())
                       .layout(it->second.options.layout())
                       .device(torch::kCPU)
                       .requires_grad(it->second.options.requires_grad())
                       .pinned_memory(it->second.options.pinned_memory());

    DLOG_TRACE("SetModuleMemoryFromDisk tensor {}", it->second.DebugString());
    auto tensor_tmp =
        torch::from_blob(static_cast<char*>(host_ptr) + param_size,
                         it->second.shape, DoNothingDeleter<void>{}, options);
    if (!it->second.tensor.defined()) {
      it->second.tensor = torch::zeros({1}, options);
    }
    it->second.tensor.set_data(tensor_tmp);
    std::int64_t size_aligned =
        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
    param_size += size_aligned;
  }
}

// DISK (already read) -> CPU views only (no disk read; buffer pre-filled)
void SetModuleMemoryFromDisk_Views(std::vector<TensorID>& tensor_ids,
                                   void* host_ptr) {
  std::int64_t param_size = 0;
  for (const auto& tensor_id : tensor_ids) {
    auto it = kTensorIndex->find(tensor_id);
    auto options = torch::TensorOptions()
                       .dtype(it->second.options.dtype())
                       .layout(it->second.options.layout())
                       .device(torch::kCPU)
                       .requires_grad(it->second.options.requires_grad())
                       .pinned_memory(it->second.options.pinned_memory());

    DLOG_TRACE("SetModuleMemoryFromDisk_Views tensor {}",
               it->second.DebugString());
    auto tensor_tmp =
        torch::from_blob((void*)((char*)host_ptr + param_size),
                         it->second.shape, DoNothingDeleter<void>{}, options);
    if (!it->second.tensor.defined()) {
      it->second.tensor = torch::zeros({1}, options);
    }
    it->second.tensor.set_data(tensor_tmp);
    std::int64_t size_aligned =
        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
    param_size += size_aligned;
  }
}

// CPU -> GPU
void SetModuleCudaMemoryFromCPU(std::vector<TensorID>& tensor_ids,
                                void* device_ptr, const torch::Device& device) {
  // DLOG_TRACE("SetModuleCudaMemoryFromCPU {} tensors", tensor_ids.size());
  std::int64_t param_size = 0;
  for (const auto& tensor_id : tensor_ids) {
    auto it = kTensorIndex->find(tensor_id);
    DLOG_TRACE("SetModuleCudaMemoryFromCPU tensor {} -> {}",
               it->second.DebugString(), device.str());
    auto tensor_options = torch::TensorOptions()
                              .dtype(it->second.options.dtype())
                              .layout(it->second.options.layout())
                              .device(device)
                              .requires_grad(it->second.options.requires_grad())
                              .pinned_memory(false);
    it->second.tensor.set_data(
        torch::from_blob((char*)device_ptr + param_size, it->second.shape,
                         DoNothingDeleter<void>{}, tensor_options));
    std::int64_t size_aligned =
        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
    param_size += size_aligned;
  }
  // DLOG_TRACE("SetModuleCudaMemoryFromCPU {} tensors done",
  // tensor_ids.size());
}

// GPU -> CPU
void SetModuleMemoryFromCuda(std::vector<TensorID>& tensor_ids,
                             void* host_ptr) {
  std::int64_t param_size = 0;
  for (const auto& tensor_id : tensor_ids) {
    // void* old_ptr = kTensorIndex->find(tensor_id)->second.tensor.data_ptr();

    auto it = kTensorIndex->find(tensor_id);
    DLOG_TRACE("SetModuleMemoryFromCuda tensor {}", it->second.DebugString());
    it->second.tensor.set_data(
        torch::from_blob((char*)host_ptr + param_size, it->second.shape,
                         DoNothingDeleter<void>{}, it->second.options));
    // kArcherTensorHandle->UpdateTensorMap(old_ptr,
    // it->second.tensor.data_ptr());
    std::int64_t size_aligned =
        (it->second.size + kAioAlignment - 1) & ~(kAioAlignment - 1);
    param_size += size_aligned;
  }
  // DLOG_TRACE("SetModuleMemoryFromCuda {} tensors done", tensor_ids.size());
}
