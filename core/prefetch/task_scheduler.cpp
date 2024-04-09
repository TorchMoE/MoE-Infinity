// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include <sstream>

#include <c10/cuda/CUDACachingAllocator.h>

#include "common/time.h"
#include "memory/memory_pool.h"
#include "task_scheduler.h"
#include "task_thread.h"
#include "utils/archer_logger.h"
#include "utils/cuda_utils.h"

std::unique_ptr<ArcherTaskPool> kTaskPool = nullptr;

ArcherTaskPool::ArcherTaskPool()
{
    unified_queue_.resize(NUM_PRIORITY);

    main_thread_stop_flag_.store(false);

    int num_gpu = GetDeviceCount();
    int num_thread_per_gpu = 1;

    for (int i = -1; i < num_gpu; ++i) {
        gpu_min_priority_.push_back(std::vector<std::uint32_t>(num_thread_per_gpu, 1000));
    }

    for (int i = -1; i < num_gpu; ++i) {
        std::list<std::thread> gpu_threads;
        for (int j = 0; j < num_thread_per_gpu; ++j) {
            auto gpu_thread = std::thread(&ArcherTaskPool::GPUThreadFunc, this, i, j);
            SetThreadAffinity(gpu_thread);
            gpu_threads.push_back(std::move(gpu_thread));
        }
        exec_threads_.push_back(std::move(gpu_threads));
    }
}

void ArcherTaskPool::FetchExec(const std::uint64_t& request_id, const NodePtr& node)
{
    auto task = std::make_shared<Task>();
    task->on_demand = false;
    task->node = node;
    task->priority = 0;
    task->src_device = node->device;
    task->dst_device = node->default_device;
    task->request_id = request_id;

    ARCHER_LOG_DEBUG("FetchExec: {}", task->DebugString());

    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (std::size_t i = 1; i < NUM_PRIORITY; ++i) {
            unified_queue_[i].erase(std::remove_if(unified_queue_[i].begin(),
                                                   unified_queue_[i].end(),
                                                   [&](auto& t) {
                                                       return (t->node == node) |
                                                              ((node->corr_id & 0xffffffff) >=
                                                               (t->node->corr_id & 0xffffffff));
                                                   }),
                                    unified_queue_[i].end());
        }
    }

    if (task->src_device == task->dst_device) {
        node->state = 0;
        node->cv.notify_all();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        unified_queue_[task->priority].push_back(task);
    }
}

void ArcherTaskPool::EnqueueTask(const TaskPtr& task)
{
    ARCHER_LOG_DEBUG("EnqueueTask: {}", task->DebugString());

    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (std::size_t i = 1; i < NUM_PRIORITY; ++i) {
            // remove any task that has the same node
            auto it =
                std::remove_if(unified_queue_[i].begin(), unified_queue_[i].end(), [&](auto& t) {
                    bool is_same_node = (t->node == task->node);
                    bool is_lower_priority = (t->priority >= task->priority);
                    bool is_outdate_layers =
                        task->remove_layer &&
                        ((t->node->corr_id & 0xffffffff) < (task->node->corr_id & 0xffffffff));
                    bool need_remove = (is_same_node && is_lower_priority) || is_outdate_layers;
                    if (need_remove) t->node->mutex.unlock();
                    return need_remove;
                });
            unified_queue_[i].erase(it, unified_queue_[i].end());
        }
    }

    if (task->src_device == task->dst_device) {
        task->node->state = 0;
        task->node->cv.notify_all();
        ARCHER_LOG_DEBUG("EnqueueTask: {} is on the same device", task->DebugString());
        return;
    }

    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        unified_queue_[task->priority].push_back(task);
    }

    ARCHER_LOG_DEBUG("EnqueueTask: finish {}", task->DebugString());
}

void ArcherTaskPool::StartExec(const std::uint64_t& request_id, const NodePtr& node)
{
    auto task = std::make_shared<Task>();
    task->on_demand = true;
    task->node = node;
    task->priority = 0;
    task->src_device = node->device;
    task->dst_device = node->default_device;
    task->request_id = request_id;

    ARCHER_LOG_DEBUG("StartExec: {}", task->DebugString());

    node->visit_count += 1;
    if (node->device.is_cuda()) { node->incache_visit_count++; }
    node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
    node->io_state = static_cast<NodeState>(node->io_state | NODE_STATE_VISITED);

    auto node_body = kTopologyHandle->GetNodeBodyFromCorrID(node->corr_id);

    node_body->visit_cnt += 1;

    /* Observation: expert IO time + GPU inference time is similar to its compute time in CPU.
     * Solution: If the node is on CPU, and no other nodes are running on CPU, do not perform H2D
     * memory copy. Allow two node run concurrently.
     */
    // if ((++cpu_running_nodes_ < 2) && task->src_device.is_cpu() && node->is_sparse) {
    //     task->dst_device = task->src_device;
    //     node_body->cpu_visit_cnt += 1;
    // }

    if (task->dst_device.is_cuda()) { node_body->gpu_visit_cnt += 1; }

    node->last_prefetch_time = MCIROSECONDS_SINCE_EPOCH;
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (std::size_t i = 0; i < NUM_PRIORITY; ++i) {
            // remove any task that has the same node
            auto it =
                std::remove_if(unified_queue_[i].begin(), unified_queue_[i].end(), [&](auto& t) {
                    return (t->node == node) |
                           ((node->corr_id & 0xffffffff) > (t->node->corr_id & 0xffffffff));
                });
            unified_queue_[i].erase(it, unified_queue_[i].end());
        }
    }

    if (task->src_device.is_cuda()) {
        // ARCHER_LOG_DEBUG("StartExec: {} is on the same device", task->DebugString());
        std::lock_guard<std::mutex> lock(exec_mutex_);
        exec_queue_.insert({node->id, task});
        node->state = 0;
        node->cv.notify_all();

        if (task->dst_device.is_cpu()) { node_body->cpu_hit_cnt += 1; }

        if (task->dst_device.is_cuda()) node_body->hit_cnt += 1;

        return;
    }

    if (task->dst_device.is_cpu()) { node_body->cpu_miss_cnt += 1; }
    if (task->dst_device.is_cuda()) { node_body->gpu_miss_cnt += 1; }

    {
        std::lock_guard<std::mutex> lock(exec_mutex_);
        if (exec_queue_.find(node->id) != exec_queue_.end()) {
            std::stringstream ss;
            ss << "Node " << std::hex << node->id << " is already in exec queue";
            ARCHER_LOG_WARN(ss.str().c_str());
            node->state = 0;
            node->cv.notify_all();
            return;
        }
        exec_queue_.insert({node->id, task});
    }

    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        unified_queue_[task->priority].push_back(task);
    }
}

void ArcherTaskPool::StopExec(const std::uint64_t& request_id, const NodePtr& node)
{
    auto task = std::make_shared<Task>();
    task->on_demand = true;
    task->node = node;
    task->priority = 0;
    task->src_device = node->device;
    task->dst_device = node->default_host;
    task->request_id = request_id;

    ARCHER_LOG_DEBUG("StopExec: {}", task->DebugString());

    node->state = 0;
    node->cv.notify_all();
    {
        std::lock_guard<std::mutex> lock(exec_mutex_);
        exec_queue_.erase(node->id);
    }

    return;
}

bool ArcherTaskPool::RemoveCachedSparseNode(const NodePtr& node, int device_id)
{
    // ARCHER_LOG_DEBUG("RemoveCachedSparseNode: {}", node->str());

    if (node->device.is_cuda()) { return true; }

    auto start_time = MILLISECONDS_SINCE_EPOCH;

    auto nodes = kTopologyHandle->GetSparseNodes();

    // get all nodes in exec queue
    std::unordered_set<NodePtr> nodes_exec;
    {
        std::lock_guard<std::mutex> lock(exec_mutex_);
        for (auto& [id, task] : exec_queue_) { nodes_exec.insert(task->node); }
    }

    if (device_id == -1) device_id = node->default_device.index();

    auto cache_limit = kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, device_id));
    cache_limit -= node->byte_size;

    int64_t cache_size = 0;
    NodePtrList device_nodes;
    for (auto& n : nodes) {
        if (n->device.is_cuda() && (n->device.index() == device_id)) {
            cache_size += n->byte_size;
            device_nodes.push_back(n);
        }
    }

    ARCHER_LOG_DEBUG("RemoveCachedSparseNode: {} {}MB {}MB {}",
                     device_id,
                     cache_size / MB,
                     cache_limit / MB,
                     device_nodes.size());

    if (cache_size > cache_limit) {
        std::vector<std::size_t> node_access_time;
        std::vector<std::size_t> node_index(device_nodes.size());
        std::iota(node_index.begin(), node_index.end(), 0);
        // for (auto& n : device_nodes) { node_access_time.push_back(n->last_access_time); }
        for (auto& n : device_nodes) { node_access_time.push_back(n->incache_visit_count); }
        std::sort(node_index.begin(), node_index.end(), [&](int i, int j) {
            return node_access_time[i] > node_access_time[j];
        });
        for (auto i : node_index) {
            auto& n = device_nodes[i];
            {
                std::lock_guard<std::mutex> lock(this->candidates_mutex_);
                if (candidates_.find(n) != candidates_.end() && !n->is_overflow) { continue; }
            }
            if (nodes_exec.find(n) != nodes_exec.end()) { continue; }
            if (n->mutex.try_lock()) {
                ARCHER_LOG_DEBUG("RemoveCachedSparseNode: {}", n->str());
                n->SetDevice(n->default_host);
                n->incache_visit_count = 0;
                n->mutex.unlock();
                cache_size -= n->byte_size;
                if ((node->io_state & NODE_STATE_VISITED) == 0) node->unused_count += 1;
            }
            if (cache_size <= cache_limit) { break; }
        }
    }

    auto end_time = MILLISECONDS_SINCE_EPOCH;
    ARCHER_LOG_DEBUG("RemoveCachedSparseNode: {}MB {}MB {} {}us",
                     cache_size / MB,
                     cache_limit / MB,
                     node->str(),
                     end_time - start_time);

    return cache_size <= cache_limit;
}

bool ArcherTaskPool::RemoveCachedDenseNode(const NodePtr& node)
{
    if (node->device.is_cuda()) { return true; }

    auto device_id = node->default_device.index();

    auto start_time = MILLISECONDS_SINCE_EPOCH;
    auto nodes = kTopologyHandle->GetDenseNodes();
    auto cache_limit = DEVICE_CACHE_LIMIT(device_id);
    int64_t cache_size = 0;

    NodePtrList device_nodes;
    for (auto& n : nodes) {
        if (n->device.is_cuda() && (n->device.index() == device_id)) {
            cache_size += n->byte_size;
            device_nodes.push_back(n);
        }
    }

    ARCHER_LOG_DEBUG("RemoveCachedDenseNode: {} {}MB {}MB {}",
                     device_id,
                     cache_size / MB,
                     cache_limit / MB,
                     device_nodes.size());

    if (cache_size > cache_limit) {
        std::sort(device_nodes.begin(), device_nodes.end(), [&](auto& a, auto& b) {
            return (a->corr_id & 0xFFFFFFFF) < (b->corr_id & 0xFFFFFFFF);
        });

        for (auto& n : device_nodes) {
            if (n->mutex.try_lock()) {
                ARCHER_LOG_DEBUG("RemoveCachedDenseNode: {} {}MB {}MB {}",
                                 device_id,
                                 cache_size / MB,
                                 cache_limit / MB,
                                 n->str());

                n->SetDevice(n->default_host);
                n->mutex.unlock();
                cache_size -= n->byte_size;
            }

            if (cache_size <= cache_limit) { break; }
        }
    }

    if (cache_size > cache_limit) {
        ARCHER_LOG_ERROR("RemoveCachedDenseNode: {} {}MB {}MB {} {}",
                         device_id,
                         cache_size / MB,
                         cache_limit / MB,
                         device_nodes.size(),
                         node->str());
    };

    auto free_memory = kDeviceMemoryPool->GetFreeMemory(torch::Device(torch::kCUDA, device_id));
    ARCHER_LOG_DEBUG("RemoveCachedDenseNode: {} {}MB {}MB {}",
                     device_id,
                     cache_size / MB,
                     cache_limit / MB,
                     free_memory / MB);

    auto end_time = MILLISECONDS_SINCE_EPOCH;
    ARCHER_LOG_DEBUG("RemoveCachedDenseNode: {} {}us", node->str(), end_time - start_time);
    return cache_size > cache_limit;
}

// void ArcherTaskPool::RemoveCachedNode(const NodePtr& node)
// {
//     if (node->device.is_cuda()) { return; }

//     auto sparse_nodes = kTopologyHandle->GetSparseNodes();
//     auto dense_nodes = kTopologyHandle->GetDenseNodes();

//     auto nodes = sparse_nodes;
//     nodes.insert(nodes.end(), dense_nodes.begin(), dense_nodes.end());

//     std::unordered_set<NodePtr> nodes_exec;
//     {
//         std::lock_guard<std::mutex> lock(exec_mutex_);
//         for (auto& [id, task] : exec_queue_) { nodes_exec.insert(task->node); }
//     }

//     auto device_id = node->default_device.index();

//     auto cache_limit = kDeviceMemoryPool->GetMemoryCapacity(torch::Device(torch::kCUDA, device_id));
//     cache_limit -= node->byte_size;
//     int64_t cache_size = 0;

//     NodePtrList device_nodes;
//     for (auto& n : nodes) {
//         if (n->device.is_cuda() && (n->device.index() == device_id)) {
//             cache_size += n->byte_size;
//             device_nodes.push_back(n);
//         }
//     }

//     ARCHER_LOG_DEBUG("RemoveCachedNode: {} {}MB {}MB {}",
//                      device_id,
//                      cache_size / MB,
//                      cache_limit / MB,
//                      device_nodes.size());

//     if (cache_size > cache_limit) {
//         std::sort(device_nodes.begin(), device_nodes.end(), [&](auto& a, auto& b) {
//             return a->last_access_time < b->last_access_time;
//         });

//         for (auto& n : device_nodes) {
//             if (nodes_exec.find(n) != nodes_exec.end()) { continue; }
//             if (n->mutex.try_lock()) {
//                 ARCHER_LOG_DEBUG("RemoveCachedNode: {} {}MB {}MB {}",
//                                  device_id,
//                                  cache_size / MB,
//                                  cache_limit / MB,
//                                  n->str());

//                 n->SetDevice(n->default_host);
//                 n->mutex.unlock();
//                 cache_size -= n->byte_size;
//             }

//             if (cache_size <= cache_limit) { break; }
//         }
//     }

//     if (cache_size > cache_limit) {
//         ARCHER_LOG_ERROR("RemoveCachedSparseNode: {} {}MB {}MB {} {}",
//                          device_id,
//                          cache_size / MB,
//                          cache_limit / MB,
//                          device_nodes.size(),
//                          node->str());
//     };
// }

void ArcherTaskPool::GPUThreadFunc(int gpu_id, int thread_id)
{
    while (!main_thread_stop_flag_.load()) {
        std::uint32_t max_priority = 1000;
        std::unique_lock<std::mutex> lock(unified_mutex_);
        for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
            if (!unified_queue_[i].empty()) {
                max_priority = i;
                break;
            }
        }

        if (max_priority == 1000) {
            lock.unlock();
            SKIP_TO_NEXT_ITERATION
        }

        // Find a task that can be executed on the current GPU
        TaskPtr task = nullptr;
        for (auto& t : unified_queue_[max_priority]) {
            if (t->dst_device.index() == gpu_id) {
                task = t;
                break;
            }
        }

        if (task == nullptr) {
            lock.unlock();
            SKIP_TO_NEXT_ITERATION
        }

        auto node = task->node;
        node->incache_visit_count += 1;

        // remove task from the queue
        for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
            unified_queue_[i].erase(std::remove_if(unified_queue_[i].begin(),
                                                   unified_queue_[i].end(),
                                                   [&, task](auto& t) {
                                                       return (t->node == node) &
                                                              (t->dst_device == task->dst_device);
                                                   }),
                                    unified_queue_[i].end());
        }

        ARCHER_LOG_DEBUG(("Execute task " + task->DebugString()).c_str());

        lock.unlock();

        if (!task->on_demand) {
            bool success = RemoveCachedSparseNode(node);
            if (!success) {
                ARCHER_LOG_DEBUG("{} evict failed, move to CPU", task->DebugString());
                continue;
            }
        }
        SetNodeDevice(task);

        if (task->on_demand) {
            node->state = 0;
            node->cv.notify_all();
        }
    }
}

void ArcherTaskPool::SetNodeDevice(const TaskPtr& task)
{
    auto node = task->node;

    ARCHER_LOG_DEBUG("SetNodeDevice: task: {}, node: {}", task->DebugString(), node->str());
    if (!task->on_demand) {
        if (!node->mutex.try_lock()) {
            ARCHER_LOG_DEBUG("SetNodeDevice: task: {}, mutex locked", task->DebugString());
            return;
        }
    }

    if (node->device.type() == task->dst_device.type()) {
        ARCHER_LOG_DEBUG("SetNodeDevice: task: {}, skip same device", task->DebugString());
        if (!task->on_demand) node->mutex.unlock();
        return;
    }

    auto start_time = MCIROSECONDS_SINCE_EPOCH;

    // node->SetDevice(task->dst_device);
    node->SetDevice(task->dst_device, task->on_demand);
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    ARCHER_LOG_DEBUG(
        "SetNodeDevice: task: {}, emplace time {} us", task->DebugString(), end_time - start_time);

    // do not unlock if node in exec queue, leave this to the release of node
    if (!task->on_demand) node->mutex.unlock();

    node->io_state = NODE_STATE_CACHED;

    if (task->priority > 0 && task->dst_device.is_cuda()) {
        auto node_body = kTopologyHandle->GetNodeBodyFromCorrID(node->corr_id);
        node_body->prefetch_cnt += 1;
        node->io_state = static_cast<NodeState>(node->io_state | NODE_STATE_PREFETCHED);
        node->last_prefetch_time = MCIROSECONDS_SINCE_EPOCH;
        ARCHER_LOG_DEBUG("Prefetch Node: task: {}, prefetch_cnt: {}",
                         task->DebugString(),
                         node_body->prefetch_cnt);
    }
}

std::string ArcherTaskPool::DebugString(const std::vector<std::deque<TaskPtr>>& queue)
{
    std::stringstream ss;
    for (std::uint32_t i = 0; i < queue.size(); ++i) {
        ss << "priority " << i << " : ";
        for (auto task : queue[i]) {
            auto node = task->node;
            if (node == nullptr && task->remove_nodes.size() == 0) { continue; }
            if (node == nullptr) { node = task->remove_nodes[0]; }
            ss << std::hex << node->id << "[" << node->device << "->" << task->dst_device.str()
               << "," << task->remove_nodes.size() << ","
               << "]"
               << " " << std::dec;
        }
        ss << std::endl;
    }
    return ss.str();
}
