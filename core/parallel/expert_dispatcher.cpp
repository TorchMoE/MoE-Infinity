// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "memory/event_pool.h"
#include "prefetch/task_scheduler.h"
#include "prefetch/task_thread.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "model/model_topology.h"
#include "model/moe.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <chrono>
#include <future>
#include <iterator>
#include <regex>
#include <sstream>
#include <thread>

extern void fp8_dequant_blockwise_cuda(const void* weight, const void* scale,
                                       void* out, int N, int K,
                                       cudaStream_t stream);

static torch::Tensor _fp8_dequant_on_device(const torch::Tensor& w_fp8,
                                            const torch::Tensor& scale,
                                            cudaStream_t stream) {
  int N = w_fp8.size(0);
  int K = w_fp8.size(1);
  auto out = torch::empty(
      {N, K},
      torch::TensorOptions().dtype(torch::kBFloat16).device(w_fp8.device()));
  auto w_u8 = w_fp8.view(torch::kUInt8).contiguous();
  auto s_f32 = scale.to(torch::kFloat32).contiguous();
  fp8_dequant_blockwise_cuda(w_u8.data_ptr(), s_f32.data_ptr(), out.data_ptr(),
                             N, K, stream);
  return out;
}

void ExpertDispatcher::SetScales(
    const std::map<std::string, torch::Tensor>& scales) {
  if (scales.empty()) return;

  static const std::regex kLayerRe(R"(layers\.(\d+)\.)");
  static const std::regex kExpertRe(R"(experts\.(\d+)\.)");
  static const std::vector<std::string> kWeightNames = {
      "gate_proj.weight", "up_proj.weight", "down_proj.weight"};

  int max_layer = 0;
  int max_expert = 0;
  for (auto& kv : scales) {
    std::smatch m;
    std::string k = kv.first;
    if (std::regex_search(k, m, kLayerRe)) {
      max_layer = std::max(max_layer, std::stoi(m[1].str()));
    }
    if (std::regex_search(k, m, kExpertRe)) {
      max_expert = std::max(max_expert, std::stoi(m[1].str()));
    }
  }

  fp8_scales_.assign(max_layer + 1,
                     std::vector<std::vector<torch::Tensor>>(
                         max_expert + 1, std::vector<torch::Tensor>(3)));

  for (auto& kv : scales) {
    std::smatch m;
    std::string k = kv.first;
    int layer_idx = -1, expert_idx = -1, weight_idx = -1;
    if (std::regex_search(k, m, kLayerRe)) layer_idx = std::stoi(m[1].str());
    if (std::regex_search(k, m, kExpertRe)) expert_idx = std::stoi(m[1].str());
    for (int wi = 0; wi < 3; ++wi) {
      if (k.find(kWeightNames[wi]) != std::string::npos) {
        weight_idx = wi;
        break;
      }
    }
    if (layer_idx >= 0 && expert_idx >= 0 && weight_idx >= 0) {
      fp8_scales_[layer_idx][expert_idx][weight_idx] = kv.second.cpu();
    }
  }

  fp8_in_store_ = true;
}

ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype,
                                   int expert_type, int num_threads)
    : pending_(0),
      num_enqueued_(0),
      start_(false),
      expert_type_(expert_type),
      dtype_(dtype),
      num_experts_(num_experts),
      cache_mutex_(kNumDevices()),
      cache_cv_(kNumDevices()),
      input_queue_(kNumDevices()),
      exec_queue_(kNumDevices()),
      cached_experts_(kNumDevices()),
      num_threads_(num_threads),
      modules_(kNumDevices() * num_threads, nullptr) {
  main_thread_stop_flag_.store(false);

  gpu_overload_ = std::make_unique<std::atomic<bool>[]>(kNumDevices());
  for (int i = 0; i < kNumDevices(); ++i) {
    gpu_overload_[i].store(false);
  }

  for (int i = 0; i < kNumDevices(); ++i) {
    cudaSetDevice(i);
    cudaStream_t fetch_stream;
    cudaStreamCreateWithFlags(&fetch_stream, cudaStreamNonBlocking);
    fetch_streams_.emplace_back(fetch_stream);

    auto thread_func = std::bind(&ExpertDispatcher::GPUFetchFunc, this, i);
    std::string thread_name = "GPUFetchFunc" + std::to_string(i);
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();

    auto cache_limit =
        kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, i));
    cache_sizes_.push_back(cache_limit);
  }

  for (int i = 0; i < kNumDevices() * num_threads; ++i) {
    cudaSetDevice(i % kNumDevices());
    cudaStream_t exec_stream;
    cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
    exec_streams_.emplace_back(exec_stream);
    cudaEvent_t exec_done_event;
    cudaEventCreateWithFlags(&exec_done_event, cudaEventDisableTiming);
    exec_done_events_.emplace_back(exec_done_event);

    modules_[i] = new MoEMLP(dtype, expert_type);

    auto thread_func =
        std::bind(&ExpertDispatcher::GPUExecFunc, this, i % kNumDevices(), i);
    std::string thread_name = "GPUExecFunc" + std::to_string(i);
    threads_.emplace_back(new base::Thread(thread_func, thread_name));
    threads_.back()->start();
  }

  {
    auto route_func = std::bind(&ExpertDispatcher::RouteFunc, this);
    threads_.emplace_back(new base::Thread(route_func, "ExpertRouteFunc"));
    threads_.back()->start();
  }
  {
    auto retire_func =
        std::bind(&ExpertDispatcher::CompletionRetirementFunc, this);
    threads_.emplace_back(
        new base::Thread(retire_func, "CompletionRetirementFunc"));
    threads_.back()->start();
  }
  {
    auto expert_retire_func =
        std::bind(&ExpertDispatcher::RetirementFunc, this);
    threads_.emplace_back(
        new base::Thread(expert_retire_func, "ExpertRetirementFunc"));
    threads_.back()->start();
  }

  at::InferenceMode infer_guard(0);

  for (int i = 0; i < num_experts; ++i) {
    experts_.emplace_back();
    for (int j = 0; j < num_layers; ++j) {
      experts_[i].emplace_back();
      experts_[i][j] = std::make_shared<ExpertNode>();
      experts_[i][j]->expert_type = expert_type;
      int expert_type = expert_type_;
      switch (expert_type) {
        case NLLB_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new NllbMoeDenseActDense(dtype);
          break;
        case FSGPT_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new FSGPTMoEDenseActDense(dtype);
          break;
        case MIXTRAL_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new MixtralMoEDenseActDense(dtype);
          break;
        case DEEPSEEK_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new DeepSeekMoEDenseActDense(dtype);
          break;
        case GPT_OSS_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new GptOssMoeDenseActDense();
          break;
        default:
          DLOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ",
                     expert_type);
      }
      experts_[i][j]->module->eval();
      experts_[i][j]->layer_idx = j;
      experts_[i][j]->expert_idx = i;
    }
  }
}

void ExpertDispatcher::EnqueueExpert(int layer_idx, int expert_idx, int gpu_id,
                                     bool remote) {
  ExpertDispatcher::CallArgs args;
  args.layer_idx = layer_idx;
  args.expert_idx = expert_idx;
  args.gpu_id = gpu_id;
  args.remote = remote;
  args.generation = current_generation_.load(std::memory_order_acquire);
  Enqueue(args);
}

void ExpertDispatcher::Enqueue(CallArgs& args) {
#ifndef NVTX_DISABLE
  nvtx3::scoped_range r("expert_enqueue");
#endif
  // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
  int layer_idx = args.layer_idx;
  int expert_idx = args.expert_idx;
  auto expert_node = experts_[expert_idx][layer_idx];

  {
    auto expected = NodeExecState::IDLE;
    if (!expert_node->node->exec_state.compare_exchange_strong(
            expected, NodeExecState::FETCHING, std::memory_order_acq_rel)) {
      if (expert_node->node->is_prefetching.load(std::memory_order_acquire)) {
        expert_node->node->pending_dispatches.fetch_add(
            1, std::memory_order_acq_rel);
        args.wait_for_prefetch = true;
        args.gpu_id = expert_node->node->default_device.index();
        input_queue_[args.gpu_id].Push(args);
        return;
      }
      DLOG_FATAL(
          "ExpertDispatcher::Enqueue: exec_state CAS failed (expert_idx ",
          expert_idx, " layer_idx ", layer_idx, " node ",
          expert_node->node->str(), " state=", static_cast<int>(expected), ")");
    }
  }
  expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

  if (expert_node->node->device.is_cuda()) {
    // Cache-hit path: expert already on GPU, skip fetch queue entirely.
    // transfer_event stays nullptr — GPUExecFunc will skip cudaStreamWaitEvent.
    args.gpu_id = expert_node->node->device.index();
    DLOG_DEBUG("cache_hit_immediate: expert_idx ", args.expert_idx,
               " layer_idx ", args.layer_idx, " gpu_id ", args.gpu_id);

    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();

    ExecArgs exec_args;
    // exec_args.hidden_states = std::move(input);
    exec_args.expert_node = expert_node;
    expert_node->SetTensorsFromBlob(expert_node->node->device);
    exec_args.out_gpu_id = original_device.index();
    exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
    exec_args.evict = false;
    exec_args.hit = true;
    exec_args.generation = args.generation;
    cache_hit_count_.fetch_add(1, std::memory_order_relaxed);
    cache_access_count_.fetch_add(1, std::memory_order_relaxed);
    // transfer_event = nullptr: expert is already on GPU, no H2D wait needed

    // module_->SetTensorsFromIds(expert_node->node->tensor_ids);

    // std::unique_lock<std::mutex> lock(exec_mutex_[args.gpu_id]);
    // exec_queue_[args.gpu_id].push_back(std::move(exec_args));
    exec_queue_[args.gpu_id].Push(exec_args);
  } else {
    // std::unique_lock<std::mutex> lock(input_mutex_[args.gpu_id]);
    // input_queue_[args.gpu_id].push_back(std::move(args));
    input_queue_[args.gpu_id].Push(args);
  }
  // input_cv_[args.gpu_id].notify_all();
  // exec_cv_[args.gpu_id].notify_all();
  // input_queue_.push_back(std::move(args));
  num_enqueued_.fetch_add(1);

  // auto& a = input_queue_.back();
  // if (expert_node->node->device.is_cuda()) {
  //   a.gpu_id = expert_node->node->device.index();
  // }
  // DLOG_TRACE("ExpertDispatcher::Enqueue: num_enqueued_ ",
  // num_enqueued_.load(),
  //            "input_queue_ ", input_queue_.size(), "gpu_id ", a.gpu_id,
  //            "layer_idx ", a.layer_idx, "expert_idx ", a.expert_idx, "remote
  //            ", a.remote);
  // lock.unlock();
  // cvs_[MUTEX_TYPE::INPUT_MUTEX].notify_all();
}

void ExpertDispatcher::RegisterExpert(
    int layer_idx, int expert_idx, const std::vector<std::uint32_t>& tensor_ids,
    std::string jit_path) {
  NodePtr cached_node = nullptr;
  for (auto tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (cached_node == nullptr) {
      cached_node = node;
      experts_[expert_idx][layer_idx]->node = node;
      // experts_[expert_idx][layer_idx]->jit_module =
      //     new torch::jit::script::Module(torch::jit::load(jit_path));
    } else if (cached_node != node) {
      DLOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
    }
  }
}

void ExpertDispatcher::NotifyFetchStart() {
  for (int i = 0; i < kNumDevices(); ++i) {
    // std::unique_lock<std::mutex> lock(input_mutex_[i]);
    input_queue_[i].NotifyAll();
  }
}

void ExpertDispatcher::ClearExpertCacheCounts() {
  for (auto& expert : experts_) {
    for (auto& expert_node : expert) {
      if (expert_node->node == nullptr) {
        continue;
      }
      expert_node->node->incache_visit_count = 0;
    }
  }
  cache_hit_count_.store(0, std::memory_order_relaxed);
  cache_access_count_.store(0, std::memory_order_relaxed);
}

std::int64_t ExpertDispatcher::GetCacheOccupancyBytes() {
  std::int64_t total = 0;
  for (auto& gpu_cache : cached_experts_) {
    for (auto key : gpu_cache) {
      int layer_idx = static_cast<int>(key >> 32);
      int expert_idx = static_cast<int>(key & 0xFFFFFFFF);
      if (expert_idx < 0 || expert_idx >= static_cast<int>(experts_.size())) {
        continue;
      }
      if (layer_idx < 0 ||
          layer_idx >= static_cast<int>(experts_[expert_idx].size())) {
        continue;
      }
      auto& expert_node = experts_[expert_idx][layer_idx];
      if (expert_node && expert_node->node) {
        total += expert_node->node->byte_size;
      }
    }
  }
  return total;
}

double ExpertDispatcher::GetCacheHitRate() const {
  std::uint64_t access = cache_access_count_.load(std::memory_order_relaxed);
  if (access == 0) {
    return 0.0;
  }
  std::uint64_t hits = cache_hit_count_.load(std::memory_order_relaxed);
  return static_cast<double>(hits) / static_cast<double>(access);
}

// void ExpertDispatcher::GPUThreadFunc(int gpu_id) {
//   while (!main_thread_stop_flag_.load()) {
//   }
// }

ExpertNodePtr ExpertDispatcher::FindExpertEvict(int gpu_id) {
  uint64_t min_visit_count = INT_MAX;
  ExpertNodePtr evict_expert_node = nullptr;

  for (auto& key : cached_experts_[gpu_id]) {
    auto layer_idx = key >> 32;
    auto expert_idx = key & 0xFFFFFFFF;
    auto node = experts_[expert_idx][layer_idx]->node;
    if (node == nullptr) continue;
    if (node->device.is_cuda() && node->incache_visit_count < min_visit_count &&
        node->pending_dispatches.load(std::memory_order_acquire) == 0 &&
        node->exec_state.load(std::memory_order_acquire) ==
            NodeExecState::IDLE) {
      evict_expert_node = experts_[expert_idx][layer_idx];
      min_visit_count = node->incache_visit_count;
    }
  }
  return evict_expert_node;
}

void ExpertDispatcher::GPUFetchFunc(int gpu_id) {
#ifndef NVTX_DISABLE
  nvtx3::scoped_range r("gpu_fetch");
#endif
  cudaSetDevice(gpu_id);
  cudaStream_t stream = fetch_streams_[gpu_id];

  while (!main_thread_stop_flag_.load(std::memory_order_acquire)) {
    // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
    // if (cache_ == nullptr) {
    //   auto cache_limit =
    //   kDeviceMemoryPool->GetSparseCacheLimit(torch::Device(torch::kCUDA,
    //   gpu_id));
    //   // get any one expert size
    //   auto num_layers = experts_[0].size();
    //   auto num_experts = experts_.size();
    //   auto expert_node = experts_[num_layers-1][num_experts-1];

    //   int cache_capacity = cache_limit / expert_node->node->byte_size;
    //   cache_capacity_ = cache_capacity;
    // }
    // std::unique_lock<std::mutex> lock(input_mutex_[gpu_id]);
    // input_cv_[gpu_id].wait(lock, [&] { return !input_queue_[gpu_id].empty();
    // });

    // CallArgs args = std::move(input_queue_[gpu_id].front());
    // input_queue_[gpu_id].pop_front();

    // lock.unlock();
    CallArgs args;
    if (!input_queue_[gpu_id].Pop(args)) {
      break;
    }

    FailureContext failure;
    failure.gpu_id = gpu_id;
    try {
      int fetch_fault = static_cast<int>(DispatchFaultPoint::FETCH_WORKER);
      if (dispatch_fault_for_test_.compare_exchange_strong(
              fetch_fault, 0, std::memory_order_acq_rel)) {
        throw std::runtime_error("injected fetch failure");
      }

      auto device = CUDA_DEVICE(gpu_id);
      auto original_device =
          (args.remote) ? CPU_DEVICE : hidden_states_.device();
      int64_t layer_idx = args.layer_idx;
      int64_t expert_idx = args.expert_idx;
      int64_t batch_size = hidden_states_.size(0);

      auto expert_node = experts_[expert_idx][layer_idx];
      failure.expert_node = expert_node;

      if (args.wait_for_prefetch) {
        while (expert_node->node->exec_state.load(std::memory_order_acquire) !=
                   NodeExecState::IDLE &&
               !main_thread_stop_flag_.load(std::memory_order_acquire)) {
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        if (main_thread_stop_flag_.load(std::memory_order_acquire)) break;
        {
          auto expected = NodeExecState::IDLE;
          expert_node->node->exec_state.compare_exchange_strong(
              expected, NodeExecState::FETCHING, std::memory_order_acq_rel);
        }
        expert_node->node->pending_dispatches.fetch_sub(
            1, std::memory_order_acq_rel);
        if (expert_node->node->device.is_cuda()) {
          expert_node->node->incache_visit_count += 1;
          expert_node->SetTensorsFromBlob(expert_node->node->device);
          ExecArgs exec_args;
          exec_args.expert_node = expert_node;
          exec_args.out_gpu_id = original_device.index();
          exec_args.out_dtype =
              c10::typeMetaToScalarType(hidden_states_.dtype());
          exec_args.evict = false;
          exec_args.hit = true;
          exec_args.generation = args.generation;
          cache_hit_count_.fetch_add(1, std::memory_order_relaxed);
          cache_access_count_.fetch_add(1, std::memory_order_relaxed);
          exec_args.transfer_event = nullptr;
          exec_queue_[gpu_id].Push(exec_args);
          continue;
        }
      }

      bool cache_hit = expert_node->node->device.is_cuda();

      // std::cerr << "ExpertDispatcher::GPUFetchFunc: gpu_id " << gpu_id
      //           << " layer_idx " << layer_idx << " expert_idx " << expert_idx
      //           << " cache_hit " << cache_hit << " node "
      //           << expert_node->node->device.str() << std::endl;
      DLOG_DEBUG("ExpertDispatcher::GPUFetchFunc: gpu_id ", gpu_id,
                 " layer_idx ", layer_idx, " expert_idx ", expert_idx,
                 "cache_hit ", cache_hit, "cache_size ", cache_sizes_[gpu_id],
                 " incache count ", cached_experts_[gpu_id].size());

      if (!cache_hit && cache_sizes_[gpu_id] < expert_node->node->byte_size) {
        if (batch_size > 1) {
          // force fetch to GPU regardless of cache size, only for prefill
          // only one extra cache slot for prefill
          DLOG_DEBUG("overloading expert cache: gpu_id ", gpu_id,
                     " cache size ", cache_sizes_[gpu_id], " incache count ",
                     cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
                     " expert_idx ", expert_idx);
          while (gpu_overload_[gpu_id].load(std::memory_order_acquire) &&
                 !main_thread_stop_flag_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
          }
          if (main_thread_stop_flag_.load(std::memory_order_acquire)) {
            break;
          }
          gpu_overload_[gpu_id].store(true, std::memory_order_release);
          failure.overload_owned = true;
        } else {
          // find the expert in gpu and min incache_visit_count
          ExpertNodePtr evict_expert_node = FindExpertEvict(gpu_id);
          if (evict_expert_node == nullptr) {
            // wait for notification that cache is available
            DLOG_WARN(
                "All cached expert locked, waiting for cache to be available. "
                "gpu_id ",
                gpu_id, " cache size ", cache_sizes_[gpu_id], " incache count ",
                cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
                " expert_idx ", expert_idx);
            {
              std::unique_lock<std::mutex> lock(cache_mutex_[gpu_id]);
              cache_cv_[gpu_id].wait(lock, [&] {
                return main_thread_stop_flag_.load(std::memory_order_acquire) ||
                       FindExpertEvict(gpu_id) != nullptr;
              });
            }
            if (main_thread_stop_flag_.load(std::memory_order_acquire)) break;
            evict_expert_node = FindExpertEvict(gpu_id);
          }
          // auto num_layers = experts_[0].size();
          // auto num_experts = experts_.size();

          // for (size_t i = 0; i < num_experts; ++i) {
          //   for (size_t j = 0; j < num_layers; ++j) {
          // auto node = experts_[i][j]->node;
          // if (node == nullptr) {
          //   // std::cerr << "ExpertDispatcher::GPUFetchFunc: node is nullptr"
          //   //           << " layer_idx " << j << " expert_idx " << i <<
          //   //           std::endl;
          //   continue;
          // }
          // if (node->device.is_cuda() &&
          //     node->incache_visit_count < min_visit_count &&
          //     node->mutex.try_lock()) {
          //   evict_node = node;
          //   min_visit_count = node->incache_visit_count;
          //   node->mutex.unlock();
          //   // std::cerr << "ExpertDispatcher::GPUFetchFunc: evict node "
          //   //           << evict_node->device.str() << " incache_visit_count
          //   "
          //   //           << min_visit_count << std::endl;
          // }
          //   }
          // }
          if (evict_expert_node == nullptr) {
            for (int retry = 0;
                 retry < 100 && evict_expert_node == nullptr &&
                 !main_thread_stop_flag_.load(std::memory_order_acquire);
                 ++retry) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
              evict_expert_node = FindExpertEvict(gpu_id);
            }
            if (main_thread_stop_flag_.load(std::memory_order_acquire)) break;
            DLOG_FATAL_IF(
                evict_expert_node == nullptr,
                "ExpertDispatcher::GPUFetchFunc: evict_node is nullptr after "
                "retries, gpu_id",
                gpu_id, "cache size", cache_sizes_[gpu_id], "in cache count",
                cached_experts_[gpu_id].size());
          }

          DLOG_DEBUG("evicting expert: gpu_id ", gpu_id, " cache size ",
                     cache_sizes_[gpu_id], " incache count ",
                     cached_experts_[gpu_id].size(), " layer_idx ", layer_idx,
                     " expert_idx ", expert_idx);

          auto evict_node = evict_expert_node->node;
          evict_node->SetDevice(evict_node->default_host);
          cache_sizes_[gpu_id] += evict_node->byte_size;
          int64_t evict_layer_idx = evict_expert_node->layer_idx;
          int64_t evict_expert_idx = evict_expert_node->expert_idx;

          // std::lock_guard<std::mutex> lock(cache_mutex_[gpu_id]);
          uint64_t evict_key = (evict_layer_idx << 32) + evict_expert_idx;
          auto it = cached_experts_[gpu_id].find(evict_key);
          if (it != cached_experts_[gpu_id].end()) {
            cached_experts_[gpu_id].erase(it);
          } else {
            DLOG_FATAL(
                "ExpertDispatcher::GPUFetchFunc: evict_key not found. "
                "layer_idx ",
                evict_layer_idx, " expert_idx ", evict_expert_idx);
          }
        }
      }

      if (!gpu_overload_[gpu_id].load(std::memory_order_acquire)) {
        cache_sizes_[gpu_id] -= expert_node->node->byte_size;
        failure.cache_slot_reserved = true;
        uint64_t key = (layer_idx << 32) + expert_idx;
        cached_experts_[gpu_id].insert(key);
        failure.cache_key_inserted = true;
      }

      cudaEvent_t transfer_done = nullptr;
      expert_node->node->SetDevice(device, true, stream, &transfer_done);
      expert_node->node->incache_visit_count += 1;
      expert_node->SetTensorsFromBlob(device);
      // module_->SetTensorsFromIds(expert_node->node->tensor_ids);

      // std::cerr << "ExpertDispatcher::GPUFetchFunc: move to device gpu_id "
      //           << gpu_id << " layer_idx " << layer_idx << " expert_idx "
      //           << expert_idx << " node "
      //           << expert_node->node->device.str() << std::endl;

      // DLOG_TRACE("ExpertDispatcher::GPUFetchFunc gpu_id ", gpu_id, "layer_idx
      // ",
      //            layer_idx, "expert_idx ", expert_idx, "input ",
      //            input.device().str(), "node ",
      //            expert_node->node->device.str());
      {
        ExecArgs exec_args;
        // exec_args.hidden_states = std::move(input);
        exec_args.expert_node = expert_node;
        exec_args.out_gpu_id = original_device.index();
        exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
        exec_args.evict = gpu_overload_[gpu_id].load(std::memory_order_acquire);
        exec_args.hit = cache_hit;
        exec_args.generation = args.generation;
        exec_args.cache_slot_reserved = failure.cache_slot_reserved;
        exec_args.cache_key_inserted = failure.cache_key_inserted;
        cache_access_count_.fetch_add(1, std::memory_order_relaxed);
        if (cache_hit) cache_hit_count_.fetch_add(1, std::memory_order_relaxed);
        exec_args.transfer_event = transfer_done;
        // std::lock_guard<std::mutex> lock(exec_mutex_[gpu_id]);
        // exec_queue_[gpu_id].emplace_back(std::move(exec_args));
        exec_queue_[gpu_id].Push(exec_args);
      }
      // exec_cv_[gpu_id].notify_all();
    } catch (...) {
      FailDispatch(args.generation, std::current_exception(), {failure});
    }
  }
}

void ExpertDispatcher::GPUExecFunc(int gpu_id, int thread_idx) {
  cudaSetDevice(gpu_id);
  cudaStream_t stream = exec_streams_[thread_idx];

  while (!main_thread_stop_flag_.load(std::memory_order_acquire)) {
    ExecArgs args;
    if (!exec_queue_[gpu_id].Pop(args)) {
      break;
    }

    if (args.expert_node == nullptr) {
      continue;
    }

    FailureContext failure{
        args.expert_node,        gpu_id,     args.cache_slot_reserved,
        args.cache_key_inserted, args.evict,
    };
    try {
      int exec_fault = static_cast<int>(DispatchFaultPoint::EXEC_WORKER);
      if (dispatch_fault_for_test_.compare_exchange_strong(
              exec_fault, 0, std::memory_order_acq_rel)) {
        throw std::runtime_error("injected exec failure");
      }

      c10::cuda::CUDAStream torch_stream =
          c10::cuda::getStreamFromExternal(stream, gpu_id);
      c10::cuda::CUDAStreamGuard guard(torch_stream);

      if (args.transfer_event != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, args.transfer_event, 0));
        kCudaEventPool->Release(args.transfer_event);
        args.transfer_event = nullptr;
      }

      int64_t batch_size = hidden_states_.size(0);
      auto device = CUDA_DEVICE(gpu_id);
      auto expert_idx = args.expert_node->expert_idx;

      // Order the exec stream after the producer stream that wrote
      // hidden_states_/router_mask_ (recorded in SetInputs).
      if (input_ready_event_ != nullptr) {
        cudaStreamWaitEvent(stream, input_ready_event_, 0);
      }

      auto token_mask = router_mask_.index({"...", expert_idx});
      torch::Tensor input = (batch_size == 1)
                                ? hidden_states_.to(device)
                                : hidden_states_.index({token_mask}).to(device);

      modules_[thread_idx]->SetTensorsFromIds(
          args.expert_node->node->tensor_ids);

      if (fp8_in_store_) {
        int64_t layer_idx = args.expert_node->layer_idx;
        int64_t expert_idx_val = args.expert_node->expert_idx;
        if (layer_idx < (int64_t)fp8_scales_.size() &&
            expert_idx_val < (int64_t)fp8_scales_[layer_idx].size()) {
          modules_[thread_idx]->SetFp8Scales(
              fp8_scales_[layer_idx][expert_idx_val]);
          modules_[thread_idx]->DequantFp8Params(stream);
        }
      }

      if (expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE) {
        modules_[thread_idx]->DequantMxfp4Params(stream);
      }

      torch::Tensor output;
      {
#ifndef NVTX_DISABLE
        nvtx3::scoped_range r("expert_compute");
#endif
        output = modules_[thread_idx]->forward(input, stream);
      }
      OutputFunc(args, output, token_mask, gpu_id, stream);
    } catch (...) {
      if (args.transfer_event != nullptr) {
        kCudaEventPool->Release(args.transfer_event);
        args.transfer_event = nullptr;
      }
      FailDispatch(args.generation, std::current_exception(), {failure});
    }
  }
}

bool ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output,
                                  torch::Tensor token_mask, int gpu_id,
                                  cudaStream_t exec_stream) noexcept {
  FailureContext failure{
      args.expert_node,        gpu_id,     args.cache_slot_reserved,
      args.cache_key_inserted, args.evict,
  };
  try {
    int output_fault = static_cast<int>(DispatchFaultPoint::OUTPUT);
    if (dispatch_fault_for_test_.compare_exchange_strong(
            output_fault, 0, std::memory_order_acq_rel)) {
      throw std::runtime_error("injected output failure");
    }

    auto output_device =
        (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
    torch::Tensor output_tensor = output.to(output_device).to(torch::kFloat32);

    int64_t expert_idx = args.expert_node->expert_idx;
    int64_t batch_size = hidden_states_.size(0);

    {
      std::lock_guard<std::mutex> lock(accum_mutex_);
      // Drop stragglers from a failed or superseded generation: after
      // FailDispatch zeros pending_, a late worker must not accumulate into a
      // reused final_hidden_states_ or leave a stray completion event that
      // would trip the next dispatch's empty-events check.
      if (current_generation_.load(std::memory_order_acquire) !=
              args.generation ||
          failed_generation_.load(std::memory_order_acquire) ==
              args.generation) {
        return true;
      }
      if (batch_size == 1) {
        final_hidden_states_.add_(
            output_tensor *
            router_weight_.index({torch::indexing::Slice(), expert_idx}));
      } else {
        auto token_indices = torch::nonzero(token_mask).squeeze(1);
        auto weights =
            router_weight_.index({token_mask, expert_idx}).unsqueeze(1);
        auto weighted_output = output_tensor * weights;
        final_hidden_states_.index_add_(0, token_indices, weighted_output);
      }

      cudaEvent_t output_done = AcquireCompletionEvent(gpu_id);
      int event_fault =
          static_cast<int>(DispatchFaultPoint::COMPLETION_EVENT_RECORD);
      if (dispatch_fault_for_test_.compare_exchange_strong(
              event_fault, 0, std::memory_order_acq_rel)) {
        ReleaseCompletionEvent(output_done, gpu_id);
        throw std::runtime_error("injected completion_event failure");
      }
      cudaError_t record_status = cudaEventRecord(output_done, exec_stream);
      if (record_status != cudaSuccess) {
        ReleaseCompletionEvent(output_done, gpu_id);
        throw std::runtime_error(
            std::string("OutputFunc: cudaEventRecord failed: ") +
            cudaGetErrorString(record_status));
      }
      {
        std::lock_guard<std::mutex> state_lock(route_state_mutex_);
        completion_events_.push_back(
            CompletionEventRecord{output_done, args.generation, gpu_id});
        completion_events_outstanding_.fetch_add(1, std::memory_order_relaxed);
      }
    }

    auto* retire = new ExpertRetireArgs{this, args.expert_node, args.generation,
                                        gpu_id, args.evict};
    pending_retirement_callbacks_.fetch_add(1, std::memory_order_acq_rel);
    int retirement_fault =
        static_cast<int>(DispatchFaultPoint::RETIREMENT_CALLBACK_LAUNCH);
    if (dispatch_fault_for_test_.compare_exchange_strong(
            retirement_fault, 0, std::memory_order_acq_rel)) {
      pending_retirement_callbacks_.fetch_sub(1, std::memory_order_acq_rel);
      delete retire;
      throw std::runtime_error("injected retirement_launch failure");
    }
    cudaError_t retire_status = cudaLaunchHostFunc(
        exec_stream, &ExpertDispatcher::ExpertReadyToRetireCallback, retire);
    if (retire_status != cudaSuccess) {
      pending_retirement_callbacks_.fetch_sub(1, std::memory_order_acq_rel);
      delete retire;
      throw std::runtime_error(
          std::string("cudaLaunchHostFunc expert retirement failed: ") +
          cudaGetErrorString(retire_status));
    }

    CompleteOne(args.generation);
    return true;
  } catch (...) {
    FailDispatch(args.generation, std::current_exception(), {failure});
    return false;
  }
}

std::vector<ExpertDispatcher::CallResult> ExpertDispatcher::Wait() {
  // int wait_count = 0;

  std::unique_lock<std::mutex> lock(pending_mutex_);
  pending_cv_.wait(lock, [&] { return pending_.load() == 0; });
  SyncExecStreamsWithCurrent();

  num_enqueued_.store(0);
  std::vector<CallResult> output_queue;
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue.swap(output_queue_);
  }

  return output_queue;
}

torch::Tensor ExpertDispatcher::WaitHiddenStates() {
#ifndef NVTX_DISABLE
  nvtx3::scoped_range range("expert_completion_handoff");
#endif
  std::unique_lock<std::mutex> lock(pending_mutex_);
  pending_cv_.wait(lock, [&] {
    return !route_pending_.load(std::memory_order_acquire) &&
           pending_.load(std::memory_order_acquire) == 0;
  });
  lock.unlock();

  const std::uint64_t generation =
      current_generation_.load(std::memory_order_acquire);
  std::exception_ptr route_error;
  std::vector<CompletionEventRecord> events;
  {
    std::lock_guard<std::mutex> state_lock(route_state_mutex_);
    route_error.swap(route_error_);
    auto it = completion_events_.begin();
    while (it != completion_events_.end()) {
      if (it->generation == generation) {
        events.push_back(*it);
        it = completion_events_.erase(it);
      } else {
        ++it;
      }
    }
  }

  const int device = final_hidden_states_.device().index();
  c10::cuda::CUDAGuard device_guard(device);
  auto caller_stream = c10::cuda::getCurrentCUDAStream(device).stream();
  std::size_t waits_inserted = 0;
  try {
    for (const CompletionEventRecord& record : events) {
      CUDA_CHECK(cudaStreamWaitEvent(caller_stream, record.event, 0));
      ++waits_inserted;
    }
    if (!events.empty()) {
      auto callback = std::make_unique<CompletionRetireBatch>(
          CompletionRetireBatch{this, std::move(events)});
      pending_completion_retirement_callbacks_.fetch_add(
          1, std::memory_order_acq_rel);
      cudaError_t status = cudaLaunchHostFunc(
          caller_stream, &ExpertDispatcher::CompletionWaitsConsumedCallback,
          callback.get());
      if (status != cudaSuccess) {
        pending_completion_retirement_callbacks_.fetch_sub(
            1, std::memory_order_acq_rel);
        events = std::move(callback->records);
        throw std::runtime_error(
            std::string("completion retirement callback launch failed: ") +
            cudaGetErrorString(status));
      }
      callback.release();
    }
  } catch (...) {
    std::exception_ptr cleanup_error = std::current_exception();
    std::vector<CompletionEventRecord> unwaited(
        std::make_move_iterator(events.begin() + waits_inserted),
        std::make_move_iterator(events.end()));
    events.erase(events.begin() + waits_inserted, events.end());
    QueueUnwaitedEventsForQuery(std::move(unwaited));
    {
      std::lock_guard<std::mutex> state_lock(route_state_mutex_);
      destructor_fallback_events_.insert(
          destructor_fallback_events_.end(),
          std::make_move_iterator(events.begin()),
          std::make_move_iterator(events.end()));
    }
    std::exception_ptr error = route_error ? route_error : cleanup_error;
    FailDispatch(generation, error);
    std::rethrow_exception(error);
  }
  num_enqueued_.store(0, std::memory_order_release);
  if (route_error) std::rethrow_exception(route_error);
  return final_hidden_states_;
}

void ExpertDispatcher::SetInputs(const torch::Tensor& hidden_states,
                                 const torch::Tensor& router_mask,
                                 const torch::Tensor& router_weight) {
  TORCH_CHECK(!route_pending_.load(std::memory_order_acquire) &&
                  pending_.load(std::memory_order_acquire) == 0,
              "SetInputs: previous dispatch is still active");
  current_generation_.store(
      dispatch_generation_.fetch_add(1, std::memory_order_acq_rel) + 1,
      std::memory_order_release);
  failed_generation_.store(0, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(route_state_mutex_);
    route_error_ = nullptr;
  }
  int device = at::cuda::current_device();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(CUDA_DEVICE(device));
  hidden_states_ = hidden_states;
  router_mask_ = router_mask;
  router_weight_ = router_weight;  // this can be float32
  final_hidden_states_ = torch::zeros_like(hidden_states, options);

  if (input_ready_event_ == nullptr) {
    cudaEventCreateWithFlags(&input_ready_event_, cudaEventDisableTiming);
  }
  cudaEventRecord(input_ready_event_, c10::cuda::getCurrentCUDAStream());
}

void ExpertDispatcher::SyncExecStreamsWithCurrent() {
  auto current = c10::cuda::getCurrentCUDAStream();
  for (size_t i = 0; i < exec_streams_.size(); ++i) {
    cudaEventRecord(exec_done_events_[i], exec_streams_[i]);
    cudaStreamWaitEvent(current.stream(), exec_done_events_[i], 0);
  }
}

void ExpertDispatcher::DispatchExperts(int layer_idx) {
  TORCH_CHECK(router_mask_.defined(),
              "DispatchExperts: SetInputs must be called first");
  TORCH_CHECK(router_mask_.is_cuda(),
              "DispatchExperts: router_mask must be CUDA-resident");
  TORCH_CHECK(router_mask_.dim() == 2 && router_mask_.size(1) == num_experts_,
              "DispatchExperts: router_mask must be [tokens, num_experts]");

  bool expected = false;
  TORCH_CHECK(route_pending_.compare_exchange_strong(expected, true,
                                                     std::memory_order_acq_rel),
              "DispatchExperts: previous dispatch has not been waited");
  const std::uint64_t generation =
      current_generation_.load(std::memory_order_acquire);
  DispatchSubmissionGuard submission(this, generation);
  try {
    int submission_fault = static_cast<int>(DispatchFaultPoint::SUBMISSION);
    if (dispatch_fault_for_test_.compare_exchange_strong(
            submission_fault, 0, std::memory_order_acq_rel)) {
      throw std::runtime_error("injected submission failure");
    }
    {
      std::lock_guard<std::mutex> lock(route_state_mutex_);
      TORCH_CHECK(completion_events_.empty(),
                  "DispatchExperts: unreaped completion events remain active");
      completion_events_.reserve(num_experts_);
    }
    const int device = router_mask_.device().index();
    c10::cuda::CUDAGuard device_guard(device);
    auto active_flags = router_mask_.to(torch::kBool).any(0).contiguous();
    auto host_options = torch::TensorOptions()
                            .dtype(torch::kBool)
                            .device(torch::kCPU)
                            .pinned_memory(true);
    auto active_flags_host = torch::empty({num_experts_}, host_options);
    active_flags_host.copy_(active_flags, true);
    auto stream = c10::cuda::getCurrentCUDAStream(device).stream();

    RouteArgs args;
    args.layer_idx = layer_idx;
    args.active_flags_host = active_flags_host;
    args.generation = generation;
    auto callback_args = std::make_unique<RouteCallbackArgs>(
        RouteCallbackArgs{this, std::move(args)});
    pending_route_callbacks_.fetch_add(1, std::memory_order_acq_rel);
    cudaError_t status = cudaLaunchHostFunc(
        stream, &ExpertDispatcher::RouteReadyCallback, callback_args.get());
    if (status != cudaSuccess) {
      pending_route_callbacks_.fetch_sub(1, std::memory_order_acq_rel);
      throw std::runtime_error(
          std::string("DispatchExperts: cudaLaunchHostFunc failed: ") +
          cudaGetErrorString(status));
    }
    callback_args.release();
    submission.Release();
  } catch (...) {
    submission.Capture(std::current_exception());
    throw;
  }
}

void ExpertDispatcher::FailDispatch(
    const std::uint64_t failing_generation, std::exception_ptr error,
    const std::vector<FailureContext>& contexts) noexcept {
  const std::uint64_t current =
      current_generation_.load(std::memory_order_acquire);
  if (failing_generation != current) {
    stale_failures_quarantined_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(route_state_mutex_);
    if (failing_generation !=
        current_generation_.load(std::memory_order_acquire)) {
      stale_failures_quarantined_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    if (!route_error_) route_error_ = std::move(error);
    last_active_experts_.clear();
  }
  failed_generation_.store(failing_generation, std::memory_order_release);
  for (const FailureContext& context : contexts) {
    if (context.expert_node && context.expert_node->node) {
      auto node = context.expert_node->node;
      if (context.overload_owned && node->device.is_cuda()) {
        try {
          node->SetDevice(node->default_host, true, nullptr);
        } catch (...) {
          // FailDispatch is noexcept; preserve the first dispatch exception
          // while still restoring terminal state below.
        }
      }
      if (context.gpu_id >= 0) {
        try {
          uint64_t key =
              (static_cast<uint64_t>(context.expert_node->layer_idx) << 32) |
              static_cast<uint32_t>(context.expert_node->expert_idx);
          std::lock_guard<std::mutex> cache_lock(cache_mutex_[context.gpu_id]);
          if (node->device.is_cuda() && !context.overload_owned) {
            cached_experts_[context.gpu_id].insert(key);
          } else if (context.cache_key_inserted) {
            cached_experts_[context.gpu_id].erase(key);
            if (context.cache_slot_reserved) {
              cache_sizes_[context.gpu_id] += node->byte_size;
            }
          }
        } catch (...) {
          // FailDispatch is noexcept; continue terminal restoration.
        }
      }
      node->exec_state.store(NodeExecState::IDLE, std::memory_order_release);
    }
    if (context.overload_owned && context.gpu_id >= 0) {
      gpu_overload_[context.gpu_id].store(false, std::memory_order_release);
    }
    if (context.gpu_id >= 0) cache_cv_[context.gpu_id].notify_all();
  }
  route_failures_.fetch_add(1, std::memory_order_relaxed);
  last_active_experts_count_.store(0, std::memory_order_relaxed);
  pending_.store(0, std::memory_order_release);
  route_pending_.store(false, std::memory_order_release);
  pending_cv_.notify_all();
  route_callback_cv_.notify_all();
  retirement_callback_cv_.notify_all();
}

void ExpertDispatcher::CompleteOne(std::uint64_t generation) noexcept {
  if (current_generation_.load(std::memory_order_acquire) != generation) return;
  if (failed_generation_.load(std::memory_order_acquire) == generation) return;
  size_t previous = pending_.fetch_sub(1, std::memory_order_acq_rel);
  if (previous <= 1) {
    pending_.store(0, std::memory_order_release);
    pending_cv_.notify_all();
  }
}

void CUDART_CB ExpertDispatcher::RouteReadyCallback(void* opaque) {
  std::unique_ptr<RouteCallbackArgs> callback(
      static_cast<RouteCallbackArgs*>(opaque));
  ExpertDispatcher* dispatcher = callback->dispatcher;
  try {
    int callback_fault = static_cast<int>(DispatchFaultPoint::ROUTE_CALLBACK);
    if (dispatcher->dispatch_fault_for_test_.compare_exchange_strong(
            callback_fault, 0, std::memory_order_acq_rel)) {
      throw std::runtime_error("injected callback routing failure");
    }
    dispatcher->route_queue_.Push(callback->route);
  } catch (...) {
    dispatcher->FailDispatch(callback->route.generation,
                             std::current_exception());
    // Hand the pinned RouteArgs to the route worker so its owning
    // torch::Tensor is destroyed off the CUDA host-callback thread, where a
    // pinned free would issue a forbidden CUDA call. RouteFunc drops already
    // failed generations without routing.
    dispatcher->route_queue_.Push(callback->route);
  }
  if (dispatcher->pending_route_callbacks_.fetch_sub(
          1, std::memory_order_acq_rel) == 1) {
    // Serialize with the destructor's predicate load to close the
    // notify-after-check lost-wakeup window on the drain condition variable.
    std::lock_guard<std::mutex> lock(dispatcher->route_callback_mutex_);
    dispatcher->route_callback_cv_.notify_all();
  }
}

void ExpertDispatcher::RouteFunc() noexcept {
  RouteArgs args;
  while (route_queue_.Pop(args)) {
    if (failed_generation_.load(std::memory_order_acquire) == args.generation &&
        args.generation != 0) {
      args.active_flags_host = torch::Tensor();
      continue;
    }
    std::vector<FailureContext> routed_contexts;
    try {
#ifndef NVTX_DISABLE
      nvtx3::scoped_range range("gpu_route_handoff");
#endif
      int worker_fault = static_cast<int>(DispatchFaultPoint::ROUTE_WORKER);
      if (dispatch_fault_for_test_.compare_exchange_strong(
              worker_fault, 0, std::memory_order_acq_rel)) {
        throw std::runtime_error("injected worker routing failure");
      }
      const auto started = std::chrono::steady_clock::now();
      std::vector<int> active_experts;
      const bool* flags = args.active_flags_host.data_ptr<bool>();
      for (int expert_idx = 0; expert_idx < num_experts_; ++expert_idx) {
        if (flags[expert_idx]) active_experts.push_back(expert_idx);
      }
      pending_.store(active_experts.size(), std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(route_state_mutex_);
        last_active_experts_ = active_experts;
      }
      for (int expert_idx : active_experts) {
        int gpu_id = expert_idx % kNumDevices();
        routed_contexts.push_back(FailureContext{
            experts_[expert_idx][args.layer_idx], gpu_id, false, false, false});
        EnqueueExpert(args.layer_idx, expert_idx, expert_idx % kNumDevices(),
                      false);
      }
      NotifyFetchStart();
      route_batches_.fetch_add(1, std::memory_order_relaxed);
      last_active_experts_count_.store(active_experts.size(),
                                       std::memory_order_relaxed);
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - started);
      last_route_handoff_us_.store(elapsed.count(), std::memory_order_relaxed);
      route_pending_.store(false, std::memory_order_release);
      pending_cv_.notify_all();
    } catch (...) {
      FailDispatch(args.generation, std::current_exception(), routed_contexts);
    }
  }
}

void ExpertDispatcher::SetDispatchFaultForTest(const std::string& stage) {
  static const std::map<std::string, DispatchFaultPoint> faults = {
      {"callback", DispatchFaultPoint::ROUTE_CALLBACK},
      {"worker", DispatchFaultPoint::ROUTE_WORKER},
      {"fetch", DispatchFaultPoint::FETCH_WORKER},
      {"exec", DispatchFaultPoint::EXEC_WORKER},
      {"output", DispatchFaultPoint::OUTPUT},
      {"completion_event", DispatchFaultPoint::COMPLETION_EVENT_RECORD},
      {"retirement_launch", DispatchFaultPoint::RETIREMENT_CALLBACK_LAUNCH},
      {"submission", DispatchFaultPoint::SUBMISSION},
  };
  auto it = faults.find(stage);
  TORCH_CHECK(it != faults.end(), "unknown dispatch fault stage: ", stage);
  dispatch_fault_for_test_.store(static_cast<int>(it->second),
                                 std::memory_order_release);
}

std::vector<int> ExpertDispatcher::TakeLastActiveExperts() {
  std::lock_guard<std::mutex> lock(route_state_mutex_);
  return last_active_experts_;
}

std::map<std::string, std::int64_t> ExpertDispatcher::GetRoutingStats() const {
  std::lock_guard<std::mutex> state_lock(route_state_mutex_);
  return {
      {"route_batches", route_batches_.load(std::memory_order_relaxed)},
      {"route_failures", route_failures_.load(std::memory_order_relaxed)},
      {"last_active_experts",
       last_active_experts_count_.load(std::memory_order_relaxed)},
      {"last_route_handoff_us",
       last_route_handoff_us_.load(std::memory_order_relaxed)},
      {"completion_events_retired",
       completion_events_retired_.load(std::memory_order_relaxed)},
      {"completion_events_outstanding",
       completion_events_outstanding_.load(std::memory_order_relaxed)},
      {"stale_failures_quarantined",
       stale_failures_quarantined_.load(std::memory_order_relaxed)},
      {"current_generation", static_cast<std::int64_t>(current_generation_.load(
                                 std::memory_order_acquire))},
      {"pending",
       static_cast<std::int64_t>(pending_.load(std::memory_order_acquire))},
      {"route_pending", route_pending_.load(std::memory_order_acquire) ? 1 : 0},
  };
}

void ExpertDispatcher::FailDispatchForTest(std::uint64_t generation,
                                           const std::string& message) {
  FailDispatch(generation,
               std::make_exception_ptr(std::runtime_error(message)));
}

void ExpertDispatcher::QueueUnwaitedEventsForQuery(
    std::vector<CompletionEventRecord> records) noexcept {
  if (records.empty()) return;
  std::vector<CompletionRetireItem> items;
  items.reserve(records.size());
  for (const CompletionEventRecord& record : records) {
    items.push_back(CompletionRetireItem{record, false});
  }
  try {
    completion_retirement_queue_.Push(items);
  } catch (...) {
    std::lock_guard<std::mutex> lock(route_state_mutex_);
    destructor_fallback_events_.insert(destructor_fallback_events_.end(),
                                       std::make_move_iterator(records.begin()),
                                       std::make_move_iterator(records.end()));
  }
}

void CUDART_CB ExpertDispatcher::CompletionWaitsConsumedCallback(void* opaque) {
  std::unique_ptr<CompletionRetireBatch> batch(
      static_cast<CompletionRetireBatch*>(opaque));
  ExpertDispatcher* dispatcher = batch->dispatcher;
  try {
    std::vector<CompletionRetireItem> items;
    items.reserve(batch->records.size());
    for (const CompletionEventRecord& record : batch->records) {
      items.push_back(CompletionRetireItem{record, true});
    }
    dispatcher->completion_retirement_queue_.Push(items);
    batch->records.clear();
  } catch (...) {
    std::exception_ptr error = std::current_exception();
    const std::uint64_t generation = batch->records.front().generation;
    {
      std::lock_guard<std::mutex> lock(dispatcher->route_state_mutex_);
      dispatcher->destructor_fallback_events_.insert(
          dispatcher->destructor_fallback_events_.end(),
          std::make_move_iterator(batch->records.begin()),
          std::make_move_iterator(batch->records.end()));
    }
    dispatcher->FailDispatch(generation, error);
  }
  if (dispatcher->pending_completion_retirement_callbacks_.fetch_sub(
          1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> lock(
        dispatcher->completion_retirement_callback_mutex_);
    dispatcher->completion_retirement_callback_cv_.notify_all();
  }
}

void ExpertDispatcher::CompletionRetirementFunc() {
  std::vector<CompletionRetireItem> items;
  while (completion_retirement_queue_.Pop(items)) {
    std::vector<CompletionRetireItem> retry;
    for (CompletionRetireItem& item : items) {
      cudaError_t status = item.caller_wait_consumed
                               ? cudaSuccess
                               : cudaEventQuery(item.record.event);
      if (status == cudaErrorNotReady) {
        retry.push_back(std::move(item));
        continue;
      }
      if (status != cudaSuccess) {
        auto error = std::make_exception_ptr(
            std::runtime_error(std::string("completion event query failed: ") +
                               cudaGetErrorString(status)));
        FailDispatch(item.record.generation, error);
        std::lock_guard<std::mutex> lock(route_state_mutex_);
        destructor_fallback_events_.push_back(std::move(item.record));
        continue;
      }
      ReleaseCompletionEvent(item.record.event, item.record.device);
      completion_events_retired_.fetch_add(1, std::memory_order_relaxed);
      completion_events_outstanding_.fetch_sub(1, std::memory_order_relaxed);
    }
    if (!retry.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      completion_retirement_queue_.Push(retry);
    }
    items.clear();
  }
}

void CUDART_CB ExpertDispatcher::ExpertReadyToRetireCallback(void* opaque) {
  std::unique_ptr<ExpertRetireArgs> retire(
      static_cast<ExpertRetireArgs*>(opaque));
  ExpertDispatcher* dispatcher = retire->dispatcher;
  try {
    dispatcher->retirement_queue_.Push(*retire);
  } catch (...) {
    // Runs on a CUDA host-callback thread, so pass no FailureContext: node/
    // cache restoration issues CUDA calls that are forbidden here. State is
    // still closed and waiters notified; the destructor resets node state.
    dispatcher->FailDispatch(retire->generation, std::current_exception());
  }
  if (dispatcher->pending_retirement_callbacks_.fetch_sub(
          1, std::memory_order_acq_rel) == 1) {
    std::lock_guard<std::mutex> lock(dispatcher->retirement_callback_mutex_);
    dispatcher->retirement_callback_cv_.notify_all();
  }
}

cudaEvent_t ExpertDispatcher::AcquireCompletionEvent(int device) {
  {
    std::lock_guard<std::mutex> lock(completion_event_pool_mutex_);
    auto it = completion_event_pool_.find(device);
    if (it != completion_event_pool_.end() && !it->second.empty()) {
      cudaEvent_t event = it->second.back();
      it->second.pop_back();
      return event;
    }
  }
  c10::cuda::CUDAGuard device_guard(device);
  cudaEvent_t event = nullptr;
  CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  return event;
}

void ExpertDispatcher::ReleaseCompletionEvent(cudaEvent_t event,
                                              int device) noexcept {
  if (event == nullptr) return;
  std::lock_guard<std::mutex> lock(completion_event_pool_mutex_);
  completion_event_pool_[device].push_back(event);
}

void ExpertDispatcher::RetirementFunc() {
  ExpertRetireArgs args;
  while (retirement_queue_.Pop(args)) {
    try {
      if (args.evict) {
        args.expert_node->node->SetDevice(args.expert_node->node->default_host,
                                          true, nullptr);
        gpu_overload_[args.gpu_id].store(false, std::memory_order_release);
      }
      args.expert_node->node->exec_state.store(NodeExecState::IDLE,
                                               std::memory_order_release);
      cache_cv_[args.gpu_id].notify_all();
    } catch (...) {
      FailDispatch(args.generation, std::current_exception(),
                   {FailureContext{args.expert_node, args.gpu_id, false, false,
                                   args.evict}});
    }
  }
}
