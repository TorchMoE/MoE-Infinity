// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "prefetch/task_scheduler.h"
#include "prefetch/task_thread.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "utils/cache.h"
#include "model/model_topology.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <future>

ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype,
                                   int expert_type)
    : pending_(0),
      num_enqueued_(0),
      start_(false),
      expert_type_(expert_type),
      input_mutex_(8),
      input_cv_(8),
      exec_mutex_(8),
      exec_cv_(8),
      input_queue_(8),
      exec_queue_(8),
      gpu_overload_(8, false) {
  main_thread_stop_flag_.store(false);

  int num_gpu = GetDeviceCount();
  for (int i = 0; i < num_gpu; ++i) {
    cudaSetDevice(i);
    cudaStream_t fetch_stream;
    cudaStreamCreateWithFlags(&fetch_stream, cudaStreamNonBlocking);
    fetch_streams_.emplace_back(fetch_stream);

    cudaStream_t out_stream;
    cudaStreamCreateWithFlags(&out_stream, cudaStreamNonBlocking);
    out_streams_.emplace_back(out_stream);

    auto thread_func = std::bind(&ExpertDispatcher::GPUFetchFunc, this, i);
    threads_.emplace_back(new base::Thread(thread_func));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());

    auto cache_limit =
        kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, i));
    cache_sizes_.push_back(cache_limit);
  }

  // std::vector<std::condition_variable> input_cv(num_gpu);
  // input_cv_.swap(input_cv);

  // std::vector<std::mutex> input_mutex(num_gpu);
  // input_mutex_.swap(input_mutex);

  // for (int i = 0; i < num_gpu; ++i) {
  //   // std::thread t(&ExpertDispatcher::GPUThreadFunc, this, i);
  //   // SetThreadAffinity(t);
  //   // threads_.emplace_back(std::move(t));
  // }

  for (int i = 0; i < num_gpu * 8; ++i) {
    cudaSetDevice(i % num_gpu);
    cudaStream_t exec_stream;
    cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
    exec_streams_.emplace_back(exec_stream);
    // cudaDeviceSynchronize();

    auto thread_func =
        std::bind(&ExpertDispatcher::GPUExecFunc, this, i % num_gpu);
    threads_.emplace_back(new base::Thread(thread_func));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());
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
        case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
          experts_[i][j]->module = new SwitchTransformersDenseActDense(dtype);
          break;
        case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
          experts_[i][j]->module =
              new SwitchTransformersDenseGatedActDense(dtype);
          break;
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
  Enqueue(args);
}

void ExpertDispatcher::Enqueue(CallArgs& args) {
  // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
  int layer_idx = args.layer_idx;
  int expert_idx = args.expert_idx;
  auto expert_node = experts_[expert_idx][layer_idx];

  // if (!expert_node->node->mutex.try_lock()) {
  //     DLOG_WARN("ExpertDispatcher::Enqueue: mutex try_lock failed (expert_idx
  //     ",
  //                     expert_idx,
  //                     " layer_idx ",
  //                     layer_idx,
  //                     "node ",
  //                     expert_node->node->str(),
  //                     ")");
  //     return;
  // }
  expert_node->node->mutex.try_lock();
  expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

  if (expert_node->node->device.is_cuda()) {
    args.gpu_id = expert_node->node->device.index();
  }

  {
    std::unique_lock<std::mutex> lock(input_mutex_[args.gpu_id]);
    input_queue_[args.gpu_id].push_back(std::move(args));
  }
  input_cv_[args.gpu_id].notify_all();
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
    int layer_idx, int expert_idx,
    const std::vector<std::uint32_t>& tensor_ids) {
  NodePtr cached_node = nullptr;
  for (auto tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (cached_node == nullptr) {
      cached_node = node;
      experts_[expert_idx][layer_idx]->node = node;
    } else if (cached_node != node) {
      DLOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
    }
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
}

// void ExpertDispatcher::GPUThreadFunc(int gpu_id) {
//   while (!main_thread_stop_flag_.load()) {
//   }
// }

void ExpertDispatcher::GPUFetchFunc(int gpu_id) {
  while (!main_thread_stop_flag_.load()) {
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
    std::unique_lock<std::mutex> lock(input_mutex_[gpu_id]);
    input_cv_[gpu_id].wait(lock, [&] { return !input_queue_[gpu_id].empty(); });

    CallArgs args = std::move(input_queue_[gpu_id].front());
    input_queue_[gpu_id].pop_front();

    lock.unlock();

    // if (input_queue_.empty()) {
    //   // cvs_[MUTEX_TYPE::INPUT_MUTEX].wait(lock, [&] { return
    //   // !input_queue_.empty(); });
    //   lock.unlock();
    //   std::this_thread::sleep_for(std::chrono::microseconds(10));
    //   continue;
    // }

    // CallArgs args;

    // // find all the args with gpu_id
    // std::vector<CallArgs> args_list;
    // auto it = input_queue_.begin();
    // while (it != input_queue_.end()) {
    //   if (it->gpu_id == gpu_id) {
    //     args_list.emplace_back(std::move(*it));
    //     it = input_queue_.erase(it);
    //   } else {
    //     ++it;
    //   }
    // }

    // lock.unlock();

    // if (args_list.empty()) {
    //   continue;
    // }

    // for (auto& a : args_list) {
    //   int layer_idx = a.layer_idx;
    //   int expert_idx = a.expert_idx;
    //   auto expert_node = experts_[expert_idx][layer_idx];

    //   if (expert_node->node->device.is_cuda()) {
    //     args = std::move(a);
    //     break;
    //   }
    // }

    // if (args.gpu_id == -1) {
    //   args = std::move(args_list[0]);
    // }

    // //  put back the rest
    // args_list.erase(std::remove_if(args_list.begin(), args_list.end(),
    //                                [&](const CallArgs& a) {
    //                                  return (a.expert_idx == args.expert_idx)
    //                                  &&
    //                                         (a.layer_idx == args.layer_idx);
    //                                }),
    //                 args_list.end());

    // lock.lock();
    // for (auto& a : args_list) {
    //   input_queue_.push_back(std::move(a));
    // }
    // lock.unlock();

    auto device = CUDA_DEVICE(gpu_id);
    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();
    int layer_idx = args.layer_idx;
    int expert_idx = args.expert_idx;

    auto expert_node = experts_[expert_idx][layer_idx];
    bool cache_hit = expert_node->node->device.is_cuda();

    // std::cerr << "ExpertDispatcher::GPUFetchFunc: gpu_id " << gpu_id
    //           << " layer_idx " << layer_idx << " expert_idx " << expert_idx
    //           << " cache_hit " << cache_hit << " node "
    //           << expert_node->node->device.str() << std::endl;

    if (!expert_node->node->device.is_cuda() &&
        cache_sizes_[gpu_id] < expert_node->node->byte_size) {
      // find the expert in gpu and min incache_visit_count
      NodePtr evict_node = nullptr;
      auto num_layers = experts_[0].size();
      auto num_experts = experts_.size();
      int min_visit_count = INT_MAX;
      for (int i = 0; i < num_experts; ++i) {
        for (int j = 0; j < num_layers; ++j) {
          auto node = experts_[i][j]->node;
          if (node == nullptr) {
            // std::cerr << "ExpertDispatcher::GPUFetchFunc: node is nullptr"
            //           << " layer_idx " << j << " expert_idx " << i <<
            //           std::endl;
            continue;
          }
          if (node->device.is_cuda() &&
              node->incache_visit_count < min_visit_count &&
              node->mutex.try_lock()) {
            evict_node = node;
            min_visit_count = node->incache_visit_count;
            node->mutex.unlock();
            // std::cerr << "ExpertDispatcher::GPUFetchFunc: evict node "
            //           << evict_node->device.str() << " incache_visit_count "
            //           << min_visit_count << std::endl;
          }
        }
      }
      assert(evict_node != nullptr);
      evict_node->SetDevice(evict_node->default_host);
      cache_sizes_[gpu_id] += evict_node->byte_size;
    }

    bool success = true;
    // if (!expert_node->node->device.is_cuda()) {
    //   success = kTaskPool->RemoveCachedSparseNode(expert_node->node, gpu_id);

    //   int wait_count = 0;
    //   while (!success && gpu_overload_[gpu_id]) {
    //     std::this_thread::sleep_for(std::chrono::microseconds(10));
    //     wait_count++;
    //     // if (wait_count % 100000 == 0) {
    //     //     DLOG_WARN(
    //     //         "ExpertDispatcher::EnqueueTask: gpu_overload_ gpu_id {}
    //     //         wait_count {}
    //     //         {}", gpu_id, wait_count, expert_node->node->str());
    //     // }
    //   }

    //   if (!success && !gpu_overload_[gpu_id]) {
    //     {
    //       std::lock_guard<std::mutex> lock(gpu_overload_mutex_);
    //       gpu_overload_[gpu_id] = true;
    //     }
    //   }

    //   // c10::cuda::CUDAStream stream =
    //   //   c10::cuda::getStreamFromExternal(exec_streams_[gpu_id + rnd],
    //   //   gpu_id);
    //   expert_node->node->SetDevice(CUDA_DEVICE(gpu_id), true,
    //                                fetch_streams_[gpu_id]);

    //   // auto task = std::make_shared<Task>();
    //   // task->priority = 0;
    //   // task->node = expert_node->node;
    //   // task->on_demand = true;
    //   // task->src_device = expert_node->node->device;
    //   // task->dst_device = CUDA_DEVICE(gpu_id);
    //   // task->remove_layer = true;
    //   // kTaskPool->EnqueueTask(task);

    //   // wait_count = 0;
    //   // while (!expert_node->node->device.is_cuda()) {
    //   //   std::this_thread::sleep_for(std::chrono::microseconds(10));
    //   //   wait_count++;
    //   //   if (wait_count % 100000 == 0) {
    //   //     DLOG_WARN("ExpertDispatcher::EnqueueTask: wait_count ",
    //   wait_count,
    //   //               expert_node->node->str());
    //   //   }
    //   // }

    //   // DLOG_TRACE(
    //   //     "ExpertDispatcher::GPUFetchFunc: move to device gpu_id {}
    //   layer_idx
    //   //     {} expert_idx
    //   //     {} node {}", gpu_id, layer_idx, expert_idx,
    //   //     expert_node->node->device.str());
    // }
    expert_node->node->SetDevice(CUDA_DEVICE(gpu_id), true,
                                 fetch_streams_[gpu_id]);
    expert_node->node->incache_visit_count += 1;
    expert_node->SetTensorsFromBlob(device);
    cache_sizes_[gpu_id] -= expert_node->node->byte_size;
    // std::cerr << "ExpertDispatcher::GPUFetchFunc: move to device gpu_id "
    //           << gpu_id << " layer_idx " << layer_idx << " expert_idx "
    //           << expert_idx << " node "
    //           << expert_node->node->device.str() << std::endl;

    int expert_type = expert_type_;
    torch::Tensor input;
    auto token_indices =
        router_mask_.index({"...", expert_idx}).to(torch::kBool);
    switch (expert_type) {
      case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
      case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
      case NLLB_MOE_DENSE_ACT_DENSE:
      case FSGPT_MOE_DENSE_ACT_DENSE:
      case MIXTRAL_MOE_DENSE_ACT_DENSE:
      case DEEPSEEK_MOE_DENSE_ACT_DENSE:
        input =
            hidden_states_.index({token_indices}).to(expert_node->node->device);
        break;
      default:
        DLOG_FATAL("ExpertDispatcher::expert_type: unknown expert type ",
                   expert_type);
    }

    DLOG_TRACE("ExpertDispatcher::GPUFetchFunc gpu_id ", gpu_id, "layer_idx ",
               layer_idx, "expert_idx ", expert_idx, "input ",
               input.device().str(), "node ", expert_node->node->device.str());
    {
      ExecArgs exec_args;
      exec_args.hidden_states = std::move(input);
      exec_args.expert_node = expert_node;
      exec_args.out_gpu_id = original_device.index();
      exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
      exec_args.evict = !success;
      exec_args.hit = cache_hit;
      std::lock_guard<std::mutex> lock(exec_mutex_[gpu_id]);
      exec_queue_[gpu_id].emplace_back(std::move(exec_args));
    }
    exec_cv_[gpu_id].notify_all();
  }
}

void ExpertDispatcher::GPUExecFunc(int gpu_id) {
  cudaSetDevice(gpu_id);
  while (!main_thread_stop_flag_.load()) {
    std::unique_lock<std::mutex> lock(exec_mutex_[gpu_id]);
    exec_cv_[gpu_id].wait(lock, [&] { return !exec_queue_[gpu_id].empty(); });

    ExecArgs args = std::move(exec_queue_[gpu_id].front());
    exec_queue_[gpu_id].pop_front();

    // if (exec_queue_.empty()) {
    //   // cvs_[MUTEX_TYPE::EXEC_MUTEX].wait(lock, [&] { return
    //   // !exec_queue_.empty(); });
    //   lock.unlock();
    //   std::this_thread::sleep_for(std::chrono::microseconds(10));
    //   continue;
    // }

    // ExecArgs args;

    // for (auto it = exec_queue_.begin(); it != exec_queue_.end(); ++it) {
    //   if (it->expert_node->node->device.index() == gpu_id) {
    //     args = std::move(*it);
    //     exec_queue_.erase(it);
    //     break;
    //   }
    // }

    lock.unlock();

    if (args.expert_node == nullptr) {
      continue;
    }

    torch::Tensor output;

    // at::InferenceMode infer_guard(true);

    // random int [0,8)
    int rnd = std::rand() % 8;
    c10::cuda::CUDAStream stream =
        c10::cuda::getStreamFromExternal(exec_streams_[gpu_id + rnd], gpu_id);

    {
      auto start = TIME_NOW;
      // c10::cuda::CUDAStreamGuard guard(stream);

      auto* expert_module = args.expert_node->module;
      int expert_type = expert_type_;
      cudaStreamSynchronize(stream);  // make sure the input is ready

      try {
        switch (expert_type) {
          case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
            output = reinterpret_cast<SwitchTransformersDenseActDense*>(
                         expert_module)
                         ->forward(args.hidden_states);
            break;
          case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
            output = reinterpret_cast<SwitchTransformersDenseGatedActDense*>(
                         expert_module)
                         ->forward(args.hidden_states);
            break;
          case NLLB_MOE_DENSE_ACT_DENSE:
            output = reinterpret_cast<NllbMoeDenseActDense*>(expert_module)
                         ->forward(args.hidden_states);
            break;
          case FSGPT_MOE_DENSE_ACT_DENSE:
            output = reinterpret_cast<FSGPTMoEDenseActDense*>(expert_module)
                         ->forward(args.hidden_states);
            break;
          case MIXTRAL_MOE_DENSE_ACT_DENSE:
            output = reinterpret_cast<MixtralMoEDenseActDense*>(expert_module)
                         ->forward(args.hidden_states);
            break;
          case DEEPSEEK_MOE_DENSE_ACT_DENSE:
            output = reinterpret_cast<DeepSeekMoEDenseActDense*>(expert_module)
                         ->forward(args.hidden_states);
            break;
          default:
            DLOG_FATAL("ExpertDispatcher::GPUExecFunc: unknown expert type",
                       expert_type);
        }

      } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "DenseActDense tensor_ids: [";
        for (auto& id : args.expert_node->node->tensor_ids) {
          ss << id << " ";
        }
        ss << "]";
        DLOG_FATAL("ExpertDispatcher::GPUExecFunc", ss.str(), "expert_type",
                   expert_type, e.what());
      }

      stream.synchronize();
      auto end = TIME_NOW;
      // DLOG_INFO("ExpertDispatcher::GPUExecFunc: forward time ",
      //                  std::chrono::duration_cast<MCIROSECONDS>(end -
      //                  start).count(), "us");
    }

    (void)std::async(std::launch::async, &ExpertDispatcher::OutputFunc, this,
                     std::move(args), std::move(output), gpu_id);
  }
}

void ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output,
                                  int gpu_id) {
  // c10::cuda::CUDAStream stream =
  // c10::cuda::getStreamFromExternal(out_streams_[gpu_id], gpu_id);
  // c10::cuda::CUDAStreamGuard guard(stream);

  auto output_device =
      (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
  torch::Tensor output_tensor = output.to(output_device).to(args.out_dtype);

  if (args.evict) {
    args.expert_node->node->SetDevice(args.expert_node->node->default_host,
                                      true, nullptr);
    {
      std::lock_guard<std::mutex> lock(gpu_overload_mutex_);
      gpu_overload_[gpu_id] = false;
    }
  }

  args.expert_node->node->mutex.unlock();

  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue_.emplace_back(std::move(output_tensor),
                               args.expert_node->layer_idx,
                               args.expert_node->expert_idx, args.hit);
    DLOG_TRACE("ExpertDispatcher::OutputFunc: output_queue_",
               output_queue_.size(), "output",
               std::get<0>(output_queue_.back()).device().str(), "evict",
               args.evict, "(", args.expert_node->layer_idx,
               args.expert_node->expert_idx, gpu_id, args.hit, ")");
  }
  // stream.synchronize();
  pending_.fetch_sub(1);
  if (pending_.load() == 0) {
    pending_cv_.notify_all();
  }
}

std::vector<ExpertDispatcher::CallResult> ExpertDispatcher::Wait() {
  int wait_count = 0;

  std::unique_lock<std::mutex> lock(pending_mutex_);
  pending_cv_.wait(lock, [&] { return pending_.load() == 0; });

  // while (pending_.load() > 0) {
  //   std::unique_lock<std::mutex> lock(pending_mutex_);
  //   std::this_thread::sleep_for(std::chrono::microseconds(10));
  //   wait_count++;
  //   if (wait_count % 1000 == 0) {
  //     DLOG_WARN("ExpertDispatcher::Wait: wait_count:", wait_count,
  //               "pending_: ", pending_.load(),
  //               "num_enqueued: ", num_enqueued_.load(),
  //               "input_queue_: ", input_queue_.size(),
  //               "exec_queue_: ", exec_queue_.size());
  //   }
  // }
  // input_queue_.clear();
  // exec_queue_.clear();
  num_enqueued_.store(0);
  std::vector<CallResult> output_queue;
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue.swap(output_queue_);
  }

  // std::stringstream ss;
  // ss << "ExpertDispatcher::Wait: output_queue_ " << output_queue.size();
  // for (auto& output : output_queue) {
  //     ss << " (" << std::get<0>(output).sizes() << "," << std::get<1>(output)
  //     << ","
  //        << std::get<2>(output) << "," << std::get<3>(output) << ")";
  // }
  // DLOG_TRACE(ss.str());
  return output_queue;
}
