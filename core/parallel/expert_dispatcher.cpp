// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "prefetch/task_scheduler.h"
#include "prefetch/task_thread.h"
#include "utils/archer_logger.h"
#include "utils/cuda_utils.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <future>

ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype, int expert_type)
    : pending_(0), num_enqueued_(0), start_(false), expert_type_(expert_type)
{
    main_thread_stop_flag_.store(false);

    int num_gpu = GetDeviceCount();
    for (int i = 0; i < num_gpu; ++i) {
        std::thread t(&ExpertDispatcher::GPUFetchFunc, this, i);
        SetThreadAffinity(t);
        threads_.emplace_back(std::move(t));

        cudaSetDevice(i);
        cudaStream_t fetch_stream;
        cudaStreamCreateWithFlags(&fetch_stream, cudaStreamNonBlocking);
        fetch_streams_.emplace_back(fetch_stream);

        cudaStream_t out_stream;
        cudaStreamCreateWithFlags(&out_stream, cudaStreamNonBlocking);
        out_streams_.emplace_back(fetch_stream);
    }

    for (int i = 0; i < num_gpu; ++i) {
        std::thread t(&ExpertDispatcher::GPUThreadFunc, this, i);
        SetThreadAffinity(t);
        threads_.emplace_back(std::move(t));

        gpu_overload_.emplace_back(false);
    }

    for (int i = 0; i < num_gpu; ++i) {
        std::thread t(&ExpertDispatcher::GPUExecFunc, this, i);
        SetThreadAffinity(t);
        threads_.emplace_back(std::move(t));

        cudaSetDevice(i);
        cudaStream_t exec_stream;
        cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
        exec_streams_.emplace_back(exec_stream);
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
                    experts_[i][j]->module = new SwitchTransformersDenseGatedActDense(dtype);
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
                default:
                    ARCHER_LOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ",
                                     expert_type);
            }
            experts_[i][j]->module->eval();
            experts_[i][j]->layer_idx = j;
            experts_[i][j]->expert_idx = i;
        }
    }
}

void ExpertDispatcher::EnqueueExpert(int layer_idx, int expert_idx, int gpu_id, bool remote)
{
    ExpertDispatcher::CallArgs args;
    args.layer_idx = layer_idx;
    args.expert_idx = expert_idx;
    args.gpu_id = gpu_id;
    args.remote = remote;
    Enqueue(args);
}

void ExpertDispatcher::Enqueue(const CallArgs& args)
{
    std::lock_guard<std::mutex> lock(input_mutex_);

    int layer_idx = args.layer_idx;
    int expert_idx = args.expert_idx;
    auto expert_node = experts_[expert_idx][layer_idx];

    expert_node->node->mutex.lock();
    expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

    input_queue_.push_back(std::move(args));
    num_enqueued_.fetch_add(1);

    auto& a = input_queue_.back();
    if (expert_node->node->device.is_cuda()) { a.gpu_id = expert_node->node->device.index(); }
    ARCHER_LOG_DEBUG(
        "ExpertDispatcher::Enqueue: num_enqueued_ ", num_enqueued_.load(),
        "input_queue_ ", input_queue_.size(), \
        "gpu_id ", a.gpu_id,
        "layer_idx ", a.layer_idx,
        "expert_idx ", a.expert_idx,
        "remote ", a.remote);
}

void ExpertDispatcher::RegisterExpert(int layer_idx,
                                      int expert_idx,
                                      const std::vector<std::uint32_t>& tensor_ids)
{
    NodePtr cached_node = nullptr;
    for (auto tensor_id : tensor_ids) {
        auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
        if (cached_node == nullptr) {
            cached_node = node;
            experts_[expert_idx][layer_idx]->node = node;
        } else if (cached_node != node) {
            ARCHER_LOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
        }
    }
}

void ExpertDispatcher::GPUThreadFunc(int gpu_id)
{
    while (!main_thread_stop_flag_.load()) {}
}

void ExpertDispatcher::GPUFetchFunc(int gpu_id)
{
    while (!main_thread_stop_flag_.load()) {
        std::unique_lock<std::mutex> lock(input_mutex_);
        if (input_queue_.empty()) {
            lock.unlock();
            continue;
        }

        CallArgs args;

        // find all the args with gpu_id
        std::vector<CallArgs> args_list;
        auto it = input_queue_.begin();
        while (it != input_queue_.end()) {
            if (it->gpu_id == gpu_id) {
                args_list.emplace_back(std::move(*it));
                it = input_queue_.erase(it);
            } else {
                ++it;
            }
        }

        lock.unlock();

        if (args_list.empty()) { continue; }

        for (auto& a : args_list) {
            int layer_idx = a.layer_idx;
            int expert_idx = a.expert_idx;
            auto expert_node = experts_[expert_idx][layer_idx];

            if (expert_node->node->device.is_cuda()) {
                args = std::move(a);
                break;
            }
        }

        if (args.gpu_id == -1) { args = std::move(args_list[0]); }

        //  put back the rest
        args_list.erase(std::remove_if(args_list.begin(),
                                       args_list.end(),
                                       [&](const CallArgs& a) {
                                           return (a.expert_idx == args.expert_idx) &&
                                                  (a.layer_idx == args.layer_idx);
                                       }),
                        args_list.end());

        lock.lock();
        for (auto& a : args_list) { input_queue_.push_back(std::move(a)); }
        lock.unlock();

        auto device = CUDA_DEVICE(gpu_id);
        auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();
        int layer_idx = args.layer_idx;
        int expert_idx = args.expert_idx;

        auto expert_node = experts_[expert_idx][layer_idx];
        bool cache_hit = expert_node->node->device.is_cuda();
        bool success = true;
        if (!expert_node->node->device.is_cuda()) {
            success = kTaskPool->RemoveCachedSparseNode(expert_node->node, gpu_id);

            int wait_count = 0;
            while (!success && gpu_overload_[gpu_id]) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                wait_count++;
                // if (wait_count % 100000 == 0) {
                //     ARCHER_LOG_WARN(
                //         "ExpertDispatcher::EnqueueTask: gpu_overload_ gpu_id {} wait_count {} {}",
                //         gpu_id,
                //         wait_count,
                //         expert_node->node->str());
                // }
            }

            if (!success && !gpu_overload_[gpu_id]) {
                {
                    std::lock_guard<std::mutex> lock(gpu_overload_mutex_);
                    gpu_overload_[gpu_id] = true;
                }
            }

            auto task = std::make_shared<Task>();
            task->priority = 0;
            task->node = expert_node->node;
            task->on_demand = true;
            task->src_device = expert_node->node->device;
            task->dst_device = CUDA_DEVICE(gpu_id);
            task->remove_layer = true;
            kTaskPool->EnqueueTask(task);

            wait_count = 0;
            while (!expert_node->node->device.is_cuda()) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                wait_count++;
                if (wait_count % 100000 == 0) {
                    ARCHER_LOG_WARN("ExpertDispatcher::EnqueueTask: wait_count ",
                                    wait_count,
                                    expert_node->node->str());
                }
            }

            // ARCHER_LOG_DEBUG(
            //     "ExpertDispatcher::GPUFetchFunc: move to device gpu_id {} layer_idx {} expert_idx
            //     {} node {}", gpu_id, layer_idx, expert_idx, expert_node->node->device.str());
        }
        expert_node->SetTensorsFromBlob(device);
        // auto& meta = kTensorIndex->find(expert_node->node->tensor_ids[0])->second;

        int expert_type = expert_type_;
        torch::Tensor input;
        auto token_indices = router_mask_.index({"...", expert_idx}).to(torch::kBool);
        switch (expert_type) {
            case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
            case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
            case NLLB_MOE_DENSE_ACT_DENSE:
            case FSGPT_MOE_DENSE_ACT_DENSE:
            case MIXTRAL_MOE_DENSE_ACT_DENSE:
                input = hidden_states_.index({token_indices}).to(expert_node->node->device);
                break;
            default:
                ARCHER_LOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ",
                                 expert_type);
        }

        ARCHER_LOG_DEBUG(
            "ExpertDispatcher::GPUFetchFunc gpu_id ",  gpu_id,
            "layer_idx ", layer_idx,
            "expert_idx ", expert_idx,
            "input ", input.device().str(),
            "node ", expert_node->node->device.str());
        {
            ExecArgs exec_args;
            exec_args.hidden_states = std::move(input);
            exec_args.expert_node = expert_node;
            exec_args.out_gpu_id = original_device.index();
            exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
            exec_args.evict = !success;
            exec_args.hit = cache_hit;
            std::lock_guard<std::mutex> lock(exec_mutex_);
            exec_queue_.emplace_back(std::move(exec_args));
        }
    }
}

void ExpertDispatcher::GPUExecFunc(int gpu_id)
{
    while (!main_thread_stop_flag_.load()) {
        std::unique_lock<std::mutex> lock(exec_mutex_);
        if (exec_queue_.empty()) {
            lock.unlock();
            continue;
        }

        ExecArgs args;

        for (auto it = exec_queue_.begin(); it != exec_queue_.end(); ++it) {
            if (it->expert_node->node->device.index() == gpu_id) {
                args = std::move(*it);
                exec_queue_.erase(it);
                break;
            }
        }

        lock.unlock();

        if (args.expert_node == nullptr) { continue; }

        torch::Tensor output;

        at::InferenceMode infer_guard(true);

        c10::cuda::CUDAStream stream =
            c10::cuda::getStreamFromExternal(fetch_streams_[gpu_id], gpu_id);

        {
            c10::cuda::CUDAStreamGuard guard(stream);

            auto* expert_module = args.expert_node->module;
            int expert_type = expert_type_;

            try {
                switch (expert_type) {
                    case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
                        output = reinterpret_cast<SwitchTransformersDenseActDense*>(expert_module)
                                     ->forward(args.hidden_states);
                        break;
                    case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
                        output =
                            reinterpret_cast<SwitchTransformersDenseGatedActDense*>(expert_module)
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
                    default:
                        ARCHER_LOG_FATAL(
                            "ExpertDispatcher::ExpertDispatcher: unknown expert type",
                            expert_type);
                }

            } catch (const std::exception& e) {
                std::stringstream ss;
                ss << "DenseActDense tensor_ids: [";
                for (auto& id : args.expert_node->node->tensor_ids) { ss << id << " "; }
                ss << "]";
                ARCHER_LOG_FATAL("ExpertDispatcher::GPUExecFunc", ss.str(), "expert_type", expert_type, e.what());
            }
        }

        (void)std::async(std::launch::async,
                         &ExpertDispatcher::OutputFunc,
                         this,
                         std::move(args),
                         std::move(output),
                         gpu_id);
    }
}

void ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output, int gpu_id)
{
    c10::cuda::CUDAStream stream = c10::cuda::getStreamFromExternal(out_streams_[gpu_id], gpu_id);
    c10::cuda::CUDAStreamGuard guard(stream);

    auto output_device = (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
    torch::Tensor output_tensor = output.to(output_device).to(args.out_dtype);

    if (args.evict) {
        args.expert_node->node->SetDevice(args.expert_node->node->default_host, true, nullptr);
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
                                   args.expert_node->expert_idx,
                                   args.hit);
        ARCHER_LOG_DEBUG(
            "ExpertDispatcher::OutputFunc: output_queue_", output_queue_.size(),
            "output", std::get<0>(output_queue_.back()).device().str(),
            "evict", args.evict,
            "(", 
            args.expert_node->layer_idx,
            args.expert_node->expert_idx,
            gpu_id,
            args.hit, ")");
    }
    pending_.fetch_sub(1);
}

std::vector<ExpertDispatcher::CallResult> ExpertDispatcher::Wait()
{
    int wait_count = 0;
    while (pending_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        wait_count++;
        // if (wait_count % 1000 == 0) {
        //     ARCHER_LOG_WARN(
        //         "ExpertDispatcher::Wait: wait_count {} pending_ {} num_enqueued {} input_queue_ "
        //         "{}, exec_queue_ {}",
        //         wait_count,
        //         pending_.load(),
        //         num_enqueued_.load(),
        //         input_queue_.size(),
        //         exec_queue_.size());
        // }
    }
    input_queue_.clear();
    exec_queue_.clear();
    num_enqueued_.store(0);
    std::vector<CallResult> output_queue;
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        output_queue.swap(output_queue_);
    }

    std::stringstream ss;
    ss << "ExpertDispatcher::Wait: output_queue_ " << output_queue.size();
    for (auto& output : output_queue) {
        ss << " (" << std::get<0>(output).sizes() << "," << std::get<1>(output) << ","
           << std::get<2>(output) << "," << std::get<3>(output) << ")";
    }
    ARCHER_LOG_DEBUG(ss.str());
    return output_queue;
}
