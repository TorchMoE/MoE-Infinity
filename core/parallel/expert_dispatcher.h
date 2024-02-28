// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "expert_module.h"
#include "utils/noncopyable.h"

class ExpertDispatcher : public noncopyable {
public:
    typedef struct {
        int layer_idx = -1;
        int expert_idx = -1;
        int gpu_id = -1;
        bool remote = false;
    } CallArgs;
    typedef struct {
        torch::Tensor hidden_states =
            torch::empty({0});  // shallow copy, real tensor in python code
        ExpertNodePtr expert_node = nullptr;
        int out_gpu_id = -1;
        torch::ScalarType out_dtype = torch::kFloat32;
        bool evict = false;
        bool hit = false;
    } ExecArgs;
    typedef std::tuple<torch::Tensor, int, int, int> CallResult;

public:
    explicit ExpertDispatcher(int num_experts, int num_layers, int dtype, int expert_type);
    ~ExpertDispatcher()
    {
        main_thread_stop_flag_.store(true);
        for (auto& thread : threads_) { thread.join(); }

        for (auto& stream : fetch_streams_) { cudaStreamDestroy(stream); }
        for (auto& stream : exec_streams_) { cudaStreamDestroy(stream); }
        for (auto& stream : out_streams_) { cudaStreamDestroy(stream); }
    }

    void SetInputs(const torch::Tensor& hidden_states, const torch::Tensor& router_mask)
    {
        hidden_states_ = hidden_states.clone();
        router_mask_ = router_mask.clone();
    }

    void EnqueueExpert(int layer_idx, int expert_idx, int gpu_id = -1, bool remote = false);

    void RegisterExpert(int layer_idx,
                        int expert_idx,
                        const std::vector<std::uint32_t>& tensor_ids);

    void SetExpectedQueue(int expected_pending = 0) { pending_.store(expected_pending); }

    std::vector<CallResult> WaitExpert() { return Wait(); }
    void SetNode(int layer_idx, int expert_idx, const NodePtr& node)
    {
        experts_[expert_idx][layer_idx]->node = node;
    }

private:
    void Enqueue(const CallArgs& args);
    std::vector<CallResult> Wait();
    void Start() { start_ = true; }

    void GPUFetchFunc(int gpu_id);
    void GPUExecFunc(int gpu_id);

    void GPUThreadFunc(int gpu_id);

    void OutputFunc(ExecArgs args, torch::Tensor output, int gpu_id);

private:
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::vector<CallArgs> input_queue_;
    std::vector<ExecArgs> exec_queue_;
    std::vector<CallResult> output_queue_;
    std::vector<std::vector<ExpertNodePtr>> experts_;
    std::atomic<size_t> pending_;
    std::atomic<size_t> num_enqueued_;
    bool start_;
    int expert_type_;
    std::atomic<bool> main_thread_stop_flag_;

    std::mutex input_mutex_;
    std::mutex output_mutex_;
    std::mutex exec_mutex_;
    std::mutex gpu_overload_mutex_;

    std::vector<cudaStream_t> fetch_streams_;
    std::vector<cudaStream_t> exec_streams_;
    std::vector<cudaStream_t> out_streams_;

    std::vector<bool> gpu_overload_;

    torch::Tensor hidden_states_;
    torch::Tensor router_mask_;
};
