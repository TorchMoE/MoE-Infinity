// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef NVTX_DISABLE
  #include <nvtx3/nvtx3.hpp>
#endif

#include "common/sync.h"
#include "base/noncopyable.h"
#include "base/thread.h"
#include "utils/threadsafe_queue.h"
#include "memory/event_pool.h"
#include "expert_module.h"

enum MUTEX_TYPE {
  INPUT_MUTEX = 0,
  OUTPUT_MUTEX = 1,
  EXEC_MUTEX = 2,
  PENDING_MUTEX = 3
};

struct CUevent_st;
using cudaEvent_t = CUevent_st*;

class ExpertDispatcher : public base::noncopyable {
 public:
  typedef struct {
    int layer_idx = -1;
    int expert_idx = -1;
    int gpu_id = -1;
    bool remote = false;
    bool wait_for_prefetch = false;
    // Stamp the creating generation so late workers cannot underflow a failed
    // generation's pending count.
    std::uint64_t generation = 0;
    bool cache_slot_reserved = false;
    bool cache_key_inserted = false;
  } CallArgs;
  typedef struct {
    torch::Tensor hidden_states =
        torch::empty({0});  // shallow copy, real tensor in python code
    ExpertNodePtr expert_node = nullptr;
    int out_gpu_id = -1;
    torch::ScalarType out_dtype = torch::kFloat32;
    bool evict = false;
    bool hit = false;
    cudaEvent_t transfer_event = nullptr;
    std::uint64_t generation = 0;
    bool cache_slot_reserved = false;
    bool cache_key_inserted = false;
  } ExecArgs;
  typedef std::tuple<torch::Tensor, int, int, int> CallResult;

  typedef struct {
    int layer_idx = -1;
    torch::Tensor active_flags_host;
    std::uint64_t generation = 0;
  } RouteArgs;

  typedef struct {
    ExpertDispatcher* dispatcher = nullptr;
    RouteArgs route;
  } RouteCallbackArgs;

  enum class DispatchFaultPoint : int {
    NONE = 0,
    ROUTE_CALLBACK = 1,
    ROUTE_WORKER = 2,
    FETCH_WORKER = 3,
    EXEC_WORKER = 4,
    OUTPUT = 5,
    COMPLETION_EVENT_RECORD = 6,
    RETIREMENT_CALLBACK_LAUNCH = 7,
    SUBMISSION = 8,
  };

  typedef struct {
    ExpertNodePtr expert_node = nullptr;
    int gpu_id = -1;
    bool cache_slot_reserved = false;
    bool cache_key_inserted = false;
    bool overload_owned = false;
  } FailureContext;

  typedef struct {
    ExpertDispatcher* dispatcher = nullptr;
    ExpertNodePtr expert_node = nullptr;
    std::uint64_t generation = 0;
    int gpu_id = -1;
    bool evict = false;
  } ExpertRetireArgs;

  struct CompletionEventRecord {
    cudaEvent_t event = nullptr;
    std::uint64_t generation = 0;
    int device = 0;
  };
  struct CompletionRetireBatch {
    ExpertDispatcher* dispatcher = nullptr;
    std::vector<CompletionEventRecord> records;
  };
  struct CompletionRetireItem {
    CompletionEventRecord record;
    bool caller_wait_consumed = false;
  };

 public:
  explicit ExpertDispatcher(int num_experts, int num_layers, int dtype,
                            int expert_type, int num_threads = 1);
  ~ExpertDispatcher() {
    // Let any in-flight dispatch finish routing and expert execution before
    // draining callbacks: OutputFunc launches retirement/completion callbacks,
    // so waiting for the pipeline to quiesce prevents a callback from being
    // launched against a half-destroyed dispatcher.
    {
      std::unique_lock<std::mutex> lock(pending_mutex_);
      pending_cv_.wait(lock, [&] {
        return !route_pending_.load(std::memory_order_acquire) &&
               pending_.load(std::memory_order_acquire) == 0;
      });
    }
    {
      std::unique_lock<std::mutex> lock(route_callback_mutex_);
      route_callback_cv_.wait(lock, [&] {
        return pending_route_callbacks_.load(std::memory_order_acquire) == 0;
      });
    }
    {
      std::unique_lock<std::mutex> lock(completion_retirement_callback_mutex_);
      completion_retirement_callback_cv_.wait(lock, [&] {
        return pending_completion_retirement_callbacks_.load(
                   std::memory_order_acquire) == 0;
      });
    }
    {
      std::unique_lock<std::mutex> lock(retirement_callback_mutex_);
      retirement_callback_cv_.wait(lock, [&] {
        return pending_retirement_callbacks_.load(std::memory_order_acquire) ==
               0;
      });
    }
    main_thread_stop_flag_.store(true, std::memory_order_release);
    for (auto& expert_list : experts_) {
      for (auto& expert_node : expert_list) {
        if (expert_node && expert_node->node) {
          expert_node->node->exec_state.store(NodeExecState::IDLE,
                                              std::memory_order_release);
        }
      }
    }
    for (int i = 0; i < static_cast<int>(input_queue_.size()); ++i) {
      input_queue_[i].Close();
    }
    for (int i = 0; i < static_cast<int>(exec_queue_.size()); ++i) {
      exec_queue_[i].Close();
    }
    route_queue_.Close();
    completion_retirement_queue_.Close();
    retirement_queue_.Close();
    for (int i = 0; i < static_cast<int>(cache_cv_.size()); ++i) {
      cache_cv_[i].notify_all();
    }
    pending_cv_.notify_all();
    for (auto& t : threads_) {
      if (t) {
        t->join();
      }
    }
    // Destructor-only synchronization: drain every device so no in-flight
    // kernel, stream callback, or completion event still references the
    // modules, streams, or events destroyed below.
    for (int device = 0; device < kNumDevices(); ++device) {
      cudaSetDevice(device);
      cudaDeviceSynchronize();
    }
    for (auto& record : completion_events_) {
      if (record.event != nullptr) {
        cudaEventSynchronize(record.event);
        cudaEventDestroy(record.event);
        completion_events_retired_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    for (auto& record : destructor_fallback_events_) {
      if (record.event != nullptr) {
        cudaEventSynchronize(record.event);
        cudaEventDestroy(record.event);
        completion_events_retired_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    for (auto& completion_pool : completion_event_pool_) {
      for (auto& event : completion_pool.second) {
        cudaEventDestroy(event);
      }
    }
    for (auto& stream : exec_streams_) {
      cudaStreamDestroy(stream);
    }
    for (auto& stream : fetch_streams_) {
      cudaStreamDestroy(stream);
    }
    for (auto& event : exec_done_events_) {
      cudaEventDestroy(event);
    }
    if (input_ready_event_ != nullptr) {
      cudaEventDestroy(input_ready_event_);
    }
    for (auto* m : modules_) {
      delete m;
    }
  }

  void SetInputs(const torch::Tensor& hidden_states,
                 const torch::Tensor& router_mask,
                 const torch::Tensor& router_weight);

  void EnqueueExpert(int layer_idx, int expert_idx, int gpu_id = -1,
                     bool remote = false);
  void NotifyFetchStart();

  void RegisterExpert(int layer_idx, int expert_idx,
                      const std::vector<std::uint32_t>& tensor_ids,
                      std::string jit_path);
  void ClearExpertCacheCounts();
  // Read-only observability accessors; neither alters routing/dispatch.
  std::int64_t GetCacheOccupancyBytes();
  double GetCacheHitRate() const;
  void SetExpectedQueue(int expected_pending = 0) {
    pending_.store(expected_pending);
  }

  void SetScales(const std::map<std::string, torch::Tensor>& scales);

  std::vector<CallResult> WaitExpert() { return Wait(); }
  torch::Tensor WaitHiddenStates();

  void DispatchExperts(int layer_idx);
  std::vector<int> TakeLastActiveExperts();
  std::map<std::string, std::int64_t> GetRoutingStats() const;
  void SetDispatchFaultForTest(const std::string& stage);
  void FailDispatchForTest(std::uint64_t generation,
                           const std::string& message);

 private:
  void Enqueue(CallArgs& args);
  std::vector<CallResult> Wait();
  void Start() { start_ = true; }

  void GPUFetchFunc(int gpu_id);
  void GPUExecFunc(int gpu_id, int thread_idx);

  // void GPUThreadFunc(int gpu_id);

  bool OutputFunc(ExecArgs args, torch::Tensor output, torch::Tensor token_mask,
                  int gpu_id, cudaStream_t exec_stream) noexcept;

  ExpertNodePtr FindExpertEvict(int gpu_id);

  void RouteFunc() noexcept;
  static void CUDART_CB RouteReadyCallback(void* opaque);
  void FailDispatch(const std::uint64_t failing_generation,
                    std::exception_ptr error,
                    const std::vector<FailureContext>& contexts = {}) noexcept;
  void CompleteOne(std::uint64_t generation) noexcept;

  static void CUDART_CB CompletionWaitsConsumedCallback(void* opaque);
  void CompletionRetirementFunc();
  void QueueUnwaitedEventsForQuery(
      std::vector<CompletionEventRecord> records) noexcept;

  static void CUDART_CB ExpertReadyToRetireCallback(void* opaque);
  void RetirementFunc();

  class DispatchSubmissionGuard {
   public:
    DispatchSubmissionGuard(ExpertDispatcher* dispatcher,
                            const std::uint64_t generation)
        : dispatcher_(dispatcher), generation_(generation) {}
    ~DispatchSubmissionGuard() {
      if (armed_) {
        dispatcher_->FailDispatch(
            generation_, error_ ? error_
                                : std::make_exception_ptr(std::runtime_error(
                                      "dispatch submission failed")));
      }
    }
    void Capture(std::exception_ptr error) noexcept {
      error_ = std::move(error);
    }
    void Release() noexcept { armed_ = false; }

   private:
    ExpertDispatcher* dispatcher_;
    const std::uint64_t generation_;
    std::exception_ptr error_;
    bool armed_ = true;
  };

 private:
  std::vector<std::unique_ptr<base::Thread>> threads_;
  std::mutex mutex_;
  // std::vector<std::deque<CallArgs>> input_queue_;
  std::vector<ThreadSafeQueue<CallArgs>> input_queue_;
  // std::vector<std::deque<ExecArgs>> exec_queue_;
  std::vector<ThreadSafeQueue<ExecArgs>> exec_queue_;
  std::vector<CallResult> output_queue_;
  std::vector<std::vector<ExpertNodePtr>> experts_;
  std::atomic<size_t> num_enqueued_;
  bool start_;
  int expert_type_;
  int dtype_;
  int num_experts_;
  std::atomic<bool> main_thread_stop_flag_;

  std::atomic<size_t> pending_;

  // Passive counters for GetCacheHitRate(); reset by ClearExpertCacheCounts().
  std::atomic<std::uint64_t> cache_hit_count_{0};
  std::atomic<std::uint64_t> cache_access_count_{0};

  std::mutex pending_mutex_;
  std::condition_variable pending_cv_;

  // std::vector<std::mutex> input_mutex_;
  // std::vector<std::mutex> exec_mutex_;
  // std::vector<std::condition_variable> input_cv_;
  // std::vector<std::condition_variable> exec_cv_;

  std::vector<std::mutex> cache_mutex_;
  std::vector<std::condition_variable> cache_cv_;

  std::mutex output_mutex_;
  std::mutex accum_mutex_;

  void SyncExecStreamsWithCurrent();

  std::vector<cudaStream_t> exec_streams_;
  std::vector<cudaEvent_t> exec_done_events_;
  cudaEvent_t input_ready_event_ = nullptr;
  std::vector<cudaStream_t> fetch_streams_;

  std::unique_ptr<std::atomic<bool>[]> gpu_overload_;

  torch::Tensor hidden_states_;
  torch::Tensor final_hidden_states_;
  torch::Tensor router_mask_;
  torch::Tensor router_weight_;

  std::vector<int64_t> cache_sizes_;
  std::vector<std::unordered_set<uint64_t>> cached_experts_;

  int cache_capacity_ = 0;
  int num_threads_ = 1;

  std::vector<MoEMLP*> modules_;

  bool fp8_in_store_ = false;
  std::vector<std::vector<std::vector<torch::Tensor>>> fp8_scales_;

  ThreadSafeQueue<RouteArgs> route_queue_;
  std::atomic<bool> route_pending_{false};
  std::atomic<std::uint64_t> dispatch_generation_{0};
  std::atomic<std::int64_t> pending_route_callbacks_{0};
  std::mutex route_callback_mutex_;
  std::condition_variable route_callback_cv_;
  mutable std::mutex route_state_mutex_;
  std::exception_ptr route_error_;
  std::atomic<std::uint64_t> current_generation_{0};
  std::atomic<std::uint64_t> failed_generation_{0};
  std::atomic<int> dispatch_fault_for_test_{0};
  std::vector<int> last_active_experts_;
  std::vector<CompletionEventRecord> completion_events_;
  std::vector<CompletionEventRecord> destructor_fallback_events_;
  ThreadSafeQueue<std::vector<CompletionRetireItem>>
      completion_retirement_queue_;
  std::atomic<std::int64_t> pending_completion_retirement_callbacks_{0};
  std::mutex completion_retirement_callback_mutex_;
  std::condition_variable completion_retirement_callback_cv_;

  ThreadSafeQueue<ExpertRetireArgs> retirement_queue_;
  std::atomic<std::int64_t> pending_retirement_callbacks_{0};
  std::mutex retirement_callback_mutex_;
  std::condition_variable retirement_callback_cv_;

  cudaEvent_t AcquireCompletionEvent(int device);
  void ReleaseCompletionEvent(cudaEvent_t event, int device) noexcept;
  std::mutex completion_event_pool_mutex_;
  std::unordered_map<int, std::vector<cudaEvent_t>> completion_event_pool_;

  std::atomic<std::int64_t> route_batches_{0};
  std::atomic<std::int64_t> route_failures_{0};
  std::atomic<std::int64_t> last_active_experts_count_{0};
  std::atomic<std::int64_t> last_route_handoff_us_{0};
  std::atomic<std::int64_t> completion_events_retired_{0};
  std::atomic<std::int64_t> completion_events_outstanding_{0};
  std::atomic<std::int64_t> stale_failures_quarantined_{0};
};

#define SET_TENSORS_AND_MODULE_FROM_BLOB(cls, module, node, device, \
                                         jit_module)                \
  do {                                                              \
    reinterpret_cast<cls*>(module)->SetTensorsFromBlob(             \
        node->device_memory_ptr, node->tensor_ids, device);         \
    reinterpret_cast<cls*>(module)->SetModuleFromBlob(jit_module);  \
  } while (0)
