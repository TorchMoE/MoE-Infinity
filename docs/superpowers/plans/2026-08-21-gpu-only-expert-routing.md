# GPU-Only Expert Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the blocking GPU-to-CPU active-expert extraction from single-host local MoE dispatch and replace the output-side GPU completion barrier with a CUDA-stream dependency, while preserving the exact routed expert set, expert order, weights, and accumulated output.

**Architecture:** Keep every model adapter's `dispatch_local(...)` / `wait_dispatch_local()` contract unchanged. When an opt-in flag is enabled and the router mask is CUDA-resident, `ExpertDispatcher` reduces the mask on the caller's CUDA stream, asynchronously copies only the per-expert activity bitmap into pinned host memory, and lets a dedicated native route thread enqueue the same ascending expert IDs that Python enqueues today; the eager Python path remains the fallback. Expert workers record completion events after output accumulation, and `WaitHiddenStates()` inserts waits for those events into the caller's current stream rather than synchronizing the GPU; the short CPU wait is only for routing/output launch handoff. This is a single-process, single-host design and does not alter the existing RPC/multi-node `dispatch()` path.

**Tech Stack:** Python 3.10+, PyTorch/ATen CUDA streams, pybind11, C++17, CUDA runtime events and pinned memory, pytest, NVTX, Nsight Systems, `IOProfiler`.

---

## Scope, invariants, and current evidence

This plan is grounded in `origin/main@b766f8f` and the active call chain:

1. `moe_infinity/models/{qwen.py:72-79,mixtral.py:128-135,deepseek.py:148-155}` constructs a `[tokens, experts]` mask, calls `DistributedExpertExecutor.dispatch_local()`, then immediately consumes `wait_dispatch_local()`.
2. `moe_infinity/distributed/expert_executor.py:205-215` computes the active set with `sum(...).cpu().numpy()` and therefore synchronizes the caller's CUDA stream before any expert is enqueued.
3. `expert_executor.py:218-243` calls `set_inputs`, `set_expected_queue`, one `enqueue_expert` per ascending ID, and `notify_fetch_start`.
4. `core/python/py_archer_prefetch.cpp:109-123` binds that eager interface to `ExpertDispatcher`.
5. `core/parallel/expert_dispatcher.cpp:180-266` maps each expert to a fetch/exec queue; `GPUFetchFunc()` and `GPUExecFunc()` already own nonblocking streams and use transfer events at lines 553-583 and 602-605.
6. `expert_dispatcher.cpp:672-717` accumulates weighted output and decrements `pending_`; `WaitHiddenStates()` at lines 735-742 waits on a CPU condition variable but does not explicitly establish a dependency from all worker streams to the caller's PyTorch stream.
7. `expert_executor.py:270-280` exposes that launch/completion handoff as the `expert_wait_barrier` / `sync_wait` region.
8. `core/parallel/expert_module.cpp:202-229` performs two unconditional `cudaStreamSynchronize(stream)` calls around every expert forward. These synchronizations make the worker-side “completion handoff” wait for GPU execution and must be removed only after all input creation is moved under the non-default execution-stream guard and expert/cache lifetime is retired asynchronously.
9. `core/utils/cuda_utils.h:44-81` exposes a blocking `GpuTimer::elapsed_millis()` (`cudaEventSynchronize`), while `cuda_utils.h:95-96` / `cuda_utils.cpp:60-65` expose `BlockingCudaCopy()`. Neither helper is currently called by `ExpertDispatcher`; the implementation must preserve that isolation and make the timer's blocking contract explicit so timing events are never confused with dispatch completion events.

The implementation must preserve these invariants:

- Active expert IDs are exactly `torch.where(router_mask.bool().any(dim=0))[0]`, in ascending expert-ID order.
- GPU ownership remains `expert_id % torch.cuda.device_count()` / `expert_id % kNumDevices()`.
- `router_mask`, `router_weights`, token selection, expert math, float32 accumulation, and adapter return dtypes are unchanged.
- Empty masks return a zero float32 output with the same shape as `hidden_states` and do not deadlock.
- A dispatch is single-flight: the next layer may begin only after `wait_dispatch_local()` has closed the previous generation. This matches all current adapters.
- The old eager methods remain bound and tested. Disabling the flag, using CPU tensors, loading an older extension without the new method, or entering route-ahead mode selects eager behavior before native submission. Once `dispatch_experts()` is called, synchronous submission errors propagate immediately after RAII cleanup and asynchronous errors rethrow from `wait_expert()`; neither is replayed eagerly because partial native state may exist.
- `DistributedExpertExecutor.dispatch()` (RPC/multi-node code at lines 293-369) is out of scope and unchanged. No result in this work may be presented as multi-node support.
- DeepEP is design motivation only: its asynchronous dispatch/combine APIs and explicit event handoff demonstrate the value of stream-ordered dependencies ([DeepEP](https://github.com/deepseek-ai/DeepEP)); no DeepEP code, API compatibility, hardware claim, or published speedup is imported into this work.

## File map

**Runtime and configuration**

- Modify `moe_infinity/utils/config.py`: add the opt-in `gpu_only_expert_routing` rollout flag.
- Modify `moe_infinity/distributed/expert_executor.py`: select native versus eager routing, preserve prefetch correction, and emit Python-visible profiler stages.
- Do not modify model adapters. `qwen.py`, `mixtral.py`, `deepseek.py`, `deepseek_v2_wrapper.py`, `deepseek_v3_wrapper.py`, `qwen3_5_moe.py`, `glm_moe_dsa.py`, `gpt_oss.py`, and `deepseek_v4/sync_moe_block.py` retain their current API and routing math.

**Native dispatcher**

- Modify `core/parallel/expert_dispatcher.h`: define generation-stamped route/worker arguments, `FailureContext`, `DispatchSubmissionGuard`, `std::exception_ptr` failure state, completion-event ownership, and read-only diagnostics.
- Modify `core/parallel/expert_dispatcher.cpp`: implement GPU activity reduction, asynchronous bitmap handoff, native enqueue, common route/fetch/exec/output failure closure, stream completion handoff, event/callback RAII, cache/node restoration, and validation.
- Modify `core/parallel/expert_module.cpp:185-247`: remove per-expert stream synchronizations after stream ownership and expert/cache retirement are made safe.
- Modify `core/utils/cuda_utils.h:44-81`: rename/document the blocking timing accessor and prohibit it from dispatch lifecycle code.
- Modify `core/python/py_archer_prefetch.cpp`: bind the new methods without removing eager bindings.

**Tests and validation**

- Modify `tests/python/unit/test_utils_config.py`: test flag defaults and JSON parsing.
- Create `tests/python/unit/test_gpu_only_expert_routing.py`: CPU-only orchestration, fallback, ordering, empty-mask, prefetch-correction, and adapter-contract tests with fakes.
- Modify `tests/python/unit/test_gpt_oss_mxfp4_dispatch.py`: CUDA parity, no caller-stream synchronization, empty-mask, and multi-GPU ownership tests using its existing real native-extension fixture.
- Create `tests/python/unit/test_gpu_routing_source_contract.py`: CPU-only source guards for route-state closure, forbidden hot-path synchronizers, explicit blocking timing APIs, and CLI/schema wiring.
- Create `tests/python/unit/test_gpu_routing_documentation.py`: executable documentation-content checks for configuration, observability, benchmark acceptance, scope, and rollback guidance.
- Modify `benchmarks/serving/validate_batched_dispatch.py`: source-contract checks for the new path and preserved adapter boundary.

**Observability, benchmarks, and docs**

- Modify `benchmarks/expert_io_microbench/bench_routing.py`: summarize `gpu_route_submit`, `gpu_route_fallback`, and native route statistics.
- Modify `benchmarks/expert_io_microbench/bench_bubble.py`: split route-handoff and expert-completion-handoff CPU time from total `sync_wait`.
- Modify `benchmarks/expert_io_microbench/nsys_parser.py`: report the new NVTX ranges and GPU-to-CPU memcpy count/bytes.
- Modify `benchmarks/expert_io_microbench/run_decision_profile.py:24-38,65-92,109-178`: wire routing mode, warmup iterations, schema, and native statistics into the exact Nsight command.
- Modify `benchmarks/serving/latency.py`: expose opt-in A/B mode and report TPOT p50/p99 explicitly.
- Create `tests/python/unit/test_gpu_routing_benchmarks.py`: CPU-only metric/parser and rollback-verdict tests.
- Modify `docs/configuration.md`, `docs/environment-variables.md`, `docs/benchmarking.md`, and `benchmarks/expert_io_microbench/README.md`: document scope, metrics, commands, and rollback.

## Native interface and ownership contract

Define, bind, and test these exact methods; do not expose an undefined “async handle” abstraction:

```cpp
// core/parallel/expert_dispatcher.h
void DispatchExperts(int layer_idx);
std::vector<int> TakeLastActiveExperts();
std::map<std::string, std::int64_t> GetRoutingStats() const;
```

`DispatchExperts(layer_idx)` requires CUDA `router_mask_`, computes `router_mask_.to(torch::kBool).any(0)` on the current PyTorch stream, copies its `num_experts_` boolean flags asynchronously into a dispatcher-owned pinned CPU tensor, and appends `cudaLaunchHostFunc` to that stream. The host callback performs no CUDA operation; after the copy is stream-complete it only transfers the owning `RouteArgs` into `route_queue_`. `RouteFunc()` is the sole consumer: it scans the ready pinned flags from expert `0` through `num_experts_ - 1`, sets `pending_` before enqueueing, uses `expert_id % kNumDevices()`, records the active list, notifies fetch queues, closes routing handoff, and wakes waiters. Thus no model-thread or route-thread CUDA synchronization is introduced, and no Python-visible `.cpu()`, `.numpy()`, `.item()`, or `.tolist()` is used on the enabled hot path.

`WaitHiddenStates()` waits only until routing has closed and every expert worker has enqueued its output accumulation plus a completion event. It then calls `cudaStreamWaitEvent()` for each completion event on `c10::cuda::getCurrentCUDAStream(final_hidden_states_.device().index())` and returns immediately; it never calls `cudaEventSynchronize`, `cudaStreamSynchronize`, `cudaDeviceSynchronize`, or `Tensor::cpu`. It moves the generation's event handles out of `completion_events_` before inspecting/rethrowing the generation error, so a retirement-callback-launch failure cannot strand an already-recorded event. A host callback placed after the caller-stream waits transfers those handles to a CPU completion-retirement queue; its worker returns every handle to `kCudaEventPool`. This retirement is autonomous and does not depend on a later dispatch. If insertion of a caller wait or its retirement callback fails, records with no inserted wait go directly to the query-based retirement queue, records whose waits may still reference them remain in the destructor fallback list, and `FailDispatch()` is called only after `route_state_mutex_` has been released. Destruction is the only path permitted to synchronize fallback events before releasing them.

`route_error_` is the generation's common failure slot despite its legacy name. Its type is always `std::exception_ptr`; `FailDispatch(const std::uint64_t failing_generation, ...)` may assign only an exception pointer, `SetInputs()` may reset it to `nullptr`, and `WaitHiddenStates()` swaps and calls `std::rethrow_exception`. Every asynchronous argument carries the immutable generation that created it and every catch passes that value to `FailDispatch`; `FailDispatch` must never infer the failing generation by loading `current_generation_`. A failure whose generation is not current is quarantined: it increments `stale_failures_quarantined`, performs no mutation of the current generation's error, active list, pending count, route flag, cache/node/overload state, or notifications, and returns. No worker logs-and-continues, no string is assigned to the slot, and no failure path decrements `pending_` directly.

`TakeLastActiveExperts()` returns the already-host-resident ascending vector after `WaitHiddenStates()`. It exists only so the existing post-dispatch `correct_prefetch(layer_id + 1, expert_list)` behavior remains exact; it must not query CUDA. `GetRoutingStats()` returns nonnegative `route_batches`, `route_failures`, `last_active_experts`, `last_route_handoff_us`, `completion_events_retired`, `completion_events_outstanding`, `stale_failures_quarantined`, `current_generation`, `pending`, and `route_pending`; the final five values are lifecycle diagnostics used by fault tests.

The route-ahead seam in `_maybe_route_ahead_prefetch()` currently calls `union_experts_from_mask(...).tolist()` before enqueue so it can pin the exact union. Until overlap-aware prefetch accepts the native host-ready vector or a GPU tensor, an active route-ahead context deliberately uses eager routing. This dependency is explicit, but this plan is independently implementable because ordinary execution and speculative prefetch correction can consume `TakeLastActiveExperts()` after dispatch. A later overlap-aware-prefetch plan may remove that guarded fallback without changing this interface.

**First-release independence and conflict rule:** GPU-only routing ships only with overlap-prefetch disabled. `ArcherConfig.__post_init__()` rejects `gpu_only_expert_routing=true` when the current `speculative_prefetch_overlap` boolean is true or when a later overlap plan contributes `overlap_prefetch_mode` with value `"observe"` or `"enforce"`. This is a hard configuration error before engine construction, not an eager fallback. GPU routing remains independently usable with ordinary deferred speculative prefetch and with all overlap fields off.

**Later reconciliation seam:** a follow-up jointly owned by both plans may relax the rejection only after it defines one generation-scoped active-list contract, proves whether overlap may consume the pinned host list before `WaitHiddenStates()`, and unifies route/completion/retirement failure ownership. Until that change lands, do not benchmark or document simultaneous enablement. DFlash route-ahead remains separate and intentionally falls back to eager routing until its pin-before-enqueue interface accepts GPU/native metadata.

The active list has three lifecycle points: GPU flags exist after `any(0)` is queued; pinned flags become CPU-readable only when `RouteReadyCallback` runs; the sorted `last_active_experts_` vector becomes immutable for Python immediately before `RouteFunc` clears `route_pending_`. `TakeLastActiveExperts()` is legal only after `WaitHiddenStates()` observes `route_pending_ == false`. It never waits on CUDA and never owns a CUDA event.

Event classes are deliberately separate:

- **Dispatch completion events:** acquired from `kCudaEventPool`, generation-stamped, recorded after output accumulation on expert execution streams, waited by the caller stream even when that generation is about to rethrow an asynchronous error, and recycled by the autonomous completion-retirement callback/queue. `DispatchExperts()` never clears an event vector. A query-based path handles records for which no caller wait was inserted, and destruction synchronizes/releases only the exceptional callback-launch fallback list. Completion events are never timed.
- **Timing events:** privately owned by `GpuTimer`, created/destroyed by that timer, and synchronized only through the explicitly named `elapsed_millis_blocking()` benchmark/debug API. Timing events never enter `completion_events_`, `completion_retirement_queue_`, `destructor_fallback_events_`, or `kCudaEventPool`.
- **Route readiness:** uses stream-ordered pinned copy plus `cudaLaunchHostFunc`, not an event and not `cudaEventSynchronize`.
- **Expert retirement:** uses a separate stream callback/counter/queue; it neither releases completion events nor changes timing-event ownership.

### Task 1: Freeze the opt-in and eager-fallback contract

**Files:**
- Modify: `tests/python/unit/test_utils_config.py`
- Create: `tests/python/unit/test_gpu_only_expert_routing.py`
- Modify: `moe_infinity/utils/config.py:17-89`
- Modify: `moe_infinity/distributed/expert_executor.py:97-117`

- [ ] **Step 1: Write failing configuration tests**

Append to `tests/python/unit/test_utils_config.py`:

```python
def test_gpu_only_expert_routing_defaults_off(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(offload_path="/tmp", use_native_engine=False)
    assert config.gpu_only_expert_routing is False


def test_gpu_only_expert_routing_loads_from_json(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig.load_from_json(
        {
            "offload_path": "/tmp",
            "use_native_engine": False,
            "gpu_only_expert_routing": True,
        }
    )
    assert config.gpu_only_expert_routing is True


def test_gpu_routing_rejects_current_overlap_boolean(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(
        ValueError,
        match="gpu_only_expert_routing cannot be combined with overlap prefetch",
    ):
        ArcherConfig(
            offload_path="/tmp",
            use_native_engine=False,
            gpu_only_expert_routing=True,
            speculative_prefetch_overlap=True,
        )


@pytest.mark.parametrize("mode", ["observe", "enforce"])
def test_gpu_routing_rejects_future_overlap_modes(mode):
    with pytest.raises(
        ValueError,
        match="gpu_only_expert_routing cannot be combined with overlap prefetch",
    ):
        ArcherConfig._validate_gpu_routing_overlap(True, False, mode)
```

- [ ] **Step 2: Run the configuration tests and verify RED**

Run:

```bash
pytest -q tests/python/unit/test_utils_config.py \
  -k 'gpu_only_expert_routing or gpu_routing_rejects'
```

Expected: default/JSON tests fail because the field is absent and conflict tests fail because no rejection helper exists.

- [ ] **Step 3: Add the rollout flag**

Insert after `speculative_prefetch_overlap` in `ArcherConfig`:

```python
    gpu_only_expert_routing: bool = field(
        default=False,
        metadata={
            "help": (
                "Use native CUDA active-expert discovery for single-host local "
                "dispatch. Falls back to eager Python routing when unavailable."
            )
        },
    )
```

Add this validator and call it at the beginning of `__post_init__()`:

```python
    @staticmethod
    def _validate_gpu_routing_overlap(
        gpu_only_expert_routing: bool,
        speculative_prefetch_overlap: bool,
        overlap_prefetch_mode: str = "off",
    ) -> None:
        if gpu_only_expert_routing and (
            speculative_prefetch_overlap
            or overlap_prefetch_mode in {"observe", "enforce"}
        ):
            raise ValueError(
                "gpu_only_expert_routing cannot be combined with overlap "
                "prefetch in the first release; disable "
                "speculative_prefetch_overlap and overlap_prefetch_mode"
            )
```

```python
        self._validate_gpu_routing_overlap(
            self.gpu_only_expert_routing,
            self.speculative_prefetch_overlap,
            getattr(self, "overlap_prefetch_mode", "off"),
        )
```

`overlap_prefetch_mode` is intentionally read with `getattr`: it is not introduced as a public field by this plan, but if the overlap plan lands first with `observe`/`enforce`, the conflict remains rejected without changing either field name.

Initialize the executor mode in `DistributedExpertExecutor.__init__`:

```python
        self._gpu_only_expert_routing = bool(
            getattr(archer_config, "gpu_only_expert_routing", False)
        )
        self._last_dispatch_used_native_routing = False
        self._gpu_route_fallback_count = 0
```

- [ ] **Step 4: Run the configuration tests and verify GREEN**

Run:

```bash
pytest -q tests/python/unit/test_utils_config.py \
  -k 'gpu_only_expert_routing or gpu_routing_rejects'
```

Expected: five tests pass (the parametrized mode test contributes two cases).

- [ ] **Step 5: Commit the rollout contract**

```bash
git add moe_infinity/utils/config.py \
  moe_infinity/distributed/expert_executor.py \
  tests/python/unit/test_utils_config.py
git commit -m "feat: add opt-in gpu expert routing mode"
```

### Task 2: Add native GPU activity discovery and background enqueue

**Files:**
- Modify: `tests/python/unit/test_gpt_oss_mxfp4_dispatch.py:134-265`
- Modify: `core/parallel/expert_dispatcher.h:40-194`
- Modify: `core/parallel/expert_dispatcher.cpp:91-290,719-755`
- Modify: `core/python/py_archer_prefetch.cpp:109-123`

- [ ] **Step 1: Write a failing CUDA test for exact native active IDs**

Change only the existing helper signature at line 134 to:

```python
def _native_dispatch(
    tmp_path,
    hidden_dtype,
    *,
    active_experts=(0,),
    gpu_routing=False,
    dispatch_fault=None,
    wait_for_result=True,
    capture_wait_error=False,
):
```

Keep its current registered-weight setup at lines 135-211, then replace the routing tail at lines 212-226 with:

```python
    router_mask = torch.zeros((3, 2), dtype=torch.bool, device="cuda:0")
    router_weights = torch.zeros(
        (3, 2), dtype=torch.bfloat16, device="cuda:0"
    )
    for expert_id in active_experts:
        router_mask[:, expert_id] = True
        router_weights[:, expert_id] = 1.0 / max(len(active_experts), 1)
    dispatcher.set_inputs(hidden_states, router_mask, router_weights)
    if gpu_routing:
        if dispatch_fault is not None:
            dispatcher._set_dispatch_fault_for_test(dispatch_fault)
        dispatcher.dispatch_experts(0)
    else:
        dispatcher.set_expected_queue(len(active_experts))
        for expert_id in sorted(active_experts):
            dispatcher.enqueue_expert(
                0, expert_id, expert_id % torch.cuda.device_count(), False
            )
        dispatcher.notify_fetch_start()
    actual = None
    if wait_for_result:
        try:
            actual = dispatcher.wait_expert()
        except RuntimeError as error:
            if not capture_wait_error:
                raise
            actual = error
    return actual, expected, hidden_states, tensors, dispatcher
```

Update the two existing callers to unpack the fifth return value:

```python
    actual, resident, hidden_states, tensors, _ = _native_dispatch(
        tmp_path, torch.bfloat16
    )
```

```python
    actual, _, _, _, _ = _native_dispatch(tmp_path, torch.float32)
```

Then add:

```python
@pytest.mark.gpu
def test_native_gpu_routing_reports_sorted_active_experts(tmp_path):
    _, _, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(1, 0),
        gpu_routing=True,
    )
    assert dispatcher.take_last_active_experts() == [0, 1]
    stats = dispatcher.get_routing_stats()
    assert stats["route_batches"] == 1
    assert stats["route_failures"] == 0
    assert stats["last_active_experts"] == 2


@pytest.mark.parametrize("fault", ["callback", "worker"])
@pytest.mark.gpu
def test_native_routing_failure_rethrows_without_deadlock(tmp_path, fault):
    with pytest.raises(RuntimeError, match=f"injected {fault} routing failure"):
        _native_dispatch(
            tmp_path / fault,
            torch.bfloat16,
            active_experts=(0,),
            gpu_routing=True,
            dispatch_fault=fault,
        )


@pytest.mark.gpu
def test_synchronous_submission_failure_closes_route_state(tmp_path):
    with pytest.raises(RuntimeError, match="injected submission failure"):
        _native_dispatch(
            tmp_path,
            torch.bfloat16,
            active_experts=(0,),
            gpu_routing=True,
            dispatch_fault="submission",
        )


@pytest.mark.gpu
def test_dispatcher_destruction_drains_inflight_route_callback(tmp_path):
    import gc

    _, _, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0,),
        gpu_routing=True,
        wait_for_result=False,
    )
    del dispatcher
    gc.collect()
```

- [ ] **Step 2: Run the CUDA test and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py::test_native_gpu_routing_reports_sorted_active_experts
```

Expected: the active-ID test fails with `AttributeError` for `dispatch_experts`; the failure tests fail because `_set_dispatch_fault_for_test` is absent; the destruction test times out or exposes the missing callback-drain contract.

- [ ] **Step 3: Define routing state and ownership in the header**

Add the following public methods, structs, private methods, and members to `ExpertDispatcher`:

```cpp
  // Add to both CallArgs and ExecArgs so late workers cannot underflow a failed
  // generation's pending count.
  std::uint64_t generation = 0;
  bool cache_slot_reserved = false;
  bool cache_key_inserted = false;

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

  void DispatchExperts(int layer_idx);
  std::vector<int> TakeLastActiveExperts();
  std::map<std::string, std::int64_t> GetRoutingStats() const;
  void SetDispatchFaultForTest(const std::string& stage);
  void FailDispatchForTest(std::uint64_t generation,
                           const std::string& message);

 private:
  void RouteFunc() noexcept;
  static void CUDART_CB RouteReadyCallback(void* opaque);
  void FailDispatch(const std::uint64_t failing_generation,
                    std::exception_ptr error,
                    const std::vector<FailureContext>& contexts = {}) noexcept;
  void CompleteOne(std::uint64_t generation) noexcept;

  class DispatchSubmissionGuard {
   public:
    DispatchSubmissionGuard(ExpertDispatcher* dispatcher,
                            const std::uint64_t generation)
        : dispatcher_(dispatcher), generation_(generation) {}
    ~DispatchSubmissionGuard() {
      if (armed_) {
        dispatcher_->FailDispatch(
            generation_,
            error_ ? error_ : std::make_exception_ptr(std::runtime_error(
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
  struct CompletionEventRecord {
    cudaEvent_t event = nullptr;
    std::uint64_t generation = 0;
  };
  struct CompletionRetireBatch {
    ExpertDispatcher* dispatcher = nullptr;
    std::vector<CompletionEventRecord> records;
  };
  struct CompletionRetireItem {
    CompletionEventRecord record;
    bool caller_wait_consumed = false;
  };
  static void CUDART_CB CompletionWaitsConsumedCallback(void* opaque);
  void CompletionRetirementFunc();
  void QueueUnwaitedEventsForQuery(
      std::vector<CompletionEventRecord> records) noexcept;
  std::vector<CompletionEventRecord> completion_events_;
  std::vector<CompletionEventRecord> destructor_fallback_events_;
  ThreadSafeQueue<std::vector<CompletionRetireItem>>
      completion_retirement_queue_;
  std::atomic<std::int64_t> pending_completion_retirement_callbacks_{0};
  std::mutex completion_retirement_callback_mutex_;
  std::condition_variable completion_retirement_callback_cv_;
  std::atomic<std::int64_t> route_batches_{0};
  std::atomic<std::int64_t> route_failures_{0};
  std::atomic<std::int64_t> last_active_experts_count_{0};
  std::atomic<std::int64_t> last_route_handoff_us_{0};
  std::atomic<std::int64_t> completion_events_retired_{0};
  std::atomic<std::int64_t> completion_events_outstanding_{0};
  std::atomic<std::int64_t> stale_failures_quarantined_{0};
```

Include `<exception>` in the header. In the destructor, first wait on `route_callback_cv_` until `pending_route_callbacks_ == 0` and on `completion_retirement_callback_cv_` until `pending_completion_retirement_callbacks_ == 0`, then close `route_queue_` and `completion_retirement_queue_`, set the stop flag, notify `pending_cv_`, and join `threads_`. Only after all workers join may it synchronize and release records in `completion_events_` or `destructor_fallback_events_`; increment `completion_events_retired_` for each release. This destructor-only host wait prevents a CUDA callback from retaining a freed dispatcher and guarantees cleanup when no later dispatch occurs; it is not on the per-layer path. Never hold `route_callback_mutex_`, `completion_retirement_callback_mutex_`, `route_state_mutex_`, or `pending_mutex_` while closing queues, synchronizing fallback events, or joining threads.

- [ ] **Step 4: Start the route worker and implement GPU reduction submission**

At the end of the constructor's stream/worker setup, start exactly one route thread:

```cpp
  auto route_func = std::bind(&ExpertDispatcher::RouteFunc, this);
  threads_.emplace_back(new base::Thread(route_func, "ExpertRouteFunc"));
  threads_.back()->start();
```

Implement `DispatchExperts` in `expert_dispatcher.cpp`:

```cpp
void ExpertDispatcher::DispatchExperts(int layer_idx) {
  TORCH_CHECK(router_mask_.defined(),
              "DispatchExperts: SetInputs must be called first");
  TORCH_CHECK(router_mask_.is_cuda(),
              "DispatchExperts: router_mask must be CUDA-resident");
  TORCH_CHECK(router_mask_.dim() == 2 &&
                  router_mask_.size(1) == num_experts_,
              "DispatchExperts: router_mask must be [tokens, num_experts]");

  bool expected = false;
  TORCH_CHECK(route_pending_.compare_exchange_strong(
                  expected, true, std::memory_order_acq_rel),
              "DispatchExperts: previous dispatch has not been waited");
  const std::uint64_t generation =
      current_generation_.load(std::memory_order_acquire);
  DispatchSubmissionGuard submission(this, generation);
  try {
    int submission_fault =
        static_cast<int>(DispatchFaultPoint::SUBMISSION);
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
    auto callback_args =
        std::make_unique<RouteCallbackArgs>(RouteCallbackArgs{this, std::move(args)});
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
```

Use the existing `CUDA_CHECK` macro from `core/utils/cuda_utils.h:38` consistently; do not define a second checker.

- [ ] **Step 5: Implement background handoff with ascending IDs and safe pending state**

Implement the common failure closure, no-CUDA host callback, `RouteFunc`, fault injector, and active-list/stat accessors. `FailDispatch(failing_generation, ...)` is the only asynchronous failure exit for route, fetch, execution, output, completion-event, and retirement-launch failures. The immutable argument comes from `RouteArgs`, `CallArgs`, `ExecArgs`, `ExpertRetireArgs`, or the submission guard—not from `current_generation_`. It stores the first exception, marks that generation failed, zeros pending work, clears `route_pending_`, restores node/cache state when supplied, and notifies every waiter only when `failing_generation` is still current. A stale failure is counted and otherwise quarantined before context restoration, because touching a node/cache slot now owned by the next generation would corrupt live work.

```cpp
void ExpertDispatcher::FailDispatch(
    const std::uint64_t failing_generation,
    std::exception_ptr error,
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
        // Preserve the first dispatch exception; state restoration below still
        // prevents waiters from deadlocking.
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
        // Preserve the first dispatch exception and continue terminal-state
        // restoration/notification; FailDispatch is noexcept.
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
    int callback_fault =
        static_cast<int>(DispatchFaultPoint::ROUTE_CALLBACK);
    if (dispatcher->dispatch_fault_for_test_.compare_exchange_strong(
            callback_fault, 0, std::memory_order_acq_rel)) {
      throw std::runtime_error("injected callback routing failure");
    }
    dispatcher->route_queue_.Push(callback->route);
  } catch (...) {
    dispatcher->FailDispatch(callback->route.generation,
                             std::current_exception());
  }
  if (dispatcher->pending_route_callbacks_.fetch_sub(
          1, std::memory_order_acq_rel) == 1) {
    dispatcher->route_callback_cv_.notify_all();
  }
}

void ExpertDispatcher::RouteFunc() noexcept {
  RouteArgs args;
  while (route_queue_.Pop(args)) {
      std::vector<FailureContext> routed_contexts;
      try {
#ifndef NVTX_DISABLE
        nvtx3::scoped_range range("gpu_route_handoff");
#endif
        int worker_fault =
            static_cast<int>(DispatchFaultPoint::ROUTE_WORKER);
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
              experts_[expert_idx][args.layer_idx], gpu_id, false, false,
              false});
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
        last_route_handoff_us_.store(elapsed.count(),
                                     std::memory_order_relaxed);
        route_pending_.store(false, std::memory_order_release);
        pending_cv_.notify_all();
      } catch (...) {
        FailDispatch(args.generation, std::current_exception(),
                     routed_contexts);
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
      {"current_generation", static_cast<std::int64_t>(
                                 current_generation_.load(
                                     std::memory_order_acquire))},
      {"pending", static_cast<std::int64_t>(
                      pending_.load(std::memory_order_acquire))},
      {"route_pending",
       route_pending_.load(std::memory_order_acquire) ? 1 : 0},
  };
}

void ExpertDispatcher::FailDispatchForTest(
    std::uint64_t generation, const std::string& message) {
  FailDispatch(generation,
               std::make_exception_ptr(std::runtime_error(message)));
}
```

Include `<chrono>` in `expert_dispatcher.cpp`. `RouteReadyCallback` must remain limited to moving `RouteArgs`, decrementing its callback counter, and notifying the condition variable; CUDA APIs, tensor operations, expert enqueue, and blocking work remain forbidden inside the CUDA callback.

At the start of `SetInputs()`, reject overlapping generations and reset failure state before storing tensors:

```cpp
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
```

Set `CallArgs::generation` in `EnqueueExpert()` from `current_generation_` and copy it into every `ExecArgs`. Replace every raw `pending_.fetch_sub(1)` in success/error paths with `CompleteOne(args.generation)` so late work from a failed generation cannot underflow zero or satisfy a later generation's wait.

- [ ] **Step 6: Bind the complete native interface**

Add these bindings after `set_inputs` in `py_archer_prefetch.cpp`:

```cpp
      .def("dispatch_experts", &ExpertDispatcher::DispatchExperts)
      .def("take_last_active_experts",
           &ExpertDispatcher::TakeLastActiveExperts)
      .def("get_routing_stats", &ExpertDispatcher::GetRoutingStats)
      .def("_set_dispatch_fault_for_test",
           &ExpertDispatcher::SetDispatchFaultForTest)
      .def("_fail_dispatch_for_test",
           &ExpertDispatcher::FailDispatchForTest)
```

Keep `set_expected_queue`, `enqueue_expert`, and `notify_fetch_start` bound for eager fallback.

- [ ] **Step 7: Build the extension and verify GREEN**

Run:

```bash
CUTLASS_DIR="$HOME/cutlass" pip install --no-build-isolation -e .
CUDA_VISIBLE_DEVICES=0 timeout 30s pytest -q \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py \
  -k 'native_gpu_routing_reports_sorted or routing_failure_rethrows or synchronous_submission or destruction_drains'
```

Expected: editable build succeeds; active IDs pass; callback and worker failures raise on the Python wait thread without a hang; destruction completes within 30 seconds.

- [ ] **Step 8: Commit native routing submission**

```bash
git add core/parallel/expert_dispatcher.h \
  core/parallel/expert_dispatcher.cpp \
  core/python/py_archer_prefetch.cpp \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py
git commit -m "feat: discover active experts on the gpu"
```

### Task 3: Replace GPU completion synchronization with stream waits

**Files:**
- Modify: `tests/python/unit/test_gpt_oss_mxfp4_dispatch.py`
- Modify: `core/parallel/expert_dispatcher.h:133-135,160-182`
- Modify: `core/parallel/expert_dispatcher.cpp:588-717,735-755`

- [ ] **Step 1: Write failing CUDA completion and empty-mask tests**

Add:

```python
@pytest.mark.gpu
def test_native_wait_returns_stream_ordered_exact_output(tmp_path):
    actual, expected, _, _, _ = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0, 1),
        gpu_routing=True,
    )
    consumer = actual.square().sum()

    torch.testing.assert_close(
        consumer, expected.square().sum(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.gpu
def test_native_gpu_routing_empty_mask_returns_zero_without_deadlock(tmp_path):
    actual, _, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(),
        gpu_routing=True,
    )
    assert dispatcher.take_last_active_experts() == []
    assert torch.count_nonzero(actual).item() == 0
```

- [ ] **Step 2: Run both tests before the event handoff and verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py \
  -k 'stream_ordered_exact_output or empty_mask_returns_zero'
```

Expected: the empty-mask test hangs or returns before routing closes, or the immediate consumer exposes missing cross-stream ordering; terminate a hang after 30 seconds and record which assertion/timeout failed.

- [ ] **Step 3: Record one completion event after each output accumulation launch**

Change the declaration and call to pass the owning execution stream:

```cpp
  bool OutputFunc(ExecArgs args, torch::Tensor output,
                  torch::Tensor token_mask, int gpu_id,
                  cudaStream_t exec_stream) noexcept;
```

```cpp
      OutputFunc(args, output, token_mask, gpu_id, stream);
```

Add this event lease in the anonymous namespace at the top of `expert_dispatcher.cpp`, after `memory/event_pool.h` is included; it releases an acquired event unless ownership is explicitly transferred:

```cpp
  class PooledCudaEventLease {
   public:
    PooledCudaEventLease() : event_(kCudaEventPool->Acquire()) {}
    ~PooledCudaEventLease() {
      if (event_ != nullptr) kCudaEventPool->Release(event_);
    }
    cudaEvent_t get() const noexcept { return event_; }
    cudaEvent_t Release() noexcept {
      cudaEvent_t event = event_;
      event_ = nullptr;
      return event;
    }

   private:
    cudaEvent_t event_ = nullptr;
  };
```

At the end of the `accum_mutex_` critical section, before `CompleteOne`, record and retain an event. The injected failure occurs after acquisition so the RED test also proves RAII release:

```cpp
  PooledCudaEventLease output_done;
  int event_fault =
      static_cast<int>(DispatchFaultPoint::COMPLETION_EVENT_RECORD);
  if (dispatch_fault_for_test_.compare_exchange_strong(
          event_fault, 0, std::memory_order_acq_rel)) {
    throw std::runtime_error("injected completion_event failure");
  }
  CUDA_CHECK(cudaEventRecord(output_done.get(), exec_stream));
  {
    std::lock_guard<std::mutex> lock(route_state_mutex_);
    completion_events_.push_back(
        CompletionEventRecord{output_done.Release(), args.generation});
    completion_events_outstanding_.fetch_add(1, std::memory_order_relaxed);
  }
  CompleteOne(args.generation);
```

Keep the existing float32 `add_` / `index_add_` code and its host launch order unchanged.

- [ ] **Step 4: Make wait close routing, surface worker errors, and hand events to the caller stream**

Replace `WaitHiddenStates()` with the following ownership order. The state lock is used only to detach this generation's error and event records; no CUDA call and no `FailDispatch()` call occurs while it is held. Events are handed off before the saved error is rethrown, including the `retirement_launch` failure path:

```cpp
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
```

Implement autonomous retirement; it must not depend on `DispatchExperts()` being called again. The CUDA host callback performs no CUDA API call and only transfers ownership to the CPU queue. The worker releases callback-ordered records immediately; `QueueUnwaitedEventsForQuery` marks records as query-required, and the worker requeues those until `cudaEventQuery` returns `cudaSuccess`. `cudaErrorNotReady` is not a dispatch failure. Any other query error calls `FailDispatch(record.generation, ...)` and moves the still-owned record to `destructor_fallback_events_` before continuing:

```cpp
void ExpertDispatcher::QueueUnwaitedEventsForQuery(
    std::vector<CompletionEventRecord> records) noexcept {
  if (records.empty()) return;
  std::vector<CompletionRetireItem> items;
  items.reserve(records.size());
  for (const CompletionEventRecord& record : records) {
    items.push_back(CompletionRetireItem{record, false});
  }
  try {
    completion_retirement_queue_.Push(std::move(items));
  } catch (...) {
    std::lock_guard<std::mutex> lock(route_state_mutex_);
    destructor_fallback_events_.insert(
        destructor_fallback_events_.end(),
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
    dispatcher->completion_retirement_queue_.Push(std::move(items));
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
        auto error = std::make_exception_ptr(std::runtime_error(
            std::string("completion event query failed: ") +
            cudaGetErrorString(status)));
        FailDispatch(item.record.generation, error);
        std::lock_guard<std::mutex> lock(route_state_mutex_);
        destructor_fallback_events_.push_back(std::move(item.record));
        continue;
      }
      kCudaEventPool->Release(item.record.event);
      completion_events_retired_.fetch_add(1, std::memory_order_relaxed);
      completion_events_outstanding_.fetch_sub(1, std::memory_order_relaxed);
    }
    if (!retry.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      completion_retirement_queue_.Push(std::move(retry));
    }
    items.clear();
  }
}
```

`QueueUnwaitedEventsForQuery()` converts each record to `CompletionRetireItem{record, false}` and pushes the batch. If that queue push throws, it moves the still-owned records into `destructor_fallback_events_` under `route_state_mutex_`; it is `noexcept`. Start `CompletionRetirementFunc` in the constructor and include `<iterator>` and `<thread>`. Maintain an atomic `completion_events_outstanding_`: increment exactly when `PooledCudaEventLease::Release()` inserts a record and decrement exactly when the retirement worker or destructor returns that record to the pool. `GetRoutingStats()` reads that atomic for `completion_events_outstanding`, so callback-owned and queued/querying handles are included. Remove `RetireCompletedEvents()` and every call to it. In particular, never replace retirement with `completion_events_.clear()`.

For the empty active set, `RouteFunc()` closes `route_pending_` and notifies with `pending_ == 0`; no completion event is required because `SetInputs()` created `final_hidden_states_` on the caller stream.

- [ ] **Step 5: Run CUDA completion tests and verify GREEN**

Run:

```bash
CUTLASS_DIR="$HOME/cutlass" pip install --no-build-isolation -e .
CUDA_VISIBLE_DEVICES=0 timeout 30s pytest -q \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py \
  -k 'stream_ordered_exact_output or empty_mask_returns_zero'
```

Expected: both tests pass within 30 seconds and no CUDA synchronization error is printed.

- [ ] **Step 6: Commit stream-ordered completion**

```bash
git add core/parallel/expert_dispatcher.h \
  core/parallel/expert_dispatcher.cpp \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py
git commit -m "feat: hand expert completion to the caller stream"
```

### Task 3A: Remove hidden expert-worker synchronizations and prove non-default stream ownership

**Files:**
- Create: `tests/python/unit/test_gpu_routing_source_contract.py`
- Modify: `tests/python/unit/test_gpt_oss_mxfp4_dispatch.py:134-265`
- Modify: `core/utils/cuda_utils.h:44-81,92-96`
- Modify: `core/parallel/expert_dispatcher.h:49-58,123-194`
- Modify: `core/parallel/expert_dispatcher.cpp:588-717`
- Modify: `core/parallel/expert_module.cpp:185-247`

- [ ] **Step 1: Write RED source-contract and non-default-stream tests**

Create `tests/python/unit/test_gpu_routing_source_contract.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _function_body(path, start, end):
    source = (ROOT / path).read_text(encoding="utf-8")
    return source[source.index(start) : source.index(end, source.index(start))]


def test_expert_forward_has_no_stream_synchronize():
    body = _function_body(
        "core/parallel/expert_module.cpp",
        "torch::Tensor MoEMLP::forward",
        "void MoEMLP::ForwardHelper",
    )
    assert "cudaStreamSynchronize" not in body


def test_exec_stream_guard_precedes_input_tensor_work():
    body = _function_body(
        "core/parallel/expert_dispatcher.cpp",
        "void ExpertDispatcher::GPUExecFunc",
        "bool ExpertDispatcher::OutputFunc",
    )
    guard_at = body.index("c10::cuda::CUDAStreamGuard guard(torch_stream)")
    token_mask_at = body.index("router_mask_.index")
    input_copy_at = body.index("hidden_states_.index")
    assert guard_at < token_mask_at < input_copy_at


def test_blocking_cuda_helpers_are_explicit_and_not_used_by_dispatch():
    utilities = (ROOT / "core/utils/cuda_utils.h").read_text(encoding="utf-8")
    dispatcher = (
        ROOT / "core/parallel/expert_dispatcher.cpp"
    ).read_text(encoding="utf-8")
    module = (ROOT / "core/parallel/expert_module.cpp").read_text(
        encoding="utf-8"
    )
    assert "elapsed_millis_blocking" in utilities
    assert "BlockingCudaCopy" not in dispatcher + module
    assert "GpuTimer" not in dispatcher + module


def test_all_dispatch_workers_use_common_failure_closure():
    path = "core/parallel/expert_dispatcher.cpp"
    fetch = _function_body(
        path,
        "void ExpertDispatcher::GPUFetchFunc",
        "void ExpertDispatcher::GPUExecFunc",
    )
    execute = _function_body(
        path,
        "void ExpertDispatcher::GPUExecFunc",
        "bool ExpertDispatcher::OutputFunc",
    )
    output = _function_body(
        path,
        "bool ExpertDispatcher::OutputFunc",
        "std::vector<ExpertDispatcher::CallResult>",
    )
    for body in (fetch, execute, output):
        assert "catch (...)" in body
        assert "FailDispatch(args.generation, std::current_exception()" in body
        assert "pending_.fetch_sub" not in body
    assert "DLOG_WARN(\"GPUExecFunc: expert forward failed" not in execute


def test_route_error_is_exception_ptr_only():
    header = (ROOT / "core/parallel/expert_dispatcher.h").read_text(
        encoding="utf-8"
    )
    source = (ROOT / "core/parallel/expert_dispatcher.cpp").read_text(
        encoding="utf-8"
    )
    assert "std::exception_ptr route_error_;" in header
    assert "route_error_ = std::string" not in source
    assert "std::rethrow_exception(route_error)" in source


def test_route_pending_submission_is_raii_guarded():
    body = _function_body(
        "core/parallel/expert_dispatcher.cpp",
        "void ExpertDispatcher::DispatchExperts",
        "void ExpertDispatcher::RouteReadyCallback",
    )
    pending_at = body.index("route_pending_.compare_exchange_strong")
    guard_at = body.index("DispatchSubmissionGuard submission(this, generation)")
    first_throwing_work = body.index("TORCH_CHECK(completion_events_.empty()")
    assert pending_at < guard_at < first_throwing_work


def test_fail_dispatch_is_generation_explicit_and_wait_never_recurses_lock():
    header = (ROOT / "core/parallel/expert_dispatcher.h").read_text(
        encoding="utf-8"
    )
    wait = _function_body(
        "core/parallel/expert_dispatcher.cpp",
        "torch::Tensor ExpertDispatcher::WaitHiddenStates",
        "void ExpertDispatcher::CompletionWaitsConsumedCallback",
    )
    assert "FailDispatch(const std::uint64_t failing_generation" in header
    assert "FailDispatch(generation, error)" in wait
    detach_end = wait.index("  }\n\n  const int device")
    fail_at = wait.index("FailDispatch(generation, error)")
    assert detach_end < fail_at
    assert "completion_events_.clear()" not in wait


def test_completion_retirement_is_autonomous():
    source = (ROOT / "core/parallel/expert_dispatcher.cpp").read_text(
        encoding="utf-8"
    )
    dispatch = _function_body(
        "core/parallel/expert_dispatcher.cpp",
        "void ExpertDispatcher::DispatchExperts",
        "void ExpertDispatcher::RouteReadyCallback",
    )
    assert "CompletionWaitsConsumedCallback" in source
    assert "CompletionRetirementFunc" in source
    assert "QueueUnwaitedEventsForQuery" in source
    assert "completion_events_.clear()" not in dispatch
```

Add to `tests/python/unit/test_gpt_oss_mxfp4_dispatch.py`:

```python
@pytest.mark.gpu
def test_native_wait_orders_output_on_non_default_caller_stream(tmp_path):
    caller_stream = torch.cuda.Stream()
    with torch.cuda.stream(caller_stream):
        actual, expected, _, _, _ = _native_dispatch(
            tmp_path,
            torch.bfloat16,
            active_experts=(0, 1),
            gpu_routing=True,
        )
        dependent = actual.float().square().sum()
        done = torch.cuda.Event()
        done.record(caller_stream)

    done.synchronize()
    torch.testing.assert_close(
        dependent, expected.float().square().sum(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize(
    "fault",
    [
        "fetch",
        "exec",
        "output",
        "completion_event",
        "retirement_launch",
    ],
)
@pytest.mark.gpu
def test_worker_and_handoff_failures_rethrow_from_wait(tmp_path, fault):
    with pytest.raises(RuntimeError, match=f"injected {fault} failure"):
        _native_dispatch(
            tmp_path / fault,
            torch.bfloat16,
            active_experts=(0,),
            gpu_routing=True,
            dispatch_fault=fault,
        )


@pytest.mark.gpu
def test_failure_cleanup_allows_dispatcher_destruction(tmp_path):
    import gc

    with pytest.raises(RuntimeError, match="injected completion_event failure"):
        _native_dispatch(
            tmp_path,
            torch.bfloat16,
            active_experts=(0,),
            gpu_routing=True,
            dispatch_fault="completion_event",
        )
    gc.collect()


@pytest.mark.gpu
def test_failed_generation_closes_state_and_next_generation_runs(tmp_path):
    error, _, hidden, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0,),
        gpu_routing=True,
        dispatch_fault="exec",
        capture_wait_error=True,
    )
    assert isinstance(error, RuntimeError)
    stats = dispatcher.get_routing_stats()
    assert stats["pending"] == 0
    assert stats["route_pending"] == 0
    failed_generation = stats["current_generation"]

    mask = torch.tensor(
        [[True, False], [True, False], [True, False]],
        dtype=torch.bool,
        device=hidden.device,
    )
    dispatcher.set_inputs(hidden, mask, mask.to(torch.bfloat16))
    dispatcher.dispatch_experts(0)
    dispatcher._fail_dispatch_for_test(
        failed_generation, "injected stale generation failure"
    )
    recovered = dispatcher.wait_expert()
    assert recovered.shape == hidden.shape
    recovered_stats = dispatcher.get_routing_stats()
    assert recovered_stats["stale_failures_quarantined"] == 1
    assert recovered_stats["pending"] == 0
    assert recovered_stats["route_pending"] == 0


@pytest.mark.gpu
def test_retirement_launch_failure_retires_recorded_event_without_next_dispatch(
    tmp_path,
):
    import time

    error, _, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0,),
        gpu_routing=True,
        dispatch_fault="retirement_launch",
        capture_wait_error=True,
    )
    assert isinstance(error, RuntimeError)
    assert "injected retirement_launch failure" in str(error)

    # This completes the caller-stream waits and their host handoff. Do not
    # start another dispatch: retirement must be autonomous.
    torch.cuda.synchronize()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        stats = dispatcher.get_routing_stats()
        if stats["completion_events_outstanding"] == 0:
            break
        time.sleep(0.01)
    assert stats["completion_events_retired"] >= 1
    assert stats["completion_events_outstanding"] == 0
```

- [ ] **Step 2: Run the synchronization tests and verify RED**

```bash
pytest -q tests/python/unit/test_gpu_routing_source_contract.py
CUDA_VISIBLE_DEVICES=0 timeout 30s pytest -q \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py \
  -k 'non_default_caller_stream or worker_and_handoff or failed_generation or retirement_launch_failure_retires'
```

Expected: source tests fail on both `cudaStreamSynchronize` calls, the late stream guard, missing `elapsed_millis_blocking`, generation-inferred failure closure, recursive wait-error locking, and next-dispatch event retirement. CUDA fault tests fail because fetch/exec/output/event/retirement errors are swallowed or unsupported, stale generation failure poisons the recovered dispatch, or the recorded completion event remains outstanding without a next dispatch.

- [ ] **Step 3: Make blocking utility contracts explicit and isolate timing events**

In `core/utils/cuda_utils.h`, rename the unused timer accessor and document both blocking helpers:

```cpp
  /// Timing-only API. Blocks the calling CPU thread until `_stop` completes.
  /// Forbidden in ExpertDispatcher and MoEMLP production paths.
  float elapsed_millis_blocking() {
    float elapsed = 0.0;
    CUDA_CHECK(cudaEventSynchronize(_stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, _start, _stop));
    return elapsed;
  }
```

```cpp
// Explicitly blocking copy for initialization/debug paths only. Never call from
// ExpertDispatcher, GPUFetchFunc, GPUExecFunc, OutputFunc, or MoEMLP::forward.
void BlockingCudaCopy(int device, void* dst, const void* src, size_t size,
                      cudaMemcpyKind kind, cudaStream_t stream);
```

`GpuTimer` owns timing events from construction through destruction and may synchronize only in `elapsed_millis_blocking()`. It must never acquire from or release to `kCudaEventPool`; pooled completion events have a separate lifecycle and are never passed to `cudaEventElapsedTime`.

- [ ] **Step 4: Put every input/dequant/forward/output launch on the execution stream**

In `GPUExecFunc`, move stream wrapping and guarding immediately after `stream` is selected and before `token_mask`, input indexing, `.to(device)`, dequantization, and expert forward:

```cpp
  c10::cuda::CUDAStream torch_stream =
      c10::cuda::getStreamFromExternal(stream, gpu_id);
  c10::cuda::CUDAStreamGuard guard(torch_stream);

  auto token_mask = router_mask_.index({"...", expert_idx});
  torch::Tensor input = (batch_size == 1)
                            ? hidden_states_.to(device)
                            : hidden_states_.index({token_mask}).to(device);
```

Delete the later duplicate guard at current lines 638-640. This guarantees that the D2D copy in `MoEMLP::forward`, all ATen kernels in `ForwardHelper`, `output_.clone()`, and `OutputFunc` launches are ordered on the same non-default stream.

- [ ] **Step 4A: Route every fetch/execute/output exception through `FailDispatch`**

At the top of each `GPUFetchFunc` iteration, create and update a failure context as cache state changes:

```cpp
    CallArgs args;
    FailureContext failure;
    failure.gpu_id = gpu_id;
    try {
      if (!input_queue_[gpu_id].Pop(args)) break;
      failure.expert_node = experts_[args.expert_idx][args.layer_idx];
      int fetch_fault =
          static_cast<int>(DispatchFaultPoint::FETCH_WORKER);
      if (dispatch_fault_for_test_.compare_exchange_strong(
              fetch_fault, 0, std::memory_order_acq_rel)) {
        throw std::runtime_error("injected fetch failure");
      }
```

Set `failure.overload_owned = true` immediately after `gpu_overload_[gpu_id].store(true)`. Set `failure.cache_slot_reserved = true` immediately after subtracting `byte_size`, and `failure.cache_key_inserted = true` immediately after inserting the key. Close the iteration with:

```cpp
    } catch (...) {
      FailDispatch(args.generation, std::current_exception(), {failure});
    }
```

Propagate `args.generation` and the three cache flags into `ExecArgs`. Replace the existing `GPUExecFunc` `catch (const std::exception&)` logging/swallow block with:

```cpp
    FailureContext failure{
        args.expert_node,
        gpu_id,
        args.cache_slot_reserved,
        args.cache_key_inserted,
        args.evict,
    };
    try {
      int exec_fault = static_cast<int>(DispatchFaultPoint::EXEC_WORKER);
      if (dispatch_fault_for_test_.compare_exchange_strong(
              exec_fault, 0, std::memory_order_acq_rel)) {
        throw std::runtime_error("injected exec failure");
      }
      // The existing stream-guarded dequant, forward, and OutputFunc call are
      // inside this try block.
    } catch (...) {
      if (args.transfer_event != nullptr) {
        kCudaEventPool->Release(args.transfer_event);
        args.transfer_event = nullptr;
      }
      FailDispatch(args.generation, std::current_exception(), {failure});
    }
```

Move the existing transfer-event `cudaStreamWaitEvent`/release block inside this `try` and wrap both calls with `CUDA_CHECK`; no CUDA/ATen operation may remain between a successful queue `Pop` and the catch boundary.

Change `OutputFunc` to return `bool` and contain its entire body in a catch-all boundary:

```cpp
bool ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output,
                                  torch::Tensor token_mask, int gpu_id,
                                  cudaStream_t exec_stream) noexcept {
  FailureContext failure{
      args.expert_node,
      gpu_id,
      args.cache_slot_reserved,
      args.cache_key_inserted,
      args.evict,
  };
  try {
    int output_fault = static_cast<int>(DispatchFaultPoint::OUTPUT);
    if (dispatch_fault_for_test_.compare_exchange_strong(
            output_fault, 0, std::memory_order_acq_rel)) {
      throw std::runtime_error("injected output failure");
    }
    // Keep the existing output conversion and float32 accumulation here.
    CompleteOne(args.generation);
    return true;
  } catch (...) {
    FailDispatch(args.generation, std::current_exception(), {failure});
    return false;
  }
}
```

The comments identify unchanged code already shown in `expert_dispatcher.cpp:615-653` and `672-712`; do not duplicate, reorder, or delete that math. `GPUExecFunc` ignores the boolean except for debug logging because the failure is retained for `WaitHiddenStates()`.

- [ ] **Step 5: Remove both worker stream synchronizations**

Replace the body segment in `MoEMLP::forward` with stream-ordered operations:

```cpp
  CUDA_CHECK(cudaMemcpyAsync(
      input_.data_ptr(), hidden_states.data_ptr(),
      hidden_states.numel() * hidden_states.element_size(),
      cudaMemcpyDeviceToDevice, stream));

  for (auto& buffer : buffer_) {
    auto shape_vec = buffer.sizes().vec();
    if (shape_vec.size() != 2) continue;
    int64_t row = buffer.size(0);
    int64_t col = buffer.size(1);
    auto dtype = buffer.dtype();
    if (row == kMaxTokens) {
      buffer.set_data(torch::from_blob(
          buffer.data_ptr(), {batch_size, col}, DoNothingDeleter<void>{},
          torch::TensorOptions().dtype(dtype).device(
              CUDA_DEVICE(at::cuda::current_device()))));
    }
  }

  ForwardHelper(stream);
  param_set_ = false;
  auto output = output_.clone();
```

Keep the existing buffer-shape restoration after `clone()`. Do not add an event synchronize: the caller stream waits on dispatch completion events established by Task 3.

- [ ] **Step 6: Retire expert/cache state only after the execution stream is complete**

Add these exact types and members to `expert_dispatcher.h`:

```cpp
  typedef struct {
    ExpertDispatcher* dispatcher = nullptr;
    ExpertNodePtr expert_node = nullptr;
    std::uint64_t generation = 0;
    int gpu_id = -1;
    bool evict = false;
  } ExpertRetireArgs;

  static void CUDART_CB ExpertReadyToRetireCallback(void* opaque);
  void RetirementFunc();
  ThreadSafeQueue<ExpertRetireArgs> retirement_queue_;
  std::atomic<std::int64_t> pending_retirement_callbacks_{0};
  std::mutex retirement_callback_mutex_;
  std::condition_variable retirement_callback_cv_;
```

In the same step, extend the Task 2 `FailDispatch()` terminal notification block now that `retirement_callback_cv_` exists:

```cpp
  pending_cv_.notify_all();
  route_callback_cv_.notify_all();
  retirement_callback_cv_.notify_all();
```

Task 2 intentionally notifies only `pending_cv_` and `route_callback_cv_`; the retirement callback condition variable and its notification have one lifetime and are introduced together here.

Start one `ExpertRetirementFunc` thread in the constructor. At the end of `OutputFunc`, after recording `output_done` and before decrementing `pending_`, enqueue a stream callback:

```cpp
  auto* retire = new ExpertRetireArgs{
      this, args.expert_node, args.generation, gpu_id, args.evict};
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
```

Remove the current early `exec_state.store(IDLE)`, overloaded-expert `SetDevice`, `gpu_overload_` clear, and `cache_cv_` notification from `OutputFunc`. Implement a no-CUDA callback and retirement worker:

```cpp
void CUDART_CB ExpertDispatcher::ExpertReadyToRetireCallback(void* opaque) {
  std::unique_ptr<ExpertRetireArgs> retire(
      static_cast<ExpertRetireArgs*>(opaque));
  ExpertDispatcher* dispatcher = retire->dispatcher;
  try {
    dispatcher->retirement_queue_.Push(*retire);
  } catch (...) {
    dispatcher->FailDispatch(
        retire->generation,
        std::current_exception(),
        {FailureContext{retire->expert_node, retire->gpu_id, false, false,
                        retire->evict}});
  }
  if (dispatcher->pending_retirement_callbacks_.fetch_sub(
          1, std::memory_order_acq_rel) == 1) {
    dispatcher->retirement_callback_cv_.notify_all();
  }
}

void ExpertDispatcher::RetirementFunc() {
  ExpertRetireArgs args;
  while (retirement_queue_.Pop(args)) {
    try {
      if (args.evict) {
        args.expert_node->node->SetDevice(
            args.expert_node->node->default_host, true, nullptr);
        gpu_overload_[args.gpu_id].store(false, std::memory_order_release);
      }
      args.expert_node->node->exec_state.store(
          NodeExecState::IDLE, std::memory_order_release);
      cache_cv_[args.gpu_id].notify_all();
    } catch (...) {
      FailDispatch(
          args.generation,
          std::current_exception(),
          {FailureContext{args.expert_node, args.gpu_id, false, false,
                          args.evict}});
    }
  }
}
```

The destructor waits with predicate `pending_retirement_callbacks_.load(std::memory_order_acquire) == 0`, closes `retirement_queue_`, and joins `ExpertRetirementFunc` before deleting modules. The callback catches all exceptions, invokes `FailDispatch` to restore state and notify waiters, decrements its counter, and never escapes across CUDA's C callback boundary.

- [ ] **Step 7: Run synchronization, correctness, and Nsight source gates**

```bash
CUTLASS_DIR="$HOME/cutlass" pip install --no-build-isolation -e .
pytest -q tests/python/unit/test_gpu_routing_source_contract.py
CUDA_VISIBLE_DEVICES=0 timeout 60s pytest -q \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py \
  -k 'non_default_caller_stream or native_and_eager or routing_failure or worker_and_handoff or failed_generation or retirement_launch_failure_retires'
```

Expected: source guards pass; native/eager output remains within `1e-2`; non-default-stream dependent work passes without a default-stream synchronize; callback failures do not escape or hang; the stale generation is quarantined; and `completion_events_outstanding` reaches zero after the retirement-launch fault without starting another dispatch.

- [ ] **Step 8: Commit synchronization removal as its own rollback unit**

```bash
git add core/utils/cuda_utils.h \
  core/parallel/expert_dispatcher.h \
  core/parallel/expert_dispatcher.cpp \
  core/parallel/expert_module.cpp \
  tests/python/unit/test_gpu_routing_source_contract.py \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py
git commit -m "perf: remove expert worker stream barriers"
```

Do not enable `gpu_only_expert_routing` in any benchmark until this task passes. If expert/cache retirement cannot be made stream-safe without reintroducing a model-thread or worker-thread synchronize, retain the eager default, report the barrier reduction as unshipped, and split retirement into a prerequisite change rather than weakening the acceptance criteria.

### Task 4: Wire native routing into the Python executor with exact fallback semantics

**Files:**
- Create: `tests/python/unit/test_gpu_only_expert_routing.py`
- Modify: `moe_infinity/distributed/expert_executor.py:187-291`

- [ ] **Step 1: Write CPU-only fake-dispatcher tests first**

Create `tests/python/unit/test_gpu_only_expert_routing.py` with these complete fakes and tests:

```python
from types import SimpleNamespace
from unittest.mock import patch

import torch

from moe_infinity.distributed.expert_executor import DistributedExpertExecutor


class FakeDispatcher:
    def __init__(self):
        self.calls = []
        self.active = [0, 2]

    def set_inputs(self, hidden, mask, weights):
        self.calls.append(("set_inputs", hidden, mask, weights))

    def set_expected_queue(self, count):
        self.calls.append(("set_expected_queue", count))

    def enqueue_expert(self, layer, expert, gpu, remote):
        self.calls.append(("enqueue_expert", layer, expert, gpu, remote))

    def notify_fetch_start(self):
        self.calls.append(("notify_fetch_start",))

    def dispatch_experts(self, layer):
        self.calls.append(("dispatch_experts", layer))

    def wait_expert(self):
        self.calls.append(("wait_expert",))
        hidden = self.calls[0][1]
        return torch.zeros_like(hidden, dtype=torch.float32)

    def take_last_active_experts(self):
        self.calls.append(("take_last_active_experts",))
        return list(self.active)

    def get_routing_stats(self):
        return {
            "route_batches": 1,
            "route_failures": 0,
            "last_active_experts": len(self.active),
            "last_route_handoff_us": 1,
            "completion_events_retired": 2,
        }


class FakePrefetcher:
    def __init__(self):
        self.corrected = []
        self.speculative = []

    def correct_prefetch(self, layer, experts):
        self.corrected.append((layer, experts))

    def speculative_prefetch(self, layer, router_logits):
        self.speculative.append((layer, router_logits))


def make_executor(enabled=True):
    config = SimpleNamespace(
        gpu_only_expert_routing=enabled,
        speculative_prefetch_overlap=False,
    )
    executor = DistributedExpertExecutor(config)
    dispatcher = FakeDispatcher()
    executor.set_expert_dispatcher(dispatcher)
    return executor, dispatcher


def test_cpu_mask_uses_eager_fallback_in_ascending_order():
    executor, dispatcher = make_executor(enabled=True)
    hidden = torch.ones(2, 4)
    mask = torch.tensor([[True, False, True], [False, False, True]])
    weights = mask.float()

    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(3, hidden, mask, weights)

    assert [call[0] for call in dispatcher.calls] == [
        "set_inputs",
        "set_expected_queue",
        "enqueue_expert",
        "enqueue_expert",
        "notify_fetch_start",
    ]
    enqueued = [call[2] for call in dispatcher.calls if call[0] == "enqueue_expert"]
    assert enqueued == [0, 2]


def test_disabled_flag_uses_eager_fallback():
    executor, dispatcher = make_executor(enabled=False)
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[False, True]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(0, hidden, mask, mask.float())
    assert not any(call[0] == "dispatch_experts" for call in dispatcher.calls)


def test_native_active_list_drives_existing_prefetch_correction():
    executor, dispatcher = make_executor(enabled=True)
    prefetcher = FakePrefetcher()
    executor._last_dispatch_used_native_routing = True
    executor._pending_prefetch = (prefetcher, 7, None, None)

    result = executor.wait_dispatch_local()

    assert result.dtype == torch.float32
    assert prefetcher.corrected == [(8, [0, 2])]
    assert [call[0] for call in dispatcher.calls] == [
        "wait_expert",
        "take_last_active_experts",
    ]


def test_missing_native_binding_uses_eager_fallback():
    executor, dispatcher = make_executor(enabled=True)
    delattr(FakeDispatcher, "dispatch_experts")
    try:
        hidden = torch.ones(1, 2)
        mask = torch.tensor([[True, False]])
        with patch("torch.cuda.device_count", return_value=1):
            executor.dispatch_local(0, hidden, mask, mask.float())
        assert any(call[0] == "enqueue_expert" for call in dispatcher.calls)
    finally:
        FakeDispatcher.dispatch_experts = lambda self, layer: self.calls.append(
            ("dispatch_experts", layer)
        )


def test_synchronous_native_submission_error_is_not_replayed_eagerly():
    executor, dispatcher = make_executor(enabled=True)
    executor._can_use_gpu_only_routing = lambda mask: True
    dispatcher.dispatch_experts = lambda layer: (_ for _ in ()).throw(
        RuntimeError("DispatchExperts: invalid mask")
    )
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[True, False]])
    with (
        patch("torch.cuda.device_count", return_value=1),
        pytest.raises(RuntimeError, match="DispatchExperts: invalid mask"),
    ):
        executor.dispatch_local(0, hidden, mask, mask.float())
    assert not any(call[0] == "enqueue_expert" for call in dispatcher.calls)
```

- [ ] **Step 2: Run the CPU-only tests and verify RED**

Run:

```bash
pytest -q tests/python/unit/test_gpu_only_expert_routing.py
```

Expected: the two eager tests pass against legacy behavior; `test_native_active_list_drives_existing_prefetch_correction` fails because `None` is passed to `correct_prefetch`.

- [ ] **Step 3: Add explicit native-path eligibility and route-ahead fallback**

Add this method to `DistributedExpertExecutor`:

```python
    def _can_use_gpu_only_routing(self, router_mask) -> bool:
        if not self._gpu_only_expert_routing:
            return False
        if not torch.is_tensor(router_mask) or not router_mask.is_cuda:
            return False
        if not hasattr(self.expert_dispatcher, "dispatch_experts"):
            return False
        route_ahead_ctx, _ = _load_route_ahead_impl()
        if route_ahead_ctx.is_active():
            return False
        return True
```

The route-ahead check is the independent-landing boundary: it preserves current pin-before-enqueue behavior and avoids moving that synchronization into this change.

- [ ] **Step 4: Split `dispatch_local` into native and eager submission without changing routing math**

Add this eager helper for pre-submission ineligibility only:

```python
    def _dispatch_eager_local(self, layer_id, router_mask, num_expert):
        expert_count = (
            torch.sum(router_mask.view((-1, num_expert)), dim=0)
            .cpu()
            .numpy()
            .flatten()
        )
        expert_list = (
            np.arange(num_expert).astype(int)[expert_count > 0].tolist()
        )
        self.expert_dispatcher.set_expected_queue(len(expert_list))
        total_gpus = torch.cuda.device_count()
        for expert_id in expert_list:
            self.expert_dispatcher.enqueue_expert(
                layer_id, expert_id, expert_id % total_gpus, False
            )
        self.expert_dispatcher.notify_fetch_start()
        return expert_list
```

Replace the active-list and enqueue portion with:

```python
        num_expert = router_mask.shape[-1]
        native_requested = bool(
            self._gpu_only_expert_routing
            and torch.is_tensor(router_mask)
            and router_mask.is_cuda
        )
        use_native_routing = self._can_use_gpu_only_routing(router_mask)
        expert_list = None

        self.expert_dispatcher.set_inputs(
            hidden_states, router_mask.bool(), router_weights
        )

        route_ahead_handled = self._maybe_route_ahead_prefetch(
            layer_id, router_mask, num_expert, prefetcher
        )

        if use_native_routing:
            with _nvtx_ctx("gpu_route_submit"):
                with (
                    profiler.time(
                        "gpu_route_submit", layer=layer_id, expert=-1
                    )
                    if profiler is not None
                    else nullcontext()
                ):
                    self.expert_dispatcher.dispatch_experts(layer_id)
        else:
            if native_requested:
                self._gpu_route_fallback_count += 1
            with _nvtx_ctx("gpu_route_fallback"):
                with (
                    profiler.time(
                        "gpu_route_fallback", layer=layer_id, expert=-1
                    )
                    if profiler is not None
                    else nullcontext()
                ):
                    expert_list = self._dispatch_eager_local(
                        layer_id, router_mask, num_expert
                    )

        self._last_dispatch_used_native_routing = use_native_routing
```

Add the benchmark-facing read-only accessor:

```python
    def get_gpu_routing_stats(self):
        stats = {
            "route_batches": 0,
            "route_failures": 0,
            "last_active_experts": 0,
            "last_route_handoff_us": 0,
            "completion_events_retired": 0,
        }
        getter = getattr(self.expert_dispatcher, "get_routing_stats", None)
        if getter is not None:
            stats.update({key: int(value) for key, value in getter().items()})
        stats["fallback_count"] = int(self._gpu_route_fallback_count)
        return stats
```

The shown `_maybe_route_ahead_prefetch()` call remains before every eager enqueue exactly as today. In the native branch it returns `False` because `_can_use_gpu_only_routing()` excluded an active route-ahead context. Retain speculative-prefetch trigger ordering. Store `(prefetcher, layer_id, expert_list, pending_router_logits)` in `_pending_prefetch`; `expert_list` is `None` only for successful native routing.

- [ ] **Step 5: Resolve the host-ready active list only after native wait**

Replace the single `wait_expert()` call inside the existing `sync_wait` context with a nested completion-handoff timer:

```python
                completion_profiler_ctx = (
                    profiler.time("expert_completion_handoff", expert=-1)
                    if profiler is not None
                    else nullcontext()
                )
                with completion_profiler_ctx:
                    result = self.expert_dispatcher.wait_expert()
```

Change the pending-prefetch block in `wait_dispatch_local()` to:

```python
        pending = getattr(self, "_pending_prefetch", None)
        if pending is not None:
            prefetcher, layer_id, expert_list, router_logits = pending
            self._pending_prefetch = None
            if expert_list is None and self._last_dispatch_used_native_routing:
                expert_list = list(
                    self.expert_dispatcher.take_last_active_experts()
                )
            if prefetcher is not None:
                prefetcher.correct_prefetch(layer_id + 1, expert_list)
            if router_logits is not None:
                self.trigger_speculative_prefetch(layer_id, router_logits)
```

Keep the outer `expert_wait_barrier` and `sync_wait` names for backward-compatible dashboards; native NVTX adds `expert_completion_handoff` inside it.

- [ ] **Step 6: Run CPU orchestration and existing route-ahead tests**

Run:

```bash
pytest -q tests/python/unit/test_gpu_only_expert_routing.py \
  tests/python/dflash/test_route_ahead_wire.py \
  tests/python/dflash/test_route_ahead_metrics.py
```

Expected: all tests pass; route-ahead still observes the same pinned/enqueued IDs and call order.

- [ ] **Step 7: Commit Python native routing selection**

```bash
git add moe_infinity/distributed/expert_executor.py \
  tests/python/unit/test_gpu_only_expert_routing.py
git commit -m "feat: select native expert routing with eager fallback"
```

### Task 5: Prove routing and output semantics on CUDA and representative adapters

**Files:**
- Modify: `tests/python/unit/test_gpt_oss_mxfp4_dispatch.py`
- Modify: `tests/python/unit/test_gpu_only_expert_routing.py`
- Modify: `benchmarks/serving/validate_batched_dispatch.py:18-183`

- [ ] **Step 1: Add a native-versus-eager CUDA parity matrix**

Add this test beside the real-extension helper. Its seeded fixture initializes the same registered weights for each call:

```python
@pytest.mark.parametrize(
    "active_experts",
    [(0,), (1,), (0, 1), ()],
)
@pytest.mark.gpu
def test_native_and_eager_dispatch_are_output_identical(tmp_path, active_experts):
    eager_output, _, _, _, _ = _native_dispatch(
        tmp_path / "eager",
        torch.bfloat16,
        active_experts=active_experts,
        gpu_routing=False,
    )
    native_output, _, _, _, native = _native_dispatch(
        tmp_path / "native",
        torch.bfloat16,
        active_experts=active_experts,
        gpu_routing=True,
    )
    assert native.take_last_active_experts() == sorted(active_experts)
    torch.testing.assert_close(
        native_output, eager_output, rtol=1e-2, atol=1e-2
    )
```

- [ ] **Step 2: Add a no-synchronizing-call source guard**

Extend `analyze_dispatch_interface_payload()` in `benchmarks/serving/validate_batched_dispatch.py` to inspect the native branch between `if use_native_routing:` and its `else:` and report:

```python
        "native_branch_has_blocking_host_extract": any(
            needle in native_branch
            for needle in (".cpu(", ".numpy(", ".item(", ".tolist(")
        ),
        "native_dispatch_binding_line": _find_line(
            expert_executor_text,
            "self.expert_dispatcher.dispatch_experts(layer_id)",
        ),
        "native_take_active_line": _find_line(
            expert_executor_text,
            "self.expert_dispatcher.take_last_active_experts()",
        ),
```

Raise `RuntimeError` if the blocking-host-extract value is true or either line is missing. Keep the existing shape and adapter markers.

- [ ] **Step 3: Add representative adapter contract assertions**

Add to `tests/python/unit/test_gpu_only_expert_routing.py`:

```python
def test_representative_adapters_keep_dispatch_wait_contract():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    for relative in (
        "moe_infinity/models/qwen.py",
        "moe_infinity/models/mixtral.py",
        "moe_infinity/models/deepseek.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        dispatch_at = source.index("dispatch_local(")
        wait_at = source.index("wait_dispatch_local()", dispatch_at)
        assert dispatch_at < wait_at
        assert "router_mask" in source[dispatch_at:wait_at]
```

This is deliberately a boundary test: adapters must not gain device-to-host routing logic or a model-specific native API.

- [ ] **Step 4: Run parity and contract tests**

Run:

```bash
pytest -q tests/python/unit/test_gpu_only_expert_routing.py \
  tests/python/unit/test_multi_gpu.py -k 'gpu_only or dispatch_local'
CUDA_VISIBLE_DEVICES=0 pytest -q \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py \
  -k 'native_and_eager or native_gpu_routing'
python benchmarks/serving/validate_batched_dispatch.py
```

Expected: CPU tests and CUDA parity tests pass; validator JSON reports `native_branch_has_blocking_host_extract: false`.

- [ ] **Step 5: Run the two-GPU ownership case when two GPUs are available**

Add a CUDA test using the registered two-expert fixture; experts 0 and 1 must execute on devices 0 and 1 under the unchanged modulo rule:

```python
@pytest.mark.multi_gpu
def test_native_routing_preserves_round_robin_gpu_ownership(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    actual, expected, _, _, dispatcher = _native_dispatch(
        tmp_path,
        torch.bfloat16,
        active_experts=(0, 1),
        gpu_routing=True,
    )
    assert dispatcher.take_last_active_experts() == [0, 1]
    torch.testing.assert_close(
        actual, expected, rtol=1e-2, atol=1e-2
    )
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 pytest -q \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py \
  -m multi_gpu -k round_robin_gpu_ownership
```

Expected: pass on a two-GPU single host; skip, not fail, when fewer than two devices are visible. Validate output parity in the same test; the active-list assertion alone is not sufficient.

- [ ] **Step 6: Test independent deferred-prefetch ordering**

Add to `tests/python/unit/test_gpu_only_expert_routing.py`:

```python
def test_gpu_routing_exposes_host_list_only_after_wait():
    executor, dispatcher = make_executor(enabled=True)
    executor._can_use_gpu_only_routing = lambda mask: True
    prefetcher = FakePrefetcher()
    executor.set_prefetcher(prefetcher)
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[3.0, 1.0, 2.0]])

    executor.dispatch_local(
        4, hidden, mask, mask.float(), router_logits=logits
    )

    assert prefetcher.speculative == []
    assert not any(
        call[0] == "take_last_active_experts" for call in dispatcher.calls
    )
    _ = executor.wait_dispatch_local()
    assert prefetcher.corrected == [(5, [0, 2])]
    assert prefetcher.speculative == [(4, logits)]
    assert dispatcher.calls[-1][0] == "take_last_active_experts"
```

Run:

```bash
pytest -q tests/python/unit/test_gpu_only_expert_routing.py \
  -k gpu_routing_exposes_host_list_only_after_wait
```

Expected: GPU routing runs independently with overlap disabled; neither deferred speculative prefetch nor the host-ready list is consumed before `WaitHiddenStates()` closes `route_pending_`; correction uses the exact sorted list and deferred logits prefetch fires afterward. Keep the existing DFlash route-ahead tests in eager fallback mode.

- [ ] **Step 7: Commit semantic parity coverage**

```bash
git add tests/python/unit/test_gpt_oss_mxfp4_dispatch.py \
  tests/python/unit/test_gpu_only_expert_routing.py \
  benchmarks/serving/validate_batched_dispatch.py
git commit -m "test: prove gpu routing matches eager semantics"
```

### Task 6: Add observability for routing handoff and the reduced barrier

**Files:**
- Create: `tests/python/unit/test_gpu_routing_benchmarks.py`
- Modify: `benchmarks/expert_io_microbench/bench_routing.py`
- Modify: `benchmarks/expert_io_microbench/bench_bubble.py`
- Modify: `benchmarks/expert_io_microbench/nsys_parser.py`

- [ ] **Step 1: Write failing CPU-only summary tests**

Create `tests/python/unit/test_gpu_routing_benchmarks.py`:

```python
from benchmarks.expert_io_microbench.bench_bubble import summarize_iterations
from benchmarks.expert_io_microbench.bench_routing import summarize_gpu_routing


def test_gpu_routing_summary_exposes_submit_fallback_and_native_stats():
    events = [
        {"stage": "gpu_route_submit", "dur_ns": 100},
        {"stage": "gpu_route_submit", "dur_ns": 300},
        {"stage": "gpu_route_fallback", "dur_ns": 900},
    ]
    summary = summarize_gpu_routing(
        events,
        {
            "route_batches": 2,
            "route_failures": 0,
            "last_active_experts": 4,
            "last_route_handoff_us": 7,
            "completion_events_retired": 8,
        },
    )
    assert summary["submit_p50_ns"] == 200.0
    assert summary["fallback_count"] == 1
    assert summary["native"]["route_failures"] == 0


def test_bubble_summary_splits_route_and_completion_handoff():
    summary = summarize_iterations(
        [
            {
                "step_total_ns": 1000,
                "expert_wait_ns": 300,
                "route_submit_ns": 40,
                "completion_handoff_ns": 60,
                "bubble_ratio": 0.3,
                "layer_wait_ns": {},
                "sync_event_count": 1,
                "io_profiler_event_count": 3,
            }
        ]
    )
    assert summary["step_decomposition_mean"]["route_submit_ns"] == 40.0
    assert (
        summary["step_decomposition_mean"]["completion_handoff_ns"] == 60.0
    )
```

- [ ] **Step 2: Run summary tests and verify RED**

Run:

```bash
pytest -q tests/python/unit/test_gpu_routing_benchmarks.py
```

Expected: import failure for `summarize_gpu_routing` or missing summary keys.

- [ ] **Step 3: Implement routing and barrier summaries**

Add `summarize_gpu_routing(events, native_stats)` to `bench_routing.py`. Filter by exact stage names, use the existing percentile helper, and return:

```python
{
    "submit_p50_ns": _percentile(submit_values, 50),
    "submit_p99_ns": _percentile(submit_values, 99),
    "fallback_count": len(fallback_values),
    "fallback_p50_ns": _percentile(fallback_values, 50),
    "native": {key: int(native_stats.get(key, 0)) for key in (
        "route_batches",
        "route_failures",
        "last_active_experts",
        "last_route_handoff_us",
        "completion_events_retired",
    )},
}
```

When collecting a real run, obtain the dispatcher through `model.engine.expert_dispatcher` as existing benchmark code does and call `get_routing_stats()` only when the method exists; older builds return a zeroed native dictionary and continue.

In `bench_bubble.py`, sum `gpu_route_submit` plus `gpu_route_fallback` events into `route_submit_ns`, sum `expert_completion_handoff` into `completion_handoff_ns`, and keep the existing `sync_wait` total. Add both component fields to per-iteration and p50/p99 summaries. Keep `expert_wait_ns` unchanged for dashboard compatibility; native background-thread `last_route_handoff_us` remains in `bench_routing.py` because it cannot be attributed to a Python iteration from `IOProfiler` alone.

- [ ] **Step 4: Extend Nsight parsing without changing existing fields**

Add these exact range names to `REQUIRED_RANGE_NAMES` in `nsys_parser.py:36-43`:

```python
"gpu_route_submit",
"gpu_route_handoff",
"gpu_route_fallback",
"expert_completion_handoff",
```

Add `CUDA_API_REPORT = "cuda_api_sum"`, query it in `parse_nsys_report()`, and return:

```python
"cuda_api": {
    "stream_synchronize_count": sum(
        int(row.get("Instances", 0) or row.get("Count", 0) or 0)
        for row in cuda_api_rows
        if "cudaStreamSynchronize" in str(
            row.get("Name") or row.get("Operation") or ""
        )
    ),
    "device_synchronize_count": sum(
        int(row.get("Instances", 0) or row.get("Count", 0) or 0)
        for row in cuda_api_rows
        if "cudaDeviceSynchronize" in str(
            row.get("Name") or row.get("Operation") or ""
        )
    ),
},
```

Add a `routing_sync` object to parsed output:

```python
"routing_sync": {
    "gpu_route_submit_ns": _range_total("gpu_route_submit"),
    "gpu_route_handoff_ns": _range_total("gpu_route_handoff"),
    "gpu_route_fallback_ns": _range_total("gpu_route_fallback"),
    "expert_completion_handoff_ns": _range_total(
        "expert_completion_handoff"
    ),
    "device_to_host_memcpy_count": d2h_count,
    "device_to_host_memcpy_bytes": d2h_bytes,
    "stream_synchronize_count": int(
        report["cuda_api"]["stream_synchronize_count"]
    ),
    "device_synchronize_count": int(
        report["cuda_api"]["device_synchronize_count"]
    ),
},
```

Read `d2h_count`/`d2h_bytes` from the existing `memcpy` summary built from `cuda_gpu_mem_size_sum` rows whose operation is `Device-to-Host`. The expected optimized trace has one asynchronous `num_experts`-byte activity-bitmap copy per native-routed layer and no blocking Python-side tensor copy. Do not claim zero D2H bytes.

- [ ] **Step 5: Run summary/parser tests and verify GREEN**

Add this synthetic parser-summary test:

```python
def test_nsys_summary_reports_activity_bitmap_d2h(monkeypatch):
    from benchmarks.expert_io_microbench import nsys_parser

    monkeypatch.setattr(
        nsys_parser,
        "parse_nsys_report",
        lambda _path: {
            "ranges": {
                "gpu_route_submit": {
                    "total_ns": 100,
                    "count": 1,
                    "mean_ns": 100,
                    "p50_ns": 100,
                }
            },
            "memcpy": {
                "h2d_bytes": 0,
                "h2d_count": 0,
                "d2h_bytes": 8,
                "d2h_count": 1,
                "d2d_bytes": 0,
                "d2d_count": 0,
            },
            "gpu_memcpy_ns": {"h2d": 0, "d2h": 10, "d2d": 0},
            "cuda_api": {
                "stream_synchronize_count": 0,
                "device_synchronize_count": 0,
            },
            "duration_ns": 100,
        },
    )
    summary = nsys_parser.summarise(
        "unused.nsys-rep", 1, {"link_width": 16, "link_gen": 4}
    )
    assert summary["routing_sync"]["device_to_host_memcpy_count"] == 1
    assert summary["routing_sync"]["device_to_host_memcpy_bytes"] == 8
    assert summary["routing_sync"]["stream_synchronize_count"] == 0
```

Then run:

```bash
pytest -q tests/python/unit/test_gpu_routing_benchmarks.py \
  tests/python/unit -k 'gpu_routing or bubble or nsys'
```

Expected: all selected CPU-only tests pass.

- [ ] **Step 6: Commit observability**

```bash
git add benchmarks/expert_io_microbench/bench_routing.py \
  benchmarks/expert_io_microbench/bench_bubble.py \
  benchmarks/expert_io_microbench/nsys_parser.py \
  tests/python/unit/test_gpu_routing_benchmarks.py
git commit -m "feat: expose gpu routing handoff metrics"
```

### Task 7: Add TPOT A/B benchmarking and objective rollback criteria

**Files:**
- Modify: `tests/python/unit/test_gpu_routing_benchmarks.py`
- Modify: `tests/python/unit/test_gpu_routing_source_contract.py`
- Modify: `benchmarks/serving/latency.py:20-65,120-145,196-280,405-436`
- Modify: `benchmarks/expert_io_microbench/run_decision_profile.py:24-38,65-92,109-178`
- Modify: `benchmarks/expert_io_microbench/nsys_parser.py:36-43,217-308`
- Modify: `docs/benchmarking.md:286-303`

- [ ] **Step 1: Write failing TPOT and verdict tests**

Append:

```python
from benchmarks.expert_io_microbench.run_decision_profile import (
    build_model_config as build_profile_model_config,
    build_profile_payload,
    parse_args as parse_profile_args,
)
from benchmarks.expert_io_microbench.nsys_parser import parse_cli_args
from benchmarks.serving.latency import (
    build_model_config as build_latency_model_config,
    build_result_payload,
    gpu_routing_verdict,
    load_routing_baseline,
    parse_args as parse_latency_args,
    run_sweep,
)


def test_gpu_routing_verdict_rolls_back_p99_regression():
    verdict = gpu_routing_verdict(
        baseline={"tpot_p50_ms": 10.0, "tpot_p99_ms": 20.0},
        candidate={"tpot_p50_ms": 9.5, "tpot_p99_ms": 21.1},
        route_failures=0,
        fallback_count=0,
    )
    assert verdict["decision"] == "ROLLBACK"
    assert "tpot_p99_regression_gt_5pct" in verdict["reasons"]


def test_gpu_routing_verdict_accepts_non_regressing_candidate():
    verdict = gpu_routing_verdict(
        baseline={"tpot_p50_ms": 10.0, "tpot_p99_ms": 20.0},
        candidate={"tpot_p50_ms": 9.0, "tpot_p99_ms": 19.0},
        route_failures=0,
        fallback_count=0,
    )
    assert verdict == {"decision": "KEEP", "reasons": []}


def test_latency_cli_wires_mode_warmups_and_paths():
    args = parse_latency_args(
        [
            "--model", "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "--offload-dir", "/tmp/moe-routing-store",
            "--gpu-only-expert-routing", "on",
            "--warmup-rounds", "3",
            "--routing-baseline-json", "/tmp/routing-off.json",
            "--output-json", "/tmp/routing-on.json",
        ]
    )
    assert args.gpu_only_expert_routing == "on"
    assert args.warmup_rounds == 3
    assert build_latency_model_config(args)["gpu_only_expert_routing"] is True
    assert build_latency_model_config(args)["speculative_prefetch_overlap"] is False


def test_decision_profile_cli_wires_mode_and_warmup_iterations():
    args = parse_profile_args(
        [
            "--model", "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "--offload-dir", "/tmp/moe-routing-store",
            "--hardware-tag", "single-host",
            "--mode", "host-only",
            "--gpu-only-expert-routing", "on",
            "--warmup-iters", "3",
            "--output-json", "/tmp/routing-profile.json",
        ]
    )
    assert args.gpu_only_expert_routing == "on"
    assert args.warmup_iters == 3
    assert build_profile_model_config(args)["gpu_only_expert_routing"] is True
    assert build_profile_model_config(args)["speculative_prefetch_overlap"] is False


def test_decision_profile_rejects_gpu_routing_with_overlap():
    args = parse_profile_args(
        [
            "--model", "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "--offload-dir", "/tmp/moe-routing-store",
            "--hardware-tag", "single-host",
            "--mode", "host-only",
            "--gpu-only-expert-routing", "on",
            "--speculative-prefetch-overlap",
            "--output-json", "/tmp/routing-profile.json",
        ]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        build_profile_model_config(args)


def test_nsys_cli_accepts_profile_schema_input():
    args = parse_cli_args(
        [
            "/tmp/gpu-routing-on.nsys-rep",
            "--steps", "96",
            "--profile-json", "/tmp/gpu-routing-profile.json",
        ]
    )
    assert args.steps == 96
    assert args.profile_json == "/tmp/gpu-routing-profile.json"


def test_decision_profile_result_schema():
    from types import SimpleNamespace

    payload = build_profile_payload(
        args=SimpleNamespace(
            gpu_only_expert_routing="on",
            warmup_iters=3,
            warmup_tokens=8,
            iters=3,
            max_new_tokens=32,
        ),
        decode_step_times_ns=[100, 110, 120],
        routing={
            "route_batches": 30,
            "route_failures": 0,
            "fallback_count": 0,
            "completion_events_retired": 60,
        },
        pcie={
            "link_width_pre": 16,
            "link_gen_pre": 4,
            "link_width_post": 16,
            "link_gen_post": 4,
        },
    )
    assert payload["schema_version"] == "gpu-routing-decision-profile-v1"
    assert payload["measurement"]["decode_step_count"] == 96
    assert payload["routing"]["route_failures"] == 0


def test_latency_result_schema_contains_tpot_samples_and_verdict():
    payload = build_latency_payload_for_test()
    assert payload["schema_version"] == "gpu-routing-latency-v1"
    assert payload["measurement"]["1"]["tpot_p50_ms"] == 9.0
    assert payload["measurement"]["1"]["tpot_p99_ms"] == 19.0
    assert payload["routing"]["route_failures"] == 0
    assert payload["verdict"]["decision"] == "KEEP"


def build_latency_payload_for_test():
    from types import SimpleNamespace

    args = SimpleNamespace(
        model="deepseek-ai/DeepSeek-V2-Lite-Chat",
        offload_dir="/tmp/moe-routing-store",
        gpu_only_expert_routing="on",
        concurrency=[1],
        prompt_length=128,
        max_new_tokens=64,
        warmup_rounds=3,
        num_rounds=30,
    )
    baseline = {"1": {"tpot_p50_ms": 10.0, "tpot_p99_ms": 20.0}}
    candidate = {
        "1": {
            "sample_count": 30,
            "ttft_p50_ms": 100.0,
            "ttft_p99_ms": 120.0,
            "tpot_p50_ms": 9.0,
            "tpot_p99_ms": 19.0,
            "itl_p50_ms": 9.0,
            "itl_p99_ms": 19.0,
        }
    }
    return build_result_payload(
        args=args,
        measurements=candidate,
        baseline_measurements=baseline,
        routing={
            "route_batches": 10,
            "route_failures": 0,
            "fallback_count": 0,
            "completion_events_retired": 20,
        },
    )


def test_profiled_loop_does_not_inject_stream_synchronize():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    source = (
        root
        / "benchmarks/expert_io_microbench/run_decision_profile.py"
    ).read_text(encoding="utf-8")
    measured = source[
        source.index("decode_step_times_ns: list[int]") :
        source.index("cudaProfilerStop()")
    ]
    assert "torch.cuda.synchronize()" not in measured


def test_latency_warmups_are_excluded(monkeypatch):
    from benchmarks.serving import latency

    samples = iter([1000.0, 1001.0, 10.0, 20.0])

    def fake_round(*args, **kwargs):
        value = next(samples)
        return [value], [value]

    monkeypatch.setattr(latency, "run_one_round", fake_round)
    result = run_sweep(
        object(),
        object(),
        concurrency_levels=[1],
        warmup_rounds=2,
        num_rounds=2,
        prompt_length=128,
        max_new_tokens=64,
    )
    assert result["1"]["sample_count"] == 2
    assert result["1"]["tpot_p50_ms"] == 15.0


def test_routing_baseline_rejects_wrong_schema(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text('{"schema_version":"baseline-performance-v1"}')
    with pytest.raises(ValueError, match="gpu-routing-latency-v1"):
        load_routing_baseline(path, expected_config={})
```

- [ ] **Step 2: Run verdict tests and verify RED**

Run:

```bash
pytest -q tests/python/unit/test_gpu_routing_benchmarks.py \
  -k gpu_routing_verdict
```

Expected: import failures for `gpu_routing_verdict`, both `build_model_config` functions, argv-aware parsers, and `build_result_payload`.

- [ ] **Step 3: Add explicit mode, warmup, TPOT fields, and verdict**

Change both scripts to `def parse_args(argv: list[str] | None = None)` and return `parser.parse_args(argv)`. Add these latency arguments while preserving the current `--baseline-json` compatibility path:

```python
    parser.add_argument(
        "--gpu-only-expert-routing",
        choices=("off", "on"),
        default="off",
    )
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument(
        "--routing-baseline-json",
        default=None,
        help="gpu-routing-latency-v1 JSON produced by an off-mode run",
    )
```

Add and test this latency config builder; change `load_model_and_tokenizer` to accept the built config rather than reconstructing it:

```python
def build_model_config(args):
    return {
        "offload_path": args.offload_dir,
        "device_memory_ratio": 0.75,
        "gpu_only_expert_routing": args.gpu_only_expert_routing == "on",
        "speculative_prefetch_overlap": False,
    }
```

Rename local `itl_ms` to `tpot_ms` and emit both names for compatibility:

```python
            "tpot_p50_ms": percentile(all_tpot, 50.0),
            "tpot_p90_ms": percentile(all_tpot, 90.0),
            "tpot_p99_ms": percentile(all_tpot, 99.0),
            "itl_p50_ms": percentile(all_tpot, 50.0),
            "itl_p90_ms": percentile(all_tpot, 90.0),
            "itl_p99_ms": percentile(all_tpot, 99.0),
```

Add `warmup_rounds: int` to `run_sweep` and place this before measured lists/rounds for each concurrency; reject negative warmups and nonpositive measured rounds in `main()`:

```python
        for _ in range(warmup_rounds):
            _ = run_one_round(
                model,
                tokenizer,
                concurrency=concurrency,
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens,
            )
        all_ttft: list[float] = []
        all_tpot: list[float] = []
```

Add the following exact result contract. `build_result_payload()` is the sole producer, and `load_routing_baseline()` rejects any baseline whose `schema_version` is not `gpu-routing-latency-v1` or whose config differs in model, prompt length, max-new-tokens, concurrency, and device-memory ratio:

```python
def build_result_payload(
    *, args, measurements, baseline_measurements, routing
):
    comparison = {}
    reasons = []
    if baseline_measurements is not None:
        for level, candidate in measurements.items():
            baseline = baseline_measurements[level]
            comparison[level] = {
                "tpot_p50_delta_pct": 100.0
                * (candidate["tpot_p50_ms"] / baseline["tpot_p50_ms"] - 1.0),
                "tpot_p99_delta_pct": 100.0
                * (candidate["tpot_p99_ms"] / baseline["tpot_p99_ms"] - 1.0),
            }
            level_verdict = gpu_routing_verdict(
                baseline,
                candidate,
                routing["route_failures"],
                routing["fallback_count"],
            )
            reasons.extend(f"concurrency_{level}:{reason}"
                           for reason in level_verdict["reasons"])
        decision = "ROLLBACK" if reasons else "KEEP"
    else:
        decision = "BASELINE"
    return {
        "schema_version": "gpu-routing-latency-v1",
        "status": "PASS",
        "config": {
            "model": args.model,
            "offload_dir": args.offload_dir,
            "gpu_only_expert_routing": args.gpu_only_expert_routing,
            "device_memory_ratio": 0.75,
            "concurrency": list(args.concurrency),
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "warmup_rounds": args.warmup_rounds,
            "num_rounds": args.num_rounds,
        },
        "measurement": measurements,
        "routing": routing,
        "comparison": comparison,
        "verdict": {"decision": decision, "reasons": reasons},
    }
```

```python
{
    "schema_version": "gpu-routing-latency-v1",
    "status": "PASS",
    "config": {
        "model": str,
        "offload_dir": str,
        "gpu_only_expert_routing": "off" | "on",
        "speculative_prefetch_overlap": False,
        "device_memory_ratio": float,
        "concurrency": list[int],
        "prompt_length": int,
        "max_new_tokens": int,
        "warmup_rounds": int,
        "num_rounds": int,
    },
    "measurement": {
        "1": {
            "sample_count": int,
            "ttft_p50_ms": float,
            "ttft_p99_ms": float,
            "tpot_p50_ms": float,
            "tpot_p99_ms": float,
            "itl_p50_ms": float,
            "itl_p99_ms": float,
        }
    },
    "routing": {
        "route_batches": int,
        "route_failures": int,
        "fallback_count": int,
        "completion_events_retired": int,
    },
    "comparison": {"1": {"tpot_p50_delta_pct": float,
                           "tpot_p99_delta_pct": float}},
    "verdict": {"decision": "KEEP" | "ROLLBACK" | "BASELINE",
                "reasons": list[str]},
}
```

For off-mode runs without `--routing-baseline-json`, emit `BASELINE`. For on-mode runs, require `--routing-baseline-json`; absence or schema/config mismatch exits 2 before model loading. Add:

```python
def gpu_routing_verdict(baseline, candidate, route_failures, fallback_count):
    reasons = []
    base_p99 = baseline.get("tpot_p99_ms")
    candidate_p99 = candidate.get("tpot_p99_ms")
    if base_p99 and candidate_p99 and candidate_p99 > base_p99 * 1.05:
        reasons.append("tpot_p99_regression_gt_5pct")
    base_p50 = baseline.get("tpot_p50_ms")
    candidate_p50 = candidate.get("tpot_p50_ms")
    if base_p50 and candidate_p50 and candidate_p50 > base_p50 * 1.02:
        reasons.append("tpot_p50_regression_gt_2pct")
    if route_failures:
        reasons.append("native_route_failure")
    if fallback_count:
        reasons.append("unexpected_eager_fallback")
    return {
        "decision": "ROLLBACK" if reasons else "KEEP",
        "reasons": reasons,
    }
```

Collect routing statistics through `model.engine.expert_executor.get_gpu_routing_stats()` (defined in Task 4) and merge native `get_routing_stats()` with the Python fallback counter. Embed mode, routing stats, fallback count, and verdict exactly as above. Do not compare a p50 against an average.

Use this shared local helper in both benchmark scripts (duplicate the six lines rather than adding a cross-benchmark dependency):

```python
def collect_routing_stats(model):
    engine = getattr(model, "engine", None)
    executor = getattr(engine, "expert_executor", None)
    getter = getattr(executor, "get_gpu_routing_stats", None)
    if getter is None:
        return {
            "route_batches": 0,
            "route_failures": 0,
            "fallback_count": 0,
            "completion_events_retired": 0,
        }
    return {key: int(value) for key, value in getter().items()}
```

- [ ] **Step 4: Wire the decision-profile command and schema before using it**

Add these arguments to `run_decision_profile.py`:

```python
    p.add_argument(
        "--gpu-only-expert-routing", choices=("off", "on"), default="off"
    )
    p.add_argument("--warmup-iters", type=int, default=3)
```

Add this exact config builder:

```python
def build_model_config(args):
    gpu_routing = args.gpu_only_expert_routing == "on"
    ArcherConfig._validate_gpu_routing_overlap(
        gpu_routing,
        args.speculative_prefetch_overlap,
        "off",
    )
    return {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
        "speculative_prefetch": args.speculative_prefetch,
        "speculative_prefetch_overlap": args.speculative_prefetch_overlap,
        "num_threads": args.num_threads,
        "gpu_only_expert_routing": gpu_routing,
    }
```

Call `build_model_config(args)` before model loading and convert its `ValueError` into `ArgumentParser.error`, so `--gpu-only-expert-routing on --speculative-prefetch-overlap` exits 2 without constructing `MoE`.

Run this exact loop before `cudaProfilerStart`, synchronize once after all warmups, and remove `torch.cuda.synchronize()` from the profiled iteration loop at current line 142 so the decision profile does not manufacture the stream synchronization it is intended to detect:

```python
    for _ in range(args.warmup_iters):
        _ = m.generate(
            ids,
            max_new_tokens=max(args.warmup_tokens, 1),
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize()
```

Make this function the sole producer:

```python
def build_profile_payload(
    *, args, decode_step_times_ns, routing, pcie
):
    return {
        "schema_version": "gpu-routing-decision-profile-v1",
        "config": {
            "gpu_only_expert_routing": args.gpu_only_expert_routing,
            "speculative_prefetch_overlap": False,
            "warmup_iters": args.warmup_iters,
            "warmup_tokens": args.warmup_tokens,
            "iters": args.iters,
            "max_new_tokens": args.max_new_tokens,
        },
        "measurement": {
            "decode_step_times_ns": list(decode_step_times_ns),
            "decode_step_total_ns": sum(decode_step_times_ns),
            "decode_step_count": args.iters * args.max_new_tokens,
        },
        "routing": dict(routing),
        "pcie": dict(pcie),
    }
```

Emit:

```python
{
    "schema_version": "gpu-routing-decision-profile-v1",
    "config": {
        "gpu_only_expert_routing": "off" | "on",
        "speculative_prefetch_overlap": False,
        "warmup_iters": int,
        "warmup_tokens": int,
        "iters": int,
        "max_new_tokens": int,
    },
    "measurement": {
        "decode_step_times_ns": list[int],
        "decode_step_total_ns": int,
        "decode_step_count": int,
    },
    "routing": {
        "route_batches": int,
        "route_failures": int,
        "fallback_count": int,
        "completion_events_retired": int,
    },
    "pcie": {
        "link_width_pre": int,
        "link_gen_pre": int,
        "link_width_post": int,
        "link_gen_post": int,
    },
}
```

Extend `nsys_parser.summarise()` with a separate `gpu_routing_verdict` object while retaining the existing IBP `verdict` key:

```python
def gpu_routing_trace_verdict(ranges, cuda_api, profile):
    if profile is None:
        return {
            "decision": "UNAVAILABLE",
            "reasons": ["profile_json_not_provided"],
        }
    routing = profile["routing"]
    enabled = profile["config"]["gpu_only_expert_routing"] == "on"
    reasons = []
    if int(ranges.get("gpu_route_fallback", {}).get("count", 0)) > 0:
        reasons.append("gpu_route_fallback_present")
    if int(cuda_api.get("stream_synchronize_count", 0)) > 0:
        reasons.append("stream_synchronize_present")
    if int(cuda_api.get("device_synchronize_count", 0)) > 0:
        reasons.append("device_synchronize_present")
    if enabled and routing["route_batches"] > 0:
        if int(ranges.get("gpu_route_handoff", {}).get("count", 0)) == 0:
            reasons.append("missing_route_handoff")
        if int(ranges.get("expert_completion_handoff", {}).get("count", 0)) == 0:
            reasons.append("missing_completion_handoff")
    if routing["route_failures"] > 0:
        reasons.append("native_route_failure")
    if routing["fallback_count"] > 0:
        reasons.append("unexpected_eager_fallback")
    return {
        "decision": "ROLLBACK" if reasons else "KEEP",
        "reasons": reasons,
    }
```

Set `result["gpu_routing_verdict"] = gpu_routing_trace_verdict(ranges, report["cuda_api"], profile)` in `summarise()`.

Append `gpu_route_fallback_present` when that range count is nonzero, `stream_synchronize_present` or `device_synchronize_present` from `cuda_api_sum`, `missing_route_handoff` when enabled-mode metadata says route batches are nonzero but the range is absent, and `missing_completion_handoff` under the analogous condition. Pass profile JSON to the parser with new `--profile-json /tmp/gpu-routing-profile.json`; validate its schema and use its routing counts/config to evaluate those conditions.

Replace the hand-rolled `sys.argv` loop in `nsys_parser._cli()` with:

```python
def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("rep_path")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--link-width", type=int, default=16)
    parser.add_argument("--link-gen", type=int, default=4)
    parser.add_argument("--real-total-ns", type=int)
    parser.add_argument("--profile-json")
    return parser.parse_args(argv)
```

When `--profile-json` is present, require `gpu-routing-decision-profile-v1` and pass it to `summarise(..., profile=profile_payload)`. An incompatible profile returns exit code 2 before invoking `nsys stats`. When omitted, preserve the current IBP summary and emit `gpu_routing_verdict: {"decision": "UNAVAILABLE", "reasons": ["profile_json_not_provided"]}`.

- [ ] **Step 5: Run CLI, schema, warmup, and verdict tests and verify GREEN**

Run:

```bash
pytest -q tests/python/unit/test_gpu_routing_benchmarks.py \
  tests/python/unit/test_gpu_routing_source_contract.py \
  -k 'gpu_routing_verdict or latency_cli or decision_profile_cli or result_schema or profiled_loop'
```

Expected: parser tests prove every documented flag is accepted, both config builders carry the mode, warmups are excluded, schemas are exact, an incompatible baseline exits before model loading, and the profiled loop contains no explicit `torch.cuda.synchronize()`.

- [ ] **Step 6: Capture staged single-host A/B data**

Use the same checkout, checkpoint, offload tree, GPU visibility, cache ratio, prompt, output length, and greedy decoding for both modes:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/serving/latency.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /tmp/moe-infinity-bench/deepseek-v2-lite-chat \
  --concurrency 1 2 4 \
  --prompt-length 128 \
  --max-new-tokens 64 \
  --warmup-rounds 3 \
  --num-rounds 30 \
  --gpu-only-expert-routing off \
  --output-json /tmp/gpu-routing-off.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/serving/latency.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /tmp/moe-infinity-bench/deepseek-v2-lite-chat \
  --concurrency 1 2 4 \
  --prompt-length 128 \
  --max-new-tokens 64 \
  --warmup-rounds 3 \
  --num-rounds 30 \
  --gpu-only-expert-routing on \
  --routing-baseline-json /tmp/gpu-routing-off.json \
  --output-json /tmp/gpu-routing-on.json
```

Expected: both files report TPOT p50/p99 for each concurrency. The candidate verdict is `KEEP` only when routing failures and unexpected fallback count are zero, TPOT p50 regresses no more than 2%, and TPOT p99 regresses no more than 5%. No minimum speedup is promised.

- [ ] **Step 7: Capture an Nsight trace and confirm synchronization moved off the Python hot path**

```bash
MOE_INFINITY_PROFILE_IO=1 MOE_INFINITY_PROFILE_IO_SAMPLE=1.0 \
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=/tmp/gpu-routing-on \
  python benchmarks/expert_io_microbench/run_decision_profile.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /tmp/moe-infinity-bench/deepseek-v2-lite-chat \
    --hardware-tag single-host \
    --mode host-only \
    --gpu-only-expert-routing on \
    --warmup-iters 3 \
    --warmup-tokens 8 \
    --iters 3 \
    --max-new-tokens 32 \
    --output-json /tmp/gpu-routing-profile.json

python benchmarks/expert_io_microbench/nsys_parser.py \
  /tmp/gpu-routing-on.nsys-rep \
  --steps 96 \
  --profile-json /tmp/gpu-routing-profile.json \
  > /tmp/gpu-routing-nsys-summary.json
```

Expected: both JSON files match their declared schemas; parser `gpu_routing_verdict.decision` is `KEEP`; `gpu_route_submit` contains no synchronous D2H operation; `gpu_route_handoff` appears on `ExpertRouteFunc`; the activity-bitmap copies are asynchronous; `expert_completion_handoff` contains stream event waits; and neither `MoEMLP::forward` nor the profiled loop emits `cudaStreamSynchronize`.

- [ ] **Step 8: Commit executable benchmark gates**

```bash
git add benchmarks/serving/latency.py \
  benchmarks/expert_io_microbench/run_decision_profile.py \
  benchmarks/expert_io_microbench/nsys_parser.py \
  tests/python/unit/test_gpu_routing_benchmarks.py \
  tests/python/unit/test_gpu_routing_source_contract.py \
  docs/benchmarking.md
git commit -m "bench: gate gpu routing on p50 and p99 tpot"
```

### Task 8: Document operations, fallback, and phased landing

**Files:**
- Create: `tests/python/unit/test_gpu_routing_documentation.py`
- Modify: `docs/configuration.md`
- Modify: `docs/environment-variables.md`
- Modify: `docs/benchmarking.md`
- Modify: `benchmarks/expert_io_microbench/README.md`

- [ ] **Step 1: Write a failing executable documentation-content test**

Create `tests/python/unit/test_gpu_routing_documentation.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_gpu_routing_configuration_and_observability_are_documented():
    configuration = _read("docs/configuration.md")
    environment = _read("docs/environment-variables.md")

    assert "`gpu_only_expert_routing`" in configuration
    assert "single-host `dispatch_local`" in configuration
    assert "`speculative_prefetch_overlap=true`" in configuration
    assert "invalid combinations raise before engine construction" in configuration
    assert "CPU masks" in configuration
    assert "older native extensions" in configuration
    assert "DFlash route-ahead" in configuration

    for stage in (
        "`gpu_route_submit`",
        "`gpu_route_fallback`",
        "`gpu_route_handoff`",
        "`expert_completion_handoff`",
    ):
        assert stage in environment
    assert "route_failures" in environment
    assert "gpu_only_expert_routing=false" in environment


def test_gpu_routing_benchmark_runbook_documents_keep_and_rollback():
    for relative_path in (
        "docs/benchmarking.md",
        "benchmarks/expert_io_microbench/README.md",
    ):
        runbook = " ".join(_read(relative_path).split())
        assert "concurrency 1, 2, and 4" in runbook
        assert "TPOT p50 regression <=2%" in runbook
        assert "TPOT p99 regression <=5%" in runbook
        assert "zero route failures" in runbook
        assert "zero unexpected eager fallbacks" in runbook
        assert "gpu_only_expert_routing=false" in runbook
        assert "speculative_prefetch_overlap=false" in runbook
        assert "single-host personal-machine offloading" in runbook
        assert "not multi-node results" in runbook
```

- [ ] **Step 2: Run the documentation-content test and verify RED**

Run:

```bash
pytest -q tests/python/unit/test_gpu_routing_documentation.py
```

Expected: both tests fail because the configuration row, profiler/NVTX stages, KEEP thresholds, first-release conflict rule, single-host scope, and rollback instruction have not all been added to the four documentation files.

- [ ] **Step 3: Add the configuration contract**

Add this row to the `ArcherConfig` table in `docs/configuration.md`:

```markdown
| `gpu_only_expert_routing` | `false` | Opt in to native CUDA active-expert discovery for single-host `dispatch_local`. First release: mutually exclusive with `speculative_prefetch_overlap=true` and overlap-prefetch `observe`/`enforce`; invalid combinations raise before engine construction. CPU masks, older native extensions, and active DFlash route-ahead contexts use eager routing. Routing IDs, weights, output accumulation, and RPC `dispatch()` are unchanged. |
```

Do not add an environment variable: the rollout is per-model configuration and must be captured in benchmark JSON.

- [ ] **Step 4: Document observability and operator action**

Add to `docs/environment-variables.md` under `MOE_INFINITY_PROFILE_IO`:

```markdown
With `gpu_only_expert_routing=true`, full sampling emits `gpu_route_submit`,
`gpu_route_fallback`, and the existing `sync_wait` stages. Native NVTX adds
`gpu_route_handoff` and `expert_completion_handoff`. A nonzero
`route_failures` count or any unexpected `gpu_route_fallback` event is a
rollback signal; set `gpu_only_expert_routing=false` and retain the eager path.
```

- [ ] **Step 5: Add the exact benchmark runbook and interpretation**

Add the Task 7 A/B and Nsight commands to `docs/benchmarking.md` and `benchmarks/expert_io_microbench/README.md`. State all of the following:

```markdown
- TPOT is decode elapsed time divided by generated tokens after the first token;
  `itl_*` remains an alias for compatibility.
- Compare off/on runs only on the same commit, process environment, checkpoint,
  offload tree, visible GPUs, cache ratio, prompt/output lengths, and greedy mode.
- KEEP requires semantic tests to pass, zero route failures, zero unexpected
  eager fallbacks, TPOT p50 regression <=2%, and TPOT p99 regression <=5% at
  concurrency 1, 2, and 4.
- Roll back by setting `gpu_only_expert_routing=false`; do not remove eager
  bindings during the rollout.
- First-release runs must keep `speculative_prefetch_overlap=false` and any
  `overlap_prefetch_mode` at `off`; `observe`/`enforce` combinations are
  configuration errors, not benchmark variants.
- A later reconciliation may permit simultaneous enablement only after both
  plans share generation, active-list, completion, retirement, and failure
  ownership contracts.
- Results apply to single-host personal-machine offloading only. They are not
  multi-node results and are not promises of DeepEP or paper-level speedups.
```

- [ ] **Step 6: Run executable documentation QA and verify GREEN**

Run:

```bash
pytest -q tests/python/unit/test_gpu_routing_documentation.py
```

Expected: `2 passed`; the test confirms the documented opt-in and fallback cases, all four profiler/NVTX stage names, route-failure rollback signal, exact TPOT p50/p99 KEEP thresholds at concurrency 1/2/4, overlap exclusion, eager rollback command, and single-host/non-multi-node scope in the intended files.

- [ ] **Step 7: Commit documentation**

```bash
git add tests/python/unit/test_gpu_routing_documentation.py \
  docs/configuration.md \
  docs/environment-variables.md \
  docs/benchmarking.md \
  benchmarks/expert_io_microbench/README.md
git commit -m "docs: document gpu routing rollout and rollback"
```

### Task 9: Final regression gate and phased rollout

**Files:**
- Test only; no new files.

- [ ] **Step 1: Run CPU-only regression coverage**

```bash
pytest -q -m "not gpu and not integration and not network" \
  tests/python/unit/test_utils_config.py \
  tests/python/unit/test_gpu_only_expert_routing.py \
  tests/python/unit/test_gpu_routing_benchmarks.py \
  tests/python/unit/test_gpu_routing_source_contract.py \
  tests/python/dflash/test_route_ahead_wire.py \
  tests/python/dflash/test_route_ahead_metrics.py
```

Expected: all selected tests pass with no unexpected warnings.

- [ ] **Step 2: Run native CUDA regression coverage**

```bash
CUTLASS_DIR="$HOME/cutlass" pip install --no-build-isolation -e .
CUDA_VISIBLE_DEVICES=0 timeout 180s pytest -q -m gpu \
  tests/python/ops/test_expert_dispatch.py \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py
```

Expected: all tests supported by the GPU pass; architecture-specific tests may skip with their existing explicit reasons, not fail.

- [ ] **Step 3: Run static quality gates**

```bash
ruff check moe_infinity/distributed/expert_executor.py \
  moe_infinity/utils/config.py \
  tests/python/unit/test_gpu_only_expert_routing.py \
  tests/python/unit/test_gpu_routing_benchmarks.py \
  tests/python/ops/test_expert_dispatch.py \
  benchmarks/serving/latency.py \
  benchmarks/expert_io_microbench/bench_routing.py \
  benchmarks/expert_io_microbench/bench_bubble.py \
  benchmarks/expert_io_microbench/nsys_parser.py \
  benchmarks/expert_io_microbench/run_decision_profile.py

python benchmarks/serving/validate_batched_dispatch.py
```

Expected: Ruff exits zero; source-contract tests report no production expert-forward synchronize; validator exits zero and reports the native branch free of `.cpu(`, `.numpy(`, `.item(`, and `.tolist(`.

- [ ] **Step 4: Land in three reversible phases**

Phase A merges Tasks 1-5, including Task 3A, with the flag defaulting to `false`; unified failure/destruction, non-default-stream, native/eager parity, configuration-conflict, and eager route-ahead fallback tests must pass. Phase B lands Tasks 6-7 so every documented latency/decision-profile/parser command and schema is tested before execution. Phase C validates GPU-only routing alone with every overlap mode off. The existing overlap-prefetch plan is validated separately with GPU routing off; any simultaneous configuration must fail before engine construction. Phase D enables GPU routing only in benchmark/example configurations after all KEEP criteria pass; changing the global default or reconciling the two plans requires a separate review and fresh benchmark evidence.

Rollback immediately to `gpu_only_expert_routing=false` if any of these occurs:

- routed IDs differ from eager routing, outputs exceed existing BF16 tolerances, an empty mask hangs, or a generation crosses dispatch state;
- native `route_failures` is nonzero, a normal non-route-ahead CUDA run emits `gpu_route_fallback`, or event counts grow without retirement;
- submission/callback/route/fetch/exec/output/completion-event/retirement-launch injection fails to close state and rethrow on `WaitHiddenStates()`, or dispatcher destruction exceeds 30 seconds;
- `route_error_` is assigned anything other than `std::exception_ptr`/`nullptr`, a failed generation underflows `pending_`, or cache/node/overload state remains locked after failure;
- TPOT p50 regresses by more than 2% or TPOT p99 by more than 5% at any measured concurrency;
- Nsight shows a blocking D2H operation in `gpu_route_submit`, `cudaStreamSynchronize` in `MoEMLP::forward` or the measured decision-profile loop, a device/stream synchronize in `expert_completion_handoff`, or unexplained D2H copies beyond routing metadata;
- latency/profile JSON fails schema validation, warmup samples enter measured TPOT, or the parser cannot reproduce the documented routing verdict;
- cache-hit rate, expert fetch count, or output dtype/shape differs from the eager run.

- [ ] **Step 5: Record the final implementation commit**

```bash
git status --short
git log --oneline -9
```

Expected: only intentional implementation/test/docs files are present and each prior task has one focused commit. If verification required a fix, commit only that fix with:

```bash
git add core/parallel/expert_dispatcher.h \
  core/parallel/expert_dispatcher.cpp \
  core/parallel/expert_module.cpp \
  core/utils/cuda_utils.h \
  core/python/py_archer_prefetch.cpp \
  moe_infinity/distributed/expert_executor.py \
  moe_infinity/utils/config.py \
  tests/python/unit/test_gpu_only_expert_routing.py \
  tests/python/unit/test_gpu_routing_benchmarks.py \
  tests/python/unit/test_gpu_routing_source_contract.py \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py \
  tests/python/ops/test_expert_dispatch.py \
  benchmarks/serving/validate_batched_dispatch.py \
  benchmarks/serving/latency.py \
  benchmarks/expert_io_microbench/bench_routing.py \
  benchmarks/expert_io_microbench/bench_bubble.py \
  benchmarks/expert_io_microbench/nsys_parser.py \
  benchmarks/expert_io_microbench/run_decision_profile.py \
  docs/configuration.md docs/environment-variables.md \
  docs/benchmarking.md benchmarks/expert_io_microbench/README.md
git commit -m "fix: close gpu routing verification gaps"
```

## Risk register

| Risk | Detection | Mitigation / rollback |
| --- | --- | --- |
| `wait_expert()` observes `pending_ == 0` before route discovery finishes | Empty-mask and immediate-wait timeout tests | Include `route_pending_` in the wait predicate and set it before launching the async copy. |
| Pinned bitmap is reused before D2H completes or callback outlives dispatcher | CUDA memcheck, generation stress test, destructor test | Move tensor ownership into callback-owned `RouteArgs`; destructor waits for the callback counter before closing the route queue. |
| Caller consumes `final_hidden_states_` before worker streams finish | Immediate downstream consumer parity test, Nsight stream graph | Record completion after accumulation and enqueue waits on the caller's current stream. |
| Completion events leak across thousands of layers/tokens or after a terminal failed dispatch | `completion_events_retired`, `completion_events_outstanding`, retirement-launch fault with no next dispatch, soak-test RSS/VRAM | Retire through the caller-wait callback/CPU queue, query unwaited records, and synchronize/release only destructor fallbacks; never clear raw handles at the next dispatch. |
| Native active IDs differ from eager `sum > 0` behavior | Parametrized masks and ascending-ID assertions | Use boolean `any(0)` only; no top-k recomputation or router-logit interpretation. |
| Route-ahead pinning loses its pre-enqueue ordering | Existing DFlash wire tests | Force eager fallback while route-ahead context is active; leave integration to the overlap-aware-prefetch follow-up. |
| Multi-GPU owner changes | Two-GPU parity and active-list tests | Use `expert_idx % kNumDevices()` exactly; do not add topology remapping. |
| A fetch/exec/output exception is logged and swallowed | Parameterized worker fault injection and source contract | Replace every worker catch with `FailDispatch`, generation-safe pending completion, cache/node restoration, waiter notification, and rethrow from `WaitHiddenStates()`. |
| Submission, callback, or route worker throws across a boundary | Submission/callback/worker injection plus 30-second destruction test | Arm `DispatchSubmissionGuard` immediately after setting `route_pending_`; catch all asynchronous exceptions; store only the first `std::exception_ptr`; close and notify through `FailDispatch`. |
| Completion-event record or retirement-callback launch fails after resources are acquired | Dedicated injected failures, event-retirement counters, destruction timeout | Use `PooledCudaEventLease` and callback ownership counters; route both failures through `OutputFunc`'s catch and `FailDispatch`. |
| A wait-handoff CUDA failure recursively locks `route_state_mutex_` | Source lock-scope contract and wait-handoff fault review | Detach error/events under the mutex, release it, perform every CUDA call, then call `FailDispatch(generation, error)` outside all route-state lock scopes. |
| Failed work completes or reports failure late and underflows/reopens/poisons the next generation | Stale-failure injection during recovery plus repeated fault/recovery cycles | Stamp every asynchronous argument, pass its immutable generation to `FailDispatch`, quarantine non-current failures before all shared-state/context mutation, and make `CompleteOne` ignore non-current or failed-generation completions. |
| Removing `MoEMLP` synchronizers exposes cross-stream input or early expert eviction | Non-default caller-stream dependent-op test, cache-pressure parity, Nsight | Move the stream guard before all input work and retire expert/cache state through its own stream callback and retirement queue. Keep the feature default-off if retirement is not stream-safe. |
| Timing event is recycled as a completion event or blocks production | Source contract and event-retirement counters | `GpuTimer` privately owns timing events; pooled completion events are never timed; route readiness uses no event. |
| GPU routing and overlap-prefetch ownership conflict | Config tests for the current boolean and future `observe`/`enforce` modes | Reject simultaneous enablement before engine construction; preserve a later reconciliation seam for a generation-scoped active-list contract. |
| Benchmark command/schema drifts from implementation | argv/config/schema unit tests and exact runbook commands | Parsers accept documented flags, builders carry mode, warmups are discarded, and schema mismatch fails before model loading. |
| Event-based completion changes numerical reduction order | Native/eager BF16 parity over repeated runs | Keep existing host launch order and float32 accumulation; rollback on tolerance failure. |
| Apparent speedup is measurement noise | 3 warmups, 30 rounds, p50/p99, fixed workload | Use objective KEEP/ROLLBACK thresholds and make no external speedup promise. |

## Definition of done

- The enabled single-host CUDA path contains no Python-side `.cpu().numpy()`, `.item()`, or `.tolist()` before native dispatch.
- Native active IDs and outputs match eager routing for single-active, multi-active, all-active, and empty masks.
- Current adapters require no code change and continue to call the same executor methods.
- `WaitHiddenStates()` establishes CUDA stream dependencies without device or stream synchronization on the caller thread.
- `MoEMLP::forward` contains no `cudaStreamSynchronize`; all input/dequant/forward/output work is guarded by the worker's non-default stream, and expert/cache retirement is stream-ordered.
- Synchronous submission and asynchronous route/fetch/exec/output/completion/retirement failures all pass their immutable originating generation to `FailDispatch`; current failures clear state, restore resources, notify waiters, and rethrow the first `std::exception_ptr`, while stale failures are quarantined without touching the current dispatch.
- `WaitHiddenStates()` never calls `FailDispatch` while holding `route_state_mutex_`.
- Completion-event record and retirement-callback launch fault tests pass without event leakage, pending underflow, swallowed exceptions, or destructor hangs; an already-inserted event reaches `completion_events_outstanding == 0` without starting another dispatch.
- Eager fallback remains available, observable, and one configuration change away.
- Every documented latency, decision-profile, and Nsight-parser command is backed by argv/config tests and emits the declared versioned schema; warmups are excluded and TPOT p50/p99 verdicts are reproducible.
- CPU-only tests, single-GPU CUDA tests, non-default-stream tests, optional two-GPU tests, build, Ruff, source validator, Nsight checks, and TPOT p50/p99 gates pass.
- GPU-only routing lands independently with every overlap-prefetch mode off; simultaneous current/`observe`/`enforce` configurations are rejected; the later reconciliation seam and eager route-ahead fallback are documented; single-host-only scope makes no multi-node or paper/DeepEP speedup claim.
