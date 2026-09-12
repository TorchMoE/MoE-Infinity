# Overlap-Aware Expert Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit speculative expert transfers only when their exact stored bytes fit a conservatively calibrated transfer window derived from measured bandwidth and the current layer's measured compute time, while leaving the model's native router, expert weights, and dispatched expert set authoritative.

**Architecture:** Add a pure-Python `OverlapBudgetController` that owns deterministic EWMA calibration, byte admission, and accounting; feed it native transfer/queue telemetry and timing-enabled CUDA measurements anchored immediately around the expert kernel body. `ExpertPrefetcher` remains a cache-warming client: it ranks existing mean/top-k, trace-predictor, or DFlash exact-route candidates, asks the controller for a byte-bounded subset, and never changes routing. Extend the native task pool with generation-scoped admission, centralized terminal byte accounting, cancellation, backpressure, and drainable telemetry; keep the existing APIs and behavior intact when the new policy is `off`.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pytest, C++17, CUDA events/streams, pybind11, GoogleTest, Nsight Systems/NVTX, existing `IOProfiler` and expert I/O microbench harness.

---

## Scope, invariants, and measured model

### Correctness invariants

1. The router remains the sole source of `router_mask`, `router_weights`, and the complete `expert_list`. No code in this plan edits router logits, masks, weights, top-k selection, token assignment, acceptance, or model outputs.
2. Budgeting applies only to early cache warming. At dispatch, every native-routed expert is still passed to `expert_dispatcher.enqueue_expert`; omitted, canceled, or late prefetches fall through to the existing on-demand path.
3. An exact route is a correction signal, not permission to speculate execution. Queued false positives may be canceled, resident false positives may later be evicted normally, and exact-route misses remain on-demand work.
4. `ReplaceCacheCandidates` protects at most one current layer. Never pin candidates from two layers together. An empty admitted set must not replace the current pin set.
5. Policy `off` is the default and preserves the existing mean/top-k and DFlash wiring byte-for-byte. It does not allocate/record/query timing events, instantiate the controller, probe/call/drain any new pybind method, or add telemetry polling. Policy `observe` computes decisions and metrics but issues the same transfers as `off`. Policy `enforce` applies admission, cancellation, and queue limits. Capability detection and every new binding call occur only in `observe|enforce`; an older extension therefore remains fully usable with `off`, `observe` records a capability miss and keeps legacy issuance, and `enforce` fails closed without issuing bounded work.
6. The first release supports eager routing only when policy is `observe` or `enforce`. Configuration rejects `gpu_only_expert_routing=True` together with either active overlap policy. `gpu_only_expert_routing=True, overlap_prefetch_policy="off"` and eager routing with either overlap policy remain independently usable. DFlash is an optional caller, not a prerequisite.

### Explicit overlap budget

For layer `l`, after warm calibration:

```text
T_window_ns(l) = max(0,
    safety_factor * compute_ewma_ns[l]
    - queue_wait_ewma_ns
    - issue_overhead_ewma_ns)

B_window(l) = floor(bandwidth_ewma_bytes_per_ns * T_window_ns(l))
B_admit(l)  = clamp(B_window(l) - current_inflight_bytes,
                    0,
                    max_prefetch_window_bytes)
```

`compute_ewma_ns[l]` is the per-destination-GPU wall-clock union of expert-kernel intervals for the most recently completed invocation of layer `l` (`max(kernel_end_offset)-min(kernel_start_offset)`). Start is recorded on the execution stream immediately before `ForwardHelper(stream)` and stop immediately after it. Correctness does not depend on either current `MoEMLP::forward` host synchronization: those synchronizations may remain or be removed by a later GPU-routing optimization, while event ordering on the stream still brackets the same kernels. The interval excludes transfer waits, input copies, buffer setup, output cloning, `OutputFunc`, accumulation, and host barrier delay. `bandwidth_ewma_bytes_per_ns` is based only on completed host/disk-to-GPU prefetch samples: `bytes / transfer_ns`. Queue wait and Python-to-native issue overhead are separately EWMAd and subtracted, preventing already queued work from being counted as overlap capacity. All EWMAs use `new = alpha * sample + (1 - alpha) * old`, reject non-positive samples, and update only after completion.

Compute timing events are not acquired from `kCudaEventPool`: `core/memory/event_pool.h` creates events with `cudaEventDisableTiming`, so elapsed-time calls on those events are invalid. Timing is disabled in native code by default. Only an explicit active-policy capability-gated call enables it; until then no epoch, pair, or quarantine-fence event is created, recorded, queried, synchronized, or drained. Once enabled, `ExpertDispatcher` owns one timing-enabled epoch event per GPU and a bounded pool of timing-enabled start/stop pairs per execution thread, all created by the worker after `cudaSetDevice` with `cudaEventCreateWithFlags(..., cudaEventDefault)`. The epoch is recorded and synchronized once; start/stop are recorded on that thread's `exec_stream`. Normal pairs move `free -> active -> pending`; only `cudaEventQuery(stop) == cudaSuccess` permits elapsed-time extraction, CPU sample publication, and `pending -> free`. `cudaErrorNotReady` leaves the pair pending. If no free pair exists, skip timing that invocation and increment `timing_samples_dropped`; never overwrite a pending event and never add a host synchronization merely to collect telemetry.

Every dispatcher input setup receives a monotonically increasing `invocation_id`, copied into every `ExecArgs` and every resulting `ExpertComputeSample`. Python stores that ID in `_pending_prefetch`. After `wait_expert`, calibration accepts only samples whose `(invocation_id, layer_id)` exactly match the detached pending dispatch. A delayed sample from an older invocation is drained and counted as stale telemetry but never updates the current layer's EWMA; layer ID alone is not a sufficient correlation key.

If `ForwardHelper` or surrounding forward code throws after either timing event may have been touched, the pair moves to `quarantined`, not `free`, and produces no sample. The catch path records a separately owned timing-disabled quarantine fence on the same execution stream after all already-submitted work. It queries only that fence—never an unrecorded/unspecified stop event. `cudaErrorNotReady` retains the pair and fence; `cudaSuccess` proves prior stream work complete, destroys the fence, and returns the timing pair to `free` without elapsed-time extraction. If fence record/query returns another CUDA error, call `cudaStreamSynchronize(stream)` as an error-recovery proof; reuse is allowed only after a successful synchronization. At shutdown, join workers, synchronize each execution stream once, and destroy free/pending/quarantined timing events plus quarantine fences only for streams proven complete. If synchronization fails, log and intentionally leave those CUDA handles unreused/undestroyed for process teardown rather than touching events with unspecified in-flight state.

Three CUDA event domains must remain distinct: (a) dispatcher-owned timing-enabled kernel events, never pooled; (b) existing transfer-completion events from `kCudaEventPool`, timing-disabled and released after the execution stream has consumed its wait; and (c) a future GPU router's timing-disabled route-completion event, router-owned until every route consumer has enqueued/finished its wait. No handle crosses domains or is used both for timing and dependency completion.

Candidates are stable-sorted by `(-score, original_position, expert_id)`. Each candidate costs `expert_nbytes_map[(layer_id, expert_id)]`; candidates with missing/non-positive costs are not admitted in `enforce`. Admission is whole-expert greedy packing—no partial expert transfer. Already resident experts cost zero and count as covered, already in-flight experts consume their outstanding bytes once, and duplicate tensor IDs are deduplicated.

Cold start is deliberately conservative: until both one valid compute sample for the target layer and one valid transfer sample exist, admit at most `overlap_prefetch_cold_start_experts` (default `1`) and still enforce `overlap_prefetch_max_inflight_bytes`. A missing byte map, missing native telemetry API, invalid sample, or non-native engine causes `enforce` to admit nothing; it never fabricates average sizes or guessed bandwidth. `off` and `observe` retain compatibility behavior.

### Metrics and definitions

Expose snapshots from `ExpertPrefetcher.overlap_prefetch_stats()` and `IOProfiler` events:

- `candidate_bytes`: exact bytes considered by policy.
- `budget_bytes`: `B_admit` before packing.
- `admitted_bytes`: bytes accepted by the native queue.
- `completed_prefetch_bytes`: admitted bytes whose transfer completed.
- `covered_route_bytes`: bytes of actual routed experts already resident at exact-route correction.
- `route_bytes`: exact bytes in the native routed union.
- `coverage = covered_route_bytes / route_bytes` (1.0 when `route_bytes == 0`).
- `wasted_prefetch_bytes`: completed speculative bytes not in the corrected native route. Canceled-before-start bytes are not waste.
- `canceled_prefetch_bytes`: queued bytes canceled before transfer starts.
- `late_prefetch_bytes`: bytes in the actual route that had been admitted speculatively but were not resident when dispatch began. Actual experts never admitted are `uncovered_route_bytes`, not “late prefetch.”
- `queue_rejected_bytes`: bytes rejected by native backpressure.
- `inflight_prefetch_bytes`, `compute_ewma_ns_by_layer`, `bandwidth_ewma_bytes_per_second`, `queue_wait_ewma_ns`, and `issue_overhead_ewma_ns`.

Counters are monotonically cumulative; gauges/EWMAs are snapshots. Metrics are observational and never feed the router.

## File map

- Create `moe_infinity/memory/overlap_budget.py`: pure deterministic controller, EWMA state, candidate packing, correction accounting, and snapshot schema.
- Create `tests/python/unit/test_overlap_budget.py`: CPU-only table tests for formulas, cold start, stable packing, missing costs, exact correction, and metrics.
- Modify `moe_infinity/utils/config.py`: default-off `off|observe|enforce` policy and validated calibration/backpressure knobs.
- Modify `tests/python/unit/test_utils_config.py`: defaults, JSON loading, and invalid-value tests.
- Modify `moe_infinity/memory/expert_prefetcher.py`: controller ownership, exact byte costing, native admission/cancellation/telemetry bridge, candidate policy, and stats accessor.
- Modify `moe_infinity/memory/expert_predictor.py`: return stable per-layer scored candidates without changing its existing matrix result.
- Modify `moe_infinity/distributed/expert_executor.py`: exact-route correction before all expert enqueues, one-layer exact-route pinning of only admitted experts, and compute-sample feedback after the wait barrier.
- Modify `moe_infinity/memory/expert_tracer.py`: accept the new metric stages and aggregate bytes consistently.
- Modify `moe_infinity/profiling/io_profiler.py`: public instant-event API for policy decisions and native telemetry.
- Modify `core/prefetch/task_scheduler.h`: generation/layer metadata, queue-byte accounting, bounded admission, cancel, and telemetry structures/APIs.
- Modify `core/prefetch/task_scheduler.cpp`: thread-safe implementation and completed/canceled/rejected accounting.
- Modify `core/prefetch/archer_prefetch_handle.h`: policy scheduling/cancel/drain methods while retaining legacy methods.
- Modify `core/prefetch/archer_prefetch_handle.cpp`: tensor-to-node dedupe, calls into bounded task-pool APIs, and plain-value snapshots.
- Modify `core/parallel/expert_dispatcher.h`: compute interval sample and drain API.
- Modify `core/parallel/expert_dispatcher.cpp`: own timing-enabled CUDA events, collect kernel-only intervals, and separately timestamp `OutputFunc`/host completion delay.
- Create `core/parallel/expert_timing.h`: testable free/pending/quarantined timing-pair lifecycle and retirement state.
- Modify `core/parallel/expert_module.h`: pass the dispatcher-owned timing start/stop events into `MoEMLP::forward`.
- Modify `core/parallel/expert_module.cpp`: record timing events immediately around `ForwardHelper` without relying on either existing host synchronization.
- Modify `core/python/py_archer_prefetch.cpp`: bind new methods under new names; do not change legacy `prefetch_tensors` return behavior.
- Create `tests/cpp/unit/prefetch/test_prefetch_queue_accounting.cpp`: deterministic accounting/cancel helper tests without a GPU.
- Modify `CMakeLists.txt`: add an opt-in `MOE_BUILD_TESTS` root build and include `tests/cpp` only when enabled.
- Create `tests/cpp/CMakeLists.txt`: find GoogleTest/CUDAToolkit and add native unit subdirectories in the already configured Torch/CUDA build.
- Create `tests/cpp/unit/prefetch/CMakeLists.txt`: link queue-accounting tests to `archer_core`, Torch, CUDA runtime, and `GTest::gtest_main`.
- Create `tests/cpp/unit/parallel/test_expert_timing_lifecycle.cpp`: fake-CUDA-status tests for normal pending retirement, exception quarantine, recovery, and destructor behavior.
- Create `tests/cpp/unit/parallel/CMakeLists.txt`: link timing lifecycle tests through the root Torch/CUDA test build.
- Modify `tests/python/dflash/test_speculative_prefetch.py`: byte-bounded legacy and exact-list unit tests.
- Modify `tests/python/dflash/test_route_ahead_wire.py`: full native route remains dispatched while only admitted experts are pinned/warmed.
- Modify `tests/python/dflash/test_route_ahead_metrics.py`: byte coverage/waste/late accounting under partial admission.
- Modify `tests/python/dflash/test_prefetch_native_gpu.py`: opt-in native admission, cancellation, backpressure, telemetry, and one-layer smoke tests.
- Create `benchmarks/expert_io_microbench/bench_overlap_prefetch.py`: paired CUDA `off|observe|enforce` benchmark and JSON metrics.
- Create `tests/python/dflash/test_overlap_prefetch_report.py`: CPU-only report/schema and decision-gate tests.
- Modify `benchmarks/expert_io_microbench/run_decision_profile.py`: policy CLI and overlap metric output.
- Modify `benchmarks/expert_io_microbench/run_all.py`: optional overlap scenario merge.
- Modify `benchmarks/expert_io_microbench/README.md`: reproducible commands, interpretation, and no performance promise.
- Modify `docs/configuration.md`: knobs and safe defaults.
- Modify `docs/environment-variables.md`: benchmark opt-in variables only.

## Motivation boundary

FineMoE ([arXiv:2502.05370](https://arxiv.org/abs/2502.05370)), PreScope ([arXiv:2509.23638](https://arxiv.org/abs/2509.23638)), and SpecPrefetch ([arXiv:2607.24787](https://arxiv.org/abs/2607.24787)) motivate treating expert movement as a bounded overlap problem rather than an unbounded top-k request. They are motivation only: this implementation derives no correctness claim, algorithm equivalence, or performance target from those papers.

## Cross-plan compatibility with GPU-only expert routing

First-release compatibility is deliberately fail-closed:

| `gpu_only_expert_routing` | `overlap_prefetch_policy` | Result |
| --- | --- | --- |
| `False` | `off`, `observe`, or `enforce` | Valid eager-routing configuration |
| `True` | `off` | Valid GPU-routing configuration; this plan is inactive |
| `True` | `observe` or `enforce` | `ValueError` during `ArcherConfig` construction/loading |

This plan adds the shared default-false config field and incompatibility validation but does not implement GPU routing. The GPU-routing plan may consume the field independently while overlap prefetch remains `off`. A later cross-feature plan may lift the rejection only after it owns and tests exact-route readiness, correction order, router completion-event ownership, transfer completion-event ownership, and dispatcher timing-event quarantine together. Until then, this plan contains no `route_ready` adapter and makes no combined-mode promise.

---

### Task 1: Build the deterministic overlap budget controller

**Files:**
- Create: `moe_infinity/memory/overlap_budget.py`
- Create: `tests/python/unit/test_overlap_budget.py`

- [ ] **Step 1: Write failing formula, cold-start, and packing tests**

```python
from moe_infinity.memory.overlap_budget import Candidate, OverlapBudgetController


def controller(**kwargs):
    return OverlapBudgetController(
        policy="enforce", alpha=0.5, safety_factor=0.8,
        max_window_bytes=10_000, max_inflight_bytes=10_000,
        cold_start_experts=1, **kwargs,
    )


def test_warm_budget_subtracts_queue_issue_and_inflight():
    c = controller()
    c.observe_compute(layer_id=3, start_ns=0, end_ns=1_000)
    c.observe_transfer(bytes_transferred=1000, transfer_ns=100,
                       queue_wait_ns=100, issue_overhead_ns=100)
    d = c.admit(3, [Candidate(2, 3.0, 3000), Candidate(1, 2.0, 2500)],
                inflight_bytes=500)
    # window=(.8*1000)-100-100=600ns; 10 B/ns => 6000B; minus 500B.
    assert d.budget_bytes == 5500
    assert d.expert_ids == (2, 1)
    assert d.admitted_bytes == 5500


def test_cold_start_admits_only_one_exact_costed_expert():
    d = controller().admit(0, [Candidate(7, 9.0, 4096), Candidate(3, 8.0, 1)])
    assert d.cold_start is True
    assert d.expert_ids == (7,)


def test_missing_cost_is_never_fabricated_in_enforce():
    d = controller().admit(0, [Candidate(7, 9.0, None)])
    assert d.expert_ids == ()
    assert d.uncosted_experts == (7,)


def test_stable_whole_expert_packing_skips_non_fitting_candidate():
    c = controller()
    c.observe_compute(1, 0, 1_000)
    c.observe_transfer(1000, 100, 0, 0)  # budget=8000
    d = c.admit(1, [Candidate(9, 3.0, 9000), Candidate(4, 2.0, 4000),
                    Candidate(2, 2.0, 4000)])
    assert d.expert_ids == (4, 2)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/python/unit/test_overlap_budget.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'moe_infinity.memory.overlap_budget'`.

- [ ] **Step 3: Implement the pure controller and immutable result types**

```python
from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Optional, Sequence


@dataclass(frozen=True)
class Candidate:
    expert_id: int
    score: float
    nbytes: Optional[int]


@dataclass(frozen=True)
class AdmissionDecision:
    expert_ids: tuple[int, ...]
    candidate_bytes: int
    budget_bytes: int
    admitted_bytes: int
    uncosted_experts: tuple[int, ...]
    cold_start: bool


class OverlapBudgetController:
    def __init__(self, *, policy: str, alpha: float, safety_factor: float,
                 max_window_bytes: int, max_inflight_bytes: int,
                 cold_start_experts: int) -> None:
        self.policy = policy
        self.alpha = alpha
        self.safety_factor = safety_factor
        self.max_window_bytes = max_window_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.cold_start_experts = cold_start_experts
        self.compute_ewma_ns: dict[int, float] = {}
        self.bandwidth_ewma_bytes_per_ns: Optional[float] = None
        self.queue_wait_ewma_ns: Optional[float] = None
        self.issue_overhead_ewma_ns: Optional[float] = None

    def _ewma(self, old: Optional[float], sample: float) -> float:
        return sample if old is None else self.alpha * sample + (1-self.alpha)*old

    def observe_compute(self, layer_id: int, start_ns: int, end_ns: int) -> None:
        sample = end_ns - start_ns
        if sample > 0:
            self.compute_ewma_ns[layer_id] = self._ewma(
                self.compute_ewma_ns.get(layer_id), float(sample))

    def observe_transfer(self, bytes_transferred: int, transfer_ns: int,
                         queue_wait_ns: int, issue_overhead_ns: int) -> None:
        if bytes_transferred <= 0 or transfer_ns <= 0:
            return
        self.bandwidth_ewma_bytes_per_ns = self._ewma(
            self.bandwidth_ewma_bytes_per_ns,
            bytes_transferred / transfer_ns)
        self.queue_wait_ewma_ns = self._ewma(
            self.queue_wait_ewma_ns, float(max(queue_wait_ns, 0)))
        self.issue_overhead_ewma_ns = self._ewma(
            self.issue_overhead_ewma_ns, float(max(issue_overhead_ns, 0)))

    def admit(self, layer_id: int, candidates: Sequence[Candidate],
              inflight_bytes: int = 0) -> AdmissionDecision:
        ordered = sorted(enumerate(candidates),
                         key=lambda item: (-item[1].score, item[0], item[1].expert_id))
        uncosted = tuple(c.expert_id for _, c in ordered
                         if c.nbytes is None or c.nbytes <= 0)
        costed = [c for _, c in ordered if c.nbytes is not None and c.nbytes > 0]
        candidate_bytes = sum(int(c.nbytes) for c in costed)
        warm = layer_id in self.compute_ewma_ns and self.bandwidth_ewma_bytes_per_ns is not None
        if warm:
            window = max(0.0, self.safety_factor*self.compute_ewma_ns[layer_id]
                         - (self.queue_wait_ewma_ns or 0.0)
                         - (self.issue_overhead_ewma_ns or 0.0))
            budget = min(self.max_window_bytes,
                         floor(self.bandwidth_ewma_bytes_per_ns * window))
            budget = max(0, min(budget - max(inflight_bytes, 0),
                                self.max_inflight_bytes - max(inflight_bytes, 0)))
            limit = len(costed)
        else:
            budget = max(0, self.max_inflight_bytes - max(inflight_bytes, 0))
            limit = self.cold_start_experts
        selected: list[int] = []
        used = 0
        for c in costed:
            cost = int(c.nbytes)
            if len(selected) < limit and used + cost <= budget:
                selected.append(c.expert_id)
                used += cost
        return AdmissionDecision(tuple(selected), candidate_bytes, budget, used,
                                 uncosted, not warm)
```

Add correction counters and `snapshot()` in the same file, with integer counters initialized to zero and coverage's empty-route convention fixed at `1.0`. Keep this module free of Torch/CUDA imports so all decisions are CPU-testable.

- [ ] **Step 4: Add failing correction and EWMA tests, then complete minimal accounting**

```python
def test_exact_route_correction_separates_coverage_waste_late_and_uncovered():
    c = controller()
    c.record_issue(layer_id=2, generation=8, expert_nbytes={1: 100, 2: 200, 3: 300})
    c.record_completion(generation=8, expert_id=1, bytes_transferred=100)
    c.correct_route(layer_id=2, generation=8, actual_expert_nbytes={1: 100, 2: 200, 4: 400})
    s = c.snapshot()
    assert s["covered_route_bytes"] == 100
    assert s["late_prefetch_bytes"] == 200
    assert s["uncovered_route_bytes"] == 400
    # Expert 3 is a false positive but never completed, so it is canceled bytes,
    # not transferred waste.
    assert s["wasted_prefetch_bytes"] == 0
    assert s["canceled_prefetch_bytes"] == 300
    assert s["coverage"] == pytest.approx(1 / 7)


def test_only_completed_false_positive_counts_as_waste():
    c = controller()
    c.record_issue(layer_id=2, generation=9, expert_nbytes={1: 100, 3: 300})
    c.record_completion(generation=9, expert_id=1, bytes_transferred=100)
    c.record_completion(generation=9, expert_id=3, bytes_transferred=300)
    c.correct_route(layer_id=2, generation=9, actual_expert_nbytes={1: 100})
    s = c.snapshot()
    assert s["wasted_prefetch_bytes"] == 300
    assert s["canceled_prefetch_bytes"] == 0


def test_ewma_uses_configured_alpha():
    c = controller()
    c.observe_compute(0, 0, 100)
    c.observe_compute(0, 0, 300)
    assert c.compute_ewma_ns[0] == 200
```

Run: `pytest -q tests/python/unit/test_overlap_budget.py`

Expected first run: FAIL because `record_issue`, `record_completion`, `correct_route`, and `snapshot` are absent. Implement those methods with generation-keyed issued/completed maps, rerun, and expect all tests to pass.

- [ ] **Step 5: Commit the pure model**

```bash
git add moe_infinity/memory/overlap_budget.py tests/python/unit/test_overlap_budget.py
git commit -m "feat: add overlap prefetch budget model"
```

---

### Task 2: Add a default-off, independently rollable configuration surface

**Files:**
- Modify: `moe_infinity/utils/config.py:30-44,112-162`
- Modify: `tests/python/unit/test_utils_config.py`
- Modify: `docs/configuration.md`

- [ ] **Step 1: Write failing config tests**

```python
def test_overlap_prefetch_defaults_are_safe(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    c = ArcherConfig(offload_path="/tmp", use_native_engine=False)
    assert c.overlap_prefetch_policy == "off"
    assert c.overlap_prefetch_ewma_alpha == pytest.approx(0.2)
    assert c.overlap_prefetch_safety_factor == pytest.approx(0.8)
    assert c.overlap_prefetch_cold_start_experts == 1
    assert c.overlap_prefetch_max_window_bytes == 256 * 1024 * 1024
    assert c.overlap_prefetch_max_inflight_bytes == 512 * 1024 * 1024
    assert c.gpu_only_expert_routing is False


@pytest.mark.parametrize("field,value", [
    ("overlap_prefetch_policy", "fast"),
    ("overlap_prefetch_ewma_alpha", 0.0),
    ("overlap_prefetch_safety_factor", 1.1),
    ("overlap_prefetch_cold_start_experts", -1),
    ("overlap_prefetch_max_window_bytes", -1),
])
def test_overlap_prefetch_rejects_invalid_values(monkeypatch, field, value):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match=field):
        ArcherConfig(offload_path="/tmp", use_native_engine=False, **{field: value})


@pytest.mark.parametrize("policy", ["observe", "enforce"])
def test_overlap_prefetch_rejects_gpu_only_routing(monkeypatch, policy):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match="gpu_only_expert_routing"):
        ArcherConfig(offload_path="/tmp", use_native_engine=False,
                     gpu_only_expert_routing=True,
                     overlap_prefetch_policy=policy)


def test_gpu_only_routing_is_independent_when_overlap_is_off(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    c = ArcherConfig(offload_path="/tmp", use_native_engine=False,
                     gpu_only_expert_routing=True,
                     overlap_prefetch_policy="off")
    assert c.gpu_only_expert_routing is True
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/python/unit/test_utils_config.py -k overlap_prefetch`

Expected: FAIL with unexpected keyword or missing attribute for `overlap_prefetch_policy`.

- [ ] **Step 3: Add fields and exact validation**

```python
overlap_prefetch_policy: str = field(
    default="off", metadata={"help": "off, observe, or enforce overlap-window byte admission"})
overlap_prefetch_ewma_alpha: float = field(default=0.2)
overlap_prefetch_safety_factor: float = field(default=0.8)
overlap_prefetch_cold_start_experts: int = field(default=1)
overlap_prefetch_max_window_bytes: int = field(default=256 * 1024 * 1024)
overlap_prefetch_max_inflight_bytes: int = field(default=512 * 1024 * 1024)
gpu_only_expert_routing: bool = field(
    default=False,
    metadata={"help": "Enable GPU-only expert routing; incompatible with active overlap-prefetch policy in the first release."},
)
```

In `__post_init__`, validate policy membership, both floats in `(0, 1]`, and all integer limits non-negative. Also require `max_window_bytes <= max_inflight_bytes` when policy is `enforce`. Raise `ValueError("gpu_only_expert_routing cannot be combined with overlap_prefetch_policy=observe|enforce in the first release")` whenever GPU-only routing is true and policy is not `off`. Do not silently downgrade either feature and do not auto-enable `speculative_prefetch`, DFlash, or routing.

- [ ] **Step 4: Run config tests and document semantics**

Run: `pytest -q tests/python/unit/test_utils_config.py`

Expected: PASS. In `docs/configuration.md`, state the formula, cold-start behavior, missing-telemetry fail-closed behavior, `off|observe|enforce` rollout, and the cross-plan compatibility table above. Explicitly state that active overlap policy is eager-routing-only in the first release and that `enforce` requires the rebuilt native extension for cancellation/backpressure.

- [ ] **Step 5: Commit config and documentation**

```bash
git add moe_infinity/utils/config.py tests/python/unit/test_utils_config.py docs/configuration.md
git commit -m "feat: configure overlap-aware prefetch policy"
```

---

### Task 3: Add native bounded admission, cancellation, and transfer telemetry

**Files:**
- Modify: `CMakeLists.txt:39-42`
- Modify: `core/prefetch/task_scheduler.h:33-121`
- Modify: `core/prefetch/task_scheduler.cpp:85-127,467-597`
- Modify: `core/prefetch/archer_prefetch_handle.h:30-65`
- Modify: `core/prefetch/archer_prefetch_handle.cpp:238-278`
- Modify: `core/python/py_archer_prefetch.cpp:93-100`
- Create: `tests/cpp/CMakeLists.txt`
- Create: `tests/cpp/unit/prefetch/test_prefetch_queue_accounting.cpp`
- Create: `tests/cpp/unit/prefetch/CMakeLists.txt`

- [ ] **Step 1: Write a GPU-free queue-accounting test**

Extract a header-only `PrefetchQueueAccounting` value helper inside `task_scheduler.h`; it owns byte counters only and has no `NodePtr` dependency in its public methods.

```cpp
TEST(PrefetchQueueAccounting, AdmissionCancellationAndCompletionAreDisjoint) {
  PrefetchQueueAccounting a;
  EXPECT_TRUE(a.TryAdmit(10, 100, 250));
  EXPECT_TRUE(a.TryAdmit(11, 100, 250));
  EXPECT_FALSE(a.TryAdmit(12, 100, 250));
  EXPECT_EQ(a.queued_bytes(), 200);
  a.MarkStarted(10);
  EXPECT_EQ(a.CancelQueued(11), 100);
  EXPECT_EQ(a.CancelQueued(10), 0);
  a.MarkCompleted(10);
  EXPECT_EQ(a.inflight_bytes(), 0);
  EXPECT_EQ(a.completed_bytes(), 100);
  EXPECT_EQ(a.canceled_bytes(), 100);
  EXPECT_EQ(a.rejected_bytes(), 100);
  EXPECT_TRUE(a.InvariantHolds());
}

TEST(PrefetchQueueAccounting, EveryQueueRemovalReasonRetiresInflightBytesOnce) {
  for (auto reason : {RemovalReason::kClear, RemovalReason::kReplaceCandidates,
                      RemovalReason::kFetchSweep, RemovalReason::kDeduplicate,
                      RemovalReason::kObsoleteLayer, RemovalReason::kExplicitCancel,
                      RemovalReason::kPopDuplicate, RemovalReason::kShutdown}) {
    PrefetchQueueAccounting a;
    ASSERT_TRUE(a.TryAdmit(10, 128, 1024));
    EXPECT_EQ(a.RetireQueued(10, reason), 128);
    EXPECT_EQ(a.RetireQueued(10, reason), 0);  // idempotent terminal transition
    EXPECT_EQ(a.inflight_bytes(), 0);
    EXPECT_EQ(a.removed_bytes(reason), 128);
    EXPECT_TRUE(a.InvariantHolds());
  }
}

TEST(PrefetchQueueAccounting, PopMovesQueuedToRunningThenAllTerminalsBalance) {
  for (auto outcome : {RunningOutcome::kCompleted, RunningOutcome::kEvictionFailed,
                       RunningOutcome::kStateConflict,
                       RunningOutcome::kAlreadyResident,
                       RunningOutcome::kTransferFailed}) {
    PrefetchQueueAccounting a;
    ASSERT_TRUE(a.TryAdmit(10, 256, 1024));
    ASSERT_TRUE(a.MarkStarted(10));
    a.RetireRunning(10, outcome, outcome == RunningOutcome::kCompleted ? 256 : 0);
    EXPECT_EQ(a.inflight_bytes(), 0);
    EXPECT_TRUE(a.InvariantHolds());
  }
}

TEST(PrefetchQueueAccounting, WorkerBoundaryRecordsStdExceptionAsFailure) {
  PrefetchQueueAccounting a;
  ASSERT_TRUE(a.TryAdmit(10, 256, 1024));
  ASSERT_TRUE(a.MarkStarted(10));
  EXPECT_NO_THROW(RunPrefetchTaskNoThrow(
      10, &a, [] { throw std::runtime_error("SetNodeDevice failed"); }));
  EXPECT_EQ(a.inflight_bytes(), 0);
  EXPECT_EQ(a.failed_bytes(), 256);
  EXPECT_TRUE(a.InvariantHolds());
}

TEST(PrefetchQueueAccounting, WorkerBoundaryRecordsUnknownExceptionAsFailure) {
  PrefetchQueueAccounting a;
  ASSERT_TRUE(a.TryAdmit(10, 256, 1024));
  ASSERT_TRUE(a.MarkStarted(10));
  EXPECT_NO_THROW(RunPrefetchTaskNoThrow(10, &a, [] { throw 7; }));
  EXPECT_EQ(a.inflight_bytes(), 0);
  EXPECT_EQ(a.failed_bytes(), 256);
  EXPECT_TRUE(a.InvariantHolds());
}
```

- [ ] **Step 2: Add the test to the repository's configured Torch/CUDA build and verify RED**

Add `option(MOE_BUILD_TESTS "Build native unit tests" OFF)` to root `CMakeLists.txt`; when enabled, call `enable_testing()` and `add_subdirectory(tests/cpp)` after `add_subdirectory(core)`. In `tests/cpp/CMakeLists.txt`, use `find_package(GTest REQUIRED)`, `find_package(CUDAToolkit REQUIRED)`, and `add_subdirectory(unit/prefetch)`. In the leaf CMake file:

```cmake
add_executable(test_prefetch_queue_accounting test_prefetch_queue_accounting.cpp)
target_link_libraries(test_prefetch_queue_accounting PRIVATE
  archer_core GTest::gtest_main CUDA::cudart ${TORCH_LIBRARIES})
target_include_directories(test_prefetch_queue_accounting PRIVATE
  ${PROJECT_SOURCE_DIR}/core ${TORCH_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
include(GoogleTest)
gtest_discover_tests(test_prefetch_queue_accounting)
```

Configure from the repository root so `Torch`, CUDA, Python, CUTLASS, `archer_core`, compile flags, and transitive includes are available:

Run: `cmake -S . -B /tmp/opencode/prefetch-unit-build -DMOE_BUILD_TESTS=ON -DCUTLASS_DIR="$CUTLASS_DIR" && cmake --build /tmp/opencode/prefetch-unit-build --target test_prefetch_queue_accounting -j2`

Expected: compile failure because `PrefetchQueueAccounting` does not exist.

- [ ] **Step 3: Add metadata and thread-safe scheduler APIs**

Add to `Task`:

```cpp
std::uint64_t generation = 0;
std::int64_t layer_id = -1;
std::int64_t scheduled_bytes = 0;
std::int64_t enqueue_ns = 0;
std::int64_t start_ns = 0;
```

Add plain structs:

```cpp
struct PrefetchAdmission {
  std::vector<std::uint32_t> accepted_tensor_ids;
  std::int64_t accepted_bytes = 0;
  std::int64_t rejected_bytes = 0;
  std::int64_t inflight_bytes = 0;
};
struct PrefetchSample {
  std::uint64_t generation;
  std::int64_t layer_id;
  std::uint32_t tensor_id;
  std::int64_t bytes;
  std::int64_t queue_wait_ns;
  std::int64_t transfer_ns;
  std::string source_device;  // disk or cpu at task start
  std::string outcome;  // completed, canceled, superseded, failed, rejected
};
```

Add `AdmitPrefetchTasks`, `CancelQueuedPrefetch(generation, layer_id, keep_nodes)`, `DrainPrefetchSamples`, and `GetInflightPrefetchBytes` to `ArcherTaskPool`. All queue scans and byte transitions occur under `unified_mutex_`; cancellation may remove only non-on-demand tasks still in `unified_queue_[1..]`, never running tasks and never queue 0. Use `steady_clock` nanoseconds, not wall time.

Centralize every mutation in two lock-required helpers:

```cpp
std::int64_t RetireQueuedTaskLocked(const TaskPtr&, RemovalReason);
template <class Predicate>
std::int64_t EraseQueuedIfLocked(std::uint32_t priority, Predicate,
                                 RemovalReason);
```

No direct `deque.clear()` or remove/erase remains. Route all current removal paths through the helpers: `ClearQueue` (`kClear`), `ReplaceCacheCandidates` (`kReplaceCandidates`), `FetchExec`'s priority-1+ sweep (`kFetchSweep`), `EnqueueTask` duplicate/outdated-layer sweep (`kDeduplicate`/`kObsoleteLayer`), `StartExec`'s all-priority promotion sweep (`kFetchSweep`), explicit generation cancellation (`kExplicitCancel`), `GPUThreadFunc`'s chosen-task pop and duplicate sweep (`MarkStarted` for the chosen task, `kPopDuplicate` for the rest), and shutdown/destruction (`kShutdown`). The chosen pop must atomically move bytes from queued to running rather than decrement inflight.

Every post-pop exit also reaches one terminal transition via an RAII guard: eviction failure, `exec_state` CAS failure, already-resident/same-device race, transfer exception, and success. Put the complete `GPUThreadFunc` worker entry behind a `noexcept` catch-all boundary and put each popped task behind `RunPrefetchTaskNoThrow`: catch both `const std::exception&` and `...`, log, mark the task `failed` through the same idempotent terminal helper, clear `is_prefetching`/restore `exec_state` as applicable, and notify an on-demand task's waiter. In particular, an exception from `SetNodeDevice` must become one failed sample and retired byte count and must never unwind through `std::thread` (which would call `std::terminate`). The outermost worker catch is a final defense for pre-pop/loop exceptions; it records a worker-failure counter and returns without throwing. Only success that actually moved bytes increments `completed_transfer_bytes`; canceled, superseded, failed, or already-resident bytes retire inflight but never become completed/waste bytes. Enforce after every mutation:

```text
inflight_bytes == queued_bytes + running_bytes
accepted_bytes == queued_bytes + running_bytes + completed_bytes
                  + canceled_bytes + superseded_bytes + failed_bytes
                  + already_resident_bytes
```

Rejected bytes are outside `accepted_bytes`. A second terminal call for the same task is a zero-byte no-op and emits no duplicate sample.

- [ ] **Step 4: Add compatibility-preserving handle and pybind APIs**

Keep `prefetch_tensors(tensor_ids, priority)` returning `None`. Add:

```cpp
PrefetchAdmission SchedulePrefetchTensors(
    const std::vector<std::uint32_t>& tensor_ids, std::uint32_t priority,
    std::uint64_t generation, std::int64_t layer_id,
    std::int64_t max_inflight_bytes);
std::int64_t CancelPrefetchGeneration(
    std::uint64_t generation, std::int64_t layer_id,
    const std::vector<std::uint32_t>& keep_tensor_ids);
std::vector<PrefetchSample> DrainPrefetchSamples();
std::int64_t GetInflightPrefetchBytes();
```

Bind them as `schedule_prefetch_tensors`, `cancel_prefetch_generation`, `drain_prefetch_samples`, and `get_inflight_prefetch_bytes`. Bind the value structs with read-only fields. Deduplicate tensor IDs before converting to nodes; reject null nodes and invalid priority without mutating accounting.

- [ ] **Step 5: Pass native unit and legacy Python API tests**

Run: `cmake --build /tmp/opencode/prefetch-unit-build --target test_prefetch_queue_accounting -j2 && ctest --test-dir /tmp/opencode/prefetch-unit-build --output-on-failure -R PrefetchQueueAccounting`

Expected: `100% tests passed`, including both `SetNodeDevice`-equivalent standard and unknown exceptions returning normally with failed bytes accounted exactly once.

Run: `pytest -q tests/python/dflash/test_speculative_prefetch.py`

Expected: PASS, proving the existing mock/legacy API contract remains intact.

- [ ] **Step 6: Commit scheduler admission**

```bash
git add CMakeLists.txt core/prefetch/task_scheduler.h core/prefetch/task_scheduler.cpp \
  core/prefetch/archer_prefetch_handle.h core/prefetch/archer_prefetch_handle.cpp \
  core/python/py_archer_prefetch.cpp tests/cpp/CMakeLists.txt tests/cpp/unit/prefetch
git commit -m "feat: bound and observe native prefetch queues"
```

---

### Task 4: Measure native current-layer expert compute intervals

**Files:**
- Create: `core/parallel/expert_timing.h`
- Modify: `core/parallel/expert_dispatcher.h`
- Modify: `core/parallel/expert_dispatcher.cpp`
- Modify: `core/parallel/expert_module.h`
- Modify: `core/parallel/expert_module.cpp:185-229`
- Modify: `core/python/py_archer_prefetch.cpp:109-123`
- Modify: `tests/cpp/CMakeLists.txt`
- Create: `tests/cpp/unit/parallel/test_expert_timing_lifecycle.cpp`
- Create: `tests/cpp/unit/parallel/CMakeLists.txt`
- Modify: `tests/python/dflash/test_prefetch_native_gpu.py`

- [ ] **Step 1: Write failing event-lifecycle tests with a fake CUDA-status adapter**

`expert_timing.h` must expose a small state machine whose CUDA calls are supplied by an adapter, allowing CPU tests to assert which event is queried/destroyed without a GPU:

```cpp
TEST(ExpertTimingLifecycle, NormalStopNotReadyIsPendingAndNeverReused) {
  FakeCuda cuda;
  ExpertTimingLifecycle lifecycle(/*pairs=*/2, &cuda);
  auto first = lifecycle.Acquire();
  lifecycle.MarkStartRecorded(first);
  lifecycle.MarkStopRecorded(first);
  cuda.stop_query = cudaErrorNotReady;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(first), TimingPairState::kPending);
  EXPECT_NE(lifecycle.Acquire().id, first.id);
  EXPECT_EQ(cuda.elapsed_calls, 0);
}

TEST(ExpertTimingLifecycle, ForwardExceptionQuarantinesUntilFenceCompletes) {
  FakeCuda cuda;
  ExpertTimingLifecycle lifecycle(/*pairs=*/1, &cuda);
  auto pair = lifecycle.Acquire();
  lifecycle.MarkStartRecorded(pair);  // stop was never recorded
  lifecycle.QuarantineAfterException(pair, /*stream=*/7);
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kQuarantined);
  EXPECT_FALSE(lifecycle.TryAcquire().has_value());
  EXPECT_EQ(cuda.stop_query_calls, 0);  // unspecified stop is never queried
  cuda.fence_query = cudaErrorNotReady;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kQuarantined);
  cuda.fence_query = cudaSuccess;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kFree);
  EXPECT_EQ(cuda.elapsed_calls, 0);  // exception sample is discarded
}

TEST(ExpertTimingLifecycle, FenceErrorRequiresSuccessfulStreamProof) {
  FakeCuda cuda;
  ExpertTimingLifecycle lifecycle(/*pairs=*/1, &cuda);
  auto pair = lifecycle.Acquire();
  lifecycle.QuarantineAfterException(pair, /*stream=*/7);
  cuda.fence_query = cudaErrorInvalidResourceHandle;
  cuda.stream_sync = cudaErrorLaunchFailure;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kQuarantined);
  cuda.stream_sync = cudaSuccess;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kFree);
}

TEST(ExpertTimingLifecycle, DestructorDoesNotDestroyUnprovenEvents) {
  FakeCuda cuda;
  cuda.stream_sync = cudaErrorLaunchFailure;
  {
    ExpertTimingLifecycle lifecycle(/*pairs=*/1, &cuda);
    auto pair = lifecycle.Acquire();
    lifecycle.QuarantineAfterException(pair, /*stream=*/7);
  }
  EXPECT_EQ(cuda.destroy_timing_calls, 0);
  EXPECT_EQ(cuda.destroy_fence_calls, 0);
}

TEST(ExpertTimingLifecycle, DisabledPolicyCreatesNoCudaTimingObjects) {
  FakeCuda cuda;
  ExpertTimingLifecycle lifecycle(/*enabled=*/false, /*pairs=*/4, &cuda);
  EXPECT_FALSE(lifecycle.TryAcquire().has_value());
  lifecycle.Poll();
  EXPECT_EQ(cuda.create_timing_calls, 0);
  EXPECT_EQ(cuda.create_fence_calls, 0);
  EXPECT_EQ(cuda.record_calls, 0);
  EXPECT_EQ(cuda.query_calls, 0);
  EXPECT_EQ(cuda.elapsed_calls, 0);
}
```

- [ ] **Step 2: Add lifecycle test CMake and verify RED through the root build**

In `tests/cpp/CMakeLists.txt`, add `add_subdirectory(unit/parallel)`. The leaf target links `archer_core`, `${TORCH_LIBRARIES}`, `CUDA::cudart`, and `GTest::gtest_main`, matching Task 3's root configuration.

Run: `cmake -S . -B /tmp/opencode/prefetch-unit-build -DMOE_BUILD_TESTS=ON -DCUTLASS_DIR="$CUTLASS_DIR" && cmake --build /tmp/opencode/prefetch-unit-build --target test_expert_timing_lifecycle -j2`

Expected: compile failure because `core/parallel/expert_timing.h` does not exist.

- [ ] **Step 3: Add the opt-in native telemetry smoke test**

```python
def test_native_compute_samples_are_drainable(offloaded_prefetcher) -> None:
    dispatcher = offloaded_prefetcher.expert_dispatcher
    dispatcher.set_overlap_compute_timing_enabled(True)
    assert dispatcher.drain_compute_samples() == []
    # The module fixture's model runs one deterministic token through existing helper.
    _run_one_token(offloaded_prefetcher)
    samples = dispatcher.drain_compute_samples()
    assert samples
    assert all(s.invocation_id > 0
               and s.kernel_end_offset_ns > s.kernel_start_offset_ns
               and s.kernel_duration_ns ==
                   s.kernel_end_offset_ns - s.kernel_start_offset_ns
               and s.forward_return_host_ns >= 0
               and s.output_complete_host_ns >= s.forward_return_host_ns
               and s.output_delay_ns ==
                   s.output_complete_host_ns - s.forward_return_host_ns
               and s.layer_id >= 0 for s in samples)
    assert len({s.invocation_id for s in samples}) == 1
    assert dispatcher.drain_compute_samples() == []
```

The CPU lifecycle test above is the allocation proof for disabled timing. Do not add a policy-`off` Python call to a timing status/drain binding merely to inspect it; Task 5's strict mock test proves the off path never probes or calls those names.

Refactor the fixture to yield `(model, prefetcher)` and add `_run_one_token` using the fixture's cached tokenizer/input. Keep module scope because native topology is process-global.

- [ ] **Step 4: Run the native smoke test and verify RED**

Run: `MOE_DFLASH_SERVING_GPU=1 pytest -q tests/python/dflash/test_prefetch_native_gpu.py -k compute_samples`

Expected: FAIL with missing `drain_compute_samples`.

- [ ] **Step 5: Implement bounded normal retirement and exception quarantine**

Add a sample that keeps GPU kernel time and host completion delay separate:

```cpp
struct ExpertComputeSample {
  std::uint64_t invocation_id;
  int layer_id;
  int expert_id;
  int gpu_id;
  std::int64_t kernel_start_offset_ns;
  std::int64_t kernel_end_offset_ns;
  std::int64_t kernel_duration_ns;
  std::int64_t forward_return_host_ns;
  std::int64_t output_complete_host_ns;
  std::int64_t output_delay_ns;
};
std::vector<ExpertComputeSample> DrainComputeSamples();
```

Add new methods `set_inputs_with_invocation(...) -> uint64_t`, `set_overlap_compute_timing_enabled(bool)`, and `drain_compute_samples`; retain the legacy `set_inputs` binding and return contract unchanged. `set_inputs_with_invocation` allocates a monotonically increasing nonzero ID and copies it through `CallArgs` and `ExecArgs`, so every sample emitted by every execution thread carries the dispatch that produced it. Active Python uses the new setup method only when the complete timing capability is present; policy `off` always uses legacy `set_inputs`.

Do not create timing objects in the constructor. Keep timing disabled by default. After an active-policy call sets the atomic enable flag, each `GPUExecFunc` worker lazily creates its timing-enabled epoch and four timing-enabled start/stop pairs after its existing `cudaSetDevice(gpu_id)` succeeds, using `cudaEventCreateWithFlags(..., cudaEventDefault)`, then records/synchronizes the epoch once. Never use or release these handles through `kCudaEventPool`. `ExpertTimingLifecycle` owns `free`, `active`, `pending`, and `quarantined` collections; an empty free collection drops telemetry for that invocation rather than blocking or reusing an event. Disabling is restart-only for this release; Python never toggles the flag back at runtime.

Change `MoEMLP::forward(hidden_states, stream)` to accept a timing ticket. Record start immediately before `ForwardHelper(stream)` and stop immediately after it on the same stream:

```cpp
cudaEventRecord(kernel_start, stream);
ForwardHelper(stream);
cudaEventRecord(kernel_stop, stream);
```

Both current stream synchronizations may remain, but timing ownership and retirement must not depend on them. `GPUExecFunc` captures `forward_return_host_ns` immediately when `forward` returns and `output_complete_host_ns` after `OutputFunc`, stores those timestamps on the pending ticket, and polls pending stops between work items. Only `cudaEventQuery(stop) == cudaSuccess` permits `cudaEventElapsedTime(epoch, start)`, `cudaEventElapsedTime(epoch, stop)`, and `cudaEventElapsedTime(start, stop)`, CPU sample publication, and pair reuse. `cudaErrorNotReady` remains pending. All CUDA return codes are checked.

Wrap forward with a ticket whose `start_recorded` and `stop_recorded` flags are set only after successful `cudaEventRecord` calls. Any exception or CUDA record error after acquisition calls `QuarantineAfterException`; it records a new timing-disabled fence on the same execution stream. The quarantined pair is never sampled or reused until fence query succeeds, or until an error-recovery `cudaStreamSynchronize` succeeds. Never query stop when `stop_recorded` is false. Never infer completion from `OutputFunc`, the Python wait barrier, thread join alone, or host time. At destruction, join workers, prove each stream complete, then destroy events; intentionally leak handles belonging to an unproven stream for process teardown.

This instrumentation measures `ForwardHelper` only. It explicitly excludes input D2D copy, any synchronization, output clone, `output.to`, accumulation mutex/body, eviction, pending notification, and Python wait-barrier wakeup. Later removal of `MoEMLP::forward` host synchronizations requires no timing redesign.

- [ ] **Step 6: Pass lifecycle and native smoke tests**

Run: `cmake --build /tmp/opencode/prefetch-unit-build --target test_expert_timing_lifecycle -j2 && ctest --test-dir /tmp/opencode/prefetch-unit-build --output-on-failure -R ExpertTimingLifecycle`

Expected: `100% tests passed`, including no reuse/query/destroy before completion proof and zero CUDA timing calls while disabled.

Run: `pip install --no-build-isolation -e .`

Expected: native extension builds successfully.

Run: `MOE_DFLASH_SERVING_GPU=1 pytest -q tests/python/dflash/test_prefetch_native_gpu.py -k compute_samples`

Expected: PASS on a configured CUDA host; clean SKIP without the opt-in environment. Nsight must show the timing interval boundaries enclosing the kernels inside `expert_compute`, while reported `output_delay_ns` remains separate and non-negative.

- [ ] **Step 7: Commit compute telemetry**

```bash
git add core/parallel/expert_timing.h core/parallel/expert_dispatcher.h \
  core/parallel/expert_dispatcher.cpp \
  core/parallel/expert_module.h core/parallel/expert_module.cpp \
  core/python/py_archer_prefetch.cpp tests/cpp/CMakeLists.txt \
  tests/cpp/unit/parallel tests/python/dflash/test_prefetch_native_gpu.py
git commit -m "feat: expose native expert compute intervals"
```

---

### Task 5: Wire byte admission and exact-route correction into the prefetcher

**Files:**
- Modify: `moe_infinity/memory/expert_prefetcher.py:50-312`
- Modify: `moe_infinity/memory/expert_predictor.py:23-63`
- Modify: `moe_infinity/runtime/model_offload.py:1141-1163`
- Modify: `tests/python/dflash/test_speculative_prefetch.py`

- [ ] **Step 1: Write failing byte-bounded prefetcher tests**

```python
class StrictLegacyExtension:
    def __init__(self, forbidden: set[str]):
        self._forbidden = forbidden
        self.forbidden_accesses: list[str] = []
        self.prefetch_tensors = MagicMock()

    def __getattr__(self, name: str):
        if name in self._forbidden:
            self.forbidden_accesses.append(name)
            raise AssertionError(f"new binding accessed while unavailable: {name}")
        raise AttributeError(name)


def test_enforce_issues_only_whole_experts_inside_budget():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.configure_overlap_policy(_config(policy="enforce"))
    prefetcher.expert_nbytes_map = {(3, 0): 4000, (3, 1): 3000,
                                    (3, 2): 3000, (3, 3): 1000}
    prefetcher.overlap_controller.observe_compute(3, 0, 1000)
    prefetcher.overlap_controller.observe_transfer(1000, 100, 0, 0)
    prefetcher.speculative_prefetch(2, LOGITS)
    assert _enqueued_tensor_ids(engine) == [301, 302]
    assert engine.schedule_prefetch_tensors.call_args.kwargs["layer_id"] == 3


def test_enforce_missing_byte_map_fails_closed():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.configure_overlap_policy(_config(policy="enforce"))
    prefetcher.expert_nbytes_map = {}
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.schedule_prefetch_tensors.assert_not_called()
    engine.prefetch_tensors.assert_not_called()


def test_observe_records_decision_but_preserves_legacy_issue():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.configure_overlap_policy(_config(policy="observe"))
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.prefetch_tensors.assert_called_once_with([301, 302], priority=2)
    assert prefetcher.overlap_prefetch_stats()["decisions"] == 1


def test_off_never_probes_calls_or_drains_new_bindings():
    prefetcher, _ = _make_prefetcher(num_layers=8, num_experts=4)
    engine = StrictLegacyExtension(
        {"schedule_prefetch_tensors", "cancel_prefetch_generation",
         "drain_prefetch_samples", "get_inflight_prefetch_bytes"})
    dispatcher = StrictLegacyExtension(
        forbidden={"set_inputs_with_invocation",
                   "set_overlap_compute_timing_enabled",
                   "drain_compute_samples"})
    prefetcher.archer_engine = engine
    prefetcher.expert_dispatcher = dispatcher
    prefetcher.configure_overlap_policy(_config(policy="off"))
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.prefetch_tensors.assert_called_once()
    assert prefetcher.overlap_controller is None
    assert dispatcher.forbidden_accesses == []
    assert engine.forbidden_accesses == []


def test_observe_with_legacy_extension_records_capability_miss_and_issues_legacy():
    prefetcher, engine = _make_prefetcher(
        num_layers=8, num_experts=4, extension="legacy")
    prefetcher.configure_overlap_policy(_config(policy="observe"))
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.prefetch_tensors.assert_called_once()
    assert prefetcher.overlap_prefetch_stats()["native_capability_misses"] == 1


def test_enforce_with_legacy_extension_fails_closed_without_new_or_legacy_issue():
    prefetcher, engine = _make_prefetcher(
        num_layers=8, num_experts=4, extension="legacy")
    prefetcher.configure_overlap_policy(_config(policy="enforce"))
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.prefetch_tensors.assert_not_called()
    assert prefetcher.overlap_prefetch_stats()["native_capability_misses"] == 1
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/dflash/test_speculative_prefetch.py -k 'enforce or observe or off_never or legacy_extension'`

Expected: FAIL because configuration/controller methods do not exist.

- [ ] **Step 3: Add controller ownership and candidate costing**

`configure_overlap_policy(config)` returns immediately for `off`: leave `overlap_controller=None`, do not use `hasattr`/`getattr` on any new binding name, do not enable timing, and retain the exact legacy issuance path. For `observe|enforce`, build one controller and a monotonic generation counter, then perform one capability discovery for the complete method sets (scheduler: `schedule_prefetch_tensors`, `cancel_prefetch_generation`, `drain_prefetch_samples`, `get_inflight_prefetch_bytes`; dispatcher timing: `set_inputs_with_invocation`, `set_overlap_compute_timing_enabled`, `drain_compute_samples`). Store booleans/callables in a `NativeOverlapCapabilities` adapter so later code never performs ad-hoc calls. Only when all dispatcher timing methods exist, call `set_overlap_compute_timing_enabled(True)`. Define `enforce_ready` as both complete scheduler and complete dispatcher-timing capabilities; absent either, `enforce` admits nothing because it cannot obtain the required native accounting/calibration telemetry. An older extension is valid: `observe` records a capability miss and continues the original `prefetch_tensors` behavior with no native timing drain; `enforce` calls neither new nor legacy scheduling APIs. Add helpers with these exact contracts:

```python
def plan_candidates(self, layer_id: int,
                    ranked_expert_ids: list[int],
                    scores: Optional[list[float]] = None) -> tuple[int, list[int]]: ...
def correct_to_native_route(self, layer_id: int,
                            actual_expert_ids: list[int]) -> None: ...
def abort_prefetch_generations(self, generations: list[int],
                               reason: str) -> None: ...
def drain_native_prefetch_samples(self) -> None: ...
def observe_compute_samples(self, samples: list[Any]) -> None: ...
def overlap_prefetch_stats(self) -> dict[str, Any]: ...
```

`plan_candidates` reads exact costs and, only through a stored active-policy capability, queries native inflight bytes, asks the controller, and in `enforce` calls `schedule_prefetch_tensors(..., generation=gen, layer_id=layer_id, max_inflight_bytes=...)`. Track only IDs actually returned as accepted. In `observe`, issue the original IDs through `prefetch_experts_list`; in `off`, do not instantiate/call the controller and do not discover, enable, call, or drain any new native API.

Return the issued generation from policy-aware `speculative_prefetch`/`trigger_speculative_prefetch` (`None` when no bounded native issue occurred). Existing callers ignore the return value. This lets the executor attach exact ownership to `_pending_prefetch` instead of canceling by layer or global queue on errors.

`correct_to_native_route` first drains completions, snapshots residency with `is_tensor_on_device(tensor_id)`, records coverage/late/uncovered/waste, and calls `cancel_prefetch_generation` for prior generations targeting this layer while keeping actual tensor IDs. It returns `None`; callers must continue using the unchanged native `expert_list`. Canceled-before-start false positives increase canceled bytes only. A false positive contributes to `wasted_prefetch_bytes` only if a native completion sample for that expert was drained before correction.

`abort_prefetch_generations` cancels only the exact generation IDs supplied by the executor, records `reason`, and drains native samples in its own `finally` only when the active-policy scheduler capability exists. It is idempotent so nested cleanup cannot double-retire bytes. Do not cancel by layer range or global queue because another request may own unrelated work. Every drain/cancel/telemetry method starts with an active-policy capability guard; `off` and missing methods return without touching the extension object.

- [ ] **Step 4: Preserve predictor authority and scoring**

Add `ExpertPredictor.ranked_candidates(expert_matrix, layer_idx)` returning `(expert_id, float_score)` sorted from the existing matrix row. Do not change `predict()`'s return value or trace similarity logic. Legacy router-logit mode uses mean logits as scores; explicit DFlash mode assigns stable descending scores preserving supplied union order. No new model/router call is introduced.

- [ ] **Step 5: Wire config after exact byte-map registration**

In `model_offload.py`, call `expert_prefetcher.configure_overlap_policy(self.archer_config)` immediately after `expert_nbytes_map` is assigned. Continue attaching the prefetcher to the executor only under the existing `speculative_prefetch` condition; policy does not auto-enable it.

- [ ] **Step 6: Pass prefetcher tests and legacy characterization**

Run: `pytest -q tests/python/dflash/test_speculative_prefetch.py tests/python/unit/test_overlap_budget.py`

Expected: PASS, including all existing mean/top-k and priority-band characterization tests.

- [ ] **Step 7: Commit Python policy wiring**

```bash
git add moe_infinity/memory/expert_prefetcher.py \
  moe_infinity/memory/expert_predictor.py moe_infinity/runtime/model_offload.py \
  tests/python/dflash/test_speculative_prefetch.py
git commit -m "feat: budget expert candidates by overlap bytes"
```

---

### Task 6: Apply correction at dispatch without changing native routing

**Files:**
- Modify: `moe_infinity/distributed/expert_executor.py:123-291`
- Modify: `tests/python/dflash/test_route_ahead_wire.py:106-298`
- Modify: `tests/python/dflash/test_route_ahead_metrics.py:130-347`

- [ ] **Step 1: Write failing dispatch-authority and one-layer-pin tests**

```python
def test_partial_exact_route_prefetch_still_dispatches_full_native_union():
    prefetcher = MagicMock()
    prefetcher.plan_candidates.return_value = (41, [0, 2])
    executor = _make_executor(prefetcher=prefetcher)
    with route_ahead_context(prefetcher=prefetcher):
        _dispatch(executor, router_logits=LOGITS)
    prefetcher.fetch_experts_lock_cache.assert_called_once_with(LAYER_ID, [0, 2])
    assert _enqueued_experts(executor) == UNION


def test_empty_budget_does_not_replace_pin_or_suppress_dispatch():
    prefetcher = MagicMock()
    prefetcher.plan_candidates.return_value = (42, [])
    executor = _make_executor(prefetcher=prefetcher)
    with route_ahead_context(prefetcher=prefetcher):
        _dispatch(executor)
    prefetcher.fetch_experts_lock_cache.assert_not_called()
    assert _enqueued_experts(executor) == UNION


def test_exact_route_correction_precedes_every_expert_enqueue():
    events = []
    prefetcher = MagicMock()
    prefetcher.correct_to_native_route.side_effect = lambda l, ids: events.append("correct") or ids
    executor = _make_executor(prefetcher=prefetcher)
    executor.expert_dispatcher.enqueue_expert.side_effect = lambda *a: events.append("enqueue")
    _dispatch(executor)
    assert events[0] == "correct"


def test_wait_error_cancels_owned_generations_drains_and_reraises():
    prefetcher = MagicMock()
    executor = _make_executor(prefetcher=prefetcher)
    executor._pending_prefetch = (
        prefetcher, LAYER_ID, UNION, None, [41, 42], None)
    executor.expert_dispatcher.wait_expert.side_effect = RuntimeError("wait failed")
    with pytest.raises(RuntimeError, match="wait failed"):
        executor.wait_dispatch_local()
    prefetcher.abort_prefetch_generations.assert_called_once_with(
        [41, 42], reason="wait_expert_error")
    prefetcher.drain_native_prefetch_samples.assert_called()
    assert executor._pending_prefetch is None


def test_cleanup_error_does_not_mask_wait_error():
    prefetcher = MagicMock()
    prefetcher.abort_prefetch_generations.side_effect = RuntimeError("cleanup failed")
    executor = _make_executor(prefetcher=prefetcher)
    executor._pending_prefetch = (
        prefetcher, LAYER_ID, UNION, None, [41], None)
    executor.expert_dispatcher.wait_expert.side_effect = RuntimeError("wait failed")
    with pytest.raises(RuntimeError, match="wait failed"):
        executor.wait_dispatch_local()
    prefetcher.drain_native_prefetch_samples.assert_called()


def test_delayed_compute_sample_from_prior_invocation_is_not_calibration():
    prefetcher = MagicMock()
    executor = _make_executor(prefetcher=prefetcher, policy="observe")
    executor._pending_prefetch = (
        prefetcher, LAYER_ID, UNION, None, [], 102)
    executor.expert_dispatcher.drain_compute_samples.return_value = [
        _compute_sample(invocation_id=101, layer_id=LAYER_ID),
        _compute_sample(invocation_id=102, layer_id=LAYER_ID),
    ]
    executor.wait_dispatch_local()
    prefetcher.observe_compute_samples.assert_called_once()
    observed = prefetcher.observe_compute_samples.call_args.args[0]
    assert [s.invocation_id for s in observed] == [102]
    prefetcher.record_stale_compute_samples.assert_called_once_with(1)


def test_same_layer_sample_without_matching_invocation_is_rejected():
    prefetcher = MagicMock()
    executor = _make_executor(prefetcher=prefetcher, policy="enforce")
    executor._pending_prefetch = (
        prefetcher, LAYER_ID, UNION, None, [], 202)
    executor.expert_dispatcher.drain_compute_samples.return_value = [
        _compute_sample(invocation_id=201, layer_id=LAYER_ID),
    ]
    executor.wait_dispatch_local()
    prefetcher.observe_compute_samples.assert_not_called()
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/dflash/test_route_ahead_wire.py -k 'partial_exact or empty_budget or correction_precedes or wait_error or cleanup_error or delayed_compute or matching_invocation'`

Expected: FAIL because dispatch does not yet call correction/planning.

- [ ] **Step 3: Correct before enqueue and budget exact-route warming**

On the supported eager path, derive `expert_list`, then before input setup/enqueue, resolve the prefetcher and call `correct_to_native_route(layer_id, expert_list)` when policy is not `off`. When the stored active-policy dispatcher capability is complete, call `set_inputs_with_invocation(...)` and retain its returned ID; otherwise call the existing `set_inputs(...)` and use `invocation_id=None`. Never probe the new method on the `off` path. In `_maybe_route_ahead_prefetch`, pass the exact union to `plan_candidates`; pin and issue only the admitted IDs, and report those IDs as `predicted_ids`. Preserve `expert_list` for `expected_wait_cnt` and the enqueue loop. Under policy `off`, retain the existing exact-union pin/prefetch sequence without extra calls. Do not add a GPU-route readiness adapter in this release; config validation prevents that combined path.

Change `_maybe_route_ahead_prefetch` to return `(fired, issued_generations)` and make the overlap-triggered legacy path capture the generation returned by `trigger_speculative_prefetch`. Store their concatenation plus `invocation_id` in `_pending_prefetch`; policy `off` stores an empty generation list and `None` invocation and continues to use legacy input setup. Exact-route correction and enqueue behavior remain unchanged.

Never call `ReplaceCacheCandidates` for an empty set. Never include a future layer in the current pin. Keep the existing legacy pooled-call suppression when any exact-route prefetch fired; if budget admits zero, leave normal pending/on-demand behavior active.

- [ ] **Step 4: Finalize compute and policy accounting on success and error**

Extend `_pending_prefetch` with the exact generation IDs issued by this dispatch. Structure `wait_dispatch_local` so `_pending_prefetch` is detached/cleared before waiting and all policy cleanup happens in `finally`:

```python
pending = self._pending_prefetch
self._pending_prefetch = None

def finalize_policy(wait_succeeded: bool) -> None:
    prefetcher, layer_id, expert_list, router_logits, generations, invocation_id = (
        pending if pending is not None else (None, -1, [], None, [], None)
    )
    compute_samples = []
    try:
        if prefetcher is not None and not wait_succeeded:
            prefetcher.abort_prefetch_generations(
                generations, reason="wait_expert_error")
        if prefetcher is not None and invocation_id is not None:
            drained = self.expert_dispatcher.drain_compute_samples()
            compute_samples = [
                sample for sample in drained
                if sample.invocation_id == invocation_id
                and sample.layer_id == layer_id
            ]
            prefetcher.record_stale_compute_samples(len(drained) - len(compute_samples))
        if prefetcher is None:
            return
        if wait_succeeded and compute_samples:
            prefetcher.observe_compute_samples(compute_samples)
        if wait_succeeded:
            prefetcher.correct_prefetch(layer_id + 1, expert_list)
            if router_logits is not None:
                self.trigger_speculative_prefetch(layer_id, router_logits)
    finally:
        if prefetcher is not None:
            prefetcher.drain_native_prefetch_samples()

try:
    result = self.expert_dispatcher.wait_expert()
except BaseException:
    try:
        finalize_policy(False)
    except BaseException:
        logger.exception("overlap-prefetch cleanup failed after wait error")
    raise  # preserve the original wait_expert traceback
else:
    finalize_policy(True)
    return result
```

Preserve the original `wait_expert` exception if cleanup also fails. Always clear pending state, cancel only owned generations, and drain native prefetch samples when the active capability exists so accepted/canceled/completed bytes reach terminal accounting. Call `drain_compute_samples` only when `invocation_id` is non-`None`; this makes policy `off` and active policy on an older extension avoid the binding entirely. On success, accept only the exact `(invocation_id, layer_id)` pair, count all other drained records as stale/delayed telemetry, group accepted samples by destination GPU, and compute each GPU's union from kernel offsets; never use output/host delay. A same-layer sample from an older invocation must never update EWMA. On failure, drain (only when capable) and discard compute samples for calibration because the layer did not complete successfully.

- [ ] **Step 5: Extend route metrics for partial byte coverage**

Update tests so `RouteAheadStats.observe_layer` receives only admitted IDs and exact byte costs. Add assertions that route coverage may be below 1.0 while dispatched experts and outputs remain unchanged; add completed unused bytes as waste, admitted unfinished actual bytes as late, and canceled queued bytes separately.

- [ ] **Step 6: Run route, metrics, and output-equivalence tests**

Run: `pytest -q tests/python/dflash/test_route_ahead_wire.py tests/python/dflash/test_route_ahead_metrics.py tests/python/integration/test_output_equivalence.py`

Expected: PASS or the repository's existing environment-based SKIP for integration models. Unit tests must pass on CPU.

- [ ] **Step 7: Commit dispatch integration**

```bash
git add moe_infinity/distributed/expert_executor.py \
  tests/python/dflash/test_route_ahead_wire.py \
  tests/python/dflash/test_route_ahead_metrics.py
git commit -m "feat: reconcile prefetch with native expert routes"
```

---

### Task 7: Publish policy and transfer metrics through existing profiling

**Files:**
- Modify: `moe_infinity/profiling/io_profiler.py:151-224`
- Modify: `moe_infinity/memory/expert_tracer.py:25-36,153-224`
- Create: `tests/python/unit/test_overlap_prefetch_profiler.py`

- [ ] **Step 1: Write failing instant-event and aggregation tests**

```python
def test_record_emits_overlap_decision_without_timing_context(monkeypatch):
    p = IOProfiler(pid=1)
    p.enabled = True
    p.record("prefetch_budget", layer=3, bytes=4096,
             fields={"budget_bytes": 8192, "generation": 7})
    assert p._events == [{
        "ts_ns": p._events[0]["ts_ns"], "stage": "prefetch_budget",
        "layer": 3, "expert": None, "dur_ns": 0, "bytes": 4096,
        "budget_bytes": 8192, "generation": 7,
    }]


def test_tracer_aggregates_overlap_byte_stages():
    tracer = ExpertTracer.__new__(ExpertTracer)
    tracer._io_profiling_enabled = True
    tracer._io_events = deque(maxlen=10)
    tracer.record_io_event(3, -1, "prefetch_late", 0, 300)
    assert tracer.get_io_stats()["prefetch_late"]["total_bytes"] == 300
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/unit/test_overlap_prefetch_profiler.py`

Expected: FAIL because `IOProfiler.record` and stages are absent.

- [ ] **Step 3: Add a public instant-event API and stages**

Implement `IOProfiler.record(stage, *, layer=None, expert=None, bytes=0, fields=None)` using `_emit_event`; reject field keys that overwrite `ts_ns`, `stage`, `layer`, `expert`, `dur_ns`, or `bytes`. Add tracer stages `prefetch_budget`, `prefetch_admit`, `prefetch_complete`, `prefetch_cancel`, `prefetch_late`, and `prefetch_waste`. Emit one event at each controller/native transition, not one per polling loop.

- [ ] **Step 4: Pass profiler tests**

Run: `pytest -q tests/python/unit/test_overlap_prefetch_profiler.py tests/python/unit/test_overlap_budget.py`

Expected: PASS with deterministic counters and no sleeps.

- [ ] **Step 5: Commit profiling support**

```bash
git add moe_infinity/profiling/io_profiler.py moe_infinity/memory/expert_tracer.py \
  tests/python/unit/test_overlap_prefetch_profiler.py
git commit -m "feat: report overlap prefetch effectiveness"
```

---

### Task 8: Add deterministic report gates and CUDA overlap benchmarks

**Files:**
- Create: `benchmarks/expert_io_microbench/bench_overlap_prefetch.py`
- Create: `tests/python/dflash/test_overlap_prefetch_report.py`
- Modify: `benchmarks/expert_io_microbench/run_decision_profile.py:24-38,83-92,157-178`
- Modify: `benchmarks/expert_io_microbench/run_all.py`
- Modify: `benchmarks/expert_io_microbench/README.md`
- Modify: `docs/environment-variables.md`

- [ ] **Step 1: Write a failing pure report-gate test**

```python
from benchmarks.expert_io_microbench.bench_overlap_prefetch import build_report


def test_report_uses_measured_metrics_without_speedup_promise():
    report = build_report(
        policy="enforce", latencies_ms=[10.0, 11.0],
        stats={"route_bytes": 1000, "covered_route_bytes": 700,
               "wasted_prefetch_bytes": 100, "late_prefetch_bytes": 200,
               "admitted_bytes": 900, "budget_bytes": 1200,
               "queue_rejected_bytes": 50},
        hardware={"gpu": "test", "pcie": "test"})
    assert report["metrics"]["coverage"] == pytest.approx(0.7)
    assert report["metrics"]["waste_ratio"] == pytest.approx(1/9)
    assert report["metrics"]["late_ratio"] == pytest.approx(0.2)
    assert report["verdict"] == "MEASURED"
    assert "speedup" not in report
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/dflash/test_overlap_prefetch_report.py`

Expected: FAIL because `bench_overlap_prefetch` is absent.

- [ ] **Step 3: Implement paired benchmark and schema**

`bench_overlap_prefetch.py` must:

1. Require local model/offload paths and never download at import time.
2. Run the same deterministic prompt, seed, warmups, token count, and iteration count for `off`, `observe`, and `enforce` in separate processes because native topology is process-global.
3. Reset cache between measured arms through existing `reset_cache`, synchronize CUDA at arm boundaries, and record commit/model/hardware/PCIe metadata.
4. Emit p50/p95 per-token latency, throughput, budget/admission/completion/coverage/waste/late/cancel/reject bytes, EWMAs, and Nsight capture path.
5. Label results `MEASURED`; do not assert a speedup. Fail only on correctness mismatch, malformed/missing metrics, queue accounting inconsistency, or crashes.

Use `off` output IDs as the oracle and require `torch.equal` for `observe` and `enforce`. Add invariants `completed <= admitted`, `covered <= route`, `late <= route`, and `wasted <= completed`.

- [ ] **Step 4: Extend the decision-profile runner**

Add `--overlap-prefetch-policy {off,observe,enforce}` and numeric knob flags. Include the exact config in model construction and append `expert_prefetcher.overlap_prefetch_stats()` to JSON. Preserve existing flags and defaults. `run_all.py --scenario overlap` shells out to the new benchmark and merges its JSON without altering other scenarios.

- [ ] **Step 5: Pass CPU report tests**

Run: `pytest -q tests/python/dflash/test_overlap_prefetch_report.py`

Expected: PASS.

- [ ] **Step 6: Run the opt-in native API test**

Run: `MOE_DFLASH_SERVING_GPU=1 pytest -q tests/python/dflash/test_prefetch_native_gpu.py`

Expected: PASS on a configured CUDA host, including bounded acceptance, cancel-before-start, no cancel-after-start, drain-once telemetry, kernel timing with timing-enabled dispatcher-owned events, non-negative separate output delay, and legacy calls; otherwise clean SKIP. CUDA API checks must fail the test if a timing-disabled pooled event is accidentally passed to `cudaEventElapsedTime`.

- [ ] **Step 7: Capture the paired CUDA/Nsight benchmark**

Run:

```bash
python benchmarks/expert_io_microbench/bench_overlap_prefetch.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /path/to/existing/offload \
  --policies off observe enforce --warmup 3 --iters 10 --max-new-tokens 32 \
  --output-json /tmp/overlap-prefetch.json
```

Expected: all three arms report identical output IDs and `MEASURED`; `enforce.metrics` contains non-negative coverage/waste/late/cancel/reject bytes. No required latency direction.

Run:

```bash
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  -o /tmp/overlap-prefetch-enforce \
  python benchmarks/expert_io_microbench/run_decision_profile.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /path/to/existing/offload --hardware-tag local \
  --mode host-only --speculative-prefetch --speculative-prefetch-overlap \
  --overlap-prefetch-policy enforce \
  --output-json /tmp/overlap-prefetch-profile.json
```

Expected: the report contains `cpu_to_gpu`, `expert_compute`, and policy NVTX/IO metrics sufficient to calculate observed overlap; absence of any required range is `BLOCKED`, not a fabricated zero.

- [ ] **Step 8: Document benchmark interpretation and environment**

Add the commands above to the microbench README. State that a candidate rollout compares measured latency/throughput distributions and effectiveness metrics on each target model/hardware combination; no universal threshold or speedup is promised. Add only benchmark opt-ins to `docs/environment-variables.md`; runtime tuning stays in config.

- [ ] **Step 9: Commit benchmark and report gates**

```bash
git add benchmarks/expert_io_microbench/bench_overlap_prefetch.py \
  benchmarks/expert_io_microbench/run_decision_profile.py \
  benchmarks/expert_io_microbench/run_all.py \
  benchmarks/expert_io_microbench/README.md docs/environment-variables.md \
  tests/python/dflash/test_overlap_prefetch_report.py \
  tests/python/dflash/test_prefetch_native_gpu.py
git commit -m "bench: measure overlap-aware expert prefetch"
```

---

### Task 9: Roll out independently with explicit rollback and final gates

**Files:**
- Modify: `docs/configuration.md`
- Test: all files listed below

- [ ] **Step 1: Run CPU deterministic gates**

Run:

```bash
pytest -q \
  tests/python/unit/test_overlap_budget.py \
  tests/python/unit/test_overlap_prefetch_profiler.py \
  tests/python/unit/test_utils_config.py \
  tests/python/dflash/test_speculative_prefetch.py \
  tests/python/dflash/test_route_ahead_wire.py \
  tests/python/dflash/test_route_ahead_metrics.py \
  tests/python/dflash/test_overlap_prefetch_report.py
```

Expected: PASS. These tests use deterministic clocks, bytes, scores, and mocked native samples—no sleeps, network, or GPU.

- [ ] **Step 2: Run native build and queue-accounting gates**

Run:

```bash
pip install --no-build-isolation -e .
cmake -S . -B /tmp/opencode/prefetch-unit-build \
  -DMOE_BUILD_TESTS=ON -DCUTLASS_DIR="$CUTLASS_DIR"
cmake --build /tmp/opencode/prefetch-unit-build \
  --target test_prefetch_queue_accounting test_expert_timing_lifecycle -j2
ctest --test-dir /tmp/opencode/prefetch-unit-build \
  --output-on-failure -R 'PrefetchQueueAccounting|ExpertTimingLifecycle'
```

Expected: extension build succeeds with the repository's Torch/CUDA/Python/CUTLASS configuration and CTest reports `100% tests passed`; queue byte invariants hold and timing pairs are never queried/reused/destroyed before completion proof.

- [ ] **Step 3: Run routing/output correctness gates**

Run: `pytest -q tests/python/integration/test_output_equivalence.py tests/python/dflash/test_qwen35_hybrid_rollback.py`

Expected: PASS where fixtures are available, otherwise documented environment SKIPs. For each enabled fixture, compare `off` and `enforce` output tensors exactly and assert identical expert enqueue lists.

- [ ] **Step 4: Apply rollout stages per model/hardware pair**

1. Ship code with `overlap_prefetch_policy="off"`; verify legacy latency and native queue behavior.
2. Enable `observe`; require output equality and complete non-negative metrics for at least the benchmark's configured 10 measured iterations. Compare decision overhead and budget distributions, but do not block on a predetermined speedup.
3. Enable `enforce` only for the same model/hardware pair with `gpu_only_expert_routing=False`; examine p50/p95 latency, throughput, coverage, waste, late bytes, cancellation, queue rejection, and cache-pressure warnings together.
4. Keep independent allowlisting by model, quantization/storage format, GPU, PCIe generation/width, and host-vs-disk mode. DFlash and non-DFlash runs receive separate evidence; neither gates the other's availability.

- [ ] **Step 5: Document rollback**

Rollback requires no code or router change: set `overlap_prefetch_policy="off"` and restart the worker. This restores the old `prefetch_tensors` path and disables controller calls, cancellation, bounded admission, and telemetry polling. If the rebuilt extension itself is suspect, deploy the prior package because old Python never calls the new binding names. Do not use `observe` as rollback: it still executes controller overhead.

Document rollback triggers as measured regressions or instability rather than promises: output mismatch, native queue/accounting invariant failure, crash/deadlock, sustained cache-lock warnings, unbounded inflight gauge, or model/hardware-specific latency/throughput regression judged unacceptable by the owner.

- [ ] **Step 6: Record risks and mitigations in configuration docs**

| Risk | Detection | Mitigation |
| --- | --- | --- |
| EWMA reacts to transient bandwidth | bandwidth/queue EWMAs and p95 late bytes | conservative safety factor; per-process warmup; switch to `off` |
| Disk and host transfer samples mix | sample source in telemetry and host-only/disk benchmark arms | treat the EWMA as conservative effective end-to-end bandwidth, reset calibration when offload mode changes, and retain the safety factor |
| Queue debt double-counts capacity | inflight invariant and queue-reject bytes | subtract native gauge; native max-inflight admission under one lock |
| Cancellation races with worker pop | cancel-after-start test | only remove queued tasks under scheduler mutex; running tasks complete |
| Queue removal bypasses byte retirement | per-reason invariant tests and zero inflight after reset/replacement | prohibit direct clear/remove-erase; use centralized terminal helpers on every path |
| `SetNodeDevice` or another worker operation throws | standard/unknown-exception no-throw worker tests and failed-byte invariant | per-task RAII failure retirement plus outer `GPUThreadFunc` `catch (...)`; never let an exception escape `std::thread` |
| Timing event accidentally comes from disabled-timing pool | CUDA elapsed-time smoke and checked CUDA return codes | dispatcher owns timing-enabled epoch/start/stop events; never pool them |
| Forward throws with timing events in unspecified state | fake-CUDA quarantine lifecycle tests | fence the same stream; retain until query/synchronize proves completion; never sample exception tickets |
| Output/host delay inflates compute budget | kernel offsets versus output-delay fields | time only `ForwardHelper`; exclude `OutputFunc` and host completion from EWMA |
| Delayed sample is attributed to a newer execution of the same layer | mixed-invocation unit test | stamp every sample with nonzero invocation ID and calibrate only an exact invocation/layer match |
| GPU-only routing conflicts with correction/event ownership | config matrix tests | reject `gpu_only_expert_routing=True` with `observe` or `enforce` in the first release |
| `wait_expert` throws with admitted work outstanding | wait-error/failing-cleanup tests | cancel owned generations and drain native samples in error/finally paths without masking original error |
| Missing/incorrect expert sizes | uncosted candidate metric | fail closed in `enforce`; never use an average |
| Partial exact-route warming lowers apparent coverage | route/output equality tests | keep full native enqueue list; treat coverage as effectiveness only |
| Pinning causes cache pressure | occupancy/waste/warnings | one current layer only; never pin empty/multiple layers |
| Telemetry adds hot-path overhead | `off` vs `observe` issuance/latency | drain in batches after barrier; default `off` |
| Multi-GPU bandwidth differs | per-device samples | key bandwidth/inflight state by destination GPU before multi-GPU enablement |
| Native extension version skew | capability detection | `enforce` fails closed; `off` uses legacy API |
| Default-off rollout still adds timing or binding overhead | disabled-lifecycle CUDA-call counters and strict legacy-extension mocks | do not instantiate timing/controller state or probe/call/drain new bindings in `off`; gate all new APIs to `observe|enforce` |

- [ ] **Step 7: Commit rollout documentation**

```bash
git add docs/configuration.md
git commit -m "docs: define overlap prefetch rollout and rollback"
```

- [ ] **Step 8: Final repository verification**

Run: `python -m compileall -q moe_infinity benchmarks/expert_io_microbench`

Expected: exit 0.

Run: `git diff --check`

Expected: no output.

Run: `git status --short`

Expected: clean after the planned commits. Stop after this successful verification; do not rerun the suite without a new change.

## Acceptance checklist

- Native route masks, weights, and complete expert enqueue lists are identical with policy `off`, `observe`, and `enforce`.
- Warm admission uses exact expert bytes and the stated compute/bandwidth/queue/issue/inflight formula.
- Cold start is capped and missing bytes/telemetry fail closed under `enforce`.
- Exact-route correction happens before expert enqueue, cancels only queued false positives, and never cancels on-demand/running work.
- First-release config accepts eager routing with active overlap policy and GPU-only routing with policy `off`, but rejects simultaneous `gpu_only_expert_routing=True` and `observe|enforce` without silent downgrade.
- At most one current layer's admitted exact-route set is pinned; empty sets never replace pins.
- Dispatcher-owned timing-enabled epoch/start/stop events bracket only `ForwardHelper` on its stream and remain correct without `MoEMLP::forward` host synchronizations.
- Policy `off` creates/records/queries no timing objects and never probes, calls, or drains a new scheduler/dispatcher binding; older extensions retain their legacy behavior. Only `observe|enforce` perform one-time capability discovery, with observe falling back to legacy issuance and enforce failing closed.
- Every `ExpertComputeSample` carries a nonzero invocation ID. Calibration consumes only the exact pending `(invocation_id, layer_id)` and rejects/counts delayed samples, including an older invocation of the same layer.
- Normal timing pairs retire only after stop query succeeds. Exception-path pairs are quarantined, never sampled or reused, and are destroyed only after a same-stream fence or stream synchronization proves completion; unproven shutdown handles are intentionally left for process teardown.
- `OutputFunc` and host completion delay are reported separately and never contribute to `compute_ewma_ns`.
- Backpressure is atomically enforced in native code; every clear/replacement/sweep/dedup/obsolete/cancel/pop/shutdown and post-pop terminal path retires bytes exactly once; inflight accounting returns to zero.
- `GPUThreadFunc` and its per-task execution boundary catch standard and unknown exceptions; `SetNodeDevice` failures become exactly one failed terminal/accounting sample and no exception escapes a native `std::thread`.
- Coverage, waste, late, canceled, rejected, candidate, admitted, and completed bytes follow the definitions in this plan; only completed false positives count as waste.
- `wait_dispatch_local` clears pending state and cancels/drains owned generations on every wait error; cleanup failure never masks the original `wait_expert` exception.
- CPU tests are deterministic; CUDA tests are opt-in and skip cleanly; benchmark reports are labeled measured.
- Native queue tests configure from repository root and link the actual `archer_core`, Torch, CUDA runtime, Python, CUTLASS, and GoogleTest environment.
- Rollout and rollback do not depend on DFlash or unfinished GPU routing; combined active overlap/GPU-only routing is explicitly deferred and rejected.
- FineMoE, PreScope, and SpecPrefetch remain motivation citations only.
