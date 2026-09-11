# Adaptive Expert/KV GPU Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, bounded controller that safely reallocates a fixed per-GPU VRAM budget between the expert cache and KV cache using measured expert-miss cost, KV pressure/swap cost, and a hard free-memory reserve.

**Architecture:** Keep policy calculation in a deterministic, CPU-only controller with independent state and targets for each CUDA device, and keep mutation in explicit serving/native adapters. A device-local resize is a transactional maintenance window: close scheduler and dispatcher admissions, drain request plus native fetch/execute queues, wait for transfer completion, record and synchronize completion events on every relevant CUDA stream, reserve eligible expert victims, shrink the donor, verify the hard reserve, grow the receiver, publish the effective per-device state, and reopen admissions. Failures before donor release roll back completely; failures after irreversible expert eviction honestly commit the smaller donor with an unchanged receiver rather than pretending evicted experts were restored. The feature is disabled by default, rate-limited, hysteretic, bounded by minimum capacities and a maximum step, observable through existing stats/Prometheus/config surfaces, and never changes KV/expert precision or model outputs.

**Tech Stack:** Python 3.10+, dataclasses, PyTorch/CUDA, existing MoE-Infinity paged KV and scheduler APIs, C++17/CUDA native extension with pybind11, pytest, deterministic trace simulation, JSON benchmark reports.

---

## Motivation and non-goals

[WiSP](https://arxiv.org/abs/2606.21868) motivates treating routed experts and KV as competing VRAM working sets. It is background motivation only: this implementation does not copy WiSP's policy, reproduce its reported results, or assume any gain.

Out of scope: KV quantization, expert/KV precision changes, unbounded optimization, per-token reallocations, resizing while a CUDA kernel or transfer references the affected storage, and performance claims without measured repository-local evidence.

## Current source map and constraints

Source map validated against `origin/main@b766f8f1f6379fac6cd23594713ba6f4c7650ad9` on 2026-08-21. Line ranges below refer to that revision.

- `moe_infinity/serving/memory_manager.py:98-152` computes one fixed split through `compute_budget()` and `get_expert_cache_ratio()`; `MemoryBudget` already subtracts model bytes and activation reserve.
- `moe_infinity/serving/engine.py:100-149` builds a fixed `PagedKVCache`; `get_stats()` at lines 653-668 and `update_config()` at lines 679-690 are the serving observability/reload seams.
- `moe_infinity/serving/kv_cache.py:48-138` owns block IDs/tables and lines 140-526 own the contiguous KV tensor, swap state, and independent `_fi_prefill`/`_fi_decode` wrappers. Replacing only `_kv_cache`, or reusing either wrapper after the page count changes, is unsafe.
- `moe_infinity/serving/scheduler.py:230-328` allocates/preempts at step boundaries; lines 474-539 swap running groups. This is the safe quiescence owner for serving KV resize.
- `moe_infinity/memory/memory_coordinator.py` independently computes fixed native expert/KV ratios.
- `moe_infinity/entrypoints/big_modeling.py:389-425` turns the native KV ratio into `KVCacheManager` and `PagedAttentionBackend` capacities.
- `moe_infinity/memory/kv_cache_manager.py` tracks referenced native KV blocks (`KVCacheBlock.ref_cnt` in `block_pool.py`) and swap metadata; `moe_infinity/engine/scheduler.py:305-392` owns native preemption/swap.
- `moe_infinity/runtime/attention_backend.py:103-163` allocates three physical stores (`k_cache`, `v_cache`, and optional `_fi_kv_cache`) and creates FlashInfer prefill/decode wrappers; lines 398-417 execute decode through those wrappers, while lines 520-587 plan them. A resize that changes only `k_cache`/`v_cache` is incomplete and unsafe.
- `core/prefetch/task_scheduler.cpp:255-329` evicts sparse nodes while excluding replacement candidates and executing nodes. `core/parallel/expert_dispatcher.cpp:341-358` additionally rejects experts with `pending_dispatches != 0` or non-`IDLE` execution state.
- `core/parallel/expert_dispatcher.cpp:91-142,180-253,361-429,590-651` owns independent per-device input/fetch queues, execute queues, fetch streams, and execution streams. Scheduler step boundaries alone do not quiesce this native concurrency.
- `core/model/model_topology.cpp:849-867` derives a static sparse cache limit; `core/parallel/expert_dispatcher.cpp:124-126` snapshots it into per-device free-byte counters.
- `moe_infinity/memory/expert_prefetcher.py:94-142` exposes hit rate/occupancy and lines 180-186 mark protected cache candidates.
- `moe_infinity/runtime/model_offload.py:473-501` exposes expert hit rate, occupancy, and synchronous miss-stall time.
- `moe_infinity/utils/config.py`, `/v1/config`, `/admin/stats`, and `/metrics` are the configuration, reload, JSON telemetry, and Prometheus paths.
- Root `CMakeLists.txt:41-42` only adds `core` and `extensions`; `core/CMakeLists.txt` builds `archer_core` but does not add `tests/cpp/unit/prefetch`. The focused test target must be linked and registered before its first RED build.
- `benchmarks/serving/memory.py:119-144` hard-codes `device_memory_ratio=0.75` while lines 495-527 report CLI ratios. The benchmark must pass requested ratios/adaptive settings into model loading and report model/runtime effective values, not merely echo CLI input.

## Hard invariants

For each GPU `d` that owns both pools, every proposed and committed state must satisfy:

```text
allocatable[d] = max(0, total[d] - model[d] - activation_reserve[d] - free_reserve[d])
expert_target[d] >= min_expert_bytes[d]
kv_target[d] >= min_kv_blocks[d] * kv_block_bytes[d]
expert_target[d] + kv_target[d] <= allocatable[d]
abs(new_expert_target[d] - old_expert_target[d]) <= max_resize_step_bytes[d]
abs(new_kv_target[d] - old_kv_target[d]) <= max_resize_step_bytes[d]
```

A device without a KV backend is still represented in controller/report state with `kv_supported=False`, `kv_blocks=0`, and `direction=HOLD`; it remains on its static expert limit and is excluded only from KV-minimum arithmetic, never substituted with another device's KV capacity.

The receiver never grows before the donor reports a committed shrink. A mutation may begin only after admissions are closed, serving/native request queues and dispatcher input/execute queues are drained, all scheduled transfers finish, and a completion event recorded on every affected PyTorch, KV-copy, expert-fetch, and expert-execution stream has synchronized successfully. Only then may old tensors, allocators, pools, or FlashInfer wrappers be replaced or released. Serving `_kv_cache`, `_fi_prefill`, and `_fi_decode` are one publication/rollback unit: both wrappers are independently reconstructed and freshly planned, while all old storage/wrappers remain strongly referenced through post-publication CUDA completion. Pinned/in-flight experts (`pending_dispatches > 0`, non-`IDLE`, task-pool execution membership, dispatcher queue membership, or current protected candidates) and referenced KV blocks (`ref_cnt > 0` or serving block-table membership) are never directly evicted by the resizer.

Expert shrink has two explicit phases. `reserve_victims()` marks a deterministic set of idle/unprotected victims so no new dispatch can claim them; this reservation is reversible until `commit_reserved_victims()` moves tensors to host and releases CUDA storage. If reserve verification or receiver growth fails after that commit, the result is `PARTIAL_DONOR_COMMITTED`: publish the reduced expert target and unchanged KV target, resume at that effective split, and allow normal future expert misses to fetch evicted experts. Never raise the expert limit or report the old split as restored unless every evicted tensor was actually made resident again. KV donor shrink remains reversible because the drained old storage is retained until replacement allocation and publication succeed.

## Controller API locked for all tasks

Create `moe_infinity/memory/adaptive_memory.py` with these stable interfaces:

```python
from dataclasses import dataclass
from enum import Enum

class ResizeDirection(str, Enum):
    HOLD = "hold"
    EXPERT_TO_KV = "expert_to_kv"
    KV_TO_EXPERT = "kv_to_expert"

class ResizeOutcome(str, Enum):
    COMMITTED = "committed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    PARTIAL_DONOR_COMMITTED = "partial_donor_committed"

@dataclass(frozen=True)
class AdaptiveMemoryConfig:
    enabled: bool = False
    interval_steps: int = 64
    cooldown_steps: int = 256
    ewma_alpha: float = 0.20
    hysteresis_ratio: float = 0.15
    max_resize_step_bytes: int = 256 * 1024**2
    min_expert_cache_bytes: int = 512 * 1024**2
    min_kv_cache_blocks: int = 128
    free_memory_reserve_bytes: int = 1024 * 1024**2
    failure_limit: int = 3

@dataclass(frozen=True)
class MemorySignals:
    device_id: int
    step: int
    expert_misses: int
    expert_accesses: int
    expert_fetch_stall_ms: float
    kv_used_blocks: int
    kv_total_blocks: int
    kv_swap_bytes: int
    kv_swap_stall_ms: float
    kv_preemptions: int
    free_gpu_bytes: int
    kv_supported: bool = True

@dataclass(frozen=True)
class MemoryTargets:
    device_id: int
    expert_bytes: int
    kv_blocks: int
    direction: ResizeDirection
    reason: str
    kv_supported: bool = True

@dataclass(frozen=True)
class ResizeResult:
    device_id: int
    outcome: ResizeOutcome
    expert_bytes: int
    kv_blocks: int
    reason: str
    kv_supported: bool = True

    @property
    def committed(self) -> bool:
        return self.outcome in {
            ResizeOutcome.COMMITTED,
            ResizeOutcome.PARTIAL_DONOR_COMMITTED,
        }

class AdaptiveMemoryController:
    def observe(self, signals: MemorySignals) -> None: ...
    def propose(self, *, device_id: int, step: int, total_bytes: int, model_bytes: int,
                 activation_reserve_bytes: int, kv_block_bytes: int,
                 current_expert_bytes: int, current_kv_blocks: int,
                 kv_supported: bool = True) -> MemoryTargets: ...
    def record_resize(self, result: ResizeResult, *, step: int) -> None: ...
    def disable_to_static(self, device_id: int, reason: str) -> None: ...
    def report(self) -> dict[int, dict[str, int | float | str | bool]]: ...
```

The controller stores warm-up, EWMA, cooldown, consecutive-failure, fallback, and last-committed state in a dictionary keyed by `device_id`; no scalar target or failure latch is shared across GPUs. Policy uses EWMA rates over controller intervals, not cumulative counters:

```text
expert_cost = expert_miss_rate * expert_fetch_stall_ms
kv_pressure = kv_used_blocks / max(1, kv_total_blocks)
kv_cost = kv_pressure * (kv_swap_stall_ms + kv_preemptions) + kv_swap_stall_ms
```

Move one `min(max_resize_step_bytes, donor_slack)` step only when the larger cost exceeds the smaller by `hysteresis_ratio * max(expert_cost, kv_cost, 1e-9)`. Hold during warm-up, cooldown, insufficient donor slack, low free-memory reserve, or equal/no signal. This is deliberately bounded and deterministic; there is no gradient search or unconstrained online optimizer.

## Exact file map

### New files

- `moe_infinity/memory/adaptive_memory.py` — pure policy/config/signals/targets/failure latch; no CUDA mutation.
- `moe_infinity/serving/memory_resize.py` — serving two-phase resize adapter and rollback.
- `moe_infinity/engine/memory_resize.py` — native scheduler/KV/backend/expert two-phase adapter.
- `tests/python/unit/test_adaptive_memory.py` — deterministic CPU policy/invariant/stability/fallback tests.
- `tests/python/serving/test_memory_resize.py` — serving quiescence, referenced-KV, reserve, rollback tests.
- `tests/python/unit/test_native_memory_resize.py` — native block references, swap failure, and rollback tests.
- `tests/python/benchmark/test_memory_benchmark_config.py` — benchmark arm model-load propagation and requested/effective config reporting.
- `tests/python/unit/test_docs.py` — source-level assertions for adaptive-memory configuration, rollout, and rollback documentation.
- `tests/cpp/unit/prefetch/test_sparse_cache_resize.cpp` — native pinned/in-flight/candidate eviction contract.
- `tests/cpp/unit/prefetch/sparse_cache_resize_fixture.h` — complete declarations for native fake nodes/cache/dispatcher and fixture result types.
- `tests/cpp/unit/prefetch/sparse_cache_resize_fixture.cpp` — fixture construction, state mutation, and adapters into the production resize APIs.
- `tests/cpp/unit/prefetch/CMakeLists.txt` — focused native test target.
- `benchmarks/serving/adaptive_memory_trace.py` — deterministic trace replay comparing adaptive and fixed splits without asserting gains.
- `tests/python/benchmark/test_adaptive_memory_trace.py` — trace report/schema/reproducibility tests.

### Modified files

- `moe_infinity/serving/memory_manager.py` — absolute safe budget and current committed targets while preserving static behavior when disabled.
- `moe_infinity/serving/kv_cache.py` — drain-only physical resize, allocator rebuild, and transactional independent FlashInfer prefill/decode wrapper reconstruction.
- `moe_infinity/serving/scheduler.py` — step-boundary quiesce/drain/restore plus resize telemetry.
- `moe_infinity/serving/engine.py` — controller wiring, observations, periodic transactions, stats, reload.
- `moe_infinity/memory/memory_coordinator.py` — native absolute budget and adaptive config construction.
- `moe_infinity/memory/block_pool.py` — explicit idle-block inspection needed by native shrink validation.
- `moe_infinity/memory/kv_cache_manager.py` — drain-only pool resize and referenced-block rejection.
- `moe_infinity/runtime/attention_backend.py` — drain-only recreation of built-in and FlashInfer KV stores plus fresh wrappers/plans, with unchanged dtype/layout.
- `moe_infinity/engine/scheduler.py` — native quiesce/drain/restore and pressure/swap counters.
- `moe_infinity/engine/expert_offload_coordinator.py` — attach the tensor's actual CUDA owner to expert transfer requests.
- `moe_infinity/engine/transfer_types.py` — carry the owning `device_id` on native transfers.
- `moe_infinity/engine/unified_transfer_scheduler.py` — validate device-qualified transfer endpoints and wait for all transfers touching one device.
- `moe_infinity/entrypoints/big_modeling.py` — instantiate and retain native controller/adapter.
- `moe_infinity/memory/expert_prefetcher.py` — Python `resize_cache()`/telemetry bridge.
- `moe_infinity/runtime/model_offload.py` — interval expert miss/stall snapshots and controller bridge.
- `core/model/model_topology.h` and `core/model/model_topology.cpp` — synchronized per-device sparse-limit override.
- `core/prefetch/task_scheduler.h` and `core/prefetch/task_scheduler.cpp` — reversible victim reservation and committed trim that respect protected/executing nodes.
- `core/prefetch/archer_prefetch_handle.h` and `core/prefetch/archer_prefetch_handle.cpp` — device-bound quiescence/reservation/commit expert resize API.
- `core/parallel/expert_dispatcher.h` and `core/parallel/expert_dispatcher.cpp` — per-device admission gate, queue/active-worker drain, CUDA-stream completion barrier, victim reservation, live limit/free-byte updates, and pinned-state reporting.
- `core/python/py_archer_prefetch.cpp` — pybind resize/result/telemetry methods.
- Root `CMakeLists.txt` and `core/CMakeLists.txt` — define the test option, enable CTest/GTest, and add the focused C++ test subdirectory before the RED build.
- `moe_infinity/utils/config.py` — feature flag and bounded policy knobs.
- `moe_infinity/entrypoints/openai/api_server_v2.py` — Prometheus fields; existing `/v1/config` remains reload entry.
- `tests/python/serving/test_memory_manager.py`, `tests/python/unit/test_memory_coordinator.py`, `tests/python/unit/test_attention_backend.py`, `tests/python/unit/test_flashinfer_attention_backend.py`, `tests/python/unit/test_scheduler.py`, `tests/python/unit/test_unified_scheduler.py`, `tests/python/unit/test_transfer_scheduler_interface.py`, `tests/python/unit/test_engine_types.py`, `tests/python/unit/test_kv_handler_registration.py`, `tests/python/integration/test_expert_kv_integration.py`, `tests/python/unit/test_utils_config.py`, `tests/python/serving/test_engine.py`, `tests/python/serving/test_hot_reload.py`, and `tests/python/serving/test_api_routes.py` — compatibility, storage/wrapper recreation, explicit transfer ownership, unequal-device routing, wiring, API stats/metrics, and config tests.
- `tests/python/e2e/test_kv_memory.py` — CUDA pressure/peak-reserve/fallback test.
- `benchmarks/serving/memory.py` — feature flag and controller telemetry in live benchmark JSON.
- `docs/configuration.md`, `docs/serving.md`, and `docs/benchmarking.md` — rollout, metrics, rollback, and evidence rules.

## Task 1: Build the deterministic bounded policy

**Files:**
- Create: `moe_infinity/memory/adaptive_memory.py`
- Create: `tests/python/unit/test_adaptive_memory.py`

- [ ] **Step 1: Write failing invariant and direction tests**

```python
from moe_infinity.memory.adaptive_memory import (
    AdaptiveMemoryConfig, AdaptiveMemoryController, MemorySignals,
    MemoryTargets, ResizeDirection, ResizeOutcome, ResizeResult,
)

MIB = 1024**2

def signal(step: int, *, misses: int, fetch_ms: float, used: int,
           swaps: int = 0, swap_ms: float = 0.0, free: int = 4096 * MIB,
           device_id: int = 0):
    return MemorySignals(device_id, step, misses, 100, fetch_ms, used, 100,
                          swaps * MIB, swap_ms, swaps, free)

def test_cost_moves_one_bounded_step_toward_kv() -> None:
    ctl = AdaptiveMemoryController(AdaptiveMemoryConfig(
        enabled=True, interval_steps=1, cooldown_steps=0,
        max_resize_step_bytes=64 * MIB, min_expert_cache_bytes=128 * MIB,
        min_kv_cache_blocks=4, free_memory_reserve_bytes=128 * MIB))
    ctl.observe(signal(1, misses=0, fetch_ms=0.0, used=99,
                       swaps=8, swap_ms=40.0))
    target = ctl.propose(device_id=0, step=1, total_bytes=2048*MIB, model_bytes=512*MIB,
        activation_reserve_bytes=128*MIB, kv_block_bytes=16*MIB,
        current_expert_bytes=704*MIB, current_kv_blocks=36)
    assert target.direction is ResizeDirection.EXPERT_TO_KV
     assert target.expert_bytes == 640*MIB
     assert target.kv_blocks == 40
     # One 64 MiB bounded step: 640 MiB + 40 * 16 MiB = 1280 MiB.
     assert target.expert_bytes + target.kv_blocks*16*MIB == 1280*MIB

def test_hysteresis_and_cooldown_prevent_oscillation() -> None:
    ctl = AdaptiveMemoryController(AdaptiveMemoryConfig(
        enabled=True, interval_steps=1, cooldown_steps=8,
        hysteresis_ratio=0.25, max_resize_step_bytes=64*MIB,
        min_expert_cache_bytes=128*MIB, min_kv_cache_blocks=4,
        free_memory_reserve_bytes=128*MIB))
    ctl.observe(signal(8, misses=80, fetch_ms=50.0, used=40))
    first = ctl.propose(device_id=0, step=8, total_bytes=2048*MIB, model_bytes=512*MIB,
        activation_reserve_bytes=128*MIB, kv_block_bytes=16*MIB,
        current_expert_bytes=640*MIB, current_kv_blocks=40)
    ctl.record_resize(ResizeResult(0, ResizeOutcome.COMMITTED,
                                   first.expert_bytes, first.kv_blocks,
                                   "committed"), step=8)
    ctl.observe(signal(9, misses=0, fetch_ms=0.0, used=99,
                       swaps=8, swap_ms=50.0))
    assert ctl.propose(device_id=0, step=9, total_bytes=2048*MIB, model_bytes=512*MIB,
        activation_reserve_bytes=128*MIB, kv_block_bytes=16*MIB,
        current_expert_bytes=first.expert_bytes,
        current_kv_blocks=first.kv_blocks).direction is ResizeDirection.HOLD

def test_three_failures_latch_static_fallback() -> None:
    ctl = AdaptiveMemoryController(AdaptiveMemoryConfig(enabled=True,
        interval_steps=1, cooldown_steps=0, failure_limit=3))
    for step in range(3):
        ctl.record_resize(ResizeResult(1, ResizeOutcome.REJECTED,
                                       10, 10, "pinned"), step=step)
    assert ctl.report()[1]["fallback_static"] is True
    assert ctl.report()[1]["fallback_reason"] == "pinned"

def test_device_state_is_independent() -> None:
    ctl = AdaptiveMemoryController(AdaptiveMemoryConfig(
        enabled=True, interval_steps=1, cooldown_steps=0, failure_limit=1))
    ctl.observe(signal(1, misses=90, fetch_ms=40.0, used=10, device_id=0))
    ctl.observe(signal(1, misses=0, fetch_ms=0.0, used=99,
                       swaps=4, swap_ms=30.0, device_id=1))
    zero = ctl.propose(device_id=0, step=1, total_bytes=2048*MIB,
        model_bytes=512*MIB, activation_reserve_bytes=128*MIB,
        kv_block_bytes=16*MIB, current_expert_bytes=640*MIB,
        current_kv_blocks=40)
    one = ctl.propose(device_id=1, step=1, total_bytes=1536*MIB,
        model_bytes=384*MIB, activation_reserve_bytes=128*MIB,
        kv_block_bytes=16*MIB, current_expert_bytes=512*MIB,
        current_kv_blocks=24)
    assert zero.device_id == 0 and zero.direction is ResizeDirection.KV_TO_EXPERT
    assert one.device_id == 1 and one.direction is ResizeDirection.EXPERT_TO_KV
    ctl.record_resize(ResizeResult(0, ResizeOutcome.REJECTED,
                                   640*MIB, 40, "pinned"), step=1)
    assert ctl.report()[0]["fallback_static"] is True
    assert ctl.report()[1]["fallback_static"] is False

def test_device_without_kv_backend_holds_static_expert_target() -> None:
    ctl = AdaptiveMemoryController(AdaptiveMemoryConfig(
        enabled=True, interval_steps=1, cooldown_steps=0))
    target = ctl.propose(device_id=1, step=1, total_bytes=2048*MIB,
        model_bytes=512*MIB, activation_reserve_bytes=128*MIB,
        kv_block_bytes=16*MIB, current_expert_bytes=640*MIB,
        current_kv_blocks=0, kv_supported=False)
    assert target == MemoryTargets(1, 640*MIB, 0, ResizeDirection.HOLD,
                                   "kv_backend_unavailable", False)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/python/unit/test_adaptive_memory.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'moe_infinity.memory.adaptive_memory'`.

- [ ] **Step 3: Implement the policy exactly as the locked API and equations above**

Implementation requirements:

```python
def _bounded_step(self, donor_slack: int) -> int:
    return max(0, min(self.config.max_resize_step_bytes, donor_slack))

def _hard_budget(self, total: int, model: int, activation: int) -> int:
    return max(0, total - model - activation
               - self.config.free_memory_reserve_bytes)
```

`observe()` validates `device_id >= 0`, non-negative counters, and `0 <= kv_used_blocks <= kv_total_blocks`, converts interval deltas to the two costs, and updates only that device's EWMA. `propose()` reads only the requested device state and returns a target carrying the same `device_id`; with `kv_supported=False` it returns `HOLD`, preserves expert bytes, and reports zero KV blocks. Otherwise it returns `HOLD` unless enabled, not latched, on an interval, outside cooldown, above reserve, and beyond hysteresis. Round KV changes down to whole blocks; if one block exceeds the byte step, hold. Clamp both minima and hard budget after rounding. `record_resize()` resets only that device's consecutive failures on `COMMITTED`, records effective targets and a failure on `PARTIAL_DONOR_COMMITTED`, and latches only that device at `failure_limit` without changing configured static targets.

- [ ] **Step 4: Run the complete CPU policy matrix**

Add parametrized cases for zero signals, malformed counters/device IDs, minimum expert bytes, minimum KV blocks, model bytes exceeding total, reserve breach, non-divisible block size, deterministic repeated traces, opposite expert-heavy pressure, two GPUs with unequal capacities, and a failure latch on one GPU that does not stop the other.

Run: `pytest -q tests/python/unit/test_adaptive_memory.py`

Expected: all policy tests pass on CPU with no CUDA import requirement.

- [ ] **Step 5: Commit the policy atomically**

```bash
git add moe_infinity/memory/adaptive_memory.py tests/python/unit/test_adaptive_memory.py
git commit -m "feat: add bounded expert kv memory policy"
```

## Task 2: Preserve static budgeting and add absolute safe targets

**Files:**
- Modify: `moe_infinity/serving/memory_manager.py:10-169`
- Modify: `moe_infinity/memory/memory_coordinator.py:10-148`
- Modify: `tests/python/serving/test_memory_manager.py`
- Modify: `tests/python/unit/test_memory_coordinator.py`

- [ ] **Step 1: Write failing compatibility and hard-budget tests**

```python
import pytest
import torch

from moe_infinity.serving.memory_manager import MemoryManager


def test_absolute_targets_cannot_consume_free_reserve() -> None:
    manager = MemoryManager(device=torch.device("cpu"),
        device_memory_ratio=1.0, kv_cache_ratio=0.5,
        activation_reserve_ratio=0.1)
    manager.total_gpu_memory_bytes = 2048 * 1024**2
    budget = manager.compute_budget(model_memory_bytes=512*1024**2,
                                    free_memory_reserve_bytes=256*1024**2)
    assert budget.expert_cache_bytes + budget.kv_cache_bytes <= 1076*1024**2

def test_static_call_keeps_existing_ratio_result() -> None:
    manager = MemoryManager(device=torch.device("cpu"),
        device_memory_ratio=0.8, kv_cache_ratio=0.25)
    manager.total_gpu_memory_bytes = 8*1024**3
    assert manager.compute_budget(0).expert_cache_ratio == pytest.approx(0.6)
```

For native coordinator, monkeypatch device 0 to 8 GiB and device 1 to 12 GiB. Assert `compute_safe_budget(device_id=0, model_bytes=2*GiB, activation_reserve_bytes=GiB, free_reserve_bytes=GiB)` returns 4 GiB, the same call for device 1 returns 8 GiB, and each device rejects targets above its own budget.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/serving/test_memory_manager.py tests/python/unit/test_memory_coordinator.py`

Expected: `compute_budget()` rejects the new keyword and `MemoryCoordinator` lacks `compute_safe_budget`.

- [ ] **Step 3: Add absolute budget fields without changing disabled defaults**

Extend `MemoryBudget` with `device_id: int = 0`, `free_memory_reserve_bytes: int = 0`, `expert_cache_target_bytes: int | None = None`, and `kv_cache_target_bytes: int | None = None`. Its byte properties use explicit targets when set, then clamp to that device's `available_bytes = total - model - activation - free reserve`; otherwise preserve current ratio behavior byte-for-byte. `MemoryManager` stores committed targets in `dict[int, tuple[int, int]]`, adds `commit_targets(device_id, expert_bytes, kv_blocks, kv_block_bytes)`, and reports configured ratios plus committed absolute targets keyed by device.

Add to `MemoryCoordinator`:

```python
def compute_safe_budget(self, *, model_bytes: int,
                        activation_reserve_bytes: int,
                        free_reserve_bytes: int,
                        device_id: int = 0) -> int:
    return max(0, self.total_gpu_memory_bytes(device_id) - model_bytes
               - activation_reserve_bytes - free_reserve_bytes)

def validate_targets(self, *, device_id: int, expert_bytes: int, kv_blocks: int,
                      kv_block_bytes: int, safe_budget_bytes: int) -> None:
    if device_id < 0:
        raise ValueError("device_id must be non-negative")
    if min(expert_bytes, kv_blocks, kv_block_bytes, safe_budget_bytes) < 0:
        raise ValueError("memory targets must be non-negative")
    if expert_bytes + kv_blocks * kv_block_bytes > safe_budget_bytes:
        raise ValueError("expert and KV targets exceed safe GPU budget")
```

- [ ] **Step 4: Run budget tests**

Run: `pytest -q tests/python/serving/test_memory_manager.py tests/python/unit/test_memory_coordinator.py tests/python/unit/test_glm_budget.py`

Expected: all pass; legacy ratio assertions remain unchanged.

- [ ] **Step 5: Commit budgeting**

```bash
git add moe_infinity/serving/memory_manager.py moe_infinity/memory/memory_coordinator.py tests/python/serving/test_memory_manager.py tests/python/unit/test_memory_coordinator.py
git commit -m "feat: enforce hard gpu memory targets"
```

## Task 3: Add safe serving KV drain and physical resize

**Files:**
- Create: `moe_infinity/serving/memory_resize.py`
- Modify: `moe_infinity/serving/kv_cache.py:48-379`
- Modify: `moe_infinity/serving/scheduler.py:158-539`
- Create: `tests/python/serving/test_memory_resize.py`

- [ ] **Step 1: Write failing drain/reference/rollback tests**

Create the support objects at the top of `tests/python/serving/test_memory_resize.py`, before every test. Do not rely on helpers from another test module:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from moe_infinity.memory.adaptive_memory import (
    MemoryTargets,
    ResizeDirection,
    ResizeOutcome,
)
from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.memory_resize import ResizeReceipt, ServingMemoryResizer
from moe_infinity.serving.scheduler import Scheduler
from moe_infinity.serving.sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
)


class FakeEvent:
    def __init__(self, *, complete: bool = True) -> None:
        self._complete = complete

    def query(self) -> bool:
        return self._complete

    def complete(self) -> None:
        self._complete = True


def completed_receipt(
    *, device_id: int, post_publish_event: FakeEvent | None = None
) -> ResizeReceipt:
    event = FakeEvent()
    return ResizeReceipt(
        device_id=device_id,
        completion_events=(event,),
        post_publish_event=post_publish_event,
        admissions_paused=True,
    )


def make_cache(*, num_blocks: int) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )


def make_running_scheduler(
    *, num_blocks: int, prompt_tokens: int
) -> tuple[PagedKVCache, Scheduler]:
    cache = make_cache(num_blocks=num_blocks)
    scheduler = Scheduler(cache, max_batch_size=8, max_tokens_per_step=128)
    sequence = SequenceData(
        seq_id=1,
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(),
    )
    scheduler.add_request(SequenceGroup(request_id="r1", sequences=[sequence]))
    scheduler.schedule()
    return cache, scheduler


@dataclass
class FakeExpertPool:
    device_id: int
    resident_bytes: int
    limit_bytes: int = field(init=False)
    evicted_experts_are_resident: bool = True

    def __post_init__(self) -> None:
        self.limit_bytes = self.resident_bytes

    def reserve_victims(self, device_id: int, target_bytes: int) -> object:
        assert device_id == self.device_id
        return SimpleNamespace(ready=True, target_bytes=target_bytes)

    def commit_reserved_victims(self, reservation: object) -> int:
        self.limit_bytes = int(reservation.target_bytes)
        self.resident_bytes = self.limit_bytes
        self.evicted_experts_are_resident = False
        return self.resident_bytes

    def cancel_reservation(self, reservation: object) -> None:
        _ = reservation


class FakeFlashinferWrapper:
    def __init__(self, workspace: torch.Tensor, layout: str) -> None:
        self.workspace = workspace
        self.layout = layout
        self.plan_calls: list[SimpleNamespace] = []
        self.released = False

    def plan(self, *args: object, **kwargs: object) -> None:
        page_indices = kwargs.get(
            "page_indices", args[2] if len(args) == 8 else args[1]
        )
        values = page_indices.tolist() if isinstance(page_indices, torch.Tensor) else page_indices
        maximum = max(values, default=-1)
        self.plan_calls.append(SimpleNamespace(max_page_index=maximum))

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        _ = kv_cache
        return query


def make_flashinfer_cache(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_blocks: int,
    next_prefill_plan_error: Exception | None = None,
) -> PagedKVCache:
    class Prefill(FakeFlashinferWrapper):
        def plan(self, *args: object, **kwargs: object) -> None:
            nonlocal next_prefill_plan_error
            if next_prefill_plan_error is not None:
                error, next_prefill_plan_error = next_prefill_plan_error, None
                raise error
            super().plan(*args, **kwargs)

    module = SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=Prefill,
        BatchDecodeWithPagedKVCacheWrapper=FakeFlashinferWrapper,
    )
    monkeypatch.setattr(
        "moe_infinity.runtime.flashinfer_utils.HAS_FLASHINFER", True
    )
    monkeypatch.setattr(
        "moe_infinity.runtime.flashinfer_utils.get_flashinfer_module",
        lambda: module,
    )
    monkeypatch.setattr(
        "moe_infinity.runtime.flashinfer_utils.get_workspace",
        lambda device: torch.empty(1, device=device),
    )
    return make_cache(num_blocks=num_blocks)


def prefill_inputs(*, num_pages: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.zeros((1, 2, 2, 8), dtype=torch.float16)
    key = torch.zeros((1, 2, num_pages * 4, 8), dtype=torch.float16)
    return query, key, key.clone()


def decode_inputs(*, num_pages: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.zeros((1, 2, 1, 8), dtype=torch.float16)
    key = torch.zeros((1, 2, num_pages * 4, 8), dtype=torch.float16)
    return query, key, key.clone()
```

`ResizeReceipt` is a production dataclass in `moe_infinity/serving/memory_resize.py` with exactly `device_id`, `completion_events`, `post_publish_event`, `admissions_paused`, `retained_objects`, and consumed/cancelled state; it provides `release_retained_objects()` and rejects reuse. The tests may duck-type CUDA events with `FakeEvent`, but no helper may be left implicit.

```python
def test_resize_rejects_live_block_tables() -> None:
    cache = make_cache(num_blocks=8)
    cache.allocate_sequence(1, num_tokens=4)
    with pytest.raises(RuntimeError, match="referenced KV blocks"):
        cache.resize_num_blocks(4, completed_receipt(device_id=0))

def test_scheduler_drain_resize_restore_preserves_tokens() -> None:
    cache, scheduler = make_running_scheduler(num_blocks=8, prompt_tokens=4)
    receipt = scheduler.quiesce_for_kv_resize()
    assert scheduler.admissions_paused is True
    assert cache.block_allocator.num_free_blocks == 8
    assert all(event.query() for event in receipt.completion_events)
    cache.resize_num_blocks(6, receipt)
    scheduler.restore_after_kv_resize(receipt)
    assert scheduler.admissions_paused is False
    assert scheduler.get_running_seq_ids() == [1]
    assert cache._require_sequence(1).num_computed_tokens() == 4

def test_receiver_growth_is_not_called_when_donor_shrink_fails() -> None:
    expert = Mock(shrink_to=Mock(return_value=False))
    kv = Mock()
    result = ServingMemoryResizer(expert, kv, reserve_probe=lambda _: 2**40).apply(
        0, MemoryTargets(0, 512, 8, ResizeDirection.EXPERT_TO_KV, "kv_pressure"),
        current_expert_bytes=1024, current_kv_blocks=4, kv_block_bytes=64)
    assert result.device_id == 0
    assert result.outcome is ResizeOutcome.REJECTED
    kv.resize_num_blocks.assert_not_called()

def test_old_storage_is_retained_until_cuda_completion_event(monkeypatch) -> None:
    cache, scheduler = make_running_scheduler(num_blocks=8, prompt_tokens=4)
    event = FakeEvent(complete=False)
    monkeypatch.setattr(scheduler, "_record_resize_completion_event",
                        lambda: event)
    with pytest.raises(TimeoutError, match="CUDA completion"):
        scheduler.quiesce_for_kv_resize(timeout_s=0.01)
    assert cache.num_blocks == 8
    assert scheduler.admissions_paused is False

def test_quiesce_failure_reopens_admissions_and_restores_queues() -> None:
    cache, scheduler = make_running_scheduler(num_blocks=8, prompt_tokens=4)
    scheduler.inject_swap_failure_after(1)
    before = scheduler.snapshot_queue_ids()
    with pytest.raises(RuntimeError, match="swap drain failed"):
        scheduler.quiesce_for_kv_resize()
    assert scheduler.snapshot_queue_ids() == before
    assert scheduler.admissions_paused is False

def test_expert_eviction_then_kv_growth_failure_reports_partial_commit() -> None:
    expert = FakeExpertPool(device_id=0, resident_bytes=1024)
    kv = Mock(resize_num_blocks=Mock(side_effect=torch.OutOfMemoryError()))
    resizer = ServingMemoryResizer(expert, kv, reserve_probe=lambda _: 2**40)
    result = resizer.apply(
        0, MemoryTargets(0, 512, 8, ResizeDirection.EXPERT_TO_KV,
                         "kv_pressure"),
        current_expert_bytes=1024, current_kv_blocks=4,
        kv_block_bytes=64)
    assert result.outcome is ResizeOutcome.PARTIAL_DONOR_COMMITTED
    assert (result.expert_bytes, result.kv_blocks) == (512, 4)
    assert expert.limit_bytes == 512
    assert expert.evicted_experts_are_resident is False

def test_serving_flashinfer_wrappers_rebuild_independently_and_old_bundle_lives(
        monkeypatch) -> None:
    cache = make_flashinfer_cache(monkeypatch, num_blocks=8)
    old_store = cache._kv_cache
    old_prefill = cache._fi_prefill
    old_decode = cache._fi_decode
    assert old_prefill is not old_decode
    post_publish = FakeEvent(complete=False)
    receipt = completed_receipt(device_id=0,
                                post_publish_event=post_publish)
    cache.resize_num_blocks(4, receipt)
    assert cache._kv_cache.shape[1] == 4
    assert cache._fi_prefill is not old_prefill
    assert cache._fi_decode is not old_decode
    assert cache._fi_prefill is not cache._fi_decode
    assert receipt.retained_objects[0] is old_store
    assert receipt.retained_objects[1] is old_prefill
    assert receipt.retained_objects[2] is old_decode
    assert old_prefill.released is False and old_decode.released is False
    post_publish.complete()
    receipt.release_retained_objects()
    cache._compute_attention(*prefill_inputs(num_pages=4))
    cache._compute_attention(*decode_inputs(num_pages=4))
    assert cache._fi_prefill.plan_calls[-1].max_page_index < 4
    assert cache._fi_decode.plan_calls[-1].max_page_index < 4

def test_serving_first_replan_failure_restores_complete_old_bundle(
        monkeypatch) -> None:
    cache = make_flashinfer_cache(monkeypatch, num_blocks=8,
        next_prefill_plan_error=RuntimeError("stale page plan"))
    old = (cache._kv_cache, cache.block_allocator,
           cache._fi_prefill, cache._fi_decode, cache.num_blocks)
    receipt = completed_receipt(device_id=0)
    with pytest.raises(RuntimeError, match="stale page plan"):
        cache.resize_num_blocks(4, receipt)
        cache._compute_attention(*prefill_inputs(num_pages=4))
    assert cache._kv_cache is old[0]
    assert cache.block_allocator is old[1]
    assert cache._fi_prefill is old[2]
    assert cache._fi_decode is old[3]
    assert cache.num_blocks == old[4]
    assert cache._fi_prefill.plan_calls[-1].max_page_index < 8
    assert cache._fi_decode.plan_calls[-1].max_page_index < 8
    assert receipt.admissions_paused is False
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/serving/test_memory_resize.py`

Expected: missing `resize_num_blocks`, `quiesce_for_kv_resize`, and `ServingMemoryResizer` failures; the FlashInfer contract cannot yet reconstruct or retain the serving wrappers.

- [ ] **Step 3: Implement drain-only resize**

`PagedKVCache.resize_num_blocks(new_num_blocks, receipt)` must require an unconsumed same-device quiescence receipt whose completion events have synchronized, require every sequence table to have no GPU block IDs, and preserve `_swapped_cpu_buffers` plus token counts. Allocate a new tensor with exactly the existing dtype/layout/device and a new `BlockAllocator`. When FlashInfer is enabled, independently construct a fresh `BatchPrefillWithPagedKVCacheWrapper` and a fresh `BatchDecodeWithPagedKVCacheWrapper` on the existing workspace; neither object may alias or inherit a plan from its predecessor or from the other wrapper. Build the complete candidate bundle before taking the publication lock, then atomically publish `_kv_cache`, `block_allocator`, `num_blocks`, `_fi_prefill`, and `_fi_decode`. The first prefill and first decode after publication each rebuild metadata and call their own fresh wrapper's `plan()` with page indices below `new_num_blocks` before `run()`.

Retain strong references to the old tensor, allocator, `_fi_prefill`, and `_fi_decode` in the transaction receipt until a post-publication event recorded on the current serving-attention stream completes. Constructor, publication, planning, or event failure before that completion restores the complete old bundle, including both old wrappers and their old storage; it never leaves one new wrapper paired with one old wrapper/storage. If a fresh wrapper's first plan fails, close admissions again through the same receipt/replan path, restore the retained old bundle, safely replan the old wrappers against the old page count, and resume. Release the old bundle only after CUDA completion proves no kernel references it. It must not cast, quantize, disable FlashInfer, or silently fall back because of resize.

`Scheduler.quiesce_for_kv_resize()` runs only between `schedule()`/`update_after_step()` calls. Under the scheduler condition lock it first sets `admissions_paused=True`, so `add_request()` queues arrivals in a maintenance backlog and `schedule()` cannot allocate or swap in. It then drains every PREFILL/DECODE group to CPU, waits for all swap futures, frees GPU blocks, moves groups to `_swapped`, and records a `torch.cuda.Event` on each serving KV-copy stream and the current model stream. Poll `event.query()` until all complete or the monotonic deadline expires (PyTorch `Event.synchronize()` has no timeout), then return an immutable receipt of queue order, statuses, token counts, stream IDs, and completion events. If any drain/event fails, restore all queues/statuses, merge the maintenance backlog in arrival order, clear the gate, and raise. `restore_after_kv_resize()` uses existing `swap_in`; groups that no longer fit remain `SWAPPED`, then it atomically merges the backlog and clears the admission gate. The gate remains closed across shrink, reserve check, receiver growth, target publication, and rollback/partial-commit publication.

`ServingMemoryResizer.apply(device_id, ...)` rejects a receipt for another device, enforces donor-first ordering, calls `torch.cuda.empty_cache()` only after the receipt proves all users complete and donor objects are released, probes `torch.cuda.mem_get_info(device_id)`, and requires `free >= configured_reserve + receiver_growth`. KV-donor failure restores the retained old KV storage before reopening admissions. Expert-donor failure after committed evictions returns `ResizeOutcome.PARTIAL_DONOR_COMMITTED` with the measured smaller expert target and unchanged KV blocks; it publishes those effective targets before reopening admissions and does not claim logical restoration. Every exit path consumes or cancels the receipt exactly once and reopens admissions.

- [ ] **Step 4: Run serving resize and existing swap suites**

Run: `pytest -q tests/python/serving/test_memory_resize.py tests/python/unit/test_kv_cache_free.py tests/python/unit/test_kv_swap_recovery.py tests/python/unit/test_kv_edge_cases.py tests/python/serving/test_scheduler.py`

Expected: all pass; swap recovery and dtype-preservation behavior is unchanged, independent serving prefill/decode wrappers are reconstructed, and the old bundle survives through CUDA completion/replan safety.

- [ ] **Step 5: Commit serving KV transaction**

```bash
git add moe_infinity/serving/memory_resize.py moe_infinity/serving/kv_cache.py moe_infinity/serving/scheduler.py tests/python/serving/test_memory_resize.py
git commit -m "feat: add safe serving kv resize transaction"
```

## Task 4: Add native expert-cache live limits and safe trimming

**Files:**
- Modify: `core/model/model_topology.h`, `core/model/model_topology.cpp`
- Modify: `core/prefetch/task_scheduler.h`, `core/prefetch/task_scheduler.cpp`
- Modify: `core/prefetch/archer_prefetch_handle.h`, `core/prefetch/archer_prefetch_handle.cpp`
- Modify: `core/parallel/expert_dispatcher.h`, `core/parallel/expert_dispatcher.cpp`
- Modify: `core/python/py_archer_prefetch.cpp`
- Modify: `moe_infinity/memory/expert_prefetcher.py`
- Create: `tests/cpp/unit/prefetch/test_sparse_cache_resize.cpp`
- Create: `tests/cpp/unit/prefetch/sparse_cache_resize_fixture.h`
- Create: `tests/cpp/unit/prefetch/sparse_cache_resize_fixture.cpp`
- Create: `tests/cpp/unit/prefetch/CMakeLists.txt`
- Modify: `CMakeLists.txt`
- Modify: `core/CMakeLists.txt`

- [ ] **Step 1: Register/link the focused target, then write failing C++ contracts**

Add to root `CMakeLists.txt` immediately before `add_subdirectory(core)` so `MOE_BUILD_TESTS`, CTest, and `GTest::gtest_main` exist when `core/CMakeLists.txt` conditionally adds the target:

```cmake
option(MOE_BUILD_TESTS "Build MoE-Infinity native unit tests" OFF)
if(MOE_BUILD_TESTS)
  include(CTest)
  enable_testing()
  find_package(GTest REQUIRED)
endif()
```

Add to `core/CMakeLists.txt` after `archer_core` is fully defined:

```cmake
if(MOE_BUILD_TESTS)
  add_subdirectory(
    ${CMAKE_SOURCE_DIR}/tests/cpp/unit/prefetch
    ${CMAKE_BINARY_DIR}/tests/cpp/unit/prefetch
  )
endif()
```

Create `tests/cpp/unit/prefetch/CMakeLists.txt` and compile the fixture source into the RED target:

```cmake
add_executable(test_sparse_cache_resize
  test_sparse_cache_resize.cpp
  sparse_cache_resize_fixture.cpp
)
target_link_libraries(test_sparse_cache_resize PRIVATE archer_core GTest::gtest_main)
target_include_directories(test_sparse_cache_resize PRIVATE ${CMAKE_SOURCE_DIR}/core)
add_test(NAME sparse_cache_resize COMMAND test_sparse_cache_resize)
```

Create `tests/cpp/unit/prefetch/sparse_cache_resize_fixture.h` before the test source. It includes the production header that fully defines `Node` and `NodeExecState`; the fixture does not invent incompatible shadows:

```cpp
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "model/model_topology.h"

struct FakeResizeResult {
  std::string outcome;
  std::int64_t resident_bytes;
  std::string reason;
};

struct FakeReservation {
  std::uint64_t id;
  bool ready;
};

struct FakeResizeToken {
  std::uint64_t id;
  int device_id;
  bool ready;
  std::string reason;
};

class FakeSparseCache {
 public:
  explicit FakeSparseCache(
      std::initializer_list<std::pair<std::size_t, std::int64_t>> specs,
      int device_id = 0);
  Node& node(std::size_t index);
  NodePtr node_ptr(std::size_t index);
  void ReplaceCacheCandidates(std::initializer_list<NodePtr> candidates);
  FakeResizeResult TrimSparseCache(int device_id, std::int64_t target_bytes);
  FakeReservation ReserveSparseCacheVictims(int device_id,
                                             std::int64_t target_bytes);
  void CancelSparseCacheReservation(std::uint64_t id);
  std::int64_t ResidentBytes(int device_id) const;
  bool AllNodesResident(int device_id) const;
  void PauseAfterExecSnapshot();
  bool WaitUntilExecSnapshotReleased(int timeout_ms);
  void ResumeAfterExecSnapshot();

 private:
  int device_id_;
  std::uint64_t next_reservation_id_{1};
  NodePtrList nodes_;
  std::unordered_set<Node*> protected_;
  std::unordered_set<Node*> reserved_;
  std::mutex exec_mutex_;
  std::mutex candidates_mutex_;
  std::mutex pause_mutex_;
  std::condition_variable pause_cv_;
  bool pause_after_exec_snapshot_{false};
  bool exec_snapshot_released_{false};
  bool resume_after_exec_snapshot_{false};
};

class FakeDispatcher {
 public:
  explicit FakeDispatcher(int devices);
  void EnqueueFetch(int device_id);
  void EnqueueExec(int device_id);
  void SetFetchEventComplete(int device_id, bool complete);
  void CompleteQueuesWorkersAndEvents(int device_id);
  FakeResizeToken BeginMemoryResize(int device_id, int timeout_ms);
  void EndMemoryResize(const FakeResizeToken& token);
  bool AdmissionsPaused(int device_id) const;

 public:
  struct DeviceState {
    int fetch_queue = 0;
    int exec_queue = 0;
    int active_fetch = 0;
    int active_exec = 0;
    bool fetch_event_complete = true;
    bool admissions_paused = false;
  };

 private:
  std::uint64_t next_token_id_{1};
  std::vector<DeviceState> devices_;
};
```

Create `tests/cpp/unit/prefetch/sparse_cache_resize_fixture.cpp` with every fixture method defined before RED:

```cpp
#include "sparse_cache_resize_fixture.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace {
FakeDispatcher::DeviceState& Checked(
    std::vector<FakeDispatcher::DeviceState>& devices, int device_id) {
  if (device_id < 0 || device_id >= static_cast<int>(devices.size()))
    throw std::out_of_range("device_id");
  return devices.at(device_id);
}
}  // namespace

FakeSparseCache::FakeSparseCache(
    std::initializer_list<std::pair<std::size_t, std::int64_t>> specs,
    int device_id)
    : device_id_(device_id) {
  for (const auto& [id, bytes] : specs) {
    auto node = std::make_shared<Node>();
    node->id = id;
    node->byte_size = bytes;
    node->device = torch::Device(torch::kCUDA, device_id);
    node->default_device = node->device;
    node->default_host = torch::Device(torch::kCPU);
    node->exec_state.store(NodeExecState::IDLE);
    node->pending_dispatches.store(0);
    nodes_.push_back(std::move(node));
  }
}

Node& FakeSparseCache::node(std::size_t index) { return *nodes_.at(index); }
NodePtr FakeSparseCache::node_ptr(std::size_t index) { return nodes_.at(index); }

void FakeSparseCache::ReplaceCacheCandidates(
    std::initializer_list<NodePtr> candidates) {
  std::lock_guard<std::mutex> lock(candidates_mutex_);
  protected_.clear();
  for (const auto& node : candidates) protected_.insert(node.get());
}

void FakeSparseCache::PauseAfterExecSnapshot() {
  std::lock_guard<std::mutex> lock(pause_mutex_);
  pause_after_exec_snapshot_ = true;
  exec_snapshot_released_ = false;
  resume_after_exec_snapshot_ = false;
}

bool FakeSparseCache::WaitUntilExecSnapshotReleased(int timeout_ms) {
  std::unique_lock<std::mutex> lock(pause_mutex_);
  return pause_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
    return exec_snapshot_released_;
  });
}

void FakeSparseCache::ResumeAfterExecSnapshot() {
  {
    std::lock_guard<std::mutex> lock(pause_mutex_);
    resume_after_exec_snapshot_ = true;
  }
  pause_cv_.notify_all();
}

std::int64_t FakeSparseCache::ResidentBytes(int device_id) const {
  std::int64_t bytes = 0;
  for (const auto& node : nodes_)
    if (node->device.is_cuda() && node->device.index() == device_id)
      bytes += node->byte_size;
  return bytes;
}

bool FakeSparseCache::AllNodesResident(int device_id) const {
  return std::all_of(nodes_.begin(), nodes_.end(), [device_id](const NodePtr& n) {
    return n->device.is_cuda() && n->device.index() == device_id;
  });
}

FakeReservation FakeSparseCache::ReserveSparseCacheVictims(
    int device_id, std::int64_t target_bytes) {
  {
    std::lock_guard<std::mutex> lock(exec_mutex_);
    // The production fixture has no execute entries; taking and releasing this
    // lock models the production exec-membership snapshot boundary.
  }
  {
    std::unique_lock<std::mutex> lock(pause_mutex_);
    if (pause_after_exec_snapshot_) {
      exec_snapshot_released_ = true;
      pause_cv_.notify_all();
      pause_cv_.wait(lock, [&] { return resume_after_exec_snapshot_; });
    }
  }
  std::unordered_set<Node*> protected_snapshot;
  {
    std::lock_guard<std::mutex> lock(candidates_mutex_);
    protected_snapshot = protected_;
  }
  reserved_.clear();
  auto remaining = ResidentBytes(device_id);
  for (const auto& node : nodes_) {
    if (remaining <= target_bytes) break;
    if (!node->device.is_cuda() || node->device.index() != device_id ||
        protected_snapshot.count(node.get()) != 0 ||
        node->pending_dispatches.load() != 0 ||
        node->exec_state.load() != NodeExecState::IDLE)
      continue;
    reserved_.insert(node.get());
    remaining -= node->byte_size;
  }
  if (remaining > target_bytes) {
    reserved_.clear();
    return {0, false};
  }
  return {next_reservation_id_++, true};
}

void FakeSparseCache::CancelSparseCacheReservation(std::uint64_t) {
  reserved_.clear();
}

FakeResizeResult FakeSparseCache::TrimSparseCache(
    int device_id, std::int64_t target_bytes) {
  const auto reservation = ReserveSparseCacheVictims(device_id, target_bytes);
  if (!reservation.ready)
    return {"rejected", ResidentBytes(device_id), "pinned_or_in_flight"};
  for (const auto& node : nodes_)
    if (reserved_.count(node.get()) != 0)
      node->device = node->default_host;
  reserved_.clear();
  return {"committed", ResidentBytes(device_id), "committed"};
}

FakeDispatcher::FakeDispatcher(int devices) : devices_(devices) {}
void FakeDispatcher::EnqueueFetch(int d) { Checked(devices_, d).fetch_queue++; }
void FakeDispatcher::EnqueueExec(int d) { Checked(devices_, d).exec_queue++; }
void FakeDispatcher::SetFetchEventComplete(int d, bool done) {
  Checked(devices_, d).fetch_event_complete = done;
}
void FakeDispatcher::CompleteQueuesWorkersAndEvents(int d) {
  auto& state = Checked(devices_, d);
  state.fetch_queue = state.exec_queue = 0;
  state.active_fetch = state.active_exec = 0;
  state.fetch_event_complete = true;
}
FakeResizeToken FakeDispatcher::BeginMemoryResize(int d, int) {
  auto& state = Checked(devices_, d);
  state.admissions_paused = true;
  if (state.fetch_queue || state.exec_queue || state.active_fetch ||
      state.active_exec || !state.fetch_event_complete) {
    state.admissions_paused = false;
    return {0, d, false, "dispatcher_drain_timeout"};
  }
  return {next_token_id_++, d, true, "ready"};
}
void FakeDispatcher::EndMemoryResize(const FakeResizeToken& token) {
  Checked(devices_, token.device_id).admissions_paused = false;
}
bool FakeDispatcher::AdmissionsPaused(int d) const {
  return devices_.at(d).admissions_paused;
}
```

Then create `test_sparse_cache_resize.cpp`. The first test is a compile contract against production so RED is caused by the missing resize APIs, not undefined fixtures:

```cpp
#include <gtest/gtest.h>

#include <chrono>
#include <future>

#include "parallel/expert_dispatcher.h"
#include "prefetch/task_scheduler.h"
#include "sparse_cache_resize_fixture.h"

TEST(SparseCacheResize, ProductionApiSignaturesExist) {
  auto reserve = &ArcherTaskPool::ReserveSparseCacheVictims;
  auto cancel = &ArcherTaskPool::CancelSparseCacheReservation;
  auto commit = &ArcherTaskPool::CommitSparseCacheReservation;
  auto begin = &ExpertDispatcher::BeginMemoryResize;
  auto end = &ExpertDispatcher::EndMemoryResize;
  EXPECT_NE(reserve, nullptr);
  EXPECT_NE(cancel, nullptr);
  EXPECT_NE(commit, nullptr);
  EXPECT_NE(begin, nullptr);
  EXPECT_NE(end, nullptr);
}

TEST(SparseCacheResize, RejectsPinnedAndExecutingNodes) {
  FakeSparseCache cache({{0, 64}, {1, 64}, {2, 64}});
  cache.node(0).pending_dispatches.store(1);
  cache.node(1).exec_state.store(NodeExecState::FETCHING);
  cache.ReplaceCacheCandidates({cache.node_ptr(2)});
  auto result = cache.TrimSparseCache(/*device=*/0, /*target_bytes=*/64);
  EXPECT_EQ(result.outcome, "rejected");
  EXPECT_EQ(result.resident_bytes, 192);
  EXPECT_EQ(result.reason, "pinned_or_in_flight");
}

TEST(SparseCacheResize, EvictsOnlyIdleUnprotectedNodes) {
  FakeSparseCache cache({{0, 64}, {1, 64}, {2, 64}});
  cache.ReplaceCacheCandidates({cache.node_ptr(2)});
  auto result = cache.TrimSparseCache(0, 128);
  EXPECT_EQ(result.outcome, "committed");
  EXPECT_LE(result.resident_bytes, 128);
  EXPECT_TRUE(cache.node(2).device.is_cuda());
}

TEST(SparseCacheResize, ReservationRollbackDoesNotEvictVictims) {
  FakeSparseCache cache({{0, 64}, {1, 64}, {2, 64}});
  auto reservation = cache.ReserveSparseCacheVictims(0, 128);
  ASSERT_TRUE(reservation.ready);
  EXPECT_EQ(cache.ResidentBytes(0), 192);
  cache.CancelSparseCacheReservation(reservation.id);
  EXPECT_EQ(cache.ResidentBytes(0), 192);
  EXPECT_TRUE(cache.AllNodesResident(0));
}

TEST(SparseCacheResize, DispatcherDrainWaitsForQueuesWorkersAndStreams) {
  FakeDispatcher dispatcher(/*devices=*/2);
  dispatcher.EnqueueFetch(/*device=*/1);
  dispatcher.EnqueueExec(/*device=*/1);
  dispatcher.SetFetchEventComplete(/*device=*/1, false);
  auto blocked = dispatcher.BeginMemoryResize(/*device=*/1, /*timeout_ms=*/1);
  EXPECT_FALSE(blocked.ready);
  EXPECT_EQ(blocked.reason, "dispatcher_drain_timeout");
  dispatcher.CompleteQueuesWorkersAndEvents(/*device=*/1);
  auto ready = dispatcher.BeginMemoryResize(/*device=*/1, /*timeout_ms=*/1000);
  EXPECT_TRUE(ready.ready);
  EXPECT_TRUE(dispatcher.AdmissionsPaused(1));
  dispatcher.EndMemoryResize(ready);
  EXPECT_FALSE(dispatcher.AdmissionsPaused(1));
}
```

- [ ] **Step 2: Verify RED through the native test target**

Run: `cmake -S . -B build -GNinja -DMOE_BUILD_TESTS=ON && cmake --build build --target test_sparse_cache_resize -j2`

Expected: CMake successfully finds the registered `test_sparse_cache_resize` target and compiles `sparse_cache_resize_fixture.cpp`; compilation of `ProductionApiSignaturesExist` then fails specifically because `ArcherTaskPool::{ReserveSparseCacheVictims,CancelSparseCacheReservation,CommitSparseCacheReservation}` and `ExpertDispatcher::{BeginMemoryResize,EndMemoryResize}` do not exist. Undefined `FakeSparseCache`, `Node`, `NodeExecState`, `FakeDispatcher`, or fixture linker symbols means Step 1 is incomplete and must be fixed before implementation. A `ninja: unknown target` failure likewise means registration is wrong.

- [ ] **Step 3: Implement synchronized native resize**

Add native `enum class ResizeOutcome { COMMITTED, REJECTED, ROLLED_BACK, PARTIAL_DONOR_COMMITTED };`, `SparseVictimReservation { uint64_t id; int device_id; int64_t target_bytes; int64_t resident_bytes; bool ready; std::string reason; std::vector<NodePtr> victims; }`, and `SparseCacheResizeResult { ResizeOutcome outcome; int device_id; int64_t target_bytes; int64_t resident_bytes; std::string reason; }`. Map the native enum to the locked Python string values in pybind. `ArcherTopologyHandle` stores per-device sparse overrides behind its existing topology mutex.

`ArcherTaskPool::ReserveSparseCacheVictims(device_id, target_bytes)` must never hold `exec_mutex_` and `candidates_mutex_` simultaneously. Match the existing `RemoveCachedSparseNode()` source order: first lock `exec_mutex_`, copy the device's `exec_queue_` membership into a local `unordered_set<NodePtr>`, and release `exec_mutex_`; then lock `candidates_mutex_`, copy `candidates_` into a local protected set, and release `candidates_mutex_`. Deterministically select only resident nodes absent from both snapshots with `pending_dispatches == 0` and `exec_state == IDLE`, CAS selected nodes from `IDLE` to new state `RESIZE_RESERVED`, and do not move or release tensors. The CAS is the authority if either snapshot becomes stale. If eligible bytes are insufficient, CAS every reserved node back to `IDLE` and reject. `CancelSparseCacheReservation(id)` reverses those CASes. `CommitSparseCacheReservation(id)` may run only with a dispatcher quiescence token for the same device; it moves reserved nodes to `default_host`, waits for each move's CUDA completion event, releases CUDA tensors, returns nodes to `IDLE`, measures resident bytes, and publishes the lower limit. Once this commit succeeds, cancellation cannot claim the experts are resident.

Document the source lock rule beside the mutex declarations in `task_scheduler.h`: `unified_mutex_` may precede `candidates_mutex_` (as in `ReplaceCacheCandidates`); `exec_mutex_` and `candidates_mutex_` are snapshot locks and are never nested in either order. Include `<functional>`, then factor the two snapshots into the private production helper used by `ReserveSparseCacheVictims()`. Under `MOE_BUILD_TESTS`, expose `SetAfterExecSnapshotHookForTest(std::function<void()>)` and `SnapshotResizeExclusionsForTest(device_id)`; the latter must call that same private helper, not duplicate its locking. Add this contract test to `test_sparse_cache_resize.cpp` so the actual source mutexes are exercised:

```cpp
TEST(SparseCacheResize, ReservationNeverNestsExecAndCandidateLocks) {
  auto* pool = ArcherTaskPool::GetInstance();
  std::promise<void> exec_snapshot_released;
  std::promise<void> resume_snapshot;
  auto resume = resume_snapshot.get_future().share();
  pool->SetAfterExecSnapshotHookForTest([&] {
    exec_snapshot_released.set_value();
    resume.wait();
  });

  auto snapshot = std::async(std::launch::async, [&] {
    return pool->SnapshotResizeExclusionsForTest(/*device_id=*/0);
  });
  ASSERT_EQ(exec_snapshot_released.get_future().wait_for(
                std::chrono::seconds(1)),
            std::future_status::ready);
  auto replace = std::async(std::launch::async, [&] {
    pool->ReplaceCacheCandidates(NodePtrList{});
  });
  EXPECT_EQ(replace.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  resume_snapshot.set_value();
  EXPECT_EQ(snapshot.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  snapshot.get();
  pool->SetAfterExecSnapshotHookForTest({});
}
```

Add `<chrono>` and `<future>` to the test imports. The already-defined fake pause methods remain a fixture-level model, but this test is authoritative because both `SnapshotResizeExclusionsForTest()` and `ReserveSparseCacheVictims()` call the same production helper. The test fails by timeout if that helper still holds `exec_mutex_` while waiting for `candidates_mutex_`, or if it acquires candidates before releasing exec. Ensure `MOE_BUILD_TESTS` is a public compile definition on `archer_core` when the focused target is enabled so declarations and definitions agree.

Add `ExpertDispatcher::BeginMemoryResize(device_id, timeout_ms)` and `EndMemoryResize(token)`. The begin call sets a per-device admission gate checked by `Enqueue()` before either `input_queue_` or `exec_queue_` receives work; new calls block on a condition variable without being dropped. Add `pending_by_device_`, `active_fetch_workers_`, and `active_exec_workers_` counters rather than inferring device-local state from existing aggregate `pending_`. Begin waits until that device's input queue and execute queue are empty, all three device-local counters are zero, and task-pool execution membership is empty. It records CUDA events on `fetch_streams_[device_id]` and every `exec_streams_[thread_idx]` where `thread_idx % kNumDevices() == device_id`, polls `cudaEventQuery` to the deadline, destroys every temporary event, and returns a device-bound token only after success. On timeout it cancels victim reservations, clears the gate, wakes blocked enqueues, and returns a rejection. `EndMemoryResize()` publishes limits before clearing the gate and waking blocked calls.

`ExpertDispatcher::SetCacheLimit(device_id, target_bytes)` requires the same device's live quiescence token, locks `cache_mutex_[device_id]`, recomputes free bytes as `max(0, target - GetCacheOccupancyBytes(device_id))`, and notifies `cache_cv_`. Make `GetCacheOccupancyBytes(device_id)` device-specific instead of summing every GPU. Do not call `DeviceMemoryPool::SetMemoryRatio()` because it resets free-memory accounting independently of live allocations.

Expose through pybind:

```cpp
.def("resize_expert_cache", &ArcherPrefetchHandle::ResizeExpertCache,
     py::arg("device_id"), py::arg("target_bytes"))
.def("get_expert_cache_limit", &ArcherPrefetchHandle::GetExpertCacheLimit)
.def("begin_memory_resize", &ArcherPrefetchHandle::BeginMemoryResize,
     py::arg("device_id"), py::arg("timeout_ms"))
.def("end_memory_resize", &ArcherPrefetchHandle::EndMemoryResize,
     py::arg("token"))
```

Add `ExpertPrefetcher.resize_cache(device_id, target_bytes) -> dict[str, object]` that returns the native result without swallowing failure.

- [ ] **Step 4: Run native and Python bridge tests**

Run: `cmake --build build --target test_sparse_cache_resize -j2 && ctest --test-dir build -R sparse_cache_resize --output-on-failure`

Expected: pass, including target registration, protected-candidate/pending-dispatch cases, reversible reservation, blocked admission, queue/active-worker drain, per-device stream events, and gate reopening.

Run: `pytest -q tests/python/dflash/test_route_ahead_wire.py tests/python/dflash/test_route_ahead_metrics.py tests/python/integration/test_expert_kv_integration.py`

Expected: pass; existing `replace_cache_candidates()` behavior remains protection, not a forced eviction list.

- [ ] **Step 5: Commit native expert resizing**

```bash
git add CMakeLists.txt core/model core/prefetch core/parallel core/python/py_archer_prefetch.cpp core/CMakeLists.txt moe_infinity/memory/expert_prefetcher.py tests/cpp/unit/prefetch
git commit -m "feat: safely resize native expert cache"
```

## Task 5: Add native KV drain/resize and joint transaction

**Files:**
- Create: `moe_infinity/engine/memory_resize.py`
- Modify: `moe_infinity/memory/block_pool.py`
- Modify: `moe_infinity/memory/kv_cache_manager.py`
- Modify: `moe_infinity/runtime/attention_backend.py`
- Modify: `moe_infinity/engine/scheduler.py`
- Modify: `moe_infinity/engine/expert_offload_coordinator.py`
- Modify: `moe_infinity/engine/transfer_types.py`
- Modify: `moe_infinity/engine/unified_transfer_scheduler.py`
- Modify: `moe_infinity/entrypoints/big_modeling.py`
- Create: `tests/python/unit/test_native_memory_resize.py`
- Modify: `tests/python/unit/test_attention_backend.py`
- Modify: `tests/python/unit/test_flashinfer_attention_backend.py`
- Modify: `tests/python/unit/test_scheduler.py`
- Modify: `tests/python/unit/test_unified_scheduler.py`
- Modify: `tests/python/unit/test_transfer_scheduler_interface.py`
- Modify: `tests/python/unit/test_engine_types.py`
- Modify: `tests/python/unit/test_kv_handler_registration.py`
- Modify: `tests/python/integration/test_expert_kv_integration.py`

- [ ] **Step 1: Write failing native safety tests**

Put the following imports and complete helpers before the tests in `tests/python/unit/test_native_memory_resize.py`. When the two scheduler tests below are placed in `test_scheduler.py`/`test_unified_scheduler.py`, copy the relevant helper definition and imports into that file rather than importing from another test module.

```python
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import pytest
import torch

from moe_infinity.engine.memory_resize import NativeMemoryResizer, ResizeReceipt
from moe_infinity.engine.scheduler import Scheduler
from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)
from moe_infinity.engine.types import Request, SamplingParams, SequenceStatus
from moe_infinity.engine.unified_transfer_scheduler import (
    TransferScheduler,
    UnifiedTransferScheduler,
)
from moe_infinity.memory.adaptive_memory import (
    MemoryTargets,
    ResizeDirection,
    ResizeOutcome,
)
from moe_infinity.memory.block_pool import BlockPool
from moe_infinity.memory.kv_cache_manager import KVCacheManager


class FakeEvent:
    def query(self) -> bool:
        return True


def completed_receipt(*, device_id: int) -> ResizeReceipt:
    return ResizeReceipt(
        device_id=device_id,
        request_queues_drained=True,
        dispatch_queues_drained=True,
        cuda_events=(FakeEvent(),),
        admissions_paused=True,
    )


def targets(*, device_id: int, expert: int, kv: int) -> MemoryTargets:
    direction = (
        ResizeDirection.EXPERT_TO_KV if kv > 8 else ResizeDirection.KV_TO_EXPERT
    )
    return MemoryTargets(device_id, expert, kv, direction, "test")


def make_request(request_id: str, *, tokens: int = 8) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(tokens)),
        sampling_params=SamplingParams(),
        arrival_time=time.time(),
    )


def make_running_request(request_id: str) -> Request:
    request = make_request(request_id)
    request.status = SequenceStatus.RUNNING
    return request


class RecordingTransferScheduler(TransferScheduler):
    def __init__(self) -> None:
        self.requests: list[TransferRequest] = []
        self._by_id: dict[str, TransferRequest] = {}

    def enqueue(self, request: TransferRequest) -> str:
        self.requests.append(request)
        self._by_id[request.transfer_id] = request
        return request.transfer_id

    def wait(self, transfer_id: str, timeout_ms: float = 5000.0) -> bool:
        _ = timeout_ms
        return transfer_id in self._by_id

    def wait_for_device(self, device_id: int, timeout_ms: float) -> bool:
        _ = timeout_ms
        return all(
            request.device_id != device_id or request.transfer_id in self._by_id
            for request in self.requests
        )

    def cancel(self, transfer_id: str) -> bool:
        return self._by_id.pop(transfer_id, None) is not None

    def shutdown(self, wait: bool = True) -> None:
        _ = wait

    def get_pending_count(self) -> dict[TransferType, int]:
        return {}

    def set_bandwidth_budget(self, expert_ratio: float, kv_ratio: float) -> None:
        _ = (expert_ratio, kv_ratio)


class FakeDispatcher:
    def __init__(self) -> None:
        self._paused: set[int] = set()
        self._fetches: dict[int, int] = {}

    def enqueue_fetch(self, *, device_id: int) -> None:
        self._fetches[device_id] = self._fetches.get(device_id, 0) + 1

    def begin_memory_resize(self, device_id: int, timeout_ms: int) -> object:
        _ = timeout_ms
        self._paused.add(device_id)
        self._fetches[device_id] = 0
        return SimpleNamespace(device_id=device_id, ready=True)

    def end_memory_resize(self, token: object) -> None:
        self._paused.remove(int(token.device_id))

    def admissions_paused(self, device_id: int) -> bool:
        return device_id in self._paused


class FakeExpertCache:
    def __init__(self, device_id: int, resident_bytes: int) -> None:
        self.device_id = device_id
        self._limit = resident_bytes

    def reserve_victims(self, device_id: int, target_bytes: int) -> object:
        assert device_id == self.device_id
        return SimpleNamespace(device_id=device_id, target_bytes=target_bytes)

    def commit_reserved_victims(self, reservation: object) -> int:
        self._limit = int(reservation.target_bytes)
        return self._limit

    def cancel_reservation(self, reservation: object) -> None:
        _ = reservation

    def limit_bytes(self, device_id: int) -> int:
        assert device_id == self.device_id
        return self._limit


class FakeAttentionBackend:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks

    def resize_num_blocks(
        self, device_id: int, target_blocks: int, receipt: ResizeReceipt
    ) -> None:
        assert receipt.device_id == device_id
        self.num_blocks = target_blocks

    def grow(self, target_blocks: int) -> None:
        self.num_blocks = target_blocks


def make_native_scheduler(
    *,
    device_id: int,
    gpu_blocks: int,
    transfer_scheduler: TransferScheduler | None = None,
) -> Scheduler:
    manager = KVCacheManager(
        num_gpu_blocks=gpu_blocks,
        num_cpu_blocks=32,
        block_size=4,
        device_id=device_id,
    )
    return Scheduler(
        kv_cache_manager=manager,
        transfer_scheduler=transfer_scheduler,
        device_id=device_id,
    )


def make_native_bundle(
    *, device_id: int, gpu_blocks: int, expert_bytes: int = 1024
) -> SimpleNamespace:
    transfer = RecordingTransferScheduler()
    scheduler = make_native_scheduler(
        device_id=device_id,
        gpu_blocks=gpu_blocks,
        transfer_scheduler=transfer,
    )
    dispatcher = FakeDispatcher()
    expert = FakeExpertCache(device_id, expert_bytes)
    attention = FakeAttentionBackend(gpu_blocks)
    resizer = NativeMemoryResizer(
        device_id=device_id,
        scheduler=scheduler,
        dispatcher=dispatcher,
        expert_cache=expert,
        kv_manager=scheduler.kv_mgr,
        attention_backend=attention,
        reserve_probe=lambda _: 2**40,
    )
    return SimpleNamespace(
        transfer=transfer,
        scheduler=scheduler,
        dispatcher=dispatcher,
        expert=expert,
        kv_mgr=scheduler.kv_mgr,
        attention=attention,
        resizer=resizer,
    )


def kv_swap_out_request(request_id: str, *, device_id: int) -> TransferRequest:
    return TransferRequest(
        transfer_id=f"swap-out-{request_id}",
        transfer_type=TransferType.KV_SWAP_OUT,
        priority=TransferPriority.HIGH,
        device_id=device_id,
        source_device=f"cuda:{device_id}",
        target_device="cpu",
    )
```

The production constructors added in this task must accept exactly the keywords used by these helpers. `ResizeReceipt` owns the fields shown above plus the scheduler/dispatcher sub-tokens and consumed state. This makes the RED failure about missing production APIs, never an undefined fixture or import.

```python
def test_native_pool_resize_rejects_referenced_blocks() -> None:
    mgr = KVCacheManager(8, 16, block_size=4)
    assert mgr.allocate_blocks_for_sequence("r1", 8)
    with pytest.raises(RuntimeError, match="referenced KV blocks"):
        mgr.resize_gpu_blocks(0, 4, completed_receipt(device_id=0))

def test_native_transaction_rolls_back_when_attention_allocation_ooms() -> None:
    bundle = make_native_bundle(device_id=0, gpu_blocks=8)
    running = make_running_request("resident")
    assert bundle.kv_mgr.allocate_blocks_for_sequence(running.request_id, 8)
    bundle.scheduler._running.append(running)
    receipt = bundle.resizer.quiesce(device_id=0)
    bundle.attention.resize_num_blocks = Mock(side_effect=torch.OutOfMemoryError())
    result = bundle.resizer.apply(
        targets(device_id=0, expert=512, kv=4), receipt=receipt)
    assert result.outcome is ResizeOutcome.ROLLED_BACK
    assert bundle.kv_mgr.num_gpu_blocks == 8
    assert bundle.scheduler.num_swapped == 1
    assert bundle.scheduler.admissions_paused is False

def test_native_shrink_never_removes_cached_block_with_reference() -> None:
    pool = BlockPool(4)
    block = pool.allocate_block()
    assert block is not None and block.ref_cnt == 1
    assert pool.removable_tail_ids(2) == []

def test_expert_donor_failure_after_eviction_commits_reduced_state() -> None:
    bundle = make_native_bundle(device_id=1, gpu_blocks=8,
                                expert_bytes=1024)
    bundle.attention.resize_num_blocks = Mock(
        side_effect=torch.OutOfMemoryError()
    )
    result = bundle.resizer.apply(targets(device_id=1, expert=512, kv=12))
    assert result.outcome is ResizeOutcome.PARTIAL_DONOR_COMMITTED
    assert result.device_id == 1
    assert result.expert_bytes == 512
    assert result.kv_blocks == 8
    assert bundle.expert.limit_bytes(1) == 512
    assert bundle.scheduler.admissions_paused is False

def test_native_quiescence_drains_transfers_and_synchronizes_streams() -> None:
    bundle = make_native_bundle(device_id=1, gpu_blocks=8)
    bundle.scheduler.add_request(make_request("r1"))
    bundle.dispatcher.enqueue_fetch(device_id=1)
    receipt = bundle.resizer.quiesce(device_id=1)
    assert bundle.scheduler.admissions_paused is True
    assert bundle.dispatcher.admissions_paused(1) is True
    assert receipt.request_queues_drained
    assert receipt.dispatch_queues_drained
    assert all(event.query() for event in receipt.cuda_events)
    bundle.resizer.resume(receipt)
    assert bundle.scheduler.admissions_paused is False
    assert bundle.dispatcher.admissions_paused(1) is False

def test_native_swap_transfers_use_owning_unequal_device() -> None:
    transfer = RecordingTransferScheduler()
    zero = make_native_scheduler(device_id=0, gpu_blocks=8,
                                 transfer_scheduler=transfer)
    one = make_native_scheduler(device_id=1, gpu_blocks=13,
                                transfer_scheduler=transfer)
    zero_req = make_running_request("zero")
    one_req = make_running_request("one")
    assert zero.kv_mgr.allocate_blocks_for_sequence("zero", 8)
    assert one.kv_mgr.allocate_blocks_for_sequence("one", 8)
    zero._preempt_with_transfer(zero_req)
    one._preempt_with_transfer(one_req)
    assert zero._swap_in_request(zero_req)
    assert one._swap_in_request(one_req)
    resize_req = make_running_request("resize-one")
    assert one.kv_mgr.allocate_blocks_for_sequence("resize-one", 8)
    one._running.append(resize_req)
    receipt = one.begin_memory_resize(device_id=1)
    one.end_memory_resize(receipt)
    assert [(r.source_device, r.target_device) for r in transfer.requests] == [
        ("cuda:0", "cpu"), ("cuda:1", "cpu"),
        ("cpu", "cuda:0"), ("cpu", "cuda:1"),
        ("cuda:1", "cpu"),
    ]
    assert zero.kv_mgr.num_gpu_blocks == 8
    assert one.kv_mgr.num_gpu_blocks == 13

def test_quiesce_waits_only_for_transfers_touching_requested_device() -> None:
    gate0, gate1 = threading.Event(), threading.Event()
    transfer = UnifiedTransferScheduler(max_workers=2)
    transfer.register_handler(
        TransferType.KV_SWAP_OUT,
        lambda request: {0: gate0, 1: gate1}[request.device_id].wait())
    id0 = transfer.enqueue(kv_swap_out_request("d0", device_id=0))
    id1 = transfer.enqueue(kv_swap_out_request("d1", device_id=1))
    gate0.set()
    assert transfer.wait(id0, timeout_ms=1000) is True
    assert transfer.wait_for_device(device_id=0, timeout_ms=10) is True
    assert transfer.wait_for_device(device_id=1, timeout_ms=1) is False
    gate1.set()
    assert transfer.wait(id1, timeout_ms=1000) is True
    assert transfer.wait_for_device(device_id=1, timeout_ms=10) is True
    transfer.shutdown()

@pytest.mark.parametrize("bad", [-1, 2])
def test_transfer_scheduler_rejects_endpoint_for_other_device(bad: int) -> None:
    transfer = UnifiedTransferScheduler()
    with pytest.raises(ValueError, match="device_id"):
        transfer.enqueue(TransferRequest(
            transfer_id="wrong", transfer_type=TransferType.KV_SWAP_OUT,
            priority=TransferPriority.HIGH, device_id=1,
            source_device=f"cuda:{bad}", target_device="cpu"))
    transfer.shutdown()
```

Place the unequal-device transfer tests in `tests/python/unit/test_scheduler.py` and `tests/python/unit/test_unified_scheduler.py`; keep transaction/quiescence assertions in `test_native_memory_resize.py`. Put the complete `RecordingTransferScheduler`, `make_request`, `make_running_request`, `make_native_scheduler`, and `kv_swap_out_request` definitions above the first test in whichever file uses them, with the exact imports shown above; do not assume pytest shares module globals. Update every existing `TransferRequest(...)` fixture in `test_transfer_scheduler_interface.py`, `test_engine_types.py`, `test_kv_handler_registration.py`, and `test_expert_kv_integration.py` to supply the endpoint's explicit `device_id`, and assert normalization/result handling preserves it. In `tests/python/unit/test_attention_backend.py` add a local `FakeEvent` and `completed_receipt` before the test, keep references to old `k_cache`/`v_cache`, prove `resize_num_blocks()` refuses an unsynchronized receipt, then prove both tensors change shape only after synchronization and retain dtype/device/layout. In `tests/python/unit/test_flashinfer_attention_backend.py`, define the fake prefill/decode wrapper classes and their constructor/`plan()` counters before the test, install them through that file's existing FlashInfer monkeypatch seam, resize 8→4 blocks, and assert `_fi_kv_cache.shape[0] == 4`, both wrappers are newly constructed, no old plan object is reused, and the first prefill/decode after resize plans against indices below 4.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/unit/test_native_memory_resize.py tests/python/unit/test_attention_backend.py tests/python/unit/test_flashinfer_attention_backend.py tests/python/unit/test_scheduler.py tests/python/unit/test_unified_scheduler.py tests/python/unit/test_transfer_scheduler_interface.py tests/python/unit/test_engine_types.py tests/python/unit/test_kv_handler_registration.py tests/python/integration/test_expert_kv_integration.py`

Expected: the resize/quiesce/removable-tail/device-drain APIs and `ResizeOutcome` behavior are missing; unequal-device assertions expose the current hard-coded `cuda:0`; attention backend tests fail because built-in/FlashInfer storage cannot be resized.

- [ ] **Step 3: Implement native drain-only storage recreation**

Add `BlockPool.referenced_block_ids()` and `removable_tail_ids(target_blocks)`. Construct `KVCacheManager(..., device_id=device_id)` and retain that immutable owner. `KVCacheManager.resize_gpu_blocks(device_id, target_blocks, receipt)`, `prepare_swap_out(device_id, ...)`, `commit_swap_out(device_id, ...)`, `prepare_swap_in(device_id, ...)`, and `commit_swap_in(device_id, ...)` reject a device other than the manager's owner. Resize additionally requires a same-device receipt, no GPU block table, no allocated/ref-counted block, no pending swap-in allocation, and synchronized KV-copy events; it retains the old `_gpu_pool` until all attention storage has published successfully.

`PagedAttentionBackend.resize_num_blocks(device_id, target_blocks, receipt)` requires the same live receipt and explicitly covers every runtime allocation in `attention_backend.py:103-163`: allocate replacement `k_cache` and `v_cache` with identical shape convention/dtype/layout/device and, when `_use_flashinfer`, allocate replacement `_fi_kv_cache` with `[target_blocks, 2, block_size, num_kv_heads, head_dim]`. Construct fresh `BatchPrefillWithPagedKVCacheWrapper` and `BatchDecodeWithPagedKVCacheWrapper` on the existing workspace; do not reuse wrapper-internal plans that reference the old page count/storage. Publish all three stores, both wrappers, and `num_gpu_blocks` under one lock only after allocations and constructors succeed. The next prefill/decode rebuilds metadata and calls the fresh wrapper's `plan()` before `run()`. Keep strong references to all old stores/wrappers until a post-publication CUDA event on the current attention stream synchronizes; on any allocation/wrapper/event failure, retain every old field. FlashInfer remains enabled after a successful resize and adaptive resizing is not silently bypassed on that path.

Add required `device_id: int` to `TransferRequest` and preserve it when `UnifiedTransferScheduler.enqueue()` normalizes a request. `engine.Scheduler(..., device_id)` stores and validates the owning CUDA index. `_preempt_with_transfer()`, normal swap-out, resize drain preemption, `_swap_in_request()`, and restore after resize pass that exact `device_id` into every KV-manager operation and every `TransferRequest`. They create transfer endpoints with `f"cuda:{device_id}"`; no native KV path may contain a literal `"cuda:0"`. `UnifiedTransferScheduler.enqueue()` parses CUDA endpoints, rejects negative/malformed indices and any request whose `device_id` disagrees with a CUDA endpoint, indexes queued/running transfer IDs by touched CUDA device, and removes them on completion/cancel. Add `wait_for_device(device_id, timeout_ms)` so quiescence waits for all queued or running H2D/D2H transfers touching that device without blocking on unrelated GPUs. Update every existing `TransferRequest(...)` construction in the repository to pass its actual owner; expert transfers derive it from their source/target tensor device rather than defaulting to device 0.

`engine.Scheduler.begin_memory_resize(device_id)` rejects a device other than its owner, first closes request admission and swap-in/allocation while preserving new arrivals in order, then uses the same device-qualified `_preempt_with_transfer()` for every running request, calls `wait_for_device(device_id, ...)`, and rejects if CPU blocks cannot hold the drain. `NativeMemoryResizer.quiesce()` composes that scheduler token with `ExpertDispatcher::BeginMemoryResize`, records/synchronizes events on that device's KV H2D/D2H streams, attention current stream, expert fetch stream, and every execution stream assigned to that device, and returns only when request, transfer, fetch, execute, and task-pool work is empty. On failure it reverses queue/status changes and both gates. It exposes per-device interval counters for used/total blocks, swap bytes/stall, and preemptions. `big_modeling.py` passes the selected native KV device into `KVCacheManager`, `PagedAttentionBackend`, `Scheduler`, `UnifiedTransferScheduler` requests, and the resizer rather than relying on defaults.

`NativeMemoryResizer.apply()` reserves expert victims before irreversible mutation. Pre-commit rejection cancels the reservation and restores queue state. After `CommitSparseCacheReservation` releases expert tensors, reserve/receiver failure publishes `PARTIAL_DONOR_COMMITTED` with measured expert bytes and unchanged KV blocks; normal fetch can reload those experts later. KV-donor failures restore retained KV pool/backend objects. In all cases publish the effective `ResizeResult` to the same device's controller before reopening dispatcher and scheduler admissions.

`big_modeling.py` creates controller/report state for every visible device, computes each device's initial budget independently, and retains `_native_memory_coordinator`, `_native_memory_controller`, and per-device `_native_memory_resizers` alongside existing native objects. Because the current native KV manager/backend is instantiated on one selected device, only that device uses `kv_supported=True`; other devices emit explicit `HOLD` targets/results with `kv_supported=False`, `kv_blocks=0`, and their unchanged static expert limits until a backend exists for them. Never apply device 0 totals to another GPU. Disabled mode constructs no controller/resizer and follows lines 389-425 unchanged.

- [ ] **Step 4: Run native scheduler/KV suites**

Run: `pytest -q tests/python/unit/test_native_memory_resize.py tests/python/unit/test_kv_cache_manager.py tests/python/unit/test_attention_backend.py tests/python/unit/test_flashinfer_attention_backend.py tests/python/unit/test_scheduler.py tests/python/unit/test_unified_scheduler.py tests/python/unit/test_transfer_scheduler_interface.py tests/python/unit/test_engine_types.py tests/python/unit/test_kv_handler_registration.py tests/python/integration/test_expert_kv_integration.py tests/python/integration/test_kv_cache_swap.py tests/python/integration/test_swap_scheduling.py tests/python/integration/test_generation_loop.py tests/python/integration/test_output_equivalence.py`

Expected: all pass; output-equivalence remains byte-for-byte, old storage survives until event completion, built-in plus FlashInfer paths use the new page count, and unequal devices 0/1 route swap-out, drain preemption, and swap-in only through their own CUDA endpoints.

- [ ] **Step 5: Commit native joint transaction**

```bash
git add moe_infinity/engine/memory_resize.py moe_infinity/memory/block_pool.py moe_infinity/memory/kv_cache_manager.py moe_infinity/runtime/attention_backend.py moe_infinity/engine/scheduler.py moe_infinity/engine/expert_offload_coordinator.py moe_infinity/engine/transfer_types.py moe_infinity/engine/unified_transfer_scheduler.py moe_infinity/entrypoints/big_modeling.py tests/python/unit/test_native_memory_resize.py tests/python/unit/test_attention_backend.py tests/python/unit/test_flashinfer_attention_backend.py tests/python/unit/test_scheduler.py tests/python/unit/test_unified_scheduler.py tests/python/unit/test_transfer_scheduler_interface.py tests/python/unit/test_engine_types.py tests/python/unit/test_kv_handler_registration.py tests/python/integration/test_expert_kv_integration.py
git commit -m "feat: add native expert kv resize transaction"
```

## Task 6: Wire interval telemetry and controller ticks

**Files:**
- Modify: `moe_infinity/runtime/model_offload.py:460-510`
- Modify: `moe_infinity/serving/engine.py:84-168,653-690`
- Modify: `moe_infinity/serving/scheduler.py`
- Modify: `tests/python/serving/test_engine.py`

- [ ] **Step 1: Write failing wiring tests with a fake clock/counters**

Extend the existing `_make_config()` and `_make_engine()` helpers in `tests/python/serving/test_engine.py` before adding these tests; do not introduce an undefined parallel factory:

```python
from unittest.mock import Mock

from moe_infinity.memory.adaptive_memory import ResizeOutcome, ResizeResult


def _make_engine(
    tokenizer: Optional[object] = None,
    *,
    adaptive: bool = False,
    interval_steps: int = 64,
    device_count: int = 1,
) -> ContinuousBatchingEngine:
    config = _make_config()
    config.update({
        "adaptive_memory_enabled": adaptive,
        "adaptive_memory_interval_steps": interval_steps,
        "adaptive_memory_device_count_for_test": device_count,
    })
    return ContinuousBatchingEngine(
        model=MockModel(),
        engine=MockOffloadEngine(),
        config=config,
        tokenizer=tokenizer,
    )
```

The test-only device-count key is consumed only by the CPU fixture seam and is never accepted as a public runtime option.

```python
def test_engine_ticks_controller_only_at_safe_interval() -> None:
    engine = _make_engine(adaptive=True, interval_steps=4)
    engine.memory_controller = Mock()
    for _ in range(3):
        engine.step()
    engine.memory_controller.propose.assert_not_called()
    engine.step()
    engine.memory_controller.propose.assert_called_once()

def test_stats_expose_last_committed_split_and_failures() -> None:
    engine = _make_engine(adaptive=True, device_count=2)
    memory = engine.get_stats()["memory"]["adaptive"]
    assert set(memory["devices"]) == {0, 1}
    assert {"enabled", "fallback_static", "expert_target_bytes",
             "kv_target_blocks", "resize_attempts", "resize_failures",
             "last_reason"}.issubset(memory["devices"][0])

def test_failure_on_one_device_does_not_latch_other_device() -> None:
    engine = _make_engine(adaptive=True, device_count=2)
    engine.memory_controller.record_resize(
        ResizeResult(0, ResizeOutcome.REJECTED, 512, 8, "pinned"), step=64)
    stats = engine.get_stats()["memory"]["adaptive"]["devices"]
    assert stats[0]["resize_failures"] == 1
    assert stats[1]["resize_failures"] == 0
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/serving/test_engine.py -k 'controller or committed_split'`

Expected: adaptive constructor/stats fields are absent.

- [ ] **Step 3: Add interval snapshots, not hot-path synchronization**

Add monotonic interval snapshots keyed by CUDA device for expert accesses/misses, `_fetch_tensors_timed()` stall, KV used/total blocks, swap bytes/stall, preemptions, and `mem_get_info(device_id)` free bytes. Controller observation/tick occurs after a serving step has completed and before the next schedule. Propose/apply/report each device independently; a device with no KV backend reports `kv_supported=false` and holds KV rather than borrowing device 0's values. Never invoke resize from expert dispatch threads, CUDA callbacks, or transfer handlers.

On `torch.OutOfMemoryError`, reserve failure, timeout, or adapter rejection before donor release, record one failed result for that device and continue at its last committed/static targets. A post-expert-eviction failure records the effective partial donor commit and does not overwrite it with the old target. After `failure_limit`, stop proposing only for that device for process lifetime unless `/v1/config` explicitly disables then re-enables the feature.

- [ ] **Step 4: Run serving engine regression tests**

Run: `pytest -q tests/python/serving/test_engine.py tests/python/serving/test_scheduler.py tests/python/serving/test_cancellation.py tests/python/unit/test_phase_cleanup.py`

Expected: all pass and disabled mode produces no resize calls.

- [ ] **Step 5: Commit runtime wiring**

```bash
git add moe_infinity/runtime/model_offload.py moe_infinity/serving/engine.py moe_infinity/serving/scheduler.py tests/python/serving/test_engine.py
git commit -m "feat: drive adaptive memory from interval telemetry"
```

## Task 7: Add config, hot reload, metrics, and static rollback

**Files:**
- Modify: `moe_infinity/utils/config.py:17-162`
- Modify: `moe_infinity/serving/engine.py:670-690`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:716-745,1183-1200,1732-1758`
- Modify: `tests/python/unit/test_utils_config.py`
- Modify: `tests/python/serving/test_hot_reload.py`
- Modify: `tests/python/serving/test_api_routes.py`

- [ ] **Step 1: Write failing config and metric tests**

Add these imports and fixtures above the new tests in `tests/python/serving/test_hot_reload.py`; the API-route formatter test continues to use that module's existing `client` fixture.

```python
from unittest.mock import Mock

from fastapi.testclient import TestClient

import moe_infinity.entrypoints.openai.api_server_v2 as srv
from moe_infinity.entrypoints.openai.api_server_v2 import _format_prometheus_metrics
from moe_infinity.utils.config import ArcherConfig


@pytest.fixture
def adaptive_engine() -> SimpleNamespace:
    trace: list[str] = []

    def restore_static_memory_targets(*, transactional: bool = True) -> None:
        assert transactional is True
        trace.extend([
            "pause_admissions", "drain_requests", "drain_dispatch",
            "sync_cuda_events", "trim", "publish_effective",
            "resume_admissions",
        ])

    engine = SimpleNamespace(
        config=SimpleNamespace(adaptive_memory_enabled=True),
        resize_trace=trace,
        restore_static_memory_targets=Mock(
            side_effect=restore_static_memory_targets
        ),
    )
    engine.update_config = Mock(side_effect=lambda values: (
        engine.restore_static_memory_targets(transactional=True)
        if values.get("adaptive_memory_enabled") is False else None
    ))
    return engine


@pytest.fixture
def client(adaptive_engine, monkeypatch):
    monkeypatch.setattr(srv, "engine", adaptive_engine)
    with TestClient(srv.app) as test_client:
        yield test_client


def test_adaptive_defaults_disabled_and_bounded(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    cfg = ArcherConfig(offload_path="/tmp", use_native_engine=False)
    assert cfg.adaptive_memory_enabled is False
    assert cfg.adaptive_memory_min_expert_cache_bytes > 0
    assert cfg.adaptive_memory_min_kv_cache_blocks > 0
    assert cfg.adaptive_memory_free_reserve_bytes > 0

def test_hot_disable_restores_static_targets(client, adaptive_engine) -> None:
    response = client.post("/v1/config",
        json={"adaptive_memory_enabled": False})
    assert response.status_code == 200
    adaptive_engine.restore_static_memory_targets.assert_called_once_with(
        transactional=True)

def test_hot_disable_keeps_admissions_closed_until_static_publish(
        adaptive_engine) -> None:
    adaptive_engine.restore_static_memory_targets()
    assert adaptive_engine.resize_trace == [
        "pause_admissions", "drain_requests", "drain_dispatch",
        "sync_cuda_events", "trim", "publish_effective",
        "resume_admissions",
    ]

def test_prometheus_contains_controller_state() -> None:
    text = _format_prometheus_metrics({"memory": {"adaptive": {
        "devices": {0: {"enabled": True, "expert_target_bytes": 10,
                        "kv_target_blocks": 4, "resize_failures": 1}}}}})
    assert 'moe_adaptive_memory_enabled{device="0"} 1' in text
    assert 'moe_adaptive_memory_resize_failures_total{device="0"} 1' in text
```

Put the direct formatter assertion beside the production formatter tests and add an endpoint assertion to the existing `tests/python/serving/test_api_routes.py::test_metrics_endpoint` fixture by returning adaptive per-device stats from `_make_mock_stats()`. Keep all API route/metrics coverage in that existing suite rather than creating a parallel API test module.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/unit/test_utils_config.py tests/python/serving/test_hot_reload.py`

Expected: `ArcherConfig` and metrics lack adaptive fields.

- [ ] **Step 3: Add validated knobs and reload semantics**

Add fields matching `AdaptiveMemoryConfig` with prefix `adaptive_memory_`. Validate positive intervals/step/minima/reserve/failure limit, `0 < ewma_alpha <= 1`, and `0 <= hysteresis_ratio <= 1`. Existing expert/KV ratio validation remains the static fallback contract.

`engine.update_config()` accepts only `adaptive_memory_enabled` at runtime. Enabling constructs fresh per-device controller state from startup-validated knobs. Disabling starts the same maintenance transaction on each supported device in deterministic device order: close scheduler/dispatcher admissions, drain request/swap/fetch/execute queues, wait transfers, synchronize all relevant CUDA completion events, cancel uncommitted reservations, trim/grow toward configured static targets, publish each device's effective result, then resume admissions. If a device cannot restore, leave its last safe or partially reduced split, set `fallback_reason="static_restore_deferred"` for that device, and still reopen both gates; never publish configured static ratios unless physical state matches. Other policy knobs require restart to avoid partially mutating controller state.

Prometheus names:

```text
moe_adaptive_memory_enabled
moe_adaptive_memory_fallback_static
moe_adaptive_memory_expert_target_bytes
moe_adaptive_memory_kv_target_blocks
moe_adaptive_memory_resize_attempts_total
moe_adaptive_memory_resize_failures_total
moe_adaptive_memory_reserve_rejections_total
moe_adaptive_memory_expert_miss_cost
moe_adaptive_memory_kv_pressure_cost
```

Every metric carries the bounded `device="<cuda index>"` label and no aggregate is presented as a per-device value. Do not put free-byte values or reasons in labels. `last_reason` stays in `/admin/stats`, not Prometheus labels, to avoid cardinality.

- [ ] **Step 4: Run config/API suites**

Run: `pytest -q tests/python/unit/test_utils_config.py tests/python/serving/test_hot_reload.py tests/python/serving/test_api_routes.py`

Expected: pass; old configurations remain accepted and adaptive mode is off.

- [ ] **Step 5: Commit control-plane rollout surfaces**

```bash
git add moe_infinity/utils/config.py moe_infinity/serving/engine.py moe_infinity/entrypoints/openai/api_server_v2.py tests/python/unit/test_utils_config.py tests/python/serving/test_hot_reload.py tests/python/serving/test_api_routes.py
git commit -m "feat: expose adaptive memory rollout controls"
```

## Task 8: Add deterministic simulations and stability gates

**Files:**
- Create: `benchmarks/serving/adaptive_memory_trace.py`
- Create: `tests/python/benchmark/test_adaptive_memory_trace.py`

- [ ] **Step 1: Write failing replay tests**

Define every trace constructor locally before the tests; only `replay()` and `compare_policies()` are production imports:

```python
import json
from pathlib import Path

from benchmarks.serving.adaptive_memory_trace import compare_policies, replay


def trace_row(
    step: int, *, device_id: int, expert_heavy: bool
) -> dict[str, int | float | bool]:
    return {
        "device_id": device_id,
        "step": step,
        "expert_misses": 80 if expert_heavy else 0,
        "expert_accesses": 100,
        "expert_fetch_stall_ms": 40.0 if expert_heavy else 0.0,
        "kv_used_blocks": 20 if expert_heavy else 63,
        "kv_total_blocks": 64,
        "kv_swap_bytes": 0 if expert_heavy else 64 * 1024**2,
        "kv_swap_stall_ms": 0.0 if expert_heavy else 30.0,
        "kv_preemptions": 0 if expert_heavy else 2,
        "free_gpu_bytes": 2 * 1024**3,
        "kv_supported": True,
        "total_bytes": 8 * 1024**3,
        "model_bytes": 2 * 1024**3,
        "activation_reserve_bytes": 1024**3,
        "kv_block_bytes": 16 * 1024**2,
        "current_expert_bytes": 3 * 1024**3,
        "current_kv_blocks": 64,
    }


def alternating_pressure_trace(
    *, steps: int
) -> list[dict[str, int | float | bool]]:
    return [trace_row(step, device_id=0, expert_heavy=(step // 128) % 2 == 0)
            for step in range(1, steps + 1)]


def mixed_trace() -> list[dict[str, int | float | bool]]:
    rows: list[dict[str, int | float | bool]] = []
    for step in range(1, 257):
        rows.append(trace_row(step, device_id=0, expert_heavy=step <= 128))
        row = trace_row(step, device_id=1, expert_heavy=step > 128)
        row["total_bytes"] = 12 * 1024**3
        rows.append(row)
    return rows


def write_trace(
    tmp_path: Path, rows: list[dict[str, int | float | bool]]
) -> Path:
    path = tmp_path / "adaptive-memory.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_same_trace_produces_identical_decisions(tmp_path: Path) -> None:
    trace = write_trace(tmp_path, alternating_pressure_trace(steps=1024))
    a = replay(trace, policy="adaptive", seed=7)
    b = replay(trace, policy="adaptive", seed=7)
    assert a["decisions"] == b["decisions"]
    assert a["hard_budget_violations"] == 0

def test_stability_gate_limits_resize_frequency(tmp_path: Path) -> None:
    report = replay(write_trace(tmp_path,
        alternating_pressure_trace(steps=1024)), policy="adaptive", seed=7)
    assert report["resize_count"] <= 1024 // report["cooldown_steps"] + 1
    assert report["minimum_capacity_violations"] == 0

def test_report_compares_without_claiming_gain(tmp_path: Path) -> None:
    report = compare_policies(write_trace(tmp_path, mixed_trace()))
    assert set(report["policies"]) == {"fixed", "adaptive"}
    assert "expected_improvement" not in report
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/python/benchmark/test_adaptive_memory_trace.py`

Expected: trace module is missing.

- [ ] **Step 3: Implement JSONL replay and report**

Each trace row contains the exact `MemorySignals` fields including `device_id`, plus that device's total/model/activation bytes, KV block bytes, and current targets. Interleaved rows from different devices update only their own controller state. CLI:

```bash
python benchmarks/serving/adaptive_memory_trace.py \
  --trace tests/fixtures/adaptive_memory/mixed.jsonl \
  --policies fixed adaptive --seed 7 \
  --output-json /tmp/adaptive-memory-trace.json
```

Report per policy and per device: decision sequence, resize count/reasons, expert miss-stall sum, KV swap-stall sum, preemptions, minimum/free-reserve/hard-budget violations, and final effective targets. Add a two-device fixture with unequal memory sizes and opposing pressure; assert no target/result from device 0 appears under device 1. Fixed policy uses configured ratios and does not resize. No threshold asserts adaptive is faster; pass/fail covers safety, determinism, isolation, and stability only.

- [ ] **Step 4: Run replay tests**

Run once: `pytest -q tests/python/benchmark/test_adaptive_memory_trace.py`

Expected: all pass. The test itself performs two in-process replays and proves identical output; do not perform an extra manual rerun.

- [ ] **Step 5: Commit simulation tooling**

```bash
git add benchmarks/serving/adaptive_memory_trace.py tests/python/benchmark/test_adaptive_memory_trace.py
git commit -m "test: add adaptive memory trace simulation"
```

## Task 9: Add CUDA pressure tests and live A/B benchmark output

**Files:**
- Modify: `tests/python/e2e/test_kv_memory.py`
- Modify: `benchmarks/serving/memory.py`
- Create: `tests/python/benchmark/test_memory_benchmark_config.py`

- [ ] **Step 1: Write failing benchmark propagation tests and CUDA pressure test before runtime changes**

Create `tests/python/benchmark/test_memory_benchmark_config.py`:

```python
from argparse import Namespace
from unittest.mock import Mock

from benchmarks.serving import memory

def test_arm_ratios_and_feature_flag_reach_model_load(monkeypatch) -> None:
    seen: list[dict[str, object]] = []
    class FakeMoE:
        def __init__(self, model_name: str, config: dict[str, object]) -> None:
            seen.append(dict(config))
            self.model = Mock(config=Mock())
            self.engine = Mock(config=Mock(
                device_memory_ratio=config["device_memory_ratio"],
                kv_cache_ratio=config["kv_cache_ratio"],
                adaptive_memory_enabled=config["adaptive_memory_enabled"]))
    monkeypatch.setattr(memory, "_moe_class", lambda: FakeMoE)
    monkeypatch.setattr(memory, "_load_tokenizer", lambda _: Mock(pad_token_id=0))
    arm = memory.ArmConfig("adaptive", 0.61, 0.37, True)
    model, _ = memory.load_model_and_tokenizer("m", "/offload", arm)
    assert seen == [{"offload_path": "/offload",
                     "device_memory_ratio": 0.61,
                     "kv_cache_ratio": 0.37,
                     "adaptive_memory_enabled": True}]
    assert memory.effective_arm_config(model, arm) == {
        "arm": "adaptive", "device_memory_ratio": 0.61,
        "kv_cache_ratio": 0.37, "adaptive_memory_enabled": True}

def test_report_separates_requested_and_effective_config(monkeypatch) -> None:
    arms = [memory.ArmConfig("fixed", 0.55, 0.20, False),
            memory.ArmConfig("adaptive", 0.55, 0.20, True)]
    monkeypatch.setattr(memory, "run_arm", lambda arm, args: {
        "requested_config": arm.as_dict(),
        "effective_config": {**arm.as_dict(), "kv_cache_ratio": 0.19},
        "output_token_ids": [1, 2], "safety": {"violations": 0}})
    report = memory.compare_arms(arms, Namespace(seed=7))
    assert report["arms"][0]["requested_config"]["kv_cache_ratio"] == 0.20
    assert report["arms"][0]["effective_config"]["kv_cache_ratio"] == 0.19
    assert report["arms"][1]["effective_config"]["adaptive_memory_enabled"] is True
```

Then add the CUDA contracts:

```python
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch


@dataclass(frozen=True)
class DevicePressureResult:
    hard_budget_violations: int


@dataclass(frozen=True)
class PressureResult:
    output_token_ids: list[int]
    hard_budget_violations: int
    minimum_capacity_violations: int
    min_free_gpu_bytes: int
    configured_reserve_bytes: int
    resize_count: int
    max_resize_count: int
    per_device: dict[int, DevicePressureResult]
    completed: bool
    fallback_static: bool
    resize_failures: int
    failure_limit: int


@pytest.fixture(scope="module")
def memory_model(memory_bundle: tuple[Any, Any]) -> tuple[Any, Any]:
    return memory_bundle


def run_pressure(
    memory_model: tuple[Any, Any], *, adaptive: bool, seed: int
) -> PressureResult:
    tokenizer, build_engine = memory_model
    engine = build_engine(kv_cache_ratio=0.25, adaptive=adaptive)
    prompts = _make_prompt_batches(tokenizer, seed=seed)
    outputs = _run_batch(engine, prompts)
    adaptive_stats = engine.get_stats()["memory"]["adaptive"]
    devices = adaptive_stats["devices"]
    return PressureResult(
        output_token_ids=[token for request in sorted(outputs)
                          for token in outputs[request]],
        hard_budget_violations=sum(
            int(item["hard_budget_violations"]) for item in devices.values()
        ),
        minimum_capacity_violations=sum(
            int(item["minimum_capacity_violations"])
            for item in devices.values()
        ),
        min_free_gpu_bytes=min(
            int(item["min_free_gpu_bytes"]) for item in devices.values()
        ),
        configured_reserve_bytes=min(
            int(item["configured_reserve_bytes"]) for item in devices.values()
        ),
        resize_count=sum(int(item["resize_count"]) for item in devices.values()),
        max_resize_count=sum(
            int(item["max_resize_count"]) for item in devices.values()
        ),
        per_device={
            int(device): DevicePressureResult(
                hard_budget_violations=int(item["hard_budget_violations"])
            )
            for device, item in devices.items()
        },
        completed=bool(adaptive_stats["completed"]),
        fallback_static=any(
            bool(item["fallback_static"]) for item in devices.values()
        ),
        resize_failures=sum(
            int(item["resize_failures"]) for item in devices.values()
        ),
        failure_limit=int(adaptive_stats["failure_limit"]),
    )


def raise_oom_on_resize(*args: object, **kwargs: object) -> torch.Tensor:
    _ = (args, kwargs)
    raise torch.OutOfMemoryError("injected adaptive resize OOM")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_adaptive_pressure_preserves_reserve_and_outputs(memory_model) -> None:
    fixed = run_pressure(memory_model, adaptive=False, seed=11)
    adaptive = run_pressure(memory_model, adaptive=True, seed=11)
    assert adaptive.output_token_ids == fixed.output_token_ids
    assert adaptive.hard_budget_violations == 0
    assert adaptive.minimum_capacity_violations == 0
    assert adaptive.min_free_gpu_bytes >= adaptive.configured_reserve_bytes
    assert adaptive.resize_count <= adaptive.max_resize_count
    assert set(adaptive.per_device) == set(range(torch.cuda.device_count()))
    assert all(d.hard_budget_violations == 0
               for d in adaptive.per_device.values())

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_injected_resize_oom_falls_back_to_static(memory_model, monkeypatch) -> None:
    monkeypatch.setattr(
        "moe_infinity.runtime.attention_backend._allocate_resize_tensor",
        raise_oom_on_resize,
    )
    result = run_pressure(memory_model, adaptive=True, seed=13)
    assert result.completed
    assert result.fallback_static
    assert result.resize_failures >= result.failure_limit
```

Update the existing `memory_bundle.build_engine` closure to accept `adaptive: bool`, pass `adaptive_memory_enabled=adaptive` into the engine config, and update `_run_batch()` to return the exact `run_until_done()` dictionary instead of discarding it. Add `_allocate_resize_tensor(*shape, dtype, device)` as the production allocation seam used only by adaptive backend recreation; normal initialization continues to call `torch.empty` directly. These definitions occur before the tests, so the RED result is missing adaptive telemetry/allocation behavior rather than missing fixtures.

- [ ] **Step 2: Verify RED for model-load propagation, then on a CUDA worker**

Run: `pytest -q tests/python/benchmark/test_memory_benchmark_config.py`

Expected: fail because `ArmConfig`, `effective_arm_config`, `run_arm`, and `compare_arms` do not exist and `load_model_and_tokenizer()` still hard-codes `0.75`.

Run: `CUDA_VISIBLE_DEVICES=0 pytest -q tests/python/e2e/test_kv_memory.py -k adaptive_pressure`

Expected: helper/result/controller telemetry is absent. On CPU-only hosts the tests skip; that is not CUDA evidence.

- [ ] **Step 3: Propagate each arm's effective model/runtime configuration and extend output without a win threshold**

Add immutable `ArmConfig(arm, device_memory_ratio, kv_cache_ratio, adaptive_memory_enabled)`. Change `load_model_and_tokenizer(model_name, offload_dir, arm)` to pass all four values shown in the test into `MoE`; remove the hard-coded `0.75`. Construct separate fixed and adaptive model/runtime instances so each arm's loader and `ContinuousBatchingEngine` receive the same requested ratios and only the adaptive flag differs. Read effective values back from the loaded `ArcherConfig`/engine config after validation/default adjustment and store both `requested_config` and `effective_config`; if they differ, keep both rather than relabeling requested values as effective.

Add `--adaptive-memory`, all bounded policy knobs, `--warmup-runs`, `--repetitions`, and `--trace-output`. Propagate every bounded knob through the adaptive arm's model config; fixed arm keeps `adaptive_memory_enabled=False`. Emit environment, seed, arm order, requested/effective model config, per-device initial/final targets, TTFT, decode throughput, p50/p95 latency, expert hit/miss/stall, KV utilization/swap/stall/preemptions, resize decisions, minimum free bytes, reserve, peak allocated/reserved bytes, fallback state, and output-token equality. Alternate arm order by seed to reduce thermal/order bias. A blocked arm still emits requested config and any effective config available before the block.

- [ ] **Step 4: Run CUDA safety test and benchmark smoke**

Run: `pytest -q tests/python/benchmark/test_memory_benchmark_config.py && CUDA_VISIBLE_DEVICES=0 pytest -q tests/python/e2e/test_kv_memory.py -k 'adaptive_pressure or injected_resize_oom'`

Expected: pass with zero hard-budget violations and equal output tokens.

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/serving/memory.py \
  --model "$MODEL" --offload-dir "$OFFLOAD_DIR" \
  --batch-size 8 --prompt-length 128 --max-new-tokens 32 \
  --adaptive-memory --warmup-runs 1 --repetitions 3 \
  --output-json /tmp/adaptive-memory-live.json
```

Expected: status `PASS` or an explicit `BLOCKED` reason, fixed and adaptive measurements, requested/effective configs proving ratios reached model loading, per-device targets/safety counters, output equality, and no asserted throughput improvement.

- [ ] **Step 5: Commit pressure evidence**

```bash
git add tests/python/e2e/test_kv_memory.py benchmarks/serving/memory.py tests/python/benchmark/test_memory_benchmark_config.py
git commit -m "test: validate adaptive memory under cuda pressure"
```

## Risks and operational rollback contract

- **Deadlock while closing admissions:** scheduler, transfer, task-pool, and dispatcher locks must be acquired in that order and never while waiting on a CUDA event. Inside `ArcherTaskPool`, the source-matching rule is stricter: snapshot `exec_queue_` under `exec_mutex_`, release it, then snapshot candidates under `candidates_mutex_`; never hold both mutexes, in either order. `unified_mutex_` may precede `candidates_mutex_` only in the existing candidate-replacement path. The focused pause-hook test proves candidate replacement completes after the exec snapshot and before reservation resumes. Every wait has a deadline; timeout cancels uncommitted reservations, restores queue order/status, clears both admission gates, and records a device-local failure.
- **Use-after-free on asynchronous CUDA work:** queue emptiness is insufficient. Tensor/pool/wrapper replacement requires synchronized completion events from KV-copy, current attention/model, expert fetch, and every device execution stream. Old objects remain strongly referenced through the post-publication event.
- **FlashInfer stale page plans:** serving `_kv_cache` plus its independent `_fi_prefill`/`_fi_decode`, and native `_fi_kv_cache` plus its independent prefill/decode wrappers, are transactional publication units. Any constructor/first-plan failure closes admissions, restores all retained old fields, safely replans the old wrappers, and resumes; mixed old/new stores or wrappers are forbidden, and old bundles survive through CUDA completion.
- **Irreversible expert shrink:** victim reservation is reversible only before host move/storage release. A later receiver failure publishes `PARTIAL_DONOR_COMMITTED`; normal misses refill experts, and telemetry/benchmarks report the reduced effective target.
- **Multi-GPU cross-talk:** signals, budgets, gates, targets, results, failures, and metrics are keyed by `device_id`. Unequal-GPU tests reject any use of aggregate/device-0 capacity for another device.
- **Wrong-GPU native swaps:** `device_id` is carried from KV-manager/scheduler ownership into preemption, swap-out/in, resize restoration, `TransferRequest`, and `UnifiedTransferScheduler`; endpoints are validated and device-local drain waits cannot be satisfied by another GPU's transfers.
- **Benchmark configuration drift:** each arm gets a fresh model load with requested ratios/flags; reports preserve both requested and effective config. A mismatch is evidence, not silently normalized.

Operational rollback is transactional, not merely a flag flip. `POST /v1/config {"adaptive_memory_enabled": false}` closes admissions, drains all relevant queues, synchronizes streams, attempts the configured static split per device, publishes physical effective state, and resumes. If exact static restoration cannot fit, serving resumes at the last safe/partially reduced split with `static_restore_deferred`; restart with adaptive absent/false remains the final rollback. No rollback claims to resurrect evicted experts synchronously.

## Task 10: Document feature-flag rollout, rollback, and evidence

**Files:**
- Modify: `docs/configuration.md`
- Modify: `docs/serving.md`
- Modify: `docs/benchmarking.md`
- Create: `tests/python/unit/test_docs.py`

- [ ] **Step 1: Write documentation assertions**

Create `tests/python/unit/test_docs.py` with the exact source-level contract:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def test_adaptive_memory_docs_cover_configuration_rollout_and_rollback() -> None:
    configuration = (ROOT / "docs/configuration.md").read_text("utf-8")
    serving = (ROOT / "docs/serving.md").read_text("utf-8")
    benchmarking = (ROOT / "docs/benchmarking.md").read_text("utf-8")

    for text in (configuration, serving, benchmarking):
        assert "adaptive_memory_enabled" in text
        assert "free-memory reserve" in text.lower()
    assert 'POST /v1/config {"adaptive_memory_enabled": false}' in serving
    assert "Stage 0" in serving and "Stage 4" in serving
    assert "drain" in serving.lower() and "CUDA completion event" in serving
    assert "partial_donor_committed" in serving.lower()
    assert "_fi_prefill" in configuration and "_fi_decode" in configuration
    assert "device_id" in serving and "cuda:0" in serving
    assert "requested_config" in benchmarking
    assert "effective_config" in benchmarking
    assert "no assumed performance gain" in benchmarking.lower()
    assert "https://arxiv.org/abs/2606.21868" in benchmarking
```

Verify RED before editing docs.

Run: `pytest -q tests/python/unit/test_docs.py -k adaptive_memory`

Expected: fail because the configuration and rollout text is absent.

- [ ] **Step 2: Document staged rollout**

Document these stages exactly:

1. **Stage 0:** default off; collect fixed-split telemetry only.
2. **Stage 1:** trace simulation in CI; require zero safety/minimum violations and deterministic decisions.
3. **Stage 2:** one CUDA canary with adaptive enabled; alert on reserve rejection, resize failure, fallback latch, output mismatch, or increased OOM/preemption rate.
4. **Stage 3:** small production percentage; compare fixed/adaptive distributions without declaring a gain from a single run.
5. **Stage 4:** broader opt-in only after model/workload-specific evidence.

Rollback is `POST /v1/config {"adaptive_memory_enabled": false}`; document that it closes admissions, drains request/transfer/fetch/execute queues, synchronizes CUDA completion events, attempts static targets per device, publishes effective state, and only then resumes. If static restoration is deferred, admissions continue at the last safe or partially reduced split and stats say so. Process restart with the feature absent/false is the final rollback. Include every metric name with its device label, the serving `_kv_cache` plus independent `_fi_prefill`/`_fi_decode` reconstruction/retention/replan contract, the native all-store/all-wrapper recreation contract, `PARTIAL_DONOR_COMMITTED`, and explain that native preemption/swap/transfer endpoints use the owning `device_id` rather than a hard-coded `cuda:0`. Explain that policy knobs other than enable require restart.

- [ ] **Step 3: Document benchmark interpretation**

State that safety/output parity are gates, performance is reported with repetitions and distributions, and regressions or no gain are valid outcomes. Require reports to show requested and effective model/runtime config for both arms and per-device safety/targets. Cite WiSP only in a motivation paragraph and do not present its numbers as MoE-Infinity expectations.

- [ ] **Step 4: Run documentation checks**

Run: `pytest -q tests/python/unit/test_docs.py -k adaptive_memory`

Expected: pass with all config names, metrics, rollout stages, and rollback command present.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/configuration.md docs/serving.md docs/benchmarking.md tests/python/unit/test_docs.py
git commit -m "docs: add adaptive memory rollout guide"
```

## Task 11: Final verification and release gate

**Files:** all files listed above; no additional production files.

- [ ] **Step 1: Run CPU policy, serving, native metadata, config, and simulation suites**

```bash
pytest -q \
  tests/python/unit/test_adaptive_memory.py \
  tests/python/serving/test_memory_manager.py \
  tests/python/unit/test_memory_coordinator.py \
  tests/python/serving/test_memory_resize.py \
  tests/python/unit/test_native_memory_resize.py \
  tests/python/unit/test_kv_cache_manager.py \
  tests/python/unit/test_attention_backend.py \
  tests/python/unit/test_flashinfer_attention_backend.py \
  tests/python/unit/test_scheduler.py \
  tests/python/unit/test_unified_scheduler.py \
  tests/python/unit/test_transfer_scheduler_interface.py \
  tests/python/unit/test_engine_types.py \
  tests/python/unit/test_kv_handler_registration.py \
  tests/python/integration/test_expert_kv_integration.py \
  tests/python/serving/test_engine.py \
  tests/python/serving/test_hot_reload.py \
  tests/python/serving/test_api_routes.py \
  tests/python/benchmark/test_adaptive_memory_trace.py \
  tests/python/benchmark/test_memory_benchmark_config.py
```

Expected: all pass on CPU; no CUDA allocation is required by controller tests.

- [ ] **Step 2: Run native extension tests**

```bash
cmake -S . -B build -GNinja -DMOE_BUILD_TESTS=ON
cmake --build build --target test_sparse_cache_resize -j2
ctest --test-dir build -R sparse_cache_resize --output-on-failure
```

Expected: configure proves the target and both fixture translation units are registered, then build and CTest pass; production resize API signatures, pinned/in-flight/candidate cases, reversible reservations, queue/worker drain, stream completion, and admission reopening are covered.

- [ ] **Step 3: Run one CUDA safety gate**

```bash
CUDA_VISIBLE_DEVICES=0 pytest -q tests/python/e2e/test_kv_memory.py \
  -k 'adaptive_pressure or injected_resize_oom'
```

Expected: pass on a CUDA worker, equal outputs, zero hard-budget/minimum violations, reserve maintained, and injected OOM falls back without terminating serving.

- [ ] **Step 4: Run static-mode regression suites**

```bash
pytest -q \
  tests/python/unit/test_kv_cache_free.py \
  tests/python/unit/test_kv_swap_recovery.py \
  tests/python/integration/test_swap_scheduling.py \
  tests/python/integration/test_output_equivalence.py \
  tests/python/dflash/test_route_ahead_wire.py
```

Expected: all pass with `adaptive_memory_enabled=False`; no resize metric increments.

- [ ] **Step 5: Inspect one live report and record evidence, not a claim**

Run the Task 9 benchmark command and attach `/tmp/adaptive-memory-live.json` to the implementation PR/artifact. Record hardware, model revision, seed, repetitions, fixed/adaptive requested and effective configs, per-device initial/final targets, raw distributions, safety counters, output parity, and any fallback/partial donor commit. Do not summarize a gain unless the measured repeated data supports it.

- [ ] **Step 6: Commit only verification fixture adjustments if required**

```bash
git status --short
git diff --check
```

Expected: no generated benchmark JSON, model cache, build output, or trace artifact is staged. If no fixture adjustment was required, do not create an empty commit.

## Acceptance checklist

- [ ] Adaptive mode is disabled by default and existing static ratios behave identically.
- [ ] Every proposal and commit obeys the hard per-GPU budget, free reserve, minima, whole KV blocks, and maximum step.
- [ ] Controller EWMA/cooldown/targets/results/failure latch/reporting are keyed per device; unequal multi-GPU tests prove isolation.
- [ ] Policy is deterministic, interval-based, hysteretic, cooled down, and bounded.
- [ ] Expert miss cost and KV pressure/swap cost both affect direction.
- [ ] Receiver growth occurs only after donor shrink and reserve verification.
- [ ] Scheduler and dispatcher admissions stop before maintenance; request, swap, fetch, execute, and task-pool queues drain before mutation; all exit paths resume without dropping or reordering arrivals.
- [ ] Completion events from KV-copy, attention/model, expert-fetch, and all affected execution streams synchronize before any tensor/pool/wrapper replacement or release.
- [ ] Serving `_kv_cache`, `_fi_prefill`, and `_fi_decode` are reconstructed and published transactionally; both independent wrappers are freshly planned, and the old storage/wrapper bundle remains retained through CUDA completion and safe replan/rollback.
- [ ] Pinned/in-flight/protected experts and referenced KV blocks cannot be directly evicted by resize.
- [ ] Expert victim reservations are reversible before eviction; post-eviction receiver failure reports and publishes `PARTIAL_DONOR_COMMITTED` rather than claiming resurrection.
- [ ] Serving and native KV storage are physically recreated only while drained; dtype/layout are unchanged; FlashInfer cache plus both wrappers are recreated and freshly planned as one unit.
- [ ] Native scheduler ownership propagates `device_id` through resize preemption, swap-out, swap-in, KV-manager calls, `TransferRequest`, and `UnifiedTransferScheduler`; no hard-coded `cuda:0` remains in those paths, and unequal-device tests prove routing/drain isolation.
- [ ] The focused native RED target compiles and links complete `FakeSparseCache`, production `Node`/`NodeExecState`, and `FakeDispatcher` fixture definitions before failing only on missing resize APIs.
- [ ] Failed/partial resizes leave recoverable request state and report physical effective targets.
- [ ] Repeated failures latch static fallback; hot disable and restart provide rollback.
- [ ] JSON stats and Prometheus expose per-device targets, costs, attempts, failures, reserve rejections, partial commits, and fallback without high-cardinality labels.
- [ ] CPU simulations prove invariants/stability/determinism; CUDA tests prove pressure safety/output parity.
- [ ] Fixed/adaptive benchmark arms propagate ratios/flags into separate model loads and report requested plus effective configuration, raw evidence, and no assumed gain.
- [ ] WiSP is cited as motivation only.
