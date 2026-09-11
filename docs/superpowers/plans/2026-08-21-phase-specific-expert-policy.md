# Phase-Specific Expert Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently tunable prefill and decode expert admission, prefetch, and eviction behavior while retaining one shared expert store and one shared GPU residency set.

**Architecture:** The serving runner derives an explicit `ExpertPhase` from `BatchMetadata.is_prefill` and scopes every model forward with that phase. Decode-first splitting of mixed batches occurs only when `phase_specific_expert_policy=True`; disabled mode retains the existing non-paged combined forward and paged prefill-then-decode split exactly. Python propagates phase through `DistributedExpertExecutor` and `ExpertPrefetcher`; both the native dispatcher and prefetch scheduler submit admission/lease/eviction operations to one synchronized `ExpertResidencyManager`, which starts without capacity and rejects admission until `SetTopology`/`SetTopologyV2` has completed and configured each device from the finalized sparse-cache limit.

**Tech Stack:** Python 3.10+, PyTorch, dataclasses/contextvars, pybind11, C++17/CUDA, pytest, GoogleTest, Prometheus text exposition.

---

## Scope and non-goals

This plan changes policy only. It does not create separate GPU pools, duplicate expert weights, disaggregate prefill and decode, alter router outputs, or change expert ownership. [DuoServe-MoE](https://arxiv.org/abs/2509.07379) motivates treating prefill and decode as different operating regimes; it is not used as evidence for a speedup, and this implementation does not reproduce that system's P/D disaggregation.

### Policy invariants

1. **Single identity and residency:** `(layer_id, expert_id)` maps to exactly one native `Node`, one host-store entry, and at most one GPU-resident copy per owning GPU. Phase is request metadata, never part of the cache key.
2. **Demand is never rejected after runtime readiness:** before capacity configuration, the manager rejects admission tickets and serving setup must not issue model work. After topology has configured the target device, every routed expert is fetched; admission controls whether a miss remains resident after use or uses the existing single transient overflow slot.
3. **Capacity and safety:** the manager starts unconfigured and rejects every admission until a successful per-device `ConfigureCapacity(gpu_id, bytes)` call. Capacity is derived only after topology initialization and device/host memory pools can report the finalized sparse-cache limit; persistent resident bytes never exceed that limit. Per-device reconfiguration is atomic and cannot install a negative capacity or a limit below already-resident bytes. At most one transient overflow expert exists per GPU. The sole production eligibility predicate rejects `is_prefetching`, non-`IDLE` `exec_state`, `pending_dispatches > 0`, `is_overflow`, active leases, and protected candidates.
4. **No phase flush:** prefill-to-decode transition does not clear the cache or reset counters. A decode hit on an expert admitted during prefill is recorded as `transition_hits`.
5. **Mixed batches:** with the feature enabled, `ServingEngine._execute_batch` executes decode rows first and prefill rows second, once each, then restores original output order. With the feature disabled, the method retains the current behavior byte-for-byte: non-paged mixed batches execute once as mixed, while paged mixed batches execute prefill then decode. Direct mixed `ModelRunner.execute` calls are tagged `mixed`, disable speculative prefetch, and use decode-safe admission/eviction only when the native policy is enabled.
6. **Determinism:** victim order is `(lowest weighted utility, oldest phase-local sequence, layer_id, expert_id)`. It never depends on wall-clock time or unordered-set iteration.
7. **Starvation protection:** on-demand band 0 remains strict. Among prefetch bands, a task bypassed `expert_policy_starvation_limit` times is promoted for one service opportunity; promotion never passes on-demand work.
8. **One authority:** `ExpertResidencyManager` is the only component allowed to mutate persistent resident membership, resident bytes, lease counts, protected-candidate state, eviction reservations, or residency counters. Dispatcher and prefetch code may transfer bytes only under a manager-issued ticket and must commit or abort that ticket.
9. **Safe rollback:** with `phase_specific_expert_policy=False`, decode-first splitting is not entered, no phase-specific native configuration is installed, legacy priority/top-2 behavior and the current `batch_size > 1` overload branch remain active, and new telemetry reports zero/disabled without changing routing or cache state.

### Existing integration points and intentional non-edits

* `moe_infinity/serving/scheduler.py:230-328` already emits distinct `prefill_seq_ids` and `decode_seq_ids`; `:381-405` performs the status transition after the prefill step. The plan consumes those facts and does not add a second scheduler or alter request admission.
* `moe_infinity/serving/batch.py:217-273` already materializes row-level `is_prefill`, and `:193-214` already splits/recombines packed batches. Task 2 reuses that representation rather than adding duplicate phase metadata.
* `moe_infinity/memory/offloading_policy.py` implements generic Python LRU/ARC containers, but the active expert residency and eviction path is the native dispatcher/task scheduler. It remains unchanged to avoid a second source of cache truth.
* `core/prefetch/task_scheduler.cpp:245-329` and `core/parallel/expert_dispatcher.cpp:341-358` are the two current native sparse-victim paths. Task 5 removes their independent victim/accounting logic in enabled mode; both call the same synchronized `ExpertResidencyManager` transaction API.
* `moe_infinity/distributed/expert_executor.py:187-291` is the last Python seam before native `enqueue_expert`; phase propagation must terminate there for local and RPC dispatch without changing router masks or weights.
* `core/prefetch/archer_prefetch_handle.cpp:20-38` creates the topology and memory-pool globals before topology exists, while `:396-408` is where `InitializeTopology`/`InitializeTopologyV2` completes. `core/model/model_topology.cpp:849-867` can compute the real per-device sparse-cache limit only after that completion. Task 5 therefore creates the shared manager unconfigured and installs capacities at the end of both topology setters, never in the handle or dispatcher constructor.

## File map

| File | Responsibility |
| --- | --- |
| `moe_infinity/memory/expert_policy.py` | Canonical phase enum, forward-scoped context, immutable Python policy settings, phase resolution. |
| `moe_infinity/utils/config.py` | Backward-compatible feature flag and validated phase policy tunables. |
| `moe_infinity/serving/model_runner.py` | Derive and scope the phase around model forward. |
| `moe_infinity/serving/engine.py` | Decode-first split/recombine for mixed batches and expert-policy stats exposure. |
| `moe_infinity/distributed/expert_executor.py` | Read current phase and pass it to prefetch and native dispatch. |
| `moe_infinity/memory/expert_prefetcher.py` | Phase-specific top-k, priority, correction, and telemetry adapter. |
| `moe_infinity/runtime/model_offload.py` | Build policy settings and configure the Python/native runtime once. |
| `core/prefetch/expert_policy.h` | Native phase/settings types and pure deterministic score/bypass helpers used by the residency authority. |
| `core/prefetch/expert_residency.{h,cpp}` | Sole synchronized authority for resident membership, bytes, leases, candidate protection, eligibility, victim reservation, commit/abort, and residency telemetry. |
| `setup.py` | Register `core/prefetch/expert_residency.cpp` in `_STORE_SOURCES` so editable/wheel builds link the authority into `moe_infinity._store`. |
| `core/model/model_topology.h` | Per-node phase metadata storage mutated only by `ExpertResidencyManager`; no phase-specific expert copies. |
| `core/prefetch/task_scheduler.{h,cpp}` | Phase-tagged prefetch tasks and bounded bypass; delegates all residency decisions/accounting to `ExpertResidencyManager`. |
| `core/parallel/expert_dispatcher.{h,cpp}` | Explicit phase on demand dispatch; delegates all residency decisions/accounting to `ExpertResidencyManager`. |
| `core/prefetch/archer_prefetch_handle.{h,cpp}` | Carry phase on prefetch requests, configure each manager device only after topology/memory limits are ready, and expose scheduler metrics. |
| `core/python/py_archer_prefetch.cpp` | Bind phase/config arguments and stats snapshots. |
| `moe_infinity/entrypoints/openai/api_server_v2.py` | Tested `--[no-]phase-specific-expert-policy` seam plus Prometheus phase labels. |
| `benchmarks/serving/phase_specific_expert_policy.py` | Reproducible off/on workload matrix with TTFT/TPOT and policy telemetry. |
| `docs/configuration.md` | Defaults, compatibility, invariants, operational rollback. |
| `docs/benchmarking.md` | Validation matrix and interpretation rules without performance claims. |
| `tests/python/unit/test_setup_sources.py` | AST-level regression that `_STORE_SOURCES` includes the residency implementation without importing `setup.py`. |
| `tests/python/unit/test_native_capacity_lifecycle.py` | Source-order regression proving constructors do not infer capacity and both topology setters configure only after initialization from sparse limits. |
| `tests/cpp/unit/prefetch/expert_residency_test_fixture.h` | Real-manager fixture with real `Node` objects and injected fake transfer operations. |
| `tests/cpp/unit/prefetch/test_expert_residency.cpp` | Unconfigured lifecycle, per-device reconfiguration, capacity, duplicate admission, reservation race, idempotence, lease, and shared demand/prefetch accounting against the real manager. |

## Adaptive-precision integration constraint

This branch is independently implementable for the current fixed-size expert `Node` representation, but it is **not composable by parallel merge** with adaptive expert precision. Adaptive precision changes a resident expert's byte size and may introduce multiple materialized precision variants; independently adding that state beside this plan's residency state would recreate two accounting and eviction authorities.

The shared prerequisite for composition is the `ExpertResidencyManager` lease/accounting substrate in Task 5: one logical expert key, variant-aware byte accounting, manager-issued transfer/compute leases, protected-candidate handling, and transactional admission/eviction. Recommended sequencing is: (1) land and validate this phase-policy branch with fixed-size entries; (2) rebase adaptive precision on the manager API; (3) extend `ResidencyEntry` with precision/variant metadata and make precision conversion a manager transaction; (4) add cross-feature tests before enabling both flags. This plan does not add adaptive-precision fields, variant selection, conversion, or configuration, so it remains independently shippable.

## Public configuration contract

Add these `ArcherConfig` fields. The master flag defaults off; therefore old JSON files, old Python constructors, and old command paths remain behaviorally compatible.

| Field | Type | Default | Validation / meaning when enabled |
| --- | --- | --- | --- |
| `phase_specific_expert_policy` | `bool` | `False` | Master gate. |
| `prefill_expert_admission` | `str` | `"transient_on_pressure"` | `cache` or `transient_on_pressure`. |
| `decode_expert_admission` | `str` | `"cache"` | `cache` or `transient_on_pressure`. |
| `prefill_expert_prefetch_top_k` | `int` | `0` | `[0, num_experts]`; zero disables predictive prefetch for prefill. |
| `decode_expert_prefetch_top_k` | `int` | `2` | `[0, num_experts]`. |
| `prefill_expert_prefetch_priority` | `int` | `2` | Native band `[1, 19]`. |
| `decode_expert_prefetch_priority` | `int` | `1` | Native band `[1, 19]`. |
| `prefill_expert_eviction_weight` | `float` | `1.0` | Finite and `> 0`. |
| `decode_expert_eviction_weight` | `float` | `4.0` | Finite and `> 0`. |
| `expert_policy_starvation_limit` | `int` | `8` | Positive maximum prefetch bypasses. |

The defaults intentionally protect decode residency and avoid speculative prefill traffic, but they are inert until the master flag is true.

---

### Task 1: Define the phase contract and validated configuration

**Files:**
- Create: `moe_infinity/memory/expert_policy.py`
- Modify: `moe_infinity/memory/__init__.py`
- Modify: `moe_infinity/utils/config.py:17-162`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:1028-1070,1868-1919`
- Modify: `tests/python/unit/test_utils_config.py`
- Create: `tests/python/unit/test_expert_policy.py`
- Modify: `tests/python/integration/test_stability_e2e.py`

- [ ] **Step 1: Write failing config and phase-context tests**

```python
# tests/python/unit/test_expert_policy.py
from moe_infinity.memory.expert_policy import (
    ExpertPhase,
    current_expert_phase,
    expert_phase_scope,
)


def test_phase_scope_restores_nested_state() -> None:
    assert current_expert_phase() is ExpertPhase.MIXED
    with expert_phase_scope(ExpertPhase.PREFILL):
        assert current_expert_phase() is ExpertPhase.PREFILL
        with expert_phase_scope(ExpertPhase.DECODE):
            assert current_expert_phase() is ExpertPhase.DECODE
        assert current_expert_phase() is ExpertPhase.PREFILL
    assert current_expert_phase() is ExpertPhase.MIXED
```

```python
# append to tests/python/unit/test_utils_config.py
def test_phase_policy_defaults_are_backward_compatible(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(offload_path="/tmp", use_native_engine=False)
    assert config.phase_specific_expert_policy is False
    assert config.prefill_expert_admission == "transient_on_pressure"
    assert config.decode_expert_admission == "cache"
    assert config.prefill_expert_prefetch_top_k == 0
    assert config.decode_expert_prefetch_top_k == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prefill_expert_admission", "drop", "must be one of"),
        ("decode_expert_prefetch_top_k", -1, "must be >= 0"),
        ("decode_expert_prefetch_priority", 0, "must be in [1, 19]"),
        ("prefill_expert_eviction_weight", 0.0, "must be finite and > 0"),
        ("expert_policy_starvation_limit", 0, "must be > 0"),
    ],
)
def test_phase_policy_rejects_invalid_values(monkeypatch, field, value, message):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    kwargs = {"offload_path": "/tmp", "use_native_engine": False, field: value}
    with pytest.raises(ValueError, match=message):
        ArcherConfig(**kwargs)
```

Add a real server CLI seam test; do not rely on an undeclared environment variable:

```python
# append to tests/python/integration/test_stability_e2e.py
def test_phase_policy_cli_defaults_off_and_can_enable(monkeypatch) -> None:
    base = [
        "api_server_v2",
        "--model", "fixture/model",
        "--offload-dir", "/tmp/moe-policy-test",
    ]
    monkeypatch.setattr(sys, "argv", base)
    assert server_module.parse_args().phase_specific_expert_policy is False

    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--phase-specific-expert-policy"],
    )
    assert server_module.parse_args().phase_specific_expert_policy is True

    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--no-phase-specific-expert-policy"],
    )
    assert server_module.parse_args().phase_specific_expert_policy is False
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest tests/python/unit/test_expert_policy.py tests/python/unit/test_utils_config.py tests/python/integration/test_stability_e2e.py::test_phase_policy_cli_defaults_off_and_can_enable -q`

Expected: collection fails because `moe_infinity.memory.expert_policy`, the new `ArcherConfig` fields, and the server CLI argument do not exist.

- [ ] **Step 3: Implement the canonical Python phase API**

```python
# moe_infinity/memory/expert_policy.py
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator


class ExpertPhase(IntEnum):
    PREFILL = 0
    DECODE = 1
    MIXED = 2


_CURRENT_EXPERT_PHASE: ContextVar[ExpertPhase] = ContextVar(
    "current_expert_phase", default=ExpertPhase.MIXED
)


def current_expert_phase() -> ExpertPhase:
    return _CURRENT_EXPERT_PHASE.get()


@contextmanager
def expert_phase_scope(phase: ExpertPhase) -> Iterator[None]:
    token = _CURRENT_EXPERT_PHASE.set(phase)
    try:
        yield
    finally:
        _CURRENT_EXPERT_PHASE.reset(token)


@dataclass(frozen=True)
class PhasePolicySettings:
    enabled: bool
    prefill_admission: str
    decode_admission: str
    prefill_prefetch_top_k: int
    decode_prefetch_top_k: int
    prefill_prefetch_priority: int
    decode_prefetch_priority: int
    prefill_eviction_weight: float
    decode_eviction_weight: float
    starvation_limit: int

    def effective_phase(self, phase: ExpertPhase) -> ExpertPhase:
        return ExpertPhase.DECODE if phase is ExpertPhase.MIXED else phase


__all__ = [
    "ExpertPhase",
    "PhasePolicySettings",
    "current_expert_phase",
    "expert_phase_scope",
]
```

Export `ExpertPhase` and `PhasePolicySettings` from `moe_infinity/memory/__init__.py`. Add the fields in the configuration table above to `ArcherConfig`, then validate admissions, non-negative top-k, priorities in `[1, 19]`, finite positive weights via `math.isfinite`, and a positive starvation limit in `__post_init__` after the existing memory-ratio validation. Add the tested server seam:

```python
parser.add_argument(
    "--phase-specific-expert-policy",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable phase-specific expert admission, prefetch, and eviction policy.",
)
```

When `api_server_v2` constructs the `MoE` config dictionary, include:

```python
"phase_specific_expert_policy": bool(args.phase_specific_expert_policy),
```

This provides both `--phase-specific-expert-policy` and `--no-phase-specific-expert-policy`; no environment-variable seam is introduced.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/python/unit/test_expert_policy.py tests/python/unit/test_utils_config.py tests/python/integration/test_stability_e2e.py::test_phase_policy_cli_defaults_off_and_can_enable -q`

Expected: all tests pass; old-config tests still pass without specifying new keys.

- [ ] **Step 5: Commit the contract**

```bash
git add moe_infinity/memory/expert_policy.py moe_infinity/memory/__init__.py moe_infinity/utils/config.py moe_infinity/entrypoints/openai/api_server_v2.py tests/python/unit/test_expert_policy.py tests/python/unit/test_utils_config.py tests/python/integration/test_stability_e2e.py
git commit -m "feat: define phase-specific expert policy config"
```

---

### Task 2: Propagate serving phase and define mixed-batch behavior

**Files:**
- Modify: `moe_infinity/serving/model_runner.py:92-154`
- Modify: `moe_infinity/serving/engine.py:726-751`
- Modify: `tests/python/serving/test_model_runner.py`
- Modify: `tests/python/serving/test_engine.py`
- Modify: `tests/python/serving/test_batch.py`

- [ ] **Step 1: Write failing phase-scope, enabled-order, and disabled-compatibility tests**

```python
# append to tests/python/serving/test_model_runner.py
def test_execute_scopes_homogeneous_phase(monkeypatch) -> None:
    observed = []
    model = MockModel(vocab_size=16, rank3_logits=False)
    original = model.forward

    def forward(*args, **kwargs):
        from moe_infinity.memory.expert_policy import current_expert_phase
        observed.append(current_expert_phase())
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    runner = ModelRunner(model, MockOffloadEngine())
    batch = _make_batch()
    batch.is_prefill = [False, False]
    _ = runner.execute(batch)
    from moe_infinity.memory.expert_policy import ExpertPhase
    assert observed == [ExpertPhase.DECODE]
```

```python
# append to tests/python/serving/test_engine.py
def make_mixed_batch(prefill_tokens, decode_tokens) -> BatchMetadata:
    tokens = [*prefill_tokens, *decode_tokens]
    return BatchMetadata(
        seq_ids=[1, 2],
        input_token_ids=tokens,
        seq_lengths=[len(prefill_tokens), len(decode_tokens)],
        context_lengths=[0, 8],
        is_prefill=[True, False],
        block_tables=[[0, 1], [2, 3]],
        token_offsets=[0, len(prefill_tokens), len(tokens)],
        sampling_params=[SamplingParams(), SamplingParams()],
    )


def test_enabled_policy_executes_mixed_decode_then_prefill() -> None:
    engine = _make_engine()
    calls = []

    def execute(batch):
        calls.append(list(batch.is_prefill))
        base = 100 if all(batch.is_prefill) else 200
        return torch.arange(batch.total_tokens).unsqueeze(1) + base

    engine.model_runner.execute = execute
    engine.config["phase_specific_expert_policy"] = True
    batch = make_mixed_batch(prefill_tokens=[11, 12], decode_tokens=[21])
    output = engine._execute_batch(batch)
    assert calls == [[False], [True]]
    assert output.squeeze(1).tolist() == [100, 101, 200]


def test_disabled_policy_keeps_nonpaged_mixed_combined() -> None:
    engine = _make_engine()
    calls = []
    engine.config["phase_specific_expert_policy"] = False
    engine.model_runner._get_paged_attention_classes = lambda: []
    engine.model_runner.execute = lambda batch: (
        calls.append(list(batch.is_prefill))
        or torch.zeros((batch.total_tokens, 1))
    )
    batch = make_mixed_batch(prefill_tokens=[11, 12], decode_tokens=[21])
    _ = engine._execute_batch(batch)
    assert calls == [[True, False]]


def test_disabled_policy_keeps_paged_prefill_then_decode() -> None:
    engine = _make_engine()
    calls = []
    engine.config["phase_specific_expert_policy"] = False
    engine.model_runner._get_paged_attention_classes = lambda: [object]
    engine.model_runner.execute = lambda batch: (
        calls.append(list(batch.is_prefill))
        or torch.zeros((batch.total_tokens, 1))
    )
    batch = make_mixed_batch(prefill_tokens=[11, 12], decode_tokens=[21])
    _ = engine._execute_batch(batch)
    assert calls == [[True], [False]]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/python/serving/test_model_runner.py tests/python/serving/test_engine.py tests/python/serving/test_batch.py -q`

Expected: phase observation is `MIXED`; enabled decode-first behavior is absent; disabled tests document the current non-paged and paged behavior before refactoring.

- [ ] **Step 3: Scope model forwards and gate decode-first splitting**

In `ModelRunner.execute`, derive the phase without inspecting tensor shapes:

```python
def _expert_phase(self, batch: BatchMetadata) -> ExpertPhase:
    if batch.is_prefill and all(batch.is_prefill):
        return ExpertPhase.PREFILL
    if batch.is_prefill and not any(batch.is_prefill):
        return ExpertPhase.DECODE
    return ExpertPhase.MIXED
```

Wrap only the actual `forward_fn(**forward_kwargs)` region in `expert_phase_scope(self._expert_phase(batch))`; keep paged-attention set/clear inside that scope so exceptions restore both contexts. In `ServingEngine._execute_batch`, preserve the existing disabled path before adding the enabled path:

```python
phase_policy_enabled = bool(
    self.config.get("phase_specific_expert_policy", False)
)

if not phase_policy_enabled:
    # Exact pre-feature behavior: a non-paged mixed batch stays combined;
    # a paged mixed batch is split prefill first, then decode.
    if not uses_paged or not (has_prefill and has_decode):
        return self.model_runner.execute(batch)
    split = split_prefill_decode_batch(batch)
    prefill_logits = (
        self.model_runner.execute(split.prefill_batch)
        if split.prefill_batch is not None
        else None
    )
    decode_logits = (
        self.model_runner.execute(split.decode_batch)
        if split.decode_batch is not None
        else None
    )
    return split.recombine_outputs(prefill_logits, decode_logits)

if not (has_prefill and has_decode):
    return self.model_runner.execute(batch)
split = split_prefill_decode_batch(batch)
decode_logits = (
    self.model_runner.execute(split.decode_batch)
    if split.decode_batch is not None
    else None
)
prefill_logits = (
    self.model_runner.execute(split.prefill_batch)
    if split.prefill_batch is not None
    else None
)
return split.recombine_outputs(prefill_logits, decode_logits)
```

Decode-first execution is therefore an enabled-policy behavior only. Keep `split_prefill_decode_batch`'s original-index recombination unchanged. Do not hoist common-looking split code across the feature gate, because the order and the non-paged combined forward are compatibility behavior.

- [ ] **Step 4: Verify serving tests**

Run: `pytest tests/python/serving/test_model_runner.py tests/python/serving/test_engine.py tests/python/serving/test_batch.py tests/python/integration/test_flashinfer_e2e.py -q`

Expected: all pass; enabled mode is decode-first, disabled non-paged mode remains one combined call, and disabled paged mode remains prefill-first.

- [ ] **Step 5: Commit phase propagation**

```bash
git add moe_infinity/serving/model_runner.py moe_infinity/serving/engine.py tests/python/serving/test_model_runner.py tests/python/serving/test_engine.py tests/python/serving/test_batch.py
git commit -m "feat: propagate expert phase from serving batches"
```

---

### Task 3: Make Python prefetch policy phase-specific

**Files:**
- Modify: `moe_infinity/memory/expert_prefetcher.py:15-312`
- Modify: `moe_infinity/distributed/expert_executor.py:97-291`
- Modify: `moe_infinity/runtime/model_offload.py:1089-1163`
- Create: `tests/python/unit/test_phase_expert_prefetcher.py`
- Modify: `tests/python/ops/test_expert_dispatch.py`
- Modify: `tests/python/dflash/test_route_ahead_wire.py`

- [ ] **Step 1: Write failing deterministic prefetch-selection tests**

```python
# tests/python/unit/test_phase_expert_prefetcher.py
import numpy as np

from moe_infinity.memory.expert_policy import ExpertPhase, PhasePolicySettings
from moe_infinity.memory.expert_prefetcher import ExpertPrefetcher


def settings() -> PhasePolicySettings:
    return PhasePolicySettings(True, "transient_on_pressure", "cache", 0, 2, 2, 1, 1.0, 4.0, 8)


def test_prefill_top_k_zero_issues_no_predictive_prefetch() -> None:
    prefetcher = object.__new__(ExpertPrefetcher)
    prefetcher.num_layers = 4
    prefetcher.num_experts = 4
    prefetcher.phase_policy = settings()
    prefetcher.archer_engine = RecordingEngine()
    prefetcher.expert_tensor_map = {(1, i): 10 + i for i in range(4)}
    prefetcher._last_speculative_prediction = {}
    prefetcher.speculative_prefetch(0, np.array([[1, 4, 3, 2]]), phase=ExpertPhase.PREFILL)
    assert prefetcher.archer_engine.calls == []


def test_decode_uses_decode_top_k_and_priority() -> None:
    prefetcher = make_prefetcher(settings())
    prefetcher.speculative_prefetch(0, np.array([[1, 4, 3, 2]]), phase=ExpertPhase.DECODE)
    assert prefetcher.archer_engine.calls == [([11, 12], 1, int(ExpertPhase.DECODE))]
```

`RecordingEngine.prefetch_tensors(tensor_ids, priority, phase)` must store sorted integer arguments. Also add an executor test asserting `enqueue_expert(layer, expert, gpu, remote, int(phase))` and route-ahead prefetch receives `DECODE`.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/python/unit/test_phase_expert_prefetcher.py tests/python/ops/test_expert_dispatch.py tests/python/dflash/test_route_ahead_wire.py -q`

Expected: `speculative_prefetch` rejects `phase=`, and dispatcher calls have no phase argument.

- [ ] **Step 3: Implement explicit phase arguments**

Add `phase_policy: PhasePolicySettings` to `ExpertPrefetcher`, defaulting to an inert settings object for direct test construction. Change signatures consistently:

```python
def prefetch_experts_list(
    self,
    layer_id: int,
    expert_list: List[int],
    priority: Optional[int] = None,
    phase: ExpertPhase = ExpertPhase.MIXED,
): ...

def speculative_prefetch(
    self,
    layer_idx: int,
    router_logits: Optional[Any] = None,
    *,
    expert_ids: Optional[List[int]] = None,
    prefetch_layer_id: Optional[int] = None,
    phase: ExpertPhase = ExpertPhase.MIXED,
): ...

def correct_prefetch(
    self,
    layer_idx: int,
    actual_expert_ids: List[int],
    predicted_expert_ids: Optional[Set[int]] = None,
    *,
    phase: ExpertPhase = ExpertPhase.MIXED,
): ...
```

When enabled, resolve `MIXED` to decode for admission but return without predictive prefetch for an actual mixed phase. Select top-k and priority from the phase settings. Keep explicit route-ahead `expert_ids` exact and tag it decode; do not apply top-k to route-ahead. Store predictions by phase (`dict[ExpertPhase, set[int]]`) so prefill correction cannot consume decode state.

In `DistributedExpertExecutor.dispatch_local`, read `phase = current_expert_phase()` once, pass it to every `enqueue_expert`, speculative prefetch, and `_pending_prefetch`; in `wait_dispatch_local`, pass the same captured phase to correction. The distributed RPC call gets the same trailing integer. No router-mask or router-weight values change.

In `model_offload.py`, construct `PhasePolicySettings` from `ArcherConfig`, assign it to the prefetcher, and call the native `expert_dispatcher.configure_phase_policy(...)` only when the master flag is true. This conditional call is the compatibility seam for disabled policy and old behavior.

- [ ] **Step 4: Verify Python dispatch, prefetch, and DFlash behavior**

Run: `pytest tests/python/unit/test_phase_expert_prefetcher.py tests/python/ops/test_expert_dispatch.py tests/python/dflash/test_speculative_prefetch.py tests/python/dflash/test_route_ahead_wire.py tests/python/dflash/test_prefetch_route.py -q`

Expected: all pass; route-ahead still prefetches the exact union and ordinary decode uses configured top-k/priority.

- [ ] **Step 5: Commit Python policy wiring**

```bash
git add moe_infinity/memory/expert_prefetcher.py moe_infinity/distributed/expert_executor.py moe_infinity/runtime/model_offload.py tests/python/unit/test_phase_expert_prefetcher.py tests/python/ops/test_expert_dispatch.py tests/python/dflash/test_route_ahead_wire.py
git commit -m "feat: apply phase-specific expert prefetch policy"
```

---

### Task 4: Add a pure native policy core with deterministic tests

**Files:**
- Create: `core/prefetch/expert_policy.h`
- Modify: `CMakeLists.txt:41-42`
- Create: `tests/cpp/unit/prefetch/CMakeLists.txt`
- Create: `tests/cpp/unit/prefetch/test_expert_policy.cpp`

- [ ] **Step 1: Write the failing GoogleTest policy specification**

```cpp
// tests/cpp/unit/prefetch/test_expert_policy.cpp
#include <gtest/gtest.h>
#include "prefetch/expert_policy.h"

TEST(ExpertPolicy, DecodeWeightsDecodeReuse) {
  PhasePolicyConfig cfg{true, AdmissionMode::TRANSIENT_ON_PRESSURE,
                        AdmissionMode::CACHE, 1.0, 4.0, 8};
  ExpertPolicyMetadata prefill_hot{10, 0, 10, 0};
  ExpertPolicyMetadata decode_hot{0, 3, 0, 9};
  EXPECT_LT(VictimUtility(prefill_hot, ExpertPhase::DECODE, cfg),
            VictimUtility(decode_hot, ExpertPhase::DECODE, cfg));
}

TEST(ExpertPolicy, StableVictimTieBreakUsesLayerThenExpert) {
  VictimCandidate a{2, 7, 1.0, 4};
  VictimCandidate b{3, 0, 1.0, 4};
  EXPECT_TRUE(VictimLess(a, b));
}

TEST(ExpertPolicy, BypassPromotionNeverPassesDemand) {
  EXPECT_EQ(ServiceClass(0, 99, 8), 0);
  EXPECT_EQ(ServiceClass(2, 8, 8), 1);
  EXPECT_EQ(ServiceClass(1, 0, 8), 1);
}
```

- [ ] **Step 2: Configure and run the test to verify RED**

Run: `cmake -S . -B build -DBUILD_TESTING=ON && cmake --build build --target test_expert_policy -j && ctest --test-dir build -R ExpertPolicyTests --output-on-failure`

Expected: compile fails because `prefetch/expert_policy.h` is absent.

- [ ] **Step 3: Implement header-only native types and helpers**

```cpp
// core/prefetch/expert_policy.h
#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>

enum class ExpertPhase : std::uint8_t { PREFILL = 0, DECODE = 1, MIXED = 2 };
enum class AdmissionMode : std::uint8_t {
  CACHE = 0,
  TRANSIENT_ON_PRESSURE = 1,
};

struct PhasePolicyConfig {
  bool enabled = false;
  AdmissionMode prefill_admission = AdmissionMode::TRANSIENT_ON_PRESSURE;
  AdmissionMode decode_admission = AdmissionMode::CACHE;
  double prefill_eviction_weight = 1.0;
  double decode_eviction_weight = 4.0;
  std::uint32_t starvation_limit = 8;
};

struct ExpertPolicyMetadata {
  std::uint64_t prefill_accesses = 0;
  std::uint64_t decode_accesses = 0;
  std::uint64_t last_prefill_sequence = 0;
  std::uint64_t last_decode_sequence = 0;
};

struct VictimCandidate {
  std::int64_t layer_id;
  std::int64_t expert_id;
  double utility;
  std::uint64_t last_sequence;
};

inline ExpertPhase EffectivePhase(ExpertPhase phase) {
  return phase == ExpertPhase::MIXED ? ExpertPhase::DECODE : phase;
}

inline double VictimUtility(const ExpertPolicyMetadata& m, ExpertPhase phase,
                            const PhasePolicyConfig& cfg) {
  const auto active = EffectivePhase(phase);
  const double prefill_weight = active == ExpertPhase::PREFILL
                                    ? cfg.prefill_eviction_weight : 1.0;
  const double decode_weight = active == ExpertPhase::DECODE
                                   ? cfg.decode_eviction_weight : 1.0;
  return prefill_weight * m.prefill_accesses +
         decode_weight * m.decode_accesses;
}

inline bool VictimLess(const VictimCandidate& a, const VictimCandidate& b) {
  return std::tie(a.utility, a.last_sequence, a.layer_id, a.expert_id) <
         std::tie(b.utility, b.last_sequence, b.layer_id, b.expert_id);
}

inline std::uint32_t ServiceClass(std::uint32_t priority,
                                  std::uint32_t bypasses,
                                  std::uint32_t limit) {
  if (priority == 0) return 0;
  return bypasses >= limit ? 1 : priority;
}
```

After `add_subdirectory(core)` in the root `CMakeLists.txt`, add `include(CTest)` and conditionally `add_subdirectory(tests/cpp/unit/prefetch)` when `BUILD_TESTING`. The test CMake file must use `find_package(GTest REQUIRED)`, build `test_expert_policy`, link `archer_core`, `GTest::gtest_main`, and `pthread`, and register `ExpertPolicyTests`:

```cmake
find_package(GTest REQUIRED)

add_executable(test_expert_policy test_expert_policy.cpp)
target_link_libraries(test_expert_policy PRIVATE archer_core GTest::gtest_main pthread)

add_test(NAME ExpertPolicyTests COMMAND test_expert_policy)
```

- [ ] **Step 4: Run the deterministic native tests**

Run: `cmake --build build --target test_expert_policy -j && ctest --test-dir build -R ExpertPolicyTests --output-on-failure`

Expected: `ExpertPolicyTests` passes.

- [ ] **Step 5: Commit the pure native policy**

```bash
git add CMakeLists.txt core/prefetch/expert_policy.h tests/cpp/unit/prefetch/CMakeLists.txt tests/cpp/unit/prefetch/test_expert_policy.cpp
git commit -m "feat: add deterministic native expert policy core"
```

---

### Task 5: Apply phase-aware shared-cache admission, eviction, and starvation protection

**Files:**
- Create: `core/prefetch/expert_residency.h`
- Create: `core/prefetch/expert_residency.cpp`
- Modify: `setup.py:177-203`
- Modify: `core/CMakeLists.txt:1-44`
- Modify: `core/model/model_topology.h:40-72`
- Modify: `core/prefetch/task_scheduler.h:24-121`
- Modify: `core/prefetch/task_scheduler.cpp:47-597`
- Modify: `core/parallel/expert_dispatcher.h:40-194`
- Modify: `core/parallel/expert_dispatcher.cpp:180-586`
- Modify: `core/prefetch/archer_prefetch_handle.h:13-80`
- Modify: `core/prefetch/archer_prefetch_handle.cpp:20-408`
- Modify: `core/python/py_archer_prefetch.cpp:18-123`
- Modify: `tests/cpp/unit/prefetch/CMakeLists.txt`
- Create: `tests/cpp/unit/prefetch/expert_residency_test_fixture.h`
- Create: `tests/cpp/unit/prefetch/test_expert_residency.cpp`
- Modify: `tests/cpp/unit/prefetch/test_expert_policy.cpp`
- Create: `tests/python/unit/test_native_phase_policy_wire.py`
- Create: `tests/python/unit/test_setup_sources.py`
- Create: `tests/python/unit/test_native_capacity_lifecycle.py`

- [ ] **Step 1: Write failing source-registration and capacity-order tests**

```python
# tests/python/unit/test_setup_sources.py
import ast
from pathlib import Path


def test_store_sources_link_expert_residency_implementation() -> None:
    setup_path = Path(__file__).resolve().parents[3] / "setup.py"
    module = ast.parse(setup_path.read_text())
    store_sources = None
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "_STORE_SOURCES"
            for target in statement.targets
        ):
            store_sources = ast.literal_eval(statement.value)
            break
    assert store_sources is not None
    assert "core/prefetch/expert_residency.cpp" in store_sources
    assert store_sources.count("core/prefetch/expert_residency.cpp") == 1
```

Run: `pytest tests/python/unit/test_setup_sources.py -q`

Expected: FAIL because `_STORE_SOURCES` does not include `core/prefetch/expert_residency.cpp`.

In the same RED step, add an explicit native lifecycle/order regression:

```python
# tests/python/unit/test_native_capacity_lifecycle.py
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "core"
SOURCE = (CORE / "prefetch/archer_prefetch_handle.cpp").read_text()


def _between(start: str, end: str) -> str:
    start_index = SOURCE.index(start)
    end_index = SOURCE.index(end, start_index + len(start))
    return SOURCE[start_index:end_index]


def test_constructor_leaves_expert_capacity_unconfigured() -> None:
    constructor = _between(
        "ArcherPrefetchHandle::ArcherPrefetchHandle(",
        "ArcherPrefetchHandle::~ArcherPrefetchHandle()",
    )
    assert "ConfigureCapacity" not in constructor
    assert "GetSparseCacheLimit" not in constructor


def test_topology_setters_configure_only_after_initialization() -> None:
    legacy = _between(
        "void ArcherPrefetchHandle::SetTopology(",
        "void ArcherPrefetchHandle::SetTopologyV2(",
    )
    v2 = _between(
        "void ArcherPrefetchHandle::SetTopologyV2(",
        "ArcherPrefetchHandle::GetTopologySnapshot()",
    )
    assert legacy.index("InitializeTopology(topology)") < legacy.index(
        "ConfigureExpertCapacityAfterTopology()"
    )
    assert v2.index("InitializeTopologyV2(topology)") < v2.index(
        "ConfigureExpertCapacityAfterTopology()"
    )


def test_capacity_helper_reads_sparse_limit_before_update() -> None:
    helper = _between(
        "void ArcherPrefetchHandle::ConfigureExpertCapacityAfterTopology()",
        "void ArcherPrefetchHandle::SetTopology(",
    )
    assert helper.index("GetSparseCacheLimit(device)") < helper.index(
        "ConfigureCapacity(gpu_id, bytes)"
    )


def test_only_topology_completion_installs_production_capacity() -> None:
    callsites = {
        str(path.relative_to(CORE)): text.count("->ConfigureCapacity(")
        for path in CORE.rglob("*.cpp")
        if (text := path.read_text()).count("->ConfigureCapacity(")
    }
    assert callsites == {"prefetch/archer_prefetch_handle.cpp": 1}
```

Run: `pytest tests/python/unit/test_setup_sources.py tests/python/unit/test_native_capacity_lifecycle.py -q`

Expected: both tests fail: the residency source is not registered, and the deferred capacity helper/calls do not exist. The source-order test intentionally protects the setup sequencing while the real-manager tests below protect runtime semantics.

- [ ] **Step 2: Write exhaustive eligibility and randomized-order tests**

Define one production predicate in `core/prefetch/expert_residency.h` and test every rejection independently. The test state mirrors the fields read from a real `Node` plus manager-owned lease/protection state:

```cpp
#include <algorithm>
#include <random>

#include "prefetch/expert_residency.h"

TEST(ExpertResidency, EligibilityAcceptsOnlyIdleUnprotectedResident) {
  EvictionState eligible{/*is_cuda=*/true, /*is_prefetching=*/false,
                         NodeExecState::IDLE, /*pending_dispatches=*/0,
                         /*is_overflow=*/false, /*lease_count=*/0,
                         /*protected_candidate=*/false};
  EXPECT_TRUE(IsEvictionEligible(eligible));

  auto state = eligible;
  state.is_cuda = false;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible; state.is_prefetching = true;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible; state.exec_state = NodeExecState::FETCHING;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible; state.exec_state = NodeExecState::EXECUTING;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible; state.pending_dispatches = 1;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible; state.is_overflow = true;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible; state.lease_count = 1;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible; state.protected_candidate = true;
  EXPECT_FALSE(IsEvictionEligible(state));
}

TEST(ExpertResidency, RandomizedCandidateOrderAlwaysSelectsStableVictim) {
  const std::vector<VictimCandidate> original{
      {4, 2, 1.0, 9}, {1, 3, 1.0, 9}, {2, 1, 2.0, 1}, {0, 7, 3.0, 2}};
  for (std::uint32_t seed = 0; seed < 100; ++seed) {
    auto candidates = original;
    std::shuffle(candidates.begin(), candidates.end(), std::mt19937(seed));
    const auto victim = SelectStableVictim(candidates);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->layer_id, 1);
    EXPECT_EQ(victim->expert_id, 3);
  }
}
```

Retain Task 4's exact bypass assertions (`7 -> class 2`, `8 -> class 1`, and demand `0 -> class 0`). In `tests/python/unit/test_native_phase_policy_wire.py`, use one recording manager fake injected into fake dispatcher and prefetch clients; issue one demand admission and one prefetch admission, then assert the same fake receives both calls and produces one non-duplicated snapshot. The fake records calls only and does not implement eligibility, scoring, or accounting policy.

- [ ] **Step 3: Define the real-manager fixture and transaction tests**

Use the real `ExpertResidencyManager`, real `Node` objects, and an injected transfer implementation that records host moves without requiring a GPU. Define the production seam and fixture:

```cpp
// core/prefetch/expert_residency.h
class ResidencyTransferOps {
 public:
  virtual ~ResidencyTransferOps() = default;
  virtual bool MoveToHost(const NodePtr& node) = 0;
};

enum class AdmissionSource : std::uint8_t { DEMAND = 0, PREFETCH = 1 };
enum class AdmissionOutcome : std::uint8_t {
  ADMIT = 0, ALREADY_RESIDENT = 1, TRANSIENT = 2, REJECTED = 3
};
```

```cpp
// tests/cpp/unit/prefetch/expert_residency_test_fixture.h
#pragma once

#include <gtest/gtest.h>
#include <atomic>
#include <memory>
#include <vector>

#include "prefetch/expert_residency.h"

class RecordingTransferOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override {
    moved_ids.push_back(node->id);
    node->device = node->default_host;
    return true;
  }
  std::vector<std::size_t> moved_ids;
};

class ExpertResidencyManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    transfer_ops = std::make_shared<RecordingTransferOps>();
    manager = std::make_shared<ExpertResidencyManager>(transfer_ops);
  }

  NodePtr MakeNode(std::size_t id, std::int64_t bytes) {
    auto node = std::make_shared<Node>();
    node->id = id;
    node->corr_id = id;
    node->byte_size = bytes;
    node->device = torch::Device(torch::kCPU);
    node->default_device = torch::Device(torch::kCUDA, 0);
    node->default_host = torch::Device(torch::kCPU);
    return node;
  }

  ResidencyTicket Begin(const NodePtr& node, AdmissionSource source) {
    return manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                   AdmissionMode::CACHE, source);
  }

  void Admit(const NodePtr& node, AdmissionSource source) {
    if (!manager->IsCapacityConfigured(0)) {
      ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
    }
    auto ticket = Begin(node, source);
    ASSERT_TRUE(ticket.valid);
    ASSERT_EQ(ticket.outcome, AdmissionOutcome::ADMIT);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(ticket));
  }

  std::shared_ptr<RecordingTransferOps> transfer_ops;
  std::shared_ptr<ExpertResidencyManager> manager;
};
```

Create `tests/cpp/unit/prefetch/test_expert_residency.cpp` with the lifecycle/order/reconfiguration cases below followed by the six existing transaction cases. These lifecycle tests are mandatory because the production handle and dispatcher can be constructed before topology is installed:

```cpp
#include "expert_residency_test_fixture.h"
#include <thread>

TEST_F(ExpertResidencyManagerTest, AdmissionIsRejectedUntilDeviceConfigured) {
  auto rejected = Begin(MakeNode(1, 40), AdmissionSource::DEMAND);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.outcome, AdmissionOutcome::REJECTED);
  EXPECT_EQ(manager->ResidentBytes(0), 0);
  EXPECT_EQ(manager->ResidentCount(0), 0);
  EXPECT_EQ(manager->Snapshot().at("decode_admissions"), 0);

  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto allowed = Begin(MakeNode(2, 40), AdmissionSource::DEMAND);
  EXPECT_TRUE(allowed.valid);
  EXPECT_EQ(allowed.outcome, AdmissionOutcome::ADMIT);
  EXPECT_TRUE(manager->AbortAdmission(allowed));
}

TEST_F(ExpertResidencyManagerTest, DevicesConfigureIndependently) {
  ASSERT_TRUE(manager->ConfigureCapacity(1, 200));
  auto gpu0 = manager->BeginAdmission(
      MakeNode(1, 40), 0, ExpertPhase::DECODE,
      AdmissionMode::CACHE, AdmissionSource::DEMAND);
  auto gpu1 = manager->BeginAdmission(
      MakeNode(2, 40), 1, ExpertPhase::DECODE,
      AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_FALSE(gpu0.valid);
  EXPECT_TRUE(gpu1.valid);
  EXPECT_TRUE(manager->AbortAdmission(gpu1));
}

TEST_F(ExpertResidencyManagerTest, CapacityUpdateIsAtomicAndPerDevice) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto resident = MakeNode(1, 60);
  Admit(resident, AdmissionSource::DEMAND);

  EXPECT_TRUE(manager->ConfigureCapacity(0, 80));
  EXPECT_EQ(manager->CapacityBytes(0), 80);
  EXPECT_FALSE(manager->ConfigureCapacity(0, 50));
  EXPECT_EQ(manager->CapacityBytes(0), 80);
  EXPECT_FALSE(manager->ConfigureCapacity(0, -1));
  EXPECT_EQ(manager->CapacityBytes(0), 80);

  auto pressure = Begin(MakeNode(2, 30), AdmissionSource::DEMAND);
  ASSERT_TRUE(pressure.valid);
  EXPECT_EQ(pressure.reserved_victim, resident);
  EXPECT_FALSE(manager->ConfigureCapacity(0, 120));
  EXPECT_EQ(manager->CapacityBytes(0), 80);
  EXPECT_TRUE(manager->AbortAdmission(pressure));
  EXPECT_TRUE(manager->ConfigureCapacity(0, 120));
  EXPECT_EQ(manager->CapacityBytes(0), 120);
}

TEST_F(ExpertResidencyManagerTest, CapacityEvictsReservedVictimBeforeCommit) {
  auto first = MakeNode(1, 60);
  Admit(first, AdmissionSource::DEMAND);
  auto second = MakeNode(2, 60);
  auto ticket = Begin(second, AdmissionSource::DEMAND);
  ASSERT_TRUE(ticket.valid);
  ASSERT_EQ(ticket.reserved_victim, first);
  EXPECT_TRUE(manager->EvictReserved(ticket));
  second->device = second->default_device;
  EXPECT_TRUE(manager->CommitAdmission(ticket));
  EXPECT_EQ(manager->ResidentBytes(0), 60);
  EXPECT_EQ(manager->ResidentCount(0), 1);
  ASSERT_EQ(transfer_ops->moved_ids.size(), 1);
  EXPECT_EQ(transfer_ops->moved_ids[0], first->id);
}

TEST_F(ExpertResidencyManagerTest, DuplicateAdmissionIsNoOp) {
  auto node = MakeNode(1, 40);
  Admit(node, AdmissionSource::DEMAND);
  auto duplicate = Begin(node, AdmissionSource::PREFETCH);
  EXPECT_TRUE(duplicate.valid);
  EXPECT_EQ(duplicate.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_FALSE(manager->CommitAdmission(duplicate));
  EXPECT_EQ(manager->ResidentBytes(0), 40);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest, VictimReservationAllowsOnlyOneRacingTicket) {
  auto victim = MakeNode(1, 100);
  Admit(victim, AdmissionSource::DEMAND);
  std::atomic<bool> go{false};
  ResidencyTicket tickets[2];
  std::thread threads[2];
  for (int i = 0; i < 2; ++i) {
    threads[i] = std::thread([&, i] {
      while (!go.load(std::memory_order_acquire)) {}
      tickets[i] = Begin(MakeNode(10 + i, 100), AdmissionSource::DEMAND);
    });
  }
  go.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();
  const int reservations =
      int(tickets[0].reserved_victim != nullptr) +
      int(tickets[1].reserved_victim != nullptr);
  EXPECT_EQ(reservations, 1);
  for (const auto& ticket : tickets) {
    if (ticket.valid) EXPECT_TRUE(manager->AbortAdmission(ticket));
  }
}

TEST_F(ExpertResidencyManagerTest, CommitAndAbortAreIdempotent) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto committed = MakeNode(1, 40);
  auto commit_ticket = Begin(committed, AdmissionSource::DEMAND);
  committed->device = committed->default_device;
  EXPECT_TRUE(manager->CommitAdmission(commit_ticket));
  EXPECT_FALSE(manager->CommitAdmission(commit_ticket));

  auto aborted = MakeNode(2, 40);
  auto abort_ticket = Begin(aborted, AdmissionSource::PREFETCH);
  EXPECT_TRUE(manager->AbortAdmission(abort_ticket));
  EXPECT_FALSE(manager->AbortAdmission(abort_ticket));
  EXPECT_EQ(manager->ResidentBytes(0), 40);
}

TEST_F(ExpertResidencyManagerTest, LeaseReleaseRestoresEvictionEligibility) {
  auto resident = MakeNode(1, 100);
  Admit(resident, AdmissionSource::DEMAND);
  const auto lease = manager->AcquireLease(resident, LeaseKind::DEMAND);
  auto blocked = Begin(MakeNode(2, 100), AdmissionSource::DEMAND);
  EXPECT_FALSE(blocked.valid);
  EXPECT_TRUE(manager->ReleaseLease(lease));
  EXPECT_FALSE(manager->ReleaseLease(lease));
  auto allowed = Begin(MakeNode(3, 100), AdmissionSource::DEMAND);
  EXPECT_TRUE(allowed.valid);
  EXPECT_EQ(allowed.reserved_victim, resident);
  EXPECT_TRUE(manager->AbortAdmission(allowed));
}

TEST_F(ExpertResidencyManagerTest, DemandAndPrefetchShareByteAccounting) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  ExpertResidencyClient dispatcher_client(manager, AdmissionSource::DEMAND);
  ExpertResidencyClient prefetch_client(manager, AdmissionSource::PREFETCH);
  auto demand = MakeNode(1, 40);
  auto demand_ticket = dispatcher_client.BeginAdmission(
      demand, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  demand->device = demand->default_device;
  ASSERT_TRUE(manager->CommitAdmission(demand_ticket));
  auto prefetched = MakeNode(2, 50);
  auto prefetch_ticket = prefetch_client.BeginAdmission(
      prefetched, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  prefetched->device = prefetched->default_device;
  ASSERT_TRUE(manager->CommitAdmission(prefetch_ticket));
  const auto stats = manager->Snapshot();
  EXPECT_EQ(manager->ResidentBytes(0), 90);
  EXPECT_EQ(manager->ResidentCount(0), 2);
  EXPECT_EQ(stats.at("resident_bytes"), 90);
  EXPECT_EQ(stats.at("resident_experts"), 2);
  EXPECT_EQ(stats.at("decode_admissions"), 1);
  EXPECT_EQ(stats.at("decode_prefetch_completed"), 1);
}
```

- [ ] **Step 4: Define CMake linkage and run all new tests RED**

Extend `tests/cpp/unit/prefetch/CMakeLists.txt` so the transaction test links the real manager through `archer_core`:

```cmake
add_executable(test_expert_residency test_expert_residency.cpp)
target_link_libraries(
  test_expert_residency PRIVATE archer_core GTest::gtest_main pthread
)
add_test(NAME ExpertResidencyTests COMMAND test_expert_residency)
```

Run:

```bash
pytest tests/python/unit/test_setup_sources.py -q
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_expert_policy test_expert_residency -j
ctest --test-dir build -R 'Expert(Policy|Residency)Tests' --output-on-failure
```

Expected: source-registration test fails until `setup.py` changes; native build fails until the real manager, transfer seam, outcomes, source accounting, and idempotent Boolean APIs exist.

- [ ] **Step 5: Register the residency source in both build graphs**

Append the implementation once to `_STORE_SOURCES` adjacent to the existing prefetch scheduler source:

```python
_STORE_SOURCES = [
    # utils and model entries above are unchanged
    # prefetch
    "core/prefetch/archer_prefetch_handle.cpp",
    "core/prefetch/task_scheduler.cpp",
    "core/prefetch/expert_residency.cpp",
    "core/prefetch/task_thread.cpp",
    # memory, parallel, aio, and base entries below are unchanged
]
```

Also add `core/prefetch/expert_residency.cpp` to `ARCHER_CORE_CXX_SOURCES`; `setup.py` is required because editable installation builds `_store` from `_STORE_SOURCES` rather than linking the CMake `archer_core` target.

- [ ] **Step 6: Run native tests and verify RED**

Run: `cmake --build build --target test_expert_policy test_expert_residency -j && ctest --test-dir build -R 'Expert(Policy|Residency)Tests' --output-on-failure`

Expected: compile fails because `expert_residency.h`, `EvictionState`, `IsEvictionEligible`, and `SelectStableVictim` do not exist.

- [ ] **Step 7: Implement the single synchronized residency authority**

Add `ExpertPolicyMetadata policy_metadata;` to `Node`; do not add phase to `corr_id`, tensor IDs, or topology keys. Create `ExpertResidencyManager` as the sole owner of per-GPU resident maps and bytes. Its public transaction surface is:

```cpp
enum class LeaseKind : std::uint8_t { DEMAND = 0, PREFETCH = 1, TRANSFER = 2 };

struct ResidencyTicket {
  std::uint64_t id = 0;
  NodePtr incoming;
  NodePtr reserved_victim;
  int gpu_id = -1;
  ExpertPhase phase = ExpertPhase::MIXED;
  AdmissionSource source = AdmissionSource::DEMAND;
  AdmissionOutcome outcome = AdmissionOutcome::REJECTED;
  bool transient = false;
  bool valid = false;
};

struct ResidencyEntry {
  NodePtr node;
  std::int64_t bytes = 0;
  std::uint32_t lease_count = 0;
};

struct LeaseRecord {
  std::uint64_t id = 0;
  NodePtr node;
  LeaseKind kind = LeaseKind::DEMAND;
};

using ExpertPolicyStats =
    std::unordered_map<std::string, std::int64_t>;

class ExpertResidencyManager {
 public:
  explicit ExpertResidencyManager(
      std::shared_ptr<ResidencyTransferOps> transfer_ops);
  bool ConfigureCapacity(int gpu_id, std::int64_t capacity_bytes);
  bool IsCapacityConfigured(int gpu_id) const;
  std::int64_t CapacityBytes(int gpu_id) const;
  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode,
                                 AdmissionSource source);
  bool EvictReserved(const ResidencyTicket& ticket);
  bool CommitAdmission(const ResidencyTicket& ticket);
  bool AbortAdmission(const ResidencyTicket& ticket);
  std::uint64_t AcquireLease(const NodePtr& node, LeaseKind kind);
  bool ReleaseLease(std::uint64_t lease_id);
  void ReplaceProtectedCandidates(const NodePtrList& candidates);
  void RecordAccess(const NodePtr& node, ExpertPhase phase, bool hit);
  ExpertPolicyStats Snapshot() const;
  std::int64_t ResidentBytes(int gpu_id) const;
  std::size_t ResidentCount(int gpu_id) const;

 private:
  mutable std::mutex mutex_;
  std::vector<std::map<std::uint64_t, ResidencyEntry>> residents_;
  std::unordered_map<std::uint64_t, LeaseRecord> leases_;
  std::unordered_set<NodePtr> protected_candidates_;
  std::unordered_map<std::uint64_t, ResidencyTicket> pending_tickets_;
  std::vector<std::optional<std::int64_t>> capacity_bytes_;
};

class ExpertResidencyClient {
 public:
  ExpertResidencyClient(std::shared_ptr<ExpertResidencyManager> manager,
                        AdmissionSource source)
      : manager_(std::move(manager)), source_(source) {}
  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode) {
    return manager_->BeginAdmission(incoming, gpu_id, phase, mode, source_);
  }
  std::shared_ptr<ExpertResidencyManager> manager() const { return manager_; }

 private:
  std::shared_ptr<ExpertResidencyManager> manager_;
  AdmissionSource source_;
};
```

The constructor allocates no inferred capacity and leaves every device unconfigured. `ConfigureCapacity` holds `mutex_`, rejects a negative device ID, negative byte count, any update while that device has pending tickets, and any value below the manager-owned resident-byte total for that device; on rejection it leaves the previous optional value unchanged. It grows the per-device vectors when a non-negative new GPU ID is configured, so GPU 1 can be configured or updated without implicitly configuring GPU 0. `BeginAdmission` checks the requested device's optional capacity before duplicate, transient, victim, or counter logic and returns an invalid `REJECTED` ticket with no state/counter mutation while unconfigured.

After that guard, `BeginAdmission` holds `mutex_`, snapshots candidates, calls the one `IsEvictionEligible(EvictionState)` predicate, deterministically selects and reserves a victim, and inserts a pending ticket. Reservation increments a manager lease so another caller cannot select the same victim. Duplicate keys return a valid `ALREADY_RESIDENT` no-op ticket. Capacity with no eligible victim returns invalid `REJECTED`; transient mode returns valid `TRANSIENT` without a victim reservation. `EvictReserved`, `CommitAdmission`, `AbortAdmission`, and `ReleaseLease` validate IDs exactly once and return `false` on repeats; every terminal path releases reservation leases. `EvictReserved` is the only API that may evict a persistent expert and decrement membership/bytes. `CommitAdmission` is the only API that may add persistent membership/bytes. `AdmissionSource` updates demand or prefetch counters in the same snapshot. Add `expert_residency.cpp` to `ARCHER_CORE_CXX_SOURCES`.

Create one shared, unconfigured `kExpertResidencyManager` alongside the native topology/task-pool lifetime. Pass the same `shared_ptr` into `ArcherTaskPool` and `ExpertDispatcher`; do not instantiate one manager per component or use any constructor-time sparse-limit reading to configure it. Existing constructor state used exclusively by the disabled legacy path remains unchanged.

At the end of each successful `ArcherPrefetchHandle::SetTopology` and `SetTopologyV2`, after `InitializeTopology`/`InitializeTopologyV2` returns, configure every CUDA device from the now-finalized topology and memory-pool limits:

```cpp
void ArcherPrefetchHandle::ConfigureExpertCapacityAfterTopology() {
  for (int gpu_id = 0; gpu_id < kNumDevices(); ++gpu_id) {
    const auto device = torch::Device(torch::kCUDA, gpu_id);
    const auto bytes = kTopologyHandle->GetSparseCacheLimit(device);
    TORCH_CHECK(
        kExpertResidencyManager->ConfigureCapacity(gpu_id, bytes),
        "failed to configure expert residency capacity for GPU ", gpu_id,
        " with ", bytes, " bytes");
  }
}

void ArcherPrefetchHandle::SetTopology(
    const std::vector<
        std::tuple<std::string, std::vector<std::vector<TensorID>>>>&
        topology) {
  kTopologyHandle->InitializeTopology(topology);
  ConfigureExpertCapacityAfterTopology();
}

void ArcherPrefetchHandle::SetTopologyV2(
    const std::vector<
        std::tuple<std::string, bool, std::vector<std::vector<TensorID>>,
                   std::vector<std::uint64_t>>>& topology) {
  kTopologyHandle->InitializeTopologyV2(topology);
  ConfigureExpertCapacityAfterTopology();
}
```

Declare `void ConfigureExpertCapacityAfterTopology();` in the private section of `archer_prefetch_handle.h` and define it immediately before `SetTopology` so the source-order regression inspects the intended method. The helper is the only production bridge from topology-derived sparse limits to `ConfigureCapacity`. A failed reconfiguration raises before serving resumes rather than leaving a silently stale limit.

Define the sole predicate in `expert_residency.h`:

```cpp
struct EvictionState {
  bool is_cuda;
  bool is_prefetching;
  NodeExecState exec_state;
  int pending_dispatches;
  bool is_overflow;
  std::uint32_t lease_count;
  bool protected_candidate;
};

inline bool IsEvictionEligible(const EvictionState& state) {
  return state.is_cuda && !state.is_prefetching &&
         state.exec_state == NodeExecState::IDLE &&
         state.pending_dispatches == 0 && !state.is_overflow &&
         state.lease_count == 0 && !state.protected_candidate;
}

inline std::optional<VictimCandidate> SelectStableVictim(
    const std::vector<VictimCandidate>& candidates) {
  if (candidates.empty()) return std::nullopt;
  return *std::min_element(candidates.begin(), candidates.end(), VictimLess);
}
```

Include `<optional>` and use this exact function in production. The production state projection is:

```cpp
EvictionState state{
    node->device.is_cuda(),
    node->is_prefetching.load(std::memory_order_acquire),
    node->exec_state.load(std::memory_order_acquire),
    node->pending_dispatches.load(std::memory_order_acquire),
    node->is_overflow,
    entry.lease_count,
    protected_candidates_.find(node) != protected_candidates_.end(),
};
if (!IsEvictionEligible(state)) continue;
```

No dispatcher or scheduler code may duplicate a subset of these checks.

- [ ] **Step 8: Route dispatcher and prefetch scheduler through the authority**

Remove enabled-mode ownership of `cached_experts_`, `cache_sizes_`, `FindExpertEvict`, and task-scheduler topology scans. Construct `ExpertDispatcher` with `ExpertResidencyClient(shared_manager, AdmissionSource::DEMAND)` and `ArcherTaskPool` with `ExpertResidencyClient(shared_manager, AdmissionSource::PREFETCH)`. For demand misses, the dispatcher client calls `BeginAdmission`, optionally `EvictReserved`, transfers the incoming node under a transfer lease, and calls `CommitAdmission` or `AbortAdmission`. The prefetch client uses the same sequence and manager. `ReplaceCacheCandidates` calls `ReplaceProtectedCandidates` on the manager before clearing obsolete prefetch queues. `GetCacheOccupancyBytes` and all residency metrics delegate to `Snapshot`.

Clients may be constructed before topology, but they must not cache a capacity or bypass the manager's unconfigured check. No dispatcher thread, prefetch thread, or Python `configure_phase_policy` call invokes `ConfigureCapacity`; only the post-initialization handle helper does so.

Keep the disabled path unchanged behind `if (!phase_policy_.enabled)`: existing dispatcher cache containers and task-scheduler removal logic remain active only for compatibility. Add debug assertions that enabled code never directly mutates legacy `cached_experts_`/`cache_sizes_`.

Each demand access calls `RecordAccess` exactly once. A decode hit whose prior decode count is zero and prefill count is nonzero increments `transition_hits` once.

Change native interfaces consistently:

```cpp
void EnqueueExpert(int layer_idx, int expert_idx, int gpu_id = -1,
                   bool remote = false,
                   ExpertPhase phase = ExpertPhase::MIXED);
void ConfigurePhasePolicy(const PhasePolicyConfig& config);
void EnqueuePrefetchTensors(const std::vector<std::uint32_t>& tensor_ids,
                            std::uint32_t priority,
                            ExpertPhase phase = ExpertPhase::MIXED);
```

Store `phase` and `bypass_count` in `Task`, and `phase` plus manager lease IDs in `CallArgs`/`ExecArgs`. Pybind accepts integer phase values, validates `0..2`, and exposes `configure_phase_policy(enabled, prefill_admission, decode_admission, prefill_weight, decode_weight, starvation_limit)`.

- [ ] **Step 9: Implement admission and victim rules through tickets**

Under `config.enabled`:

* `cache`: on pressure, `BeginAdmission` reserves the eligible candidate with minimum stable `VictimCandidate`; `EvictReserved` evicts it; then `CommitAdmission` admits the miss persistently.
* `transient_on_pressure`: if enough free cache bytes exist, admit normally; otherwise use the existing single `gpu_overload_` slot, execute, and return the node to host in `OutputFunc` without changing the manager's persistent resident map/bytes.
* `mixed`: resolve to decode.
* Prefetch is admitted only if the shared manager can issue a ticket; failed prefetch returns without affecting demand state and increments `prefetch_rejected` in the manager.

When disabled, retain the current `batch_size > 1` branch and LFU victim selection byte-for-byte in an explicit `if (!phase_policy_.enabled)` path.

- [ ] **Step 10: Implement bounded prefetch bypass**

In `GPUThreadFunc`, always service priority 0 first. For priorities `1..19`, increment bypass counts on queued tasks when another prefetch is selected; if any task reaches the configured limit, select the stable oldest promoted task `(request_id, layer_id, expert_id)` from service class 1. Reset only the selected task. This provides bounded prefetch waiting without weakening demand priority.

- [ ] **Step 11: Expose one shared-residency snapshot**

Bind `ExpertResidencyManager::Snapshot()` as native `get_expert_policy_stats()` returning a flat `dict[str, int64_t]` with:

```text
enabled, resident_bytes, resident_experts,
prefill_accesses, prefill_hits, prefill_misses, prefill_admissions,
prefill_transient, prefill_evictions, prefill_prefetch_issued,
prefill_prefetch_completed, prefill_prefetch_rejected,
decode_accesses, decode_hits, decode_misses, decode_admissions,
decode_transient, decode_evictions, decode_prefetch_issued,
decode_prefetch_completed, decode_prefetch_rejected,
mixed_accesses, transition_hits, starvation_promotions
```

`resident_bytes` and `resident_experts` are totals, not sums of phase counters. Do not publish `prefill_resident_bytes` or `decode_resident_bytes`, because residency is shared.

- [ ] **Step 12: Verify source registration, real transactions, and Python wiring**

Run: `pytest tests/python/unit/test_setup_sources.py tests/python/unit/test_native_capacity_lifecycle.py tests/python/unit/test_native_phase_policy_wire.py tests/python/unit/test_phase_expert_prefetcher.py -q && cmake --build build --target test_expert_policy test_expert_residency -j && ctest --test-dir build -R 'Expert(Policy|Residency)Tests' --output-on-failure`

Expected: all pass; setup registers the source exactly once; constructors leave capacity unconfigured; both topology setters initialize before reading sparse limits and configuring each device; pre-configuration admission is rejected without mutation; devices configure independently; valid reconfiguration updates one device atomically while invalid/pending-ticket updates preserve the prior limit; every ineligibility state is rejected; 100 randomized candidate orders select the same victim; all nine real-manager tests pass; and demand/prefetch admissions contribute to one byte/count snapshot without duplication. The source-registration and six pre-existing transaction cases remain intact.

- [ ] **Step 13: Build the editable extension once**

Run: `pip install --no-build-isolation -e .`

Expected: `_store` compiles and links with `expert_residency.cpp`; importing `moe_infinity._store` has no undefined `ExpertResidencyManager` symbols.

- [ ] **Step 14: Commit native policy integration**

```bash
git add setup.py core/CMakeLists.txt core/model/model_topology.h core/prefetch/expert_policy.h core/prefetch/expert_residency.h core/prefetch/expert_residency.cpp core/prefetch/task_scheduler.h core/prefetch/task_scheduler.cpp core/parallel/expert_dispatcher.h core/parallel/expert_dispatcher.cpp core/prefetch/archer_prefetch_handle.h core/prefetch/archer_prefetch_handle.cpp core/python/py_archer_prefetch.cpp tests/cpp/unit/prefetch/CMakeLists.txt tests/cpp/unit/prefetch/expert_residency_test_fixture.h tests/cpp/unit/prefetch/test_expert_policy.cpp tests/cpp/unit/prefetch/test_expert_residency.cpp tests/python/unit/test_native_phase_policy_wire.py tests/python/unit/test_setup_sources.py tests/python/unit/test_native_capacity_lifecycle.py
git commit -m "feat: enforce phase-aware shared expert residency"
```

---

### Task 6: Publish phase telemetry through engine stats and Prometheus

**Files:**
- Modify: `moe_infinity/memory/expert_prefetcher.py:94-156`
- Modify: `moe_infinity/serving/engine.py:653-668`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:716-745`
- Modify: `tests/python/serving/test_engine.py`
- Modify: `tests/python/integration/test_stability_e2e.py`
- Modify: `tests/python/dflash/test_native_stat_accessors.py`

- [ ] **Step 1: Write failing telemetry tests**

```python
def test_policy_stats_fallback_is_disabled_and_zero() -> None:
    prefetcher = _bare_prefetcher(archer_engine=object(), expert_dispatcher=object())
    stats = prefetcher.get_policy_stats()
    assert stats["enabled"] == 0
    assert stats["resident_bytes"] == 0
    assert stats["decode_hits"] == 0
```

```python
def test_prometheus_formats_phase_policy_metrics() -> None:
    body = _format_prometheus_metrics({
        "expert_policy": {
            "enabled": 1,
            "resident_bytes": 4096,
            "prefill_hits": 3,
            "decode_hits": 7,
            "transition_hits": 2,
        }
    })
    assert 'moe_expert_cache_hits_total{phase="prefill"} 3' in body
    assert 'moe_expert_cache_hits_total{phase="decode"} 7' in body
    assert "moe_expert_cache_resident_bytes 4096" in body
    assert "moe_expert_cache_transition_hits_total 2" in body
```

- [ ] **Step 2: Run telemetry tests and verify RED**

Run: `pytest tests/python/dflash/test_native_stat_accessors.py tests/python/serving/test_engine.py tests/python/integration/test_stability_e2e.py -q`

Expected: missing `get_policy_stats` and absent Prometheus series.

- [ ] **Step 3: Add stable snapshots and labeled metrics**

`ExpertPrefetcher.get_policy_stats()` reads one `ExpertResidencyManager::Snapshot()` through the prefetch handle; it must not merge dispatcher and scheduler totals because both are clients of the same authority and merging would double-count. The dispatcher accessor, if retained for compatibility, returns the same manager snapshot and its manager identity must match the prefetch handle in tests. Return every documented key with integer zero fallback. `ServingEngine.get_stats()` reads `self.model_runner.engine.expert_prefetcher` if present and adds `"expert_policy": snapshot`; resident-only models return the all-zero disabled snapshot.

Add these Prometheus families:

```text
moe_expert_phase_policy_enabled
moe_expert_cache_resident_bytes
moe_expert_cache_resident_experts
moe_expert_cache_accesses_total{phase="prefill|decode|mixed"}
moe_expert_cache_hits_total{phase="prefill|decode"}
moe_expert_cache_misses_total{phase="prefill|decode"}
moe_expert_cache_admissions_total{phase="prefill|decode"}
moe_expert_cache_transient_total{phase="prefill|decode"}
moe_expert_cache_evictions_total{phase="prefill|decode"}
moe_expert_prefetch_total{phase="prefill|decode",result="issued|completed|rejected"}
moe_expert_cache_transition_hits_total
moe_expert_prefetch_starvation_promotions_total
```

Use a fixed label allowlist; never interpolate arbitrary configuration strings.

- [ ] **Step 4: Verify stats and Prometheus tests**

Run: `pytest tests/python/dflash/test_native_stat_accessors.py tests/python/serving/test_engine.py tests/python/integration/test_stability_e2e.py -q`

Expected: all pass, including empty/resident fallback.

- [ ] **Step 5: Commit telemetry**

```bash
git add moe_infinity/memory/expert_prefetcher.py moe_infinity/serving/engine.py moe_infinity/entrypoints/openai/api_server_v2.py tests/python/dflash/test_native_stat_accessors.py tests/python/serving/test_engine.py tests/python/integration/test_stability_e2e.py
git commit -m "feat: expose phase-specific expert policy metrics"
```

---

### Task 7: Add deterministic integration coverage for transitions and starvation

**Files:**
- Create: `tests/python/integration/test_phase_specific_expert_policy.py`
- Modify: `tests/python/integration/test_flashinfer_offload_wiring.py`
- Modify: `tests/python/integration/test_model_consistency.py`

- [ ] **Step 1: Add a fake-native transition test**

```python
def test_prefill_to_decode_reuses_shared_resident_expert() -> None:
    runtime = FakePolicyRuntime(capacity=2, decode_weight=4.0, starvation_limit=2)
    runtime.dispatch(phase="prefill", layer=0, experts=[1, 2])
    before = runtime.resident_keys()
    runtime.dispatch(phase="decode", layer=0, experts=[1])
    assert before == {(0, 1), (0, 2)}
    assert runtime.resident_keys() == {(0, 1), (0, 2)}
    assert runtime.stats()["transition_hits"] == 1
    assert runtime.unique_store_keys() == {(0, 1), (0, 2)}
```

Add tests for: enabled mixed input becomes two forwards in decode/prefill order; disabled non-paged input remains one mixed forward; disabled paged input remains prefill then decode; disabled policy records no phase actions; decode-weighted eviction keeps a decode-hot expert; transient prefill pressure does not evict it; and a prefill prefetch is promoted after the configured bypass bound.

Add two explicitly configured model-consistency cases rather than an environment-variable switch:

```python
def test_model_consistency_with_explicit_phase_policy(model_fixture) -> None:
    outputs = []
    for enabled in (False, True):
        config = ArcherConfig.load_from_json({
            "offload_path": str(model_fixture.offload_path),
            "use_native_engine": True,
            "device_memory_ratio": 0.75,
            "kv_cache_memory_ratio": 0.15,
            "phase_specific_expert_policy": enabled,
        })
        outputs.append(model_fixture.generate(
            config=config,
            seed=0,
            do_sample=False,
            prompt_token_ids=model_fixture.prompt_token_ids,
            max_new_tokens=16,
        ))
    assert outputs[0] == outputs[1]
```

- [ ] **Step 2: Run integration tests and verify RED**

Run: `pytest tests/python/integration/test_phase_specific_expert_policy.py tests/python/integration/test_flashinfer_offload_wiring.py -q`

Expected: fake-native adapter lacks the new phase/config semantics or the transition counter.

- [ ] **Step 3: Complete test adapters and assertions without changing production policy**

Implement `FakePolicyRuntime` entirely in the test file using the same stable victim tuple and one shared `set[(layer, expert)]`. Use it to test policy invariants independent of CUDA scheduling. Extend FlashInfer offload wiring assertions to observe one prefill phase followed by decode phases. Make the model-consistency fixture construct `ArcherConfig` with the explicit Boolean shown above; do not add an environment-variable seam. Off/on runs use the same seed, greedy decoding, prompt, and output length, then assert token IDs match; do not assert latency improvement.

- [ ] **Step 4: Run deterministic integration coverage**

Run: `pytest tests/python/integration/test_phase_specific_expert_policy.py tests/python/integration/test_flashinfer_offload_wiring.py tests/python/integration/test_model_consistency.py -q`

Expected: all pass; correctness is identical with policy off/on for the deterministic fixture.

- [ ] **Step 5: Commit integration tests**

```bash
git add tests/python/integration/test_phase_specific_expert_policy.py tests/python/integration/test_flashinfer_offload_wiring.py tests/python/integration/test_model_consistency.py
git commit -m "test: cover phase policy transitions and starvation"
```

---

### Task 8: Add the TTFT/TPOT workload matrix and reporting

**Files:**
- Create: `benchmarks/serving/phase_specific_expert_policy.py`
- Create: `tests/python/benchmarks/test_phase_specific_expert_policy.py`
- Modify: `docs/benchmarking.md`

- [ ] **Step 1: Write failing benchmark-matrix tests**

```python
from benchmarks.serving.phase_specific_expert_policy import build_matrix, summarize


def test_matrix_covers_prefill_decode_and_mixed_pressure() -> None:
    cells = build_matrix()
    assert {(c.prompt_tokens, c.output_tokens, c.concurrency) for c in cells} == {
        (128, 16, 1), (2048, 16, 1), (128, 256, 1),
        (2048, 256, 1), (128, 256, 8), (2048, 256, 8),
    }


def test_summary_uses_stream_timestamps_for_ttft_and_tpot() -> None:
    row = summarize(submitted=1.0, token_times=[2.0, 2.2, 2.5])
    assert row["ttft_s"] == 1.0
    assert row["tpot_s"] == 0.25
```

- [ ] **Step 2: Run benchmark unit tests and verify RED**

Run: `pytest tests/python/benchmarks/test_phase_specific_expert_policy.py -q`

Expected: import fails because the benchmark module does not exist.

- [ ] **Step 3: Implement the benchmark harness**

The script must:

* launch or target one existing OpenAI server; never launch separate prefill/decode workers;
* run policy `off` and `on` as separate server invocations with the same model, offload directory, `device_memory_ratio`, GPU visibility, seed, and greedy sampling;
* execute the six matrix cells above, with one warmup plus five measured repeats;
* submit concurrency-8 requests simultaneously with a barrier and record each SSE token timestamp;
* calculate per-request TTFT, TPOT as the mean gap after the first token, E2E, and output token count; report p50/p90/p99 across requests;
* scrape `/admin/stats` after each cell for phase accesses/hits/misses/admissions/transient/evictions/prefetch outcomes, transition hits, starvation promotions, and shared resident bytes;
* record commit, full commands, model, tokenizer, CUDA/PyTorch/Transformers versions, GPU/CPU, offload medium, raw rows, and summary JSON;
* reject comparisons if generated token IDs, request counts, prompt/output lengths, or environment fingerprints differ;
* print deltas only, never label them as speedups or wins.

CLI:

```text
--server-url --model --output-json --policy {off,on}
--prompt-lengths 128 2048 --output-lengths 16 256
--concurrency 1 8 --warmup 1 --repeats 5 --seed 0
```

The matrix intentionally includes prefill-heavy `(2048,16,1)`, decode-heavy `(128,256,1)`, balanced `(2048,256,1)`, and mixed continuous-batching pressure at concurrency 8.

- [ ] **Step 4: Verify benchmark helpers**

Run: `pytest tests/python/benchmarks/test_phase_specific_expert_policy.py -q`

Expected: all pass without a server or GPU.

- [ ] **Step 5: Document benchmark interpretation**

Add the exact matrix, warmup/repeat rules, TTFT/TPOT definitions, required telemetry, environment matching, and token-parity gate to `docs/benchmarking.md`. State that DuoServe-MoE is motivation only and that measured results from this repository must stand on their own.

- [ ] **Step 6: Commit benchmark and runbook**

```bash
git add benchmarks/serving/phase_specific_expert_policy.py tests/python/benchmarks/test_phase_specific_expert_policy.py docs/benchmarking.md
git commit -m "bench: add phase expert policy workload matrix"
```

---

### Task 9: Document safe defaults, compatibility, and rollback

**Files:**
- Modify: `docs/configuration.md`
- Modify: `docs/serving.md`
- Modify: `docs/troubleshooting.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add source-of-truth configuration documentation**

Document every field from the public configuration contract, including types, defaults, ranges, and the fact that all subordinate values are inert while `phase_specific_expert_policy=False`. Add this minimal enablement example:

```json
{
  "phase_specific_expert_policy": true,
  "prefill_expert_admission": "transient_on_pressure",
  "decode_expert_admission": "cache",
  "prefill_expert_prefetch_top_k": 0,
  "decode_expert_prefetch_top_k": 2
}
```

- [ ] **Step 2: Document operational invariants and rollback**

Add an operator section stating:

1. The expert store and GPU cache remain shared; there are no phase pools.
2. Enabled mixed batches execute decode then prefill and are recombined; disabled mode restores the original non-paged combined behavior and paged prefill-then-decode order.
3. `ExpertResidencyManager` is the one enabled-mode authority for residency, bytes, leases, candidate protection, and eviction; dispatcher/prefetch counters are views of its snapshot.
4. Disable with `--no-phase-specific-expert-policy` (or `"phase_specific_expert_policy": false` for in-process config) and restart; subordinate keys may remain in JSON.
5. Rollback requires no cache deletion or offload-store migration because phase is not persisted in tensor IDs, topology, or checkpoint metadata.
6. If `starvation_promotions`, `prefetch_rejected`, TPOT, or TTFT regress, capture `/admin/stats`, `/metrics`, config, and benchmark JSON before disabling.
7. Do not tune `device_memory_ratio` between A/B runs.
8. Adaptive expert precision is not parallel-merge composable with this branch. Land this fixed-size phase-policy manager first, then rebase adaptive precision onto the same lease/accounting transactions and add combined tests before co-enabling.

- [ ] **Step 3: Update changelog without a performance claim**

Add an Unreleased entry describing an opt-in phase-specific policy, shared residency, telemetry, and compatibility. Do not include speedup numbers or comparative claims.

- [ ] **Step 4: Check documentation links and prohibited claims**

Run: `python -m compileall moe_infinity benchmarks/serving/phase_specific_expert_policy.py`

Expected: Python compilation passes. Manually open each newly added relative Markdown link and confirm its target is one of the existing files listed in this task.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/configuration.md docs/serving.md docs/troubleshooting.md CHANGELOG.md
git commit -m "docs: explain phase-specific expert policy rollback"
```

---

### Task 10: Perform CPU, CUDA, model, and benchmark validation

**Files:**
- No new files; record raw outputs under the operator-selected benchmark output directory, not in the repository.

- [ ] **Step 1: Run the CPU regression suite**

Run:

```bash
pytest \
  tests/python/unit/test_expert_policy.py \
  tests/python/unit/test_utils_config.py \
  tests/python/unit/test_phase_expert_prefetcher.py \
  tests/python/unit/test_native_phase_policy_wire.py \
  tests/python/unit/test_native_capacity_lifecycle.py \
  tests/python/unit/test_setup_sources.py \
  tests/python/serving/test_batch.py \
  tests/python/serving/test_model_runner.py \
  tests/python/serving/test_engine.py \
  tests/python/integration/test_phase_specific_expert_policy.py \
  tests/python/dflash/test_speculative_prefetch.py \
  tests/python/dflash/test_route_ahead_wire.py \
  tests/python/dflash/test_native_stat_accessors.py \
  tests/python/benchmarks/test_phase_specific_expert_policy.py -q
```

Expected: all pass with no xfails added for this feature.

- [ ] **Step 2: Run native deterministic tests and rebuild**

Run:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --target test_expert_policy test_expert_residency -j
ctest --test-dir build -R 'Expert(Policy|Residency)Tests' --output-on-failure
pip install --no-build-isolation -e .
python -c "import moe_infinity._store"
```

Expected: policy and all nine real-manager lifecycle/transaction tests pass; editable `_store` builds, links `expert_residency.cpp`, and imports without an undefined manager symbol.

- [ ] **Step 3: Run CUDA store/dispatch validation**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 pytest \
  tests/docker/test_io_integration.py \
  tests/python/ops/test_expert_dispatch.py \
  tests/python/dflash/test_prefetch_native_gpu.py -q
```

Expected: all available CUDA tests pass; skips are acceptable only for explicitly documented optional kernels, not for phase argument or pybind failures.

- [ ] **Step 4: Run model correctness with policy off and on**

Use the smallest supported offloaded MoE fixture available locally. The test itself constructs `ArcherConfig` twice with `phase_specific_expert_policy=False` and `True`, while holding greedy decoding and seed zero fixed:

```bash
CUDA_VISIBLE_DEVICES=0 pytest tests/python/integration/test_model_consistency.py::test_model_consistency_with_explicit_phase_policy -q
```

Expected: the test's explicit off and on configurations produce identical token IDs. No undeclared environment variable controls policy state.

- [ ] **Step 5: Run the TTFT/TPOT matrix**

Start the disabled server with the tested Boolean CLI seam:

```bash
CUDA_VISIBLE_DEVICES=0 python -m moe_infinity.entrypoints.openai.api_server_v2 \
  --host 127.0.0.1 --port 8000 \
  --model "$MODEL" --offload-dir "$OFFLOAD_DIR" \
  --device-memory-ratio 0.75 \
  --no-phase-specific-expert-policy
```

In a second shell, run:

```bash
python benchmarks/serving/phase_specific_expert_policy.py \
  --server-url http://127.0.0.1:8000 \
  --model "$MODEL" \
  --policy off \
  --prompt-lengths 128 2048 \
  --output-lengths 16 256 \
  --concurrency 1 8 \
  --warmup 1 --repeats 5 --seed 0 \
  --output-json "$RESULT_DIR/policy-off.json"
```

After the off run, stop the server and restart the same server command with only `--no-phase-specific-expert-policy` changed to `--phase-specific-expert-policy`. Then run:

```bash
python benchmarks/serving/phase_specific_expert_policy.py \
  --server-url http://127.0.0.1:8000 \
  --model "$MODEL" \
  --policy on \
  --prompt-lengths 128 2048 \
  --output-lengths 16 256 \
  --concurrency 1 8 \
  --warmup 1 --repeats 5 --seed 0 \
  --output-json "$RESULT_DIR/policy-on.json"
```

The benchmark's `--policy` value labels and validates the expected server state from `/admin/stats`; it does not mutate server configuration.

Expected: both JSON files contain all six cells, matching environment fingerprints and token IDs, TTFT/TPOT percentiles, shared residency, and per-phase policy counters. Treat the results as validation data, not a guaranteed improvement.

- [ ] **Step 6: Exercise rollback**

Restart with the explicit `--no-phase-specific-expert-policy` CLI while leaving subordinate policy values at their defaults. Run one prefill+decode request and one mixed-batch request.

Expected: requests succeed, non-paged mixed execution is one combined forward, paged mixed execution remains prefill then decode, `moe_expert_phase_policy_enabled 0`, phase-specific counters stay zero, legacy hit-rate/occupancy remain available, and the existing offload directory is reused without migration or deletion.

- [ ] **Step 7: Run final static diagnostics**

Run:

```bash
python -m compileall moe_infinity benchmarks/serving/phase_specific_expert_policy.py
ruff check moe_infinity tests/python benchmarks/serving/phase_specific_expert_policy.py
mypy moe_infinity/utils/config.py
```

Expected: all commands pass without new warnings or errors.

- [ ] **Step 8: Commit only if validation required a test/runbook correction**

If validation changed tracked tests or docs, stage only the validation files named by Tasks 7-9 with:

```bash
git add tests/python/integration/test_phase_specific_expert_policy.py tests/python/integration/test_flashinfer_offload_wiring.py tests/python/integration/test_model_consistency.py tests/python/benchmarks/test_phase_specific_expert_policy.py docs/benchmarking.md docs/configuration.md docs/serving.md docs/troubleshooting.md CHANGELOG.md
git commit -m "test: finalize phase expert policy validation"
```

Do not commit benchmark outputs, local offload data, model weights, or environment-specific paths.

---

## Acceptance checklist

- Existing configs omit all new fields and retain legacy behavior.
- `phase_specific_expert_policy=False` bypasses new admission, prefetch, eviction, starvation, and decode-first splitting logic.
- Disabled non-paged mixed batches remain one combined forward; disabled paged mixed batches remain prefill then decode.
- Prefill, decode, and mixed phase values reach every local and RPC native dispatcher call.
- Enabled production mixed batches execute decode then prefill once each and recombine outputs exactly.
- The prefill-to-decode transition preserves shared residency and records transition reuse.
- After runtime readiness, admission changes residency only; it never changes router selections or denies routed demand.
- One shared `ExpertResidencyManager` instance is used by dispatcher and prefetch scheduler and is the sole enabled-mode authority for membership, bytes, leases, protection, victim reservation, commit/abort, eviction, and counters.
- The shared manager is constructed unconfigured; admissions on an unconfigured device return invalid `REJECTED` tickets without mutating residency or policy counters.
- `SetTopology` and `SetTopologyV2` call `ConfigureCapacity(gpu_id, bytes)` only after their topology initializer returns and `GetSparseCacheLimit` can observe finalized topology plus memory-pool limits; no constructor-time value and no dispatcher, scheduler, or Python policy setup call configures the manager. Disabled-only legacy cache bookkeeping remains unchanged.
- Capacity can be configured and updated independently per device. Negative limits, limits below resident bytes, and updates while that device has pending tickets fail atomically and preserve the previous capacity.
- `setup.py::_STORE_SOURCES` and `core/CMakeLists.txt::ARCHER_CORE_CXX_SOURCES` each register `core/prefetch/expert_residency.cpp` exactly once; editable `_store` imports without unresolved manager symbols.
- The sole production eligibility predicate rejects non-CUDA, `is_prefetching`, `FETCHING`, `EXECUTING`, `pending_dispatches > 0`, `is_overflow`, leased, and protected candidates; tests cover every state.
- Victim selection returns the same `(layer_id, expert_id)` across 100 randomized candidate orders.
- Real-manager GoogleTests retain capacity enforcement, duplicate admission, concurrent victim reservation, commit/abort idempotence, lease release idempotence, and shared byte accounting through production `ExpertResidencyClient` instances, and add unconfigured rejection, independent device configuration, and atomic reconfiguration coverage.
- Prefetch starvation is bounded while on-demand remains strict priority.
- Telemetry exposes per-phase actions but only aggregate shared residency.
- CPU tests, native policy tests, extension build, CUDA dispatch/store tests, and model token parity pass.
- Validation uses explicit `ArcherConfig` values and the tested `--[no-]phase-specific-expert-policy` CLI; no undefined environment seam remains.
- Benchmark output includes the full TTFT/TPOT matrix, telemetry, environment fingerprint, and no paper-derived or unmeasured speedup claim.
- Rollback is `--no-phase-specific-expert-policy` plus restart, restores both legacy mixed-batch paths, and requires no store migration.
- Adaptive precision is documented as non-composable by parallel merge; this fixed-size branch lands independently first, and adaptive precision must later rebase onto the shared residency/lease/accounting transaction substrate before co-enablement.
