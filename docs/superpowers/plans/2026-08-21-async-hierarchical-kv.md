# Asynchronous Hierarchical KV Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace serving's blocking KV swap copies with bounded, event-driven GPU↔pinned-host transfers that preserve paged-cache ownership and correctness, while retaining a synchronous fallback and defining—but not implementing—a host↔external-store extension boundary.

**Architecture:** `PagedKVCache` remains the sole owner of serving KV tensors and block tables, and delegates data movement to a shared transfer module with a bounded pinned-buffer pool, one CUDA transfer stream per device, completion events, and an injectable CPU fake backend. Serving scheduling polls completions and admits only GPU-resident sequences; GPU blocks remain owned until D2H completion, destination blocks remain reserved until H2D completion, and cancellation uses tombstones so in-flight DMA never targets recycled storage. The existing native-engine coordinator reuses the shared transfer primitives, while `UnifiedTransferScheduler` remains the expert/KV admission queue rather than becoming the CUDA-completion owner.

**Tech Stack:** Python 3.10+, PyTorch tensors/`torch.cuda.Stream`/`torch.cuda.Event`, pinned host memory, pytest CPU and CUDA markers, existing continuous-batching scheduler, Prometheus text exposition.

---

## Scope, repository findings, and fixed decisions

- The active serving layout in `moe_infinity/serving/kv_cache.py:185-196` is `[num_layers, num_blocks, 2, block_size, num_heads, head_dim]`. Preserve this layout and copy `self._kv_cache[:, block_ids, ...]` without changing attention metadata or adding KV quantization.
- Current `swap_out()` blocks on `.to("cpu").clone()` at `moe_infinity/serving/kv_cache.py:341-352`; current `swap_in()` blocks on `.to(device)` and assignment at `:354-379`. Keep these semantics behind the `sync` backend.
- `Scheduler._preempt_oldest_running_group()` currently frees blocks immediately after `swap_out()` (`moe_infinity/serving/scheduler.py:474-496`). Async mode must delay that free until the D2H event completes.
- `Scheduler._recover_swapped_groups()` currently makes a sequence runnable immediately after `swap_in()` (`moe_infinity/serving/scheduler.py:500-539`). Async mode must wait for H2D completion and metadata validation.
- Keep serving `SequenceStatus` unchanged. `SequenceStatus.SWAPPED` means “not runnable because its KV is moving or host-resident”; detailed transfer state belongs to `PagedKVCache`, not the user-visible request lifecycle.
- The older `KVCacheOffloadCoordinator` creates streams and then synchronizes them (`moe_infinity/engine/kv_cache_offload_coordinator.py:49-123`). Refactor it to reuse the shared copy backend, but do not route serving transfers through `UnifiedTransferScheduler`: its worker calls `future.result()` (`moe_infinity/engine/unified_transfer_scheduler.py:249-253`) and would serialize event completion. `UnifiedTransferScheduler` continues to schedule native-engine/expert work and receives accurate byte/failure results from coordinator handlers.
- Pinned memory is a hard-cap pool. Pool exhaustion returns backpressure; it never silently allocates pageable memory or exceeds the configured cap.
- That pinned-memory rule applies only to `kv_swap_mode="async"`. Default `sync` owns one explicit `PageableCPUBufferRecord` per host-resident sequence, creates it with the current blocking `.detach().to("cpu").clone()` expression, restores with the current blocking `.to(device, dtype)` assignment, and releases it by dropping its tensor reference. Sync construction, reservation, swap, free, and shutdown never construct, acquire, release, or account a `PinnedBufferPool`/`PinnedBufferLease`; therefore `kv_swap_host_memory_bytes` and `kv_swap_max_inflight_bytes` are accepted configuration but inactive in sync mode.
- `kv_swap_max_inflight_bytes` is a second hard cap over tickets whose CUDA events are incomplete; host-resident leases still count against `kv_swap_host_memory_bytes` but not against the in-flight cap. Exceeding either cap returns backpressure before state or ownership changes.
- CUDA operations cannot be cancelled after submission. Cancellation marks a record `CANCEL_PENDING`; event polling later reclaims source/destination blocks and host leases.
- Every sequence allocation receives a monotonically increasing generation. Completion records include `(seq_id, generation)` so a late event cannot mutate a reused sequence ID (ABA protection).
- Metadata format version is `1`. Shape, dtype, byte length, generation, and block count are always validated. CRC32 is opt-in (`kv_swap_checksum=False` by default) because it consumes host CPU time; when present, mismatch prevents H2D submission and triggers recovery.
- Async mode is opt-in. `kv_swap_mode="sync"` is the default. `kv_swap_mode="async"` falls back to sync at initialization when CUDA or pinned allocation is unavailable if `kv_swap_allow_sync_fallback=True`; runtime data corruption never falls back to using suspect bytes.
- The external tier is only a protocol for a later pinned-host↔external transfer. This change does not implement SSD, RDMA, object storage, distributed coordination, or multi-node behavior. [Mooncake](https://arxiv.org/abs/2407.00079) motivates separating transfer control from storage tiers; it is not a dependency and is not evidence for a performance claim.

## Exact transfer state, sequence recovery, and ownership

`KVTransferState` describes storage/DMA only. It never contains `WAITING`, `PREFILL`, `DECODE`, `DRAFT`, or `VERIFY`, and no transfer-state arrow targets a `SequenceStatus`. Scheduler recovery is a separate operation after transfer cleanup.

```text
GPU_RESIDENT
  --group swap-out reserved/submitted--> SWAP_OUT_IN_FLIGHT
  --reservation backpressure-----------> GPU_RESIDENT (no mutation)

SWAP_OUT_IN_FLIGHT
  --ticket complete + metadata valid---> HOST_RESIDENT
  --submit/event failure---------------> GPU_RESIDENT
  --free_sequence/cancel---------------> CANCEL_PENDING

HOST_RESIDENT
  --group swap-in reserved/submitted----> SWAP_IN_IN_FLIGHT
  --no destination capacity-------------> HOST_RESIDENT (no mutation)
  --metadata/checksum invalid-----------> FAILED
  --free_sequence/cancel----------------> CANCELLED

SWAP_IN_IN_FLIGHT
  --ticket complete---------------------> GPU_RESIDENT
  --submit/event failure before retry---> HOST_RESIDENT
  --retry exhaustion or bad metadata----> FAILED
  --free_sequence/cancel----------------> CANCEL_PENDING

CANCEL_PENDING
  --ticket complete, retired, reclaimed-> CANCELLED

FAILED
  --discard_failed_for_reprefill--------> record removed
```

After `discard_failed_for_reprefill(seq_id, generation)` removes the failed transfer record and releases only resources that are no longer DMA-owned, `Scheduler` independently performs `SequenceStatus.SWAPPED -> SequenceStatus.WAITING`, resets `SequenceData.num_computed_tokens`, and moves the whole group from `_swapped` to `_waiting`. No transfer-state transition is performed during that scheduler recovery.

Block ownership is:

```text
FREE -> ALLOCATED -> EVICTING -> FREE
FREE -> RESTORING -> ALLOCATED
EVICTING/RESTORING --cancel requested--> unchanged until ticket retirement
EVICTING/RESTORING --retired cancel/failure--> FREE
```

### Authoritative `free_sequence()` / cancellation rules

`free_sequence()` is the single cache-destruction entry point; `cancel_sequence()` delegates to it. It first removes the sequence from active lookup so it cannot be scheduled, but retains a generation-keyed tombstone in `_retiring_records` whenever DMA is in flight.

| State at call | Immediate action | Deferred action |
| --- | --- | --- |
| `GPU_RESIDENT` | Free ALLOCATED blocks and remove table/record because no ticket owns them. | None. |
| `HOST_RESIDENT` | Async: release the pinned lease. Sync: call `PageableCPUBufferRecord.release()`. Then remove metadata/table/record because no ticket owns them. | None. |
| `FAILED` | Release the non-DMA-owned mode-specific host owner (pinned lease or pageable record), release restoring blocks, and remove record. | None. |
| `SWAP_OUT_IN_FLIGHT` | Set `CANCEL_PENDING`; move record/table to `_retiring_records[(seq_id, generation)]`; retain EVICTING source blocks, host lease, ticket, stream, event, and staging tensors. | After event completion: retire ticket, discard copied host bytes, free source blocks, release lease, remove tombstone. |
| `SWAP_IN_IN_FLIGHT` | Set `CANCEL_PENDING`; move record/table to `_retiring_records`; retain RESTORING destination blocks, host lease, ticket, stream, event, and staging tensors. | After event completion: retire ticket, free destination blocks without publication, release lease, remove tombstone. |
| `CANCEL_PENDING` / `CANCELLED` | Idempotent no-op. | Existing retirement continues. |

Neither `free_sequence()`, `abort_request()`, generation reuse, retry, nor shutdown may free a block, release an async pinned lease, drop the final staging-tensor reference, or destroy a stream before its ticket is retired. Sync has no in-flight DMA: its pageable record is released immediately on completed swap-in, host-resident `free_sequence()`, or shutdown. A new allocation of the same `seq_id` gets a new generation and cannot address the old tombstone.

### Group-atomic scheduler rules

Add `_SwappedGroupRecord(group, prior_status_by_seq, phase)` with phases `OUT_IN_FLIGHT`, `HOST_RESIDENT`, `IN_IN_FLIGHT`, `ROLLBACK_IN_FLIGHT`, and `REPREFILL_PENDING`.

```python
class SwapGroupPhase(str, Enum):
    OUT_IN_FLIGHT = "out_in_flight"
    HOST_RESIDENT = "host_resident"
    IN_IN_FLIGHT = "in_in_flight"
    ROLLBACK_IN_FLIGHT = "rollback_in_flight"
    REPREFILL_PENDING = "reprefill_pending"


@dataclass
class _SwappedGroupRecord:
    group: SequenceGroup
    prior_status_by_seq: dict[int, SequenceStatus]
    phase: SwapGroupPhase
```

1. At the start of exactly one `schedule()` pass, call `kv_cache.poll_transfers()` once and fold all completions into group phases; do not move queues while iterating individual completions. Complete group success/rollback/reprefill reservations before admitting new waiting work so blocks freed by a partial D2H cannot be stolen before rollback.
2. Preemption calls `reserve_swap_out_group(seq_ids)` to reserve every lease and all in-flight bytes before any submission. On reservation failure, the group remains in `_running` with every status unchanged. On success, submit every member, atomically remove the group from `_running`, append it once to `_swapped`, save prior statuses, set every member to `SequenceStatus.SWAPPED`, and set `OUT_IN_FLIGHT`.
3. `OUT_IN_FLIGHT` remains in `_swapped` until every member settles. All `HOST_RESIDENT` becomes group `HOST_RESIDENT`. Any D2H failure starts rollback: host-resident members are group-restored, GPU-resident members remain owned, phase becomes `ROLLBACK_IN_FLIGHT`, and no member is runnable.
4. Recovery calls `reserve_swap_in_group(seq_ids)` only when the group is wholly `HOST_RESIDENT` and enough blocks exist for every member. On success phase becomes `IN_IN_FLIGHT`; the group remains once in `_swapped` and all statuses remain `SWAPPED`.
5. Only when every member is `GPU_RESIDENT` does one atomic commit remove the group from `_swapped`, append it once to `_running`, restore every prior status, and permit decode/prefill selection. Partial completion never changes a member status or queue.
6. Terminal H2D/metadata failure sets `REPREFILL_PENDING`; wait for every other ticket to retire, call `discard_failed_for_reprefill()`/`discard_host_copy()` for every member, reset all sequence accounting, atomically remove from `_swapped`, append once to `_waiting`, and set every member `SWAPPED -> WAITING`.
7. Group cancellation removes the group from scheduler queues immediately but cache tombstones retain DMA ownership until retirement.

## Public/internal API contract

Create the shared API in `moe_infinity/engine/kv_transfer.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import torch

KV_FORMAT_VERSION = 1


class KVTransferState(str, Enum):
    GPU_RESIDENT = "gpu_resident"
    SWAP_OUT_IN_FLIGHT = "swap_out_in_flight"
    HOST_RESIDENT = "host_resident"
    SWAP_IN_IN_FLIGHT = "swap_in_in_flight"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BlockOwnership(str, Enum):
    FREE = "free"
    ALLOCATED = "allocated"
    EVICTING = "evicting"
    RESTORING = "restoring"


@dataclass(frozen=True)
class KVObjectKey:
    seq_id: int
    generation: int


@dataclass(frozen=True)
class KVObjectMetadata:
    format_version: int
    key: KVObjectKey
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    num_tokens: int
    block_count: int
    block_size: int
    valid_tokens_last_block: int
    checksum_crc32: int | None


@dataclass
class PinnedBufferLease:
    tensor: torch.Tensor
    nbytes: int
    lease_id: int


@dataclass
class PageableCPUBufferRecord:
    """Sync-only owner of one pageable host copy; never belongs to a pool."""

    tensor: torch.Tensor | None
    nbytes: int

    @classmethod
    def from_blocking_clone(cls, source: torch.Tensor) -> "PageableCPUBufferRecord":
        tensor = source.detach().to("cpu").clone()
        return cls(tensor=tensor, nbytes=tensor.numel() * tensor.element_size())

    def require_tensor(self) -> torch.Tensor:
        if self.tensor is None:
            raise RuntimeError("pageable CPU buffer has been released")
        return self.tensor

    def release(self) -> None:
        self.tensor = None


class CompletionEvent(Protocol):
    def query(self) -> bool: ...
    def synchronize(self) -> None: ...


@dataclass
class CopyTicket:
    device: torch.device
    stream: torch.cuda.Stream | None
    event: CompletionEvent
    owned_staging_tensors: tuple[torch.Tensor, ...]
    submitted_ns: int
    nbytes: int
    retired: bool = field(default=False, init=False)

    def query(self) -> bool:
        return self.event.query()

    def wait_on_consumer(self, consumer_stream: torch.cuda.Stream) -> None:
        if self.retired:
            raise RuntimeError("cannot wait on a retired copy ticket")
        consumer_stream.wait_event(self.event)

    def retire(self, *, synchronize: bool = False) -> bool:
        if self.retired:
            return True
        if synchronize:
            self.event.synchronize()
        elif not self.event.query():
            return False
        self.owned_staging_tensors = ()
        self.retired = True
        return True


class KVTransferBackend(Protocol):
    @property
    def asynchronous(self) -> bool: ...

    def submit_d2h(
        self,
        source_cache: torch.Tensor,
        destination: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket: ...

    def submit_h2d(
        self,
        source: torch.Tensor,
        destination_cache: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket: ...

    def close(self) -> None: ...


class ExternalKVStore(Protocol):
    """Future host↔external boundary; no implementation in this change."""

    def submit_put(self, key: KVObjectKey, source: memoryview, metadata: KVObjectMetadata) -> str: ...
    def submit_get(self, key: KVObjectKey, destination: memoryview, metadata: KVObjectMetadata) -> str: ...
    def poll(self, operation_id: str) -> str: ...
    def cancel(self, operation_id: str) -> bool: ...
    def delete(self, key: KVObjectKey) -> None: ...
    def close(self) -> None: ...
```

Ticket lifecycle is exact: `CopyTicket` owns its event and staging tensors and holds a strong reference to its device/transfer stream; the backend owns final stream destruction. The producer stream records prior KV writes; the transfer stream waits for that producer before gather/copy; the event is recorded on the transfer stream; an early consumer calls `wait_on_consumer()` before reading destination bytes; serving normally publishes blocks only after `query()` succeeds; `retire()` drops staging ownership only after completion but keeps device/stream/event available for diagnostics until the record is removed. Shutdown ordering is: reject submissions → cancel scheduler visibility → synchronize every unretired event → retire every ticket → finalize cancelled/failed records → release leases/blocks → remove ticket records → close/destroy transfer streams → release the pool.

Metadata validation receives `(metadata, payload, expected_key, block_ids, total_blocks)` and independently verifies: format version; exact generation/key; six-dimensional serving shape and block-axis count; exact dtype; exact padded payload bytes; block count; positive block size; unique block IDs within `[0, total_blocks)`; `0 <= num_tokens <= block_count * block_size`; `block_count == ceil(num_tokens / block_size)` (zero maps to zero blocks); and `valid_tokens_last_block == 0` for zero blocks or `((num_tokens - 1) % block_size) + 1` otherwise. Transfers/checksums cover the full padded blocks, while attention uses only `num_tokens`; padding is not interpreted as valid KV.

```python
@dataclass
class KVSwapReservation:
    direction: str  # exactly "out" or "in"
    keys: tuple[KVObjectKey, ...]
    host_leases: dict[KVObjectKey, PinnedBufferLease]  # async mode only
    block_ids: dict[KVObjectKey, list[int]]
    total_nbytes: int
    submitted: bool = False


@dataclass(frozen=True)
class KVTransferCompletion:
    seq_id: int
    generation: int
    direction: str
    success: bool
    cancelled: bool
    error: str | None
    bytes_transferred: int
```

`PagedKVCache` exposes:

```python
def reserve_swap_out_group(self, seq_ids: list[int]) -> KVSwapReservation | None: ...
def submit_swap_out_group(self, reservation: KVSwapReservation) -> None: ...
def reserve_swap_in_group(self, seq_ids: list[int]) -> KVSwapReservation | None: ...
def submit_swap_in_group(self, reservation: KVSwapReservation) -> None: ...
def request_swap_out(self, seq_id: int) -> bool: ...  # one-member wrapper
def request_swap_in(self, seq_id: int) -> bool: ...  # one-member wrapper
def poll_transfers(self) -> list[KVTransferCompletion]: ...
def transfer_state(self, seq_id: int) -> KVTransferState: ...
def is_gpu_ready(self, seq_id: int) -> bool: ...
def discard_failed_for_reprefill(self, seq_id: int, generation: int) -> None: ...
def discard_host_copy(self, seq_id: int, generation: int) -> None: ...
def free_sequence(self, seq_id: int) -> None: ...
def cancel_sequence(self, seq_id: int) -> None: ...
def has_pending_transfers(self) -> bool: ...
def wait_for_transfer_progress(self, timeout_ms: float) -> bool: ...
def shutdown(self, timeout_ms: float = 5000.0) -> None: ...
def get_swap_stats(self) -> dict[str, object]: ...
```

For direction `"out"`, an **async** `KVSwapReservation` owns newly acquired host leases and rollback releases them; for `"in"`, it references existing async host leases but owns newly reserved destination block IDs, so rollback frees only those blocks. Reject double submission by checking `submitted`. Sync never puts a pageable record in `KVSwapReservation.host_leases`: its reservation validates/reserves GPU ownership only, and synchronous submission creates or consumes `PageableCPUBufferRecord` directly. A sequence record must own exactly one of `pageable_buffer` and `host_lease`, selected once from the effective mode; mixed ownership is an invariant violation.

## File structure

| File | Responsibility |
| --- | --- |
| `moe_infinity/engine/kv_transfer.py` (create) | Shared metadata, state enums, sync-only pageable record, async-only pinned pool/lease, sync/CUDA/fake backend contracts, validation, CRC32, future external-store protocol. |
| `moe_infinity/serving/kv_cache.py` | Per-generation records, block ownership, host-resident leases, submit/poll/cancel/finalize APIs, compatibility wrappers. |
| `moe_infinity/serving/scheduler.py` | Poll completions, delay block reuse/readiness, apply backpressure, retry swap-in, reprefill on terminal restore failure. |
| `moe_infinity/serving/engine.py` | Consume all six validated config fields, construct the selected backend/pool, expose stats, avoid false “no progress,” and drain on shutdown/reload. |
| `moe_infinity/engine/kv_cache_offload_coordinator.py` | Reuse shared backend for native stacked/tuple tensors; remove internal stream synchronization and silent missing-data returns. |
| `moe_infinity/engine/unified_transfer_scheduler.py` | Preserve queue role, accept handler byte counts/error text, report real bytes and failures; do not own CUDA event polling. |
| `moe_infinity/engine/transfer_types.py` | Structured `TransferResult` status/error/bytes without changing transfer priority values. |
| `moe_infinity/utils/config.py` | Source defaults and validation for async mode, host cap, in-flight cap, checksum, retries, and fallback. |
| `moe_infinity/entrypoints/big_modeling.py` | Pass `ArcherConfig` values through `MoE.serve()` and install the native coordinator against real paged KV tensors. |
| `moe_infinity/entrypoints/openai/api_server_v2.py` | CLI/programmatic config propagation, engine config construction, Prometheus metrics, and shutdown drain. |
| `tests/python/unit/test_kv_transfer.py` (create) | Pool, metadata, fake events, state-independent transfer primitives. |
| `tests/python/serving/test_async_kv_swap.py` (create) | CPU fake-backend lifecycle, ownership, readiness, backpressure, cancellation, retry, ABA tests. |
| `tests/python/integration/test_async_kv_swap_cuda.py` (create) | Real pinned-memory/CUDA-stream equivalence and race tests. |
| `tests/python/unit/test_kv_handler_registration.py` | Native coordinator compatibility and explicit failures. |
| `tests/python/unit/test_unified_scheduler.py` | Real byte/failure metrics and no queue regression. |
| `tests/python/serving/test_cancellation.py` | Engine cancellation with in-flight transfers and leak checks. |
| `benchmarks/serving/kv_offload_benchmark.py` | Sync/async A/B, warmup/trials, swap p50/p95/p99, backpressure, bytes, overlap accounting, JSON output. |
| `docs/configuration.md`, `docs/serving.md`, `docs/benchmarking.md`, `CHANGELOG.md` | Operator contract, caveats, metrics, runbook, rollout. |

### Task 1: Add transfer metadata, bounded pinned pooling, and backend interfaces

**Files:**
- Create: `moe_infinity/engine/kv_transfer.py`
- Create: `tests/python/unit/test_kv_transfer.py`

- [ ] **Step 1: Write failing tests for pool accounting and metadata validation**

```python
import torch

from moe_infinity.engine.kv_transfer import (
    CopyTicket,
    KV_FORMAT_VERSION,
    KVObjectKey,
    KVObjectMetadata,
    PageableCPUBufferRecord,
    PinnedBufferPool,
    validate_metadata,
)


def test_pool_enforces_cap_and_reuses_exact_shape() -> None:
    allocations: list[torch.Tensor] = []

    def allocate(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype)
        allocations.append(tensor)
        return tensor

    pool = PinnedBufferPool(capacity_bytes=64, allocator=allocate)
    first = pool.acquire((8,), torch.float32)
    second = pool.acquire((8,), torch.float32)
    third = pool.acquire((1,), torch.float32)
    assert first is not None and second is not None
    assert third is None
    assert pool.in_use_bytes == 64
    pool.release(first)
    reused = pool.acquire((8,), torch.float32)
    assert reused is not None
    assert reused.lease_id != first.lease_id
    assert len(allocations) == 2


def test_pageable_record_owns_and_releases_blocking_clone() -> None:
    source = torch.arange(8, dtype=torch.float32)
    record = PageableCPUBufferRecord.from_blocking_clone(source)
    owned = record.require_tensor()
    assert owned.device.type == "cpu"
    assert not owned.is_pinned()
    assert owned.data_ptr() != source.data_ptr()
    torch.testing.assert_close(owned, source)
    assert record.nbytes == source.numel() * source.element_size()
    record.release()
    assert record.tensor is None
    try:
        record.require_tensor()
    except RuntimeError as exc:
        assert "released" in str(exc)
    else:
        raise AssertionError("released pageable record must not expose a tensor")


def make_metadata(**updates: object) -> tuple[KVObjectMetadata, torch.Tensor]:
    payload = torch.zeros((1, 2, 2, 4, 1, 8), dtype=torch.float16)
    values: dict[str, object] = {
        "format_version": KV_FORMAT_VERSION,
        "key": KVObjectKey(seq_id=3, generation=11),
        "shape": tuple(payload.shape),
        "dtype": payload.dtype,
        "nbytes": payload.numel() * payload.element_size(),
        "num_tokens": 7,
        "block_count": 2,
        "block_size": 4,
        "valid_tokens_last_block": 3,
        "checksum_crc32": None,
    }
    values.update(updates)
    return KVObjectMetadata(**values), payload


def assert_invalid(field: str, **updates: object) -> None:
    metadata, payload = make_metadata(**updates)
    try:
        validate_metadata(
            metadata,
            payload,
            expected_key=KVObjectKey(3, 11),
            block_ids=[1, 4],
            total_blocks=8,
        )
    except ValueError as exc:
        assert field in str(exc)
    else:
        raise AssertionError(f"{field} mismatch must be rejected")


def test_metadata_rejects_shape_independently() -> None:
    assert_invalid("shape", shape=(1, 3, 2, 4, 1, 8))


def test_metadata_rejects_dtype_independently() -> None:
    assert_invalid("dtype", dtype=torch.bfloat16)


def test_metadata_rejects_byte_length_independently() -> None:
    assert_invalid("nbytes", nbytes=255)


def test_metadata_rejects_block_count_independently() -> None:
    assert_invalid("block_count", block_count=3)


def test_metadata_rejects_generation_independently() -> None:
    assert_invalid("generation", key=KVObjectKey(3, 12))


def test_metadata_rejects_block_bounds_and_duplicates() -> None:
    metadata, payload = make_metadata()
    for block_ids in ([-1, 4], [1, 8], [1, 1]):
        try:
            validate_metadata(
                metadata,
                payload,
                expected_key=KVObjectKey(3, 11),
                block_ids=block_ids,
                total_blocks=8,
            )
        except ValueError as exc:
            assert "block_ids" in str(exc)
        else:
            raise AssertionError("invalid block IDs must be rejected")


def test_metadata_validates_padding_and_token_bounds() -> None:
    assert_invalid("num_tokens", num_tokens=9)
    assert_invalid("block_count", num_tokens=4, block_count=2)
    assert_invalid("valid_tokens_last_block", valid_tokens_last_block=4)
    metadata, payload = make_metadata()
    validate_metadata(
        metadata,
        payload,
        expected_key=KVObjectKey(3, 11),
        block_ids=[1, 4],
        total_blocks=8,
    )


class ManualEvent:
    def __init__(self, done: bool) -> None:
        self.done = done

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.done = True


def test_copy_ticket_cannot_retire_before_completion() -> None:
    event = ManualEvent(done=False)
    staging = torch.ones(4)
    ticket = CopyTicket(
        device=torch.device("cpu"),
        stream=None,
        event=event,
        owned_staging_tensors=(staging,),
        submitted_ns=1,
        nbytes=staging.numel() * staging.element_size(),
    )
    assert ticket.retire() is False
    assert ticket.owned_staging_tensors[0] is staging
    event.done = True
    assert ticket.retire() is True
    assert ticket.retired
    assert ticket.owned_staging_tensors == ()
```

- [ ] **Step 2: Run the unit test and verify RED**

Run: `python -m pytest -q tests/python/unit/test_kv_transfer.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'moe_infinity.engine.kv_transfer'`.

- [ ] **Step 3: Implement the module contract**

Implement the exact API shown above plus:

```python
class PinnedBufferPool:
    def __init__(self, capacity_bytes: int, allocator=None) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be > 0")
        self.capacity_bytes = capacity_bytes
        self._allocator = allocator or (
            lambda shape, dtype: torch.empty(shape, dtype=dtype, pin_memory=True)
        )
        self._cached: dict[tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]] = {}
        self._leased: dict[int, PinnedBufferLease] = {}
        self._allocated_bytes = 0
        self._next_lease_id = 1
        self.peak_in_use_bytes = 0
        self.backpressure_total = 0

    @property
    def in_use_bytes(self) -> int:
        return sum(lease.nbytes for lease in self._leased.values())

    def acquire(self, shape, dtype) -> PinnedBufferLease | None:
        key = (tuple(shape), dtype)
        nbytes = int(torch.empty((), dtype=dtype).element_size())
        for dimension in shape:
            nbytes *= int(dimension)
        cached = self._cached.get(key, [])
        if cached:
            tensor = cached.pop()
        elif self._allocated_bytes + nbytes <= self.capacity_bytes:
            tensor = self._allocator(tuple(shape), dtype)
            self._allocated_bytes += nbytes
        else:
            self.backpressure_total += 1
            return None
        lease = PinnedBufferLease(tensor, nbytes, self._next_lease_id)
        self._next_lease_id += 1
        self._leased[lease.lease_id] = lease
        self.peak_in_use_bytes = max(self.peak_in_use_bytes, self.in_use_bytes)
        return lease

    def release(self, lease: PinnedBufferLease) -> None:
        owned = self._leased.pop(lease.lease_id, None)
        if owned is None:
            raise ValueError(f"unknown or released lease {lease.lease_id}")
        key = (tuple(owned.tensor.shape), owned.tensor.dtype)
        self._cached.setdefault(key, []).append(owned.tensor)
```

Implement `PageableCPUBufferRecord` exactly as shown in the public contract, plus `SyncKVTransferBackend`, `CudaKVTransferBackend`, and `validate_metadata()`. CUDA D2H must allocate no host tensor: first call `transfer_stream.wait_stream(torch.cuda.current_stream(device))` so prior KV writes happen-before the gather; then use `index_select(1, block_ids)` to create a contiguous GPU gather staging tensor, call `destination.copy_(staging, non_blocking=True)`, record a CUDA event, retain the staging tensor in `CopyTicket`, and call `staging.record_stream(stream)`. H2D copies pinned host bytes into a contiguous GPU staging tensor and then calls base-cache `index_copy_(1, block_ids_tensor, staging)` on the same transfer stream before recording the event; do not pass an advanced-indexing result as a destination because it is not a writable view of the base cache. `close()` synchronizes only during shutdown.

`SyncKVTransferBackend` performs only blocking tensor copy operations and immediate-ticket signaling; it must not import, instantiate, retain, acquire, or release `PinnedBufferPool` or `PinnedBufferLease`. The cache owns sync allocation through `PageableCPUBufferRecord.from_blocking_clone()` so the exact existing `.detach().to("cpu").clone()` behavior remains visible and testable. Sync H2D reads `record.require_tensor()`, performs `.to(device=..., dtype=...)` followed by base-cache assignment, and returns only after assignment completes. `PageableCPUBufferRecord.release()` is the sole sync host-release operation and only drops the owned tensor reference; it has no pool callback or byte-budget mutation.

Implement each `validate_metadata()` check as a separate branch with the field name in its `ValueError`, in the order format/key, shape, dtype, bytes, block count/size, block-ID bounds/uniqueness, token bounds, and final-block padding. This ordering makes every independent test above diagnose one invariant instead of being masked by another.

- [ ] **Step 4: Run transfer primitive tests and verify GREEN**

Run: `python -m pytest -q tests/python/unit/test_kv_transfer.py`

Expected: all tests pass on CPU; no CUDA test is collected in this file.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/engine/kv_transfer.py tests/python/unit/test_kv_transfer.py
git commit -m "feat: add bounded KV transfer primitives"
```

### Task 2: Add per-generation transfer records and block ownership to serving KV cache

**Files:**
- Modify: `moe_infinity/serving/kv_cache.py:48-379`
- Create: `tests/python/serving/test_async_kv_swap.py`

- [ ] **Step 1: Write fake-backend tests for delayed ownership and readiness**

```python
import torch

from moe_infinity.engine.kv_transfer import (
    CopyTicket,
    KVTransferState,
    SyncKVTransferBackend,
)
from moe_infinity.serving.kv_cache import PagedKVCache


class FakeEvent:
    def __init__(self) -> None:
        self.done = False

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.done = True


class FakeBackend:
    asynchronous = True

    def __init__(self) -> None:
        self.events: list[FakeEvent] = []

    def submit_d2h(
        self,
        source_cache: torch.Tensor,
        destination: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        source = source_cache.index_select(
            block_dim, torch.tensor(block_ids, dtype=torch.long)
        )
        destination.copy_(source)
        event = FakeEvent()
        self.events.append(event)
        return CopyTicket(
            device=source.device,
            stream=None,
            event=event,
            owned_staging_tensors=(),
            submitted_ns=1,
            nbytes=source.numel() * source.element_size(),
        )

    def submit_h2d(
        self,
        source: torch.Tensor,
        destination_cache: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        destination_cache.index_copy_(
            block_dim,
            torch.tensor(block_ids, dtype=torch.long),
            source,
        )
        event = FakeEvent()
        self.events.append(event)
        return CopyTicket(
            device=destination_cache.device,
            stream=None,
            event=event,
            owned_staging_tensors=(),
            submitted_ns=1,
            nbytes=source.numel() * source.element_size(),
        )

    def close(self) -> None:
        return None


def make_cache(backend: FakeBackend) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=2,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
        transfer_backend=backend,
        host_pool_bytes=1024,
    )


def test_swap_out_holds_blocks_until_event_completion() -> None:
    backend = FakeBackend()
    cache = make_cache(backend)
    cache.allocate_sequence(7, num_tokens=8)
    assert cache.request_swap_out(7)
    assert cache.transfer_state(7) is KVTransferState.SWAP_OUT_IN_FLIGHT
    assert cache.block_allocator.num_free_blocks == 0
    assert cache.poll_transfers() == []
    backend.events[0].done = True
    completion = cache.poll_transfers()
    assert completion[0].success
    assert cache.transfer_state(7) is KVTransferState.HOST_RESIDENT
    assert cache.block_allocator.num_free_blocks == 2
    assert cache.get_block_table(7) == []


def test_swap_in_publishes_blocks_only_after_event() -> None:
    backend = FakeBackend()
    cache = make_cache(backend)
    cache.allocate_sequence(9, num_tokens=8)
    assert cache.request_swap_out(9)
    backend.events[-1].done = True
    cache.poll_transfers()
    assert cache.request_swap_in(9)
    assert cache.get_block_table(9) == []
    assert not cache.is_gpu_ready(9)
    backend.events[-1].done = True
    cache.poll_transfers()
    assert cache.is_gpu_ready(9)
    assert len(cache.get_block_table(9)) == 2


def test_sync_round_trip_is_blocking_pageable_and_never_allocates_pool() -> None:
    pool_factory_calls: list[int] = []

    def forbidden_pool_factory(capacity_bytes: int):
        pool_factory_calls.append(capacity_bytes)
        raise AssertionError("sync mode must not construct a pinned pool")

    cache = PagedKVCache(
        num_blocks=2,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        transfer_backend=SyncKVTransferBackend(),
        host_pool_bytes=1,
        host_pool_factory=forbidden_pool_factory,
    )
    cache.allocate_sequence(12, num_tokens=8)
    original_ids = cache.get_block_table(12)
    expected = torch.arange(32, dtype=torch.float32).reshape(1, 2, 2, 4, 1, 2)
    cache.get_kv_cache_tensors()[:, original_ids, ...].copy_(expected)

    cache.swap_out(12)

    record = cache._kv_records[12]
    assert record.state is KVTransferState.HOST_RESIDENT
    assert record.host_lease is None
    assert record.pageable_buffer is not None
    assert not record.pageable_buffer.require_tensor().is_pinned()
    assert pool_factory_calls == []
    assert cache.get_swap_stats()["host_capacity_bytes"] == 0
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0

    cache.free_gpu_blocks(12)
    cache.get_kv_cache_tensors().fill_(-1)
    cache.swap_in(12)

    restored_ids = cache.get_block_table(12)
    torch.testing.assert_close(
        cache.get_kv_cache_tensors()[:, restored_ids, ...], expected
    )
    assert record.pageable_buffer is None
    assert cache.is_gpu_ready(12)
    assert pool_factory_calls == []


def test_sync_host_resident_free_releases_pageable_record_without_pool() -> None:
    cache = PagedKVCache(
        num_blocks=1,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        transfer_backend=SyncKVTransferBackend(),
        host_pool_bytes=1,
        host_pool_factory=lambda _capacity: (_ for _ in ()).throw(
            AssertionError("sync mode must not construct a pinned pool")
        ),
    )
    cache.allocate_sequence(13, num_tokens=4)
    cache.swap_out(13)
    pageable = cache._kv_records[13].pageable_buffer
    assert pageable is not None

    cache.free_sequence(13)

    assert pageable.tensor is None
    assert cache.block_allocator.num_free_blocks == 1
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0
```

- [ ] **Step 2: Run the serving test and verify RED**

Run: `python -m pytest -q tests/python/serving/test_async_kv_swap.py`

Expected: fails because `PagedKVCache.__init__` does not accept `transfer_backend` and the async methods do not exist.

- [ ] **Step 3: Implement records, generations, and ownership transitions**

Add `seq_id` to `BlockTable`, add optional `owner_seq_id`/`ownership` tracking to `BlockAllocator`, and preserve `BlockAllocator.allocate(num_blocks)` compatibility by defaulting `owner_seq_id=None`. Add internal records:

```python
@dataclass
class _SequenceKVRecord:
    key: KVObjectKey
    state: KVTransferState
    num_tokens: int
    active_block_ids: list[int]
    restoring_block_ids: list[int]
    pageable_buffer: PageableCPUBufferRecord | None
    host_lease: PinnedBufferLease | None
    metadata: KVObjectMetadata | None
    ticket: CopyTicket | None
    retries: int = 0
    cancelled: bool = False
    retire_requested: bool = False
    last_error: str | None = None
```

Replace `_swapped_cpu_buffers`, `_swapped_num_tokens`, and `_swapped_out_sequences` with `_kv_records`, `_retiring_records`, `_next_generation`, an optional async-only pinned pool, and backend. Add `pinned_pool: PinnedBufferPool | None = None` and injectable `host_pool_factory: Callable[[int], PinnedBufferPool]` constructor fields. If `transfer_backend.asynchronous` is false, reject a non-`None` supplied pool, set `_pinned_pool = None`, and do not call the factory; if true, use the supplied pool or construct exactly one through the factory. Enforce `bool(record.pageable_buffer) != bool(record.host_lease)` whenever a record is host-resident: sync owns only the former, async owns only the latter.

Centralize host ownership release so no caller can accidentally send a pageable record to the pool:

```python
def _release_host_storage(self, record: _SequenceKVRecord) -> None:
    if record.pageable_buffer is not None and record.host_lease is not None:
        raise RuntimeError("KV record cannot own pageable and pinned host storage")
    if record.pageable_buffer is not None:
        record.pageable_buffer.release()
        record.pageable_buffer = None
        return
    if record.host_lease is not None:
        if self._pinned_pool is None:
            raise RuntimeError("pinned lease exists without an async pool")
        self._pinned_pool.release(record.host_lease)
        record.host_lease = None
```

Call this helper from successful sync H2D, host-resident/failed `free_sequence()`, cancellation finalization, reprefill discard, and shutdown. Only async ticket finalizers may reach its pinned branch; sync has no ticket/tombstone lifetime.

Async group reservation pre-acquires every pinned lease or destination block and checks aggregate in-flight bytes; it rolls back all reservations if any member cannot reserve. Sync group reservation never calls pool code and never applies either configured byte cap. Sync D2H submission executes `PageableCPUBufferRecord.from_blocking_clone(self._kv_cache[:, block_ids, ...])` before returning, records `HOST_RESIDENT`, and emits/consumes its immediate completion in the same public call. Sync H2D submission reads `pageable_buffer.require_tensor()`, performs the blocking `.to(device=self._kv_cache.device, dtype=self._kv_cache.dtype)` and base-cache assignment before publishing restored block IDs, calls `pageable_buffer.release()`, clears the field, and returns GPU-ready. Thus compatibility `swap_out()`/`swap_in()` retain the old return-time guarantees and require no `poll_transfers()` call in sync mode.

Async submission marks source blocks `EVICTING` or destination blocks `RESTORING`, and passes base cache plus block IDs to the backend so H2D uses `index_copy_`. `poll_transfers()` queries without synchronization, calls `ticket.retire()` before releasing any owned resource, and emits completions; H2D success publishes block tables only after retirement.

Implement `free_sequence()` exactly from the authoritative table above. For an in-flight async record, pop only the active `_sequence_tables`/`_kv_records` mapping, move the full record plus table to `_retiring_records[key]`, set `CANCEL_PENDING`, and return without calling `BlockAllocator.free()` or `PinnedBufferPool.release()`. `poll_transfers()` owns the only path that retires the ticket and then frees those resources. For a sync host-resident record, call `pageable_buffer.release()` immediately and clear it; never route this through pool release. `shutdown()` invokes the same async finalizers after synchronized retirement, releases every remaining sync pageable record directly, and only closes/releases `_pinned_pool` when it is non-`None`.

Keep compatibility wrappers:

```python
def swap_out(self, seq_id: int) -> None:
    if not self.request_swap_out(seq_id):
        raise RuntimeError(f"KV swap-out backpressure for sequence {seq_id}")
    # Sync submission has already completed and published HOST_RESIDENT.

def swap_in(self, seq_id: int) -> None:
    if not self.request_swap_in(seq_id):
        raise RuntimeError(f"KV swap-in unavailable for sequence {seq_id}")
    # Sync submission has already restored/published and released pageable bytes.
```

Do not use broad `except Exception: pass`. Convert backend exceptions into failed completion records with `type(exc).__name__` and message; retain valid source data according to the state machine.

- [ ] **Step 4: Add backpressure, failure, checksum, and ABA tests**

Add the following tests with the specified setup and assertions:

```text
test_pool_backpressure_leaves_sequence_gpu_resident
test_swap_out_failure_keeps_source_blocks_allocated
test_swap_in_failure_releases_restoring_blocks_and_keeps_host_copy
test_checksum_mismatch_never_submits_h2d
test_free_sequence_during_d2h_retains_blocks_lease_and_ticket_until_retired
test_free_sequence_during_h2d_retains_blocks_lease_and_ticket_until_retired
test_cancel_sequence_is_idempotent_while_ticket_is_pending
test_stale_generation_completion_cannot_mutate_reused_seq_id
test_free_sequence_reclaims_host_resident_lease
```

For both in-flight free tests, capture block count, lease ID, ticket, and `_retiring_records` key; assert they remain owned while `event.query()` is false, then complete the event, poll, and assert `ticket.retired`, tombstone removal, allocator restoration, and pool `in_use_bytes == 0`. Every other test asserts state, free-block count, pool bytes, and block table before and after completion.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m pytest -q tests/python/unit/test_kv_transfer.py tests/python/serving/test_async_kv_swap.py tests/python/serving/test_kv_cache.py tests/python/unit/test_kv_edge_cases.py`

Expected: all tests pass; existing allocator and synchronous cache tests remain unchanged, both new sync lifecycle tests observe blocking round-trip data before return, and the forbidden pool factory is never called.

- [ ] **Step 6: Commit**

```bash
git add moe_infinity/serving/kv_cache.py tests/python/serving/test_async_kv_swap.py
git commit -m "feat: track asynchronous serving KV residency"
```

### Task 3: Make scheduling event-driven and backpressure-safe

**Files:**
- Modify: `moe_infinity/serving/scheduler.py:158-588`
- Modify: `tests/python/unit/test_kv_swap_recovery.py`
- Modify: `tests/python/serving/test_scheduler.py`

- [ ] **Step 1: Write failing scheduler tests**

Add fake-backend tests proving:

```python
def test_preemption_does_not_reuse_evicting_blocks(fake_async_cache) -> None:
    scheduler, first, second, backend = build_pressure_case(fake_async_cache)
    scheduler.add_request(first)
    scheduler.schedule()
    scheduler.update_after_step([], [first.sequence_ids[0]])
    scheduler.add_request(second)
    output = scheduler.schedule()
    assert output.preempted_seq_ids == [first.sequence_ids[0]]
    assert output.prefill_seq_ids == []
    assert scheduler.kv_cache.block_allocator.num_free_blocks == 0
    backend.complete_next()
    output = scheduler.schedule()
    assert output.prefill_seq_ids == [second.sequence_ids[0]]


def test_swapped_sequence_not_decoded_before_h2d_completion(fake_async_cache) -> None:
    scheduler, group, backend = build_host_resident_case(fake_async_cache)
    output = scheduler.schedule()
    assert output.decode_seq_ids == []
    assert group.sequences[0].status is SequenceStatus.SWAPPED
    backend.complete_next()
    output = scheduler.schedule()
    assert output.decode_seq_ids == [group.sequence_ids[0]]
    assert group.sequences[0].status is SequenceStatus.DECODE
```

- [ ] **Step 2: Run scheduler tests and verify RED**

Run: `python -m pytest -q tests/python/unit/test_kv_swap_recovery.py tests/python/serving/test_scheduler.py`

Expected: the preemption test observes premature block reuse or decode readiness.

- [ ] **Step 3: Implement scheduler polling and readiness gates**

At the beginning of `schedule()`, call `poll_transfers()` exactly once, aggregate completions by `_SwappedGroupRecord`, and run queue transitions only after aggregation. Change preemption to an all-member reservation/submission:

```python
seqs = [
    sequence
    for sequence in group.sequences
    if sequence.status in {
        SequenceStatus.PREFILL,
        SequenceStatus.DECODE,
        SequenceStatus.DRAFT,
        SequenceStatus.VERIFY,
    }
]
reservation = self.kv_cache.reserve_swap_out_group(
    [sequence.seq_id for sequence in seqs]
)
if reservation is None:
    return []
prior = {sequence.seq_id: sequence.status for sequence in seqs}
self.kv_cache.submit_swap_out_group(reservation)
_ = self._running.popleft()
for sequence in seqs:
    sequence.set_status(SequenceStatus.SWAPPED)
self._swapped.append(group)
self._swapped_groups[group.request_id] = _SwappedGroupRecord(
    group=group,
    prior_status_by_seq=prior,
    phase=SwapGroupPhase.OUT_IN_FLIGHT,
)
preempted_seq_ids = [sequence.seq_id for sequence in seqs]
```

Never call `free_gpu_blocks()` from preemption. Implement the seven group rules above in `_advance_swapped_groups(completions)`. `_swapped_groups` is the authoritative phase map; `_swapped`, `_running`, and `_waiting` each contain a request at most once. A transfer `FAILED` record is first discarded/removed by the cache; only then does scheduler recovery transition `SequenceStatus.SWAPPED -> WAITING`. Restore the saved prior status on successful/rollback swap-in rather than forcing every sequence to `DECODE`.

Change `has_work()` to include `_swapped` and `kv_cache.has_pending_transfers()`. Add `has_runnable_work()` that returns true only when `_waiting` or GPU-ready `_running` sequences can produce a batch; use it in the engine loop so transfer-only iterations yield instead of busy-spinning. Do not report a moving request as finished.

- [ ] **Step 4: Add group atomicity and retry tests**

Add six tests: (1) reservation backpressure leaves the group once in `_running` with all prior statuses; (2) accepted preemption moves the group once from `_running` to `_swapped` and sets every member `SWAPPED`; (3) one of two D2H/H2D completions never changes queues/statuses; (4) all H2D completions atomically move `_swapped -> _running` and restore mixed prior `DECODE`/`DRAFT` statuses; (5) partial D2H failure performs rollback and publishes no member until all rollback H2D tickets retire; and (6) retry exhaustion waits for all tickets, removes transfer records, resets all sequence accounting, and atomically moves `_swapped -> _waiting` with every status `WAITING`. After every pass assert each request ID occurs in exactly one queue.

- [ ] **Step 5: Run scheduler regression tests and verify GREEN**

Run: `python -m pytest -q tests/python/unit/test_kv_swap_recovery.py tests/python/serving/test_scheduler.py tests/python/serving/test_dflash_deficit_scheduler.py`

Expected: all tests pass, including existing FCFS, verify-budget, and recovery coverage.

- [ ] **Step 6: Commit**

```bash
git add moe_infinity/serving/scheduler.py tests/python/unit/test_kv_swap_recovery.py tests/python/serving/test_scheduler.py
git commit -m "feat: gate serving scheduling on KV transfer events"
```

### Task 4: Wire configuration, synchronous fallback, progress, cancellation, and shutdown

**Files:**
- Modify: `moe_infinity/utils/config.py:45-162`
- Modify: `moe_infinity/entrypoints/big_modeling.py:915-969`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:475-526,1040-1094,1778-1943`
- Modify: `moe_infinity/serving/engine.py:84-168,594-668`
- Modify: `tests/python/unit/test_kv_config_wiring.py`
- Modify: `tests/python/serving/test_api_routes.py`
- Modify: `tests/python/serving/test_cancellation.py`
- Modify: `tests/python/serving/test_engine.py`

- [ ] **Step 1: Write failing config and engine lifecycle tests**

Test these exact defaults and validations:

```python
assert ArcherConfig().kv_swap_mode == "sync"
assert ArcherConfig().kv_swap_host_memory_bytes == 512 * 1024 * 1024
assert ArcherConfig().kv_swap_max_inflight_bytes == 256 * 1024 * 1024
assert ArcherConfig().kv_swap_checksum is False
assert ArcherConfig().kv_swap_max_retries == 2
assert ArcherConfig().kv_swap_allow_sync_fallback is True
```

Add this default-construction test to `tests/python/unit/test_kv_config_wiring.py` with `from moe_infinity.engine.kv_transfer import SyncKVTransferBackend` and `from moe_infinity.serving.engine import build_kv_transfer_resources`:

```python
def test_default_sync_does_not_construct_or_budget_a_pinned_pool() -> None:
    config = ArcherConfig()
    pool_factory_calls: list[int] = []

    def forbidden_pool_factory(capacity_bytes: int):
        pool_factory_calls.append(capacity_bytes)
        raise AssertionError("default sync must not construct a pinned pool")

    backend, pool, fallback_reason = build_kv_transfer_resources(
        config=config,
        device=torch.device("cpu"),
        pool_factory=forbidden_pool_factory,
    )

    assert config.kv_swap_mode == "sync"
    assert isinstance(backend, SyncKVTransferBackend)
    assert pool is None
    assert fallback_reason is None
    assert pool_factory_calls == []
```

Reject modes outside `{"sync", "async"}`, nonpositive byte caps, in-flight cap greater than host cap, and negative retries. These source values remain validated for deterministic config compatibility, but document and test that both byte caps are inactive when the effective mode is sync. Add an engine test where backend construction raises `RuntimeError("pin unavailable")`: fallback enabled selects sync, returns `pool is None`, records `fallback_reason`, and discards any partially constructed async pool; fallback disabled raises the same error.

Add a propagation test with six non-default sentinel values on `moe_model.engine_config`, call `MoE.serve()` without swap overrides, and assert the same values at every boundary: `ArcherConfig` → resolved `MoE.serve()` values → `initialize_with_model()` namespace → `_build_engine_config()` dictionary → `ContinuousBatchingEngine.config` → `PagedKVCache`/backend/pool fields. Add an explicit-override test and a CLI parser test for `--kv-swap-mode`, `--kv-swap-host-memory-bytes`, `--kv-swap-max-inflight-bytes`, `--kv-swap-checksum`, `--kv-swap-max-retries`, and `--no-kv-swap-sync-fallback`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/python/unit/test_kv_config_wiring.py tests/python/serving/test_api_routes.py tests/python/serving/test_engine.py tests/python/serving/test_cancellation.py`

Expected: fails because the new config fields and engine shutdown/progress behavior are absent.

- [ ] **Step 3: Wire backend selection and no-progress handling**

Wire all six fields through both server paths:

```python
swap_config = {
    "kv_swap_mode": args.kv_swap_mode,
    "kv_swap_host_memory_bytes": args.kv_swap_host_memory_bytes,
    "kv_swap_max_inflight_bytes": args.kv_swap_max_inflight_bytes,
    "kv_swap_checksum": args.kv_swap_checksum,
    "kv_swap_max_retries": args.kv_swap_max_retries,
    "kv_swap_allow_sync_fallback": args.kv_swap_allow_sync_fallback,
}
config.update(swap_config)
```

Extend `MoE.serve()` with optional swap overrides defaulting to `None`; resolve each `None` from `self.engine_config` so the `ArcherConfig` defaults/non-defaults are retained. Extend `initialize_with_model()` with concrete resolved values; extend CLI `parse_args()` with the six source defaults; include them in CLI `moe_config` and `_build_engine_config()`; parse them in `ContinuousBatchingEngine.__init__`; and pass the resolved mode/host/in-flight/checksum/retry values into `PagedKVCache`.

Implement one shared `build_kv_transfer_resources(config, device, *, pool_factory=PinnedBufferPool)` factory returning `(backend, pool_or_none, fallback_reason)`. Its first branch must be `if config.kv_swap_mode == "sync": return SyncKVTransferBackend(), None, None`; it must not evaluate `pool_factory`, probe pinned allocation, or reserve/account either byte cap on this branch. Only the async branch calls `pool_factory(config.kv_swap_host_memory_bytes)`, constructs `CudaKVTransferBackend`, and supplies the returned pool to `PagedKVCache`; therefore async reservations use `PinnedBufferLease` exclusively. For async initialization failure, close/discard a partially created pool, choose `SyncKVTransferBackend(), None, fallback_reason` only when fallback is enabled, and otherwise raise before requests are admitted. Pass `pool_or_none` into `PagedKVCache` rather than allowing the cache to silently create a pool in sync mode.

Modify `run_until_done()` so an empty scheduler output is not an error while `kv_cache.has_pending_transfers()` is true:

```python
if self.kv_cache.has_pending_transfers():
    progressed = self.kv_cache.wait_for_transfer_progress(timeout_ms=100.0)
    if progressed:
        continue
```

Retain the existing no-progress `RuntimeError` when there is neither runnable work nor a pending transfer. Add `shutdown()` that aborts pending requests, calls `kv_cache.shutdown()`, and is idempotent. `timeout_ms` is a warning threshold, not permission to release DMA-owned memory: after the deadline, log the outstanding count and synchronize the backend before reclaiming resources. Module reload does not serialize live KV; operational shutdown/model replacement must drain the old engine before replacing it.

- [ ] **Step 4: Implement cancellation tests for both transfer directions**

Cancel during D2H and H2D with an incomplete fake event. Immediately assert request metadata is removed from the engine but block/pool resources remain held; complete the event, call `poll_transfers()`, and assert all blocks and leases are reclaimed. Call `shutdown()` twice and assert no exception or leak.

- [ ] **Step 5: Run lifecycle tests and verify GREEN**

Run: `python -m pytest -q tests/python/unit/test_kv_config_wiring.py tests/python/serving/test_api_routes.py tests/python/serving/test_engine.py tests/python/serving/test_cancellation.py`

Expected: all tests pass; cancellation remains a no-op for unknown request IDs; the default-config test returns sync with `pool is None` and zero pool-factory calls; the two sync cache lifecycle tests prove data is copied/restored before each call returns and pageable ownership is released without pool accounting.

- [ ] **Step 6: Commit**

```bash
git add moe_infinity/utils/config.py moe_infinity/entrypoints/big_modeling.py moe_infinity/entrypoints/openai/api_server_v2.py moe_infinity/serving/engine.py tests/python/unit/test_kv_config_wiring.py tests/python/serving/test_api_routes.py tests/python/serving/test_engine.py tests/python/serving/test_cancellation.py
git commit -m "feat: wire async KV lifecycle and fallback"
```

### Task 5: Reuse transfer primitives in the native coordinator and preserve unified scheduling

**Files:**
- Modify: `moe_infinity/entrypoints/big_modeling.py:401-470`
- Modify: `moe_infinity/engine/kv_cache_offload_coordinator.py:1-149`
- Modify: `moe_infinity/engine/transfer_types.py:21-36`
- Modify: `moe_infinity/engine/unified_transfer_scheduler.py:64-230`
- Modify: `tests/python/unit/test_kv_handler_registration.py`
- Modify: `tests/python/unit/test_unified_scheduler.py`
- Modify: `tests/python/integration/test_swap_scheduling.py`
- Modify: `tests/python/integration/test_flashinfer_offload_wiring.py`

- [ ] **Step 1: Write failing coordinator and result tests**

Require missing KV tensors/cache entries to produce explicit failed `TransferResult`, not silent success. Require bytes to equal tensor bytes rather than `len(block_ids)`:

```python
result = scheduler.get_result(transfer_id)
assert result is not None
assert result.status == "COMPLETED"
assert result.bytes_transferred == selected.numel() * selected.element_size()
assert result.error is None
```

Add a handler failure test asserting `status == "FAILED"`, `error == "RuntimeError: missing host KV for transfer missing"`, and `failures == 1` in metrics.

Add a pending-cancellation test asserting `status == "CANCELLED"`, `bytes_transferred == 0`, `cancelled == 1`, and that the skipped handler cannot later increment `count`, `bytes`, or `failures`. Inspect scheduler pending/results and assert no value is a `CopyTicket` and no result exposes a CUDA event/stream.

Add an installation test that enables native KV offload, constructs a fake `PagedAttentionBackend` with concrete `k_cache`/`v_cache`, and asserts the coordinator stores that tuple before handler registration. Add a negative test that enabled offload with no attention backend raises `RuntimeError("KV offload requires initialized paged KV tensors")` instead of registering no-op handlers.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/python/unit/test_kv_handler_registration.py tests/python/unit/test_unified_scheduler.py tests/python/integration/test_swap_scheduling.py`

Expected: byte accounting is block-count based and handler exceptions lose their error text.

- [ ] **Step 3: Refactor coordinator and scheduler result accounting**

Install the native coordinator only after `PagedAttentionBackend` construction:

```python
if enable_kv_cache_offload:
    if attention_backend is None:
        raise RuntimeError("KV offload requires initialized paged KV tensors")
    kv_offload_coordinator = KVCacheOffloadCoordinator(
        kv_tensors=(attention_backend.k_cache, attention_backend.v_cache),
        block_pool=kv_cache_manager,
        config=engine_config,
        transfer_backend=native_kv_transfer_backend,
    )
    kv_offload_coordinator.register_with_scheduler(transfer_scheduler)
```

Create `native_kv_transfer_backend` with the same validated factory used by serving, using `attention_backend.k_cache.device`; pass the six `ArcherConfig` fields directly. Because native `engine/scheduler.py` still waits for handler completion, the coordinator uses the backend asynchronously internally but synchronizes/retires before its handler returns.

Remove the `kv_tensors=None` no-op installation. Change coordinator handlers to return exact completed payload bytes: stacked layout uses `selected.numel() * selected.element_size()`; tuple layout submits/owns one K ticket and one V ticket and sums each tensor's bytes independently. The coordinator—not `UnifiedTransferScheduler`—owns every `CopyTicket`: it waits/synchronizes all tickets at the native synchronous-handler boundary, retires all of them, commits/removes host data, and only then returns bytes. The scheduler never stores, queries, waits on, retires, or destroys CUDA events/streams.

Extend `TransferResult` with `bytes_transferred: int = 0` and `error: str | None = None`. In `_run_request`, set `COMPLETED` and increment `count`/`bytes` only from the handler's returned byte count; on exception set `FAILED`, preserve `bytes_transferred=0`, increment `count` and `failures`, store `"TypeName: message"`, and signal the waiter. Pending cancellation increments `cancelled` once, records zero bytes, and never later increments completed bytes when the queued item is skipped.

Do not make `UnifiedTransferScheduler` poll CUDA events and do not change priority enum values. Replace broad exception swallowing with stored error text; keep worker liveness after a failed request. Metrics `bytes` means payload bytes, never block count.

- [ ] **Step 4: Run native-path regression tests and verify GREEN**

Run: `python -m pytest -q tests/python/unit/test_kv_handler_registration.py tests/python/unit/test_unified_scheduler.py tests/python/integration/test_swap_scheduling.py tests/python/integration/test_expert_kv_integration.py tests/python/integration/test_flashinfer_offload_wiring.py`

Expected: all tests pass; expert and KV requests still interleave and cancellation remains observable.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/entrypoints/big_modeling.py moe_infinity/engine/kv_cache_offload_coordinator.py moe_infinity/engine/transfer_types.py moe_infinity/engine/unified_transfer_scheduler.py tests/python/unit/test_kv_handler_registration.py tests/python/unit/test_unified_scheduler.py tests/python/integration/test_swap_scheduling.py tests/python/integration/test_flashinfer_offload_wiring.py
git commit -m "refactor: share KV transfer backend across engines"
```

### Task 6: Add real CUDA equivalence and race coverage

**Files:**
- Create: `tests/python/integration/test_async_kv_swap_cuda.py`

- [ ] **Step 1: Write CUDA-gated correctness tests**

Mark every test `@pytest.mark.gpu` and skip when CUDA is unavailable. Cover FP16 and BF16. Fill only allocated blocks with deterministic values, save a GPU reference, swap out, poll until event completion with a bounded 5-second deadline, overwrite freed GPU slots, swap in to newly allocated IDs, and use `torch.testing.assert_close(restored, reference, rtol=0, atol=0)`.

Assert the host lease tensor is CPU and `is_pinned()`. Assert the API call returns before event completion by launching `torch.cuda._sleep()` on the transfer stream when available; if `_sleep` is unavailable, skip that single nonblocking assertion rather than using a flaky wall-clock threshold.

- [ ] **Step 2: Add CUDA race tests**

Add:

```text
test_cancel_during_d2h_does_not_reuse_source_blocks_early
test_cancel_during_h2d_does_not_publish_partial_block_table
test_ticket_consumer_stream_waits_for_restore_event_before_read
test_ticket_retire_keeps_staging_alive_until_event_completion
test_shutdown_synchronizes_then_retires_before_releasing_pool_or_blocks
test_repeated_swap_cancel_cycles_have_no_block_or_pinned_leak
```

For the consumer test, submit H2D, call `ticket.wait_on_consumer(torch.cuda.current_stream())`, launch a dependent checksum on that consumer stream, and verify restored bytes without host synchronization before the checksum. For retirement, hold a weak reference to the gather staging tensor, assert `retire()` returns false and the reference remains live before completion, then synchronize/retire and assert `retired` plus released staging ownership. Instrument shutdown callbacks and assert the order `event.synchronize`, `ticket.retire`, cache resource finalization, backend stream close, pool release.

- [ ] **Step 3: Run CUDA tests and verify RED, then GREEN after minimal fixes**

Run: `python -m pytest -q -m gpu tests/python/integration/test_async_kv_swap_cuda.py`

Expected before fixes: at least one readiness/event-order assertion fails. After fixes: all tests pass on a CUDA host; on CPU-only hosts pytest reports all tests skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/python/integration/test_async_kv_swap_cuda.py moe_infinity/serving/kv_cache.py
git commit -m "test: cover CUDA KV transfer ordering and races"
```

### Task 7: Add telemetry and operator-visible failure signals

**Files:**
- Modify: `moe_infinity/serving/engine.py:653-668`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:716-745`
- Modify: `tests/python/serving/test_api_routes.py`

- [ ] **Step 1: Write failing stats and Prometheus tests**

Require `get_stats()["kv_swap"]` to contain:

```text
mode, fallback_reason, host_capacity_bytes, host_in_use_bytes,
host_peak_in_use_bytes, inflight, inflight_bytes, retiring_records,
host_resident, backpressure_total,
swap_out_started_total, swap_out_completed_total, swap_out_failed_total,
swap_in_started_total, swap_in_completed_total, swap_in_failed_total,
cancelled_total, checksum_failures_total, d2h_bytes_total, h2d_bytes_total,
d2h_duration_ms_sum, h2d_duration_ms_sum
```

Require Prometheus names:

```text
moe_kv_swap_inflight
moe_kv_swap_inflight_bytes
moe_kv_swap_retiring_records
moe_kv_swap_host_resident
moe_kv_swap_host_bytes
moe_kv_swap_host_capacity_bytes
moe_kv_swap_backpressure_total
moe_kv_swap_out_completed_total
moe_kv_swap_in_completed_total
moe_kv_swap_failures_total{direction="out"}
moe_kv_swap_failures_total{direction="in"}
moe_kv_swap_bytes_total{direction="d2h"}
moe_kv_swap_bytes_total{direction="h2d"}
moe_kv_swap_duration_seconds_sum{direction="d2h"}
moe_kv_swap_duration_seconds_sum{direction="h2d"}
```

- [ ] **Step 2: Run API tests and verify RED**

Run: `python -m pytest -q tests/python/serving/test_api_routes.py`

Expected: swap metrics are missing.

- [ ] **Step 3: Implement stats and exposition**

Counters are monotonic; gauges reflect current records/pool bytes. In effective sync mode, `host_capacity_bytes`, `host_in_use_bytes`, `host_peak_in_use_bytes`, `inflight`, and `inflight_bytes` are all zero because pageable records are deliberately outside the async pinned-pool budget; sync payload byte counters may still report completed blocking copies. Duration is measured from monotonic submission to event observation and documented as observed completion latency, not pure PCIe time. Do not include sequence IDs, request IDs, token content, or checksums in metrics. Emit a warning with direction/state/error and generation on failures, and one warning when initialization falls back to sync.

- [ ] **Step 4: Run API tests and verify GREEN**

Run: `python -m pytest -q tests/python/serving/test_api_routes.py tests/python/serving/test_engine.py`

Expected: all tests pass and existing metric names remain present.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/serving/engine.py moe_infinity/entrypoints/openai/api_server_v2.py tests/python/serving/test_api_routes.py
git commit -m "feat: expose asynchronous KV swap telemetry"
```

### Task 8: Upgrade the KV benchmark for p99 and A/B evidence

**Files:**
- Modify: `benchmarks/serving/kv_offload_benchmark.py:20-412`
- Create: `tests/python/unit/test_kv_offload_benchmark.py`

- [ ] **Step 1: Write failing parser/statistics tests**

Test `percentiles([1.0, 2.0, 3.0, 4.0]) == {"p50": 2.5, "p95": 4.0, "p99": 4.0}` using the documented nearest-rank/median rules. Require CLI options `--kv-swap-mode {sync,async}`, `--warmup-requests`, `--trials`, `--host-memory-mib`, `--max-inflight-mib`, `--checksum`, `--max-retries`, and `--no-sync-fallback`.

- [ ] **Step 2: Run benchmark unit tests and verify RED**

Run: `python -m pytest -q tests/python/unit/test_kv_offload_benchmark.py`

Expected: parser options and percentile helper are absent.

- [ ] **Step 3: Implement benchmark measurements**

Run warmup outside measurement. Pass every benchmark option into the corresponding `ContinuousBatchingEngine` config key, including byte conversion for MiB values. For every trial record raw swap-out and swap-in observed durations, end-to-end request latency, generated tokens, backpressure count, D2H/H2D bytes, pinned peak bytes, GPU peak bytes, and failures. JSON contains environment, git commit, all six resolved swap config values, raw samples, p50/p95/p99, and status. Emit one self-contained result per mode. Compare the two JSON files offline; label any difference observed overlap rather than guaranteed speedup.

Do not assert performance thresholds in pytest. The benchmark exits nonzero on correctness mismatch, transfer failure, checksum failure, or leaked block/lease accounting.

- [ ] **Step 4: Run benchmark unit tests and verify GREEN**

Run: `python -m pytest -q tests/python/unit/test_kv_offload_benchmark.py`

Expected: all parser/statistics/schema tests pass.

- [ ] **Step 5: Run the CUDA A/B benchmark**

Run:

```bash
python benchmarks/serving/kv_offload_benchmark.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /path/to/offload/dir \
  --num-requests 64 \
  --prompt-length 2048 \
  --max-new-tokens 128 \
  --warmup-requests 8 \
  --trials 5 \
  --host-memory-mib 2048 \
  --max-inflight-mib 1024 \
  --kv-swap-mode async \
  --output-json /tmp/kv-swap-async.json
```

Repeat with `--kv-swap-mode sync --output-json /tmp/kv-swap-sync.json`. Expected: both runs report `status=PASS`, exact output equivalence, zero leaks/failures, and measured swap p50/p95/p99 values. Record results without promising an improvement.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/serving/kv_offload_benchmark.py tests/python/unit/test_kv_offload_benchmark.py
git commit -m "bench: measure async KV swap tail latency"
```

### Task 9: Document operation, recovery, extension boundary, and rollout

**Files:**
- Modify: `docs/configuration.md`
- Modify: `docs/serving.md`
- Modify: `docs/benchmarking.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Document the configuration contract**

Add the six exact config keys/defaults from Task 4, validation constraints, pinned-memory accounting, sync fallback behavior, and an example with `kv_swap_mode="async"`. State that async mode requires CUDA and sufficient locked/pinned host memory. State explicitly that default sync retains pageable `.to("cpu").clone()`/blocking restore semantics, owns `PageableCPUBufferRecord` objects directly, never constructs or consumes the pinned pool, and therefore reports zero pinned capacity/in-use bytes even though pageable host copies exist.

- [ ] **Step 2: Document lifecycle and recovery**

Add the transfer-only state diagram, separate `SequenceStatus` recovery, group phase/queue rules, ticket producer/consumer/retirement ordering, cancellation tombstones, shutdown drain, retry count, and reprefill after terminal H2D/checksum failure. Explain that `/v1/reload` reloads Python modules rather than migrating live KV; operators must drain/replace the engine for model reload.

- [ ] **Step 3: Document telemetry and benchmarking**

List every metric from Task 7 and the two A/B commands from Task 8. Define p99 as measured per-swap observed completion latency. State that results are hardware/workload specific and are not an SLA or performance promise.

- [ ] **Step 4: Document the external-tier boundary and non-goals**

Describe `ExternalKVStore` as the later host↔external seam. Cite Mooncake (`https://arxiv.org/abs/2407.00079`) only as architectural motivation for decoupled transfer/storage. Explicitly state that this release contains no external-store implementation, distributed storage, multi-node protocol, or KV quantization.

- [ ] **Step 5: Document rollout stages**

Use this rollout:

```text
Stage 0: default sync; merge CPU/CUDA correctness tests and telemetry.
Stage 1: async opt-in on one canary; alert on failures, checksum failures,
         backpressure, pinned utilization, and p99 swap latency.
Stage 2: async opt-in fleet expansion only after workload-specific A/B review;
         retain immediate config rollback to sync.
Stage 3: consider changing defaults in a separate change with production data.
```

Rollback is a config change to `kv_swap_mode="sync"` plus engine restart/drain; never switch backends while transfers are in flight.

- [ ] **Step 6: Run executable documentation QA**

From the repository root, run this documentation contract check before committing:

```bash
python - <<'PY'
from pathlib import Path
import re

docs = {
    "configuration": Path("docs/configuration.md").read_text(),
    "serving": Path("docs/serving.md").read_text(),
    "benchmarking": Path("docs/benchmarking.md").read_text(),
    "changelog": Path("CHANGELOG.md").read_text(),
}

required = {
    "configuration": [
        "kv_swap_mode", "kv_swap_host_memory_bytes", "kv_swap_max_inflight_bytes",
        "kv_swap_checksum", "kv_swap_max_retries", "kv_swap_allow_sync_fallback",
        "PageableCPUBufferRecord", "pinned", "sync",
    ],
    "serving": [
        "GPU_RESIDENT", "HOST_RESIDENT", "SWAP_OUT_IN_FLIGHT",
        "SWAP_IN_IN_FLIGHT", "CANCEL_PENDING", "SequenceStatus",
        "poll_transfers", "retire", "rollback", "reprefill", "reload",
    ],
    "benchmarking": [
        "p99", "kv_swap", "warmup", "backpressure", "bytes",
    ],
    "changelog": ["async", "hierarchical", "KV"],
}
for name, needles in required.items():
    missing = [needle for needle in needles if needle not in docs[name]]
    assert not missing, f"{name} missing documented terms: {missing}"

assert "ExternalKVStore" in docs["serving"] or "ExternalKVStore" in docs["configuration"]
assert "no external" in docs["serving"].lower() or "no external" in docs["configuration"].lower()
assert "kv_swap_mode=\"sync\"" in docs["serving"] or "kv_swap_mode=\"sync\"" in docs["configuration"]
assert "restart" in docs["serving"].lower() or "restart" in docs["configuration"].lower()

for path, text in docs.items():
    for target in re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (Path("docs") / target).resolve() if path != "changelog" else Path(target).resolve()
        assert resolved.exists(), f"{path}: broken Markdown link target {target}"
print("documentation contract, lifecycle, metrics/benchmark, external-tier limits, rollout/rollback, and links: PASS")
PY
git diff --check
```

Expected: the Python command prints the `documentation contract ... PASS` line, exits 0, and `git diff --check` exits 0 with no output. If the check fails, update the documentation (or add the missing focused assertion to this command), rerun it, and do not commit until both commands pass.

- [ ] **Step 7: Commit**

```bash
git add docs/configuration.md docs/serving.md docs/benchmarking.md CHANGELOG.md
git commit -m "docs: add async hierarchical KV swap runbook"
```

### Task 10: Run full verification and inspect resource safety

**Files:**
- Verify all files listed above

- [ ] **Step 1: Run formatting and static checks**

Run: `pre-commit run --all-files`

Expected: all hooks pass with no modified files left by formatters. If formatters change files, review those changes, rerun the focused tests once, and include them in the relevant task commit rather than creating a formatting-only commit.

- [ ] **Step 2: Run CPU suites**

Run:

```bash
python -m pytest -q \
  tests/python/unit/test_kv_transfer.py \
  tests/python/unit/test_kv_config_wiring.py \
  tests/python/unit/test_kv_edge_cases.py \
  tests/python/unit/test_kv_swap_recovery.py \
  tests/python/unit/test_kv_handler_registration.py \
  tests/python/unit/test_unified_scheduler.py \
  tests/python/unit/test_kv_offload_benchmark.py \
  tests/python/serving/test_async_kv_swap.py \
  tests/python/serving/test_kv_cache.py \
  tests/python/serving/test_scheduler.py \
  tests/python/serving/test_engine.py \
  tests/python/serving/test_cancellation.py \
  tests/python/serving/test_api_routes.py \
  tests/python/integration/test_swap_scheduling.py \
  tests/python/integration/test_flashinfer_offload_wiring.py \
  tests/python/integration/test_expert_kv_integration.py
```

Expected: all selected tests pass with no warnings about leaked leases, failed transfers, or pending events.

- [ ] **Step 3: Run CUDA equivalence/race tests**

Run: `python -m pytest -q -m gpu tests/python/integration/test_async_kv_swap_cuda.py`

Expected: all tests pass on CUDA. On a CPU-only developer host, record the skips and require this command on a CUDA CI runner before rollout.

- [ ] **Step 4: Run serving and integration suites**

Run:

```bash
python -m pytest -q tests/python/serving/
python -m pytest -q tests/python/integration/
```

Expected: both suites pass; no regression in FlashInfer layout, DFlash scheduling, native swap, or cancellation.

- [ ] **Step 5: Inspect final accounting after stress**

Run a 1,000-cycle fake-backend stress test alternating swap-out, swap-in, cancellation, failure, and sequence-ID reuse. Expected final assertions:

```python
assert cache.block_allocator.num_free_blocks == cache.num_blocks
assert cache.get_swap_stats()["host_in_use_bytes"] == 0
assert cache.get_swap_stats()["inflight"] == 0
assert cache.get_swap_stats()["host_resident"] == 0
assert cache.get_swap_stats()["retiring_records"] == 0
assert all(ticket.retired for ticket in completed_tickets)
```

- [ ] **Step 6: Commit final test-only adjustments, if any**

```bash
git add tests/python
git commit -m "test: finalize async KV swap verification"
```

Skip this commit when Step 1-5 require no changes; do not create an empty commit.

## Risks and mitigations

| Risk | Mitigation and required evidence |
| --- | --- |
| Transfer state accidentally coupled to request status | Distinct enums/modules; no transfer arrow names a `SequenceStatus`; scheduler tests assert cache-record removal precedes `SWAPPED -> WAITING`. |
| Use-after-free from cancellation or shutdown | Generation-keyed tombstones; ticket retains stream/event/staging ownership; block/lease release occurs only after `retire()`; CPU fake and CUDA shutdown-order tests. |
| Partial group publication or duplicate queue membership | Group reservation before submission, `_SwappedGroupRecord` phase aggregation, one poll per schedule pass, queue uniqueness assertions after every partial/failure completion. |
| Producer/consumer stream race | Transfer stream waits producer; early consumer waits event; normal serving publishes only after event query; CUDA checksum race test. |
| Async pinned-memory or in-flight exhaustion | Two async-only hard byte caps checked before mutation; atomic group reservation rollback; backpressure telemetry and canary alerting. |
| Default sync accidentally consumes async pinned budget | Separate `PageableCPUBufferRecord` and `PinnedBufferLease` fields with a mutual-exclusion invariant; sync-first factory branch returns `pool=None`; config and lifecycle tests use a forbidden pool factory and assert zero pinned accounting. |
| Corrupt/stale/padded payload accepted | Independent validation of shape, dtype, bytes, blocks, generation, bounds, token limits, and final-block padding; optional full-padded-payload CRC32. |
| Config accepted but dropped before backend construction | Non-default sentinel propagation test across ArcherConfig, both server entry points, engine config, engine, cache, backend, and pool. |
| Native scheduler reports block counts as bytes or owns CUDA events | Coordinator installs against concrete K/V tensors, owns wait/retire, and returns exact payload bytes; Unified scheduler tests assert zero bytes on failure/cancel and no ticket/event fields. |
| Shutdown exceeds warning deadline | Deadline emits an operational warning but does not permit unsafe release; synchronize/retire/finalize order remains mandatory. |
| Benchmark interpreted as an SLA | Emit raw samples/environment/commit and label comparisons workload-specific; no performance threshold in pytest or rollout promise. |

## Failure recovery matrix

| Failure | Preserved source | Immediate action | Retry/fallback | Terminal action |
| --- | --- | --- | --- | --- |
| Pinned pool exhausted | GPU blocks | Return `False`; increment backpressure | Scheduler may try another victim/later cycle | No state mutation |
| Sync pageable allocation/restore failure | Existing GPU blocks or pageable record | Propagate the blocking exception; never create/acquire a pinned lease | No implicit async retry | Release only an already-created pageable record through `PageableCPUBufferRecord.release()` when the sequence is discarded/shut down |
| D2H submit/event failure | GPU blocks | Keep group `SWAPPED`; retire submitted tickets; restore `EVICTING→ALLOCATED` | Group rollback restores host-resident peers; sync selection requires restart/config rollback | Atomically restore prior statuses and `_running` membership |
| `free_sequence()`/cancellation during D2H | GPU blocks, lease, ticket, stream/event/staging | Move generation to `_retiring_records`; mark `CANCEL_PENDING`; release nothing | No retry | After event and ticket retirement, free blocks/lease and remove tombstone |
| Host metadata/version/bounds/padding mismatch | Host lease, no active ticket | Mark transfer `FAILED`; never submit H2D | No suspect-byte fallback | Cache removes failed record/resources, then scheduler independently moves whole group `SWAPPED -> WAITING` |
| CRC32 mismatch | Host lease, no active ticket | Increment checksum failure; mark transfer `FAILED` | No suspect-byte fallback | Cache removes failed record/resources, then scheduler independently requeues whole group |
| No GPU blocks for H2D | Host lease | Return `False` | Retry on later schedule | Stay `HOST_RESIDENT` |
| H2D submit/event failure | Host lease; submitted tickets retain destination ownership | Retire completed tickets; free only retired RESTORING blocks | Retry whole group up to `kv_swap_max_retries` | After all tickets retire, remove cache records, then independently requeue group for prefill |
| `free_sequence()`/cancellation during H2D | Host lease, RESTORING blocks, ticket, stream/event/staging | Move generation to tombstone; release nothing | No retry | After event and retirement, free unpublished blocks/lease and remove tombstone |
| Shutdown with in-flight DMA | All blocks, leases, tickets, streams/events/staging | Stop admission; synchronize events; retire tickets; run cache finalizers | Warn if deadline crossed | Only then close streams and release pool |

## Acceptance criteria

1. `KVTransferState` has no transition to `SequenceStatus`; failed-record removal and scheduler `SWAPPED -> WAITING` recovery are separately tested.
2. The serving hot path contains no `.to("cpu").clone()` or host-side stream synchronization in async mode; D2H/H2D uses pinned tensors, nonblocking copies, producer/transfer/consumer ordering, and events.
3. `CopyTicket` owns device, stream, event, and staging references until explicit retirement; shutdown synchronizes, retires, finalizes resources, closes streams, then releases the pool in that order.
4. `free_sequence()`/cancellation never frees EVICTING/RESTORING blocks or leases before ticket retirement, including sequence-ID reuse and shutdown.
5. Scheduler performs group-atomic reservation, queue/status transitions, completion aggregation, rollback, success, and reprefill; no partial group becomes runnable and no group appears in multiple queues.
6. Async backpressure is bounded by host and in-flight caps, observable, and leaves group state unchanged on reservation failure; sync does not consult or consume either cap.
7. Independent tests reject incorrect metadata shape, dtype, bytes, block count, generation, block bounds/duplicates, token bounds, and final-block padding; suspect bytes never reach H2D.
8. All six config fields reach backend/cache construction through CLI and `MoE.serve()` paths, with sync default and explicit initialization fallback behavior; the sync-first factory branch returns `SyncKVTransferBackend`, `pool=None`, and never invokes the pool factory.
9. Native coordinator installation uses concrete paged K/V tensors, owns ticket completion/retirement, and reports exact payload bytes; `UnifiedTransferScheduler` owns queue/result accounting only.
10. CPU fake tests cover ownership/state/group failures; CUDA tests prove exact equivalence, stream waits, staging lifetime, cancellation races, and shutdown order.
11. The benchmark reports measured p99 swap latency and correctness/accounting without asserting a performance gain.
12. `ExternalKVStore` is defined/documented, but no external/distributed store, KV quantization, multi-node implementation, empty exception path, or silent failure is introduced.
13. Default sync swap-out uses an explicit pageable record created by the existing blocking `.detach().to("cpu").clone()` operation; sync swap-in is complete before return; swap-in/free/shutdown drop pageable ownership directly; sync never constructs/acquires/releases a pinned pool or lease and reports zero pinned-budget usage. Async reservations continue to use bounded `PinnedBufferLease` ownership exclusively.
