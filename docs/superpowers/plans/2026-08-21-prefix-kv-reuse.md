# Correctness-Preserving Prefix KV Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire opt-in, correctness-preserving longest-prefix KV reuse into the active OpenAI continuous-batching path, with exact cold/warm equivalence and unchanged disabled or unsupported-runtime behavior.

**Architecture:** A namespace-scoped prefix tree indexes immutable, layer-complete physical KV blocks by exact parent-entry identity and token block. Lookup returns a pinned lease before any eviction or allocation; the scheduler admits every sequence in a request group atomically, while the batch/runtime metadata keeps submitted query lengths separate from total KV lengths. The engine publishes only successfully committed full-block ranges, and reference counting plus copy-on-write keeps completion, cancellation, preemption, reload, and eviction safe.

**Tech Stack:** Python 3.10+, PyTorch, FastAPI/OpenAI v2 serving, Qwen3 MoE paged attention, FlashInfer paged prefill/decode, pytest, Prometheus, JSON benchmarks.

---

## Current seams and corrected design constraints

The active path is `moe_infinity/entrypoints/openai/api_server_v2.py` → `ContinuousBatchingEngine` → `serving/Scheduler` → `BatchBuilder` → `ModelRunner`. The current scaffold is not safe to wire directly:

- `serving/prefix_cache.py` has no active caller and hashes each block independently.
- `runtime/attention_backend.py:_build_flashinfer_metadata()` incorrectly derives `qo_indptr` from total sequence lengths; a warm suffix query therefore masquerades as a full-prompt query.
- `PagedAttentionBackend.k_cache`, `v_cache`, and `_fi_kv_cache` have no layer dimension, while class-level paged context is shared across attention modules; one layer can overwrite another's physical slots.
- `PagedKVCache._kv_cache` is not the physical storage consumed by `ModelRunner`, so it cannot be authoritative for COW or swap restore.
- Scheduler admission allocates sequences independently, which is incompatible with atomic `n>1` group admission.

### Invariants

1. **Compatibility namespace:** A hit requires exact model ID/revision, tokenizer ID/revision and special-token/chat-template digest, adapter ID/revision, dtype, layer count, KV geometry, attention layout/backend, RoPE/scaling/sliding-window configuration, and process runtime epoch. Missing identity becomes process-local identity, never a wildcard.
2. **Exact path identity:** Every cache node has a unique `entry_id`; a child's logical identity is `(namespace, parent_entry_id, exact token tuple)`. SHA-256 is only a bucket accelerator. Digest equality never substitutes for parent-entry or token equality.
3. **Safe prefix length:** Reuse only complete blocks and leave at least one prompt token to execute: `floor((prompt_len - 1) / block_size) * block_size`.
4. **Chunk-plan metadata contract:** Both plans construct one canonical `PagedBatchLengths(query_lengths, query_offsets, context_lengths, kv_seq_lengths)`. `query_lengths[i]` is the number of submitted tokens, `query_offsets` is its exclusive prefix sum beginning at zero, `context_lengths[i]` is committed KV before this forward, and `kv_seq_lengths[i] == context_lengths[i] + query_lengths[i]`. FlashInfer receives `qo_indptr=query_offsets`; KV page metadata is derived only from `kv_seq_lengths` and block tables.
5. **Layer completeness:** One physical block ID names K/V for every model layer in one validated `LayeredPagedKVStore`. Export/import, checkpoint/restore, COW, publication, and eviction operate across all layers or fail closed.
6. **Pinned lookup:** Matching blocks gain lease references while the prefix-index lock is held. Only after every sequence in a group is pinned may eviction or allocation run. Eviction can remove cache ownership but cannot reclaim leased blocks.
7. **Atomic groups:** All sequences in a `SequenceGroup`, including `n>1`, are admitted with their pinned prefixes and suffix blocks together, or none are admitted and every lease/allocation is rolled back.
8. **Ownership:** Allocation creates one sequence reference; a lookup lease creates one temporary sequence reference; publication creates one cache reference. Lease adoption is two phase: every lease first prepares against one group owner token, then the cache publishes every staged block table and every prepared lease commits without further validation. Prepare failure or pre-commit failure aborts every open/prepared lease and releases every staged allocation. Physical storage returns to the allocator only at refcount zero.
9. **COW:** Indexed blocks are immutable. Any append or truncate that would write a shared partial tail first copies every layer to a private physical block.
10. **Committed publication:** After a successful paged forward, record the contiguous committed KV range `[context_len, context_len + query_len)`. Publish a block only when all its token positions are committed and the block belongs to the prompt. Failed forwards and uncommitted DFlash verify tails are never published.
11. **Eviction:** LRU node removal recursively removes descendants by `entry_id` and releases cache references. Active lease/sequence references survive metadata eviction. Memory-pressure eviction continues until actual free-block count satisfies the atomic group or the index is empty.
12. **ContextPilot separation:** CP scores may reorder waiting groups but never authorize a block hit. Only exact namespace/path/token lookup does.
13. **Validated binding:** `ContinuousBatchingEngine` resolves the production `PagedAttentionBackend`, asks that backend's store factory for its one `LayeredPagedKVStore`, and calls `PagedKVCache.set_block_store(store, owner=backend)` before constructing a scheduler. Binding verifies `backend.block_store is store`, `store.owner is backend`, complete geometry, and `logical_num_blocks <= store.num_blocks`; it disables/removes the cache's independent tensor/wrappers so backend and cache cannot remain active against different storage. Absence, attempted rebinding, active-table binding, or mismatch leaves the provider `None` and executes cold.
14. **Disabled fallback:** With `enable_prefix_caching=False`, or without a complete Qwen3 paged-layer registry plus real FlashInfer prefix-aware prefill, the engine executes the existing cold path and reports a stable disabled reason.

## Reconciliation decisions

- **One cross-plan contract:** At integration, both branches use `moe_infinity/serving/prefix_contract.py` and `tests/python/serving/test_prefix_contract.py`. Any older chunk-plan spelling of `prefix_lease.py`, or local redefinition of `PrefixMatch`, `PrefixLease`, or `PrefixLeaseProvider`, is replaced by imports from this contract; `PrefixCache` and `CacheNamespace` remain in `prefix_cache.py` and are not duplicated by chunk scheduling.
- **Chunked prefill:** Prefix reuse and chunking consume the same optional `PrefixLeaseProvider` from `serving/prefix_contract.py`. A synthetic `PagedBatchLengths([16], [0,16], [64], [80])` produces `qo_indptr=[0,16]`, not `[0,80]`, and commits/publishes only `[64,80)`. The non-chunking scheduler requests one lease for the complete remaining prefill; a chunk scheduler requests the same lease once before its first chunk and carries the committed block-table ownership across later chunks.
- **COW and references:** Lookup pins precede eviction. `prepare_group()` prepares already-retained lease IDs without retaining twice; `commit_group()` publishes every table and transfers every prepared lease together. Partial-tail COW copies all layers, swaps one sequence table entry, then releases that sequence's old reference.
- **Committed ranges:** `SequenceData.committed_kv_tokens` advances only after successful model execution. Prefix publication uses this field, not scheduled length, requested prompt length, or generated output length.
- **DFlash:** Delegated DFlash owns a separate `DynamicCache`. It remains eligible only for a cold singleton prefill where `batch.lengths.kv_seq_lengths[0] == batch.lengths.query_lengths[0]`, no prefix lease, and the existing greedy gate. Reused-prefix requests use ordinary paged execution. Delegated and verify-session paths do not publish into the paged prefix cache; rejected/uncommitted verify rows therefore cannot leak into entries.

### Mandatory implementation ordering

Use this dependency order: first add Task 1's `prefix_contract.py`/`prefix_cache.py` production contracts and Task 2's `LayeredPagedKVStore` production contract; then create the complete shared test utility and run Tasks 1–2 tests. Continue with Tasks 3, 4, 5, and 6 numerically. Task 7 is the first task allowed to activate the feature in `ContinuousBatchingEngine`: it must validate and bind the store before constructing the scheduler with the provider argument. Benchmarks and rollout tests follow only after this order is green. This interleaving prevents the shared utility from importing either contract before both exist.

## File responsibility map

- Create `moe_infinity/serving/prefix_contract.py`: the only definition site for shared `PrefixLease`, `PrefixMatch`, and optional `PrefixLeaseProvider` consumed by normal and chunked prefill.
- Modify `moe_infinity/serving/prefix_cache.py`: `PrefixLeaseProvider` implementation, exact-path tree, collision buckets, subtree LRU, metrics.
- Modify `moe_infinity/serving/kv_cache.py`: block refcounts, atomic group allocation, lease adoption, layer-complete COW/swap lifecycle.
- Modify `moe_infinity/runtime/attention_types.py`: canonical `PagedBatchLengths(query_lengths, query_offsets, context_lengths, kv_seq_lengths)` shared with the chunk plan.
- Modify `moe_infinity/runtime/attention_backend.py`: validated `LayeredPagedKVStore`, layer registry, query-vs-KV FlashInfer metadata, exact export/import/checkpoint/restore API.
- Modify `moe_infinity/models/qwen3_paged_attention.py`: pass the concrete module `layer_idx` into the shared backend.
- Modify `moe_infinity/serving/model_runner.py`: build query/total metadata and validate a complete, unambiguous Qwen3 layer registry.
- Modify `moe_infinity/serving/sequence.py`: track committed KV range and pinned-prefix state.
- Modify `moe_infinity/serving/scheduler.py`: pin-before-evict and all-or-nothing group admission.
- Modify `moe_infinity/serving/batch.py`: suffix/chunk query lengths distinct from total KV lengths.
- Modify `moe_infinity/serving/engine.py`: capability gate, post-forward commit/publication, DFlash exclusion, lifecycle and stats.
- Modify `moe_infinity/entrypoints/openai/api_server_v2.py`: config, reload invalidation, metrics.
- Create `tests/python/serving/prefix_cache_test_utils.py`: complete CPU stores/providers/builders shared by prefix and chunk-compatible tests.
- Create `tests/python/serving/test_prefix_contract.py`: provider/lease contract shared by normal and chunked prefill.
- Modify `tests/python/serving/test_prefix_cache.py`, `test_kv_cache.py`, `test_scheduler.py`, `test_batch.py`, `test_engine.py`, `test_correctness.py`, `test_cancellation.py`, `test_hot_reload.py`, `test_api_routes.py`, `test_flashinfer_model_runner.py`.
- Create `tests/python/serving/test_qwen3_paged_attention.py`: verifies each Qwen attention module forwards its concrete `layer_idx`.
- Modify `tests/python/unit/test_flashinfer_attention_backend.py`, `test_v2_lifespan.py`, `test_kv_swap_recovery.py`.
- Create `tests/python/serving/test_prefix_cache_cuda.py` and `test_prefix_cache_benchmark.py`.
- Create `benchmarks/serving/prefix_cache_benchmark.py`.
- Modify `docs/serving.md`, `docs/benchmarking.md`, `README.md`, `ARCHITECTURE.md`, and `CHANGELOG.md`.

### Task 1: Build exact-path prefix entries and pinned leases

**Files:**
- Create: `moe_infinity/serving/prefix_contract.py`
- Create: `tests/python/serving/prefix_cache_test_utils.py`
- Modify: `moe_infinity/serving/prefix_cache.py:1-111`
- Modify: `tests/python/serving/test_prefix_cache.py:1-125`
- Create: `tests/python/serving/test_prefix_contract.py`

- [ ] **Step 1: Define test helpers and write failing multilevel-collision/lease tests**

The following is the complete shared utility content. Per the dependency order above, write its production-independent cache helpers with Task 1, then add the store-dependent definitions immediately after Task 2 creates `LayeredPagedKVStore`; import this module in tests only after both additions are present.

```python
# tests/python/serving/prefix_cache_test_utils.py
import dataclasses
from collections.abc import Callable
from types import SimpleNamespace

import torch

import moe_infinity.serving.prefix_cache as prefix_cache_module
from moe_infinity.serving.prefix_cache import CacheNamespace, PrefixCache
from moe_infinity.serving.prefix_contract import PrefixLease, PrefixMatch
from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.batch import BatchMetadata
from moe_infinity.serving.scheduler import Scheduler
from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.model_runner import ModelRunner
from moe_infinity.serving.sequence import SamplingParams, SequenceData, SequenceGroup, SequenceStatus
from moe_infinity.runtime.attention_backend import (
    LayeredPagedKVCheckpoint, LayeredPagedKVPayload, LayeredPagedKVStore,
    PagedAttentionBackend,
)
from moe_infinity.runtime.attention_types import KVCacheSpec, PagedBatchLengths

def make_namespace(**changes: object) -> CacheNamespace:
    base = CacheNamespace(
        model_id="Qwen/Qwen3-30B-A3B", model_revision="rev-a",
        tokenizer_id="Qwen/Qwen3-30B-A3B", tokenizer_revision="rev-a",
        tokenizer_config_digest="tok-digest", adapter_id=None,
        adapter_revision=None, dtype="bfloat16", block_size=4,
        num_layers=2, num_kv_heads=2, head_dim=8,
        attention_backend="flashinfer-paged", attention_layout="NHD",
        position_config_digest="rope-default;window=none", runtime_epoch="epoch-1",
    )
    return dataclasses.replace(base, **changes)

class RefRecorder:
    def __init__(self) -> None:
        self.retained: list[list[int]] = []
        self.released: list[list[int]] = []
    def retain(self, ids: list[int]) -> None: self.retained.append(list(ids))
    def release(self, ids: list[int]) -> None: self.released.append(list(ids))

class RecordingLayeredPagedKVStore(LayeredPagedKVStore):
    def __init__(
        self, num_layers: int = 3, num_blocks: int = 8, block_size: int = 4,
        num_kv_heads: int = 2, head_dim: int = 8,
        dtype: torch.dtype = torch.float32, owner: object | None = None,
    ) -> None:
        owner = owner or SimpleNamespace()
        super().__init__(
            owner=owner, num_layers=num_layers, num_blocks=num_blocks, block_size=block_size,
            num_kv_heads=num_kv_heads, head_dim=head_dim, dtype=dtype,
            device=torch.device("cpu"), use_flashinfer=True,
        )
        setattr(owner, "block_store", self)
        self.copies: list[tuple[int, int, tuple[int, ...]]] = []

    def import_blocks(self, block_ids: list[int], payload: LayeredPagedKVPayload) -> None:
        super().import_blocks(block_ids, payload)
        if len(block_ids) == 1 and len(payload.source_block_ids) == 1:
            self.copies.append((payload.source_block_ids[0], block_ids[0], tuple(range(self.num_layers))))

    def layer_values(self, block_ids: list[int]) -> list[list[float]]:
        return self.fi_kv_cache[:, block_ids, 0, 0, 0, 0].tolist()

class RecordingPrefixLeaseProvider:
    def __init__(self, cache: PrefixCache) -> None:
        self.cache = cache
        self.events: list[str] = []

    @property
    def open_leases(self) -> int:
        return self.cache.open_leases

    def acquire_prefix_lease(
        self, namespace: CacheNamespace, token_ids: list[int], max_prefix_tokens: int
    ) -> PrefixLease:
        self.events.append(f"pin:{token_ids[-1]}")
        return self.cache.acquire_prefix_lease(namespace, token_ids, max_prefix_tokens)

    def evict_until(self, predicate: Callable[[], bool]) -> None:
        self.events.append("evict")
        self.cache.evict_until(predicate)

def make_cache(
    store: RecordingLayeredPagedKVStore | None = None, num_blocks: int = 8
) -> PagedKVCache:
    cache = PagedKVCache(
        num_blocks=num_blocks, block_size=4, num_layers=3,
        num_heads=2, head_dim=8, dtype=torch.float32, device=torch.device("cpu"),
    )
    if store is not None:
        cache.set_block_store(store, owner=store.owner)
    return cache

def make_paged_backend(num_blocks: int) -> PagedAttentionBackend:
    return PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2, head_dim=8, dtype=torch.float32, block_size=4,
        ),
        num_gpu_blocks=num_blocks,
        device=torch.device("cpu"),
    )

def make_group(request_id: str, rows: list[tuple[int, list[int]]]) -> SequenceGroup:
    return SequenceGroup(
        request_id=request_id,
        sequences=[
            SequenceData(seq_id=seq_id, prompt_token_ids=tokens, sampling_params=SamplingParams())
            for seq_id, tokens in rows
        ],
    )

def make_qwen_runner(layer_indices: list[int], expected_layers: int) -> ModelRunner:
    class FakeQwen3PagedAttention:
        @classmethod
        def set_paged_context(cls, backend, metadata) -> None:
            cls.backend, cls.metadata = backend, metadata
        @classmethod
        def clear_paged_context(cls) -> None:
            cls.backend, cls.metadata = None, None
    FakeQwen3PagedAttention.__name__ = "Qwen3PagedAttention"
    modules = []
    for layer_idx in layer_indices:
        module = FakeQwen3PagedAttention()
        module.layer_idx = layer_idx
        modules.append(module)
    model = SimpleNamespace(
        modules=lambda: iter(modules),
        config=SimpleNamespace(num_hidden_layers=expected_layers, sliding_window=None),
    )
    backend = SimpleNamespace()
    store = RecordingLayeredPagedKVStore(
        num_layers=expected_layers, owner=backend
    )
    backend.num_gpu_blocks = store.num_blocks
    backend.spec = SimpleNamespace(
        block_size=store.block_size, num_kv_heads=store.num_kv_heads,
        head_dim=store.head_dim, dtype=store.dtype,
    )
    backend.register_layers = lambda registrations: None
    backend.create_layered_store = lambda *, layer_count: store
    backend._flashinfer_enabled = lambda: True
    owner = SimpleNamespace(get_attention_backend=lambda: backend, expert_layer_modules=[])
    return ModelRunner(model, owner, device=torch.device("cpu"))

def make_prefill_batch(sequence: SequenceData, context_len: int, query_tokens: list[int], block_table: list[int]) -> BatchMetadata:
    return BatchMetadata(
        seq_ids=[sequence.seq_id], input_token_ids=list(query_tokens),
        lengths=PagedBatchLengths(
            [len(query_tokens)], [0, len(query_tokens)], [context_len],
            [context_len + len(query_tokens)],
        ),
        is_prefill=[True],
        block_tables=[list(block_table)], sampling_params=[sequence.sampling_params],
    )

SHARED = list(range(8))

def make_seeded_scheduler(
    *, num_blocks: int, max_batch_size: int
) -> tuple[Scheduler, PagedKVCache, RecordingPrefixLeaseProvider, CacheNamespace]:
    store = RecordingLayeredPagedKVStore(num_blocks=num_blocks)
    cache = make_cache(store=store, num_blocks=num_blocks)
    namespace = make_namespace(num_layers=3)
    prefix = PrefixCache(
        block_size=4, max_entries=8,
        on_retain=cache.block_allocator.retain,
        on_release=cache.block_allocator.release,
    )
    cache.allocate_sequence(999, 9)
    seed_blocks = cache.get_block_table(999)
    prefix.insert(namespace, SHARED + [99], seed_blocks, committed_tokens=8)
    cache.free_sequence(999)
    provider = RecordingPrefixLeaseProvider(prefix)
    scheduler = Scheduler(
        cache, max_batch_size=max_batch_size, max_tokens_per_step=64,
        prefix_lease_provider=provider, cache_namespace=namespace,
    )
    return scheduler, cache, provider, namespace

def bind_prefix_runtime(engine: ContinuousBatchingEngine) -> RecordingLayeredPagedKVStore:
    cache = engine.kv_cache
    store = RecordingLayeredPagedKVStore(
        num_layers=cache.num_layers, num_blocks=cache.num_blocks,
        block_size=cache.block_size, num_kv_heads=cache.num_heads,
        head_dim=cache.head_dim, dtype=cache.dtype,
    )
    owner = store.owner
    cache.set_block_store(store, owner=owner)
    namespace = make_namespace(
        num_layers=cache.num_layers, num_kv_heads=cache.num_heads,
        head_dim=cache.head_dim, dtype=str(cache.dtype).removeprefix("torch."),
    )
    prefix = PrefixCache(
        cache.block_size, 32,
        on_retain=cache.block_allocator.retain,
        on_release=cache.block_allocator.release,
    )
    engine.prefix_cache, engine.cache_namespace = prefix, namespace
    engine.scheduler.prefix_lease_provider = prefix
    engine.scheduler.cache_namespace = namespace
    return store

def make_prefix_capable_engine(cb_engine_factory) -> ContinuousBatchingEngine:
    engine = cb_engine_factory(config_overrides={"enable_prefix_caching": False})
    bind_prefix_runtime(engine)
    return engine

def add_prefill(
    engine: ContinuousBatchingEngine, prompt: list[int], committed: int
) -> SequenceData:
    request_id = f"synthetic-{len(engine._request_to_seq_ids)}"
    engine.add_request(request_id, prompt, SamplingParams(temperature=0.0, max_tokens=1))
    seq_id = engine._request_to_seq_ids[request_id][0]
    sequence = engine._sequences[seq_id]
    sequence.set_status(SequenceStatus.PREFILL)
    sequence.num_computed_tokens = committed
    sequence.committed_kv_tokens = committed
    engine.kv_cache.allocate_sequence(seq_id, len(prompt))
    if committed:
        engine.prefix_cache.insert(
            engine.cache_namespace, prompt,
            engine.kv_cache.get_block_table(seq_id), committed_tokens=committed,
        )
    return sequence

def make_dflash_engine_and_batch(cb_engine_factory, context_len: int, has_prefix_lease: bool):
    engine = cb_engine_factory()
    engine.speculative_draft = object()
    sequence = SequenceData(
        seq_id=77, prompt_token_ids=list(range(context_len + 1)),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        status=SequenceStatus.PREFILL, num_computed_tokens=context_len,
    )
    sequence.has_prefix_lease = has_prefix_lease
    batch = make_prefill_batch(sequence, context_len, [context_len], [0] * ((context_len + 4) // 4))
    return engine, batch
```

```python
def test_forced_digest_collision_preserves_multilevel_parent_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefix_cache_module, "_digest_block", lambda *args: "same")
    cache = PrefixCache(4, 32)
    ns = make_namespace()
    a = [1,1,1,1, 7,7,7,7, 9,9,9,9, 20]
    b = [2,2,2,2, 7,7,7,7, 9,9,9,9, 21]
    cache.insert(ns, a, [10,11,12], committed_tokens=12)
    cache.insert(ns, b, [20,21,22], committed_tokens=12)
    lease_a = cache.acquire_prefix_lease(ns, a, max_prefix_tokens=12)
    lease_b = cache.acquire_prefix_lease(ns, b, max_prefix_tokens=12)
    assert lease_a.match.block_ids == (10,11,12)
    assert lease_b.match.block_ids == (20,21,22)
    lease_a.abort(); lease_b.abort()

def test_lookup_pins_before_lock_is_released_and_abort_balances_refs() -> None:
    refs = RefRecorder()
    cache = PrefixCache(4, 8, on_retain=refs.retain, on_release=refs.release)
    ns = make_namespace()
    cache.insert(ns, [1,2,3,4,5], [9], committed_tokens=4)
    lease = cache.acquire_prefix_lease(ns, [1,2,3,4,6], max_prefix_tokens=4)
    assert lease.match.block_ids == (9,)
    assert refs.retained == [[9], [9]]  # cache ownership, then lease ownership
    lease.abort()
    assert refs.released == [[9]]

def test_lease_prepare_requires_same_owner_for_commit_or_abort() -> None:
    released: list[list[int]] = []
    lease = PrefixLease(
        PrefixMatch(4, (9,), (1,)), released.append, lambda: None
    )
    owner = object()
    other = object()
    assert lease.prepare_adoption(owner).block_ids == (9,)
    with pytest.raises(RuntimeError, match="owner/state mismatch"):
        lease.commit_adoption(other)
    with pytest.raises(RuntimeError, match="owner mismatch"):
        lease.abort(other)
    lease.abort(owner)
    assert released == [[9]]

def test_prepared_lease_commit_transfers_without_release() -> None:
    released: list[list[int]] = []
    terminals: list[str] = []
    lease = PrefixLease(
        PrefixMatch(4, (9,), (1,)), released.append,
        lambda: terminals.append("closed"),
    )
    owner = object()
    lease.prepare_adoption(owner)
    lease.commit_adoption(owner)
    assert lease.state == "committed"
    assert released == []
    assert terminals == ["closed"]
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/python/serving/test_prefix_cache.py`

Expected: FAIL because `entry_id`, `acquire_prefix_lease`, `PrefixLease`, and `PrefixLeaseProvider` do not exist in the canonical contract module.

- [ ] **Step 3: Implement the exact APIs**

```python
# moe_infinity/serving/prefix_contract.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from moe_infinity.serving.prefix_cache import CacheNamespace

OwnerToken = object

@dataclass(frozen=True)
class PrefixMatch:
    num_tokens: int
    block_ids: tuple[int, ...]
    entry_ids: tuple[int, ...]

@dataclass
class PrefixLease:
    match: PrefixMatch
    _release: Callable[[list[int]], None]
    _terminal: Callable[[], None]
    _state: str = "open"
    _prepared_owner: OwnerToken | None = None

    @classmethod
    def empty(cls) -> "PrefixLease":
        return cls(PrefixMatch(0, (), ()), lambda ids: None, lambda: None)

    @property
    def state(self) -> str:
        return self._state

    def is_prepared_for(self, owner: OwnerToken) -> bool:
        return self._state == "prepared" and self._prepared_owner is owner

    def prepare_adoption(self, owner: OwnerToken) -> PrefixMatch:
        if self._state != "open":
            raise RuntimeError(f"lease is already {self._state}")
        if owner is None:
            raise ValueError("lease adoption owner must not be None")
        self._prepared_owner = owner
        self._state = "prepared"
        return self.match

    def commit_adoption(self, owner: OwnerToken) -> PrefixMatch:
        if self._state != "prepared" or self._prepared_owner is not owner:
            raise RuntimeError("lease adoption owner/state mismatch")
        self._state = "committed"
        self._prepared_owner = None
        self._terminal()
        return self.match

    def abort(self, owner: OwnerToken | None = None) -> None:
        if self._state not in {"open", "prepared"}:
            raise RuntimeError(f"lease is already {self._state}")
        if self._state == "prepared" and self._prepared_owner is not owner:
            raise RuntimeError("lease adoption owner mismatch")
        self._release(list(self.match.block_ids))
        self._state = "aborted"
        self._prepared_owner = None
        self._terminal()

class PrefixLeaseProvider(Protocol):
    def acquire_prefix_lease(
        self,
        namespace: CacheNamespace,
        token_ids: list[int],
        max_prefix_tokens: int,
    ) -> PrefixLease: ...
```

```python
# moe_infinity/serving/prefix_cache.py
EntryId = int

@dataclass
class _CacheEntry:
    entry_id: EntryId
    namespace: CacheNamespace
    parent_entry_id: EntryId
    digest: str
    token_block: tuple[int, ...]
    block_id: int
    child_entry_ids: set[EntryId] = field(default_factory=set)
```

`PrefixMatch`, `PrefixLease`, and `PrefixLeaseProvider` are defined nowhere except `moe_infinity/serving/prefix_contract.py`; `prefix_cache.py`, schedulers, engines, and both plans import them. `PrefixCache` implements the provider method plus `insert(namespace, token_ids, block_ids, committed_tokens) -> None`. Use monotonic `entry_id`, one root entry per namespace, `dict[digest, list[entry_id]]` only for candidate lookup, and exact filtering by `parent_entry_id` plus `token_block`. Under one `RLock`, resolve at most `max_prefix_tokens`, invoke `on_retain` for all matched blocks, increment `open_leases`, and only then return the lease. `prepare_adoption` binds retained IDs to exactly one group owner without ending the lease; `commit_adoption` is called only after all leases and tables prepare, transfers those existing references without another retain, and closes the lease; `abort(owner)` releases open/prepared references. Test `prefix_lease_provider=None` with `PrefixLease.empty()` as the identity path used by normal and chunked prefill.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest -q tests/python/serving/test_prefix_cache.py tests/python/serving/test_prefix_contract.py`

Expected: PASS, including forced three-level digest collisions and balanced leases.

```bash
git add moe_infinity/serving/prefix_contract.py moe_infinity/serving/prefix_cache.py tests/python/serving/prefix_cache_test_utils.py tests/python/serving/test_prefix_cache.py tests/python/serving/test_prefix_contract.py
git commit -m "feat(serving): add pinned exact-path prefix index"
```

### Task 2: Make physical KV storage layer-complete

**Files:**
- Modify: `moe_infinity/runtime/attention_backend.py:79-293,443-610`
- Modify: `moe_infinity/models/qwen3_paged_attention.py:24-120`
- Modify: `moe_infinity/serving/model_runner.py:120-315`
- Modify: `tests/python/unit/test_flashinfer_attention_backend.py`
- Modify: `tests/python/serving/test_flashinfer_model_runner.py`
- Create: `tests/python/serving/test_qwen3_paged_attention.py`

- [ ] **Step 1: Write failing all-layer storage and registry tests**

```python
def test_export_import_checkpoint_restore_cover_every_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = PagedAttentionBackend(
        spec=KVCacheSpec(2, 8, torch.float16, 4),
        num_gpu_blocks=4, device=torch.device("cpu"),
    )
    store = backend.create_layered_store(layer_count=3)
    assert backend.block_store is store
    assert store.owner is backend
    assert backend.k_cache is store.k_cache
    assert backend.v_cache is store.v_cache
    assert backend._fi_kv_cache is store.fi_kv_cache
    for layer in range(3):
        store.k_cache[layer, 1].fill_(10 + layer)
        store.v_cache[layer, 1].fill_(20 + layer)
        store.fi_kv_cache[layer, 1].fill_(30 + layer)
    payload = store.export_blocks([1])
    store.import_blocks([2], payload)
    checkpoint = store.checkpoint([2])
    store.zero_blocks([2])
    store.restore([2], checkpoint)
    for layer in range(3):
        assert torch.all(store.k_cache[layer, 2] == 10 + layer)
        assert torch.all(store.v_cache[layer, 2] == 20 + layer)
        assert torch.all(store.fi_kv_cache[layer, 2] == 30 + layer)

def test_prefix_capability_rejects_missing_or_duplicate_qwen_layers() -> None:
    runner = make_qwen_runner(layer_indices=[0, 1, 1], expected_layers=3)
    capability = runner.get_prefix_reuse_capability(make_cache())
    assert capability.supported is False
    assert capability.reason == "incomplete-paged-layer-registry"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/python/unit/test_flashinfer_attention_backend.py tests/python/serving/test_flashinfer_model_runner.py tests/python/serving/test_qwen3_paged_attention.py`

Expected: FAIL because storage lacks a layer dimension and no registry validates module ownership.

- [ ] **Step 3: Implement layer-complete storage and registry**

```python
@dataclass(frozen=True)
class LayerRegistration:
    layer_idx: int
    module_id: int

@dataclass(frozen=True)
class PrefixReuseCapability:
    supported: bool
    reason: str
    backend: PagedAttentionBackend | None
    block_store: LayeredPagedKVStore | None

    @classmethod
    def active(
        cls, backend: PagedAttentionBackend, store: LayeredPagedKVStore
    ) -> "PrefixReuseCapability":
        if backend.block_store is not store or store.owner is not backend:
            raise ValueError("capability backend/store ownership mismatch")
        return cls(True, "active", backend, store)

    @classmethod
    def disabled(cls, reason: str) -> "PrefixReuseCapability":
        return cls(False, reason, None, None)

@dataclass(frozen=True)
class LayeredPagedKVPayload:
    source_block_ids: list[int]
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    fi_kv_cache: torch.Tensor | None

@dataclass(frozen=True)
class LayeredPagedKVCheckpoint:
    source_block_ids: list[int]
    k_cache_cpu: torch.Tensor
    v_cache_cpu: torch.Tensor
    fi_kv_cache_cpu: torch.Tensor | None

class LayeredPagedKVStore:
    def __init__(
        self, *, owner: PagedAttentionBackend, num_layers: int, num_blocks: int, block_size: int,
        num_kv_heads: int, head_dim: int, dtype: torch.dtype,
        device: torch.device, use_flashinfer: bool,
    ) -> None:
        self.owner = owner
        self.num_layers, self.num_blocks, self.block_size = num_layers, num_blocks, block_size
        self.num_kv_heads, self.head_dim = num_kv_heads, head_dim
        self.dtype, self.device = dtype, device
        self.k_cache = torch.zeros(num_layers, num_blocks, num_kv_heads, head_dim // 8, block_size, 8, dtype=dtype, device=device)
        self.v_cache = torch.zeros(num_layers, num_blocks, num_kv_heads, head_dim, block_size, dtype=dtype, device=device)
        self.fi_kv_cache = (
            torch.zeros(num_layers, num_blocks, 2, block_size, num_kv_heads, head_dim, dtype=dtype, device=device)
            if use_flashinfer else None
        )

    def _validate_ids(self, block_ids: list[int]) -> list[int]:
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("block ids must be unique")
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in block_ids):
            raise ValueError("block id is outside the layered store")
        return list(block_ids)

    def _validate_payload_geometry(self, ids: list[int], payload: LayeredPagedKVPayload) -> None:
        if len(ids) != len(payload.source_block_ids):
            raise ValueError("source and destination block counts differ")
        if payload.k_cache.shape != self.k_cache[:, ids].shape or payload.v_cache.shape != self.v_cache[:, ids].shape:
            raise ValueError("K/V payload geometry mismatch")
        if payload.k_cache.dtype != self.dtype or payload.v_cache.dtype != self.dtype:
            raise ValueError("K/V payload dtype mismatch")
        if payload.k_cache.device != self.device or payload.v_cache.device != self.device:
            raise ValueError("K/V payload device mismatch")
        if (payload.fi_kv_cache is None) != (self.fi_kv_cache is None):
            raise ValueError("FlashInfer payload/store mismatch")
        if payload.fi_kv_cache is not None and self.fi_kv_cache is not None:
            if payload.fi_kv_cache.shape != self.fi_kv_cache[:, ids].shape:
                raise ValueError("FlashInfer payload geometry mismatch")
            if payload.fi_kv_cache.dtype != self.dtype or payload.fi_kv_cache.device != self.device:
                raise ValueError("FlashInfer payload dtype/device mismatch")

    def export_blocks(self, block_ids: list[int]) -> LayeredPagedKVPayload:
        ids = self._validate_ids(block_ids)
        fi = None if self.fi_kv_cache is None else self.fi_kv_cache[:, ids].clone()
        return LayeredPagedKVPayload(block_ids, self.k_cache[:, ids].clone(), self.v_cache[:, ids].clone(), fi)

    def import_blocks(self, block_ids: list[int], payload: LayeredPagedKVPayload) -> None:
        ids = self._validate_ids(block_ids)
        self._validate_payload_geometry(ids, payload)
        self.k_cache[:, ids].copy_(payload.k_cache)
        self.v_cache[:, ids].copy_(payload.v_cache)
        if payload.fi_kv_cache is not None:
            if self.fi_kv_cache is None:
                raise ValueError("FlashInfer payload cannot be imported into a non-FlashInfer store")
            self.fi_kv_cache[:, ids].copy_(payload.fi_kv_cache)

    def checkpoint(self, block_ids: list[int]) -> LayeredPagedKVCheckpoint:
        payload = self.export_blocks(block_ids)
        fi = None if payload.fi_kv_cache is None else payload.fi_kv_cache.detach().cpu()
        return LayeredPagedKVCheckpoint(
            payload.source_block_ids, payload.k_cache.detach().cpu(),
            payload.v_cache.detach().cpu(), fi,
        )

    def restore(self, block_ids: list[int], checkpoint: LayeredPagedKVCheckpoint) -> None:
        payload = LayeredPagedKVPayload(
            checkpoint.source_block_ids,
            checkpoint.k_cache_cpu.to(self.device, self.dtype),
            checkpoint.v_cache_cpu.to(self.device, self.dtype),
            None if checkpoint.fi_kv_cache_cpu is None else checkpoint.fi_kv_cache_cpu.to(self.device, self.dtype),
        )
        self.import_blocks(block_ids, payload)

class PagedAttentionBackend:
    def create_layered_store(self, *, layer_count: int) -> LayeredPagedKVStore:
        if layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if getattr(self, "_block_store", None) is not None:
            raise RuntimeError("paged backend store is already initialized")
        self._block_store = LayeredPagedKVStore(
            owner=self,
            num_layers=layer_count,
            num_blocks=self.num_gpu_blocks,
            block_size=self.spec.block_size,
            num_kv_heads=self.spec.num_kv_heads,
            head_dim=self.spec.head_dim,
            dtype=self.spec.dtype,
            device=self.device,
            use_flashinfer=self._flashinfer_enabled(),
        )
        # Replace the constructor's legacy single-layer tensors; no second
        # attention-visible storage remains active.
        self.k_cache = self._block_store.k_cache
        self.v_cache = self._block_store.v_cache
        self._fi_kv_cache = self._block_store.fi_kv_cache
        return self._block_store

    @property
    def block_store(self) -> LayeredPagedKVStore:
        if self._block_store is None:
            raise RuntimeError("paged backend store is not initialized")
        return self._block_store

    def register_layers(self, registrations: list[LayerRegistration]) -> None:
        by_layer = {item.layer_idx: item.module_id for item in registrations}
        expected_layers = self.block_store.num_layers
        if len(by_layer) != len(registrations) or set(by_layer) != set(range(expected_layers)):
            raise ValueError("paged layer registry must contain each layer exactly once")
        self._layer_registry = by_layer
```

`LayeredPagedKVStore.__init__` also requires `owner: PagedAttentionBackend` and stores it by identity. `LayeredPagedKVStore` owns `k_cache`, `v_cache`, and `fi_kv_cache` with leading dimensions `[num_layers, num_blocks]` and exposes exactly `export_blocks`, `import_blocks`, `checkpoint`, and `restore`; no alternate copy/snapshot names remain. `_validate_payload_geometry` checks layer count, block count, dtype, device, K/V shapes, and FlashInfer presence before the first write. Keep the existing supported `PagedAttentionBackend(spec, num_gpu_blocks, device)` constructor unchanged: it does **not** accept `num_layers`. Immediately after discovering the complete Qwen registry, call `backend.create_layered_store(layer_count=expected_layers)` once; the factory replaces the constructor's legacy tensor attributes with aliases to the new store, and test helpers use this factory rather than adding an unsupported backend constructor argument. `PagedAttentionBackend.block_store` is that sole store instance used by every registered Qwen layer. `register_layers` requires exactly one module identity for each index `0..num_layers-1`; duplicates, gaps, or out-of-range IDs disable reuse. Add `layer_idx` to backend `forward`/`write_kv` and select only that layer's storage during attention.

In `Qwen3PagedAttention.forward`, pass `layer_idx=int(self.layer_idx)` to the backend. In `ModelRunner`, enumerate actual `Qwen3PagedAttention` module instances, register `(layer_idx, id(module))`, and stop treating class-level context alone as proof of layer completeness.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest -q tests/python/unit/test_flashinfer_attention_backend.py tests/python/serving/test_flashinfer_model_runner.py tests/python/serving/test_qwen3_paged_attention.py tests/python/serving/test_model_runner.py`

Expected: PASS; every layer survives export/import and checkpoint/restore independently.

```bash
git add moe_infinity/runtime/attention_backend.py moe_infinity/models/qwen3_paged_attention.py moe_infinity/serving/model_runner.py tests/python/unit/test_flashinfer_attention_backend.py tests/python/serving/test_flashinfer_model_runner.py tests/python/serving/test_qwen3_paged_attention.py
git commit -m "fix(serving): make paged KV storage layer complete"
```

### Task 3: Separate query lengths from total KV lengths

**Files:**
- Modify: `moe_infinity/runtime/attention_types.py:20-34`
- Modify: `moe_infinity/serving/batch.py:33-81,217-273`
- Modify: `moe_infinity/serving/model_runner.py:185-236`
- Modify: `moe_infinity/runtime/attention_backend.py:295-553`
- Modify: `tests/python/serving/test_batch.py`
- Modify: `tests/python/serving/test_flashinfer_model_runner.py`
- Modify: `tests/python/unit/test_flashinfer_attention_backend.py`

- [ ] **Step 1: Write failing cold/warm/chunk metadata tests with a real FlashInfer wrapper call shape**

```python
import itertools

class RecordingPrefill:
    def plan(self, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, *args, **kwargs) -> None:
        self.plan_args = SimpleNamespace(
            qo_indptr=qo_indptr.clone(), kv_indptr=kv_indptr.clone(),
            kv_indices=kv_indices.clone(), kv_last_page_len=kv_last_page_len.clone(),
        )
    def run(self, query, kv_cache):
        return query

def make_recording_backend(monkeypatch: pytest.MonkeyPatch):
    prefill, decode = RecordingPrefill(), Mock(plan=Mock(), run=Mock())
    module = SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=lambda workspace, layout: prefill,
        BatchDecodeWithPagedKVCacheWrapper=lambda workspace, layout: decode,
    )
    monkeypatch.setattr(flashinfer_utils, "HAS_FLASHINFER", True)
    monkeypatch.setattr(flashinfer_utils, "get_flashinfer_module", lambda: module)
    monkeypatch.setattr(flashinfer_utils, "get_workspace", lambda device: torch.empty(1))
    spec = KVCacheSpec(num_kv_heads=2, head_dim=8, dtype=torch.float16, block_size=16)
    backend = PagedAttentionBackend(spec, num_gpu_blocks=16, device=torch.device("cpu"))
    backend.create_layered_store(layer_count=1)
    return backend, prefill

def make_tables(kv_seq_lengths: list[int], block_size: int) -> torch.Tensor:
    rows = [(length + block_size - 1) // block_size for length in kv_seq_lengths]
    table = torch.zeros(len(rows), max(rows), dtype=torch.int32)
    cursor = 0
    for row, count in enumerate(rows):
        table[row, :count] = torch.arange(cursor, cursor + count, dtype=torch.int32)
        cursor += count
    return table

def make_slots(query_lengths: list[int], kv_seq_lengths: list[int]) -> torch.Tensor:
    slots = []
    for query_len, kv_len in zip(query_lengths, kv_seq_lengths):
        slots.extend(range(kv_len - query_len, kv_len))
    return torch.tensor(slots, dtype=torch.int64)

def make_q(tokens: int) -> torch.Tensor:
    return torch.zeros(tokens, 2, 8, dtype=torch.float16)

make_k = make_q
make_v = make_q

@pytest.mark.parametrize(
    ("query_lengths", "kv_seq_lengths", "expected_qo", "expected_last_page"),
    [([80], [80], [0, 80], [16]), ([16], [80], [0, 16], [16]), ([3, 5], [67, 21], [0, 3, 8], [3, 5])],
)
def test_flashinfer_qo_uses_query_and_kv_pages_use_total(
    query_lengths, kv_seq_lengths, expected_qo, expected_last_page,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, recording_prefill = make_recording_backend(monkeypatch)
    query_offsets = [0, *itertools.accumulate(query_lengths)]
    context_lengths = [kv - query for query, kv in zip(query_lengths, kv_seq_lengths)]
    metadata = AttentionMetadata(
        block_tables=make_tables(kv_seq_lengths, block_size=16),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor(query_lengths, dtype=torch.int32),
            query_offsets=torch.tensor(query_offsets, dtype=torch.int32),
            context_lengths=torch.tensor(context_lengths, dtype=torch.int32),
            kv_seq_lengths=torch.tensor(kv_seq_lengths, dtype=torch.int32),
        ),
        max_seq_len=max(kv_seq_lengths), num_prefill_tokens=sum(query_lengths),
        num_decode_tokens=0, slot_mapping=make_slots(query_lengths, kv_seq_lengths), is_prefill=True,
    )
    backend.forward(query=make_q(sum(query_lengths)), key=make_k(sum(query_lengths)),
                    value=make_v(sum(query_lengths)), metadata=metadata, layer_idx=0)
    assert recording_prefill.plan_args.qo_indptr.tolist() == expected_qo
    assert recording_prefill.plan_args.kv_last_page_len.tolist() == expected_last_page
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/python/unit/test_flashinfer_attention_backend.py tests/python/serving/test_flashinfer_model_runner.py tests/python/serving/test_batch.py`

Expected: FAIL because runtime metadata lacks `query_lengths`/`query_offsets`/`kv_seq_lengths` and `qo_indptr` uses total sequence lengths.

- [ ] **Step 3: Implement the metadata contract**

```python
@dataclass(frozen=True)
class PagedBatchLengths:
    query_lengths: torch.Tensor | list[int]
    query_offsets: torch.Tensor | list[int]
    context_lengths: torch.Tensor | list[int]
    kv_seq_lengths: torch.Tensor | list[int]

    def validate(self) -> None:
        query = [int(value) for value in self.query_lengths]
        offsets = [int(value) for value in self.query_offsets]
        context = [int(value) for value in self.context_lengths]
        kv = [int(value) for value in self.kv_seq_lengths]
        if len(context) != len(query) or len(kv) != len(query):
            raise ValueError("paged length vectors must have equal batch size")
        expected_offsets = [0]
        for length in query:
            if length < 0:
                raise ValueError("query lengths must be non-negative")
            expected_offsets.append(expected_offsets[-1] + length)
        if offsets != expected_offsets:
            raise ValueError("query_offsets must be the prefix sum of query_lengths")
        if any(prior < 0 for prior in context):
            raise ValueError("context lengths must be non-negative")
        if any(total != prior + current for total, prior, current in zip(kv, context, query)):
            raise ValueError("kv_seq_lengths must equal context_lengths + query_lengths")

@dataclass
class AttentionMetadata:
    block_tables: torch.Tensor
    lengths: PagedBatchLengths
    max_seq_len: int
    num_prefill_tokens: int
    num_decode_tokens: int
    slot_mapping: torch.Tensor
    is_prefill: bool

@dataclass(frozen=True)
class FlashInferPlanMetadata:
    lengths: PagedBatchLengths
    kv_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
```

Define `PagedBatchLengths` once in `moe_infinity/runtime/attention_types.py`; both `BatchMetadata.lengths` and runtime `AttentionMetadata.lengths` use it, and consumers access `metadata.lengths.query_lengths`, `.query_offsets`, `.context_lengths`, and `.kv_seq_lengths`. Do not duplicate these four fields on either metadata class and do not retain `seq_lengths`/`token_offsets` aliases. `BatchBuilder` constructs exactly `PagedBatchLengths(query_lengths, query_offsets, context_lengths, kv_seq_lengths)` and calls `validate()`. `ModelRunner` tensorizes the same object without changing meanings. In `_build_flashinfer_metadata`, pass `lengths.query_offsets` as `qo_indptr`, and compute page counts, `kv_indptr`, indices, and last-page lengths from `lengths.kv_seq_lengths`. Validate `query_offsets[-1] == query.shape[0]` before calling real/fake FlashInfer.

Add an internal immutable `FlashInferPlanMetadata(lengths: PagedBatchLengths, kv_indptr, kv_last_page_len)` snapshot as `PagedAttentionBackend.last_flashinfer_plan` after each successful plan. It contains the canonical four length vectors plus page indices only, no prompt tokens, and gives CUDA tests a defined way to prove real cold/warm query geometry.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest -q tests/python/unit/test_flashinfer_attention_backend.py tests/python/serving/test_flashinfer_model_runner.py tests/python/serving/test_batch.py tests/python/serving/test_model_runner.py`

Expected: PASS for cold full query, warm suffix query, mixed query lengths, and synthetic chunked prefill.

```bash
git add moe_infinity/runtime/attention_types.py moe_infinity/serving/batch.py moe_infinity/serving/model_runner.py moe_infinity/runtime/attention_backend.py tests/python/unit/test_flashinfer_attention_backend.py tests/python/serving/test_flashinfer_model_runner.py tests/python/serving/test_batch.py
git commit -m "fix(serving): separate query and KV lengths"
```

### Task 4: Add refcounts, layer-complete COW, and preemption restore

**Files:**
- Modify: `moe_infinity/serving/kv_cache.py:48-379`
- Modify: `tests/python/serving/test_kv_cache.py`
- Modify: `tests/python/unit/test_kv_swap_recovery.py`

- [ ] **Step 1: Write failing ownership/COW/all-layer preemption tests**

```python
def test_partial_tail_cow_copies_all_layers() -> None:
    recording_layered_store = RecordingLayeredPagedKVStore(num_layers=3)
    cache = make_cache(store=recording_layered_store, num_blocks=4)
    cache.allocate_sequence(1, 3)
    old = cache.get_block_table(1)[0]
    cache.block_allocator.retain([old])
    cache.append_tokens(1, 1)
    new = cache.get_block_table(1)[0]
    assert recording_layered_store.copies == [(old, new, (0, 1, 2))]
    assert cache.block_allocator.ref_count(old) == 1

def test_swap_restore_preserves_every_layer_and_references() -> None:
    recording_layered_store = RecordingLayeredPagedKVStore(num_layers=3)
    cache = make_cache(store=recording_layered_store, num_blocks=6)
    cache.allocate_sequence(7, 8)
    before = recording_layered_store.layer_values(cache.get_block_table(7))
    cache.swap_out(7); cache.free_gpu_blocks(7); cache.swap_in(7)
    assert recording_layered_store.layer_values(cache.get_block_table(7)) == before
    assert all(cache.block_allocator.ref_count(i) == 1 for i in cache.get_block_table(7))

def test_binding_uses_one_owner_and_disables_independent_storage() -> None:
    backend = PagedAttentionBackend(
        KVCacheSpec(2, 8, torch.float32, 4),
        num_gpu_blocks=8, device=torch.device("cpu"),
    )
    store = backend.create_layered_store(layer_count=3)
    cache = make_cache(num_blocks=6)
    cache.set_block_store(store, owner=backend)
    assert cache.block_store is backend.block_store
    assert cache.block_store.owner is backend
    assert cache.num_blocks == 6 < store.num_blocks == 8
    assert cache._kv_cache is None
    assert cache._fi_prefill is None and cache._fi_decode is None

def test_binding_rejects_wrong_owner_rebind_and_active_tables() -> None:
    backend = PagedAttentionBackend(
        KVCacheSpec(2, 8, torch.float32, 4),
        num_gpu_blocks=8, device=torch.device("cpu"),
    )
    store = backend.create_layered_store(layer_count=3)
    cache = make_cache(num_blocks=6)
    with pytest.raises(ValueError, match="owner"):
        cache.set_block_store(store, owner=object())
    cache.allocate_sequence(1, 1)
    with pytest.raises(RuntimeError, match="before allocation"):
        cache.set_block_store(store, owner=backend)

def test_binding_rejects_logical_capacity_larger_than_physical() -> None:
    backend = PagedAttentionBackend(
        KVCacheSpec(2, 8, torch.float32, 4),
        num_gpu_blocks=4, device=torch.device("cpu"),
    )
    store = backend.create_layered_store(layer_count=3)
    with pytest.raises(ValueError, match="logical cache exceeds"):
        make_cache(num_blocks=6).set_block_store(store, owner=backend)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/python/serving/test_kv_cache.py tests/python/unit/test_kv_swap_recovery.py tests/python/dflash/test_kv_truncate.py`

Expected: FAIL because allocator references and authoritative all-layer store operations are absent.

- [ ] **Step 3: Implement exact ownership APIs**

```python
def set_block_store(
    self, store: LayeredPagedKVStore, *, owner: PagedAttentionBackend
) -> None:
    if self._block_store is store:
        if self._block_store_owner is not owner:
            raise ValueError("layered KV store owner mismatch")
        return
    if self._sequence_tables or self._swapped_out_sequences:
        raise RuntimeError("block store must be bound before allocation")
    if self._block_store is not None and self._block_store is not store:
        raise RuntimeError("paged KV cache cannot be rebound to another store")
    if getattr(owner, "block_store", None) is not store or store.owner is not owner:
        raise ValueError("layered KV store owner mismatch")
    if self.num_blocks > store.num_blocks:
        raise ValueError("logical cache exceeds layered store capacity")
    expected = (
        self.num_layers, self.block_size, self.num_heads,
        self.head_dim, self.dtype, self.device,
    )
    actual = (
        store.num_layers, store.block_size, store.num_kv_heads,
        store.head_dim, store.dtype, store.device,
    )
    if actual != expected:
        raise ValueError(
            f"layered KV store geometry mismatch: expected={expected}, actual={actual}"
        )
    self._block_store = store
    self._block_store_owner = owner
    self._kv_cache = None
    self._use_flashinfer = False
    self._fi_workspace = None
    self._fi_prefill = None
    self._fi_decode = None

def _copy_on_write_tail(self, block_table: BlockTable) -> None:
    old = block_table.get_block_ids()[-1]
    if self.block_allocator.ref_count(old) <= 1:
        return
    new = self.block_allocator.allocate(1)[0]
    try:
        payload = self._require_block_store().export_blocks([old])
        self._require_block_store().import_blocks([new], payload)
        block_table.replace_tail(new)
        self.block_allocator.release([old])
    except Exception:
        self.block_allocator.release([new])
        raise
```

Add checked `BlockAllocator.retain(ids)`, `release(ids)`, and `ref_count(id)`. Keep `free()` as a checked alias of `release()`. Change `_kv_cache` to `torch.Tensor | None`; after binding, `get_kv_cache_tensors`, swap, COW, checkpoint, restore, and all attention-visible paths must route to `block_store` or fail rather than reactivate `_kv_cache`. `swap_out` calls `store.checkpoint(tuple(block_ids))`; `swap_in` allocates destination IDs and calls `store.restore(tuple(destination_ids), checkpoint)` before exposing the table. Validate all IDs before count mutation. `truncate_tokens` calls COW before retaining a partial shared tail; full shared blocks remain immutable. Rebinding the same `(store, owner)` is an idempotent no-op; a different store/owner is rejected.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest -q tests/python/serving/test_kv_cache.py tests/python/unit/test_kv_swap_recovery.py tests/python/dflash/test_kv_truncate.py tests/python/unit/test_kv_cache_free.py tests/python/unit/test_kv_edge_cases.py`

Expected: PASS; copy/preemption/restoration verifies every layer and refcount.

```bash
git add moe_infinity/serving/kv_cache.py tests/python/serving/test_kv_cache.py tests/python/unit/test_kv_swap_recovery.py
git commit -m "feat(serving): own shared KV blocks safely"
```

### Task 5: Make sequence-group admission transactional

**Files:**
- Modify: `moe_infinity/serving/sequence.py:34-56`
- Modify: `moe_infinity/serving/scheduler.py:212-328,381-599`
- Modify: `moe_infinity/serving/kv_cache.py`
- Modify: `tests/python/serving/test_scheduler.py`
- Modify: `tests/python/serving/test_engine.py`

- [ ] **Step 1: Write failing pin-before-evict and `n>1` atomicity tests**

```python
def test_group_pins_all_matches_before_eviction() -> None:
    scheduler, _, prefix, _ = make_seeded_scheduler(num_blocks=3, max_batch_size=2)
    scheduler.add_request(make_group("n", [(1, SHARED+[10]), (2, SHARED+[11])]))
    scheduler.schedule()
    assert prefix.events.index("pin:10") < prefix.events.index("evict")
    assert prefix.events.index("pin:11") < prefix.events.index("evict")

def test_n_group_allocation_failure_rolls_back_every_sequence_and_lease() -> None:
    scheduler, cache, prefix, _ = make_seeded_scheduler(num_blocks=3, max_batch_size=2)
    group = make_group("n", [(1, SHARED+[10]), (2, SHARED+[11])])
    initial_free = cache.block_allocator.num_free_blocks
    initial_refs = tuple(cache.block_allocator.ref_count(i) for i in range(cache.num_blocks))
    scheduler.add_request(group)
    output = scheduler.schedule()
    assert output.prefill_seq_ids == []
    assert [seq.status for seq in group.sequences] == [SequenceStatus.WAITING, SequenceStatus.WAITING]
    assert cache.block_allocator.num_free_blocks == initial_free
    assert tuple(cache.block_allocator.ref_count(i) for i in range(cache.num_blocks)) == initial_refs
    assert prefix.open_leases == 0

def test_warm_n_group_commits_all_leases_together() -> None:
    scheduler, cache, prefix, _ = make_seeded_scheduler(num_blocks=8, max_batch_size=3)
    group = make_group("n", [(1, SHARED+[10]), (2, SHARED+[11]), (3, SHARED+[12])])
    scheduler.add_request(group)
    output = scheduler.schedule()
    assert output.prefill_seq_ids == [1, 2, 3]
    assert {tuple(cache.get_block_table(seq_id)[:2]) for seq_id in (1, 2, 3)} == {(0, 1)}
    assert cache.block_allocator.ref_count(0) == 4  # cache + three sequences
    assert prefix.open_leases == 0

def test_second_lease_prepare_failure_aborts_whole_group(monkeypatch) -> None:
    scheduler, cache, prefix, _ = make_seeded_scheduler(num_blocks=8, max_batch_size=2)
    group = make_group("n", [(1, SHARED+[10]), (2, SHARED+[11])])
    original = PrefixLease.prepare_adoption
    calls = 0
    def fail_second(self, owner):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second lease prepare failed")
        return original(self, owner)
    monkeypatch.setattr(PrefixLease, "prepare_adoption", fail_second)
    initial_free = cache.block_allocator.num_free_blocks
    initial_refs = tuple(cache.block_allocator.ref_count(i) for i in range(cache.num_blocks))
    scheduler.add_request(group)
    with pytest.raises(RuntimeError, match="second lease prepare failed"):
        scheduler.schedule()
    assert not cache.has_sequence(1) and not cache.has_sequence(2)
    assert cache.block_allocator.num_free_blocks == initial_free
    assert tuple(cache.block_allocator.ref_count(i) for i in range(cache.num_blocks)) == initial_refs
    assert prefix.open_leases == 0
    assert [seq.status for seq in group.sequences] == [SequenceStatus.WAITING, SequenceStatus.WAITING]

def test_explicit_group_abort_rolls_back_staged_tables_and_prepared_leases() -> None:
    _, cache, prefix, namespace = make_seeded_scheduler(num_blocks=8, max_batch_size=2)
    leases = [
        prefix.acquire_prefix_lease(namespace, SHARED+[10], 8),
        prefix.acquire_prefix_lease(namespace, SHARED+[11], 8),
    ]
    plans = [
        SequenceAllocationPlan(1, 9, 8, list(leases[0].match.block_ids)),
        SequenceAllocationPlan(2, 9, 8, list(leases[1].match.block_ids)),
    ]
    initial_free = cache.block_allocator.num_free_blocks
    receipt = cache.prepare_group(plans, leases)
    assert all(lease.state == "prepared" for lease in leases)
    assert not cache.has_sequence(1) and not cache.has_sequence(2)
    cache.abort_group(receipt)
    assert all(lease.state == "aborted" for lease in leases)
    assert cache.block_allocator.num_free_blocks == initial_free
    assert prefix.open_leases == 0

def test_chunked_prefill_uses_same_optional_provider_once() -> None:
    scheduler, _, provider, _ = make_seeded_scheduler(num_blocks=8, max_batch_size=1)
    sequence = make_group("chunk", [(7, SHARED+[10,11,12,13])]).sequences[0]
    first = scheduler._acquire_prefill_lease(sequence, max_prefix_tokens=8)
    assert first.match.num_tokens == 8
    owner = object()
    first.prepare_adoption(owner); first.commit_adoption(owner)
    sequence.has_prefix_lease = True
    second = scheduler._acquire_prefill_lease(sequence, max_prefix_tokens=8)
    assert second.match.num_tokens == 0
    second_owner = object()
    second.prepare_adoption(second_owner); second.commit_adoption(second_owner)
    assert [event for event in provider.events if event.startswith("pin:")] == ["pin:13"]
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/python/serving/test_scheduler.py tests/python/serving/test_engine.py`

Expected: FAIL because lookup is not leased and admission mutates one sequence at a time.

- [ ] **Step 3: Implement atomic planning/adoption**

```python
@dataclass(frozen=True)
class SequenceAllocationPlan:
    seq_id: int
    total_tokens: int
    prefix_tokens: int
    pinned_block_ids: list[int]

@dataclass(frozen=True)
class GroupAllocationReceipt:
    owner: object
    seq_ids: list[int]
    new_block_ids: list[int]
    staged_tables: dict[int, BlockTable]
    leases: tuple[PrefixLease, ...]
    state: str = "prepared"

def _acquire_prefill_lease(self, sequence: SequenceData, max_prefix_tokens: int) -> PrefixLease:
    provider = self.prefix_lease_provider
    namespace = self.cache_namespace
    if provider is None or namespace is None or sequence.has_prefix_lease:
        return PrefixLease.empty()
    return provider.acquire_prefix_lease(
        namespace, sequence.prompt_token_ids, max_prefix_tokens=max_prefix_tokens
    )

def prepare_group(
    self, plans: list[SequenceAllocationPlan], leases: list[PrefixLease]
) -> GroupAllocationReceipt:
    if len(plans) != len(leases):
        raise ValueError("one prefix lease is required per sequence plan")
    seq_ids = [plan.seq_id for plan in plans]
    if len(set(seq_ids)) != len(seq_ids) or any(seq_id in self._sequence_tables for seq_id in seq_ids):
        raise ValueError("group sequence ids must be unique and unallocated")
    suffix_counts: list[int] = []
    for plan in plans:
        if plan.prefix_tokens < 0 or plan.prefix_tokens > plan.total_tokens:
            raise ValueError("invalid pinned prefix length")
        if plan.prefix_tokens % self.block_size != 0:
            raise ValueError("pinned prefixes must end on a block boundary")
        if len(plan.pinned_block_ids) != plan.prefix_tokens // self.block_size:
            raise ValueError("pinned block count does not match prefix length")
        if any(self.block_allocator.ref_count(block_id) <= 0 for block_id in plan.pinned_block_ids):
            raise ValueError("pinned block lost its lease reference")
        suffix = plan.total_tokens - plan.prefix_tokens
        suffix_counts.append((suffix + self.block_size - 1) // self.block_size)

    owner = object()
    new_ids: list[int] = []
    staged: dict[int, BlockTable] = {}
    try:
        for plan, lease in zip(plans, leases):
            match = lease.prepare_adoption(owner)
            if match.num_tokens != plan.prefix_tokens or match.block_ids != tuple(plan.pinned_block_ids):
                raise ValueError("lease match does not match sequence allocation plan")
        new_ids = self.block_allocator.allocate(sum(suffix_counts))
        cursor = 0
        for plan, count in zip(plans, suffix_counts):
            owned = new_ids[cursor : cursor + count]
            cursor += count
            table = BlockTable(self.block_allocator)
            table.restore_blocks([*plan.pinned_block_ids, *owned], plan.total_tokens)
            staged[plan.seq_id] = table
        return GroupAllocationReceipt(
            owner, list(seq_ids), list(new_ids), staged, tuple(leases)
        )
    except Exception:
        if new_ids:
            self.block_allocator.release(new_ids)
        for lease in reversed(leases):
            if lease.state == "prepared":
                lease.abort(owner)
            elif lease.state == "open":
                lease.abort()
        raise

def commit_group(self, receipt: GroupAllocationReceipt) -> None:
    if receipt.state != "prepared":
        raise RuntimeError("group allocation receipt is not prepared")
    if any(not lease.is_prepared_for(receipt.owner) for lease in receipt.leases):
        raise RuntimeError("all group leases must be prepared before commit")
    if any(seq_id in self._sequence_tables for seq_id in receipt.seq_ids):
        raise RuntimeError("group sequence became allocated before commit")
    # commit_adoption is no-fail after this preflight: it only changes local state
    # and invokes the PrefixCache terminal callback, which must be no-throw.
    for lease in receipt.leases:
        lease.commit_adoption(receipt.owner)
    self._sequence_tables.update(receipt.staged_tables)
    object.__setattr__(receipt, "state", "committed")

def abort_group(self, receipt: GroupAllocationReceipt) -> None:
    if receipt.state != "prepared":
        raise RuntimeError("only a prepared group may be aborted")
    for seq_id in receipt.seq_ids:
        self._sequence_tables.pop(seq_id, None)
    if receipt.new_block_ids:
        self.block_allocator.release(list(receipt.new_block_ids))
    for lease in reversed(receipt.leases):
        if lease.state == "prepared":
            lease.abort(receipt.owner)
        elif lease.state == "open":
            lease.abort()
    object.__setattr__(receipt, "state", "aborted")
```

Scheduler order is mandatory: (a) obtain one lease per sequence from the optional `PrefixLeaseProvider`; with `None`, create open zero-block leases locally; (b) compute aggregate suffix-token/block budget; (c) evict while leases remain pinned; (d) preempt if still required; (e) call `prepare_group(plans, leases)`, which prepares every lease against one opaque owner and stages allocations/tables without publishing them; (f) call `commit_group(receipt)` only after every sequence prepared, then mutate all statuses and committed prefix counters together. Any exception before commit calls `abort_group(receipt)` when a receipt exists; `prepare_group` itself aborts every open/prepared lease and allocation if it cannot return a receipt. Do not schedule a subset of a group. `commit_group` has no fallible work after its complete preflight; terminal callbacks are required to be no-throw. CP ordering runs before this sequence and does not provide matches. A chunked-prefill scheduler uses the same `prepare_group`/`commit_group` protocol before its first chunk and stores committed block-table ownership on `SequenceData`; execution rollback later releases ordinary sequence ownership, never an already-committed lease.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest -q tests/python/serving/test_scheduler.py tests/python/serving/test_engine.py tests/python/contextpilot/test_cp_scheduler_v2.py tests/python/contextpilot/test_request_id_lifecycle.py`

Expected: PASS; event order proves all pins precede eviction and failed `n>1` admission leaves no tables, refs, or status changes.

```bash
git add moe_infinity/serving/sequence.py moe_infinity/serving/scheduler.py moe_infinity/serving/kv_cache.py tests/python/serving/test_scheduler.py tests/python/serving/test_engine.py
git commit -m "feat(serving): admit shared-prefix groups atomically"
```

### Task 6: Commit ranges, publish safely, and reconcile DFlash

**Files:**
- Modify: `moe_infinity/serving/sequence.py`
- Modify: `moe_infinity/serving/batch.py`
- Modify: `moe_infinity/serving/engine.py:208-305,307-555`
- Modify: `tests/python/serving/test_engine.py`
- Modify: `tests/python/serving/test_correctness.py`
- Modify: `tests/python/serving/test_dflash_engine_step5.py`

- [ ] **Step 1: Write failing committed-range/chunk/DFlash tests**

```python
def test_publication_uses_successfully_committed_range_only(cb_engine_factory) -> None:
    engine = make_prefix_capable_engine(cb_engine_factory)
    seq = add_prefill(engine, prompt=list(range(20)), committed=8)
    batch = make_prefill_batch(
        seq, context_len=8, query_tokens=list(range(8, 12)),
        block_table=engine.kv_cache.get_block_table(seq.seq_id),
    )
    engine._execute_and_commit(batch)
    assert seq.committed_kv_tokens == 12
    lease = engine.prefix_cache.acquire_prefix_lease(
        engine.cache_namespace, list(range(20)), max_prefix_tokens=16
    )
    assert lease.match.num_tokens == 12
    lease.abort()

def test_failed_chunk_does_not_advance_or_publish(monkeypatch: pytest.MonkeyPatch, cb_engine_factory) -> None:
    engine = make_prefix_capable_engine(cb_engine_factory)
    seq = add_prefill(engine, prompt=list(range(20)), committed=8)
    monkeypatch.setattr(engine, "_execute_batch", Mock(side_effect=RuntimeError("forward failed")))
    batch = make_prefill_batch(seq, 8, [8,9,10,11], engine.kv_cache.get_block_table(seq.seq_id))
    with pytest.raises(RuntimeError): engine._execute_and_commit(batch)
    assert seq.committed_kv_tokens == 8

def test_reused_prefix_disables_dflash_delegation(cb_engine_factory) -> None:
    engine, batch = make_dflash_engine_and_batch(cb_engine_factory, context_len=16, has_prefix_lease=True)
    assert engine._can_delegate_speculative(batch) is False
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/python/serving/test_engine.py tests/python/serving/test_correctness.py tests/python/serving/test_dflash_engine_step5.py`

Expected: FAIL because committed KV range and publication boundaries are not represented.

- [ ] **Step 3: Implement post-forward commit/publication**

Add `SequenceData.committed_kv_tokens` and `has_prefix_lease`. Add `_execute_and_commit(batch)`: call `_execute_batch`, and only on return advance each prefill sequence exactly by its `batch.lengths.query_lengths` value and publish newly complete prompt blocks up to `committed_kv_tokens`. Route ordinary `step()` through this method. Repeated publication is idempotent by exact entry path. If a synthetic chunk ends mid-block, defer that block until a later successful chunk completes it. Sample a first output only when the submitted query reaches prompt end; the current non-chunking scheduler always does so.

Keep `_can_delegate_speculative` requiring `batch.lengths.kv_seq_lengths == batch.lengths.query_lengths`, `has_prefix_lease=False`, fresh singleton, and all existing greedy constraints. Delegated DFlash returns before paged execution and therefore does not call publication. Verify-session `committed_counts` affect DFlash's own cache only.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest -q tests/python/serving/test_engine.py tests/python/serving/test_correctness.py tests/python/serving/test_batch.py tests/python/serving/test_dflash_engine_step5.py tests/python/dflash/test_kv_truncate.py tests/python/dflash/test_spec_verify.py`

Expected: PASS; failed or partial chunks do not over-publish and reused prefixes never enter DFlash.

```bash
git add moe_infinity/serving/sequence.py moe_infinity/serving/batch.py moe_infinity/serving/engine.py tests/python/serving/test_engine.py tests/python/serving/test_correctness.py tests/python/serving/test_dflash_engine_step5.py
git commit -m "feat(serving): publish committed prefix KV ranges"
```

### Task 7: Gate/configure the active OpenAI path and expose lifecycle metrics

**Files:**
- Modify: `moe_infinity/serving/engine.py:84-168,619-690`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:480-526,716-739,1020-1099,1708-1919`
- Modify: `tests/python/serving/test_hot_reload.py`
- Modify: `tests/python/serving/test_api_routes.py`
- Modify: `tests/python/serving/test_cancellation.py`
- Modify: `tests/python/serving/test_engine.py`
- Modify: `tests/python/unit/test_v2_lifespan.py`

- [ ] **Step 1: Write failing capability/binding/reload/refcount/metrics tests**

```python
def test_engine_binds_validated_store_before_scheduler(monkeypatch, cb_base_config, mock_model, mock_engine_obj) -> None:
    events: list[str] = []
    store = RecordingLayeredPagedKVStore(num_layers=2, num_blocks=32)
    monkeypatch.setattr(ModelRunner, "get_prefix_reuse_capability", lambda self: PrefixReuseCapability.active(store.owner, store))
    original_bind = PagedKVCache.set_block_store
    monkeypatch.setattr(PagedKVCache, "set_block_store", lambda self, value, *, owner: (events.append("bind"), original_bind(self, value, owner=owner))[1])
    original_init = Scheduler.__init__
    def scheduler_init(self, *args, **kwargs):
        events.append("scheduler")
        assert args[0]._block_store is store
        assert kwargs["prefix_lease_provider"] is not None
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(Scheduler, "__init__", scheduler_init)
    ContinuousBatchingEngine(mock_model, mock_engine_obj, {**cb_base_config, "enable_prefix_caching": True})
    assert events == ["bind", "scheduler"]

def test_store_geometry_mismatch_fails_closed(monkeypatch, cb_engine_factory) -> None:
    bad_store = RecordingLayeredPagedKVStore(num_layers=99, num_blocks=32)
    monkeypatch.setattr(ModelRunner, "get_prefix_reuse_capability", lambda self: PrefixReuseCapability.active(bad_store.owner, bad_store))
    engine = cb_engine_factory(config_overrides={"enable_prefix_caching": True})
    assert engine.prefix_cache is None
    assert engine.scheduler.prefix_lease_provider is None
    assert engine.get_stats()["prefix_cache_disabled_reason"] == "kv-store-binding-mismatch"

@pytest.mark.parametrize(
    ("budget_blocks", "store_blocks", "expected_logical"),
    [(6, 8, 6), (12, 8, 8)],
)
def test_prefix_cache_physical_capacity_may_exceed_logical(
    monkeypatch, cb_engine_factory, budget_blocks, store_blocks, expected_logical
) -> None:
    backend = make_paged_backend(num_blocks=store_blocks)
    store = backend.create_layered_store(layer_count=2)
    monkeypatch.setattr(
        ModelRunner, "get_prefix_reuse_capability",
        lambda self: PrefixReuseCapability.active(backend, store),
    )
    engine = cb_engine_factory(config_overrides={
        "enable_prefix_caching": True, "num_kv_blocks": budget_blocks,
    })
    assert engine.kv_cache.num_blocks == expected_logical
    assert engine.kv_cache.block_store is backend.block_store
    assert engine.scheduler.prefix_lease_provider is engine.prefix_cache
```

Also test `--prefix-cache-max-entries` default/validation, disabled zero-cost construction, `incomplete-paged-layer-registry`, `prefix-aware-prefill-unavailable`, successful `/v1/reload` invalidation, failed reload preservation, and exact completion/cancellation reference drops.

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/python/serving/test_engine.py tests/python/serving/test_hot_reload.py tests/python/serving/test_api_routes.py tests/python/serving/test_cancellation.py tests/python/unit/test_v2_lifespan.py`

Expected: FAIL because the active engine does not gate or report real reuse.

- [ ] **Step 3: Wire config, invalidation, and observability**

Enable only for `Qwen3PagedAttention` with complete layer registry, matching geometry, and real FlashInfer prefill/decode. Construct `ModelRunner` first, resolve the existing production backend, register its complete layer set, and initialize storage exactly once through `backend.create_layered_store(layer_count=expected_layers)`. Never pass `num_layers` to `PagedAttentionBackend(...)` and never create a second backend/store for prefix reuse. Set `logical_num_blocks=min(memory_budget_blocks, backend.block_store.num_blocks)` and require physical capacity greater than or equal to logical capacity. Add `--prefix-cache-max-entries` default `1000`, minimum `1`, startup-only. Namespace from model/tokenizer/adapter/runtime values; increment runtime epoch on invalidation.

```python
self.model_runner = ModelRunner(model, engine, device=self.device)
capability = self.model_runner.get_prefix_reuse_capability()
physical_blocks = (
    capability.block_store.num_blocks
    if capability.supported and capability.block_store is not None
    else memory_budget_blocks
)
logical_num_blocks = min(memory_budget_blocks, physical_blocks)
self.kv_cache = PagedKVCache(
    num_blocks=logical_num_blocks, block_size=block_size, num_layers=num_layers,
    num_heads=num_kv_heads, head_dim=head_dim, dtype=self.dtype, device=self.device,
)
provider: PrefixLeaseProvider | None = None
if self._get_bool_config("enable_prefix_caching", False) and capability.supported:
    assert capability.backend is not None and capability.block_store is not None
    try:
        self.kv_cache.set_block_store(
            capability.block_store, owner=capability.backend
        )
    except (RuntimeError, ValueError):
        self._prefix_cache_disabled_reason = "kv-store-binding-mismatch"
    else:
        self.prefix_cache = PrefixCache(
            block_size=self.kv_cache.block_size,
            max_entries=self._get_int_config("prefix_cache_max_entries", 1000),
            on_retain=self.kv_cache.block_allocator.retain,
            on_release=self.kv_cache.block_allocator.release,
        )
        provider = self.prefix_cache
self.scheduler = Scheduler(
    self.kv_cache,
    max_batch_size=self._get_int_config("max_batch_size", 32),
    max_tokens_per_step=self._get_int_config("max_tokens_per_step", 2048),
    prefix_lease_provider=provider,
    cache_namespace=self.cache_namespace if provider is not None else None,
)
```

No constructor path may pass a provider before `set_block_store` succeeds. `PrefixReuseCapability.active(backend, store)` carries both identities and validates `backend.block_store is store` plus `store.owner is backend`; unsupported capabilities carry both as `None` and a stable reason. Add tests for physical capacities `(budget=6, store=8) -> logical=6` and `(budget=12, store=8) -> logical=8`, and reject only `logical > physical`, owner mismatch, active-table binding, or attempted rebinding.

Expose `prefix_cache_enabled`, `active`, `disabled_reason`, `entries`, `open_leases`, `hits_total`, `misses_total`, `matched_tokens_total`, `inserted_blocks_total`, `evicted_blocks_total`, `collision_checks_total`, and `invalidations_total` in `/admin/stats` and zero-safe `moe_prefix_cache_*` Prometheus metrics without request/token labels. `/v1/reload` calls `invalidate_prefix_cache("module-reload")` once after any successful module reload.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest -q tests/python/serving/test_engine.py tests/python/serving/test_hot_reload.py tests/python/serving/test_api_routes.py tests/python/serving/test_cancellation.py tests/python/unit/test_v2_lifespan.py tests/python/contextpilot/test_eviction_sync.py tests/python/contextpilot/test_eviction_parity.py`

Expected: PASS with balanced refs and unchanged CP lifecycle.

```bash
git add moe_infinity/serving/engine.py moe_infinity/entrypoints/openai/api_server_v2.py tests/python/serving/test_engine.py tests/python/serving/test_hot_reload.py tests/python/serving/test_api_routes.py tests/python/serving/test_cancellation.py tests/python/unit/test_v2_lifespan.py
git commit -m "feat(serving): operate prefix KV reuse"
```

### Task 8: Prove real-FlashInfer cold/warm execution and logits on supported Qwen3

**Files:**
- Create: `tests/python/serving/test_prefix_cache_cuda.py`

- [ ] **Step 1: Create a self-contained CUDA fixture and non-tautological tests**

Use the supported checkpoint `Qwen/Qwen3-30B-A3B` explicitly; do not accept an arbitrary model environment variable. The fixture loads `AutoTokenizer` and `MoE`, requires `Qwen3PagedAttention`, real FlashInfer wrappers, complete layer registry, and `prefix_cache_active=True`, otherwise fails rollout rather than silently skipping after CUDA is requested.

```python
MODEL = "Qwen/Qwen3-30B-A3B"

@pytest.fixture
def qwen3_engine_factory(tmp_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    def factory(*, enable_prefix_caching: bool, prefix_cache_max_entries: int = 1000):
        owner = MoE(MODEL, {
            "offload_path": str(tmp_path / ("enabled" if enable_prefix_caching else "disabled")),
            "device_memory_ratio": 0.5,
        })
        args = SimpleNamespace(
            device_memory_ratio=0.5, kv_cache_ratio=0.25, max_batch_size=4,
            enable_prefix_caching=enable_prefix_caching,
            prefix_cache_max_entries=prefix_cache_max_entries,
        )
        config = _build_engine_config(args, owner.model)
        engine = ContinuousBatchingEngine(owner.model, owner.engine, config, tokenizer=tokenizer)
        if enable_prefix_caching:
            assert engine.get_stats()["prefix_cache_active"] is True
            assert engine.model_runner.get_prefix_reuse_capability(engine.kv_cache).reason == "active"
        return engine
    return factory, tokenizer

def tokenize_to_at_least_64_tokens(tokenizer) -> list[int]:
    ids = tokenizer.encode("Layer-complete shared prefix for exact KV reuse. ", add_special_tokens=False)
    return (ids * ((64 + len(ids) - 1) // len(ids)))[:64]

def unrelated_64_tokens(tokenizer) -> list[int]:
    ids = tokenizer.encode("Unrelated eviction pressure sequence. ", add_special_tokens=False)
    return (ids * ((64 + len(ids) - 1) // len(ids)))[:64]

def run_with_logits(engine, request_id: str, prompt: list[int]) -> tuple[list[int], torch.Tensor, list[FlashInferPlanMetadata]]:
    captured: list[torch.Tensor] = []
    plans: list[FlashInferPlanMetadata] = []
    original = engine._execute_batch
    def capture(batch):
        logits = original(batch)
        captured.append(engine._extract_last_token_logits(logits, batch).detach().float().cpu())
        plans.append(engine.model_runner._get_attention_backend().last_flashinfer_plan)
        return logits
    engine._execute_batch = capture
    try:
        engine.add_request(request_id, prompt, SamplingParams(temperature=0.0, max_tokens=4))
        return cast(list[int], engine.run_until_done()[request_id]), torch.cat(captured), plans
    finally:
        engine._execute_batch = original

def test_real_flashinfer_warm_suffix_matches_cold_logits(qwen3_engine_factory) -> None:
    factory, tokenizer = qwen3_engine_factory
    shared = tokenize_to_at_least_64_tokens(tokenizer)
    cold = factory(enable_prefix_caching=False)
    warm = factory(enable_prefix_caching=True)
    run_with_logits(warm, "prime", shared + [100])
    cold_tokens, cold_logits, cold_plans = run_with_logits(cold, "cold", shared + [101])
    warm_tokens, warm_logits, warm_plans = run_with_logits(warm, "warm", shared + [101])
    assert warm_tokens == cold_tokens
    torch.testing.assert_close(warm_logits, cold_logits, rtol=2e-3, atol=2e-3)
    cold_prefill, warm_prefill = cold_plans[0], warm_plans[0]
    assert cold_prefill.lengths.query_offsets.tolist() == [0, len(shared) + 1]
    assert warm_prefill.lengths.query_offsets.tolist() == [0, 1]
    assert warm_prefill.lengths.context_lengths.tolist() == [len(shared)]
    assert warm_prefill.lengths.kv_seq_lengths.tolist() == [len(shared) + 1]

def test_active_reference_survives_lru_eviction(qwen3_engine_factory) -> None:
    factory, tokenizer = qwen3_engine_factory
    baseline = factory(enable_prefix_caching=False)
    engine = factory(enable_prefix_caching=True, prefix_cache_max_entries=4)
    shared = tokenize_to_at_least_64_tokens(tokenizer)
    expected, _, _ = run_with_logits(baseline, "expected", shared + [200])
    run_with_logits(engine, "prime", shared + [199])
    engine.add_request("active", shared + [200], SamplingParams(temperature=0.0, max_tokens=4))
    engine.step()
    active_seq = engine._request_to_seq_ids["active"][0]
    shared_block = engine.kv_cache.get_block_table(active_seq)[0]
    before = engine.kv_cache.block_allocator.ref_count(shared_block)
    engine.add_request("pressure", unrelated_64_tokens(tokenizer), SamplingParams(temperature=0.0, max_tokens=1))
    engine.step()  # pressure publishes and evicts old cache ownership; active still decodes
    during_active = engine.kv_cache.block_allocator.ref_count(shared_block)
    outputs = engine.run_until_done()
    after_eviction = engine.kv_cache.block_allocator.ref_count(shared_block)
    assert outputs["active"] == expected
    assert before >= 2
    assert during_active == 1  # active sequence lease/reference only
    assert after_eviction == 0  # active completion released the final sequence ref
```

- [ ] **Step 2: Run CPU collection and CUDA execution**

Run: `python -m pytest -q tests/python/serving/test_prefix_cache_cuda.py`

Expected: SKIPPED only because `MOE_PREFIX_CACHE_CUDA=1` is absent.

Run: `MOE_PREFIX_CACHE_CUDA=1 python -m pytest -q tests/python/serving/test_prefix_cache_cuda.py`

Expected: PASS on a Qwen3/FlashInfer CUDA runner; cold and warm execute real wrappers, logits/tokens match, and active-reference eviction has exact refcount transitions.

- [ ] **Step 3: Commit**

```bash
git add tests/python/serving/test_prefix_cache_cuda.py
git commit -m "test(serving): prove Qwen3 FlashInfer prefix parity"
```

### Task 9: Add benchmark, documentation, rollback, and motivation

**Files:**
- Create: `benchmarks/serving/prefix_cache_benchmark.py`
- Create: `tests/python/serving/test_prefix_cache_benchmark.py`
- Modify: `docs/serving.md:49-60,86,150-157,189-200`
- Modify: `docs/benchmarking.md:142-178`
- Modify: `README.md:42,276-318`
- Modify: `ARCHITECTURE.md:71-90,194-216`
- Modify: `CHANGELOG.md:5-32`

- [ ] **Step 1: Write failing dry-run schema test**

```python
def test_dry_run_isolates_modes_and_reports_geometry(tmp_path: Path) -> None:
    output = tmp_path / "prefix.json"
    proc = subprocess.run(
        [sys.executable, "benchmarks/serving/prefix_cache_benchmark.py",
         "--dry-run", "--output-json", str(output)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output.read_text())
    assert list(payload["modes"]) == ["disabled", "enabled_cold", "enabled_warm"]
    ids = [payload["modes"][name]["engine_instance_id"] for name in payload["modes"]]
    assert len(set(ids)) == 3
    assert payload["modes"]["disabled"]["prefix_cache_active"] is False
    assert payload["modes"]["enabled_cold"]["hits_total"] == 0
    assert payload["modes"]["enabled_warm"]["hits_total"] > 0
    warm = payload["modes"]["enabled_warm"]
    assert warm["geometry"]["query_offsets"][-1] == sum(warm["geometry"]["query_lengths"])
    assert warm["geometry"]["context_lengths"][0] + warm["geometry"]["query_lengths"][0] == warm["geometry"]["kv_seq_lengths"][0]
    assert warm["geometry"]["query_lengths"][0] < warm["geometry"]["kv_seq_lengths"][0]
    assert warm["refcount_high_water"] >= 2
    assert payload["correctness"] == {"token_digests_equal": True, "logit_digests_equal": True}

def test_dry_run_aborts_on_digest_mismatch(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "benchmarks/serving/prefix_cache_benchmark.py",
         "--dry-run", "--dry-run-force-mismatch", "--output-json", str(tmp_path / "bad.json")],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 2
    assert "disabled/cold/warm digest mismatch" in proc.stderr
    assert not (tmp_path / "bad.json").exists()
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest -q tests/python/serving/test_prefix_cache_benchmark.py`

Expected: FAIL because the benchmark script does not exist.

- [ ] **Step 3: Implement benchmark and docs**

```python
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from moe_infinity.entrypoints.big_modeling import MoE
from moe_infinity.entrypoints.openai.api_server_v2 import _build_engine_config
from moe_infinity.runtime.attention_backend import FlashInferPlanMetadata
from moe_infinity.runtime.attention_types import PagedBatchLengths
from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.sequence import SamplingParams

@dataclass(frozen=True)
class Geometry:
    query_lengths: list[int]
    query_offsets: list[int]
    context_lengths: list[int]
    kv_seq_lengths: list[int]

@dataclass(frozen=True)
class ModeResult:
    engine_instance_id: str
    prefix_cache_active: bool
    geometry: Geometry
    hits_total: int
    matched_tokens_total: int
    open_leases: int
    refcount_high_water: int
    token_digest: str
    logit_digest: str
    ttft_ms: list[float]
    e2e_ms: list[float]

@dataclass(frozen=True)
class RunSample:
    token_ids: torch.Tensor
    last_token_logits: torch.Tensor
    plan: FlashInferPlanMetadata
    ttft_ms: list[float]
    e2e_ms: list[float]

class DryBenchmarkEngine:
    def __init__(self, enabled: bool) -> None:
        self.enabled, self.primed = enabled, False
        self.instance_id = str(uuid.uuid4())
        self.refcount_high_water = 1
        self._hits = 0

    def run(self, prompt: list[int], measured: bool) -> RunSample:
        warm = measured and self.enabled and self.primed
        if warm:
            self._hits += 1
            self.refcount_high_water = 2
        query = [1] if warm else [len(prompt)]
        plan = FlashInferPlanMetadata(
            lengths=PagedBatchLengths(
                query_lengths=torch.tensor(query),
                query_offsets=torch.tensor([0, query[0]]),
                context_lengths=torch.tensor([len(prompt) - query[0]]),
                kv_seq_lengths=torch.tensor([len(prompt)]),
            ),
            kv_indptr=torch.tensor([0, 1]),
            kv_last_page_len=torch.tensor([len(prompt) % 16 or 16]),
        )
        self.primed = self.primed or not measured
        return RunSample(
            token_ids=torch.tensor([7, 8]), last_token_logits=torch.tensor([[0.25, 0.75]]),
            plan=plan, ttft_ms=[1.0], e2e_ms=[2.0],
        )

    def stats(self) -> dict[str, object]:
        return {
            "prefix_cache_active": self.enabled,
            "prefix_cache_hits_total": self._hits,
            "prefix_cache_matched_tokens_total": 64 if self._hits else 0,
            "prefix_cache_open_leases": 0,
        }

class RealBenchmarkEngine:
    def __init__(self, engine: ContinuousBatchingEngine) -> None:
        self.engine, self.instance_id = engine, str(uuid.uuid4())
        self.refcount_high_water = 0

    def run(self, prompt: list[int], measured: bool) -> RunSample:
        request_id, plans, logits, outputs = str(uuid.uuid4()), [], [], []
        started = time.perf_counter_ns()
        first_token_ns: int | None = None
        original = self.engine._execute_batch
        def capture(batch):
            result = original(batch)
            plans.append(self.engine.model_runner._get_attention_backend().last_flashinfer_plan)
            logits.append(self.engine._extract_last_token_logits(result, batch).detach().float().cpu())
            return result
        def on_token(output):
            nonlocal first_token_ns
            first_token_ns = first_token_ns or time.perf_counter_ns()
            outputs.append(output.token_id)
        self.engine._execute_batch = capture
        try:
            self.engine.add_request(request_id, prompt, SamplingParams(temperature=0.0, max_tokens=2), on_token=on_token)
            while self.engine.has_pending_requests():
                self.engine.step()
                allocator = self.engine.kv_cache.block_allocator
                self.refcount_high_water = max(
                    self.refcount_high_water,
                    max((allocator.ref_count(i) for i in range(allocator.num_blocks)), default=0),
                )
        finally:
            self.engine._execute_batch = original
        ended = time.perf_counter_ns()
        assert first_token_ns is not None and plans and logits
        return RunSample(
            token_ids=torch.tensor(outputs), last_token_logits=torch.cat(logits), plan=plans[0],
            ttft_ms=[(first_token_ns - started) / 1e6], e2e_ms=[(ended - started) / 1e6],
        )

    def stats(self) -> dict[str, object]:
        return self.engine.get_stats()

def digest_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()

def run_mode(
    name: str, factory: Callable[[bool], DryBenchmarkEngine | RealBenchmarkEngine],
    prime_prompt: list[int], measured_prompt: list[int], force_mismatch: bool,
) -> ModeResult:
    enabled = name != "disabled"
    engine = factory(enabled)
    if name == "enabled_warm":
        engine.run(prime_prompt, measured=False)
    sample = engine.run(measured_prompt, measured=True)
    token_digest = digest_tensor(sample.token_ids)
    if force_mismatch and name == "enabled_warm":
        token_digest = "forced-mismatch"
    stats = engine.stats()
    return ModeResult(
        engine_instance_id=engine.instance_id,
        prefix_cache_active=bool(stats["prefix_cache_active"]),
        geometry=Geometry(
            query_lengths=sample.plan.lengths.query_lengths.tolist(),
            query_offsets=sample.plan.lengths.query_offsets.tolist(),
            context_lengths=sample.plan.lengths.context_lengths.tolist(),
            kv_seq_lengths=sample.plan.lengths.kv_seq_lengths.tolist(),
        ),
        hits_total=int(stats["prefix_cache_hits_total"]),
        matched_tokens_total=int(stats["prefix_cache_matched_tokens_total"]),
        open_leases=int(stats["prefix_cache_open_leases"]),
        refcount_high_water=engine.refcount_high_water,
        token_digest=token_digest,
        logit_digest=digest_tensor(sample.last_token_logits),
        ttft_ms=list(sample.ttft_ms), e2e_ms=list(sample.e2e_ms),
    )

def run_suite(factory, prime_prompt, measured_prompt, force_mismatch=False) -> dict[str, object]:
    modes = {
        name: run_mode(name, factory, prime_prompt, measured_prompt, force_mismatch)
        for name in ("disabled", "enabled_cold", "enabled_warm")
    }
    if len({result.engine_instance_id for result in modes.values()}) != 3:
        raise RuntimeError("benchmark modes must use fresh engine instances")
    token_equal = len({result.token_digest for result in modes.values()}) == 1
    logit_equal = len({result.logit_digest for result in modes.values()}) == 1
    if not token_equal or not logit_equal:
        raise BenchmarkMismatch("disabled/cold/warm digest mismatch")
    if any(result.open_leases != 0 for result in modes.values()):
        raise RuntimeError("benchmark completed with open prefix leases")
    return {
        "modes": {name: dataclasses.asdict(result) for name, result in modes.items()},
        "correctness": {"token_digests_equal": token_equal, "logit_digests_equal": logit_equal},
    }

class BenchmarkMismatch(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)

class PrefixCapabilityError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--offload-dir", default="/tmp/moe-prefix-benchmark")
    parser.add_argument("--shared-prefix-tokens", type=int, default=64)
    parser.add_argument("--suffix-tokens", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-force-mismatch", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()

def make_dry_factory():
    return lambda enabled: DryBenchmarkEngine(enabled)

def make_real_factory(args: argparse.Namespace, tokenizer):
    def factory(enabled: bool) -> RealBenchmarkEngine:
        owner = MoE(args.model, {
            "offload_path": str(Path(args.offload_dir) / str(uuid.uuid4())),
            "device_memory_ratio": 0.5,
        })
        config = _build_engine_config(SimpleNamespace(
            device_memory_ratio=0.5, kv_cache_ratio=0.25, max_batch_size=1,
            enable_prefix_caching=enabled, prefix_cache_max_entries=1000,
        ), owner.model)
        engine = ContinuousBatchingEngine(owner.model, owner.engine, config, tokenizer=tokenizer)
        if enabled and not engine.get_stats()["prefix_cache_active"]:
            raise PrefixCapabilityError(str(engine.get_stats()["prefix_cache_disabled_reason"]))
        return RealBenchmarkEngine(engine)
    return factory

def exact_tokens(tokenizer, text: str, count: int) -> list[int]:
    if count <= 0:
        return []
    seed = tokenizer.encode(text, add_special_tokens=False)
    if not seed:
        raise ValueError("tokenizer returned an empty benchmark seed")
    return (seed * ((count + len(seed) - 1) // len(seed)))[:count]

def build_prompt_pair(
    args: argparse.Namespace, tokenizer=None
) -> tuple[list[int], list[int]]:
    if tokenizer is None:
        shared = [100 + (index % 100) for index in range(args.shared_prefix_tokens)]
        prime_suffix = [200] * args.suffix_tokens
        measured_suffix = [201] * args.suffix_tokens
    else:
        shared = exact_tokens(tokenizer, "Exact shared prefix. ", args.shared_prefix_tokens)
        prime_suffix = exact_tokens(tokenizer, "Prime suffix. ", args.suffix_tokens)
        measured_suffix = exact_tokens(tokenizer, "Measured suffix. ", args.suffix_tokens)
    return shared + prime_suffix, shared + measured_suffix

def main() -> int:
    args = parse_args()
    tokenizer = None if args.dry_run else AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    factory = (
        make_dry_factory()
        if args.dry_run
        else make_real_factory(args, tokenizer)
    )
    prime_prompt, measured_prompt = build_prompt_pair(args, tokenizer)
    try:
        payload = run_suite(
            factory, prime_prompt, measured_prompt,
            force_mismatch=args.dry_run_force_mismatch,
        )
    except (BenchmarkMismatch, PrefixCapabilityError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
```

`--dry-run` uses three distinct deterministic `DryBenchmarkEngine` instances; cold reports a full query, warm reports a one-block-or-shorter suffix with the same `kv_seq_lengths`, and its allocator simulator records cache+sequence refcount high-water `2`. Real mode tokenizes exact-length, vocabulary-valid shared and divergent suffixes with the served tokenizer, uses fresh offload directories/engines, captures canonical `PagedBatchLengths`, sampled logits, TTFT/E2E, and the maximum `BlockAllocator.ref_count` observed after each step. `main()` catches `BenchmarkMismatch`, prints its message to stderr, exits `2`, writes no JSON on mismatch, and has the executable `if __name__ == '__main__': raise SystemExit(main())` guard. Any model override must yield an active validated capability in enabled modes or exit `2` before measurements.

Benchmark `Qwen/Qwen3-30B-A3B` by default. Report percentiles and measured ratios only for that workload; never state a universal speedup.

Document default-off behavior, Qwen3/FlashInfer rollout scope, exact namespace/path matching, layer completeness, pin-before-evict atomic groups, chunk-compatible query metadata, committed-range publication, COW, reload invalidation, DFlash exclusion, metrics, and rollback. Cite SGLang RadixAttention (<https://lmsys.org/blog/2024-01-17-sglang/>) and vLLM APC (<https://docs.vllm.ai/en/stable/examples/features/automatic_prefix_caching>) as motivation, not performance guarantees. Rollback is removing `--enable-prefix-caching`, restarting, and confirming `moe_prefix_cache_active 0`; there is no persisted state migration.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest -q tests/python/serving/test_prefix_cache_benchmark.py tests/python/serving/test_api_routes.py tests/python/serving/test_hot_reload.py`

Expected: PASS; dry-run schema distinguishes query and total KV metadata.

```bash
git add benchmarks/serving/prefix_cache_benchmark.py tests/python/serving/test_prefix_cache_benchmark.py docs/serving.md docs/benchmarking.md README.md ARCHITECTURE.md CHANGELOG.md
git commit -m "docs(serving): benchmark and document prefix KV reuse"
```

### Task 10: Full touched-suite and rollout verification

**Files:**
- Verify every source, test, benchmark, and doc path listed above.

Acceptance requires all of the following: (a) active engines satisfy `engine.kv_cache.block_store is capability.backend.block_store is capability.block_store`, `capability.block_store.owner is capability.backend`, and independent cache tensors/wrappers are disabled before scheduler construction; mismatch yields a `None` provider and cold execution; (b) no test helper referenced in this plan is implicit or undefined; (c) group prepare failure and explicit abort restore initial tables, free count, refcounts, statuses, and open-lease count, while commit transfers every prepared lease under the same owner without duplicate retain; (d) cold/warm FlashInfer plans use canonical `PagedBatchLengths(query_lengths, query_offsets, context_lengths, kv_seq_lengths)` and layer-complete store APIs exclusively; (e) physical store capacity may exceed logical allocator capacity but never vice versa; (f) dry and real benchmark modes use distinct engines, report geometry/refcount/digests, execute through `raise SystemExit(main())`, and abort before JSON write on mismatch; (g) normal and chunked prefill import the same optional `PrefixLeaseProvider` only from `serving/prefix_contract.py`; and (h) all touched suites below pass.

- [ ] **Step 1: Run every touched CPU suite**

Run: `python -m pytest -q tests/python/serving tests/python/contextpilot tests/python/unit/test_flashinfer_attention_backend.py tests/python/unit/test_v2_lifespan.py tests/python/unit/test_kv_swap_recovery.py tests/python/unit/test_kv_cache_free.py tests/python/unit/test_kv_edge_cases.py tests/python/dflash/test_kv_truncate.py tests/python/dflash/test_spec_verify.py`

Expected: PASS with only documented optional CUDA/dependency skips.

- [ ] **Step 2: Run static diagnostics for every touched Python file**

Run: `python -m ruff check moe_infinity/serving/prefix_contract.py moe_infinity/serving/prefix_cache.py moe_infinity/serving/kv_cache.py moe_infinity/runtime/attention_types.py moe_infinity/runtime/attention_backend.py moe_infinity/models/qwen3_paged_attention.py moe_infinity/serving/model_runner.py moe_infinity/serving/sequence.py moe_infinity/serving/scheduler.py moe_infinity/serving/batch.py moe_infinity/serving/engine.py moe_infinity/entrypoints/openai/api_server_v2.py benchmarks/serving/prefix_cache_benchmark.py tests/python/serving tests/python/unit/test_flashinfer_attention_backend.py tests/python/unit/test_v2_lifespan.py tests/python/unit/test_kv_swap_recovery.py`

Expected: PASS with no diagnostics.

- [ ] **Step 3: Run real Qwen3/FlashInfer parity**

Run: `MOE_PREFIX_CACHE_CUDA=1 python -m pytest -q tests/python/serving/test_prefix_cache_cuda.py`

Expected: PASS. Skip, incomplete layer registry, fake FlashInfer, or unsupported checkpoint blocks rollout.

- [ ] **Step 4: Run real disabled/cold/warm benchmark**

Run: `python benchmarks/serving/prefix_cache_benchmark.py --model Qwen/Qwen3-30B-A3B --offload-dir /tmp/moe-prefix-benchmark --shared-prefix-tokens 1024 --suffix-tokens 64 --output-json prefix-cache-results.json`

Expected: exit 0; equal output/logit digests; cold `query_offsets` reflects the full query; warm `query_offsets` reflects the suffix query while `kv_seq_lengths` remains full; nonzero matched tokens; zero open leases after completion. No minimum speedup is required.

- [ ] **Step 5: Exercise rollback**

Start without `--enable-prefix-caching`, verify ordinary completions and `moe_prefix_cache_active 0`; start with the flag and verify `active 1` on Qwen3/FlashInfer; remove the flag, restart, and verify `active 0` again.

- [ ] **Step 6: Commit only tracked verification fixes**

```bash
git add moe_infinity tests benchmarks docs README.md ARCHITECTURE.md CHANGELOG.md
git commit -m "test(serving): complete prefix reuse rollout checks"
```

Do not create an empty commit.
