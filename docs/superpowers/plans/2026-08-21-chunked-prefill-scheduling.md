# Bounded Chunked Prefill Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, bounded prompt chunking to MoE-Infinity continuous serving so active decode rows run every feasible scheduler step while long prefills make fair progress within the existing per-step token and batch limits.

**Architecture:** Keep the whole-prefill default path separate. The opt-in path records committed prompt progress, reserves logical blocks against the production layered store, emits exact chunk/query-vs-KV metadata, prioritizes decode, and samples only decode or terminal-prefill rows. Scheduler-owned KV transactions checkpoint every touched block and publish prompt progress only after execution and terminal sampling succeed. The first release rejects simultaneous prefix caching at validated startup, so this plan is executable from base without importing or recreating any prefix implementation.

**Tech Stack:** Python 3.10+, dataclasses and deque-based scheduling, PyTorch, MoE-Infinity paged KV and attention metadata, pytest, CUDA + FlashInfer integration tests, OpenAI-compatible streaming benchmark client.

---

## Scope, invariants, and file map

This is one serving-scheduler change, not P/D worker disaggregation. It does not change model partitioning, expert offload placement, speculative verification budgets, or the native non-serving generation loop.

### Scheduling contract

1. `enable_chunked_prefill=False` selects the existing whole-prefill algorithm without reordering, partial state, or changed admission behavior.
2. With chunking active, `max_tokens_per_step` counts every scheduled prompt token plus one token for every decode row. `max_batch_size` counts scheduled sequence rows.
3. Decode SLO policy is step-based: every runnable decode is selected before any prefill, in running FCFS order, unless the configured token/row cap is already exhausted. Prefill never preempts or displaces a selected decode.
4. A prefill chunk has `0 < num_tokens <= prefill_chunk_size`, starts exactly at `SequenceData.num_computed_tokens`, and ends no later than `prompt_length`. A terminal chunk ends exactly at `prompt_length`.
5. Scheduler output is a transaction lease. A scheduled chunk is removed from the ready queue, recorded in `_inflight_prefill`, and retains its prior reserved length plus backend checkpoint. Prompt progress commits only after model execution succeeds and terminal sampling succeeds. Execution or sampling failure restores every active backend cache representation, truncates newly reserved blocks, clears the in-flight lease, and requeues the same sequence at the head with unchanged committed progress.
6. While `status is PREFILL`, `output_token_ids` is empty and `0 <= num_computed_tokens <= prompt_length`. The first sampled token is produced only by a terminal chunk; afterward `num_computed_tokens == prompt_length + len(output_token_ids)`.
7. Fairness is round-robin among partial prefills. `prefill_starvation_threshold_steps` promotes an aged prefill ahead of other prefills only; it never jumps ahead of decode. The guarantee is bounded progress when at least one row and one token of post-decode capacity are feasible. Saturated decode load is reported as prefill backpressure, not hidden by violating decode priority.
8. `PagedKVCache` owns allocation, references, block tables, and leases; the bound `PagedAttentionBackend` owns the actual K/V tensors used by model attention (standard and FlashInfer layouts). Swap/export/import/checkpoint/rollback always operate through that bound active storage. The allocator-only cache must never snapshot a disconnected `_kv_cache` while the model writes elsewhere.
9. Query shape and KV history use exactly one canonical `PagedBatchLengths(query_lengths, query_offsets, context_lengths, kv_seq_lengths)` value from `runtime/attention_types.py`. Query fields describe packed tokens executed now; `context_lengths` describes prior committed history; `kv_seq_lengths=context+query` describes total visible KV after the forward. `BatchMetadata` and `AttentionMetadata` each carry this value as `lengths`; they do not redeclare or alias its four fields. FlashInfer passes `qo_indptr=lengths.query_offsets` and derives page metadata only from `lengths.kv_seq_lengths`.
10. Mixed paged batches continue to split into homogeneous prefill/decode launches. Chunk slot mappings cover only query positions; committed history remains in KV page tables and `kv_seq_lengths` but never appears in query offsets or slot mappings.
11. The first release requires `not (enable_chunked_prefill and enable_prefix_caching)`. Engine construction and every CLI/programmatic startup path reject co-enablement with `ValueError("enable_chunked_prefill and enable_prefix_caching cannot both be true in the first release")` before scheduler or KV allocation begins.
12. Chunking owns only scheduler rows and KV reservations/checkpoints. It does not import, create, adapt, or duplicate prefix contracts, cache indexes, providers, refcounts, or test helpers. PR #181 is referenced only as future reconciliation design input, never as an implementation prerequisite.
13. DFlash delegation remains eligible only when the combined predicate is true: singleton terminal one-shot prefill, `start_pos == 0`, no previous chunk, and all existing sampling constraints. Partial requests use ordinary paged prefill/decode; no DFlash session is entered midway through a prompt.

### Files to create or modify

- Modify `moe_infinity/serving/sequence.py`: prompt-progress helpers and invariants; no new sequence status.
- Modify `moe_infinity/serving/kv_cache.py`: idempotent incremental token reservation and observable reserved-token count.
- Modify `moe_infinity/serving/batch.py`: immutable `PrefillChunk`, scheduler output mapping, and exact chunk slicing.
- Modify `moe_infinity/serving/scheduler.py`: preserve legacy schedule path; add decode-first chunk path, round-robin/age accounting, partial-KV admission, commit, cancellation, and preemption recovery.
- Modify `moe_infinity/serving/model_runner.py`: canonical metadata and complete real-Qwen3 layer registry used by the engine gate.
- Modify `moe_infinity/runtime/attention_types.py`: canonical shared `PagedBatchLengths(query_lengths, query_offsets, context_lengths, kv_seq_lengths)` value used unchanged by batch and runtime metadata.
- Modify `moe_infinity/runtime/attention_backend.py`: query-correct FlashInfer planning plus canonical `LayeredPagedKVStore` transaction/swap protocol.
- Modify `moe_infinity/models/qwen3_paged_attention.py`: pass concrete layer identity into the canonical layered store backend.
- Modify `moe_infinity/serving/engine.py`: configuration wiring, chunk commit, terminal-only sampling, progress-only steps, DFlash guard, stats, and runtime config validation.
- Modify `moe_infinity/entrypoints/openai/api_server_v2.py`: CLI/programmatic feature controls and engine config propagation.
- Modify `moe_infinity/entrypoints/big_modeling.py`: `MoE.serve()` feature controls and documentation.
- Modify `tests/python/serving/test_sequence.py`: prompt-progress unit tests.
- Modify `tests/python/serving/test_batch.py`: chunk slicing/validation tests.
- Modify `tests/python/serving/test_scheduler.py`: disabled parity, budget, decode priority, fairness, and partial-prefill tests.
- Modify `tests/python/unit/test_kv_edge_cases.py`: incremental reservation tests.
- Modify `tests/python/unit/test_kv_swap_recovery.py`: partial-prefill recovery tests.
- Modify `tests/python/unit/test_flashinfer_attention_backend.py`: real backend metadata, transaction, and swap storage tests.
- Modify `tests/python/serving/test_cancellation.py`: cancellation and no-leak tests during partial prefill.
- Modify `tests/python/serving/test_engine.py`: progress-only steps, terminal sampling, eager fallback, and DFlash eligibility tests.
- Modify `tests/python/serving/test_flashinfer_model_runner.py`: query/KV metadata and real `Qwen3PagedAttention` detection.
- Modify `tests/python/serving/test_api_routes.py`: startup/config propagation and co-enablement rejection tests.
- Create `tests/python/serving/test_qwen3_paged_attention_cuda.py`: real Qwen3 invocation through real CUDA FlashInfer and the production paged backend.
- Modify `tests/python/integration/test_flashinfer_e2e.py`: CUDA paged mixed chunk/decode metadata and output parity.
- Create `benchmarks/serving/chunked_prefill_latency.py`: paired disabled/enabled streaming TTFT/TPOT-tail workload.
- Create `tests/python/serving/test_chunked_prefill_benchmark.py`: CPU tests for benchmark aggregation and paired result schema.
- Modify `docs/serving.md`: semantics, limitations, controls, metrics, and rollout.
- Modify `docs/benchmarking.md`: reproducible paired benchmark commands and interpretation.

### Mandatory implementation order

Execute Tasks 1, 2, 2A, 2B, and 3 in order, then continue with Tasks 4–10. Task 2B is an isolated configuration guard and has no prefix-file prerequisite; Tasks 2A–3 establish the layered-store and metadata contracts used by standalone chunk scheduling. This order is executable directly from base `b766f8f`.

### Source-review blocker closure

| Blocker | Plan closure |
|---|---|
| Query work conflated with total KV length | Tasks 3 and 7 use canonical `PagedBatchLengths(query_lengths, query_offsets, context_lengths, kv_seq_lengths)`, assert FlashInfer `qo_indptr=[0,2,5]` for contexts `[4,6]` and total KV `[6,9]`, and exercise the real backend object. |
| Chunks committed before terminal sampling | Tasks 4 and 5 add scheduler-owned in-flight leases and one rollback path for execution, sampling, cancellation, COW, reservation, and backend writes. |
| Swap snapshots the wrong tensor owner | Task 2A binds allocator metadata to layer-aware `PagedAttentionBackend` storage and tests export/import plus uninterrupted-vs-preempted logits/output parity in Task 7. |
| Capability detection omits Qwen3 | Tasks 5 and 7 add real `Qwen3PagedAttention` detection and a separately collected real-class forward test. |
| Invalid commands and incomplete benchmarks | Tasks 8–10 use existing test modules, exact token-ID prompts from the served tokenizer, measured output-token throughput, and polled peak KV block/utilization fields. |
| Absent prefix files block execution | Task 2B adds only a validated co-enablement rejection. No prefix production file, test helper, import, or copied implementation is required from base. |
| Prepare order and multi-row atomicity | Task 4 creates/reserves each table first, records the row immediately, then COW/checkpoints; a transaction-level exception handler restores all earlier rows. |
| Production capacity mismatch | Task 2A uses canonical `LayeredPagedKVStore.physical_capacity`, chooses `min(memory_budget_blocks, physical_capacity)`, and passes that clamped logical capacity to `set_block_store`; tests prove unequal logical/physical capacities activate safely. |
| Canonical shared interfaces | Tasks 2A–3 use only `LayeredPagedKVStore` and `PagedBatchLengths(query_lengths, query_offsets, context_lengths, kv_seq_lengths)`; chunking adds no cross-plan interface. |
| Partial scheduler commit | Tasks 4 and 5 inject every fallible scheduler-preflight position and require progress publication only after complete execution and sampling. |
| Real Qwen path still mocked | Task 7 invokes actual `Qwen3PagedAttention` through actual CUDA FlashInfer wrappers and production `PagedAttentionBackend`; recording/fake backends cannot satisfy rollout. |
| Prefix/chunk coexistence undefined | Task 2B rejects both flags together at config/startup with one exact error and documents rollback plus future reconciliation against PR #181 as design only. |
| Cross-plan test/acceptance drift | Tasks 7–10 use canonical file/API names and require prepare rollback, unequal-capacity activation, real-Qwen PASS, co-enablement rejection, throughput, and peak-KV evidence. |

## Task 1: Make prompt progress explicit on sequences

**Files:**
- Modify: `tests/python/serving/test_sequence.py`
- Modify: `moe_infinity/serving/sequence.py:34-115`

- [ ] **Step 1: Write failing prompt-progress tests**

Append these tests to `tests/python/serving/test_sequence.py`:

```python
def test_prefill_progress_advances_without_creating_output() -> None:
    sequence = SequenceData(
        seq_id=200,
        prompt_token_ids=[10, 11, 12, 13, 14],
        sampling_params=SamplingParams(),
    )
    sequence.set_status(SequenceStatus.PREFILL)

    sequence.advance_prefill(2)
    assert sequence.num_computed_tokens == 2
    assert sequence.committed_kv_tokens == 2
    assert sequence.remaining_prefill_tokens == 3
    assert sequence.prefill_complete is False
    assert sequence.output_token_ids == []

    sequence.advance_prefill(3)
    assert sequence.num_computed_tokens == 5
    assert sequence.remaining_prefill_tokens == 0
    assert sequence.prefill_complete is True


def test_prefill_progress_rejects_overrun_and_non_prefill_state() -> None:
    sequence = SequenceData(
        seq_id=201,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )
    with pytest.raises(RuntimeError, match="requires prefill status"):
        sequence.advance_prefill(1)

    sequence.set_status(SequenceStatus.PREFILL)
    with pytest.raises(ValueError, match="exceeds prompt length"):
        sequence.advance_prefill(4)
    with pytest.raises(ValueError, match="must be > 0"):
        sequence.advance_prefill(0)
```

- [ ] **Step 2: Run the tests and verify the missing interface fails**

Run: `pytest -q tests/python/serving/test_sequence.py -k prefill_progress`

Expected: FAIL with `AttributeError: 'SequenceData' object has no attribute 'advance_prefill'`.

- [ ] **Step 3: Implement the sequence invariants**

Add `committed_kv_tokens: int = 0` to `SequenceData`. Add these members immediately before `_validate_transition`:

```python
    @property
    def remaining_prefill_tokens(self) -> int:
        if self.output_token_ids:
            return 0
        return max(0, self.prompt_length - self.num_computed_tokens)

    @property
    def prefill_complete(self) -> bool:
        return self.num_computed_tokens >= self.prompt_length

    def advance_prefill(self, num_tokens: int) -> None:
        if self.status is not SequenceStatus.PREFILL:
            raise RuntimeError("advance_prefill requires prefill status")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
        new_total = self.num_computed_tokens + num_tokens
        if new_total > self.prompt_length:
            raise ValueError(
                f"prefill progress {new_total} exceeds prompt length "
                f"{self.prompt_length}"
            )
        if self.output_token_ids:
            raise RuntimeError("prefill progress cannot advance after decode")
        self.num_computed_tokens = new_total
        self.committed_kv_tokens = new_total
```

Update `append_output_token` to set both `num_computed_tokens` and `committed_kv_tokens` to prompt plus output length. Scheduling and reservation never mutate either counter; only successful transaction publication does.

- [ ] **Step 4: Run the focused and existing sequence tests**

Run: `pytest -q tests/python/serving/test_sequence.py`

Expected: PASS, including existing `WAITING -> PREFILL -> DECODE`, DFlash, swap, and cancellation transitions.

- [ ] **Step 5: Commit the sequence-state unit**

```bash
git add moe_infinity/serving/sequence.py tests/python/serving/test_sequence.py
git commit -m "feat(serving): track partial prefill progress"
```

## Task 2: Reserve paged KV capacity incrementally

**Files:**
- Modify: `tests/python/unit/test_kv_edge_cases.py`
- Modify: `moe_infinity/serving/kv_cache.py:101-257`

- [ ] **Step 1: Write failing incremental-reservation tests**

Append to `tests/python/unit/test_kv_edge_cases.py`:

```python
def test_ensure_sequence_capacity_grows_only_at_page_boundaries() -> None:
    cache = _make_kv_cache(num_blocks=4)

    cache.ensure_sequence_capacity(seq_id=9, total_tokens=3)
    assert cache.get_num_reserved_tokens(9) == 3
    assert cache.get_block_table(9) == [0]

    cache.ensure_sequence_capacity(seq_id=9, total_tokens=4)
    assert cache.get_block_table(9) == [0]
    cache.ensure_sequence_capacity(seq_id=9, total_tokens=5)
    assert cache.get_num_reserved_tokens(9) == 5
    assert cache.get_block_table(9) == [0, 1]


def test_ensure_sequence_capacity_is_idempotent_and_never_shrinks() -> None:
    cache = _make_kv_cache(num_blocks=4)
    cache.ensure_sequence_capacity(seq_id=10, total_tokens=5)
    free_after_first_reservation = cache.block_allocator.num_free_blocks

    cache.ensure_sequence_capacity(seq_id=10, total_tokens=5)
    assert cache.block_allocator.num_free_blocks == free_after_first_reservation
    with pytest.raises(ValueError, match="cannot shrink"):
        cache.ensure_sequence_capacity(seq_id=10, total_tokens=4)


def test_reservation_failure_leaves_no_partial_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _make_kv_cache(num_blocks=4)
    monkeypatch.setattr(
        cache.block_allocator, "allocate",
        lambda count: (_ for _ in ()).throw(RuntimeError("allocation failed")),
    )
    with pytest.raises(RuntimeError, match="allocation failed"):
        cache.ensure_sequence_capacity(seq_id=11, total_tokens=5)
    assert cache.has_sequence(11) is False
    assert cache.block_allocator.num_free_blocks == 4
```

Use the file's existing cache helper; if it is named differently, call that exact existing helper rather than adding a duplicate.

- [ ] **Step 2: Verify the new KV interface is absent**

Run: `pytest -q tests/python/unit/test_kv_edge_cases.py -k ensure_sequence_capacity`

Expected: FAIL with `AttributeError` for `ensure_sequence_capacity`.

- [ ] **Step 3: Implement idempotent reservation**

Add this method to `BlockTable` after `num_computed_tokens`:

```python
    def ensure_num_tokens(self, total_tokens: int) -> None:
        if total_tokens < self._num_tokens:
            raise ValueError(
                f"cannot shrink reservation from {self._num_tokens} "
                f"to {total_tokens}"
            )
        current_blocks = len(self._block_ids)
        required_blocks = (total_tokens + self.block_size - 1) // self.block_size
        new_ids = self.block_allocator.allocate(required_blocks - current_blocks)
        self._block_ids.extend(new_ids)
        self._num_tokens = total_tokens
```

Add these methods to `PagedKVCache` after `allocate_sequence` and make `allocate_sequence` call `block_table.ensure_num_tokens(num_tokens)` instead of its token loop:

```python
    def ensure_sequence_capacity(self, seq_id: int, total_tokens: int) -> None:
        if total_tokens < 0:
            raise ValueError(f"total_tokens must be >= 0, got {total_tokens}")
        block_table = self._sequence_tables.get(seq_id)
        if block_table is None:
            block_table = BlockTable(block_allocator=self.block_allocator)
            block_table.ensure_num_tokens(total_tokens)
            self._sequence_tables[seq_id] = block_table
            return
        block_table.ensure_num_tokens(total_tokens)

    def get_num_reserved_tokens(self, seq_id: int) -> int:
        return self._require_sequence(seq_id).num_computed_tokens()

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self._sequence_tables

    def get_block_ids_for_range(
        self, seq_id: int, start: int, end: int
    ) -> list[int]:
        if not 0 <= start <= end <= self.get_num_reserved_tokens(seq_id):
            raise ValueError("range is outside reserved sequence capacity")
        table = self.get_block_table(seq_id)
        first = start // self.block_size
        last = (end + self.block_size - 1) // self.block_size
        return list(dict.fromkeys(table[first:last]))

    def rollback_sequence_reservation(
        self, *, seq_id: int, prior_table_existed: bool,
        prior_block_ids: tuple[int, ...], prior_reserved_tokens: int,
        cow_block_ids: tuple[int, ...],
    ) -> None:
        table = self._require_sequence(seq_id)
        current_ids = tuple(table.get_block_ids())
        private_new_ids = tuple(
            block_id for block_id in current_ids
            if block_id not in set(prior_block_ids)
        )
        restored_shared_ids = tuple(
            block_id for block_id in prior_block_ids
            if block_id not in set(current_ids)
        )
        if cow_block_ids:
            self.block_allocator.retain(list(restored_shared_ids))
        table.restore_blocks(list(prior_block_ids), prior_reserved_tokens)
        if private_new_ids:
            if cow_block_ids:
                self.block_allocator.release(list(private_new_ids))
            else:
                self.block_allocator.free(list(private_new_ids))
        if not prior_table_existed:
            self._sequence_tables.pop(seq_id, None)
```

Keep `append_tokens` as the decode-facing compatibility method; implement it as `ensure_sequence_capacity(seq_id, get_num_reserved_tokens(seq_id) + num_new_tokens)`. Use the allocator's existing ownership-aware release operation for COW blocks; `cow_block_ids` must be a subset of `private_new_ids` or rollback raises before mutation.

- [ ] **Step 4: Run KV unit tests**

Run: `pytest -q tests/python/unit/test_kv_edge_cases.py tests/python/unit/test_kv_swap_recovery.py`

Expected: PASS; allocation, truncation, free, and swap behavior remain green.

- [ ] **Step 5: Commit incremental KV accounting**

```bash
git add moe_infinity/serving/kv_cache.py tests/python/unit/test_kv_edge_cases.py
git commit -m "feat(serving): reserve paged KV incrementally"
```

## Task 2A: Bind allocation metadata to the runtime backend that owns K/V tensors

**Files:**
- Modify: `tests/python/unit/test_flashinfer_attention_backend.py`
- Modify: `tests/python/unit/test_kv_swap_recovery.py`
- Modify: `moe_infinity/runtime/attention_backend.py:78-518`
- Modify: `moe_infinity/serving/kv_cache.py:13-379`
- Modify: `moe_infinity/serving/engine.py:122-156`
- Modify: `moe_infinity/models/qwen3_paged_attention.py:24-155`
- Modify: `moe_infinity/entrypoints/big_modeling.py:401-425`

- [ ] **Step 1: Write failing active-storage transaction and swap tests**

Append to `tests/python/unit/test_flashinfer_attention_backend.py`:

```python
def test_layered_store_checkpoint_restores_both_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=4, device=torch.device("cpu")
    )
    checkpoint = backend.block_store.checkpoint([1])
    key = torch.full((2, 2, 8), 7.0)
    value = torch.full((2, 2, 8), 9.0)
    slots = torch.tensor([4, 5])
    backend.write_kv(key, value, slots)
    backend.write_kv_flashinfer(key, value, slots)

    backend.block_store.restore([1], checkpoint)

    payload = backend.block_store.export_blocks([1])
    assert torch.count_nonzero(payload.k_cache) == 0
    assert torch.count_nonzero(payload.v_cache) == 0
    assert torch.count_nonzero(payload.fi_kv_cache) == 0


def test_swap_exports_and_restores_runtime_backend_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=4, device=torch.device("cpu")
    )
    cache = _make_serving_cache(num_blocks=4, block_size=4)
    cache.set_block_store(backend.block_store, logical_capacity=cache.num_blocks)
    cache.allocate_sequence(3, num_tokens=4)
    key = torch.arange(64, dtype=torch.float32).reshape(4, 2, 8)
    value = key + 100.0
    backend.write_kv(key, value, torch.arange(4))
    backend.write_kv_flashinfer(key, value, torch.arange(4))

    cache.swap_out(3)
    cache.free_gpu_blocks(3)
    cache.swap_in(3)

    restored = backend.block_store.export_blocks(
        cache.get_block_table(3)
    )
    torch.testing.assert_close(
        restored.fi_kv_cache[0, :, 0], key.reshape(1, 4, 2, 8)
    )
    torch.testing.assert_close(
        restored.fi_kv_cache[0, :, 1], value.reshape(1, 4, 2, 8)
    )
```

Add `_make_serving_cache` to this test file using the existing `PagedKVCache` constructor with one layer, two heads, head dimension eight, and CPU float32. The test intentionally checks the real `PagedAttentionBackend` storage rather than `PagedKVCache._kv_cache`.

- [ ] **Step 2: Verify ownership tests fail on the absent binding protocol**

Run: `python -m pytest -q tests/python/unit/test_flashinfer_attention_backend.py -k 'layered_store or swap_exports'`

Expected: FAIL with missing canonical `LayeredPagedKVStore` or `set_block_store`.

- [ ] **Step 3: Define one active paged-KV storage protocol**

Add these payload/checkpoint/store classes to `moe_infinity/runtime/attention_backend.py`:

```python
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
    num_layers: int
    physical_capacity: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    def export_blocks(
        self, block_ids: list[int]
    ) -> LayeredPagedKVPayload: ...

    def import_blocks(
        self, block_ids: list[int], payload: LayeredPagedKVPayload
    ) -> None: ...

    def checkpoint(
        self, block_ids: list[int]
    ) -> LayeredPagedKVCheckpoint: ...

    def restore(
        self,
        block_ids: list[int],
        checkpoint: LayeredPagedKVCheckpoint,
    ) -> None: ...
```

Use exactly the canonical shared implementation: `PagedAttentionBackend.block_store` is one `LayeredPagedKVStore` whose `physical_capacity` describes immutable physical tensor capacity and whose mutation API is exactly `export_blocks`, `import_blocks`, `checkpoint`, and `restore`. Add `layer_idx` to backend forward/write calls, register every Qwen layer exactly once, and pass `layer_idx=self.layer_idx` from `Qwen3PagedAttention`. The store owns standard and FlashInfer tensors across all layers. Chunking requires real FlashInfer because the current SDPA prefill fallback cannot consume historical pages. Do not add logical-capacity state or a second capacity alias to the store.

- [ ] **Step 4: Bind `PagedKVCache` and route swap through active storage**

Add to `PagedKVCache`:

```python
    _block_store: LayeredPagedKVStore | None = field(
        init=False, default=None
    )

    def set_block_store(
        self, store: LayeredPagedKVStore, *, logical_capacity: int
    ) -> None:
        if logical_capacity != self.num_blocks:
            raise ValueError("logical capacity must match allocator capacity")
        if logical_capacity <= 0 or logical_capacity > store.physical_capacity:
            raise ValueError("logical cache exceeds layered store capacity")
        if store.num_layers != self.num_layers:
            raise ValueError("paged KV layer-count mismatch")
        if store.block_size != self.block_size:
            raise ValueError("paged KV block-size mismatch")
        if store.num_kv_heads != self.num_heads:
            raise ValueError("paged KV head-count mismatch")
        if store.head_dim != self.head_dim:
            raise ValueError("paged KV head-dimension mismatch")
        if store.dtype != self.dtype or store.device != self.device:
            raise ValueError("paged KV dtype/device mismatch")
        self._block_store = store

    @property
    def block_store(self) -> LayeredPagedKVStore:
        if self._block_store is None:
            raise RuntimeError("layered paged KV store is not bound")
        return self._block_store
```

Replace `swap_out` reads with `store.checkpoint(block_ids)` and `swap_in` writes with `store.restore(replacement_ids, checkpoint)`. COW uses `payload=store.export_blocks([source])` then `store.import_blocks([destination], payload)`. Keep `_kv_cache` solely for the disabled legacy internal path; once a store is set, swap/COW/transaction code never reads it.

In `ContinuousBatchingEngine.__init__`, resolve the production backend/store before creating logical `PagedKVCache`, as specified in the geometry step below. If no validated `LayeredPagedKVStore` exists, leave chunking inactive with fallback reason `layered_paged_kv_store_unavailable`.

- [ ] **Step 5: Size logical allocation from production store capacity**

Add to `tests/python/serving/test_engine.py`:

```python
@pytest.mark.parametrize(
    ("budget_blocks", "store_blocks", "expected_blocks"),
    [(6, 8, 6), (12, 8, 8)],
)
def test_chunk_cache_uses_minimum_of_budget_and_bound_store(
    budget_blocks: int, store_blocks: int, expected_blocks: int
) -> None:
    backend = make_flashinfer_backend(num_blocks=store_blocks, num_layers=1)
    offload = MockOffloadEngine(attention_backend=backend)
    config = {
        **_make_config(),
        "num_kv_blocks": budget_blocks,
        "enable_chunked_prefill": True,
    }

    model = make_tiny_qwen3_paged_model(num_layers=1)
    engine = ContinuousBatchingEngine(model, offload, config)

    assert engine.kv_cache.num_blocks == expected_blocks
    assert engine.kv_cache.block_allocator.num_blocks == expected_blocks
    assert engine.model_runner.get_attention_backend() is backend
    assert engine.scheduler.chunked_prefill_enabled is True


def test_set_block_store_rejects_unclamped_logical_capacity() -> None:
    store = make_flashinfer_backend(num_blocks=8, num_layers=1).block_store
    cache = make_serving_cache(num_blocks=6, num_layers=1)
    with pytest.raises(ValueError, match="logical capacity must match"):
        cache.set_block_store(store, logical_capacity=8)
    cache.set_block_store(store, logical_capacity=6)
    assert cache.num_blocks == 6
    assert cache.block_store.physical_capacity == 8
```

`make_tiny_qwen3_paged_model` contains an actual `Qwen3PagedAttention(layer_idx=0)` and reports `config.num_hidden_layers=1`; `make_flashinfer_backend` installs the existing fake wrappers only for this CPU geometry unit. The separate CUDA test remains the proof of real FlashInfer execution.

Refactor backend discovery into `ModelRunner.resolve_attention_backend(engine)`. Before constructing `PagedKVCache`, `ContinuousBatchingEngine` resolves the backend and validated `backend.block_store`, computes the existing memory/config budget, then uses:

```python
logical_num_blocks = (
    memory_budget_blocks
    if backend is None
    else min(memory_budget_blocks, backend.num_gpu_blocks)
)
```

Validate once that `backend.num_gpu_blocks == backend.block_store.physical_capacity`, then construct the logical allocator with `logical_num_blocks` and call `set_block_store(block_store, logical_capacity=logical_num_blocks)`. Geometry validation compares layer count, block size, heads, head dimension, dtype, and device; it requires `0 < logical_num_blocks <= store.physical_capacity`, not logical/physical equality. `set_block_store` rechecks the caller-provided clamped logical capacity against both `PagedKVCache.num_blocks` and physical capacity before binding. The engine must never resize the store or create a second production backend merely to match an independently selected logical count. A zero budget or zero physical capacity remains a hard initialization error.

- [ ] **Step 6: Run storage, geometry, and swap tests**

Run: `python -m pytest -q tests/python/unit/test_flashinfer_attention_backend.py tests/python/unit/test_kv_swap_recovery.py tests/python/serving/test_kv_cache.py tests/python/serving/test_engine.py -k 'layered_store or swap_exports or minimum_of_budget or kv_cache'`

Expected: PASS; logical capacity is the min of budget/store, valid unequal capacities activate chunking, swap round-trips production storage, and unbound legacy tests remain unchanged.

- [ ] **Step 7: Commit active-storage ownership**

```bash
git add moe_infinity/runtime/attention_backend.py moe_infinity/serving/kv_cache.py moe_infinity/serving/model_runner.py moe_infinity/serving/engine.py moe_infinity/models/qwen3_paged_attention.py moe_infinity/entrypoints/big_modeling.py tests/python/unit/test_flashinfer_attention_backend.py tests/python/unit/test_kv_swap_recovery.py tests/python/serving/test_engine.py
git commit -m "fix(serving): unify paged KV storage ownership"
```

## Task 2B: Reject chunked-prefill and prefix-caching co-enablement

**Files:**
- Modify: `tests/python/serving/test_engine.py`
- Modify: `tests/python/serving/test_api_routes.py`
- Modify: `moe_infinity/serving/engine.py:84-156`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:475-522,1849-1919`
- Modify: `docs/serving.md`

- [ ] **Step 1: Write failing engine and startup validation tests**

Add `import pytest` if the module does not already import it, then append to `tests/python/serving/test_engine.py`:

```python
def test_engine_rejects_chunked_prefill_with_prefix_caching() -> None:
    config = _make_config()
    config.update(
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )

    with pytest.raises(
        ValueError,
        match=(
            "enable_chunked_prefill and enable_prefix_caching cannot both "
            "be true in the first release"
        ),
    ):
        ContinuousBatchingEngine(
            model=MockModel(), engine=MockOffloadEngine(), config=config
        )
```

Add `import pytest` if needed, then append to `tests/python/serving/test_api_routes.py`, using its existing configurable mock model helper:

```python
def test_build_engine_config_rejects_chunk_and_prefix_coenablement() -> None:
    args = SimpleNamespace(
        device_memory_ratio=0.75,
        kv_cache_ratio=0.25,
        max_batch_size=8,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        prefill_chunk_size=256,
        prefill_starvation_threshold_steps=4,
    )

    with pytest.raises(
        ValueError,
        match=(
            "enable_chunked_prefill and enable_prefix_caching cannot both "
            "be true in the first release"
        ),
    ):
        srv._build_engine_config(args=args, model=_ConfigurableMockModel())
```

- [ ] **Step 2: Run RED and verify co-enablement reaches startup**

Run: `python -m pytest -q tests/python/serving/test_engine.py tests/python/serving/test_api_routes.py -k 'rejects_chunked_prefill_with_prefix_caching or rejects_chunk_and_prefix_coenablement'`

Expected: FAIL because both flags are currently accepted; the expected error is `ValueError: enable_chunked_prefill and enable_prefix_caching cannot both be true in the first release`.

- [ ] **Step 3: Add one shared validation rule before resource construction**

Add this function in `moe_infinity/serving/engine.py` and import it in `api_server_v2.py`:

```python
CHUNKED_PREFIX_INCOMPATIBLE = (
    "enable_chunked_prefill and enable_prefix_caching cannot both be true "
    "in the first release"
)


def validate_chunked_prefill_config(config: dict[str, object]) -> None:
    if bool(config.get("enable_chunked_prefill", False)) and bool(
        config.get("enable_prefix_caching", False)
    ):
        raise ValueError(CHUNKED_PREFIX_INCOMPATIBLE)
```

Call `validate_chunked_prefill_config(config)` as the first statement in `ContinuousBatchingEngine.__init__`, before backend discovery, scheduler construction, or KV allocation. In `_build_engine_config`, build the dictionary into a local `config`, include both `"enable_chunked_prefill": bool(getattr(args, "enable_chunked_prefill", False))` and `"enable_prefix_caching": bool(getattr(args, "enable_prefix_caching", False))`, call the same validator, and then return `config`. These `getattr` defaults keep this task executable even when the independent prefix-caching CLI has not landed. Task 8 later adds the chunk CLI/programmatic arguments and numeric controls; every path then converges on this validator and engine construction. Do not add a second error string or silently disable either requested feature.

- [ ] **Step 4: Run GREEN and independent-mode regression tests**

Run: `python -m pytest -q tests/python/serving/test_engine.py tests/python/serving/test_api_routes.py -k 'chunked_prefill or prefix_caching or config'`

Expected: PASS. Chunking-only, prefix-caching-only, and both-disabled configurations remain accepted; only both-enabled startup raises the exact error above.

- [ ] **Step 5: Document rollback and future reconciliation**

Add a “Chunked prefill compatibility” subsection to `docs/serving.md` (Task 8 later incorporates it into the full experimental-feature section):

```markdown
The first release does not support simultaneous chunked prefill and prefix
caching. Startup rejects `enable_chunked_prefill=true` together with
`enable_prefix_caching=true` with `ValueError: enable_chunked_prefill and
enable_prefix_caching cannot both be true in the first release`. Roll back by
removing `--enable-chunked-prefill` (or disabling prefix caching) and restart;
no request or cache-format migration is required. PR #181 is design input for
a future transaction-contract reconciliation only. This release neither
depends on that PR nor copies any prefix implementation into chunk scheduling.
```

- [ ] **Step 6: Commit the compatibility guard**

```bash
git add moe_infinity/serving/engine.py moe_infinity/entrypoints/openai/api_server_v2.py tests/python/serving/test_engine.py tests/python/serving/test_api_routes.py docs/serving.md
git commit -m "fix(serving): reject chunk and prefix co-enablement"
```

## Task 3: Carry exact chunk descriptors through batch construction

**Files:**
- Modify: `tests/python/serving/test_batch.py`
- Modify: `moe_infinity/serving/batch.py:13-273`

- [ ] **Step 1: Write failing descriptor and slicing tests**

Add `import pytest`, import `PrefillChunk` from `_BATCH_MODULE`, then append:

```python
def test_batch_builder_slices_exact_prefill_chunk() -> None:
    cache = _make_cache()
    sequence = _make_sequence(
        40,
        [10, 11, 12, 13, 14, 15],
        status=SequenceStatus.PREFILL,
        num_computed_tokens=2,
    )
    cache.allocate_sequence(40, num_tokens=5)
    output = SchedulerOutput(
        prefill_seq_ids=[40],
        prefill_chunks={
            40: PrefillChunk(start_pos=2, num_tokens=3, is_terminal=False)
        },
        num_prefill_tokens=3,
    )

    metadata = BatchBuilder.from_scheduler_output(output, {40: sequence}, cache)

    assert metadata.input_token_ids == [12, 13, 14]
    assert metadata.lengths == PagedBatchLengths(
        query_lengths=[3],
        query_offsets=[0, 3],
        context_lengths=[2],
        kv_seq_lengths=[5],
    )
    assert metadata.prefill_is_terminal == [False]


def test_scheduler_output_rejects_mismatched_chunk_ids() -> None:
    with pytest.raises(ValueError, match="prefill_chunks keys"):
        SchedulerOutput(
            prefill_seq_ids=[1],
            prefill_chunks={
                2: PrefillChunk(start_pos=0, num_tokens=1, is_terminal=True)
            },
        )
```

- [ ] **Step 2: Verify descriptor tests fail**

Run: `pytest -q tests/python/serving/test_batch.py -k 'chunk or descriptor'`

Expected: FAIL because `PrefillChunk` and `prefill_is_terminal` do not exist.

- [ ] **Step 3: Add the exact canonical metadata interface**

Add before `SchedulerOutput`:

```python
@dataclass(frozen=True)
class PrefillChunk:
    start_pos: int
    num_tokens: int
    is_terminal: bool

    def __post_init__(self) -> None:
        if self.start_pos < 0:
            raise ValueError("start_pos must be >= 0")
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
```

Add `prefill_chunks: dict[int, PrefillChunk] = field(default_factory=dict)` and `prefill_transaction_id: int | None = None` to `SchedulerOutput`, copy the mapping in `__post_init__`, reject nonempty mappings whose keys differ from `prefill_seq_ids`, and require a transaction ID whenever the mapping is nonempty. In `moe_infinity/runtime/attention_types.py`, define the one shared value exactly as follows; both serving and runtime metadata import it rather than recreating its fields:

```python
@dataclass(frozen=True)
class PagedBatchLengths:
    query_lengths: list[int] | torch.Tensor
    query_offsets: list[int] | torch.Tensor
    context_lengths: list[int] | torch.Tensor
    kv_seq_lengths: list[int] | torch.Tensor

    def __post_init__(self) -> None:
        def values(value: list[int] | torch.Tensor) -> list[int]:
            if isinstance(value, torch.Tensor):
                if value.ndim != 1:
                    raise ValueError("paged batch lengths must be rank one")
                return [int(item) for item in value.detach().cpu().tolist()]
            return list(value)

        query_lengths = values(self.query_lengths)
        query_offsets = values(self.query_offsets)
        context_lengths = values(self.context_lengths)
        kv_seq_lengths = values(self.kv_seq_lengths)
        batch_size = len(query_lengths)
        if len(query_offsets) != batch_size + 1:
            raise ValueError("query_offsets must have batch_size + 1 entries")
        if len(context_lengths) != batch_size:
            raise ValueError("context_lengths must match query_lengths")
        if len(kv_seq_lengths) != batch_size:
            raise ValueError("kv_seq_lengths must match query_lengths")
        if query_offsets[:1] != [0]:
            raise ValueError("query_offsets must start at zero")
        running = 0
        for index, query_length in enumerate(query_lengths):
            if query_length <= 0 or context_lengths[index] < 0:
                raise ValueError("paged batch lengths must be non-negative")
            running += query_length
            if query_offsets[index + 1] != running:
                raise ValueError("query_offsets must sum query_lengths")
            if kv_seq_lengths[index] != context_lengths[index] + query_length:
                raise ValueError("kv_seq_lengths must equal context plus query")
```

Use this exact metadata shape:

```python
@dataclass
class BatchMetadata:
    seq_ids: list[int]
    input_token_ids: list[int]
    lengths: PagedBatchLengths
    is_prefill: list[bool]
    block_tables: list[list[int]]
    sampling_params: list[SamplingParams]
    prefill_is_terminal: list[bool] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return len(self.input_token_ids)
```

Require `len(seq_ids) == len(lengths.query_lengths)` and `lengths.query_offsets[-1] == len(input_token_ids)`. Do not retain direct `query_lengths`, `query_offsets`, `context_lengths`, or `kv_seq_lengths` fields on `BatchMetadata`/`AttentionMetadata`; do not retain `seq_lengths` or `token_offsets` aliases. Update direct constructors in `tests/python/serving/test_batch.py`, `tests/python/serving/test_model_runner.py`, `tests/python/serving/test_flashinfer_model_runner.py`, `tests/python/serving/test_flashinfer_mixed_batch.py`, and `tests/python/integration/test_flashinfer_e2e.py` in this same commit so there is exactly one shared lengths contract.

Replace the prefill slicing loop in `BatchBuilder.from_scheduler_output` with:

```python
        for seq_id in scheduler_output.prefill_seq_ids:
            sequence = sequences[seq_id]
            chunk = scheduler_output.prefill_chunks.get(seq_id)
            if chunk is None:
                start = sequence.num_computed_tokens
                end = sequence.prompt_length
                terminal = True
            else:
                start = chunk.start_pos
                end = start + chunk.num_tokens
                terminal = chunk.is_terminal
            tokens = sequence.prompt_token_ids[start:end]
            if len(tokens) != end - start:
                raise ValueError(f"invalid prefill chunk for seq_id={seq_id}")
            input_token_ids.extend(tokens)
            query_lengths.append(len(tokens))
            context_lengths.append(start)
            kv_seq_lengths.append(end)
            is_prefill.append(True)
            prefill_is_terminal.append(terminal)
            block_tables.append(kv_cache.get_block_table(seq_id))
            sampling_params.append(sequence.sampling_params)
```

After both loops, construct `lengths=PagedBatchLengths(query_lengths, query_offsets, context_lengths, kv_seq_lengths)`. For decode rows append query length `1`, query offset growth by `1`, `context_length=sequence.num_computed_tokens`, `kv_seq_length=context+1`, and `False` to `prefill_is_terminal`. Export `PrefillChunk` in `__all__`. `_slice_batch` builds one new `PagedBatchLengths` from selected rows and rebuilt query offsets; it copies selected total KV lengths unchanged. `SplitBatchMetadata.recombine_outputs` uses `batch.lengths.query_offsets`, never total KV lengths.

- [ ] **Step 4: Run batch and model-runner metadata tests**

Run: `pytest -q tests/python/serving/test_batch.py tests/python/serving/test_model_runner.py tests/python/serving/test_flashinfer_model_runner.py tests/python/serving/test_flashinfer_mixed_batch.py`

Expected: PASS; mixed split/recombine order and packed slot mappings remain unchanged.

- [ ] **Step 5: Commit the chunk data contract**

```bash
git add moe_infinity/runtime/attention_types.py moe_infinity/serving/batch.py tests/python/serving/test_batch.py tests/python/serving/test_model_runner.py tests/python/serving/test_flashinfer_model_runner.py tests/python/serving/test_flashinfer_mixed_batch.py tests/python/integration/test_flashinfer_e2e.py
git commit -m "feat(serving): carry exact prefill chunk metadata"
```

## Task 4: Add the opt-in decode-first bounded scheduler

**Files:**
- Modify: `tests/python/serving/test_scheduler.py`
- Modify: `moe_infinity/serving/scheduler.py:158-602`

- [ ] **Step 1: Write disabled-parity, budget, and decode-priority tests**

Add `import pytest`; import `PagedAttentionBackend`, `KVCacheSpec`, and `PrefillChunk`; then append:

```python
def _make_paged_backend(num_blocks: int) -> PagedAttentionBackend:
    return PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2, head_dim=8, dtype=torch.float16, block_size=4
        ),
        num_gpu_blocks=num_blocks,
        num_layers=1,
        device=torch.device("cpu"),
    )


def _make_chunk_cache(num_blocks: int = 8) -> PagedKVCache:
    cache = _make_cache(num_blocks=num_blocks)
    cache.set_block_store(
        _make_paged_backend(num_blocks=num_blocks).block_store,
        logical_capacity=cache.num_blocks,
    )
    return cache


def _make_chunk_scheduler(*, max_batch_size: int = 8) -> Scheduler:
    return Scheduler(
        _make_chunk_cache(), max_batch_size=max_batch_size,
        max_tokens_per_step=8, enable_chunked_prefill=True,
        prefill_chunk_size=4,
    )


def test_chunking_disabled_preserves_whole_prefill_blocking() -> None:
    scheduler = Scheduler(
        _make_cache(), max_batch_size=8, max_tokens_per_step=4,
        enable_chunked_prefill=False, prefill_chunk_size=2,
    )
    scheduler.add_request(_make_group("long", 1, 5))

    output = scheduler.schedule()

    assert output.prefill_seq_ids == []
    assert output.decode_seq_ids == []
    assert output.prefill_chunks == {}


def test_chunked_prefill_never_exceeds_step_budget() -> None:
    scheduler = Scheduler(
        _make_chunk_cache(), max_batch_size=8, max_tokens_per_step=4,
        enable_chunked_prefill=True, prefill_chunk_size=3,
    )
    scheduler.add_request(_make_group("long", 1, 8))

    first = scheduler.schedule()
    assert first.prefill_chunks == {
        1: PrefillChunk(start_pos=0, num_tokens=3, is_terminal=False)
    }
    assert first.num_prefill_tokens == 3
    scheduler.commit_prefill_step(first.prefill_transaction_id)

    second = scheduler.schedule()
    assert second.prefill_chunks[1].start_pos == 3
    assert second.num_prefill_tokens <= 4


def test_decode_rows_are_scheduled_before_prefill_budget() -> None:
    scheduler = Scheduler(
        _make_chunk_cache(), max_batch_size=4, max_tokens_per_step=3,
        enable_chunked_prefill=True, prefill_chunk_size=3,
    )
    decode_group = _make_group("decode", 1, 1)
    scheduler.add_request(decode_group)
    first = scheduler.schedule()
    scheduler.commit_prefill_step(first.prefill_transaction_id)
    scheduler.update_after_step([], [1])
    scheduler.add_request(_make_group("prefill", 2, 8))

    output = scheduler.schedule()

    assert output.decode_seq_ids == [1]
    assert output.prefill_chunks[2].num_tokens == 2
    assert output.num_decode_tokens + output.num_prefill_tokens == 3
```

- [ ] **Step 2: Write fairness and commit-validation tests**

Append:

```python
def test_partial_prefills_rotate_and_aged_prefill_stays_ahead_of_new_prefill() -> None:
    scheduler = Scheduler(
        _make_chunk_cache(num_blocks=16), max_batch_size=1, max_tokens_per_step=2,
        enable_chunked_prefill=True, prefill_chunk_size=2,
        prefill_starvation_threshold_steps=2,
    )
    scheduler.add_request(_make_group("a", 1, 6))
    scheduler.add_request(_make_group("b", 2, 6))
    seen: list[int] = []
    for _ in range(2):
        output = scheduler.schedule()
        seen.extend(output.prefill_seq_ids)
        scheduler.commit_prefill_step(output.prefill_transaction_id)
    scheduler.add_request(_make_group("new", 3, 2))
    third = scheduler.schedule()

    assert seen == [1, 2]
    assert third.prefill_seq_ids == [1]


def test_commit_rejects_already_completed_transaction() -> None:
    scheduler = Scheduler(
        _make_chunk_cache(), max_batch_size=2, max_tokens_per_step=2,
        enable_chunked_prefill=True, prefill_chunk_size=2,
    )
    scheduler.add_request(_make_group("req", 1, 4))
    output = scheduler.schedule()

    scheduler.commit_prefill_step(output.prefill_transaction_id)
    with pytest.raises(RuntimeError, match="unknown prefill transaction"):
        scheduler.commit_prefill_step(output.prefill_transaction_id)


@pytest.mark.parametrize("fail_row", [0, 1])
def test_row_preflight_failure_aborts_entire_group(
    fail_row: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _make_chunk_scheduler(max_batch_size=2)
    scheduler.add_request(_make_group("a", 1, 4))
    scheduler.add_request(_make_group("b", 2, 4))
    before_free = scheduler.kv_cache.block_allocator.num_free_blocks
    output = scheduler.schedule()
    original = _PrefillRowCommit.prepare_commit
    calls = 0

    def fail_selected_preflight(participant: _PrefillRowCommit) -> None:
        nonlocal calls
        current = calls
        calls += 1
        original(participant)
        if current == fail_row:
            raise RuntimeError("injected row preflight failure")

    monkeypatch.setattr(_PrefillRowCommit, "prepare_commit", fail_selected_preflight)
    with pytest.raises(RuntimeError, match="injected row preflight failure"):
        scheduler.commit_prefill_step(output.prefill_transaction_id)

    assert scheduler.kv_cache.block_allocator.num_free_blocks == before_free
    assert scheduler.inflight_prefill_seq_ids == []
    assert [scheduler._sequence_map[i].num_computed_tokens for i in (1, 2)] == [0, 0]
    assert scheduler.schedule().prefill_seq_ids == [1, 2]


def test_scheduled_chunk_stays_inflight_until_commit_or_rollback() -> None:
    scheduler = Scheduler(
        _make_chunk_cache(), max_batch_size=2, max_tokens_per_step=2,
        enable_chunked_prefill=True, prefill_chunk_size=2,
    )
    scheduler.add_request(_make_group("req", 1, 4))
    output = scheduler.schedule()

    assert scheduler.inflight_prefill_seq_ids == [1]
    assert scheduler.schedule().prefill_seq_ids == []
    scheduler.rollback_prefill_step(output.prefill_transaction_id)
    assert scheduler.inflight_prefill_seq_ids == []
    retried = scheduler.schedule()
    assert retried.prefill_chunks[1].start_pos == 0


def test_row_is_recorded_after_reservation_before_cow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _make_chunk_scheduler(max_batch_size=1)
    scheduler.add_request(_make_group("req", 1, 4))
    original = scheduler.kv_cache.ensure_writable_range

    def assert_recorded(seq_id: int, start: int, end: int):
        assert scheduler.kv_cache.has_sequence(seq_id)
        assert scheduler.kv_cache.get_num_reserved_tokens(seq_id) == end
        assert seq_id in scheduler._inflight_prefill
        return original(seq_id, start, end)

    monkeypatch.setattr(
        scheduler.kv_cache, "ensure_writable_range", assert_recorded
    )
    scheduler.schedule()


def test_later_row_prepare_failure_rolls_back_every_prepared_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _make_chunk_scheduler(max_batch_size=2)
    scheduler.add_request(_make_group("a", 1, 4))
    scheduler.add_request(_make_group("b", 2, 4))
    initial_free = scheduler.kv_cache.block_allocator.num_free_blocks
    original = scheduler.kv_cache.block_store.checkpoint
    calls = 0

    def fail_second(block_ids: list[int]):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second-row checkpoint failed")
        return original(block_ids)

    monkeypatch.setattr(scheduler.kv_cache.block_store, "checkpoint", fail_second)
    with pytest.raises(RuntimeError, match="second-row checkpoint failed"):
        scheduler.schedule()

    assert scheduler.inflight_prefill_seq_ids == []
    assert scheduler.kv_cache.block_allocator.num_free_blocks == initial_free
    assert not scheduler.kv_cache.has_sequence(1)
    assert not scheduler.kv_cache.has_sequence(2)
    assert scheduler._sequence_map[1].status is SequenceStatus.WAITING
    assert scheduler._sequence_map[2].status is SequenceStatus.WAITING
    assert list(scheduler._prefill_queue)[:2] == [1, 2]
```

- [ ] **Step 3: Verify scheduler tests fail for the new constructor and methods**

Run: `python -m pytest -q tests/python/serving/test_scheduler.py -k 'chunked or chunking or decode_rows or partial_prefills or completed_transaction or stays_inflight or recorded_after_reservation or later_row_prepare_failure'`

Expected: FAIL with unexpected constructor keyword `enable_chunked_prefill`.

- [ ] **Step 4: Add configuration and isolate the unchanged legacy path**

Extend `Scheduler.__init__` with:

```python
        enable_chunked_prefill: bool = False,
        prefill_chunk_size: int = 512,
        prefill_starvation_threshold_steps: int = 8,
```

Validate positive sizes, store `chunked_prefill_requested`, `chunked_prefill_enabled`, and initialize:

```python
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_starvation_threshold_steps = prefill_starvation_threshold_steps
        self._prefill_queue: deque[int] = deque()
        self._prefill_wait_steps: dict[int, int] = {}
        self._swapped_resume_status: dict[int, SequenceStatus] = {}
        self._inflight_prefill: dict[int, InFlightPrefill] = {}
        self._next_prefill_transaction_id = 0
        self._schedule_steps = 0
```

Move the current body of `schedule()` unchanged into `_schedule_whole_prefill()`. The public method must be exactly:

```python
    def schedule(self) -> SchedulerOutput:
        if not self.chunked_prefill_enabled:
            return self._schedule_whole_prefill()
        return self._schedule_chunked_prefill()

    def set_chunked_prefill_runtime_enabled(self, enabled: bool) -> None:
        self.chunked_prefill_enabled = self.chunked_prefill_requested and enabled
```

This separation is the disabled-mode compatibility boundary; do not share reordered admission loops with legacy mode.

- [ ] **Step 5: Implement decode-first scheduling and commit**

On `add_request`, append each sequence ID to `_prefill_queue` and initialize its wait counter. Implement `_schedule_chunked_prefill` with these concrete phases:

```python
    def _schedule_chunked_prefill(self) -> SchedulerOutput:
        output = SchedulerOutput()
        self._schedule_steps += 1
        self._recover_swapped_groups(list(self._swapped))

        scheduled_rows = 0
        scheduled_tokens = 0
        for group in self._running:
            for sequence in group.sequences:
                if sequence.status is not SequenceStatus.DECODE:
                    continue
                if scheduled_rows >= self.max_batch_size:
                    break
                if scheduled_tokens + 1 > self.max_tokens_per_step:
                    break
                output.decode_seq_ids.append(sequence.seq_id)
                output.num_decode_tokens += 1
                scheduled_rows += 1
                scheduled_tokens += 1

        available_rows = self.max_batch_size - scheduled_rows
        available_tokens = self.max_tokens_per_step - scheduled_tokens
        if available_rows <= 0 or available_tokens <= 0:
            self._age_unscheduled_prefills(set())
            self._apply_verify_scheduling(output)
            return output

        ordered = self._ordered_prefill_candidates()
        selected: set[int] = set()
        transaction_id = self._next_prefill_transaction_id
        self._next_prefill_transaction_id += 1
        try:
            for seq_id in ordered:
                if available_rows <= 0 or available_tokens <= 0:
                    break
                sequence = self._sequence_map.get(seq_id)
                if sequence is None or sequence.status not in {
                    SequenceStatus.WAITING, SequenceStatus.PREFILL
                }:
                    continue
                count = min(
                    self.prefill_chunk_size,
                    sequence.remaining_prefill_tokens,
                    available_tokens,
                )
                if count <= 0:
                    continue
                end = sequence.num_computed_tokens + count
                required_blocks = self._incremental_required_blocks(sequence, end)
                if self.kv_cache.block_allocator.num_free_blocks < required_blocks:
                    continue
                chunk = PrefillChunk(
                    start_pos=sequence.num_computed_tokens,
                    num_tokens=count,
                    is_terminal=end == sequence.prompt_length,
                )
                existed = self.kv_cache.has_sequence(seq_id)
                prior_reserved = (
                    self.kv_cache.get_num_reserved_tokens(seq_id) if existed else 0
                )
                prior_block_ids = (
                    tuple(self.kv_cache.get_block_table(seq_id)) if existed else ()
                )

                # Mandatory order: create/reserve first, then record immediately,
                # then COW and checkpoint against a valid physical block table.
                self.kv_cache.ensure_sequence_capacity(seq_id, end)
                lease = InFlightPrefill(
                    transaction_id=transaction_id,
                    seq_id=seq_id,
                    chunk=chunk,
                    prior_table_existed=existed,
                    prior_reserved_tokens=prior_reserved,
                    prior_block_ids=prior_block_ids,
                )
                self._inflight_prefill[seq_id] = lease
                cow = self.kv_cache.ensure_writable_range(
                    seq_id, chunk.start_pos, end
                )
                lease.cow_block_ids = cow.new_block_ids
                touched = self.kv_cache.get_block_ids_for_range(
                    seq_id, chunk.start_pos, end
                )
                lease.checkpoint_block_ids = touched
                lease.write_checkpoint = self.kv_cache.block_store.checkpoint(
                    touched
                )

                output.prefill_seq_ids.append(seq_id)
                output.prefill_chunks[seq_id] = chunk
                output.num_prefill_tokens += count
                selected.add(seq_id)
                available_rows -= 1
                available_tokens -= count
        except BaseException:
            self._rollback_prepared_transaction(transaction_id)
            raise

        if output.prefill_chunks:
            for seq_id in output.prefill_seq_ids:
                sequence = self._sequence_map[seq_id]
                if sequence.status is SequenceStatus.WAITING:
                    sequence.set_status(SequenceStatus.PREFILL)
                    self._move_request_to_running(seq_id)
            output.prefill_transaction_id = transaction_id
        self._remove_inflight_from_ready_queue(output.prefill_chunks)
        for seq_id in output.prefill_seq_ids:
            self._inflight_prefill[seq_id].was_removed_from_ready_queue = True
        self._age_unscheduled_prefills(selected)
        self._apply_verify_scheduling(output)
        return output

    def commit_prefill_step(self, transaction_id: int | None) -> None:
        rows = self._leases_for_transaction(transaction_id)
        row_commits = [_PrefillRowCommit(self, row) for row in rows]
        try:
            for participant in row_commits:
                participant.prepare_commit()
        except BaseException as original:
            rollback_errors: list[BaseException] = []
            for participant in reversed(row_commits):
                try:
                    participant.abort()
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            try:
                self._rollback_prepared_transaction(transaction_id)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
            if rollback_errors:
                details = "; ".join(repr(error) for error in rollback_errors)
                raise RuntimeError(
                    f"{original}; transaction rollback errors: {details}"
                ) from original
            raise
        # Everything below publishes scheduler-owned progress after preflight.
        for participant in row_commits:
            participant.commit()
        for row in rows:
            self._inflight_prefill.pop(row.seq_id)
            if not row.chunk.is_terminal:
                self._enqueue_prefill_once(row.seq_id)

    def rollback_prefill_step(self, transaction_id: int | None) -> None:
        leases = self._leases_for_transaction(transaction_id)
        for lease in reversed(leases):
            if lease.write_checkpoint is not None:
                self.kv_cache.block_store.restore(
                    lease.checkpoint_block_ids, lease.write_checkpoint
                )
            self.kv_cache.rollback_sequence_reservation(
                seq_id=lease.seq_id,
                prior_table_existed=lease.prior_table_existed,
                prior_block_ids=lease.prior_block_ids,
                prior_reserved_tokens=lease.prior_reserved_tokens,
                cow_block_ids=lease.cow_block_ids,
            )
            self._inflight_prefill.pop(lease.seq_id)
            if (
                lease.was_removed_from_ready_queue
                and self._sequence_map.get(lease.seq_id) is not None
            ):
                self._prefill_queue.appendleft(lease.seq_id)
```

Define the scheduler-owned lease exactly as:

```python
@dataclass
class InFlightPrefill:
    transaction_id: int
    seq_id: int
    chunk: PrefillChunk
    prior_table_existed: bool
    prior_reserved_tokens: int
    prior_block_ids: tuple[int, ...]
    checkpoint_block_ids: list[int] = field(default_factory=list)
    write_checkpoint: LayeredPagedKVCheckpoint | None = None
    cow_block_ids: tuple[int, ...] = ()
    was_removed_from_ready_queue: bool = False
    prior_num_computed_tokens: int = 0
    prior_committed_kv_tokens: int = 0
```

Define `_PrefillRowCommit.prepare_commit()` as fallible preflight only: validate `chunk.start_pos`, every table/checkpoint/publication precondition, and snapshot all sequence fields. After every row passes, `_PrefillRowCommit.commit()` publishes scheduler-owned progress. Before that boundary, `_PrefillRowCommit.abort()` restores scheduler state.

`_rollback_prepared_transaction` uses the same reverse-order physical restore loop as `rollback_prefill_step`. Thus a second/third row reservation, COW, checkpoint, prepare, or commit exception restores every earlier row before propagating. No in-flight ID remains in `_prefill_queue`. `abort_request` encountering an in-flight transaction marks it cancelled and invokes the same rollback before freeing its sequence reservation; it cannot free blocks while a model call owns the checkpoint.

Add the concrete helpers below. Existing admission scoring may reorder never-started groups before IDs enter the round-robin order, but must not reorder partial prefills.

```python
    def _ordered_prefill_candidates(self) -> list[int]:
        unique = list(dict.fromkeys(self._prefill_queue))
        live = [
            seq_id for seq_id in unique
            if (sequence := self._sequence_map.get(seq_id)) is not None
            and sequence.status in {SequenceStatus.WAITING, SequenceStatus.PREFILL}
            and sequence.remaining_prefill_tokens > 0
        ]
        aged = [
            seq_id for seq_id in live
            if self._prefill_wait_steps.get(seq_id, 0)
            >= self.prefill_starvation_threshold_steps
        ]
        aged_set = set(aged)
        return [*aged, *(seq_id for seq_id in live if seq_id not in aged_set)]

    def _remove_inflight_from_ready_queue(
        self, chunks: dict[int, PrefillChunk]
    ) -> None:
        selected = set(chunks)
        self._prefill_queue = deque(
            seq_id for seq_id in dict.fromkeys(self._prefill_queue)
            if seq_id not in selected and seq_id in self._sequence_map
        )

    def _age_unscheduled_prefills(self, selected: set[int]) -> None:
        for seq_id in self._ordered_prefill_candidates():
            if seq_id in selected:
                self._prefill_wait_steps[seq_id] = 0
            else:
                self._prefill_wait_steps[seq_id] = (
                    self._prefill_wait_steps.get(seq_id, 0) + 1
                )

    def _move_request_to_running(self, seq_id: int) -> None:
        request_id = next(
            request_id for request_id, group in self._request_map.items()
            if seq_id in group.sequence_ids
        )
        group = self._request_map[request_id]
        if group in self._waiting:
            self._waiting.remove(group)
        if group not in self._running:
            self._running.append(group)

    def _incremental_required_blocks(
        self, sequence: SequenceData, end: int
    ) -> int:
        current = (
            len(self.kv_cache.get_block_table(sequence.seq_id))
            if self.kv_cache.has_sequence(sequence.seq_id)
            else 0
        )
        return max(0, ceil(end / self.kv_cache.block_size) - current)

    def _enqueue_prefill_once(self, seq_id: int) -> None:
        if seq_id not in self._prefill_queue:
            self._prefill_queue.append(seq_id)
        self._prefill_wait_steps.setdefault(seq_id, 0)
```

The scheduler uses the public `PagedKVCache.has_sequence` interface from Task 2 and never reads `_sequence_tables` directly.

- [ ] **Step 6: Run scheduler tests, including legacy cases**

Run: `pytest -q tests/python/serving/test_scheduler.py tests/python/contextpilot/test_cp_scheduler_v2.py tests/python/serving/test_dflash_deficit_scheduler.py`

Expected: PASS. Existing FCFS whole-prefill and verify-deficit assertions remain byte-for-byte unchanged when chunking is disabled.

- [ ] **Step 7: Commit the bounded policy**

```bash
git add moe_infinity/serving/scheduler.py tests/python/serving/test_scheduler.py
git commit -m "feat(serving): schedule bounded prefills behind decode"
```

## Task 5: Execute chunks without sampling intermediate boundaries

**Files:**
- Modify: `tests/python/serving/test_engine.py`
- Modify: `tests/python/serving/test_api_routes.py`
- Modify: `moe_infinity/serving/model_runner.py:27-154`
- Modify: `moe_infinity/serving/engine.py:84-305,594-690,726-766`

- [ ] **Step 1: Write failing engine progress and eager-fallback tests**

Add `import pytest` to `tests/python/serving/test_engine.py` (used here and in Task 8), then append:

```python
def test_partial_prefill_step_commits_progress_without_emitting_token() -> None:
    config = _make_config()
    config.update(
        enable_chunked_prefill=True,
        prefill_chunk_size=2,
        max_tokens_per_step=2,
    )
    engine = ContinuousBatchingEngine(
        model=MockModel(), engine=MockOffloadEngine(), config=config
    )
    engine.scheduler.set_chunked_prefill_runtime_enabled(True)
    engine.add_request(
        "long", [10, 11, 12, 13, 14],
        SamplingParams(temperature=0.0, max_tokens=1),
    )

    assert engine.step() == []
    sequence = engine._sequences[0]
    assert sequence.num_computed_tokens == 2
    assert sequence.output_token_ids == []
    assert engine.has_pending_requests()


def test_terminal_prefill_is_the_only_prefill_chunk_sampled() -> None:
    config = _make_config()
    config.update(
        enable_chunked_prefill=True,
        prefill_chunk_size=2,
        max_tokens_per_step=2,
    )
    engine = ContinuousBatchingEngine(
        model=MockModel(), engine=MockOffloadEngine(), config=config
    )
    engine.scheduler.set_chunked_prefill_runtime_enabled(True)
    engine.add_request(
        "long", [10, 11, 12, 13, 14],
        SamplingParams(temperature=0.0, max_tokens=1),
    )

    outputs = engine.run_until_done()

    assert outputs == {"long": [15]}
    assert engine.get_stats()["num_prefill_chunks"] == 3


def test_eager_model_disables_requested_chunking() -> None:
    config = _make_config()
    config.update(enable_chunked_prefill=True, prefill_chunk_size=2)
    engine = ContinuousBatchingEngine(
        model=MockModel(), engine=MockOffloadEngine(), config=config
    )

    assert engine.scheduler.chunked_prefill_enabled is False
    assert engine.get_stats()["chunked_prefill_active"] is False
    assert engine.get_stats()["chunked_prefill_fallback_reason"] == (
        "incomplete_qwen3_paged_layer_registry"
    )
```

- [ ] **Step 2: Write failing DFlash partial-prefill guard test**

Append:

```python
def test_dflash_is_not_delegated_after_partial_prefill() -> None:
    speculator = MockSpeculator()
    config = _make_config()
    config.update(
        enable_chunked_prefill=True,
        prefill_chunk_size=2,
        max_tokens_per_step=2,
    )
    engine = ContinuousBatchingEngine(
        model=MockModel(), engine=MockOffloadEngine(), config=config,
        speculative_draft=speculator,
    )
    engine.scheduler.set_chunked_prefill_runtime_enabled(True)
    engine.add_request(
        "long", [10, 11, 12, 13, 14],
        SamplingParams(temperature=0.0, max_tokens=1),
    )

    _ = engine.run_until_done()

    assert speculator.calls == 0


def test_execution_exception_rolls_back_and_requeues_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _make_chunk_engine(prompt=[10, 11, 12, 13], chunk_size=2)
    monkeypatch.setattr(
        engine, "_execute_batch",
        lambda batch: (_ for _ in ()).throw(RuntimeError("forward failed")),
    )

    with pytest.raises(RuntimeError, match="forward failed"):
        engine.step()

    sequence = engine._sequences[0]
    assert sequence.num_computed_tokens == 0
    assert engine.scheduler.inflight_prefill_seq_ids == []
    assert engine.scheduler.schedule().prefill_chunks[0].start_pos == 0


def test_terminal_sampling_exception_rolls_back_and_requeues_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _make_chunk_engine(prompt=[10, 11], chunk_size=2)
    monkeypatch.setattr(
        engine.sampler, "sample",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("sampling failed")
        ),
    )

    with pytest.raises(RuntimeError, match="sampling failed"):
        engine.step()

    sequence = engine._sequences[0]
    assert sequence.num_computed_tokens == 0
    assert sequence.output_token_ids == []
    assert engine.kv_cache.get_num_reserved_tokens(0) == 0
    assert engine.scheduler.inflight_prefill_seq_ids == []
```

`_make_chunk_engine` is a test helper that builds the existing mock engine, binds `PagedAttentionBackend.block_store` through `engine.kv_cache.set_block_store`, enables chunking, and adds request ID `long` as sequence zero. This keeps exception tests on the canonical rollback protocol while model logits remain deterministic.

- [ ] **Step 3: Verify engine behavior fails before implementation**

Run: `pytest -q tests/python/serving/test_engine.py -k 'partial_prefill or terminal_prefill or eager_model or dflash_is_not or exception_rolls_back'`

Expected: FAIL because the engine neither wires chunk config nor supports progress-only steps.

- [ ] **Step 4: Add the paged-attention capability gate**

Add to `ModelRunner`:

```python
    def get_attention_backend(self) -> object | None:
        return self._get_attention_backend()

    def supports_chunked_prefill(self) -> bool:
        backend = self._get_attention_backend()
        qwen_modules = self._get_qwen3_paged_attention_modules()
        expected_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        layer_indices = [int(module.layer_idx) for module in qwen_modules]
        return bool(
            qwen_modules
            and sorted(layer_indices) == list(range(expected_layers))
            and len(set(layer_indices)) == len(layer_indices)
            and isinstance(backend, PagedAttentionBackend)
            and isinstance(backend.block_store, LayeredPagedKVStore)
            and backend.supports_chunked_prefill()
        )

    def chunked_prefill_unavailable_reason(self) -> str:
        qwen_modules = self._get_qwen3_paged_attention_modules()
        expected_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        indices = [int(module.layer_idx) for module in qwen_modules]
        if sorted(indices) != list(range(expected_layers)):
            return "incomplete_qwen3_paged_layer_registry"
        backend = self._get_attention_backend()
        if not isinstance(backend, PagedAttentionBackend) or not isinstance(
            backend.block_store, LayeredPagedKVStore
        ):
            return "layered_paged_kv_store_unavailable"
        if not backend.supports_chunked_prefill():
            return "paged_backend_lacks_chunk_history"
        return "none"
```

In `ContinuousBatchingEngine.__init__`, construct `ModelRunner` and resolve/register the production Qwen backend before computing logical cache geometry. Then construct `PagedKVCache`, call `set_block_store`, construct `Scheduler`, and apply the runtime gate:

```python
        backend = self.model_runner.get_attention_backend()
        if isinstance(backend, PagedAttentionBackend):
            backend.register_layers(
                [
                    LayerRegistration(int(module.layer_idx), id(module))
                    for module in self.model_runner._get_qwen3_paged_attention_modules()
                ]
            )
            self.kv_cache.set_block_store(
                backend.block_store,
                logical_capacity=self.kv_cache.num_blocks,
            )
        paged_chunking = self.model_runner.supports_chunked_prefill()
        self.scheduler.set_chunked_prefill_runtime_enabled(paged_chunking)
        self._chunked_prefill_fallback_reason = (
            None
            if self.scheduler.chunked_prefill_enabled
            or not self.scheduler.chunked_prefill_requested
            else self.model_runner.chunked_prefill_unavailable_reason()
        )
        self._num_prefill_chunks = 0
```

- [ ] **Step 5: Keep chunks in-flight through execution and terminal sampling**

Replace the ordinary execution/sampling region in `step()` with a transaction boundary. Compute sampled tokens into local values before mutating `SequenceData`, request output maps, callbacks, or committed progress:

```python
        transaction_id = scheduler_output.prefill_transaction_id
        sampled_indices = self._sampled_row_indices(batch)
        try:
            logits = self._execute_batch(batch)
            sampler_output = None
            if sampled_indices:
                sampled_logits = self._extract_last_token_logits(
                    logits, batch, sampled_indices
                )
                sampled_params = [
                    batch.sampling_params[index] for index in sampled_indices
                ]
                sampler_output = self.sampler.sample(
                    sampled_logits, sampled_params
                )
        except BaseException:
            self.scheduler.rollback_prefill_step(transaction_id)
            raise

        # If scheduler preflight fails, commit_prefill_step restores KV/sequence
        # state, requeues every row, and re-raises.
        self.scheduler.commit_prefill_step(transaction_id)
        self._num_prefill_chunks += len(scheduler_output.prefill_chunks)
        self._num_steps += 1
        if sampler_output is None:
            return []
```

`commit_prefill_step` must not return until all row-progress participants commit. A preflight or publication exception is fully compensated by the scheduler; the engine must not append sampled output, mutate request output maps, invoke callbacks, or increment counters before it returns successfully. Add an engine test that injects a second-row publication failure after valid model execution/sampling and asserts no output/callback/counter/progress survives and both rows are requeued.

Iterate over `sampled_indices` rather than every batch row when appending output. Change `_extract_last_token_logits` to accept `row_indices: list[int] | None = None` and select only `batch.lengths.query_offsets[index + 1] - 1`. Remove the old unconditional `_num_steps += 1` at the bottom of `step()`. `_step_speculative` and `_step_speculative_session` receive the transaction ID: successful completion commits the same group transaction, while generator/session exceptions roll it back before re-raising.

In `run_until_done`, distinguish progress from deadlock:

```python
            steps_before = self._num_steps
            outputs = self.step()
            if outputs or self._num_steps > steps_before:
                continue
```

Add `num_prefill_chunks`, `chunked_prefill_requested`, `chunked_prefill_active`, `chunked_prefill_fallback_reason`, and scheduler prefill-backpressure steps to `get_stats()`.

- [ ] **Step 6: Tighten DFlash eligibility**

Change `_can_delegate_speculative` to receive `scheduler_output` and require:

```python
        chunk = scheduler_output.prefill_chunks.get(batch.seq_ids[0])
        if chunk is not None and (chunk.start_pos != 0 or not chunk.is_terminal):
            return False
        if batch.lengths.context_lengths != [0] or batch.prefill_is_terminal != [True]:
            return False
```

Pass `scheduler_output` at its call site. This combined predicate preserves one-shot DFlash behavior and explicitly rejects mid-prompt delegation.

- [ ] **Step 7: Run CPU engine, batch, and correctness tests**

Run: `pytest -q tests/python/serving/test_engine.py tests/python/serving/test_correctness.py tests/python/serving/test_batch.py`

Expected: PASS; partial steps return no token, terminal chunks emit exactly one first token, and eager default behavior is unchanged.

- [ ] **Step 8: Commit engine integration**

```bash
git add moe_infinity/serving/model_runner.py moe_infinity/serving/engine.py tests/python/serving/test_engine.py
git commit -m "feat(serving): execute partial prefills without sampling"
```

## Task 6: Preserve partial state through cancellation and preemption

**Files:**
- Modify: `tests/python/serving/test_cancellation.py`
- Modify: `tests/python/unit/test_kv_swap_recovery.py`
- Modify: `moe_infinity/serving/scheduler.py:381-584`

- [ ] **Step 1: Write a cancellation no-leak test**

Append to `tests/python/serving/test_cancellation.py`:

```python
def test_cancel_partial_prefill_frees_reserved_chunks() -> None:
    engine = _make_engine(num_kv_blocks=4, max_batch_size=1)
    engine.kv_cache.set_block_store(
        _make_paged_backend(num_blocks=4).block_store,
        logical_capacity=engine.kv_cache.num_blocks,
    )
    engine.scheduler.chunked_prefill_requested = True
    engine.scheduler.prefill_chunk_size = 4
    engine.scheduler.max_tokens_per_step = 4
    engine.scheduler.set_chunked_prefill_runtime_enabled(True)
    original_free = engine.kv_cache.block_allocator.num_free_blocks
    engine.add_request(
        "partial", list(range(10)),
        SamplingParams(max_tokens=2, temperature=0.0),
    )

    assert engine.step() == []
    assert engine.kv_cache.block_allocator.num_free_blocks < original_free
    engine.abort_request("partial")

    assert engine.kv_cache.block_allocator.num_free_blocks == original_free
    assert engine.has_pending_requests() is False
```

Define `_make_paged_backend` in `test_cancellation.py` with the same cache geometry as `_make_engine`; the test must not manually enable transactional chunking against an unbound cache.

- [ ] **Step 2: Write a partial-prefill swap recovery test**

Append to `tests/python/unit/test_kv_swap_recovery.py`:

```python
def test_partial_prefill_recovers_to_prefill_at_same_offset() -> None:
    cache = _make_kv_cache(4)
    cache.set_block_store(
        _make_paged_backend(num_blocks=4).block_store,
        logical_capacity=cache.num_blocks,
    )
    scheduler = Scheduler(
        kv_cache=cache, max_batch_size=1,
        max_tokens_per_step=4, enable_chunked_prefill=True,
        prefill_chunk_size=4,
    )
    group = _make_group("partial", seq_id=41, prompt_len=8)
    scheduler.add_request(group)
    first = scheduler.schedule()
    scheduler.commit_prefill_step(first.prefill_transaction_id)
    assert group.sequences[0].num_computed_tokens == 4

    preempted = scheduler._preempt_oldest_running_group()
    assert preempted == [41]
    assert group.sequences[0].status is SequenceStatus.SWAPPED
    scheduler._recover_swapped_groups([group])

    assert group.sequences[0].status is SequenceStatus.PREFILL
    resumed = scheduler.schedule()
    assert resumed.prefill_chunks[41].start_pos == 4
```

Add `_make_paged_backend` in this test file with `KVCacheSpec(num_kv_heads=1, head_dim=8, dtype=torch.float16, block_size=4)` and a CPU `PagedAttentionBackend`. This ensures swap recovery exercises the bound runtime storage rather than the allocator's legacy tensor.

- [ ] **Step 3: Verify recovery currently returns to decode**

Run: `pytest -q tests/python/serving/test_cancellation.py -k partial_prefill; pytest -q tests/python/unit/test_kv_swap_recovery.py -k partial_prefill`

Expected: cancellation test may expose stale queue metadata; recovery test FAILS because `_recover_swapped_groups` currently forces `DECODE`.

- [ ] **Step 4: Record and restore the exact preemption state**

Before setting a sequence to `SWAPPED` in `_preempt_oldest_running_group`, store:

```python
                self._swapped_resume_status[sequence.seq_id] = sequence.status
```

During recovery replace the unconditional decode transition with:

```python
                resume = self._swapped_resume_status.pop(
                    sequence.seq_id, SequenceStatus.DECODE
                )
                sequence.set_status(resume)
                if resume is SequenceStatus.PREFILL:
                    self._enqueue_prefill_once(sequence.seq_id)
```

In `abort_request`, completion pruning, and `_drop_request_metadata`, remove IDs from `_prefill_queue`, `_prefill_wait_steps`, `_swapped_resume_status`, and `_verify_demands`. Ensure partial prefills are preferred preemption victims over decode rows, and make chunk admission defer rather than preempt a decode. Preserve the legacy preemption order inside `_schedule_whole_prefill`.

- [ ] **Step 5: Run cancellation, swap, scheduler, and ContextPilot lifecycle tests**

Run: `pytest -q tests/python/serving/test_cancellation.py tests/python/unit/test_kv_swap_recovery.py tests/python/serving/test_scheduler.py tests/python/contextpilot/test_request_id_lifecycle.py tests/python/contextpilot/test_eviction_sync.py`

Expected: PASS with all blocks restored, no cancelled ID rescheduled, and partial offset preserved.

- [ ] **Step 6: Commit lifecycle safety**

```bash
git add moe_infinity/serving/scheduler.py tests/python/serving/test_cancellation.py tests/python/unit/test_kv_swap_recovery.py
git commit -m "fix(serving): preserve partial prefill lifecycle state"
```

## Task 7: Validate paged mixed execution on CPU metadata and CUDA

**Files:**
- Modify: `tests/python/serving/test_model_runner.py`
- Modify: `tests/python/serving/test_flashinfer_model_runner.py`
- Modify: `tests/python/unit/test_flashinfer_attention_backend.py`
- Modify: `tests/python/integration/test_flashinfer_e2e.py`
- Create: `tests/python/serving/test_qwen3_paged_attention_cuda.py`
- Modify: `moe_infinity/runtime/attention_types.py:25-33`
- Modify: `moe_infinity/runtime/attention_backend.py:451-518`
- Modify: `moe_infinity/serving/model_runner.py:185-253,286-315`

- [ ] **Step 1: Add a CPU slot-mapping test for nonzero chunk context**

Add to `tests/python/serving/test_model_runner.py` using its existing runner fixture:

```python
def test_runtime_metadata_maps_partial_prefill_slots() -> None:
    engine = MockOffloadEngine()
    engine.kv_cache = types.SimpleNamespace(block_size=4)
    runner = ModelRunner(MockModel(), engine, device=torch.device("cpu"))
    batch = BatchMetadata(
        seq_ids=[7], input_token_ids=[30, 31, 32],
        lengths=PagedBatchLengths([3], [0, 3], [5], [8]),
        is_prefill=[True],
        prefill_is_terminal=[False], block_tables=[[4, 9]],
        sampling_params=[SamplingParams()],
    )

    metadata = runner._build_runtime_attention_metadata(batch)

    assert metadata.lengths.kv_seq_lengths.tolist() == [8]
    assert metadata.lengths.query_lengths.tolist() == [3]
    assert metadata.lengths.query_offsets.tolist() == [0, 3]
    assert metadata.num_prefill_tokens == 3
    assert metadata.num_decode_tokens == 0
    assert metadata.slot_mapping.tolist() == [9 * 4 + 1, 9 * 4 + 2, 9 * 4 + 3]
```

The local engine uses block size four, so context positions 5, 6, and 7 map to physical page 9 offsets 1, 2, and 3.

- [ ] **Step 2: Make FlashInfer `qo_indptr` query-only and test the real backend object**

Replace runtime metadata with one canonical `PagedBatchLengths` containing the four tensors. `ModelRunner._build_runtime_attention_metadata` converts the serving lists once and assigns `AttentionMetadata.lengths`; it does not expose duplicate direct fields. Replace `_build_flashinfer_metadata` query planning with:

```python
        query_lengths = metadata.lengths.query_lengths.to(
            self.device, dtype=torch.int32
        ).reshape(-1)
        qo_indptr = metadata.lengths.query_offsets.to(
            self.device, dtype=torch.int32
        ).reshape(-1)
        expected_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(query_lengths, dim=0),
            ]
        )
        if not torch.equal(qo_indptr, expected_offsets):
            raise ValueError("query_offsets must be the prefix sum of query_lengths")
```

After each successful FlashInfer `plan`, store a length-only diagnostic snapshot used by real CUDA tests:

```python
@dataclass(frozen=True)
class FlashInferPlanMetadata:
    query_lengths: torch.Tensor
    query_offsets: torch.Tensor
    kv_seq_lengths: torch.Tensor
    kv_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
```

Set `backend.last_flashinfer_plan` only after the wrapper plan succeeds; clone/detach these tensors and include no token IDs.

Add to `tests/python/unit/test_flashinfer_attention_backend.py`:

```python
def test_flashinfer_qo_indptr_uses_chunk_queries_not_total_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=8, device=torch.device("cpu")
    )
    metadata = AttentionMetadata(
        block_tables=torch.tensor([[0, 1], [2, 3, 4]], dtype=torch.int32),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([2, 3], dtype=torch.int32),
            query_offsets=torch.tensor([0, 2, 5], dtype=torch.int32),
            context_lengths=torch.tensor([4, 6], dtype=torch.int32),
            kv_seq_lengths=torch.tensor([6, 9], dtype=torch.int32),
        ),
        max_seq_len=9, num_prefill_tokens=5, num_decode_tokens=0,
        slot_mapping=torch.tensor([4, 5, 14, 15, 16]), is_prefill=True,
    )
    backend.forward(
        query=torch.randn(5, 4, 8),
        key=torch.randn(5, 2, 8),
        value=torch.randn(5, 2, 8),
        attention_metadata=metadata,
    )

    assert backend._fi_prefill is not None
    plan_args = backend._fi_prefill.plan_args[0]
    assert plan_args[0].tolist() == [0, 2, 5]
    assert plan_args[1].tolist() == [0, 2, 5]
    assert plan_args[3].tolist() == [2, 1]
```

The assertions deliberately distinguish query lengths `[2, 3]`, contexts `[4, 6]`, and total KV lengths `[6, 9]`. Update every existing `AttentionMetadata(...)` test constructor to pass exactly one canonical `lengths=PagedBatchLengths(...)`; direct length fields are forbidden.

- [ ] **Step 3: Detect and exercise the real `Qwen3PagedAttention` class**

Add `"Qwen3PagedAttention"` to `ModelRunner._get_paged_attention_classes`. Replace name-only capability with a class protocol check plus this allowlist so arbitrary synthetic classes cannot activate the feature. Add to `tests/python/serving/test_flashinfer_model_runner.py`:

```python
def test_model_runner_detects_real_qwen3_paged_attention() -> None:
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )
    from moe_infinity.models.qwen3_paged_attention import Qwen3PagedAttention

    config = Qwen3MoeConfig(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, num_hidden_layers=1, intermediate_size=64,
        moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
    )
    attention = Qwen3PagedAttention(config, layer_idx=0)
    model = torch.nn.Module()
    model.add_module("qwen_attention", attention)
    backend = _make_transactional_backend(block_size=4)
    runner = ModelRunner(model, _MockEngine(backend), device=torch.device("cpu"))

    assert runner._get_paged_attention_classes() == [Qwen3PagedAttention]
    assert runner.supports_chunked_prefill() is True
```

Create `tests/python/serving/test_qwen3_paged_attention_cuda.py` with an actual Qwen attention invocation through actual FlashInfer wrappers and `PagedAttentionBackend` (no recording backend and no fake wrapper):

```python
import torch
import pytest
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from moe_infinity.models.qwen3_paged_attention import Qwen3PagedAttention
from moe_infinity.runtime.attention_backend import (
    LayerRegistration,
    PagedAttentionBackend,
)
from moe_infinity.runtime.attention_types import AttentionMetadata
from moe_infinity.runtime.attention_types import KVCacheSpec
from moe_infinity.runtime.flashinfer_utils import HAS_FLASHINFER


@pytest.mark.skipif(
    not (HAS_FLASHINFER and torch.cuda.is_available()),
    reason="requires real FlashInfer and CUDA",
)
def test_real_qwen3_paged_attention_runs_chunk_through_flashinfer() -> None:
    device = torch.device("cuda")
    config = Qwen3MoeConfig(
        hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, num_hidden_layers=1, intermediate_size=64,
        moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
    )
    attention = Qwen3PagedAttention(config, layer_idx=0).to(
        device=device, dtype=torch.float16
    ).eval()
    backend = PagedAttentionBackend(
        spec=KVCacheSpec(2, 8, torch.float16, 4),
        num_gpu_blocks=4,
        num_layers=1,
        device=device,
    )
    assert backend._flashinfer_enabled()
    backend.register_layers([LayerRegistration(0, id(attention))])
    prefix_k = torch.randn(4, 2, 8, device=device, dtype=torch.float16)
    prefix_v = torch.randn_like(prefix_k)
    prefix_slots = torch.arange(4, device=device)
    backend.write_kv(prefix_k, prefix_v, prefix_slots, layer_idx=0)
    backend.write_kv_flashinfer(
        prefix_k, prefix_v, prefix_slots, layer_idx=0
    )
    metadata = AttentionMetadata(
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([2], dtype=torch.int32, device=device),
            query_offsets=torch.tensor([0, 2], dtype=torch.int32, device=device),
            context_lengths=torch.tensor([4], dtype=torch.int32, device=device),
            kv_seq_lengths=torch.tensor([6], dtype=torch.int32, device=device),
        ),
        max_seq_len=6, num_prefill_tokens=2, num_decode_tokens=0,
        slot_mapping=torch.tensor([4, 5], device=device), is_prefill=True,
    )
    hidden = torch.randn(1, 2, 32, device=device, dtype=torch.float16)
    cos = torch.ones(1, 2, 8, device=device, dtype=torch.float16)
    sin = torch.zeros(1, 2, 8, device=device, dtype=torch.float16)
    Qwen3PagedAttention.set_paged_context(backend, metadata)
    try:
        output, weights = attention(
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )
    finally:
        Qwen3PagedAttention.clear_paged_context()

    assert output.shape == hidden.shape
    assert weights is None
    assert torch.isfinite(output).all()
    assert backend.last_flashinfer_plan.query_offsets.tolist() == [0, 2]
    assert backend.last_flashinfer_plan.kv_seq_lengths.tolist() == [6]
```

This test is the activation proof. Fake-wrapper CPU tests remain metadata unit tests only and cannot satisfy the rollout gate.

- [ ] **Step 4: Add CUDA/FlashInfer chunk-plus-decode and preemption parity tests**

Add beside `test_e2e_flashinfer_prefill_decode_loop`:

```python
@pytest.mark.skipif(
    not (HAS_FLASHINFER and torch.cuda.is_available()),
    reason="requires FlashInfer and CUDA",
)
def test_e2e_flashinfer_chunked_prefill_with_active_decode() -> None:
    device = torch.device("cuda")
    backend = _make_backend(device=device, dtype=torch.float16)
    engine = ContinuousBatchingEngine(
        model=_MockPagedModel(vocab_size=97, device=device),
        engine=_MockOffloadEngine(attention_backend=backend),
        config={
            **_make_engine_config(dtype="float16"),
            "enable_chunked_prefill": True,
            "prefill_chunk_size": 2,
            "max_tokens_per_step": 3,
        },
    )
    engine.add_request(
        "decode", [10], SamplingParams(temperature=0.0, max_tokens=4)
    )
    first = engine.step()
    assert [row.token_id for row in first] == [11]
    engine.add_request(
        "long", [20, 21, 22, 23, 24],
        SamplingParams(temperature=0.0, max_tokens=1),
    )

    observed = [*first]
    while engine.has_pending_requests():
        observed.extend(engine.step())

    assert engine.get_request_n_outputs("decode") == [[11, 12, 13, 14]]
    assert engine.get_request_n_outputs("long") == [[25]]
    assert engine.get_stats()["num_prefill_chunks"] >= 4


def test_partial_prefill_preemption_preserves_logits_and_output() -> None:
    uninterrupted = _make_chunked_flashinfer_engine()
    resumed = _make_chunked_flashinfer_engine()
    _add_parity_request(uninterrupted)
    _add_parity_request(resumed)

    assert uninterrupted.step() == []
    assert resumed.step() == []
    preempted = resumed.scheduler._preempt_oldest_running_group()
    assert preempted == [0]
    resumed.scheduler._recover_swapped_groups(list(resumed.scheduler._swapped))

    uninterrupted_logits, uninterrupted_output = _finish_and_capture_terminal(
        uninterrupted
    )
    resumed_logits, resumed_output = _finish_and_capture_terminal(resumed)

    torch.testing.assert_close(resumed_logits, uninterrupted_logits)
    assert resumed_output == uninterrupted_output
    assert resumed.kv_cache.block_allocator.num_free_blocks == (
        uninterrupted.kv_cache.block_allocator.num_free_blocks
    )
```

Implement the three test helpers locally in `test_flashinfer_e2e.py` with the file's real `_MockPagedModel`, `_MockOffloadEngine`, and `PagedAttentionBackend`. `_finish_and_capture_terminal` wraps `engine._execute_batch` to clone the terminal row logits before sampling. Mark both tests with the existing CUDA + FlashInfer skip condition.

- [ ] **Step 5: Run CPU metadata/backend/Qwen tests first**

Run: `python -m pytest -q tests/python/serving/test_model_runner.py tests/python/serving/test_flashinfer_model_runner.py tests/python/unit/test_flashinfer_attention_backend.py`

Expected: PASS; `qo_indptr` equals canonical `query_offsets`, page metadata uses `kv_seq_lengths`, and real Qwen3 class/layer detection is covered.

- [ ] **Step 6: Run the CUDA/model integration test**

Run: `python -m pytest -q tests/python/serving/test_qwen3_paged_attention_cuda.py tests/python/integration/test_flashinfer_e2e.py -k 'qwen3_paged_attention_runs_chunk or chunked_prefill_with_active_decode or partial_prefill_preemption or prefill_decode_loop'`

Expected on the rollout CUDA + real-FlashInfer environment: PASS, including the direct Qwen invocation. CPU-only developer runs may SKIP with the explicit requirement reason, but a skip does not satisfy canary acceptance.

- [ ] **Step 7: Commit paged-attention interfaces and coverage**

```bash
git add moe_infinity/runtime/attention_types.py moe_infinity/runtime/attention_backend.py moe_infinity/serving/model_runner.py tests/python/serving/test_model_runner.py tests/python/serving/test_flashinfer_model_runner.py tests/python/unit/test_flashinfer_attention_backend.py tests/python/serving/test_qwen3_paged_attention_cuda.py tests/python/integration/test_flashinfer_e2e.py
git commit -m "fix(serving): separate chunk query and KV metadata"
```

## Task 8: Expose a disabled-by-default feature and document rollout

**Files:**
- Modify: `tests/python/serving/test_engine.py`
- Modify: `moe_infinity/serving/engine.py:679-690`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:475-522,1849-1919`
- Modify: `moe_infinity/entrypoints/big_modeling.py:915-969`
- Modify: `docs/serving.md`

- [ ] **Step 1: Write config validation and default tests**

Append to `tests/python/serving/test_engine.py`:

```python
def test_chunked_prefill_defaults_disabled() -> None:
    engine = _make_engine()
    assert engine.scheduler.chunked_prefill_requested is False
    assert engine.get_config().get("enable_chunked_prefill", False) is False


@pytest.mark.parametrize(
    ("key", "value"),
    [("prefill_chunk_size", 0), ("prefill_starvation_threshold_steps", 0)],
)
def test_chunked_prefill_config_requires_positive_integers(
    key: str, value: int
) -> None:
    config = _make_config()
    config["enable_chunked_prefill"] = True
    config[key] = value
    with pytest.raises(ValueError, match=key):
        ContinuousBatchingEngine(
            model=MockModel(), engine=MockOffloadEngine(), config=config
        )


def test_build_engine_config_propagates_chunked_prefill_flags() -> None:
    args = SimpleNamespace(
        device_memory_ratio=0.75, kv_cache_ratio=0.25, max_batch_size=8,
        enable_prefix_caching=False, enable_chunked_prefill=True,
        prefill_chunk_size=256, prefill_starvation_threshold_steps=4,
    )
    config = srv._build_engine_config(args=args, model=_ConfigurableMockModel())
    assert config["enable_chunked_prefill"] is True
    assert config["prefill_chunk_size"] == 256
    assert config["prefill_starvation_threshold_steps"] == 4
```

Place the final test in `tests/python/serving/test_api_routes.py` and use that file's existing configurable mock model/config helper rather than introducing another server fixture.

- [ ] **Step 2: Verify invalid settings are currently accepted**

Run: `pytest -q tests/python/serving/test_engine.py -k 'chunked_prefill_defaults or chunked_prefill_config'`

Expected: FAIL on the non-positive numeric controls; Task 2B already added the boolean flag and co-enablement guard, but chunk-size validation is not implemented yet.

- [ ] **Step 3: Wire CLI and programmatic controls**

Add these CLI arguments in `parse_args()`:

```python
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument(
        "--prefill-starvation-threshold-steps", type=int, default=8
    )
```

Add matching keyword arguments and defaults to `initialize_with_model()` and `MoE.serve()` and place them on the namespace passed to `_build_engine_config`. Task 2B already propagates and validates `enable_chunked_prefill`; add the two numeric controls here:

```python
        "prefill_chunk_size": int(args.prefill_chunk_size),
        "prefill_starvation_threshold_steps": int(
            args.prefill_starvation_threshold_steps
        ),
```

Keep all defaults disabled. Do not make `enable_chunked_prefill` mutable through `/v1/config`; scheduler queue-mode changes require a new engine. Keep the existing runtime-mutability whitelist restricted to `max_batch_size` and `max_tokens_per_step`.

- [ ] **Step 4: Document semantics, risks, and rollout**

Add a “Chunked prefill (experimental)” section to `docs/serving.md` containing:

```markdown
### Chunked prefill (experimental)

Chunked prefill is disabled by default. Enable it with
`--enable-chunked-prefill`; tune the hard per-row bound with
`--prefill-chunk-size` (default `512`) and prefill-only age promotion with
`--prefill-starvation-threshold-steps` (default `8`). `max_tokens_per_step`
remains the total decode-plus-prefill budget. Runnable decode rows consume one
token each before prefill receives the remaining budget.

Activation requires complete Qwen3 paged-layer registration, real FlashInfer,
and the canonical production `LayeredPagedKVStore`. Logical block capacity is
`min(memory_budget_blocks, block_store.physical_capacity)`; unequal capacities are
valid. If any capability is unavailable,
the engine reports `chunked_prefill_active=false` and
an explicit `chunked_prefill_fallback_reason`, then uses the
unchanged whole-prefill path. The first release rejects simultaneous
`enable_chunked_prefill=true` and `enable_prefix_caching=true` at startup with
`ValueError: enable_chunked_prefill and enable_prefix_caching cannot both be
true in the first release`. Chunk transactions are standalone scheduler/KV
transactions and do not import or duplicate a prefix implementation. Roll back
by removing `--enable-chunked-prefill` (or disabling prefix caching) and
restarting; no request or cache-format migration is required. PR #181 is design
input for a future reconciliation only, not an implementation dependency.
DFlash is used only for singleton prompts completed in one prefill launch;
partially prefetched prompts stay on ordinary paged decode.

Operational risks are extra scheduler launches for long prompts, page-boundary
fragmentation, prefill backpressure under saturated decode load, model/backend
metadata incompatibility, and latency regressions from a poorly chosen chunk
size. Roll out first in shadow benchmarks, then a paged-backend canary, then a
small opt-in production cohort. Roll back by removing
`--enable-chunked-prefill`; no persisted cache or request format migration is
required.
```

- [ ] **Step 5: Run engine and entrypoint tests**

Run: `pytest -q tests/python/serving/test_engine.py tests/python/serving/test_api_routes.py -k 'chunked_prefill or config'`

Expected: PASS. Both command paths are existing collected test modules.

- [ ] **Step 6: Commit the opt-in surface and docs**

```bash
git add moe_infinity/serving/engine.py moe_infinity/entrypoints/openai/api_server_v2.py moe_infinity/entrypoints/big_modeling.py tests/python/serving/test_engine.py tests/python/serving/test_api_routes.py docs/serving.md
git commit -m "feat(serving): expose experimental chunked prefill"
```

## Task 9: Add paired TTFT/TPOT tail benchmarks without performance claims

**Files:**
- Create: `benchmarks/serving/chunked_prefill_latency.py`
- Create: `tests/python/serving/test_chunked_prefill_benchmark.py`
- Modify: `docs/benchmarking.md`

- [ ] **Step 1: Write failing aggregation tests**

Create `tests/python/serving/test_chunked_prefill_benchmark.py`:

```python
import pytest

from benchmarks.serving.chunked_prefill_latency import (
    build_prompt_token_ids,
    summarize_requests,
)


def test_summarize_requests_reports_ttft_and_tpot_tails() -> None:
    summary = summarize_requests(
        [
            {"started_at": 1.0, "token_times": [1.1, 1.2, 1.4]},
            {"started_at": 2.0, "token_times": [2.2, 2.5, 2.9]},
        ]
    )

    assert summary["request_count"] == 2
    assert summary["ttft_p50_ms"] == pytest.approx(150.0)
    assert summary["ttft_p99_ms"] == pytest.approx(199.0)
    assert summary["tpot_p50_ms"] == pytest.approx(250.0)
    assert summary["tpot_p99_ms"] == pytest.approx(397.0)


def test_summarize_requests_keeps_single_token_tpot_null() -> None:
    summary = summarize_requests(
        [{"started_at": 1.0, "token_times": [1.1]}]
    )
    assert summary["ttft_p99_ms"] == pytest.approx(100.0)
    assert summary["tpot_p99_ms"] is None


def test_build_prompt_token_ids_is_tokenizer_verified_exact_length() -> None:
    class Tokenizer:
        vocab_size = 32
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return [3, 5, 7]

    prompt = build_prompt_token_ids(Tokenizer(), target_length=8)

    assert prompt == [3, 5, 7, 3, 5, 7, 3, 5]
    assert len(prompt) == 8
```

- [ ] **Step 2: Verify the benchmark module is absent**

Run: `pytest -q tests/python/serving/test_chunked_prefill_benchmark.py`

Expected: collection FAIL with `ModuleNotFoundError: benchmarks.serving.chunked_prefill_latency`.

- [ ] **Step 3: Implement deterministic aggregation and paired workload CLI**

Create `benchmarks/serving/chunked_prefill_latency.py` with:

```python
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = p / 100.0 * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_requests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    for row in rows:
        started = float(row["started_at"])
        token_times = [float(value) for value in row["token_times"]]
        if not token_times:
            continue
        ttft_ms.append((token_times[0] - started) * 1000.0)
        tpot_ms.extend(
            (right - left) * 1000.0
            for left, right in zip(token_times, token_times[1:])
        )
    return {
        "request_count": len(rows),
        "ttft_p50_ms": percentile(ttft_ms, 50.0),
        "ttft_p90_ms": percentile(ttft_ms, 90.0),
        "ttft_p99_ms": percentile(ttft_ms, 99.0),
        "tpot_p50_ms": percentile(tpot_ms, 50.0),
        "tpot_p90_ms": percentile(tpot_ms, 90.0),
        "tpot_p99_ms": percentile(tpot_ms, 99.0),
    }


async def stream_request(
    client: httpx.AsyncClient, url: str,
    prompt_token_ids: list[int], max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    token_times: list[float] = []
    token_payloads: list[str] = []
    payload = {
        "model": "benchmark",
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    async with client.stream("POST", f"{url}/v1/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                token_times.append(time.perf_counter())
                event = json.loads(line.removeprefix("data: "))
                choice = event["choices"][0]
                token_payloads.append(str(choice.get("text", "")))
    return {
        "started_at": started,
        "token_times": token_times,
        "token_payloads": token_payloads,
    }
```

Complete the same file with this concrete paired runner:

```python
def build_prompt_token_ids(tokenizer: Any, target_length: int) -> list[int]:
    if target_length <= 0:
        raise ValueError("prompt token target must be positive")
    base = tokenizer.encode(
        "MoE Infinity deterministic chunked prefill benchmark",
        add_special_tokens=False,
    )
    if not base:
        raise ValueError("tokenizer returned an empty benchmark prompt")
    prompt = [int(base[index % len(base)]) for index in range(target_length)]
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and any(
        token < 0 or token >= vocab_size for token in prompt
    ):
        raise ValueError("benchmark prompt contains token outside vocabulary")
    if len(prompt) != target_length:
        raise AssertionError("benchmark prompt length is not exact")
    return prompt


async def poll_stats(
    client: httpx.AsyncClient, url: str, stop: asyncio.Event,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while not stop.is_set():
        response = await client.get(f"{url}/admin/stats")
        response.raise_for_status()
        samples.append(response.json())
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.01)
        except asyncio.TimeoutError:
            pass
    return samples


async def run_arm(
    url: str, tokenizer: Any, *, short_requests: int, long_requests: int,
    short_prompt_tokens: int, long_prompt_tokens: int, max_tokens: int,
) -> dict[str, Any]:
    work = [
        ("long", build_prompt_token_ids(tokenizer, long_prompt_tokens))
        for index in range(long_requests)
    ] + [
        ("short", build_prompt_token_ids(tokenizer, short_prompt_tokens))
        for index in range(short_requests)
    ]
    work.sort(key=lambda item: (item[0] != "long", item[1]))
    timeout = httpx.Timeout(600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        config_response = await client.get(f"{url}/v1/config")
        config_response.raise_for_status()

        async def delayed(index: int, prompt: list[int]) -> dict[str, Any]:
            await asyncio.sleep(index * 0.005)
            try:
                return await stream_request(client, url, prompt, max_tokens)
            except Exception as exc:
                return {
                    "started_at": time.perf_counter(),
                    "token_times": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

        stop = asyncio.Event()
        stats_task = asyncio.create_task(poll_stats(client, url, stop))
        arm_started = time.perf_counter()
        rows = await asyncio.gather(
            *(delayed(index, prompt) for index, (_, prompt) in enumerate(work))
        )
        arm_seconds = time.perf_counter() - arm_started
        stop.set()
        stats_samples = await stats_task
    total_output_tokens = sum(len(row.get("token_payloads", [])) for row in rows)
    peak_used_blocks = max(
        (
            int(sample["kv_cache_num_blocks"])
            - int(sample["kv_cache_free_blocks"])
            for sample in stats_samples
        ),
        default=0,
    )
    return {
        "config": config_response.json(),
        "stats_samples": stats_samples,
        "raw_requests": rows,
        "summary": summarize_requests(rows),
        "error_count": sum("error" in row for row in rows),
        "wall_time_s": arm_seconds,
        "total_output_tokens": total_output_tokens,
        "throughput_tokens_per_s": (
            total_output_tokens / arm_seconds if arm_seconds > 0 else None
        ),
        "peak_kv_used_blocks": peak_used_blocks,
        "peak_kv_utilization": max(
            (
                1.0 - int(sample["kv_cache_free_blocks"])
                / int(sample["kv_cache_num_blocks"])
                for sample in stats_samples
                if int(sample["kv_cache_num_blocks"]) > 0
            ),
            default=0.0,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--short-requests", type=int, default=64)
    parser.add_argument("--long-requests", type=int, default=16)
    parser.add_argument("--short-prompt-tokens", type=int, default=128)
    parser.add_argument("--long-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-label", default="unspecified")
    parser.add_argument("--gpu-label", default="unspecified")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


async def run_paired(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    arms: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    kwargs = {
        "short_requests": args.short_requests,
        "long_requests": args.long_requests,
        "short_prompt_tokens": args.short_prompt_tokens,
        "long_prompt_tokens": args.long_prompt_tokens,
        "max_tokens": args.max_tokens,
    }
    for _ in range(args.rounds):
        arms["baseline"].append(
            await run_arm(args.baseline_url, tokenizer, **kwargs)
        )
        arms["candidate"].append(
            await run_arm(args.candidate_url, tokenizer, **kwargs)
        )
    baseline_rows = [
        row for run in arms["baseline"] for row in run["raw_requests"]
    ]
    candidate_rows = [
        row for run in arms["candidate"] for row in run["raw_requests"]
    ]
    baseline = summarize_requests(baseline_rows)
    candidate = summarize_requests(candidate_rows)
    output_parity = [
        row.get("token_payloads", []) for row in baseline_rows
    ] == [row.get("token_payloads", []) for row in candidate_rows]
    deltas = {
        key: (
            None if baseline[key] is None or candidate[key] is None
            else candidate[key] - baseline[key]
        )
        for key in ("ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
                    "tpot_p50_ms", "tpot_p90_ms", "tpot_p99_ms")
    }
    return {
        "model_label": args.model_label,
        "gpu_label": args.gpu_label,
        "workload": kwargs | {"rounds": args.rounds},
        "baseline": baseline,
        "candidate": candidate,
        "output_parity": output_parity,
        "baseline_error_count": sum("error" in row for row in baseline_rows),
        "candidate_error_count": sum("error" in row for row in candidate_rows),
        "baseline_throughput_tokens_per_s": [
            run["throughput_tokens_per_s"] for run in arms["baseline"]
        ],
        "candidate_throughput_tokens_per_s": [
            run["throughput_tokens_per_s"] for run in arms["candidate"]
        ],
        "baseline_peak_kv_used_blocks": max(
            run["peak_kv_used_blocks"] for run in arms["baseline"]
        ),
        "candidate_peak_kv_used_blocks": max(
            run["peak_kv_used_blocks"] for run in arms["candidate"]
        ),
        "baseline_peak_kv_utilization": max(
            run["peak_kv_utilization"] for run in arms["baseline"]
        ),
        "candidate_peak_kv_utilization": max(
            run["peak_kv_utilization"] for run in arms["candidate"]
        ),
        "candidate_minus_baseline_ms": deltas,
        "runs": arms,
    }


def main() -> int:
    args = parse_args()
    for name in ("short_requests", "long_requests", "short_prompt_tokens",
                 "long_prompt_tokens", "max_tokens", "rounds"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    payload = asyncio.run(run_paired(args))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Do not calculate or print “speedup”; the output contains neutral candidate-minus-baseline milliseconds, complete raw timings, config snapshots, and explicit model/GPU labels.

- [ ] **Step 4: Run benchmark unit tests**

Run: `pytest -q tests/python/serving/test_chunked_prefill_benchmark.py`

Expected: PASS with exact interpolated percentiles and null TPOT for a single-token response.

- [ ] **Step 5: Document the paired benchmark and acceptance gates**

Add to `docs/benchmarking.md`:

````markdown
### Chunked-prefill TTFT/TPOT tails

Start two identical paged-attention servers: baseline without the feature and
candidate with `--enable-chunked-prefill --prefill-chunk-size 512`. Warm both
servers with the same requests, then run:

```bash
python benchmarks/serving/chunked_prefill_latency.py \
  --baseline-url http://127.0.0.1:8000 \
  --candidate-url http://127.0.0.1:8001 \
  --tokenizer /models/the-exact-served-tokenizer \
  --short-requests 64 --long-requests 16 \
  --short-prompt-tokens 128 --long-prompt-tokens 8192 \
  --max-tokens 128 --rounds 5 \
  --output-json chunked-prefill-paired.json
```

Repeat for chunk sizes 128, 256, 512, and 1024 at fixed hardware, model,
offload layout, tokenizer, request trace, and seed. Requests are sent as exact
token-ID arrays produced and range-checked by that tokenizer. Report p50/p90/p99
TTFT and TPOT, measured output tokens/second, errors, prefill-backpressure steps,
peak KV used blocks, and peak KV utilization. Do not infer a speedup from a
single run.

Before latency canarying, the direct real-Qwen3/real-FlashInfer test must PASS
(not skip), the valid unequal-capacity test must report
`logical_blocks=min(memory_budget_blocks, block_store.physical_capacity)`, and
the later-row reservation/checkpoint plus every scheduler-preflight failure test
must restore all progress, rows, block references, and tables. Confirm that a
server with both feature flags fails startup with the documented exact error;
benchmark only the supported both-disabled and chunking-only configurations.

Latency acceptance requires output-token parity, zero request errors, no final
KV-block leak, non-null measured throughput and peak-KV fields, candidate p99
TPOT no more than 5% above baseline, and candidate p99 TTFT no more than 5%
above baseline. If any gate fails, keep the default disabled and retain the
paired JSON for diagnosis.
````

- [ ] **Step 6: Commit benchmark coverage**

```bash
git add benchmarks/serving/chunked_prefill_latency.py tests/python/serving/test_chunked_prefill_benchmark.py docs/benchmarking.md
git commit -m "bench(serving): measure chunked prefill TTFT and TPOT tails"
```

## Task 10: Final compatibility and rollout verification

**Files:**
- Verify all files listed in the file map; no additional production files.

- [ ] **Step 1: Run the complete CPU serving suite with the default disabled**

Run: `python -m pytest -q tests/python/serving tests/python/unit/test_kv_edge_cases.py tests/python/unit/test_kv_swap_recovery.py tests/python/contextpilot/test_cp_scheduler_v2.py tests/python/contextpilot/test_request_id_lifecycle.py`

Expected: PASS. Existing whole-prefill scheduling, eager execution, DFlash deficit scheduling, cancellation, ContextPilot ordering, and KV lifecycle tests remain green.

The collected config tests must include the both-enabled startup case and assert the exact first-release incompatibility error. A silent fallback, automatic flag override, or missing-prefix import is a failure.

- [ ] **Step 2: Run static diagnostics on every changed Python file**

Run: `python -m compileall -q moe_infinity/serving moe_infinity/runtime/attention_backend.py moe_infinity/runtime/attention_types.py moe_infinity/models/qwen3_paged_attention.py moe_infinity/entrypoints/openai/api_server_v2.py moe_infinity/entrypoints/big_modeling.py benchmarks/serving/chunked_prefill_latency.py`

Expected: exit code 0 and no output. Then run the repository's configured formatter/type checker on the changed files; if no aggregate command is documented, run `ruff check` and `pyright` on the exact changed Python paths. Expected: zero new errors.

- [ ] **Step 3: Run model/CUDA correctness gates**

Run: `python -m pytest -q tests/python/serving/test_qwen3_paged_attention_cuda.py tests/python/integration/test_flashinfer_e2e.py tests/python/integration/test_model_smoke.py`

Expected on the rollout GPU environment: PASS with no skip for `test_qwen3_paged_attention_cuda.py`. CPU-only local runs may skip hardware tests, but cannot satisfy rollout.

- [ ] **Step 4: Run paired latency canary measurements**

Run the command documented in Task 9 for disabled/enabled servers, then repeat chunk sizes 128, 256, 512, and 1024.

Expected: valid JSON for both arms with identical request counts and no errors. Treat TTFT/TPOT values as measurements, not promised improvements; advance rollout only if every documented acceptance gate passes.

- [ ] **Step 5: Execute rollout and rollback checkpoints**

1. Merge with `enable_chunked_prefill=False` and observe default-path error rate/KV usage.
2. Enable only on a paged-attention staging server and compare deterministic output tokens.
3. Canary a small explicit production cohort while monitoring p99 TTFT, p99 TPOT, prefill-backpressure steps, free KV blocks, cancellation errors, and fallback reason.
4. Expand only after the paired benchmark gates and canary gates pass across representative prompt mixes.
5. Roll back immediately by removing `--enable-chunked-prefill` if any gate fails; no stored data migration or cache conversion is needed.

- [ ] **Step 6: Stop after the first clean verification**

Do not create an empty or verification-only commit. A failed gate returns work to the task that owns the failing implementation/test pair, which must repeat its red-green cycle and atomic checkpoint before this final verification is rerun.

## Risks and explicit non-goals

- **Decode saturation:** strict decode priority can indefinitely defer prefill when decode alone consumes all row/token capacity. The scheduler exposes backpressure steps; the first release does not weaken decode SLO to guarantee prefill service.
- **KV reservation lead:** capacity is reserved before model execution and committed afterward. Any prepare/execution/sampling/participant-commit exception aborts every scheduler participant, restores the canonical store checkpoint, sequence snapshot, and pre-transaction block table, then requeues the entire group. No asynchronous second batch may be scheduled before rollback/commit resolves.
- **Page fragmentation:** small chunks can reserve partially filled pages and increase launch count. Chunk-size sweeps, free-block telemetry, and rollback gates address this without claiming a universal optimum.
- **Attention compatibility:** eager/Hugging Face cache execution lacks the paged history contract required for independent chunks. It falls back to whole prefill rather than attempting incorrect attention.
- **Mixed-batch correctness:** paged prefill and decode remain separate model-runner launches and are recombined in original row order. Chunking does not introduce a mixed kernel.
- **DFlash:** no partial-prompt DFlash handoff and no changes to DRAFT/VERIFY deficit scheduling. One-shot singleton eligibility remains intact.
- **Prefix caching coexistence:** The first release rejects co-enablement before scheduler/KV initialization. It does not copy, import, or modify prefix-caching code. Reconcile the two transaction models in a future plan using PR #181 as design input only.
- **No P/D disaggregation:** no new workers, RPC, cache transfer protocol, or process topology.
- **No performance promise:** benchmark thresholds are rollout safety criteria. Results must be measured on named hardware/model/workloads before any claim.
