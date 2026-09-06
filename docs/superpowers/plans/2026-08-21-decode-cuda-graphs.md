# Decode CUDA Graphs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate decode-only CUDA graph capture/replay for resident models using the native paged-attention path into active continuous-batching serving, with explicit capability proof and eager execution as the authoritative fallback.

**Architecture:** First make one `PagedKVStorage` object the sole owner of block allocation and per-layer K/V tensors, then register only ordinary-GQA `Qwen3PagedAttention` modules by exact class, integer `layer_idx`, layer-bound backend proxy, and storage slice. The native backend launches an allocation-free in-place KV-write kernel for the current token at `slot_mapping` before each layer's decode attention reads the cache. Keep scheduling, sampling, callbacks, and CPU state eager, while `CudaGraphRunner` owns same-device fixed-address inputs/metadata/outputs; eligibility requires explicit runtime, Qwen3-GQA class-registration, per-layer-write, backend, storage, and exact-device proofs. DeepSeek V2/V3 MLA remains eager with `mla_layout_unsupported`.

**Tech Stack:** Python 3, PyTorch CUDA Graphs, MoE-Infinity continuous batching, paged KV cache, native paged attention, pytest, Prometheus text exposition, CUDA events/profiler timing.

---

## Findings and non-negotiable design decisions

1. **The current `CudaGraphRunner` is not semantically sufficient for serving.** Its private `_forward_decode()` calls `model.forward()` directly, bypassing `ModelRunner.execute()` responsibilities: expert tracer setup, offload request-id advancement, paged-attention context install/cleanup, runtime attention metadata, packed-logit validation, and the active engine's mixed prefill/decode split. It only stores `input_ids`, `position_ids`, and `attention_mask`; changing block tables, sequence lengths, and slot mappings are therefore invisible to replay. It is also not consumed by `ContinuousBatchingEngine`.
2. **Sampling stays outside the graph.** `Sampler.sample()` contains per-row Python control flow, `.item()`, top-k/top-p filtering, multinomial RNG, and optional logprob materialization. The graph returns logits for real rows; existing eager sampling behavior remains unchanged.
3. **Padding is compute padding, never scheduler padding.** A runtime decode batch of size `N` may use the smallest configured batch bucket `B >= N`. Rows `[0:N]` retain their order and real KV metadata. Rows `[N:B]` use unique graph scratch pages allocated by the same `PagedKVStorage.block_allocator` whose tensors native attention reads. Dummy rows never enter `BatchMetadata.seq_ids`, scheduler accounting, request callbacks, output usage, or generated-token counters.
4. **Context bucketing is required for native paged attention.** Graph keys are `(batch_bucket, context_bucket)`. `context_bucket` is the smallest configured upper bound at least `max(context_length + 1)`; the fixed block-table width is `ceil(context_bucket / block_size)`. Actual `seq_lens` remain tensor inputs used by kernel masks. A batch larger than the largest bucket, a context longer than the largest context bucket, or a block table wider than the key falls back eagerly.
5. **There is one allocator/storage owner, not two caches with coincidentally similar block IDs.** Today `serving.PagedKVCache` owns `_kv_cache` plus `BlockAllocator`, while `runtime.PagedAttentionBackend` separately owns `k_cache`, `v_cache`, optional `_fi_kv_cache`, and `num_gpu_blocks`. Reserving a block in the former does not prove that native attention reads the corresponding page in the latter. Before graph work, introduce `PagedKVStorage` with a unique `owner_id`, one `BlockAllocator`, and per-layer native K/V tensors; both `PagedKVCache` and `PagedAttentionBackend` receive the same instance. Startup fails the graph capability with `kv_storage_mismatch` unless object identity, owner ID, spec, block count, and tensor pointers agree.
6. **Capture never uses live request KV pages.** Warmup and capture use one unique reserved scratch page per bucket row with synthetic `seq_len=1`. Context-sized block-table tensors keep stable shapes, but only column zero is used by synthetic/padded rows; real replay rows supply actual page lists. Scratch reservation is deducted from the authoritative allocator's free count, writes into the authoritative native K/V tensors, is reported separately, and is released by `close()`.
7. **First rollout is native-paged ordinary GQA Qwen3 only.** The sole eligible attention family is exact-type `moe_infinity.models.qwen3_paged_attention.Qwen3PagedAttention` with ordinary GQA tensors sharing one `head_dim`. DeepSeek V2/V3 paged attention is MLA: its query/key/value layout cannot be represented by the first-rollout `PagedKVStorageSpec`, so it is rejected as `mla_layout_unsupported`; never coerce MLA dimensions into one `head_dim`. Non-paged models are ineligible (`native_paged_required`) because their `use_cache=True` output/cache persistence and pointer stability are not designed or tested here. FlashInfer is ineligible (`flashinfer_plan_path`) because `plan()` and metadata/workspace management allocate and specialize dynamically. Offloaded MoE is ineligible until a future piecewise design places graph breaks around routing, expert transfer/dispatch, and attention as TensorRT-LLM does.
8. **Capability is explicit and deny-by-default.** `CudaGraphRunner` consumes a `DecodeGraphCapability`; it never guesses safety using `getattr`. The model runtime and backend each construct typed capability evidence. Tests require individual rejection reasons for active model hooks, Archer begin/end callbacks, transfer scheduler, expert dispatcher, KV offload, FlashInfer plan path, dynamic allocations, missing provider, non-native-paged execution, and KV ownership mismatch.
9. **No automatic speedup claim or automatic enablement.** The default is disabled. Benchmarks report raw paired measurements, replay coverage, capture memory, and launch counts; operators decide whether to enable a qualified resident ordinary-GQA Qwen3 configuration.
10. **The sole supported paged class is an exact type, not a name heuristic.** The eligible registry resolves only `moe_infinity.models.qwen3_paged_attention.Qwen3PagedAttention`. It separately recognizes installed exact DeepSeek V2/V3 paged classes only to return `mla_layout_unsupported`, never to bind them or issue a write proof. Any other paged class is eager-only with `paged_class_unregistered`. Every Qwen3 instance must expose a unique in-range integer `layer_idx` and receive a layer-bound backend/storage lookup.
11. **Decode must append before it attends.** The current `PagedAttentionBackend.forward()` writes K/V only in the prefill branch and sends decode directly to `_decode_forward()`, so replay can read stale cache pages. Eligible decode must first launch the graph-capturable `paged_kv_write` kernel for every registered layer using that layer's current-token K/V and authoritative `slot_mapping`, then launch attention. Persistence tests inspect the exact page/offset after eager and replay steps.
12. **Device identity is exact, not merely CUDA-compatible.** Eligibility requires `PagedKVStorage.spec.device == ModelRunner.device` after canonical CUDA-index resolution and requires every graph-owned input, metadata, scratch-facing, and output tensor to have that exact device. A `cuda:0` storage with a `cuda:1` runner or any graph buffer on another device is `kv_storage_mismatch`; capture is never attempted.

TensorRT-LLM's [Torch Compile & Piecewise CUDA Graph guide](https://nvidia.github.io/TensorRT-LLM/features/torch_compile_and_piecewise_cuda_graph.html) motivates the safety boundary: unsupported dynamic components remain eager, stable pointers are required across graph boundaries, padding trades execution coverage against memory/compute, and more capture points consume memory and can reduce concurrency. This plan is deliberately narrower: whole-decode capture is useful only for resident ordinary-GQA Qwen3 paged attention. DeepSeek MLA and offloaded MoE remain eager until separate designs define their layouts and eager boundaries.

## Acceptance gates

The implementation is not releasable until all of these are true:

1. Scheduler block tables, graph scratch IDs, native attention reads/writes, and memory accounting all reference the same `PagedKVStorage.owner_id`, allocator instance, block count, and per-layer tensor pointers.
2. Every explicit capability reason has a deterministic CPU test: `missing_capability`, `active_model_hooks`, `archer_callbacks`, `transfer_scheduler`, `expert_dispatcher`, `kv_offload`, `flashinfer_plan_path`, `dynamic_allocations`, `native_paged_required`, `mla_layout_unsupported`, `kv_storage_mismatch`, `paged_class_unregistered`, `layer_idx_invalid`, and `layer_write_unproven`.
3. Only `eligible` resident ordinary-GQA Qwen3 batches can allocate scratch pages or enter capture. DeepSeek V2/V3 MLA, non-paged, FlashInfer, and offloaded MoE tests prove zero captures and eager output parity.
4. CPU gate tests prove exact device equality among storage, `ModelRunner`, and every graph buffer; CUDA tests prove token/position/block-table/sequence-length updates and padded replay match eager native-paged decode while retaining pointer identity. For every registered Qwen3 layer, current-token K/V appears at the exact authoritative `slot_mapping` page/offset before attention and persists into the following token's eager/replay result.
5. Application shutdown and both engine initialization/replacement paths call the old/current runner's idempotent close path after active steps finish; tests prove scratch blocks return to the authoritative allocator and lock order cannot deadlock.
6. `--mode fixture` runs without model/offload arguments using the formally defined persistent native-paged fixture; `--mode model` rejects missing `--model` or `--offload-dir` and reports offloaded capability fallback honestly.
7. Metrics expose capability reason, fallbacks, replay coverage, graph bytes, and authoritative scratch bytes without unbounded labels. Rollout documentation states the utility boundary as resident ordinary-GQA Qwen3 only and makes no speedup claim.

## File map

- Create `moe_infinity/runtime/paged_kv_storage.py`: own the one block allocator, owner identity, storage spec, per-layer native K/V tensors, and graph scratch reservations.
- Create `moe_infinity/models/paged_attention_registry.py`: resolve exact ordinary-GQA Qwen3 support, explicitly classify DeepSeek V2/V3 MLA as unsupported, create one layer-bound subclass/proxy per eligible Qwen3 module instance, validate unique `layer_idx`, install batch metadata, and emit per-layer write proofs.
- Create `moe_infinity/kernel/paged_kv_write.py`: graph-capturable allocation-free Triton current-token K/V write into authoritative per-layer native storage.
- Modify `moe_infinity/runtime/attention_types.py`: define `DecodeGraphCapability`, fixed rejection reasons, and the runtime capability-provider protocol.
- Modify `moe_infinity/serving/model_runner.py`: consume explicit capability evidence and provide the shared prepared native-paged decode-forward API; preserve request-id and paged-context semantics.
- Modify `moe_infinity/runtime/attention_backend.py`: consume the authoritative `PagedKVStorage`, index it by layer, expose native/FlashInfer capability evidence, and use prebuilt fixed-address metadata.
- Modify `moe_infinity/models/qwen3_paged_attention.py`: pass `layer_idx` and graph mode to the backend so every layer addresses its own authoritative storage tensors.
- Modify `moe_infinity/runtime/model_offload.py`: recognize optional upstream `DeepseekV2PagedAttention` and `DeepseekV3PagedAttention` by exact type and report `mla_layout_unsupported`; do not register, adapt, or graph them.
- Modify `moe_infinity/serving/kv_cache.py`: keep `storage: PagedKVStorage | None = None` constructor compatibility for existing callers, manage sequence block tables through authoritative storage when bound, retain legacy self-owned allocation when omitted, and require bound storage for graph eligibility.
- Modify `moe_infinity/runtime/model_offload.py`: implement an explicit deny-by-default capability provider with distinct hook/Archer/transfer/dispatcher/KV-offload reasons.
- Modify `moe_infinity/entrypoints/big_modeling.py`: construct one `PagedKVStorage` for the native paged backend, expose it for continuous serving adoption, and report the still-separate native cache manager/transfer scheduler as explicitly ineligible.
- Rewrite `moe_infinity/serving/cuda_graph.py`: eligibility, bucket selection, stable buffers, lazy warmup/capture, replay, locking, invalidation, failure quarantine, and metrics.
- Modify `moe_infinity/serving/engine.py`: instantiate the runner, route pure decode sub-batches through it, retain eager prefill/mixed recombination and sampler boundaries, expose stats, and close/invalidate safely.
- Modify `moe_infinity/serving/memory_manager.py`: report graph private-pool and scratch-page reservations separately from model/expert/KV budgets.
- Modify `moe_infinity/entrypoints/openai/api_server_v2.py`: add opt-in CLI/config, export graph metrics, invalidate on reload, and close old/current runners during hot replacement and application shutdown under a documented lock order.
- Create `benchmarks/serving/decode_cuda_graph_fixture.py`: define a persistent resident/native-paged benchmark fixture requiring no model checkpoint or offload directory.
- Modify `tests/python/serving/test_model_runner.py`: prepared-forward parity and serving-side-effect tests.
- Modify `tests/python/serving/test_flashinfer_model_runner.py`: stable metadata/context tests and FlashInfer safety rejection.
- Create `tests/python/serving/test_paged_kv_storage.py`: allocator/tensor ownership, per-layer mapping, and scratch lifecycle tests.
- Create `tests/python/serving/test_paged_attention_registry.py`: exact Qwen3 registration, DeepSeek V2/V3 MLA rejection, generated Qwen3 layer-bound subclasses, unique layer indices, and missing-proof rejection.
- Create `tests/python/ops/test_paged_kv_write.py`: allocation-free current-token K/V write layout and CUDA graph replay persistence tests.
- Modify `tests/python/integration/test_flashinfer_model_attention.py`: use installed upstream DeepSeek paged variants when available and prove each receives `mla_layout_unsupported` while retaining eager attention coverage.
- Modify `tests/python/serving/test_kv_cache.py`: authoritative allocator delegation tests.
- Modify `tests/python/integration/test_paged_attention_backend.py`: backend storage identity, exact device equality, and per-layer native tensor tests.
- Create `tests/python/serving/test_model_offload_capability.py`: explicit rejection reason tests for every offload/runtime hazard.
- Rewrite `tests/python/serving/test_cuda_graph.py`: deterministic CPU gate/bucket/lifecycle tests plus CUDA replay equivalence.
- Modify `tests/python/serving/test_engine.py`: active-serving graph selection, slicing, eager fallback, sampler boundary, and mixed-batch tests.
- Modify `tests/python/serving/test_hot_reload.py`: invalidation-before-reload and lock-order tests.
- Modify `tests/python/serving/test_memory_manager.py`: graph-memory accounting tests.
- Modify `tests/python/serving/test_api_routes.py`: opt-in configuration and Prometheus metric tests.
- Create `benchmarks/serving/decode_cuda_graph.py`: paired eager/replay launch-overhead benchmark with validated `fixture` and `model` modes, machine-readable output, and no pass/fail speedup assertion.
- Modify `docs/serving.md`: operator enablement, safety limitations, metrics, rollout, and rollback.
- Modify `docs/benchmarking.md`: reproducible decode graph benchmark command and interpretation.

### Task 1: Unify block allocation and native attention KV storage

**Files:**
- Create: `moe_infinity/runtime/paged_kv_storage.py`
- Create: `moe_infinity/kernel/paged_kv_write.py`
- Create: `moe_infinity/models/paged_attention_registry.py`
- Modify: `moe_infinity/serving/kv_cache.py:48-338`
- Modify: `moe_infinity/runtime/attention_backend.py:78-255`
- Modify: `moe_infinity/models/qwen3_paged_attention.py:24-188`
- Modify: `moe_infinity/entrypoints/big_modeling.py:401-425`
- Test: `tests/python/serving/test_paged_kv_storage.py`
- Test: `tests/python/serving/test_paged_attention_registry.py`
- Test: `tests/python/ops/test_paged_kv_write.py`
- Test: `tests/python/integration/test_flashinfer_model_attention.py`
- Test: `tests/python/serving/test_kv_cache.py`
- Test: `tests/python/integration/test_paged_attention_backend.py`

- [x] **Step 1: Write failing identity, per-layer, and scratch ownership tests**

```python
def test_scheduler_backend_and_scratch_share_one_storage_owner() -> None:
    storage = _make_storage(num_layers=2, num_blocks=16, block_size=4)
    cache = PagedKVCache(
        num_blocks=storage.spec.num_blocks,
        block_size=storage.spec.block_size,
        num_layers=storage.spec.num_layers,
        num_heads=storage.spec.num_kv_heads,
        head_dim=storage.spec.head_dim,
        dtype=storage.spec.dtype,
        device=storage.spec.device,
        storage=storage,
    )
    backend = PagedAttentionBackend(storage=storage, use_flashinfer=False)
    scratch = storage.reserve_graph_scratch_blocks(2)

    assert cache.storage is backend.storage is storage
    assert cache.block_allocator is storage.block_allocator
    assert backend.storage.owner_id == cache.storage.owner_id
    assert storage.block_allocator.num_free_blocks == 14
    assert all(0 <= block_id < storage.num_blocks for block_id in scratch)


def test_paged_kv_cache_legacy_constructor_remains_supported_but_unbound() -> None:
    cache = PagedKVCache(
        num_blocks=8,
        block_size=4,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache.allocate_sequence(seq_id=7, num_tokens=1)
    assert cache.get_block_table(7)
    assert cache.storage is None
    assert cache.has_bound_storage is False


def test_native_attention_reads_the_exact_page_reserved_by_allocator() -> None:
    storage = _make_storage(num_layers=2, num_blocks=8, block_size=4)
    block_id = storage.reserve_graph_scratch_blocks(1)[0]
    slot = block_id * storage.block_size
    key = torch.full((1, storage.num_kv_heads, storage.head_dim), 3.0)
    value = torch.full_like(key, 5.0)

    storage.write_kv(layer_idx=1, key=key, value=value, slot_mapping=torch.tensor([slot]))

    assert torch.all(storage.key_cache[1, block_id, :, :, 0, :] == 3.0)
    assert torch.all(storage.value_cache[1, block_id, :, :, 0] == 5.0)


def test_backend_rejects_block_tables_from_another_owner() -> None:
    first = _make_storage(num_layers=2, num_blocks=8, block_size=4)
    second = _make_storage(num_layers=2, num_blocks=8, block_size=4)
    backend = PagedAttentionBackend(storage=first, use_flashinfer=False)
    metadata = _metadata(owner_id=second.owner_id)
    query = torch.zeros(1, first.spec.num_kv_heads, first.spec.head_dim)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    with pytest.raises(ValueError, match="KV storage owner mismatch"):
        backend.forward(
            query=query,
            key=key,
            value=value,
            kv_cache=None,
            attention_metadata=metadata,
            layer_idx=0,
        )


def test_registry_supports_only_exact_qwen3_paged_attention() -> None:
    assert tuple(SUPPORTED_PAGED_CLASS_SPECS) == (
        ("moe_infinity.models.qwen3_paged_attention", "Qwen3PagedAttention"),
    )


def test_registry_binds_unique_layer_idx_and_storage_slice() -> None:
    model, storage, backend = _make_two_layer_qwen3_paged_model()
    registry = PagedAttentionLayerRegistry.register(model, backend, storage)
    assert [binding.layer_idx for binding in registry.bindings] == [0, 1]
    assert registry.bindings[0].backend.layer_idx == 0
    assert registry.bindings[1].backend.layer_idx == 1
    assert registry.bindings[0].storage_owner_id == storage.owner_id
    assert registry.bindings[0].bound_class is not registry.bindings[1].bound_class


def test_qwen3_routes_to_registered_layer_bound_backend() -> None:
    class_fqn = "moe_infinity.models.qwen3_paged_attention.Qwen3PagedAttention"
    module = _instantiate_supported_class(class_fqn, layer_idx=1)
    registry, recorder = _register_single_module(module, num_layers=2)
    _run_minimal_paged_forward(module)
    assert recorder.events[0][:2] == ("write", 1)
    assert recorder.events[1] == ("attention", 1)
    assert registry.bindings[0].class_fqn == class_fqn


@pytest.mark.parametrize("family", ["deepseek_v2", "deepseek_v3"])
def test_deepseek_mla_is_recognized_but_never_registered(family: str) -> None:
    module = _instantiate_deepseek_paged_attention_or_skip(family, layer_idx=0)
    result = PagedAttentionLayerRegistry.inspect_module(module)
    assert result.binding is None
    assert result.reason == "mla_layout_unsupported"
```

- [x] **Step 2: Run ownership tests and verify RED**

```bash
pytest -q tests/python/serving/test_paged_kv_storage.py \
  tests/python/serving/test_paged_attention_registry.py \
  tests/python/ops/test_paged_kv_write.py \
  tests/python/integration/test_flashinfer_model_attention.py \
  tests/python/serving/test_kv_cache.py \
  tests/python/integration/test_paged_attention_backend.py \
  -k 'storage_owner or legacy_constructor or exact_page or another_owner or registry or qwen3 or deepseek_mla'
```

Expected: FAIL because `PagedKVStorage`, exact class specs, per-instance layer bindings, shared ownership, layer-indexed tensors, and metadata owner IDs do not exist.

- [x] **Step 3: Implement the single authoritative storage object**

Create these interfaces in `paged_kv_storage.py`:

```python
@dataclass(frozen=True)
class PagedKVStorageSpec:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device


class PagedKVStorage:
    def __init__(self, spec: PagedKVStorageSpec) -> None:
        self.spec = spec
        self.owner_id = uuid.uuid4().hex
        self.block_allocator = BlockAllocator(
            num_blocks=spec.num_blocks,
            block_size=spec.block_size,
            device=spec.device,
        )
        x = 8
        self.key_cache = torch.zeros(
            spec.num_layers, spec.num_blocks, spec.num_kv_heads,
            spec.head_dim // x, spec.block_size, x,
            dtype=spec.dtype, device=spec.device,
        )
        self.value_cache = torch.zeros(
            spec.num_layers, spec.num_blocks, spec.num_kv_heads,
            spec.head_dim, spec.block_size,
            dtype=spec.dtype, device=spec.device,
        )
        self._graph_scratch_blocks: set[int] = set()

    def reserve_graph_scratch_blocks(self, count: int) -> list[int]:
        block_ids = self.block_allocator.allocate(count)
        self._graph_scratch_blocks.update(block_ids)
        return block_ids

    def release_graph_scratch_blocks(self, block_ids: list[int]) -> None:
        unknown = set(block_ids) - self._graph_scratch_blocks
        if unknown:
            raise ValueError(f"graph scratch blocks are not reserved: {sorted(unknown)}")
        self._graph_scratch_blocks.difference_update(block_ids)
        self.block_allocator.free(block_ids)

    @property
    def num_graph_scratch_blocks(self) -> int:
        return len(self._graph_scratch_blocks)
```

Define the exact class registry:

```python
SUPPORTED_PAGED_CLASS_SPECS = {
    ("moe_infinity.models.qwen3_paged_attention", "Qwen3PagedAttention"):
        PagedClassSpec("moe_infinity.models.qwen3_paged_attention", "Qwen3PagedAttention"),
}
```

Resolve optional DeepSeek V2/V3 paged types with `importlib` into a separate `UNSUPPORTED_MLA_CLASS_TYPES` tuple; absence means unavailable and no string-name fallback is allowed. If `type(module)` is one of those exact types, return `mla_layout_unsupported` before reading or synthesizing any `head_dim`. Do not create a DeepSeek layout adapter, storage spec, layer binding, or write proof. The sole `PagedClassSpec` uses Qwen3's ordinary NHD GQA resolver and validates that key and value share `storage.spec.head_dim` without allocating or converting them. For each actual Qwen3 module instance, require `type(module)` to be the resolved supported base type and `layer_idx` to be a unique integer in `[0, storage.spec.num_layers)`. Create a unique generated subclass per instance so class-level paged context holds a stable `LayerBoundPagedBackend(backend, layer_idx, storage.owner_id)`; retain `base_class`, `bound_class`, module, and layer index in `PagedLayerBinding`. `ModelRunner` installs/clears metadata through registry bindings, not a deduplicated class-name set.

Move `BlockAllocator` to authoritative storage. Keep eager `write_kv()` allowed to normalize inputs before the graph boundary. Put scratch reserve/release methods on `PagedKVStorage`, not on an unrelated cache wrapper. Task 3 supplies the allocation-free current-token writer used by every layer-bound backend.

Add `storage: PagedKVStorage | None = None` as the final defaulted `PagedKVCache` dataclass field so every existing positional/keyword constructor remains valid. When `storage is None`, preserve the current legacy behavior exactly: resolve `device`, create the cache's own `BlockAllocator` and `_kv_cache`, and keep existing tests/callers working. When storage is supplied, validate all cache dimensions/dtype/device against it, use `storage.block_allocator`, and route swap pages through `storage.key_cache`/`storage.value_cache`; expose `has_bound_storage` and make decode-graph capability return `kv_storage_mismatch` unless it is true and object identity matches the backend. Add tests that instantiate the old signature without `storage`, exercise allocation/write/swap behavior, and prove it is graph-ineligible, plus tests for the bound path. Make `PagedAttentionBackend` require authoritative storage for the graph-capable path and use `storage.key_cache[layer_idx]` and `storage.value_cache[layer_idx]`. Add `kv_storage_owner_id` to runtime attention metadata and require equality before reads/writes. `Qwen3PagedAttention.forward()` passes `self.layer_idx` to its bound proxy, which rejects a supplied index differing from registration. DeepSeek V2/V3 never reaches this proxy.

Canonicalize devices to explicit indexed `torch.device` values before comparison (`cuda` becomes the current indexed CUDA device). Require `storage.spec.device == backend.device == ModelRunner.device`; every allocated key/value cache tensor must also report that device. Mismatch fails capability as `kv_storage_mismatch` before scratch reservation or graph-buffer allocation.

In `big_modeling.py`, construct one `PagedKVStorage` after sizing blocks and pass it to the native paged backend. In Task 5, `ContinuousBatchingEngine` adopts that exact backend storage for its scheduler-facing `PagedKVCache`; it never allocates a second serving cache. Existing native-engine `KVCacheManager`/transfer-scheduler paths remain explicitly ineligible until they also consume this allocator, so they cannot accidentally qualify based on numeric block-ID overlap.

- [x] **Step 4: Run ownership suites and verify GREEN**

```bash
pytest -q tests/python/serving/test_paged_kv_storage.py \
  tests/python/serving/test_paged_attention_registry.py \
  tests/python/ops/test_paged_kv_write.py \
  tests/python/integration/test_flashinfer_model_attention.py \
  tests/python/serving/test_kv_cache.py \
  tests/python/integration/test_paged_attention_backend.py \
  tests/python/serving/test_flashinfer_model_runner.py
```

Expected: PASS; tests assert legacy constructor compatibility, bound allocator identity, owner ID, exact device, block count, tensor pointers, Qwen3-only registration, MLA rejection, and per-layer writes.

- [x] **Step 5: Commit authoritative KV ownership**

```bash
git add moe_infinity/runtime/paged_kv_storage.py \
  moe_infinity/kernel/paged_kv_write.py \
  moe_infinity/models/paged_attention_registry.py \
  moe_infinity/serving/kv_cache.py \
  moe_infinity/runtime/attention_backend.py \
  moe_infinity/models/qwen3_paged_attention.py \
  moe_infinity/entrypoints/big_modeling.py \
  tests/python/serving/test_paged_kv_storage.py \
  tests/python/serving/test_paged_attention_registry.py \
  tests/python/ops/test_paged_kv_write.py \
  tests/python/integration/test_flashinfer_model_attention.py \
  tests/python/serving/test_kv_cache.py \
  tests/python/integration/test_paged_attention_backend.py
git commit -m "refactor(serving): unify paged KV allocation and storage"
```

### Task 2: Define an explicit deny-by-default graph capability

**Files:**
- Modify: `moe_infinity/runtime/attention_types.py`
- Modify: `moe_infinity/runtime/model_offload.py:523-592,1689-1732,2081-2192`
- Modify: `moe_infinity/entrypoints/big_modeling.py`
- Modify: `moe_infinity/runtime/attention_backend.py`
- Modify: `moe_infinity/serving/model_runner.py`
- Test: `tests/python/serving/test_model_offload_capability.py`
- Test: `tests/python/serving/test_flashinfer_model_runner.py`
- Test: `tests/python/serving/test_model_runner.py`
- Test: `tests/python/serving/test_paged_attention_registry.py`

- [x] **Step 1: Write one failing test for every rejection reason**

```python
@pytest.mark.parametrize(
    ("runtime_state", "reason"),
    [
        ({"active_model_hooks": True}, "active_model_hooks"),
        ({"archer_callbacks": True}, "archer_callbacks"),
        ({"transfer_scheduler_active": True}, "transfer_scheduler"),
        ({"expert_dispatcher_active": True}, "expert_dispatcher"),
        ({"kv_offload_active": True}, "kv_offload"),
        ({"dynamic_allocations": True}, "dynamic_allocations"),
    ],
)
def test_runtime_capability_rejects_each_hazard(runtime_state, reason: str) -> None:
    provider = _ExplicitCapabilityProvider(**runtime_state)
    capability = provider.decode_graph_capability()
    assert capability == DecodeGraphCapability(False, reason)


def test_missing_capability_provider_is_rejected() -> None:
    runner = ModelRunner(_native_paged_model(), engine=object(), device=torch.device("cuda"))
    assert runner.decode_graph_capability().reason == "missing_capability"


def test_non_paged_model_is_rejected_even_when_resident() -> None:
    provider = _ExplicitCapabilityProvider()
    runner = ModelRunner(_non_paged_model(), provider, torch.device("cuda"))
    assert runner.decode_graph_capability().reason == "native_paged_required"


def test_legacy_unbound_paged_kv_cache_is_graph_ineligible() -> None:
    cache = _make_legacy_paged_kv_cache(storage=None)
    runner = _native_paged_runner(kv_cache=cache, backend_storage=_make_storage())
    assert runner.decode_graph_capability().reason == "kv_storage_mismatch"


def test_flashinfer_plan_path_is_rejected() -> None:
    backend = _native_backend(use_flashinfer=True)
    assert backend.decode_graph_capability().reason == "flashinfer_plan_path"


def test_storage_identity_mismatch_is_rejected() -> None:
    runner = _native_paged_runner(runtime_storage=_make_storage(), backend_storage=_make_storage())
    assert runner.decode_graph_capability().reason == "kv_storage_mismatch"


def test_storage_and_model_runner_device_must_match_exactly() -> None:
    storage = _make_storage(device=torch.device("cuda:0"))
    runner = _native_paged_runner(
        runtime_storage=storage,
        backend_storage=storage,
        runner_device=torch.device("cuda:1"),
    )
    assert runner.decode_graph_capability().reason == "kv_storage_mismatch"


@pytest.mark.parametrize("family", ["deepseek_v2", "deepseek_v3"])
def test_deepseek_mla_layout_is_explicitly_rejected(family: str) -> None:
    runner = _runner_with_exact_deepseek_paged_attention_or_skip(family)
    capability = runner.decode_graph_capability()
    assert capability.safe is False
    assert capability.reason == "mla_layout_unsupported"
    assert capability.layer_write_proofs == ()


@pytest.mark.parametrize(
    ("registry", "reason"),
    [
        (_registry_with_unknown_paged_class(), "paged_class_unregistered"),
        (_registry_with_duplicate_or_missing_layer_idx(), "layer_idx_invalid"),
        (_registry_without_write_proof(), "layer_write_unproven"),
    ],
)
def test_capability_rejects_missing_per_layer_write_proof(registry, reason: str) -> None:
    runner = _runner_with_registry(registry)
    assert runner.decode_graph_capability().reason == reason
```

- [x] **Step 2: Run capability tests and verify RED**

```bash
pytest -q tests/python/serving/test_model_offload_capability.py \
  tests/python/serving/test_model_runner.py \
  tests/python/serving/test_flashinfer_model_runner.py \
  tests/python/serving/test_paged_attention_registry.py \
  -k 'capability or non_paged or storage_identity or device_must_match or mla_layout or write_proof'
```

Expected: FAIL because typed capability evidence and fixed rejection reasons do not exist.

- [x] **Step 3: Implement typed capability composition**

Add:

```python
DECODE_GRAPH_REASONS = (
    "eligible", "missing_capability", "active_model_hooks",
    "archer_callbacks", "transfer_scheduler", "expert_dispatcher",
    "kv_offload", "flashinfer_plan_path", "dynamic_allocations",
    "native_paged_required", "mla_layout_unsupported", "kv_storage_mismatch",
    "paged_class_unregistered", "layer_idx_invalid", "layer_write_unproven",
)


@dataclass(frozen=True)
class PagedLayerWriteProof:
    class_fqn: str
    layer_idx: int
    storage_owner_id: str
    writer: str
    writes_before_attention: bool
    allocation_free: bool


@dataclass(frozen=True)
class DecodeGraphCapability:
    safe: bool
    reason: str
    storage_owner_id: str | None = None
    layer_write_proofs: tuple[PagedLayerWriteProof, ...] = ()


@runtime_checkable
class DecodeGraphCapabilityProvider(Protocol):
    def decode_graph_capability(self) -> DecodeGraphCapability: ...
```

`ModelRunner` receives a separate `decode_graph_capability_provider: DecodeGraphCapabilityProvider | None` constructor argument and combines that evidence with the backend's evidence; it must not inspect arbitrary runtime attributes or assume that the `engine` object is the full owner. `ModelOffload` reports its hook handles, Archer callbacks, dispatcher, KV offload, and dynamic transfers. The top-level `MoE`/native-runtime owner in `big_modeling.py` composes that with `_native_transfer_scheduler`, KV/expert coordinators, and native storage identity, returning the first active reason in the exact order above. Update explicit booleans when mechanisms are installed or removed; do not infer safety from attribute presence. `initialize_with_model()` and CLI initialization pass the top-level provider into `ContinuousBatchingEngine`, which passes it to `ModelRunner`; omission yields `missing_capability`.

`PagedAttentionBackend.decode_graph_capability()` returns `flashinfer_plan_path` when FlashInfer is active. Before ordinary paged checks, exact DeepSeek V2/V3 paged types return `mla_layout_unsupported`; capability composition must preserve that reason instead of attempting to derive one common `head_dim`, registering a layer, or falling through to `paged_class_unregistered`. Otherwise, the registry must produce exactly one `PagedLayerWriteProof` for every Qwen3 paged module and every expected layer index; each proof must name only `moe_infinity.models.qwen3_paged_attention.Qwen3PagedAttention`, match `storage.owner_id`, identify `moe_infinity.kernel.paged_kv_write.paged_kv_write_`, and set both booleans true. Missing/unregistered classes, duplicate/out-of-range/missing layer indices, absent proofs, or a backend that reads before writing are denied. `ModelRunner` requires proof coverage equal to the registered binding set, a bound (non-`None`) `PagedKVCache.storage`, object identity with backend storage, and exact equality of canonical `PagedKVStorage.spec.device`, `ModelRunner.device`, backend device, and cache tensor devices.

- [x] **Step 4: Run capability tests and verify GREEN**

```bash
pytest -q tests/python/serving/test_model_offload_capability.py \
  tests/python/serving/test_model_runner.py \
  tests/python/serving/test_flashinfer_model_runner.py \
  tests/python/serving/test_paged_attention_registry.py
```

Expected: PASS; each unsafe mechanism has a distinct deterministic reason and no non-paged model is eligible.

- [x] **Step 5: Commit explicit capability evidence**

```bash
git add moe_infinity/runtime/attention_types.py \
  moe_infinity/runtime/model_offload.py \
  moe_infinity/entrypoints/big_modeling.py \
  moe_infinity/runtime/attention_backend.py \
  moe_infinity/serving/model_runner.py \
  tests/python/serving/test_model_offload_capability.py \
  tests/python/serving/test_model_runner.py \
  tests/python/serving/test_flashinfer_model_runner.py \
  tests/python/serving/test_paged_attention_registry.py
git commit -m "feat(serving): require explicit decode graph capability"
```

### Task 3: Write current-token KV per layer before prepared decode attention

**Files:**
- Modify: `moe_infinity/serving/model_runner.py:44-315`
- Modify: `moe_infinity/runtime/attention_backend.py:255-518`
- Modify: `moe_infinity/models/paged_attention_registry.py`
- Modify: `moe_infinity/kernel/paged_kv_write.py`
- Test: `tests/python/serving/test_model_runner.py`
- Test: `tests/python/serving/test_flashinfer_model_runner.py`
- Test: `tests/python/serving/test_paged_attention_registry.py`
- Test: `tests/python/ops/test_paged_kv_write.py`

- [x] **Step 1: Write failing fixed-pointer and semantic parity tests**

```python
def test_prepared_native_paged_decode_preserves_side_effects_and_pointers() -> None:
    runner, batch, storage = _make_graph_safe_native_paged_runner()
    prepared = runner.allocate_decode_buffers(batch_bucket=2, context_bucket=16)
    pointers = prepared.data_ptrs()
    runner.copy_decode_batch(batch, prepared, scratch_block_ids=[])
    runner.prepare_batch_side_effects(batch)
    logits = runner.forward_prepared_decode(prepared)

    assert logits.shape[0] == 2
    assert prepared.data_ptrs() == pointers
    assert prepared.attention_metadata.kv_storage_owner_id == storage.owner_id
    assert prepared.attention_metadata.seq_lens.tolist() == [9, 4]


def test_copy_decode_batch_rejects_block_id_outside_authoritative_storage() -> None:
    runner, batch, storage = _make_graph_safe_native_paged_runner()
    batch.block_tables[0] = [storage.num_blocks]
    prepared = runner.allocate_decode_buffers(batch_bucket=2, context_bucket=16)
    with pytest.raises(ValueError, match="block id"):
        runner.copy_decode_batch(batch, prepared, scratch_block_ids=[])


def test_allocate_decode_buffers_requires_exact_runner_storage_device() -> None:
    runner, _, _ = _make_graph_safe_native_paged_runner(
        storage_device=torch.device("cuda:0"),
        runner_device=torch.device("cuda:1"),
    )
    with pytest.raises(ValueError, match="device"):
        runner.allocate_decode_buffers(batch_bucket=2, context_bucket=16)


def test_every_prepared_buffer_uses_exact_storage_device() -> None:
    runner, batch, storage = _make_graph_safe_native_paged_runner()
    prepared = runner.allocate_decode_buffers(batch_bucket=2, context_bucket=16)
    tensors = prepared.tensor_values()
    assert tensors
    assert all(tensor.device == storage.spec.device for tensor in tensors)


def test_each_layer_writes_current_token_before_decode_attention() -> None:
    backend, storage, metadata = _recording_two_layer_backend()
    for layer_idx in (0, 1):
        bound = LayerBoundPagedBackend(backend, layer_idx, storage.owner_id)
        _ = bound.forward(
            query=_query(layer_idx),
            key=_key(layer_idx),
            value=_value(layer_idx),
            attention_metadata=metadata,
            graph_mode=True,
        )
    assert backend.events == [
        ("write", 0, metadata.slot_mapping.data_ptr()),
        ("attention", 0),
        ("write", 1, metadata.slot_mapping.data_ptr()),
        ("attention", 1),
    ]


@requires_cuda
def test_graph_safe_kv_write_persists_current_token_per_layer() -> None:
    storage = _make_cuda_storage(num_layers=2, num_blocks=4, block_size=4)
    slots = torch.tensor([1, 6], dtype=torch.int64, device="cuda")
    key = torch.arange(2 * 2 * 8, device="cuda").reshape(2, 2, 8)
    value = key + 100
    paged_kv_write_(storage, layer_idx=1, key=key, value=value, slot_mapping=slots)
    _assert_slots_equal(storage, layer_idx=1, slots=slots, key=key, value=value)
    assert torch.count_nonzero(storage.value_cache[0]).item() == 0


@requires_cuda
def test_second_decode_token_eager_and_replay_observe_first_token_kv() -> None:
    eager, graph, storage = _make_two_step_native_paged_fixture()
    first = _decode_batch(token=17, context_len=3, slot=3)
    second = _decode_batch(token=18, context_len=4, slot=4)
    eager_first = eager.execute(first)
    replay_first = graph.try_execute(first)
    assert replay_first is not None
    _assert_all_registered_layers_persisted(storage, first.attention_slot_mapping())
    eager_second = eager.execute(second)
    replay_second = graph.try_execute(second)
    assert replay_second is not None
    torch.testing.assert_close(replay_first, eager_first, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(replay_second, eager_second, rtol=1e-4, atol=1e-4)
```

- [x] **Step 2: Run prepared-forward tests and verify RED**

```bash
pytest -q tests/python/serving/test_model_runner.py \
  tests/python/serving/test_flashinfer_model_runner.py \
  tests/python/serving/test_paged_attention_registry.py \
  tests/python/ops/test_paged_kv_write.py \
  -k 'prepared_native_paged or authoritative_storage or writes_current or persists or second_decode'
```

Expected: FAIL because prepared buffers do not carry authoritative storage identity, layer-bound backends do not exist, and decode does not write current-token K/V before attention.

- [x] **Step 3: Implement prepared buffers and allocation-free graph mode**

Define this exact buffer boundary:

```python
@dataclass
class PreparedDecodeBuffers:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    active_rows: torch.Tensor
    attention_metadata: RuntimeAttentionMetadata
    batch_bucket: int
    context_bucket: int
    real_batch_size: int = 0

    def data_ptrs(self) -> tuple[int, ...]:
        return tuple(
            tensor.data_ptr()
            for tensor in (
                self.input_ids,
                self.position_ids,
                self.attention_mask,
                self.active_rows,
                self.attention_metadata.block_tables,
                self.attention_metadata.seq_lens,
                self.attention_metadata.slot_mapping,
            )
        )

    def tensor_values(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_ids,
            self.position_ids,
            self.attention_mask,
            self.active_rows,
            self.attention_metadata.block_tables,
            self.attention_metadata.seq_lens,
            self.attention_metadata.slot_mapping,
        )


def allocate_decode_buffers(
    self, *, batch_bucket: int, context_bucket: int
) -> PreparedDecodeBuffers:
    storage = self._require_paged_kv_storage()
    max_blocks = math.ceil(context_bucket / storage.spec.block_size)
    metadata = RuntimeAttentionMetadata(
        block_tables=torch.zeros(
            (batch_bucket, max_blocks), dtype=torch.int32, device=storage.spec.device
        ),
        seq_lens=torch.ones(batch_bucket, dtype=torch.int32, device=storage.spec.device),
        max_seq_len=context_bucket,
        num_prefill_tokens=0,
        num_decode_tokens=batch_bucket,
        slot_mapping=torch.zeros(batch_bucket, dtype=torch.int64, device=storage.spec.device),
        is_prefill=False,
        kv_storage_owner_id=storage.owner_id,
    )
    return PreparedDecodeBuffers(
        input_ids=torch.zeros((batch_bucket, 1), dtype=torch.long, device=storage.spec.device),
        position_ids=torch.zeros((batch_bucket, 1), dtype=torch.long, device=storage.spec.device),
        attention_mask=torch.ones((batch_bucket, 1), dtype=torch.long, device=storage.spec.device),
        active_rows=torch.zeros(batch_bucket, dtype=torch.bool, device=storage.spec.device),
        attention_metadata=metadata,
        batch_bucket=batch_bucket,
        context_bucket=context_bucket,
    )


def copy_decode_batch(
    self,
    batch: BatchMetadata,
    buffers: PreparedDecodeBuffers,
    scratch_block_ids: list[int],
) -> None:
    """Validate against authoritative storage and copy only in place."""
```

`allocate_decode_buffers()` first canonicalizes and compares `self.device` and `storage.spec.device`; unequal devices raise before allocation. Every tensor in `PreparedDecodeBuffers`, including all `RuntimeAttentionMetadata` tensors and the later static output tensor, is allocated on exactly `storage.spec.device`. `copy_decode_batch()` rechecks those tensor devices, pure one-token decode, every block ID range, owner identity, context bucket, and scratch ownership, then uses only `zero_()`, `fill_()`, indexed assignment, and `copy_()` on existing tensors. It requires one unique reserved scratch ID for each padded row and sets `real_batch_size` after successful copies.

Implement `paged_kv_write_()` as a Triton kernel launch whose grid is fixed by captured token/head dimensions. Each program loads one `slot_mapping[token]`, computes `block_id = slot // block_size` and `token_offset = slot % block_size`, and writes the supplied current-token key into packed `key_cache[layer_idx, block_id, head, dim // 8, token_offset, dim % 8]` and value into `value_cache[layer_idx, block_id, head, dim, token_offset]`. The Python launcher validates shape/device/dtype before capture and allocates no tensor, list, workspace, or converted view during the captured call.

Define the layer-bound call order exactly:

```python
def forward_layer(
    self,
    *,
    layer_idx: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_metadata: RuntimeAttentionMetadata,
    scale: float | None,
    graph_mode: bool,
) -> torch.Tensor:
    if attention_metadata.is_prefill:
        self.write_kv_eager(layer_idx, key, value, attention_metadata.slot_mapping)
        return self._prefill_forward(query, key, value, attention_metadata, scale)
    paged_kv_write_(
        self.storage,
        layer_idx=layer_idx,
        key=key,
        value=value,
        slot_mapping=attention_metadata.slot_mapping,
    )
    return self._decode_forward_layer(
        layer_idx, query, attention_metadata, scale=scale
    )
```

`LayerBoundPagedBackend.forward()` supplies its immutable registered `layer_idx`; each eligible Qwen3 generated bound subclass receives a different proxy. DeepSeek V2/V3 is rejected before binding as `mla_layout_unsupported`. Extract eager/paged model execution into `_forward_with_optional_paged_context(...)`; `forward_prepared_decode()` installs fixed metadata through all Qwen3 registry bindings. In graph mode, no `.to()`, `.item()`, `torch.tensor`, concatenation, Python data-dependent loop, FlashInfer `plan()`, or output allocation occurs. Sampling remains outside.

- [x] **Step 4: Run prepared/native suites and verify GREEN**

```bash
pytest -q tests/python/serving/test_model_runner.py \
  tests/python/integration/test_paged_attention_backend.py \
  tests/python/serving/test_flashinfer_model_runner.py \
  tests/python/serving/test_paged_attention_registry.py \
  tests/python/ops/test_paged_kv_write.py
```

Expected: PASS with eager semantics unchanged and graph mode native-paged only.

- [x] **Step 5: Commit the prepared boundary**

```bash
git add moe_infinity/serving/model_runner.py \
  moe_infinity/runtime/attention_backend.py \
  moe_infinity/models/paged_attention_registry.py \
  moe_infinity/kernel/paged_kv_write.py \
  tests/python/serving/test_model_runner.py \
  tests/python/integration/test_paged_attention_backend.py \
  tests/python/serving/test_flashinfer_model_runner.py \
  tests/python/serving/test_paged_attention_registry.py \
  tests/python/ops/test_paged_kv_write.py
git commit -m "refactor(serving): prepare native paged decode for replay"
```

### Task 4: Replace `CudaGraphRunner` with a safe lazy lifecycle

**Files:**
- Rewrite: `moe_infinity/serving/cuda_graph.py`
- Rewrite: `tests/python/serving/test_cuda_graph.py`

- [x] **Step 1: Write deterministic CPU tests for config, gates, buckets, fallback reasons, and lifecycle**

Use injected CUDA operations and an explicit `_eligible_capability_for_registry(registry)` provider containing complete per-layer write proofs so gate logic runs without a GPU and never relies on heuristic model inspection:

```python
@dataclass(frozen=True)
class _FakeCudaOps:
    available: bool = True

    def memory_allocated(self, device: torch.device) -> int:
        return 100


def test_select_key_uses_next_batch_and_context_bucket() -> None:
    runner = _make_cpu_gate_runner(
        batch_buckets=(1, 2, 4), context_buckets=(16, 32)
    )
    decision = runner.check_eligibility(
        _make_decode_batch(batch_size=3, context_lengths=[3, 17, 7])
    )
    assert decision.eligible
    assert decision.key == GraphKey(batch_size=4, context_size=32)


@pytest.mark.parametrize(
    ("batch", "reason"),
    [
        (_make_prefill_batch(), "not_decode"),
        (_make_decode_batch(batch_size=5), "no_batch_bucket"),
        (_make_decode_batch(batch_size=1, context_lengths=[64]), "no_context_bucket"),
    ],
)
def test_gate_returns_bounded_fallback_reason(batch, reason: str) -> None:
    runner = _make_cpu_gate_runner(
        batch_buckets=(1, 2, 4), context_buckets=(16, 32)
    )
    assert runner.check_eligibility(batch).reason == reason


def test_disabled_by_default_and_environment_kill_switch(monkeypatch) -> None:
    runner = _make_cpu_gate_runner(enabled=False)
    assert runner.check_eligibility(_make_decode_batch()).reason == "disabled"
    monkeypatch.setenv("MOE_DISABLE_CUDA_GRAPHS", "1")
    runner = _make_cpu_gate_runner(enabled=True)
    assert runner.check_eligibility(_make_decode_batch()).reason == "kill_switch"


def test_capture_failure_quarantines_key_and_returns_eager_signal(monkeypatch) -> None:
    runner = _make_cpu_gate_runner(enabled=True)
    monkeypatch.setattr(runner, "_capture_key", Mock(side_effect=RuntimeError("capture")))
    result = runner.try_execute(_make_decode_batch())
    assert result is None
    assert runner.stats()["fallback_reasons"]["capture_failed"] == 1
    assert runner.check_eligibility(_make_decode_batch()).reason == "quarantined"


def test_scratch_reservation_failure_falls_back_as_insufficient_memory(monkeypatch) -> None:
    runner = _make_cpu_gate_runner(enabled=True)
    monkeypatch.setattr(
        runner.storage,
        "reserve_graph_scratch_blocks",
        Mock(side_effect=RuntimeError("BlockAllocator exhausted")),
    )
    assert runner.try_execute(_make_decode_batch()) is None
    assert runner.stats()["fallback_reasons"]["insufficient_memory"] == 1


def test_graph_gate_rejects_storage_runner_device_mismatch() -> None:
    runner = _make_cpu_gate_runner(
        storage_device=torch.device("cuda:0"),
        runner_device=torch.device("cuda:1"),
    )
    decision = runner.check_eligibility(_make_decode_batch())
    assert decision.reason == "kv_storage_mismatch"
    assert runner.storage.num_graph_scratch_blocks == 0


def test_graph_gate_rejects_any_buffer_on_another_device() -> None:
    runner = _make_cpu_gate_runner(device=torch.device("cuda:0"))
    state = _fake_captured_state_with_one_buffer_on(torch.device("cuda:1"))
    assert runner._validate_state_devices(state) is False
    assert runner.check_state_eligibility(state).reason == "kv_storage_mismatch"


def test_invalidate_waits_for_replay_lock_and_advances_generation() -> None:
    runner = _make_cpu_gate_runner(enabled=True)
    generation = runner.generation
    runner.invalidate("module_reload")
    assert runner.generation == generation + 1
    assert runner.stats()["graphs"] == 0
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q tests/python/serving/test_cuda_graph.py \
  -k 'select_key or gate or disabled or capture_failure or invalidate'
```

Expected: FAIL because the new decision, key, injected-ops, stats, and lazy execution APIs do not exist.

- [x] **Step 3: Implement exact public types and bounded reasons**

Use these public interfaces:

```python
@dataclass(frozen=True, order=True)
class GraphKey:
    batch_size: int
    context_size: int


@dataclass(frozen=True)
class GraphDecision:
    eligible: bool
    reason: str
    key: GraphKey | None = None


@dataclass
class GraphExecutionStats:
    captures: int = 0
    replays: int = 0
    capture_failures: int = 0
    graph_pool_bytes: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)


class _GraphMemoryUnavailable(RuntimeError):
    pass


@dataclass
class _CapturedGraphState:
    graph: torch.cuda.CUDAGraph
    buffers: PreparedDecodeBuffers
    output_logits: torch.Tensor
    generation: int
```

The only emitted fallback reasons are:

```python
FALLBACK_REASONS = (
    "disabled",
    "kill_switch",
    "non_cuda",
    "not_decode",
    "no_batch_bucket",
    "no_context_bucket",
    "block_table_too_wide",
    "missing_capability",
    "active_model_hooks",
    "archer_callbacks",
    "transfer_scheduler",
    "expert_dispatcher",
    "kv_offload",
    "flashinfer_plan_path",
    "dynamic_allocations",
    "native_paged_required",
    "mla_layout_unsupported",
    "kv_storage_mismatch",
    "paged_class_unregistered",
    "layer_idx_invalid",
    "layer_write_unproven",
    "insufficient_memory",
    "capture_failed",
    "quarantined",
    "invalidated",
)
```

Construct with explicit dependencies:

```python
def __init__(
    self,
    model_runner: ModelRunner,
    storage: PagedKVStorage | None,
    *,
    enabled: bool = False,
    batch_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    context_buckets: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096),
    warmup_iters: int = 2,
    max_graph_memory_bytes: int = 0,
) -> None:
```

Normalize sorted positive unique buckets; reject empty enabled bucket sets and `warmup_iters < 1`. `storage=None` is accepted only to preserve the eager engine path around a legacy `PagedKVCache`; it always returns the model capability reason (normally `native_paged_required`) or `kv_storage_mismatch`, never reserves scratch, and never captures. At every eligibility check with bound storage, obtain `model_runner.decode_graph_capability()` and return its exact typed reason unless it is safe, its `storage_owner_id == storage.owner_id`, its proof set exactly covers the registry's Qwen3 `(class_fqn, layer_idx)` bindings, and canonical `storage.spec.device == model_runner.device`. Immediately after buffer creation and immediately before every capture/replay, walk all `PreparedDecodeBuffers` tensors plus `_CapturedGraphState.output_logits` and require `tensor.device == storage.spec.device`; any mismatch returns `kv_storage_mismatch` before capture/replay and before scratch allocation where possible. Recheck proof coverage and device equality immediately before first capture after invalidation/reload. Read `MOE_DISABLE_CUDA_GRAPHS` on every check so rollback does not require restart.

- [x] **Step 4: Implement stable scratch layout and lazy capture**

Reserve once, on first eligible capture, one unique scratch block for each row of the largest configured batch bucket. Convert allocator exhaustion to `_GraphMemoryUnavailable`; `try_execute()` catches that exception, records `insufficient_memory`, and returns `None` without quarantining a graph key:

```python
count = max(self.batch_buckets)
self._scratch_block_ids = self.storage.reserve_graph_scratch_blocks(count)
```

Map row `r` to one unique scratch ID. Synthetic capture and padded replay rows use `seq_len=1`, put that ID in block-table column zero, zero remaining columns, and map their slot to `block_id * block_size`; real replay rows overwrite their full required block-table prefix. `_capture_key(key)` must:

1. Allocate one `PreparedDecodeBuffers` object.
2. Populate every row with scratch metadata.
3. Call `model_runner.prepare_batch_side_effects()` **zero times** for synthetic warmup/capture.
4. Warm up `warmup_iters` on a side stream and synchronize that stream.
5. Measure `torch.cuda.memory_allocated(device)` before/after capture.
6. Allocate `static_output_logits` before capture with `device=storage.spec.device`; assert every `buffers.tensor_values()` tensor and `static_output_logits` has that exact device. Inside capture run `captured_logits = model_runner.forward_prepared_decode(buffers)` followed by `static_output_logits.copy_(captured_logits)` so the replay output pointer is stable.
7. Reject and free the just-created state when cumulative graph private-pool bytes exceed nonzero `max_graph_memory_bytes`.
8. Install state only if its captured generation still equals `self.generation`.

Use one `threading.RLock` around scratch reservation, capture, replay, invalidation, and close. Assert that every scratch ID remains in `storage.graph_scratch_blocks` immediately before metadata copy and replay. Never hold the graph lock while calling application lifecycle code; application lock ordering is defined in Task 6.

- [x] **Step 5: Implement replay with eager signal, real-row slicing, and exception quarantine**

```python
def try_execute(self, batch: BatchMetadata) -> torch.Tensor | None:
    decision = self.check_eligibility(batch)
    if not decision.eligible or decision.key is None:
        self._record_fallback(decision.reason)
        return None
    with self._lock:
        state = self._graphs.get(decision.key)
        if state is None:
            try:
                state = self._capture_key(decision.key)
            except _GraphMemoryUnavailable:
                self._record_fallback("insufficient_memory")
                return None
            except (RuntimeError, ValueError) as exc:
                self._quarantined[decision.key] = type(exc).__name__
                self._stats.capture_failures += 1
                self._record_fallback("capture_failed")
                return None
        if state.generation != self.generation:
            self._record_fallback("invalidated")
            return None
        if not self._validate_state_devices(state):
            self._record_fallback("kv_storage_mismatch")
            return None
        scratch_rows = self._scratch_rows(
            start=len(batch.seq_ids),
            stop=decision.key.batch_size,
            context_size=decision.key.context_size,
        )
        self.model_runner.copy_decode_batch(batch, state.buffers, scratch_rows)
        self.model_runner.prepare_batch_side_effects(batch)
        state.graph.replay()
        self._stats.replays += 1
        return state.output_logits[: len(batch.seq_ids)].clone()
```

Input validation/copy failures that occur before `graph.replay()` may quarantine the key and return `None` for eager execution. Any exception raised by `graph.replay()` is treated as potentially partially launched: quarantine and remove the state, then re-raise so the server marks the step unhealthy rather than eagerly running the same token twice.

Implement `invalidate(reason)` to synchronize the device only when graphs exist, clear graphs/quarantine, increment generation, and retain scratch pages for recapture. Implement `close()` to invalidate, release scratch pages, and make future checks return `disabled`.

- [x] **Step 6: Run CPU lifecycle tests and verify GREEN**

Run:

```bash
pytest -q tests/python/serving/test_cuda_graph.py \
  -k 'not capture_and_replay_cuda'
```

Expected: CPU-safe tests PASS without requiring CUDA.

- [x] **Step 7: Commit the runner lifecycle**

```bash
git add moe_infinity/serving/cuda_graph.py tests/python/serving/test_cuda_graph.py
git commit -m "feat(serving): add safe lazy decode CUDA graph runner"
```

### Task 5: Integrate graph replay into active continuous batching

**Files:**
- Modify: `moe_infinity/serving/engine.py:65-168,726-751`
- Modify: `tests/python/serving/test_engine.py`

- [x] **Step 1: Write failing active-path tests**

```python
def test_pure_decode_uses_graph_logits_before_existing_sampler() -> None:
    engine = _make_graph_safe_native_paged_engine()
    graph_logits = torch.zeros(2, 100)
    graph_logits[:, 41] = 1.0
    engine.cuda_graph_runner = SimpleNamespace(
        try_execute=Mock(return_value=graph_logits),
        stats=Mock(return_value={}),
    )
    batch = _decode_batch_for_engine([10, 11])

    logits = engine._execute_batch(batch)

    torch.testing.assert_close(logits, graph_logits)
    engine.cuda_graph_runner.try_execute.assert_called_once_with(batch)


def test_graph_miss_runs_exact_same_decode_batch_eagerly() -> None:
    engine = _make_graph_safe_native_paged_engine()
    engine.cuda_graph_runner = SimpleNamespace(try_execute=Mock(return_value=None))
    engine.model_runner.execute = Mock(return_value=torch.ones(2, 100))
    batch = _decode_batch_for_engine([10, 11])

    result = engine._execute_batch(batch)

    engine.model_runner.execute.assert_called_once_with(batch)
    assert result.shape == (2, 100)


def test_mixed_batch_graphs_decode_partition_and_recombines_order() -> None:
    engine = _make_paged_engine()
    batch = _mixed_batch_for_engine()
    engine.model_runner.execute = Mock(return_value=torch.tensor([[10.0], [11.0]]))
    engine.cuda_graph_runner.try_execute = Mock(
        return_value=torch.tensor([[20.0], [21.0]])
    )

    result = engine._execute_batch(batch)

    assert result.tolist() == [[10.0], [20.0], [11.0], [21.0]]


def test_sampler_receives_only_real_rows_from_padded_graph() -> None:
    engine = _make_graph_safe_native_paged_engine()
    engine.cuda_graph_runner.try_execute = Mock(return_value=torch.zeros(3, 100))
    engine.sampler.sample = Mock(return_value=SamplerOutput(torch.tensor([1, 2, 3])))
    _install_three_decode_sequences(engine)

    engine.step()

    sampled_logits = engine.sampler.sample.call_args.args[0]
    assert sampled_logits.shape == (3, 100)


def test_non_paged_decode_is_always_eager() -> None:
    engine = _make_resident_non_paged_engine(enable_decode_cuda_graphs=True)
    engine.model_runner.execute = Mock(return_value=torch.ones(2, 100))
    result = engine._execute_batch(_decode_batch_for_engine([10, 11]))
    assert result.shape == (2, 100)
    assert engine.cuda_graph_runner.stats()["captures"] == 0
    assert engine.cuda_graph_runner.stats()["fallback_reasons"]["native_paged_required"] == 1
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q tests/python/serving/test_engine.py -k 'graph or padded_graph'
```

Expected: FAIL because the engine has no `cuda_graph_runner` and never calls it.

- [x] **Step 3: Instantiate opt-in runner and preserve mixed-batch order**

In `ContinuousBatchingEngine.__init__`, resolve storage before creating the scheduler-facing cache and model runner:

```python
backend = self._resolve_attention_backend(engine)
backend_storage = backend.storage if isinstance(backend, PagedAttentionBackend) else None
self.kv_cache = PagedKVCache(
    num_blocks=num_blocks,
    block_size=block_size,
    num_layers=num_layers,
    num_heads=num_kv_heads,
    head_dim=head_dim,
    dtype=self.dtype,
    device=self.device,
    storage=backend_storage,
)
storage = self.kv_cache.storage
self.paged_attention_registry = (
    PagedAttentionLayerRegistry.empty(reason="native_paged_required")
    if storage is None
    else PagedAttentionLayerRegistry.register_qwen3(
        model=model, backend=backend, storage=storage
    )
)
self.model_runner = ModelRunner(
    model,
    engine,
    device=self.device,
    paged_kv_storage=storage,
    paged_attention_registry=self.paged_attention_registry,
    decode_graph_capability_provider=decode_graph_capability_provider,
)
```

If the backend already exists, validate all dimensions and exact canonical device against serving config and fail graph capability on mismatch rather than silently renumbering blocks or moving tensors. Registration runs once before warmup/capture and freezes Qwen3 bindings for the engine generation. Existing non-graph callers may still construct `PagedKVCache(...)` without `storage`; that legacy path remains functional but `storage is None` makes graphs ineligible. Non-paged models remain `native_paged_required`; exact DeepSeek V2/V3 paged classes remain eager with `mla_layout_unsupported`; unknown paged classes are `paged_class_unregistered`; invalid/missing/duplicate Qwen3 layer indices are `layer_idx_invalid`. Construct `CudaGraphRunner` with the possibly-`None` cache storage; its `storage=None` gate records the capability reason and never reserves scratch:

```python
self.cuda_graph_runner = CudaGraphRunner(
    self.model_runner,
    self.kv_cache.storage,
    enabled=self._get_bool_config("enable_decode_cuda_graphs", False),
    batch_buckets=self._get_int_tuple_config(
        "decode_cuda_graph_batch_sizes", (1, 2, 4, 8, 16, 32)
    ),
    context_buckets=self._get_int_tuple_config(
        "decode_cuda_graph_context_sizes", (128, 256, 512, 1024, 2048, 4096)
    ),
    warmup_iters=self._get_int_config("decode_cuda_graph_warmup_iters", 2),
    max_graph_memory_bytes=self._get_int_config(
        "decode_cuda_graph_max_memory_bytes", 0
    ),
)
```

Before constructing the runner, if storage is bound, require that `self.kv_cache.storage is self.model_runner.get_paged_kv_storage()` and canonical `self.device == storage.spec.device`; if not, the runner remains present but its capability is `kv_storage_mismatch` and every batch is eager. If storage is `None`, the runner is eager-only without dereferencing `storage.spec`. Add strict bool/list parsing helpers; booleans are not accepted as integers. Replace only decode execution:

```python
def _execute_decode_batch(self, batch: BatchMetadata) -> torch.Tensor:
    graph_logits = self.cuda_graph_runner.try_execute(batch)
    if graph_logits is not None:
        return graph_logits
    return self.model_runner.execute(batch)
```

Update `_execute_batch()` rules:

- all-prefill: eager `model_runner.execute(batch)`;
- all-decode with explicit resident ordinary-GQA Qwen3 capability: `_execute_decode_batch(batch)`;
- mixed batch with the same Qwen3 capability: existing split, eager prefill, graph-attempt decode, existing `recombine_outputs()`;
- DeepSeek V2/V3 MLA, non-paged, FlashInfer, offloaded, or missing-capability execution: preserve the existing eager behavior and record the explicit reason; never attempt capture.

Do not move `_extract_last_token_logits()` or `Sampler.sample()` into `CudaGraphRunner`.

- [x] **Step 4: Verify active-path GREEN**

Run:

```bash
pytest -q tests/python/serving/test_engine.py \
  tests/python/serving/test_flashinfer_mixed_batch.py \
  tests/python/serving/test_paged_kv_storage.py
```

Expected: all tests PASS, including pre-existing speculative decode tests.

- [x] **Step 5: Commit active serving integration**

```bash
git add moe_infinity/serving/engine.py \
  tests/python/serving/test_engine.py
git commit -m "feat(serving): replay eligible decode batches"
```

### Task 6: Close graph resources on reload, shutdown, and hot replacement

**Files:**
- Modify: `moe_infinity/serving/engine.py`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:475-527,1708-1729`
- Modify: `tests/python/serving/test_hot_reload.py`
- Modify: `tests/python/serving/test_api_routes.py`

- [x] **Step 1: Write failing reload ordering tests**

```python
def test_reload_invalidates_graphs_before_importlib_reload(monkeypatch) -> None:
    events: list[str] = []
    fake_engine = SimpleNamespace(
        invalidate_cuda_graphs=lambda reason: events.append(f"invalidate:{reason}")
    )
    monkeypatch.setattr(srv, "engine", fake_engine)
    monkeypatch.setattr(importlib, "reload", lambda module: events.append("reload") or module)

    with TestClient(srv.app) as client:
        response = client.post("/v1/reload", json={"modules": ["json"]})

    assert response.status_code == 200
    assert events == ["invalidate:module_reload", "reload"]


def test_failed_reload_leaves_graphs_invalidated(monkeypatch) -> None:
    invalidate = Mock()
    monkeypatch.setattr(
        srv, "engine", SimpleNamespace(invalidate_cuda_graphs=invalidate)
    )
    monkeypatch.setattr(importlib, "reload", Mock(side_effect=RuntimeError("bad")))
    with TestClient(srv.app) as client:
        response = client.post("/v1/reload", json={"modules": ["json"]})
    assert response.json()["status"] == "partial"
    invalidate.assert_called_once_with("module_reload")


def test_application_shutdown_closes_current_graph_runner(monkeypatch) -> None:
    current = SimpleNamespace(shutdown=Mock())
    monkeypatch.setattr(srv, "engine", current)
    with TestClient(srv.app):
        pass
    current.shutdown.assert_called_once_with()


def test_hot_replacement_closes_old_engine_before_return(monkeypatch) -> None:
    events: list[str] = []
    storage = _make_storage(num_blocks=8)
    scratch = storage.reserve_graph_scratch_blocks(2)
    old = _EngineWithGraphRunner(storage=storage, events=events)
    new = SimpleNamespace()
    monkeypatch.setattr(srv, "engine", old)
    srv._replace_engine(new)
    assert srv.engine is new
    assert events == ["old:close"]
    assert storage.num_graph_scratch_blocks == 0
    assert storage.block_allocator.num_free_blocks == storage.spec.num_blocks
    assert all(block_id < storage.spec.num_blocks for block_id in scratch)


def test_hot_replacement_waits_for_active_step_and_obeys_lock_order(monkeypatch) -> None:
    step_entered = threading.Event()
    release_step = threading.Event()
    events: list[str] = []
    old = _BlockingEngine(step_entered, release_step, events)
    new = SimpleNamespace()
    monkeypatch.setattr(srv, "engine", old)
    step_thread = threading.Thread(target=srv._run_engine_step_once)
    replace_thread = threading.Thread(target=lambda: srv._replace_engine(new))
    step_thread.start()
    assert step_entered.wait(timeout=1.0)
    replace_thread.start()
    assert "old:close" not in events
    release_step.set()
    step_thread.join(timeout=1.0)
    replace_thread.join(timeout=1.0)
    assert events == ["step:exit", "graph:close", "old:close"]
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/python/serving/test_hot_reload.py \
  tests/python/serving/test_api_routes.py \
  -k 'reload or shutdown or replacement or lock_order'
```

Expected: FAIL because reload does not invalidate graphs, shutdown calls only optional `engine.shutdown`, direct engine assignment does not close the old runner, and no lifecycle lock serializes an active step with replacement.

- [x] **Step 3: Implement lifecycle hooks and lock ordering**

Add to the engine:

```python
def invalidate_cuda_graphs(self, reason: str) -> None:
    self.cuda_graph_runner.invalidate(reason)


def shutdown(self) -> None:
    self.cuda_graph_runner.close()
```

Make `shutdown()` idempotent and have it close graph states, release authoritative-storage scratch blocks, and prevent recapture. Add an application-level `_engine_lifecycle_lock = threading.RLock()` and exact helper:

```python
def _replace_engine(new_engine: ContinuousBatchingEngine) -> None:
    global engine
    with _engine_lifecycle_lock:
        old_engine = engine
        engine = new_engine
        if old_engine is not None and old_engine is not new_engine:
            old_engine.shutdown()
```

The engine loop acquires `_engine_lifecycle_lock` around reading the global engine and calling one `step()`. Both `initialize_with_model()` and `_initialize_model()` call `_replace_engine(initialized_engine)` instead of assigning `engine` directly. The application shutdown handler first signals and awaits `_engine_task` without holding the lifecycle lock, then acquires the lifecycle lock and calls `engine.shutdown()` exactly once. `/v1/reload` acquires the same lifecycle lock, calls `engine.invalidate_cuda_graphs("module_reload")`, then reloads modules; it does not recapture.

Document and test the sole lock order: **application `_engine_lifecycle_lock` → engine step ownership → `CudaGraphRunner._lock`**. `CudaGraphRunner` never acquires the lifecycle lock or calls scheduler/request/application methods while holding its lock. Shutdown does not await a task while holding either lock. This makes replacement wait for an active replay/step, then closes the old runner before returning.

- [x] **Step 4: Run reload tests and verify GREEN**

Run:

```bash
pytest -q tests/python/serving/test_hot_reload.py \
  tests/python/serving/test_api_routes.py \
  tests/python/serving/test_stream.py \
  tests/python/serving/test_cancellation.py
```

Expected: all tests PASS.

- [x] **Step 5: Commit lifecycle invalidation**

```bash
git add moe_infinity/serving/engine.py \
  moe_infinity/entrypoints/openai/api_server_v2.py \
  tests/python/serving/test_hot_reload.py
git commit -m "fix(serving): close CUDA graphs across engine lifecycle"
```

### Task 7: Account memory and expose bounded metrics

**Files:**
- Modify: `moe_infinity/serving/memory_manager.py:10-169`
- Modify: `moe_infinity/serving/engine.py:653-668`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:716-745`
- Modify: `tests/python/serving/test_memory_manager.py`
- Modify: `tests/python/serving/test_api_routes.py`

- [x] **Step 1: Write failing memory and metric tests**

```python
def test_report_includes_graph_pool_and_reserved_scratch_bytes() -> None:
    _, MemoryManager = _load_classes()
    manager = MemoryManager(device=torch.device("cpu"))
    manager.set_cuda_graph_usage(graph_pool_bytes=4096, scratch_kv_bytes=2048)
    report = manager.report()
    assert report["cuda_graph_pool_bytes"] == 4096
    assert report["cuda_graph_scratch_kv_bytes"] == 2048
    assert report["cuda_graph_total_bytes"] == 6144


def test_metrics_endpoint_exports_graph_counters_and_bounded_reasons(client) -> None:
    srv.engine.get_stats.return_value = {
        **_make_mock_stats(),
        "cuda_graph": {
            "captures": 2,
            "replays": 9,
            "capture_failures": 1,
            "graphs": 2,
            "graph_pool_bytes": 4096,
            "scratch_kv_bytes": 2048,
            "fallback_reasons": {"expert_dispatcher": 7, "not_decode": 3},
            "capability_reason": "expert_dispatcher",
        },
    }
    response = client.get("/metrics")
    assert 'moe_cuda_graph_replays_total 9' in response.text
    assert 'moe_cuda_graph_fallback_total{reason="expert_dispatcher"} 7' in response.text
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/python/serving/test_memory_manager.py \
  tests/python/serving/test_api_routes.py \
  -k 'graph or metrics_endpoint'
```

Expected: FAIL because graph memory fields and metrics are absent.

- [x] **Step 3: Implement memory accounting and stats wiring**

Add initialized counters and a validated setter:

```python
def set_cuda_graph_usage(
    self, *, graph_pool_bytes: int, scratch_kv_bytes: int
) -> None:
    if graph_pool_bytes < 0 or scratch_kv_bytes < 0:
        raise ValueError("CUDA graph memory usage must be non-negative")
    self._cuda_graph_pool_bytes = graph_pool_bytes
    self._cuda_graph_scratch_kv_bytes = scratch_kv_bytes
```

`report()` returns those two fields and their sum. `ContinuousBatchingEngine.get_stats()` obtains `cuda_graph_runner.stats()`, computes scratch bytes as:

```python
scratch_kv_bytes = (
    storage.num_graph_scratch_blocks
    * storage.spec.block_size
    * storage.spec.num_layers
    * 2
    * storage.spec.num_kv_heads
    * storage.spec.head_dim
    * torch.empty((), dtype=storage.spec.dtype).element_size()
)
```

Update `MemoryManager` before returning stats. Keep scratch bytes distinct from graph-pool bytes because authoritative native K/V storage was preallocated but reservation reduces request capacity. Include `kv_storage_owner_id`, `capability_safe`, bounded `capability_reason`, `registered_paged_layers`, and `proved_write_layers` in JSON stats; eligibility requires the latter counts to be equal and nonzero. Do not expose owner ID or class names as Prometheus labels.

- [x] **Step 4: Export fixed-name Prometheus metrics**

Add counters/gauges:

- `moe_cuda_graph_captures_total`
- `moe_cuda_graph_replays_total`
- `moe_cuda_graph_capture_failures_total`
- `moe_cuda_graph_instances`
- `moe_cuda_graph_pool_bytes`
- `moe_cuda_graph_scratch_kv_bytes`
- `moe_cuda_graph_fallback_total` with one `reason` label sample for every constant in `FALLBACK_REASONS`, including zero values

Never use exception text, model IDs, graph keys, request IDs, or arbitrary strings as labels.

- [x] **Step 5: Run memory/API tests and verify GREEN**

Run:

```bash
pytest -q tests/python/serving/test_memory_manager.py \
  tests/python/serving/test_api_routes.py
```

Expected: all tests PASS and JSON stats remain serializable.

- [x] **Step 6: Commit accounting and observability**

```bash
git add moe_infinity/serving/memory_manager.py \
  moe_infinity/serving/engine.py \
  moe_infinity/entrypoints/openai/api_server_v2.py \
  tests/python/serving/test_memory_manager.py \
  tests/python/serving/test_api_routes.py
git commit -m "feat(serving): report decode CUDA graph usage"
```

### Task 8: Add opt-in configuration and rollback controls

**Files:**
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:475-506,1778-1919`
- Modify: `tests/python/serving/test_api_routes.py`

- [x] **Step 1: Write failing CLI/config tests**

```python
def test_decode_cuda_graphs_are_disabled_by_default(monkeypatch) -> None:
    args = _parse_args(monkeypatch, required_only=True)
    assert args.enable_decode_cuda_graphs is False
    config = srv._build_engine_config(args, _mock_model())
    assert config["enable_decode_cuda_graphs"] is False


def test_decode_cuda_graph_cli_values_reach_engine_config(monkeypatch) -> None:
    args = _parse_args(
        monkeypatch,
        extra=[
            "--enable-decode-cuda-graphs",
            "--decode-cuda-graph-batch-sizes", "1", "2", "4",
            "--decode-cuda-graph-context-sizes", "128", "256",
            "--decode-cuda-graph-max-memory-bytes", "1073741824",
        ],
    )
    config = srv._build_engine_config(args, _mock_model())
    assert config["enable_decode_cuda_graphs"] is True
    assert config["decode_cuda_graph_batch_sizes"] == (1, 2, 4)
    assert config["decode_cuda_graph_context_sizes"] == (128, 256)
    assert config["decode_cuda_graph_max_memory_bytes"] == 1073741824


def test_enable_flag_does_not_override_unsafe_runtime_capability() -> None:
    engine = _make_offloaded_moe_engine(enable_decode_cuda_graphs=True)
    assert engine.cuda_graph_runner.check_eligibility(_make_decode_batch()).reason in {
        "active_model_hooks", "archer_callbacks", "transfer_scheduler",
        "expert_dispatcher", "kv_offload", "dynamic_allocations",
    }
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/python/serving/test_api_routes.py -k decode_cuda_graph
```

Expected: FAIL because parser/config fields are absent.

- [x] **Step 3: Add explicit opt-in fields**

Add CLI arguments:

```python
parser.add_argument("--enable-decode-cuda-graphs", action="store_true", default=False)
parser.add_argument(
    "--decode-cuda-graph-batch-sizes", nargs="+", type=int,
    default=[1, 2, 4, 8, 16, 32],
)
parser.add_argument(
    "--decode-cuda-graph-context-sizes", nargs="+", type=int,
    default=[128, 256, 512, 1024, 2048, 4096],
)
parser.add_argument("--decode-cuda-graph-warmup-iters", type=int, default=2)
parser.add_argument("--decode-cuda-graph-max-memory-bytes", type=int, default=0)
```

Copy these into `_build_engine_config()` and add keyword arguments with the same defaults to `initialize_with_model()`. Both initialization paths pass the top-level MoE/native-runtime object as `decode_graph_capability_provider`; tests verify it is not silently replaced by the lower-level offload engine. Validate positivity in `CudaGraphRunner`, the single owner of bucket policy. The enable flag only permits capability evaluation; it never marks a runtime safe. CLI-created MoE/offload engines therefore remain eager until they provide an `eligible` resident/native-paged capability.

Rollback hierarchy:

1. Start without `--enable-decode-cuda-graphs` (default).
2. For an already-configured process, set `MOE_DISABLE_CUDA_GRAPHS=1`; the gate reads it per batch and immediately routes eager.
3. Call `engine.invalidate_cuda_graphs("operator_rollback")` to release captured graph states; call `shutdown()` only during engine replacement/application shutdown to close the runner and release authoritative-storage scratch reservations.

- [x] **Step 4: Run config tests and verify GREEN**

Run:

```bash
pytest -q tests/python/serving/test_api_routes.py -k 'decode_cuda_graph or initialize_with_model'
```

Expected: all selected tests PASS.

- [x] **Step 5: Commit opt-in controls**

```bash
git add moe_infinity/entrypoints/openai/api_server_v2.py \
  tests/python/serving/test_api_routes.py
git commit -m "feat(serving): add opt-in decode CUDA graph controls"
```

### Task 9: Prove CUDA replay equivalence and eager fallback on real devices

**Files:**
- Modify: `tests/python/serving/test_cuda_graph.py`
- Modify: `tests/python/serving/test_flashinfer_model_runner.py`
- Modify: `tests/python/serving/test_paged_kv_storage.py`
- Modify: `tests/python/integration/test_paged_attention_backend.py`
- Modify: `tests/python/serving/test_paged_attention_registry.py`
- Modify: `tests/python/ops/test_paged_kv_write.py`

- [x] **Step 1: Write CUDA equivalence tests before completing device behavior**

```python
@requires_cuda
@pytest.mark.parametrize("real_batch_size", [1, 2, 3, 4])
def test_capture_and_replay_matches_eager_with_padding(real_batch_size: int) -> None:
    eager_runner, graph_runner, storage = _make_resident_native_paged_cuda_fixture(
        batch_buckets=(1, 2, 4), context_buckets=(16, 32)
    )
    assert graph_runner.storage is storage
    assert eager_runner.get_paged_kv_storage() is storage
    batch = _make_decode_batch(
        batch_size=real_batch_size,
        context_lengths=[2 + i * 3 for i in range(real_batch_size)],
        input_token_ids=[20 + i for i in range(real_batch_size)],
    )
    expected = eager_runner.execute(batch)
    actual = graph_runner.try_execute(batch)
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert actual.shape[0] == real_batch_size


@requires_cuda
def test_replay_observes_updated_tokens_positions_and_kv_metadata() -> None:
    eager_runner, graph_runner, storage = _make_resident_native_paged_cuda_fixture()
    first = _make_decode_batch(context_lengths=[3, 7], input_token_ids=[20, 30])
    second = _make_decode_batch(
        context_lengths=[4, 10],
        input_token_ids=[21, 31],
        block_tables=[[1, 4, 5], [2, 6, 7]],
    )
    assert graph_runner.try_execute(first) is not None
    actual = graph_runner.try_execute(second)
    expected = eager_runner.execute(second)
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert graph_runner.stats()["kv_storage_owner_id"] == storage.owner_id
    for binding in graph_runner.model_runner.paged_attention_registry.bindings:
        _assert_current_token_slot_written(
            storage,
            layer_idx=binding.layer_idx,
            slot_mapping=second.attention_slot_mapping(),
        )


@requires_cuda
def test_capture_state_devices_exactly_match_storage_and_runner() -> None:
    eager_runner, graph_runner, storage = _make_resident_native_paged_cuda_fixture()
    assert eager_runner.device == storage.spec.device
    assert graph_runner.try_execute(_make_decode_batch()) is not None
    for state in graph_runner._graphs.values():
        assert state.output_logits.device == storage.spec.device
        assert all(
            tensor.device == storage.spec.device
            for tensor in state.buffers.tensor_values()
        )


@requires_cuda
def test_replayed_kv_write_persists_into_following_decode_for_every_layer() -> None:
    eager, graph, storage = _make_resident_native_paged_cuda_fixture(num_layers=3)
    first = _make_decode_batch(context_lengths=[3], input_token_ids=[20])
    second = _make_decode_batch(context_lengths=[4], input_token_ids=[21])
    replay_first = graph.try_execute(first)
    assert replay_first is not None
    snapshots = _snapshot_written_slots(storage, first.attention_slot_mapping())
    replay_second = graph.try_execute(second)
    eager_second = eager.execute(second)
    assert replay_second is not None
    _assert_snapshots_unchanged(storage, snapshots)
    torch.testing.assert_close(replay_second, eager_second, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_unproven_paged_class_never_captures() -> None:
    engine = _make_engine_with_unregistered_paged_class()
    _ = engine._execute_batch(_make_decode_batch())
    assert engine.cuda_graph_runner.stats()["captures"] == 0
    assert engine.cuda_graph_runner.stats()["fallback_reasons"]["paged_class_unregistered"] == 1


@requires_cuda
@pytest.mark.parametrize("family", ["deepseek_v2", "deepseek_v3"])
def test_deepseek_mla_never_captures(family: str) -> None:
    engine = _make_resident_deepseek_paged_engine_or_skip(
        family, enable_decode_cuda_graphs=True
    )
    _ = engine._execute_batch(_make_decode_batch())
    assert engine.cuda_graph_runner.stats()["captures"] == 0
    assert engine.cuda_graph_runner.stats()["fallback_reasons"]["mla_layout_unsupported"] == 1


@requires_cuda
def test_offloaded_moe_executes_eager_without_capture() -> None:
    engine = _make_cuda_offload_engine(enable_decode_cuda_graphs=True)
    engine.model_runner.execute = Mock(wraps=engine.model_runner.execute)
    batch = _make_decode_batch()
    result = engine._execute_batch(batch)
    assert result.shape[0] == len(batch.seq_ids)
    engine.model_runner.execute.assert_called_once_with(batch)
    assert engine.cuda_graph_runner.stats()["captures"] == 0
    capability = engine.model_runner.decode_graph_capability()
    assert capability.safe is False
    assert capability.reason in {
        "active_model_hooks", "archer_callbacks", "transfer_scheduler",
        "expert_dispatcher", "kv_offload", "dynamic_allocations",
    }
    assert engine.cuda_graph_runner.stats()["fallback_reasons"][capability.reason] == 1


@requires_cuda
def test_resident_non_paged_model_remains_eager() -> None:
    engine = _make_cuda_resident_non_paged_engine(enable_decode_cuda_graphs=True)
    _ = engine._execute_batch(_make_decode_batch())
    assert engine.cuda_graph_runner.stats()["captures"] == 0
    assert engine.cuda_graph_runner.stats()["fallback_reasons"]["native_paged_required"] == 1
```

- [x] **Step 2: Run CUDA tests and verify RED against incomplete device behavior**

Run:

```bash
pytest -q tests/python/serving/test_cuda_graph.py \
  -k 'capture_and_replay_matches or observes_updated or state_devices or persists_into_following or unproven_paged or deepseek_mla or offloaded_moe or non_paged'
```

Expected on a CUDA host: at least one FAIL until stable output, metadata copies, and padding are complete. Expected on a CPU-only host: SKIP with the existing `requires_cuda` marker.

- [x] **Step 3: Finish only the minimal device fixes exposed by RED**

Fix production behavior rather than weakening tolerances or expected values. In particular:

- retain captured input, metadata, and output tensor objects for the state lifetime;
- use only in-place `zero_()`, `fill_()`, and `copy_()` updates before replay;
- call `torch.cuda.synchronize(device)` after warmup/capture during lifecycle setup, not on every replay;
- clone only the real output rows after replay so a later replay cannot mutate a caller's logits;
- never include padded rows in sampler input;
- verify storage owner identity immediately before every capture/replay;
- write current-token K/V to authoritative `slot_mapping` for every proof-covered layer before each decode attention launch;
- preserve previous-token pages across following eager/replay steps;
- never capture DeepSeek V2/V3 MLA, non-paged, offload/dispatcher/Archer/transfer-scheduler/KV-offload, dynamic-allocation, or FlashInfer-plan paths;
- keep Qwen3's per-layer current-token write-before-attention and following-token persistence unchanged.

- [x] **Step 4: Run CUDA equivalence suite and verify GREEN**

Run:

```bash
pytest -q tests/python/serving/test_cuda_graph.py \
  tests/python/serving/test_flashinfer_model_runner.py \
  tests/python/serving/test_paged_kv_storage.py \
  tests/python/integration/test_paged_attention_backend.py \
  tests/python/serving/test_paged_attention_registry.py \
  tests/python/ops/test_paged_kv_write.py
```

Expected on CUDA: all tests PASS. Expected without CUDA: deterministic CPU tests PASS and CUDA tests SKIP.

- [x] **Step 5: Commit device equivalence coverage**

```bash
git add moe_infinity/serving/cuda_graph.py \
  moe_infinity/serving/model_runner.py \
  tests/python/serving/test_cuda_graph.py \
  tests/python/serving/test_flashinfer_model_runner.py \
  tests/python/serving/test_paged_kv_storage.py \
  tests/python/integration/test_paged_attention_backend.py \
  tests/python/serving/test_paged_attention_registry.py \
  tests/python/ops/test_paged_kv_write.py
git commit -m "test(serving): prove decode CUDA graph replay parity"
```

### Task 10: Add a paired launch-overhead benchmark

**Files:**
- Create: `benchmarks/serving/decode_cuda_graph.py`
- Create: `benchmarks/serving/decode_cuda_graph_fixture.py`
- Test: `tests/python/serving/test_cuda_graph.py`

- [x] **Step 1: Write a failing CPU schema/CLI test**

```python
def test_benchmark_result_schema_does_not_claim_speedup() -> None:
    module = _load_benchmark_module("benchmarks/serving/decode_cuda_graph.py")
    result = module.build_result(
        config={"batch_size": 4},
        eager_us=[100.0, 101.0],
        replay_us=[90.0, 91.0],
        graph_stats={"replays": 2},
        environment={"gpu": "test"},
    )
    assert result["schema_version"] == 1
    assert result["measurements"]["eager_us"] == [100.0, 101.0]
    assert result["measurements"]["replay_us"] == [90.0, 91.0]
    assert "claimed_speedup" not in result


def test_benchmark_fixture_mode_needs_no_model_or_offload_args() -> None:
    module = _load_benchmark_module("benchmarks/serving/decode_cuda_graph.py")
    args = module.parse_args(["--mode", "fixture", "--output-json", "/tmp/out.json"])
    module.validate_args(args)
    assert args.model is None
    assert args.offload_dir is None


def test_benchmark_model_mode_requires_model_and_offload_dir() -> None:
    module = _load_benchmark_module("benchmarks/serving/decode_cuda_graph.py")
    args = module.parse_args(["--mode", "model", "--output-json", "/tmp/out.json"])
    with pytest.raises(ValueError, match="--model and --offload-dir"):
        module.validate_args(args)
```

- [x] **Step 2: Run the schema test and verify RED**

Run:

```bash
pytest -q tests/python/serving/test_cuda_graph.py -k benchmark_result_schema
```

Expected: FAIL because the benchmark module does not exist.

- [x] **Step 3: Implement the benchmark with paired controls**

The CLI must accept:

```text
--model --offload-dir --batch-sizes --context-sizes --warmup-iters
--measure-iters --output-json --mode {fixture,model}
```

Define modes formally:

- `fixture`: imports `build_fixture()` from `decode_cuda_graph_fixture.py`; constructs a deterministic two-layer resident `Qwen3PagedAttention` model, a native `PagedAttentionBackend`, and one persistent `PagedKVStorage` shared by allocator/cache/backend. It registers both layers with indices `0,1`, requires two `paged_kv_write_` proofs, and records a post-replay KV-slot checksum for each layer. It accepts no model checkpoint and no offload directory.
- `model`: requires both `--model MODEL_ID_OR_PATH` and `--offload-dir PATH`. It uses the real loader and records capability evidence. Current offloaded MoE loaders are expected to report an unsafe reason and execute eager-only; the benchmark must not force capture or describe that result as a graph comparison.

`build_fixture()` returns `BenchmarkFixture(model_runner, graph_runner, storage, make_batch)` and tests assert all three runtime components share `storage.owner_id` and tensor pointers for the fixture lifetime.

For each exact `(batch_size, context_size)` point:

1. Build one deterministic decode batch and preallocate all tensors.
2. Warm eager and graph paths independently.
3. Alternate measurement order per repetition (`eager,graph` then `graph,eager`) to reduce drift.
4. Time GPU work with `torch.cuda.Event(enable_timing=True)` and synchronize the stop event.
5. Record raw microseconds, p50/p90/p99, graph replay coverage, captures, fallbacks, graph pool bytes, scratch KV bytes, CUDA/PyTorch versions, GPU name, model identifier, dtype, and bucket configuration.
6. Record `torch.profiler` CUDA kernel/launch counts for one eager and one replay iteration when `--profile-launches` is supplied.
7. Never assert or print that replay is faster. Print the observed ratio as `eager_p50_us / replay_p50_us` with the label `observed_ratio`, including values below 1.0.
8. Fail the command only for correctness mismatch, malformed configuration, missing CUDA, or runtime error—not for performance.

Use `build_result(...)` as a pure function so schema tests run on CPU.

- [x] **Step 4: Verify schema and run a smoke benchmark**

Run CPU schema test:

```bash
pytest -q tests/python/serving/test_cuda_graph.py -k benchmark_result_schema
```

Expected: PASS.

Run CUDA fixture smoke test:

```bash
python benchmarks/serving/decode_cuda_graph.py \
  --mode fixture \
  --batch-sizes 1 2 4 \
  --context-sizes 128 512 \
  --warmup-iters 5 \
  --measure-iters 20 \
  --output-json /tmp/decode-cuda-graph-smoke.json
```

Expected: exit 0 on CUDA without model/offload arguments; JSON identifies `mode=fixture`, `capability_reason=eligible`, two registered/proved layers, the persistent storage owner, per-layer KV checksums, eager/replay raw samples, and equality checks; no performance threshold or claimed speedup.

Run real-loader capability smoke with valid required arguments:

```bash
python benchmarks/serving/decode_cuda_graph.py \
  --mode model \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /tmp/moe-offload \
  --batch-sizes 1 \
  --context-sizes 128 \
  --warmup-iters 1 \
  --measure-iters 2 \
  --output-json /tmp/decode-cuda-graph-model-capability.json
```

Expected: valid CLI; for an offloaded MoE runtime the JSON reports the explicit unsafe capability and eager-only samples with zero captures/replays.

- [x] **Step 5: Commit benchmark harness**

```bash
git add benchmarks/serving/decode_cuda_graph.py \
  benchmarks/serving/decode_cuda_graph_fixture.py \
  tests/python/serving/test_cuda_graph.py
git commit -m "bench(serving): compare eager and graph decode launches"
```

### Task 11: Document rollout, monitoring, limitations, and rollback

**Files:**
- Modify: `docs/serving.md`
- Modify: `docs/benchmarking.md`

- [x] **Step 1: Add operator documentation**

Add a `Decode CUDA graphs (experimental, opt-in)` section containing:

```text
Default: disabled.
Eligible: CUDA resident model; pure single-token decode; configured batch/context bucket;
explicit safe runtime capability; one shared PagedKVStorage; exact registered
ordinary-GQA Qwen3PagedAttention layers only; exact storage/runner/graph-buffer device equality;
unique valid layer_idx and allocation-free write-before-attention proof for every layer.
Always eager: non-paged models, prefill, speculative verify, FlashInfer plan/run path,
DeepseekV2PagedAttention and DeepseekV3PagedAttention MLA (`mla_layout_unsupported`),
active model hooks, Archer begin/end, transfer scheduler, expert dispatcher, expert/KV
offload, dynamic allocations, storage-owner mismatch, unknown paged class, invalid layer_idx,
missing per-layer KV-write proof, oversized shapes, capture failure.
Emergency rollback: export MOE_DISABLE_CUDA_GRAPHS=1.
```

Document all CLI flags, the sole supported Qwen3 class FQN, exact DeepSeek V2/V3 MLA rejection reason, generated per-layer Qwen3 bindings, write-before-attention semantics, one-owner KV persistence, exact device equality, padding, scratch reservation, lazy capture latency, reload/hot-replacement/application-shutdown cleanup, lock order, and Prometheus metrics. State that utility is limited to resident/native-paged ordinary-GQA Qwen3 models with complete write proofs, does not graph sampling, does not support DeepSeek MLA or offloaded MoE, and does not guarantee a speedup. Link future offloaded-MoE work to a separate piecewise design with explicit eager attention/routing/transfer boundaries.

- [x] **Step 2: Add staged rollout and acceptance checklist**

Document these stages exactly:

1. **Shadow qualification:** disabled in production; require exact ordinary-GQA Qwen3, `capability_reason=eligible`, complete `(class_fqn, layer_idx, writer)` proof coverage, allocator/backend owner identity, exact storage/runner/graph-buffer device equality, per-layer current-token persistence, two-step CUDA equivalence, shutdown/replacement cleanup, and paired fixture/model evidence for each resident Qwen3/GPU/dtype/bucket set.
2. **Canary:** enable on one resident/native-paged replica; alert on `capture_failures_total > 0`, any non-`eligible` capability reason after capture begins, request errors, owner mismatch, graph pool/scratch memory beyond budget, or replay coverage below the workload target.
3. **Limited rollout:** expand only when output parity remains clean, old-engine scratch blocks return to the authoritative allocator after replacement/shutdown, and observed p50/p99 ITL, throughput, GPU memory, and request concurrency are acceptable.
4. **General opt-in:** retain kill switch and per-model allowlist; DeepSeek V2/V3 MLA, non-paged, FlashInfer, and offloaded MoE stay unsupported regardless of benchmark results.
5. **Rollback:** set `MOE_DISABLE_CUDA_GRAPHS=1`, verify replay counter stops increasing and eager fallback counter increases, then restart without the enable flag if memory must be reclaimed immediately.

- [x] **Step 3: Add benchmark documentation and citation**

Document the exact valid `fixture` command and the `model` command with both `--model` and `--offload-dir`, required metadata, capability outcome, raw-result retention, no performance pass threshold, and the TensorRT-LLM motivation link. Explain that offloaded MoE model mode currently measures/records eager fallback rather than graph utility. Explain the capture-point tradeoff: denser buckets reduce padding but consume more graph memory and authoritative scratch KV capacity.

- [x] **Step 4: Validate documentation references**

Run:

```bash
python -m compileall -q moe_infinity benchmarks/serving/decode_cuda_graph.py \
  benchmarks/serving/decode_cuda_graph_fixture.py
pytest -q tests/python/serving
```

Expected: compile command exits 0; serving suite PASS with CUDA-only tests either PASS on CUDA or SKIP without CUDA.

- [x] **Step 5: Commit operator documentation**

```bash
git add docs/serving.md docs/benchmarking.md
git commit -m "docs(serving): add decode CUDA graph rollout guide"
```

### Task 12: Run final verification

**Files:**
- Verify only; no new files.

- [x] **Step 1: Run static diagnostics on every changed Python file**

Use the repository's configured language server on:

```text
moe_infinity/serving/model_runner.py
moe_infinity/runtime/paged_kv_storage.py
moe_infinity/runtime/attention_types.py
moe_infinity/runtime/attention_backend.py
moe_infinity/runtime/model_offload.py
moe_infinity/kernel/paged_kv_write.py
moe_infinity/models/paged_attention_registry.py
moe_infinity/models/qwen3_paged_attention.py
moe_infinity/serving/kv_cache.py
moe_infinity/serving/cuda_graph.py
moe_infinity/serving/engine.py
moe_infinity/serving/memory_manager.py
moe_infinity/entrypoints/openai/api_server_v2.py
moe_infinity/entrypoints/big_modeling.py
benchmarks/serving/decode_cuda_graph.py
benchmarks/serving/decode_cuda_graph_fixture.py
```

Expected: zero errors.

- [x] **Step 2: Run the complete serving test suite once**

```bash
pytest -q tests/python/serving \
  tests/python/serving/test_model_offload_capability.py \
  tests/python/integration/test_paged_attention_backend.py \
  tests/python/ops/test_paged_kv_write.py \
  tests/python/integration/test_flashinfer_model_attention.py
```

Expected: PASS; every capability reason, owner-identity assertion, shutdown/hot-replacement lock test, and benchmark CLI validation passes. CUDA-specific tests SKIP only when CUDA requirements are unavailable; FlashInfer tests prove eager fallback rather than graph eligibility.

- [x] **Step 3: Run one CUDA equivalence/benchmark qualification when a supported GPU is available**

```bash
pytest -q tests/python/serving/test_cuda_graph.py \
  tests/python/ops/test_paged_kv_write.py \
  -k 'capture_and_replay or persists or current_token'
python benchmarks/serving/decode_cuda_graph.py \
  --mode fixture \
  --batch-sizes 1 2 4 8 \
  --context-sizes 128 512 2048 \
  --warmup-iters 10 \
  --measure-iters 100 \
  --profile-launches \
  --output-json /tmp/decode-cuda-graph-qualification.json
```

Expected: every registered layer's current-token write and following-token persistence test PASS; benchmark exits 0 with two proved Qwen3 layers, per-layer KV checksums, and raw paired measurements. Record observations; do not convert them into a universal speedup claim.

- [x] **Step 4: Verify rollback behavior**

```bash
MOE_DISABLE_CUDA_GRAPHS=1 pytest -q tests/python/serving/test_cuda_graph.py \
  -k 'kill_switch or offloaded_moe or non_paged'
```

Expected: PASS, with no capture/replay attempted.

- [x] **Step 5: Commit any verification-only fixture corrections, if tests required them**

If no correction was needed, do not create an empty commit. If a fixture correction was required:

```bash
git add tests/python/serving benchmarks/serving/decode_cuda_graph.py \
  benchmarks/serving/decode_cuda_graph_fixture.py
git commit -m "test(serving): finalize CUDA graph qualification fixtures"
```
