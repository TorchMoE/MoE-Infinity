from __future__ import annotations

import dataclasses
from collections.abc import Callable
from types import SimpleNamespace

import torch

from tests.python.serving._prefix_bootstrap import (
    BatchMetadata,
    CacheNamespace,
    ContinuousBatchingEngine,
    KVCacheSpec,
    LayeredPagedKVPayload,
    LayeredPagedKVStore,
    ModelRunner,
    PagedAttentionBackend,
    PagedBatchLengths,
    PagedKVCache,
    PrefixCache,
    PrefixLease,
    PrefixMatch,
    PrefixReuseCapability,
    SamplingParams,
    Scheduler,
    SequenceAllocationPlan,
    SequenceData,
    SequenceGroup,
    SequenceStatus,
)

SHARED = list(range(8))

__all__ = [
    "SHARED",
    "BatchMetadata",
    "CacheNamespace",
    "ContinuousBatchingEngine",
    "ModelRunner",
    "PagedBatchLengths",
    "PagedKVCache",
    "PrefixCache",
    "PrefixLease",
    "PrefixMatch",
    "PrefixReuseCapability",
    "RecordingLayeredPagedKVStore",
    "RecordingPrefixLeaseProvider",
    "RefRecorder",
    "SamplingParams",
    "Scheduler",
    "SequenceAllocationPlan",
    "SequenceData",
    "SequenceGroup",
    "SequenceStatus",
    "add_prefill",
    "bind_prefix_runtime",
    "make_cache",
    "make_dflash_engine_and_batch",
    "make_group",
    "make_namespace",
    "make_paged_backend",
    "make_prefill_batch",
    "make_prefix_capable_engine",
    "make_qwen_runner",
    "make_seeded_scheduler",
]


def make_namespace(**changes: object) -> CacheNamespace:
    base = CacheNamespace(
        model_id="Qwen/Qwen3-30B-A3B",
        model_revision="rev-a",
        tokenizer_id="Qwen/Qwen3-30B-A3B",
        tokenizer_revision="rev-a",
        tokenizer_config_digest="tok-digest",
        adapter_id=None,
        adapter_revision=None,
        dtype="bfloat16",
        block_size=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        attention_backend="flashinfer-paged",
        attention_layout="NHD",
        position_config_digest="rope-default;window=none",
        runtime_epoch="epoch-1",
    )
    return dataclasses.replace(base, **changes)


class RefRecorder:
    def __init__(self) -> None:
        self.retained: list[list[int]] = []
        self.released: list[list[int]] = []

    def retain(self, ids: list[int]) -> None:
        self.retained.append(list(ids))

    def release(self, ids: list[int]) -> None:
        self.released.append(list(ids))


class RecordingLayeredPagedKVStore(LayeredPagedKVStore):
    def __init__(
        self,
        num_layers: int = 3,
        num_blocks: int = 8,
        block_size: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 8,
        dtype: torch.dtype = torch.float32,
        owner: object | None = None,
        device: torch.device | None = None,
    ) -> None:
        owner = owner or SimpleNamespace()
        super().__init__(
            owner=owner,
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device or torch.device("cpu"),
            use_flashinfer=True,
        )
        setattr(owner, "block_store", self)
        self.copies: list[tuple[int, int, tuple[int, ...]]] = []

    def import_blocks(
        self, block_ids: list[int], payload: LayeredPagedKVPayload
    ) -> None:
        super().import_blocks(block_ids, payload)
        if len(block_ids) == 1 and len(payload.source_block_ids) == 1:
            self.copies.append(
                (
                    payload.source_block_ids[0],
                    block_ids[0],
                    tuple(range(self.num_layers)),
                )
            )

    def layer_values(self, block_ids: list[int]) -> list[list[float]]:
        return self.fi_kv_cache[:, block_ids, 0, 0, 0, 0].tolist()


def make_cache(
    store: RecordingLayeredPagedKVStore | None = None, num_blocks: int = 8
) -> PagedKVCache:
    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=3,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    if store is not None:
        cache.set_block_store(store, owner=store.owner)
    return cache


def make_paged_backend(num_blocks: int) -> PagedAttentionBackend:
    return PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            block_size=4,
        ),
        num_gpu_blocks=num_blocks,
        device=torch.device("cpu"),
    )


def make_qwen_runner(
    layer_indices: list[int], expected_layers: int
) -> ModelRunner:
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
        config=SimpleNamespace(
            num_hidden_layers=expected_layers, sliding_window=None
        ),
    )
    backend = SimpleNamespace()
    store = RecordingLayeredPagedKVStore(
        num_layers=expected_layers, owner=backend
    )
    backend.num_gpu_blocks = store.num_blocks
    backend.spec = SimpleNamespace(
        block_size=store.block_size,
        num_kv_heads=store.num_kv_heads,
        head_dim=store.head_dim,
        dtype=store.dtype,
    )
    backend.register_layers = lambda registrations: None
    backend.create_layered_store = lambda *, layer_count: store
    backend._flashinfer_enabled = lambda: True
    owner = SimpleNamespace(
        get_attention_backend=lambda: backend, expert_layer_modules=[]
    )
    return ModelRunner(model, owner, device=torch.device("cpu"))


class RecordingPrefixLeaseProvider:
    def __init__(self, cache: PrefixCache) -> None:
        self.cache = cache
        self.events: list[str] = []

    @property
    def open_leases(self) -> int:
        return self.cache.open_leases

    def acquire_prefix_lease(
        self,
        namespace: CacheNamespace,
        token_ids: list[int],
        max_prefix_tokens: int,
    ) -> PrefixLease:
        self.events.append(f"pin:{token_ids[-1]}")
        return self.cache.acquire_prefix_lease(
            namespace, token_ids, max_prefix_tokens
        )

    def evict_until(self, predicate: Callable[[], bool]) -> None:
        self.events.append("evict")
        self.cache.evict_until(predicate)


def make_group(
    request_id: str, rows: list[tuple[int, list[int]]]
) -> SequenceGroup:
    return SequenceGroup(
        request_id=request_id,
        sequences=[
            SequenceData(
                seq_id=seq_id,
                prompt_token_ids=tokens,
                sampling_params=SamplingParams(),
            )
            for seq_id, tokens in rows
        ],
    )


def make_seeded_scheduler(
    *, num_blocks: int, max_batch_size: int
) -> tuple[
    Scheduler, PagedKVCache, RecordingPrefixLeaseProvider, CacheNamespace
]:
    store = RecordingLayeredPagedKVStore(num_blocks=num_blocks)
    cache = make_cache(store=store, num_blocks=num_blocks)
    namespace = make_namespace(num_layers=3)
    prefix = PrefixCache(
        block_size=4,
        max_entries=8,
        on_retain=cache.block_allocator.retain,
        on_release=cache.block_allocator.release,
    )
    cache.allocate_sequence(999, 9)
    seed_blocks = cache.get_block_table(999)
    prefix.insert(namespace, SHARED + [99], seed_blocks, committed_tokens=8)
    cache.free_sequence(999)
    provider = RecordingPrefixLeaseProvider(prefix)
    scheduler = Scheduler(
        cache,
        max_batch_size=max_batch_size,
        max_tokens_per_step=64,
        prefix_lease_provider=provider,
        cache_namespace=namespace,
    )
    return scheduler, cache, provider, namespace


def make_prefill_batch(
    sequence: SequenceData,
    context_len: int,
    query_tokens: list[int],
    block_table: list[int],
) -> BatchMetadata:
    return BatchMetadata(
        seq_ids=[sequence.seq_id],
        input_token_ids=list(query_tokens),
        lengths=PagedBatchLengths(
            [len(query_tokens)],
            [0, len(query_tokens)],
            [context_len],
            [context_len + len(query_tokens)],
        ),
        is_prefill=[True],
        block_tables=[list(block_table)],
        sampling_params=[sequence.sampling_params],
    )


def bind_prefix_runtime(
    engine: ContinuousBatchingEngine,
) -> RecordingLayeredPagedKVStore:
    cache = engine.kv_cache
    store = RecordingLayeredPagedKVStore(
        num_layers=cache.num_layers,
        num_blocks=cache.num_blocks,
        block_size=cache.block_size,
        num_kv_heads=cache.num_heads,
        head_dim=cache.head_dim,
        dtype=cache.dtype,
        device=cache.device,
    )
    owner = store.owner
    cache.set_block_store(store, owner=owner)
    namespace = make_namespace(
        num_layers=cache.num_layers,
        num_kv_heads=cache.num_heads,
        head_dim=cache.head_dim,
        dtype=str(cache.dtype).removeprefix("torch."),
    )
    prefix = PrefixCache(
        cache.block_size,
        32,
        on_retain=cache.block_allocator.retain,
        on_release=cache.block_allocator.release,
    )
    engine.prefix_cache, engine.cache_namespace = prefix, namespace
    engine.scheduler.prefix_lease_provider = prefix
    engine.scheduler.cache_namespace = namespace
    return store


def make_prefix_capable_engine(cb_engine_factory) -> ContinuousBatchingEngine:
    engine = cb_engine_factory(
        config_overrides={"enable_prefix_caching": False}
    )
    bind_prefix_runtime(engine)
    return engine


def add_prefill(
    engine: ContinuousBatchingEngine, prompt: list[int], committed: int
) -> SequenceData:
    request_id = f"synthetic-{len(engine._request_to_seq_ids)}"
    engine.add_request(
        request_id, prompt, SamplingParams(temperature=0.0, max_tokens=1)
    )
    seq_id = engine._request_to_seq_ids[request_id][0]
    sequence = engine._sequences[seq_id]
    sequence.set_status(SequenceStatus.PREFILL)
    sequence.num_computed_tokens = committed
    sequence.committed_kv_tokens = committed
    engine.kv_cache.allocate_sequence(seq_id, len(prompt))
    if committed:
        engine.prefix_cache.insert(
            engine.cache_namespace,
            prompt,
            engine.kv_cache.get_block_table(seq_id),
            committed_tokens=committed,
        )
    return sequence


def make_dflash_engine_and_batch(
    cb_engine_factory, context_len: int, has_prefix_lease: bool
):
    engine = cb_engine_factory()
    engine.speculative_draft = object()
    sequence = SequenceData(
        seq_id=77,
        prompt_token_ids=list(range(context_len + 1)),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        status=SequenceStatus.PREFILL,
        num_computed_tokens=context_len,
    )
    sequence.has_prefix_lease = has_prefix_lease
    batch = make_prefill_batch(
        sequence, context_len, [context_len], [0] * ((context_len + 4) // 4)
    )
    return engine, batch
