import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest
import torch

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
KV_CACHE_PATH = Path(ROOT) / "moe_infinity" / "serving" / "kv_cache.py"
_MISSING_MODULE = object()


class BlockAllocatorProtocol(Protocol):
    @property
    def num_free_blocks(self) -> int: ...

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        device: torch.device,
    ) -> None: ...

    def allocate(self, num_blocks: int) -> list[int]: ...

    def free(self, block_ids: list[int]) -> None: ...


class PagedKVCacheProtocol(Protocol):
    block_allocator: BlockAllocatorProtocol

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None: ...

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> None: ...

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> None: ...

    def free_sequence(self, seq_id: int) -> None: ...

    def free_gpu_blocks(self, seq_id: int) -> None: ...

    def get_block_table(self, seq_id: int) -> list[int]: ...

    def swap_out(self, seq_id: int) -> None: ...

    def swap_in(self, seq_id: int) -> None: ...


def _load_classes() -> (
    tuple[
        type[BlockAllocatorProtocol],
        type[PagedKVCacheProtocol],
    ]
):
    module_name = "task4_kv_cache"
    spec = importlib.util.spec_from_file_location(module_name, KV_CACHE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {KV_CACHE_PATH}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name, _MISSING_MODULE)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_module is _MISSING_MODULE:
            _ = sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = cast(ModuleType, previous_module)
    return (
        cast(type[BlockAllocatorProtocol], getattr(module, "BlockAllocator")),
        cast(type[PagedKVCacheProtocol], getattr(module, "PagedKVCache")),
    )


def test_block_allocation_and_deallocation():
    BlockAllocator, _ = _load_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allocator = BlockAllocator(num_blocks=4, block_size=16, device=device)

    allocated = allocator.allocate(3)
    assert allocated == [0, 1, 2]
    assert allocator.num_free_blocks == 1

    allocator.free([1, 2])
    assert allocator.num_free_blocks == 3

    recycled = allocator.allocate(2)
    assert recycled == [1, 2]

    allocator.free([0, *recycled])
    assert allocator.num_free_blocks == 4


def test_block_pool_exhaustion():
    BlockAllocator, _ = _load_classes()
    allocator = BlockAllocator(
        num_blocks=2,
        block_size=8,
        device=torch.device("cpu"),
    )

    _ = allocator.allocate(2)
    with pytest.raises(RuntimeError):
        _ = allocator.allocate(1)


def test_sequence_lifecycle():
    _, PagedKVCache = _load_classes()
    cache = PagedKVCache(
        num_blocks=8,
        block_size=4,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )

    cache.allocate_sequence(seq_id=42, num_tokens=5)
    assert cache.get_block_table(42) == [0, 1]

    cache.swap_out(42)
    cache.swap_in(42)
    cache.free_sequence(42)

    assert cache.block_allocator.num_free_blocks == 8
    with pytest.raises(KeyError):
        _ = cache.get_block_table(42)


def test_append_tokens_across_block_boundary():
    _, PagedKVCache = _load_classes()
    cache = PagedKVCache(
        num_blocks=6,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )

    cache.allocate_sequence(seq_id=7, num_tokens=4)
    assert cache.get_block_table(7) == [0]

    cache.append_tokens(seq_id=7, num_new_tokens=1)
    assert cache.get_block_table(7) == [0, 1]


def test_free_sequence_returns_blocks():
    _, PagedKVCache = _load_classes()
    cache = PagedKVCache(
        num_blocks=8,
        block_size=4,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )

    cache.allocate_sequence(seq_id=1, num_tokens=7)
    cache.allocate_sequence(seq_id=2, num_tokens=1)
    assert cache.block_allocator.num_free_blocks == 5

    cache.free_sequence(1)
    assert cache.block_allocator.num_free_blocks == 7
    assert len(cache.get_block_table(2)) == 1

    cache.free_sequence(2)
    assert cache.block_allocator.num_free_blocks == 8


def test_free_gpu_blocks_releases_blocks():
    _, PagedKVCache = _load_classes()
    cache = PagedKVCache(
        num_blocks=4,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )

    cache.allocate_sequence(seq_id=1, num_tokens=8)
    assert cache.block_allocator.num_free_blocks == 2

    cache.free_gpu_blocks(1)
    assert cache.block_allocator.num_free_blocks == 4
    assert cache.get_block_table(1) == []


def test_swap_out_free_gpu_blocks_swap_in_round_trip():
    _, PagedKVCache = _load_classes()
    cache = PagedKVCache(
        num_blocks=4,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )

    cache.allocate_sequence(seq_id=1, num_tokens=8)
    assert cache.block_allocator.num_free_blocks == 2

    cache.swap_out(1)
    cache.free_gpu_blocks(1)
    assert cache.block_allocator.num_free_blocks == 4

    cache.allocate_sequence(seq_id=2, num_tokens=4)
    assert cache.block_allocator.num_free_blocks == 3

    cache.swap_in(1)
    assert cache.block_allocator.num_free_blocks == 1
    assert len(cache.get_block_table(1)) == 2

    cache.free_sequence(1)
    cache.free_sequence(2)
    assert cache.block_allocator.num_free_blocks == 4


from moe_infinity.runtime.attention_backend import (  # noqa: E402
    KVCacheSpec,
    PagedAttentionBackend,
)
from tests.python.serving.prefix_cache_test_utils import (  # noqa: E402
    RecordingLayeredPagedKVStore,
    make_cache,
)


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
    cache.swap_out(7)
    cache.free_gpu_blocks(7)
    cache.swap_in(7)
    assert (
        recording_layered_store.layer_values(cache.get_block_table(7)) == before
    )
    assert all(
        cache.block_allocator.ref_count(i) == 1
        for i in cache.get_block_table(7)
    )


def test_binding_uses_one_owner_and_disables_independent_storage() -> None:
    backend = PagedAttentionBackend(
        KVCacheSpec(2, 8, torch.float32, 4),
        num_gpu_blocks=8,
        device=torch.device("cpu"),
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
        num_gpu_blocks=8,
        device=torch.device("cpu"),
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
        num_gpu_blocks=4,
        device=torch.device("cpu"),
    )
    store = backend.create_layered_store(layer_count=3)
    with pytest.raises(ValueError, match="logical cache exceeds"):
        make_cache(num_blocks=6).set_block_store(store, owner=backend)
