from __future__ import annotations

import pytest
import torch

from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)
from moe_infinity.runtime.paged_kv_storage import (
    PagedKVStorage,
    PagedKVStorageSpec,
)
from moe_infinity.serving.kv_cache import PagedKVCache


def _make_storage(
    *,
    num_layers: int = 2,
    num_blocks: int = 16,
    block_size: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    device: torch.device | None = None,
) -> PagedKVStorage:
    spec = PagedKVStorageSpec(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device=device or torch.device("cpu"),
    )
    return PagedKVStorage(spec)


def _metadata(*, owner_id: str) -> RuntimeAttentionMetadata:
    return RuntimeAttentionMetadata(
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        is_prefill=False,
        kv_storage_owner_id=owner_id,
    )


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


def test_paged_kv_cache_legacy_constructor_remains_supported_but_unbound() -> (
    None
):
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

    storage.write_kv(
        layer_idx=1,
        key=key,
        value=value,
        slot_mapping=torch.tensor([slot]),
    )

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


def test_reserve_and_release_scratch_round_trip() -> None:
    storage = _make_storage(num_blocks=8)
    scratch = storage.reserve_graph_scratch_blocks(3)
    assert storage.num_graph_scratch_blocks == 3
    assert storage.block_allocator.num_free_blocks == 5
    storage.release_graph_scratch_blocks(scratch)
    assert storage.num_graph_scratch_blocks == 0
    assert storage.block_allocator.num_free_blocks == 8


def test_release_unreserved_scratch_raises() -> None:
    storage = _make_storage(num_blocks=8)
    with pytest.raises(ValueError, match="not reserved"):
        storage.release_graph_scratch_blocks([0])
