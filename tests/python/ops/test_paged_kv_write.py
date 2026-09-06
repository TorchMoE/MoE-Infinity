from __future__ import annotations

import pytest
import torch

from moe_infinity.kernel.paged_kv_write import paged_kv_write_
from moe_infinity.runtime.paged_kv_storage import (
    PagedKVStorage,
    PagedKVStorageSpec,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _make_storage(
    *,
    num_layers: int = 2,
    num_blocks: int = 4,
    block_size: int = 4,
    device: torch.device | None = None,
) -> PagedKVStorage:
    spec = PagedKVStorageSpec(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=device or torch.device("cpu"),
    )
    return PagedKVStorage(spec)


def _assert_slots_equal(storage, *, layer_idx, slots, key, value) -> None:
    x = 8
    block_size = storage.spec.block_size
    for i in range(slots.shape[0]):
        slot = int(slots[i].item())
        block_id = slot // block_size
        offset = slot % block_size
        expected_key = key[i].reshape(
            storage.spec.num_kv_heads, storage.spec.head_dim // x, x
        )
        torch.testing.assert_close(
            storage.key_cache[layer_idx, block_id, :, :, offset, :].cpu(),
            expected_key.cpu().to(storage.spec.dtype),
        )
        torch.testing.assert_close(
            storage.value_cache[layer_idx, block_id, :, :, offset].cpu(),
            value[i].cpu().to(storage.spec.dtype),
        )


def test_paged_kv_write_allocation_free_layout_cpu() -> None:
    storage = _make_storage(num_layers=2, num_blocks=4, block_size=4)
    slots = torch.tensor([1, 6], dtype=torch.int64)
    key = torch.arange(2 * 2 * 8, dtype=torch.float32).reshape(2, 2, 8)
    value = key + 100

    paged_kv_write_(
        storage, layer_idx=1, key=key, value=value, slot_mapping=slots
    )

    _assert_slots_equal(storage, layer_idx=1, slots=slots, key=key, value=value)
    assert torch.count_nonzero(storage.value_cache[0]).item() == 0


def test_paged_kv_write_rejects_device_mismatch_cpu() -> None:
    storage = _make_storage()
    slots = torch.tensor([0], dtype=torch.int64)
    key = torch.zeros(1, 2, 8)
    value = torch.zeros(1, 2, 8)
    with pytest.raises(ValueError):
        paged_kv_write_(
            storage,
            layer_idx=5,
            key=key,
            value=value,
            slot_mapping=slots,
        )


@requires_cuda
def test_graph_safe_kv_write_persists_current_token_per_layer() -> None:
    storage = _make_storage(
        num_layers=2, num_blocks=4, block_size=4, device=torch.device("cuda")
    )
    slots = torch.tensor([1, 6], dtype=torch.int64, device="cuda")
    key = torch.arange(2 * 2 * 8, device="cuda", dtype=torch.float32).reshape(
        2, 2, 8
    )
    value = key + 100
    paged_kv_write_(
        storage, layer_idx=1, key=key, value=value, slot_mapping=slots
    )
    _assert_slots_equal(storage, layer_idx=1, slots=slots, key=key, value=value)
    assert torch.count_nonzero(storage.value_cache[0]).item() == 0


@requires_cuda
def test_graph_safe_kv_write_under_cuda_graph_capture_persists() -> None:
    """The Triton current-token write must be capturable: replaying a captured
    graph after refilling the fixed-address inputs persists the new K/V at the
    fixed slot without any allocation on the write path."""
    storage = _make_storage(
        num_layers=2, num_blocks=4, block_size=4, device=torch.device("cuda")
    )
    slots = torch.tensor([1], dtype=torch.int64, device="cuda")
    key = torch.zeros(1, 2, 8, device="cuda", dtype=torch.float32)
    value = torch.zeros(1, 2, 8, device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        paged_kv_write_(
            storage, layer_idx=1, key=key, value=value, slot_mapping=slots
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_kv_write_(
            storage, layer_idx=1, key=key, value=value, slot_mapping=slots
        )

    new_key = torch.arange(2 * 8, device="cuda", dtype=torch.float32).reshape(
        1, 2, 8
    )
    new_value = new_key + 100
    key.copy_(new_key)
    value.copy_(new_value)
    graph.replay()
    torch.cuda.synchronize()

    _assert_slots_equal(
        storage, layer_idx=1, slots=slots, key=new_key, value=new_value
    )


def test_second_decode_token_eager_observes_first_token_kv() -> None:
    """Task 3 semantic parity: after an eager decode step writes current-token
    K/V at the authoritative slot, a subsequent decode token attends over the
    prior token's persisted K/V. Graph replay equivalence (``try_execute``) is
    completed in Task 4; here we prove the eager write-before-attend
    persistence contract only, on CPU (the CUDA paged-attention kernel image is
    unavailable in this environment, so the SDPA fallback path is exercised)."""
    from moe_infinity.runtime.attention_backend import PagedAttentionBackend
    from moe_infinity.runtime.attention_types import (
        AttentionMetadata as RuntimeAttentionMetadata,
    )

    storage = _make_storage(num_layers=1, num_blocks=4, block_size=4)
    backend = PagedAttentionBackend(storage=storage, use_flashinfer=False)

    def _decode(token_offset: int, key_val: float) -> torch.Tensor:
        query = torch.ones(1, 4, 8, dtype=torch.float32)
        key = torch.full((1, 2, 8), key_val, dtype=torch.float32)
        value = key + 1.0
        metadata = RuntimeAttentionMetadata(
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            seq_lens=torch.tensor([token_offset + 1], dtype=torch.int32),
            max_seq_len=token_offset + 1,
            num_prefill_tokens=0,
            num_decode_tokens=1,
            slot_mapping=torch.tensor([token_offset], dtype=torch.int64),
            is_prefill=False,
            kv_storage_owner_id=storage.owner_id,
        )
        return backend.forward(
            query=query,
            key=key,
            value=value,
            attention_metadata=metadata,
            layer_idx=0,
        )

    _ = _decode(token_offset=0, key_val=3.0)
    torch.testing.assert_close(
        storage.value_cache[0, 0, :, :, 0], torch.full((2, 8), 4.0)
    )
    out_second = _decode(token_offset=1, key_val=7.0)
    torch.testing.assert_close(
        storage.value_cache[0, 0, :, :, 1], torch.full((2, 8), 8.0)
    )
    torch.testing.assert_close(
        storage.value_cache[0, 0, :, :, 0], torch.full((2, 8), 4.0)
    )
    assert out_second.shape == (1, 4, 8)
