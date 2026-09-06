import torch

from moe_infinity.runtime.attention_backend import (
    AttentionBackend,
    PagedAttentionBackend,
)
from moe_infinity.runtime.attention_types import (
    AttentionMetadata,
    KVCacheSpec,
    PagedBatchLengths,
)


def _make_backend() -> PagedAttentionBackend:
    spec = KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        block_size=4,
    )
    return PagedAttentionBackend(
        spec=spec,
        num_gpu_blocks=10,
        device=torch.device("cpu"),
    )


def test_write_kv_and_read_back() -> None:
    backend = _make_backend()

    key = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
    value = key + 1000.0
    slot_mapping = torch.tensor([0, 5, 7], dtype=torch.long)

    backend.write_kv(key, value, slot_mapping)

    for i in range(slot_mapping.shape[0]):
        slot = int(slot_mapping[i].item())
        block_id = slot // backend.spec.block_size
        token_offset = slot % backend.spec.block_size
        k_cached = backend.k_cache[block_id, :, :, token_offset, :].reshape(
            2, 8
        )
        v_cached = backend.v_cache[block_id, :, :, token_offset]
        torch.testing.assert_close(k_cached, key[i])
        torch.testing.assert_close(v_cached, value[i])


def test_prefill_forward() -> None:
    backend = _make_backend()

    query = torch.randn(4, 4, 8)
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)

    metadata = AttentionMetadata(
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([4], dtype=torch.int32),
            query_offsets=torch.tensor([0, 4], dtype=torch.int32),
            context_lengths=torch.tensor([0], dtype=torch.int32),
            kv_seq_lengths=torch.tensor([4], dtype=torch.int32),
        ),
        max_seq_len=4,
        num_prefill_tokens=4,
        num_decode_tokens=0,
        slot_mapping=torch.arange(4, dtype=torch.long),
        is_prefill=True,
    )

    out = backend.forward(
        query=query,
        key=key,
        value=value,
        attention_metadata=metadata,
    )
    assert isinstance(backend, AttentionBackend)
    assert out.shape == (4, 4, 8)


def test_decode_forward_cpu() -> None:
    backend = _make_backend()

    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    backend.write_kv(key, value, torch.arange(4, dtype=torch.long))

    query = torch.randn(1, 4, 8)
    metadata = AttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([1], dtype=torch.int32),
            query_offsets=torch.tensor([0, 1], dtype=torch.int32),
            context_lengths=torch.tensor([3], dtype=torch.int32),
            kv_seq_lengths=torch.tensor([4], dtype=torch.int32),
        ),
        max_seq_len=4,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([3], dtype=torch.long),
        is_prefill=False,
    )

    out = backend.forward(
        query=query,
        key=key[:1],
        value=value[:1],
        attention_metadata=metadata,
    )
    assert out.shape == (1, 4, 8)


def test_kv_cache_shape() -> None:
    spec = KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        block_size=4,
    )
    k_shape, v_shape = PagedAttentionBackend.get_kv_cache_shape(
        spec=spec,
        num_gpu_blocks=10,
    )
    assert k_shape == (10, 2, 1, 4, 8)
    assert v_shape == (10, 2, 8, 4)
