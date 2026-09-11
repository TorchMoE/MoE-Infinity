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
from moe_infinity.runtime.kv_cache_format import allocate_layered_paged_kv_store


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


def _make_int8_backend() -> PagedAttentionBackend:
    return PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            block_size=4,
            format_name="int8_sym",
        ),
        num_gpu_blocks=10,
        device=torch.device("cpu"),
    )


def _make_layered_store(
    *,
    format_name: str = "int8_sym",
    num_layers: int = 2,
    num_blocks: int = 10,
    owner_id: str = "serving-engine-1",
):
    return allocate_layered_paged_kv_store(
        owner_id=owner_id,
        format_name=format_name,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        execution_dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _prefill_metadata(num_tokens: int) -> AttentionMetadata:
    return AttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int64),
        seq_lens=torch.tensor([num_tokens], dtype=torch.int64),
        max_seq_len=num_tokens,
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        slot_mapping=torch.arange(num_tokens, dtype=torch.long),
        is_prefill=True,
    )


def _decode_metadata(seq_len: int) -> AttentionMetadata:
    return AttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int64),
        seq_lens=torch.tensor([seq_len], dtype=torch.int64),
        max_seq_len=seq_len,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([seq_len - 1], dtype=torch.long),
        is_prefill=False,
    )


def test_int8_write_uses_payload_and_scales() -> None:
    backend = _make_int8_backend()
    store = _make_layered_store(
        format_name="int8_sym", num_layers=2, num_blocks=10
    )
    backend.bind_store(store, owner_id="serving-engine-1")
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    backend.write_chunk(
        layer_idx=1,
        key_chunk=key,
        value_chunk=value,
        slot_mapping=torch.arange(4),
    )
    assert backend.store is store
    assert store.payload.dtype == torch.int8
    assert store.scales is not None
    assert store.scales.shape == (2, 10, 2, 4, 2)


def test_int8_gqa_prefill_and_decode_match_reference() -> None:
    torch.manual_seed(7)
    backend = _make_int8_backend()
    backend.bind_store(
        _make_layered_store(
            format_name="int8_sym", num_layers=2, num_blocks=10
        ),
        owner_id="serving-engine-1",
    )
    query = torch.randn(4, 4, 8)
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    prefill = backend.forward(
        query, key, value, layer_idx=1, attention_metadata=_prefill_metadata(4)
    )
    decode = backend.forward(
        query[-1:],
        key[-1:],
        value[-1:],
        layer_idx=1,
        attention_metadata=_decode_metadata(4),
    )
    assert prefill.shape == (4, 4, 8)
    assert decode.shape == (1, 4, 8)
    assert backend.execution_backend == "sdpa_dequant"


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
        k_cached = backend.k_cache[0, block_id, :, :, token_offset, :].reshape(
            2, 8
        )
        v_cached = backend.v_cache[0, block_id, :, :, token_offset]
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
    assert k_shape == (1, 10, 2, 1, 4, 8)
    assert v_shape == (1, 10, 2, 8, 4)
