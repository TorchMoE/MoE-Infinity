import math

import pytest
import torch

from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import AttentionMetadata, KVCacheSpec
from moe_infinity.runtime.kv_cache_format import allocate_layered_paged_kv_store


def _native_backend(num_kv_heads, head_dim, block_size, num_blocks, dtype):
    spec = KVCacheSpec(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        block_size=block_size,
    )
    return PagedAttentionBackend(spec, num_blocks, torch.device("cpu"))


def _int8_backend(num_kv_heads, head_dim, block_size, num_blocks, dtype):
    spec = KVCacheSpec(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        block_size=block_size,
        format_name="int8_sym",
    )
    backend = PagedAttentionBackend(spec, num_blocks, torch.device("cpu"))
    store = allocate_layered_paged_kv_store(
        owner_id="quality-int8",
        format_name="int8_sym",
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        execution_dtype=dtype,
        device=torch.device("cpu"),
    )
    backend.bind_store(store, owner_id="quality-int8")
    return backend


def _prefill_meta(num_tokens, block_table):
    return AttentionMetadata(
        block_tables=block_table,
        seq_lens=torch.tensor([num_tokens], dtype=torch.int64),
        max_seq_len=num_tokens,
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        slot_mapping=torch.arange(num_tokens, dtype=torch.long),
        is_prefill=True,
    )


def _decode_meta(seq_len, block_table):
    return AttentionMetadata(
        block_tables=block_table,
        seq_lens=torch.tensor([seq_len], dtype=torch.int64),
        max_seq_len=seq_len,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([seq_len - 1], dtype=torch.long),
        is_prefill=False,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (32, 8)])
@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 257])
def test_int8_decode_matches_native_reference(
    dtype, q_heads, kv_heads, seq_len
):
    torch.manual_seed(seq_len * 100 + q_heads)
    head_dim = 16
    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size + 1
    block_table = torch.arange(num_blocks, dtype=torch.int64).reshape(1, -1)

    key = torch.randn(seq_len, kv_heads, head_dim, dtype=dtype)
    value = torch.randn(seq_len, kv_heads, head_dim, dtype=dtype)
    query = torch.randn(1, q_heads, head_dim, dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)

    int8 = _int8_backend(kv_heads, head_dim, block_size, num_blocks, dtype)
    int8.write_chunk(
        layer_idx=0,
        key_chunk=key,
        value_chunk=value,
        slot_mapping=torch.arange(seq_len),
    )
    int8_out = int8.forward(
        query,
        query,
        query,
        layer_idx=0,
        attention_metadata=_decode_meta(seq_len, block_table),
        scale=scale,
    )

    k32 = key.float().permute(1, 0, 2)
    v32 = value.float().permute(1, 0, 2)
    repeat = q_heads // kv_heads
    if repeat > 1:
        k32 = k32.repeat_interleave(repeat, dim=0)
        v32 = v32.repeat_interleave(repeat, dim=0)
    q32 = query[0].float().unsqueeze(0).unsqueeze(2)
    native_out = torch.nn.functional.scaled_dot_product_attention(
        q32, k32.unsqueeze(0), v32.unsqueeze(0), scale=scale, is_causal=False
    )[0, :, 0, :]

    int8_flat = int8_out[0].float()
    torch.testing.assert_close(int8_flat, native_out, atol=2e-2, rtol=2e-2)
    cosine = torch.nn.functional.cosine_similarity(
        int8_flat.flatten(), native_out.flatten(), dim=0
    )
    assert float(cosine) >= 0.999


def test_int8_zero_vectors_reconstruct_exactly():
    head_dim, block_size, kv_heads, seq_len = 16, 16, 2, 8
    num_blocks = 2
    backend = _int8_backend(
        kv_heads, head_dim, block_size, num_blocks, torch.float16
    )
    key = torch.zeros(seq_len, kv_heads, head_dim, dtype=torch.float16)
    value = torch.zeros(seq_len, kv_heads, head_dim, dtype=torch.float16)
    backend.write_chunk(
        layer_idx=0,
        key_chunk=key,
        value_chunk=value,
        slot_mapping=torch.arange(seq_len),
    )
    k_hat, v_hat = backend.store.read_prefix(
        layer_idx=0,
        block_table=torch.tensor([0, 1]),
        seq_len=seq_len,
        execution_dtype=torch.float32,
    )
    assert torch.equal(k_hat, torch.zeros_like(k_hat))
    assert torch.equal(v_hat, torch.zeros_like(v_hat))


def test_int8_non_contiguous_block_table_decode():
    head_dim, block_size, kv_heads = 16, 4, 2
    q_heads, seq_len = 2, 8
    num_blocks = 6
    backend = _int8_backend(
        kv_heads, head_dim, block_size, num_blocks, torch.float16
    )
    key = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.float16)
    value = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.float16)
    slots = torch.tensor([0, 1, 2, 3, 20, 21, 22, 23])
    backend.write_chunk(
        layer_idx=0, key_chunk=key, value_chunk=value, slot_mapping=slots
    )
    block_table = torch.tensor([[0, 5]], dtype=torch.int64)
    query = torch.randn(1, q_heads, head_dim, dtype=torch.float16)
    out = backend.forward(
        query,
        query,
        query,
        layer_idx=0,
        attention_metadata=_decode_meta(seq_len, block_table),
        scale=1.0 / math.sqrt(head_dim),
    )
    assert out.shape == (1, q_heads, head_dim)
    assert torch.all(torch.isfinite(out))
