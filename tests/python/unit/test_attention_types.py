import torch

from moe_infinity.runtime.attention_types import (
    AttentionMetadata,
    KVCacheSpec,
    PagedBatchLengths,
)


def test_page_size_bytes_mha():
    spec = KVCacheSpec(
        num_kv_heads=8, head_dim=128, dtype=torch.float16, block_size=16
    )
    assert spec.page_size_bytes == 65536


def test_page_size_bytes_gqa():
    spec = KVCacheSpec(
        num_kv_heads=2, head_dim=64, dtype=torch.float16, block_size=16
    )
    assert spec.page_size_bytes == 8192


def test_page_size_bytes_mla_deepseek_v2():
    spec = KVCacheSpec(
        num_kv_heads=1, head_dim=512, dtype=torch.float16, block_size=16
    )
    assert spec.page_size_bytes == 32768


def test_page_size_bytes_bf16_matches_fp16():
    fp16_spec = KVCacheSpec(
        num_kv_heads=4, head_dim=32, dtype=torch.float16, block_size=8
    )
    bf16_spec = KVCacheSpec(
        num_kv_heads=4, head_dim=32, dtype=torch.bfloat16, block_size=8
    )
    assert fp16_spec.page_size_bytes == bf16_spec.page_size_bytes
    assert fp16_spec.page_size_bytes == 4096


def test_attention_metadata_creation():
    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    metadata = AttentionMetadata(
        block_tables=block_tables,
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([3, 1], dtype=torch.int32),
            query_offsets=torch.tensor([0, 3, 4], dtype=torch.int32),
            context_lengths=torch.tensor([13, 7], dtype=torch.int32),
            kv_seq_lengths=torch.tensor([16, 8], dtype=torch.int32),
        ),
        max_seq_len=16,
        num_prefill_tokens=3,
        num_decode_tokens=1,
        slot_mapping=slot_mapping,
        is_prefill=True,
    )

    assert metadata.block_tables.dtype == torch.int32
    assert metadata.lengths.kv_seq_lengths.dtype == torch.int32
    assert metadata.slot_mapping.dtype == torch.int64
    assert metadata.is_prefill is True
