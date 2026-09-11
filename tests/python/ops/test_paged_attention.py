import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip(
    "moe_infinity.kernel.paged_attention_ops",
    reason="paged_attention_ops required",
)

from moe_infinity.kernel.paged_attention_ops import paged_attention_fwd
from moe_infinity.runtime.kv_cache_format import quantize_tokenwise_symmetric


def make_paged_kv(
    batch: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = (seq_len + block_size - 1) // block_size * batch
    x = max(1, 16 // (head_dim * 2))
    k_cache = torch.randn(num_blocks, num_heads, head_dim // x, block_size, x)
    v_cache = torch.randn(num_blocks, num_heads, head_dim, block_size)
    block_tables = torch.arange(num_blocks).reshape(batch, -1).int()
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)
    return k_cache, v_cache, block_tables, seq_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_paged_vs_sdpa():
    batch, num_heads, head_dim, seq_len = 2, 8, 64, 32
    block_size = 16
    query = torch.randn(
        batch,
        num_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    k_cache, v_cache, block_tables, seq_lens = make_paged_kv(
        batch, num_heads, head_dim, seq_len, block_size
    )
    k_cache, v_cache = k_cache.cuda().half(), v_cache.cuda().half()
    scale = 1.0 / math.sqrt(float(head_dim))
    out = paged_attention_fwd(
        query,
        k_cache,
        v_cache,
        block_tables.cuda(),
        seq_lens.cuda(),
        scale,
        num_heads,
        block_size,
    )
    assert out.shape == query.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gqa_config():
    batch, num_heads, num_kv_heads, head_dim, seq_len = 2, 32, 8, 128, 16
    block_size = 16
    query = torch.randn(
        batch,
        num_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    )
    k_cache, v_cache, block_tables, seq_lens = make_paged_kv(
        batch, num_kv_heads, head_dim, seq_len, block_size
    )
    k_cache, v_cache = k_cache.cuda().half(), v_cache.cuda().half()
    scale = 1.0 / math.sqrt(float(head_dim))
    out = paged_attention_fwd(
        query,
        k_cache,
        v_cache,
        block_tables.cuda(),
        seq_lens.cuda(),
        scale,
        num_kv_heads,
        block_size,
    )
    assert out.shape == query.shape


@pytest.mark.gpu
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (32, 8)])
def test_int8_paged_attention_matches_dequantized_sdpa(num_heads, num_kv_heads):
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from moe_infinity.kernel.paged_attention_ops import (
        probe_native_int8_binding,
    )

    available, _ = probe_native_int8_binding()
    if not available:
        pytest.skip("native paged_attention_int8_v1 binding unavailable")
    torch.manual_seed(19)
    batch, head_dim, seq_len, block_size = 2, 128, 33, 16
    query = torch.randn(
        batch, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k, v, block_tables, seq_lens = make_paged_kv(
        batch, num_kv_heads, head_dim, seq_len, block_size
    )
    k = k.cuda().half()
    v = v.cuda().half()
    k_vectors = k.permute(0, 3, 1, 2, 4).reshape(-1, num_kv_heads, head_dim)
    v_vectors = v.permute(0, 3, 1, 2).reshape(-1, num_kv_heads, head_dim)
    k_q_vectors, k_scale_vectors = quantize_tokenwise_symmetric(k_vectors)
    v_q_vectors, v_scale_vectors = quantize_tokenwise_symmetric(v_vectors)
    num_blocks = k.shape[0]
    k_q = (
        k_q_vectors.reshape(
            num_blocks, block_size, num_kv_heads, head_dim // 8, 8
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_q = v_q_vectors.reshape(
        num_blocks, block_size, num_kv_heads, head_dim
    ).permute(0, 2, 3, 1)
    k_scale = (
        k_scale_vectors.reshape(num_blocks, block_size, num_kv_heads)
        .permute(0, 2, 1)
        .contiguous()
    )
    v_scale = (
        v_scale_vectors.reshape(num_blocks, block_size, num_kv_heads)
        .permute(0, 2, 1)
        .contiguous()
    )
    actual = paged_attention_fwd(
        query,
        k_q,
        v_q,
        block_tables.cuda(),
        seq_lens.cuda(),
        1.0 / math.sqrt(head_dim),
        num_kv_heads,
        block_size,
        key_scale=k_scale,
        value_scale=v_scale,
    )
    references = []
    for seq_idx in range(batch):
        logical_len = int(seq_lens[seq_idx])
        page_count = math.ceil(logical_len / block_size)
        page_ids = block_tables[seq_idx, :page_count].to(
            k_q.device, dtype=torch.long
        )
        k_tokens = (
            k_q.index_select(0, page_ids)
            .permute(0, 3, 1, 2, 4)
            .reshape(page_count * block_size, num_kv_heads, head_dim)[
                :logical_len
            ]
            .float()
        )
        v_tokens = (
            v_q.index_select(0, page_ids)
            .permute(0, 3, 1, 2)
            .reshape(page_count * block_size, num_kv_heads, head_dim)[
                :logical_len
            ]
            .float()
        )
        k_scales = (
            k_scale.index_select(0, page_ids)
            .permute(0, 2, 1)
            .reshape(page_count * block_size, num_kv_heads)[:logical_len]
            .float()
        )
        v_scales = (
            v_scale.index_select(0, page_ids)
            .permute(0, 2, 1)
            .reshape(page_count * block_size, num_kv_heads)[:logical_len]
            .float()
        )
        k_tokens = k_tokens * k_scales.unsqueeze(-1)
        v_tokens = v_tokens * v_scales.unsqueeze(-1)
        repeat = num_heads // num_kv_heads
        q_sdpa = (
            query[seq_idx].to(dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        )
        k_sdpa = (
            k_tokens.repeat_interleave(repeat, dim=1)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .to(dtype=torch.float32)
        )
        v_sdpa = (
            v_tokens.repeat_interleave(repeat, dim=1)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .to(dtype=torch.float32)
        )
        assert q_sdpa.shape == (1, num_heads, 1, head_dim)
        assert k_sdpa.shape == (1, num_heads, logical_len, head_dim)
        assert v_sdpa.shape == (1, num_heads, logical_len, head_dim)
        assert q_sdpa.dtype == k_sdpa.dtype == v_sdpa.dtype == torch.float32
        out_sdpa_fp32 = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            scale=1.0 / math.sqrt(head_dim),
            is_causal=False,
        )
        assert out_sdpa_fp32.shape == (1, num_heads, 1, head_dim)
        references.append(out_sdpa_fp32[0, :, 0, :].to(dtype=query.dtype))
    reference = torch.stack(references)
    assert reference.shape == actual.shape == (batch, num_heads, head_dim)
    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)


def test_fallback_import():
    from moe_infinity.kernel.paged_attention_ops import (
        HAS_PAGED_ATTN,
        paged_attention_fwd,
    )

    assert isinstance(HAS_PAGED_ATTN, bool)
    assert callable(paged_attention_fwd)


def test_fallback_cpu():
    batch, num_heads, head_dim, seq_len = 1, 4, 64, 8
    block_size = 8
    query = torch.randn(batch, num_heads, head_dim)
    k_cache, v_cache, block_tables, seq_lens = make_paged_kv(
        batch, num_heads, head_dim, seq_len, block_size
    )
    scale = 1.0 / math.sqrt(float(head_dim))
    out = paged_attention_fwd(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        scale,
        num_heads,
        block_size,
    )
    assert out.shape == query.shape
