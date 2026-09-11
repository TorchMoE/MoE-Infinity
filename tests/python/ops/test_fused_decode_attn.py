import math

import pytest
import torch
import torch.nn.functional as F

from tests.python.ops.conftest import (
    BF16_ATOL,
    BF16_RTOL,
    requires_cuda,
    requires_triton,
)


def _run_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    try:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
            is_causal=False,
        )
    except TypeError:
        return F.scaled_dot_product_attention(
            query * scale,
            key,
            value,
            is_causal=False,
        )


def _make_paged_kv(
    batch: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks_per_seq = math.ceil(seq_len / block_size)
    num_blocks = batch * blocks_per_seq
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    value_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    physical_blocks = torch.randperm(
        num_blocks, device=device, dtype=torch.int64
    )
    block_tables = physical_blocks.reshape(batch, blocks_per_seq).to(
        torch.int32
    )
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    return key_cache, value_cache, block_tables, seq_lens


def _reference_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, _, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, _ = key_cache.shape
    query_group_size = num_heads // num_kv_heads

    outputs: list[torch.Tensor] = []
    for batch_idx in range(batch):
        seq_len = int(seq_lens[batch_idx].item())
        blocks_per_seq = math.ceil(seq_len / block_size)
        k_blocks: list[torch.Tensor] = []
        v_blocks: list[torch.Tensor] = []

        for block_idx in range(blocks_per_seq):
            physical_block = int(block_tables[batch_idx, block_idx].item())
            valid_tokens = min(block_size, seq_len - block_idx * block_size)
            k_blocks.append(key_cache[physical_block, :valid_tokens])
            v_blocks.append(value_cache[physical_block, :valid_tokens])

        key = torch.cat(k_blocks, dim=0).permute(1, 0, 2)
        value = torch.cat(v_blocks, dim=0).permute(1, 0, 2)
        if query_group_size > 1:
            key = key.repeat_interleave(query_group_size, dim=0)
            value = value.repeat_interleave(query_group_size, dim=0)

        q = query[batch_idx, 0].unsqueeze(1)
        out = _run_sdpa(q, key, value, scale)
        outputs.append(out.squeeze(1))

    return torch.stack(outputs, dim=0).reshape(batch, num_heads, head_dim)


@requires_cuda
@requires_triton
@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("seq_len", [32, 128, 512, 2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("block_size", [16])
def test_fused_decode_attention_matches_reference(
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> None:
    from moe_infinity.kernel.fused_decode_attn import fused_decode_attention

    device = torch.device("cuda")
    query = torch.randn(
        batch,
        1,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    key_cache, value_cache, block_tables, seq_lens = _make_paged_kv(
        batch=batch,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        device=device,
    )
    scale = 1.0 / math.sqrt(float(head_dim))

    output = fused_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
    )
    reference = _reference_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
    )

    assert output.shape == (batch, num_heads, head_dim)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output.float(),
        reference.float(),
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )


@pytest.mark.parametrize("num_kv_heads", [8, 32])
def test_fused_decode_attention_falls_back_to_eager_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    num_kv_heads: int,
) -> None:
    """``MOE_DISABLE_FUSED_KERNELS=1`` must still compute decode attention.

    The fallback runs without CUDA or Triton, so this covers the CPU-only
    path that the parametrized kernel test above cannot reach.
    """
    from moe_infinity import kernel

    monkeypatch.setattr(kernel, "_FUSED_KERNELS_DISABLED", True)

    batch, seq_len, num_heads, head_dim, block_size = 2, 40, 32, 128, 16
    device = torch.device("cpu")
    query = torch.randn(
        batch,
        1,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    key_cache, value_cache, block_tables, seq_lens = _make_paged_kv(
        batch=batch,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        device=device,
    )
    scale = 1.0 / math.sqrt(float(head_dim))

    output = kernel.fused_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
    )
    reference = _reference_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
    )

    assert output.shape == (batch, num_heads, head_dim)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output.float(),
        reference.float(),
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )
