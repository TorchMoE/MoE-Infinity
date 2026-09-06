from __future__ import annotations

import pytest
import torch

from moe_infinity.serving.mla_cache import MLAPagedKVCache


def _cache() -> MLAPagedKVCache:
    return MLAPagedKVCache(
        num_blocks=4,
        block_size=2,
        num_layers=2,
        latent_dim=3,
        rope_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_mla_cache_writes_external_slots_without_cross_layer_aliasing() -> None:
    cache = _cache()
    cache.allocate_sequence(7, 3)
    block_table = cache.get_block_table(7)
    slots = torch.tensor(
        [block_table[0] * 2, block_table[1] * 2], dtype=torch.int64
    )
    latent = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rope = torch.tensor([[7.0, 8.0], [9.0, 10.0]])

    cache.write(1, latent, rope, slots)

    packed = cache.get_mla_cache_tensors()
    assert packed.shape == (2, 4, 2, 5)
    assert torch.equal(
        packed[1, block_table[0], 0], torch.tensor([1, 2, 3, 7, 8])
    )
    assert torch.equal(
        packed[1, block_table[1], 0], torch.tensor([4, 5, 6, 9, 10])
    )
    assert torch.count_nonzero(packed[0]) == 0


def test_mla_cache_reads_block_table_order_and_truncates_logically() -> None:
    cache = _cache()
    cache.allocate_sequence(9, 4)
    block_table = cache.get_block_table(9)
    slots = torch.tensor(
        [
            block_table[0] * 2,
            block_table[0] * 2 + 1,
            block_table[1] * 2,
            block_table[1] * 2 + 1,
        ]
    )
    latent = torch.arange(12, dtype=torch.float32).view(4, 3)
    rope = torch.arange(8, dtype=torch.float32).view(4, 2) + 20
    cache.write(0, latent, rope, slots)

    read_latent, read_rope = cache.read(0, block_table, seq_len=4)
    assert torch.equal(read_latent, latent)
    assert torch.equal(read_rope, rope)

    cache.truncate_tokens(9, 2)
    assert cache.get_block_table(9) == block_table[:1]
    assert cache.block_allocator.num_free_blocks == 3


def test_mla_cache_rejects_invalid_layer_and_slot() -> None:
    cache = _cache()
    with pytest.raises(IndexError, match="layer_idx"):
        cache.write(2, torch.zeros(1, 3), torch.zeros(1, 2), torch.tensor([0]))
    with pytest.raises(ValueError, match="past allocated pages"):
        cache.write(0, torch.zeros(1, 3), torch.zeros(1, 2), torch.tensor([8]))


def test_mla_cache_vectorized_write_matches_reference_with_duplicate_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _cache()
    slots = torch.tensor([3, 0, 3, 6], dtype=torch.int64)
    latent = torch.arange(12, dtype=torch.float32).view(4, 3)
    rope = torch.arange(8, dtype=torch.float32).view(4, 2) + 30
    packed = torch.cat((latent, rope), dim=-1)
    expected = torch.zeros_like(cache.get_mla_cache_tensors())
    for token_idx, slot in enumerate((3, 0, 3, 6)):
        page, offset = divmod(slot, cache.block_size)
        expected[1, page, offset] = packed[token_idx]

    original_tolist = torch.Tensor.tolist

    def reject_tolist(tensor: torch.Tensor):
        if tensor is slots:
            raise AssertionError(
                "MLA write must not copy slot_mapping to a Python list"
            )
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", reject_tolist)
    cache.write(1, latent, rope, slots)

    assert torch.equal(cache.get_mla_cache_tensors(), expected)
    assert torch.count_nonzero(cache.get_mla_cache_tensors()[0]) == 0


@pytest.mark.parametrize(
    ("slot", "message"), [(-1, "negative"), (8, "past allocated")]
)
def test_mla_cache_vectorized_write_preserves_bounds_errors(
    slot: int, message: str
) -> None:
    cache = _cache()
    with pytest.raises(ValueError, match=message):
        cache.write(
            0,
            torch.zeros(1, 3),
            torch.zeros(1, 2),
            torch.tensor([slot]),
        )


def test_mla_cache_exposes_free_blocks_across_sequence_lifecycle() -> None:
    cache = _cache()

    assert cache.free_block_count == 4
    cache.allocate_sequence(21, 3)
    assert cache.free_block_count == 2
    cache.append_tokens(21, 2)
    assert cache.free_block_count == 1
    cache.free_sequence(21)
    assert cache.free_block_count == 4
