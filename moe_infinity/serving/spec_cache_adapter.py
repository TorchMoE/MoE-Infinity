from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_infinity.runtime.attention_types import AttentionMetadata
from moe_infinity.spec_decode.protocols import CacheSnapshot

from .mla_cache import MLAPagedKVCache

EXECUTION_CONTEXT_PAGED_MLA = "paged_mla"


@dataclass(frozen=True)
class PagedCacheSnapshot(CacheSnapshot):
    block_table: tuple[int, ...]


class PagedCacheAdapter:
    """Per-sequence speculative handle over one engine-owned MLA cache."""

    cache_kind = "paged"
    mode = EXECUTION_CONTEXT_PAGED_MLA

    def __init__(
        self,
        cache: MLAPagedKVCache,
        seq_id: int,
        initial_length: int,
    ) -> None:
        if initial_length < 0:
            raise ValueError("initial_length must be >= 0")
        self.cache = cache
        self.seq_id = int(seq_id)
        self._logical_length = int(initial_length)
        self._released = False
        cache.allocate_sequence(self.seq_id, self._logical_length)

    def _ensure_active(self) -> None:
        if self._released:
            raise RuntimeError("paged cache adapter has been released")

    def snapshot(self) -> PagedCacheSnapshot:
        self._ensure_active()
        return PagedCacheSnapshot(
            logical_length=self._logical_length,
            block_table=tuple(self.cache.get_block_table(self.seq_id)),
        )

    def restore(self, snapshot: CacheSnapshot) -> None:
        self._ensure_active()
        if not isinstance(snapshot, PagedCacheSnapshot):
            raise TypeError("paged cache restore requires PagedCacheSnapshot")
        current = tuple(self.cache.get_block_table(self.seq_id))
        if current[: len(snapshot.block_table)] != snapshot.block_table:
            raise RuntimeError(
                "paged cache block-table prefix changed since snapshot"
            )
        self.truncate(snapshot.logical_length)

    def append(self, token_count: int) -> None:
        self._ensure_active()
        if token_count < 0:
            raise ValueError("token_count must be >= 0")
        self.cache.append_tokens(self.seq_id, token_count)
        self._logical_length += token_count

    def truncate(self, logical_length: int) -> None:
        self._ensure_active()
        if logical_length < 0 or logical_length > self._logical_length:
            raise ValueError(
                "logical_length must be between 0 and the current logical length"
            )
        self.cache.truncate_tokens(self.seq_id, logical_length)
        self._logical_length = logical_length

    def logical_length(self) -> int:
        self._ensure_active()
        return self._logical_length

    def build_attention_metadata(
        self, *, query_length: int, is_prefill: bool
    ) -> AttentionMetadata:
        self._ensure_active()
        if query_length < 0 or query_length > self._logical_length:
            raise ValueError(
                "query_length must be between 0 and the allocated logical length"
            )
        block_table = self.cache.get_block_table(self.seq_id)
        device = self.cache.device
        table_tensor = torch.tensor(
            [block_table], dtype=torch.int32, device=device
        )
        start = self._logical_length - query_length
        slots = [
            block_table[position // self.cache.block_size]
            * self.cache.block_size
            + position % self.cache.block_size
            for position in range(start, self._logical_length)
        ]
        return AttentionMetadata(
            block_tables=table_tensor,
            seq_lens=torch.tensor(
                [self._logical_length], dtype=torch.int32, device=device
            ),
            max_seq_len=self._logical_length,
            num_prefill_tokens=query_length if is_prefill else 0,
            num_decode_tokens=0 if is_prefill else query_length,
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
            is_prefill=is_prefill,
            seq_id=self.seq_id,
        )

    def swap_out(self) -> bool:
        """MLA pages currently stay resident during scheduler preemption."""
        self._ensure_active()
        return False

    def swap_in(self) -> bool:
        self._ensure_active()
        return False

    def release(self) -> None:
        if self._released:
            return
        self.cache.free_sequence(self.seq_id)
        self._logical_length = 0
        self._released = True


__all__ = [
    "EXECUTION_CONTEXT_PAGED_MLA",
    "PagedCacheAdapter",
    "PagedCacheSnapshot",
]
