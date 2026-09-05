from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .kv_cache import BlockAllocator, BlockTable


@dataclass
class MLAPagedKVCache:
    """Engine-owned DeepSeek MLA pages stored as ``[kv_c_normed | k_pe]``."""

    num_blocks: int
    block_size: int
    num_layers: int
    latent_dim: int
    rope_dim: int
    dtype: torch.dtype
    device: torch.device | None = None
    block_allocator: BlockAllocator = field(init=False)
    _sequence_tables: dict[int, BlockTable] = field(
        init=False, default_factory=dict
    )
    _mla_cache: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "num_blocks",
            "block_size",
            "num_layers",
            "latent_dim",
            "rope_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.block_allocator = BlockAllocator(
            self.num_blocks, self.block_size, self.device
        )
        self._mla_cache = torch.zeros(
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.latent_dim + self.rope_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> None:
        if seq_id in self._sequence_tables:
            raise ValueError(f"sequence {seq_id} already exists")
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        table = BlockTable(self.block_allocator)
        for _ in range(num_tokens):
            table.append_token()
        self._sequence_tables[seq_id] = table

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> None:
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must be >= 0")
        table = self._require_sequence(seq_id)
        for _ in range(num_new_tokens):
            table.append_token()

    def truncate_tokens(self, seq_id: int, new_len: int) -> None:
        table = self._require_sequence(seq_id)
        current = table.num_computed_tokens()
        if new_len < 0 or new_len > current:
            raise ValueError(
                "truncate_tokens requires 0 <= new_len <= current length"
            )
        blocks_needed = (new_len + self.block_size - 1) // self.block_size
        block_ids = table.get_block_ids()
        freed = block_ids[blocks_needed:]
        if freed:
            self.block_allocator.free(freed)
        table.restore_blocks(block_ids[:blocks_needed], new_len)

    def free_sequence(self, seq_id: int) -> None:
        table = self._sequence_tables.pop(seq_id, None)
        if table is not None:
            table.release()

    def get_block_table(self, seq_id: int) -> list[int]:
        return self._require_sequence(seq_id).get_block_ids()

    def get_mla_cache_tensors(self) -> torch.Tensor:
        return self._mla_cache

    @property
    def free_block_count(self) -> int:
        """Return currently unowned blocks without exposing allocator mutation."""
        return int(self.block_allocator.num_free_blocks)

    def validate_owned_access(
        self,
        seq_id: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        total_len: int,
    ) -> None:
        table = self._require_sequence(seq_id)
        if total_len > table.num_computed_tokens():
            raise ValueError(
                f"total_len {total_len} exceeds allocated sequence length "
                f"{table.num_computed_tokens()} for seq_id {seq_id}"
            )
        needed = (total_len + self.block_size - 1) // self.block_size
        expected = torch.tensor(
            table.get_block_ids()[:needed],
            dtype=torch.long,
            device=block_table.device,
        )
        actual = block_table.reshape(-1)[:needed].to(dtype=torch.long)
        if actual.numel() != expected.numel() or not torch.equal(
            actual, expected
        ):
            raise ValueError(
                f"block_tables do not match allocated sequence {seq_id} pages"
            )
        slot_pages = torch.div(
            slot_mapping.to(device=expected.device, dtype=torch.long),
            self.block_size,
            rounding_mode="floor",
        )
        if slot_pages.numel() and not bool(
            torch.isin(slot_pages, expected).all().item()
        ):
            raise ValueError(
                f"slot_mapping references pages not owned by sequence {seq_id}"
            )

    def write(
        self,
        layer_idx: int,
        latent: torch.Tensor,
        rope: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._validate_layer(layer_idx)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent must have shape [num_tokens, latent_dim]")
        if rope.ndim != 2 or rope.shape != (latent.shape[0], self.rope_dim):
            raise ValueError("rope must have shape [num_tokens, rope_dim]")
        if slot_mapping.ndim != 1 or slot_mapping.shape[0] != latent.shape[0]:
            raise ValueError("slot_mapping must have shape [num_tokens]")
        packed = torch.cat((latent, rope), dim=-1).to(
            device=self.device, dtype=self.dtype
        )
        slots = slot_mapping.to(device=self.device, dtype=torch.long)
        if slots.numel() == 0:
            return
        if bool((slots < 0).any().item()):
            raise ValueError("slot_mapping contains negative slot")
        if bool((slots >= self.num_blocks * self.block_size).any().item()):
            raise ValueError("slot_mapping points past allocated pages")

        # Advanced assignment with duplicate indices has backend-dependent write
        # ordering. Select each slot's final source row first so semantics match
        # the former left-to-right loop deterministically on CPU and GPU.
        unique_slots, inverse = torch.unique(
            slots, sorted=False, return_inverse=True
        )
        source_rows = torch.arange(slots.numel(), device=self.device)
        final_rows = torch.full_like(unique_slots, -1)
        final_rows.scatter_reduce_(
            0, inverse, source_rows, reduce="amax", include_self=True
        )
        final_slots = slots[final_rows]
        pages = torch.div(final_slots, self.block_size, rounding_mode="floor")
        offsets = torch.remainder(final_slots, self.block_size)
        self._mla_cache[layer_idx, pages, offsets] = packed[final_rows]

    def read(
        self,
        layer_idx: int,
        block_table: list[int] | torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_layer(layer_idx)
        if seq_len < 0:
            raise ValueError("seq_len must be >= 0")
        page_ids = torch.as_tensor(
            block_table, device=self.device, dtype=torch.long
        ).reshape(-1)
        needed = (seq_len + self.block_size - 1) // self.block_size
        if page_ids.numel() < needed:
            raise ValueError("block_table is too short for seq_len")
        selected_pages = page_ids[:needed]
        if selected_pages.numel() and bool(
            ((selected_pages < 0) | (selected_pages >= self.num_blocks))
            .any()
            .item()
        ):
            raise ValueError(
                "block_table contains page outside allocated cache"
            )
        if seq_len == 0:
            packed = self._mla_cache.new_empty(
                (0, self.latent_dim + self.rope_dim)
            )
        else:
            packed = self._mla_cache[layer_idx, selected_pages].reshape(
                -1, self.latent_dim + self.rope_dim
            )[:seq_len]
        return packed[:, : self.latent_dim], packed[:, self.latent_dim :]

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= int(layer_idx) < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} is outside the MLA cache")

    def _require_sequence(self, seq_id: int) -> BlockTable:
        try:
            return self._sequence_tables[seq_id]
        except KeyError as error:
            raise KeyError(f"unknown sequence id: {seq_id}") from error


__all__ = ["MLAPagedKVCache"]
