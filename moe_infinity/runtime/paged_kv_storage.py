from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from moe_infinity.serving.kv_cache import BlockAllocator


def canonical_device(device: torch.device | str | None) -> torch.device:
    """Resolve a device to an explicit indexed ``torch.device``.

    ``cuda`` (without an index) resolves to the current indexed CUDA device.
    CUDA identity is preserved even when CUDA is unavailable so CPU-only gate
    tests can still distinguish ``cuda:0`` from CPU or another CUDA index.
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        index = device.index
        if index is None:
            index = (
                torch.cuda.current_device() if torch.cuda.is_available() else 0
            )
        return torch.device("cuda", index)

    return device


@dataclass(frozen=True)
class PagedKVStorageSpec:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {self.num_blocks}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.num_kv_heads <= 0:
            raise ValueError(
                f"num_kv_heads must be > 0, got {self.num_kv_heads}"
            )
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {self.head_dim}")
        if self.head_dim % 8 != 0:
            raise ValueError(
                f"head_dim must be divisible by 8, got {self.head_dim}"
            )
        object.__setattr__(self, "device", canonical_device(self.device))


class PagedKVStorage:
    """The single authoritative owner of paged block allocation and per-layer
    native K/V tensors.

    Both the scheduler-facing ``PagedKVCache`` and the native
    ``PagedAttentionBackend`` receive the SAME instance so that reserving a
    block is proven to correspond to the exact page native attention reads.
    """

    _X: int = 8

    def __init__(self, spec: PagedKVStorageSpec) -> None:
        from moe_infinity.serving.kv_cache import BlockAllocator

        self.spec = spec
        self.owner_id = uuid.uuid4().hex
        self.block_allocator = BlockAllocator(
            num_blocks=spec.num_blocks,
            block_size=spec.block_size,
            device=spec.device,
        )
        x = self._X
        self.key_cache = torch.zeros(
            spec.num_layers,
            spec.num_blocks,
            spec.num_kv_heads,
            spec.head_dim // x,
            spec.block_size,
            x,
            dtype=spec.dtype,
            device=spec.device,
        )
        self.value_cache = torch.zeros(
            spec.num_layers,
            spec.num_blocks,
            spec.num_kv_heads,
            spec.head_dim,
            spec.block_size,
            dtype=spec.dtype,
            device=spec.device,
        )
        self._graph_scratch_blocks: set[int] = set()

    @property
    def num_layers(self) -> int:
        return self.spec.num_layers

    @property
    def num_blocks(self) -> int:
        return self.spec.num_blocks

    @property
    def block_size(self) -> int:
        return self.spec.block_size

    @property
    def num_kv_heads(self) -> int:
        return self.spec.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.spec.head_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.spec.dtype

    @property
    def device(self) -> torch.device:
        return self.spec.device

    def reserve_graph_scratch_blocks(self, count: int) -> list[int]:
        block_ids = self.block_allocator.allocate(count)
        self._graph_scratch_blocks.update(block_ids)
        return block_ids

    def release_graph_scratch_blocks(self, block_ids: list[int]) -> None:
        unknown = set(block_ids) - self._graph_scratch_blocks
        if unknown:
            raise ValueError(
                f"graph scratch blocks are not reserved: {sorted(unknown)}"
            )
        self._graph_scratch_blocks.difference_update(block_ids)
        self.block_allocator.free(list(block_ids))

    @property
    def num_graph_scratch_blocks(self) -> int:
        return len(self._graph_scratch_blocks)

    @property
    def graph_scratch_blocks(self) -> frozenset[int]:
        return frozenset(self._graph_scratch_blocks)

    def write_kv(
        self,
        *,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Eager, allocation-tolerant current-token K/V write.

        Normalizes inputs (device/dtype) before the graph boundary. The
        graph-capturable allocation-free variant lives in
        ``moe_infinity.kernel.paged_kv_write``.
        """
        if not 0 <= layer_idx < self.spec.num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range "
                f"[0, {self.spec.num_layers})"
            )
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        if key.ndim != 3:
            raise ValueError(
                "key/value must have shape [num_tokens, num_kv_heads, head_dim]"
            )
        num_tokens, num_kv_heads, head_dim = key.shape
        if num_kv_heads != self.spec.num_kv_heads:
            raise ValueError("num_kv_heads mismatch with storage spec")
        if head_dim != self.spec.head_dim:
            raise ValueError("head_dim mismatch with storage spec")
        if slot_mapping.ndim != 1 or slot_mapping.shape[0] != num_tokens:
            raise ValueError("slot_mapping must have shape [num_tokens]")

        x = self._X
        block_size = self.spec.block_size
        k_src = key.to(device=self.spec.device, dtype=self.spec.dtype)
        v_src = value.to(device=self.spec.device, dtype=self.spec.dtype)
        slots = slot_mapping.to(device=self.spec.device, dtype=torch.long)

        for i in range(num_tokens):
            slot = int(slots[i].item())
            if slot < 0:
                raise ValueError("slot_mapping contains negative slot index")
            block_id = slot // block_size
            token_offset = slot % block_size
            if block_id >= self.spec.num_blocks:
                raise ValueError("slot_mapping points past allocated blocks")
            self.key_cache[layer_idx, block_id, :, :, token_offset, :] = k_src[
                i
            ].reshape(self.spec.num_kv_heads, self.spec.head_dim // x, x)
            self.value_cache[layer_idx, block_id, :, :, token_offset] = v_src[i]


__all__ = ["PagedKVStorage", "PagedKVStorageSpec", "canonical_device"]
