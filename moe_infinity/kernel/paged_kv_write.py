from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

if TYPE_CHECKING:
    from moe_infinity.runtime.paged_kv_storage import PagedKVStorage

_X = 8


def _validate(
    storage: "PagedKVStorage",
    layer_idx: int,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if not 0 <= layer_idx < storage.spec.num_layers:
        raise ValueError(
            f"layer_idx {layer_idx} out of range [0, {storage.spec.num_layers})"
        )
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if key.ndim != 3:
        raise ValueError(
            "key/value must have shape [num_tokens, num_kv_heads, head_dim]"
        )
    num_tokens, num_kv_heads, head_dim = key.shape
    if num_kv_heads != storage.spec.num_kv_heads:
        raise ValueError("num_kv_heads mismatch with storage spec")
    if head_dim != storage.spec.head_dim:
        raise ValueError("head_dim mismatch with storage spec")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != num_tokens:
        raise ValueError("slot_mapping must have shape [num_tokens]")
    if key.device != storage.spec.device or value.device != storage.spec.device:
        raise ValueError("key/value must be on the storage device")
    if slot_mapping.device != storage.spec.device:
        raise ValueError("slot_mapping must be on the storage device")
    if key.dtype != storage.spec.dtype or value.dtype != storage.spec.dtype:
        raise ValueError("key/value dtype must match the storage dtype")


if _HAS_TRITON:

    @triton.jit
    def _paged_kv_write_kernel(
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        slot_mapping_ptr,
        layer_idx,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        x: tl.constexpr,
        block_size: tl.constexpr,
        num_blocks: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)

        slot = tl.load(slot_mapping_ptr + token)
        block_id = slot // block_size
        token_offset = slot % block_size

        dim = tl.arange(0, head_dim)
        src_offset = token * (num_kv_heads * head_dim) + head * head_dim + dim
        key_vals = tl.load(key_ptr + src_offset)
        value_vals = tl.load(value_ptr + src_offset)

        layer_block_stride = num_kv_heads * (head_dim // x) * block_size * x
        block_stride = (head_dim // x) * block_size * x
        head_stride = block_size * x
        outer = dim // x
        inner = dim % x
        key_dst = (
            layer_idx * (num_blocks * layer_block_stride)
            + block_id * layer_block_stride
            + head * block_stride
            + outer * head_stride
            + token_offset * x
            + inner
        )
        tl.store(key_cache_ptr + key_dst, key_vals)

        v_layer_block_stride = num_kv_heads * head_dim * block_size
        v_block_stride = head_dim * block_size
        v_head_stride = head_dim * block_size
        value_dst = (
            layer_idx * (num_blocks * v_layer_block_stride)
            + block_id * v_layer_block_stride
            + head * v_head_stride
            + dim * block_size
            + token_offset
        )
        tl.store(value_cache_ptr + value_dst, value_vals)


def _write_cpu(
    storage: "PagedKVStorage",
    layer_idx: int,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    block_size = storage.spec.block_size
    x = _X
    key_cache = storage.key_cache
    value_cache = storage.value_cache
    num_tokens = key.shape[0]
    key_view = key.view(
        num_tokens, storage.spec.num_kv_heads, storage.spec.head_dim // x, x
    )
    slots = slot_mapping.to(dtype=torch.long)
    block_ids = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots - block_ids * block_size
    for i in range(num_tokens):
        block_id = block_ids[i]
        offset = offsets[i]
        key_cache[layer_idx, block_id, :, :, offset, :] = key_view[i]
        value_cache[layer_idx, block_id, :, :, offset] = value[i]


def paged_kv_write_(
    storage: "PagedKVStorage",
    *,
    layer_idx: int,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Allocation-free in-place current-token K/V write, safe under CUDA graph
    capture: validation precedes all writes and no new tensor/list/workspace is
    allocated on the write path.

    On CUDA the write is a Triton kernel launch whose grid is fixed by the
    captured ``[num_tokens, num_kv_heads]`` dimensions. On CPU (or when Triton
    is unavailable) the equivalent allocation-free indexed write is used."""
    _validate(storage, layer_idx, key, value, slot_mapping)

    if not (_HAS_TRITON and key.is_cuda):
        _write_cpu(storage, layer_idx, key, value, slot_mapping)
        return

    num_tokens, num_kv_heads, head_dim = key.shape
    grid = (num_tokens, num_kv_heads)
    _paged_kv_write_kernel[grid](
        key,
        value,
        storage.key_cache,
        storage.value_cache,
        slot_mapping,
        layer_idx,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        x=_X,
        block_size=storage.spec.block_size,
        num_blocks=storage.spec.num_blocks,
    )


__all__ = ["paged_kv_write_"]
