from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

DECODE_GRAPH_REASONS = (
    "eligible",
    "missing_capability",
    "active_model_hooks",
    "archer_callbacks",
    "transfer_scheduler",
    "expert_dispatcher",
    "kv_offload",
    "flashinfer_plan_path",
    "dynamic_allocations",
    "native_paged_required",
    "mla_layout_unsupported",
    "kv_storage_mismatch",
    "paged_class_unregistered",
    "layer_idx_invalid",
    "layer_write_unproven",
)


@dataclass(frozen=True)
class PagedLayerWriteProof:
    class_fqn: str
    layer_idx: int
    storage_owner_id: str
    writer: str
    writes_before_attention: bool
    allocation_free: bool


@dataclass(frozen=True)
class DecodeGraphCapability:
    safe: bool
    reason: str
    storage_owner_id: str | None = None
    layer_write_proofs: tuple[PagedLayerWriteProof, ...] = ()


@runtime_checkable
class DecodeGraphCapabilityProvider(Protocol):
    def decode_graph_capability(self) -> DecodeGraphCapability: ...


@dataclass
class KVCacheSpec:
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    block_size: int

    @property
    def page_size_bytes(self) -> int:
        dtype_size = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}[
            self.dtype
        ]
        return (
            self.block_size * self.num_kv_heads * self.head_dim * dtype_size * 2
        )


@dataclass
class AttentionMetadata:
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    num_prefill_tokens: int
    num_decode_tokens: int
    slot_mapping: torch.Tensor
    is_prefill: bool
    kv_storage_owner_id: str | None = None
    seq_id: int | None = None
