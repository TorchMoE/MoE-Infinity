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


@dataclass(frozen=True)
class PagedBatchLengths:
    query_lengths: torch.Tensor | list[int]
    query_offsets: torch.Tensor | list[int]
    context_lengths: torch.Tensor | list[int]
    kv_seq_lengths: torch.Tensor | list[int]

    def validate(self) -> None:
        query = [int(value) for value in self.query_lengths]
        offsets = [int(value) for value in self.query_offsets]
        context = [int(value) for value in self.context_lengths]
        kv = [int(value) for value in self.kv_seq_lengths]
        if len(context) != len(query) or len(kv) != len(query):
            raise ValueError("paged length vectors must have equal batch size")
        expected_offsets = [0]
        for length in query:
            if length < 0:
                raise ValueError("query lengths must be non-negative")
            expected_offsets.append(expected_offsets[-1] + length)
        if offsets != expected_offsets:
            raise ValueError(
                "query_offsets must be the prefix sum of query_lengths"
            )
        if any(prior < 0 for prior in context):
            raise ValueError("context lengths must be non-negative")
        if any(
            total != prior + current
            for total, prior, current in zip(kv, context, query)
        ):
            raise ValueError(
                "kv_seq_lengths must equal context_lengths + query_lengths"
            )


@dataclass
class AttentionMetadata:
    block_tables: torch.Tensor
    lengths: PagedBatchLengths | None = None
    max_seq_len: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    slot_mapping: torch.Tensor | None = None
    is_prefill: bool = False
    seq_lens: torch.Tensor | None = None
    kv_storage_owner_id: str | None = None
    seq_id: int | None = None

    def __post_init__(self) -> None:
        if self.lengths is None and self.seq_lens is None:
            raise ValueError(
                "AttentionMetadata requires either lengths or seq_lens"
            )


@dataclass(frozen=True)
class FlashInferPlanMetadata:
    lengths: PagedBatchLengths
    kv_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
