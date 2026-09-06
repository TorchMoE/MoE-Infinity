from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class PagedBatchLengths:
    query_lengths: list[int] | torch.Tensor
    query_offsets: list[int] | torch.Tensor
    context_lengths: list[int] | torch.Tensor
    kv_seq_lengths: list[int] | torch.Tensor

    def __post_init__(self) -> None:
        def values(value: list[int] | torch.Tensor) -> list[int]:
            if isinstance(value, torch.Tensor):
                if value.ndim != 1:
                    raise ValueError("paged batch lengths must be rank one")
                return [int(item) for item in value.detach().cpu().tolist()]
            return list(value)

        query_lengths = values(self.query_lengths)
        query_offsets = values(self.query_offsets)
        context_lengths = values(self.context_lengths)
        kv_seq_lengths = values(self.kv_seq_lengths)
        batch_size = len(query_lengths)
        if len(query_offsets) != batch_size + 1:
            raise ValueError("query_offsets must have batch_size + 1 entries")
        if len(context_lengths) != batch_size:
            raise ValueError("context_lengths must match query_lengths")
        if len(kv_seq_lengths) != batch_size:
            raise ValueError("kv_seq_lengths must match query_lengths")
        if query_offsets[:1] != [0]:
            raise ValueError("query_offsets must start at zero")
        running = 0
        for index, query_length in enumerate(query_lengths):
            if query_length <= 0 or context_lengths[index] < 0:
                raise ValueError("paged batch lengths must be non-negative")
            running += query_length
            if query_offsets[index + 1] != running:
                raise ValueError("query_offsets must sum query_lengths")
            if kv_seq_lengths[index] != context_lengths[index] + query_length:
                raise ValueError("kv_seq_lengths must equal context plus query")


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
    max_seq_len: int
    num_prefill_tokens: int
    num_decode_tokens: int
    slot_mapping: torch.Tensor
    is_prefill: bool
    seq_lens: torch.Tensor | None = None
    lengths: PagedBatchLengths | None = field(default=None)
    kv_storage_owner_id: str | None = None
    seq_id: int | None = None

    def __post_init__(self) -> None:
        if self.lengths is not None and self.seq_lens is None:
            self.seq_lens = torch.tensor(
                [int(v) for v in _as_int_list(self.lengths.kv_seq_lengths)],
                dtype=torch.int32,
            )
        if self.seq_lens is None:
            raise ValueError("AttentionMetadata requires seq_lens or lengths")


def _as_int_list(value: list[int] | torch.Tensor) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().tolist()]
    return list(value)
    kv_storage_owner_id: str | None = None
    seq_id: int | None = None
