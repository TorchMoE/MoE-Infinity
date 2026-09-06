from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

import torch
import torch.nn.functional as F

from moe_infinity.kernel.paged_attention_ops import paged_attention_fwd
from moe_infinity.runtime import flashinfer_utils
from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)
from moe_infinity.runtime.attention_types import (
    FlashInferPlanMetadata,
    KVCacheSpec,
    PagedBatchLengths,
)

__all__ = [
    "FlashInferPlanMetadata",
    "LayerRegistration",
    "LayeredPagedKVCheckpoint",
    "LayeredPagedKVPayload",
    "LayeredPagedKVStore",
    "PagedAttentionBackend",
    "PagedBatchLengths",
    "PrefixReuseCapability",
]

if TYPE_CHECKING:
    pass
if TYPE_CHECKING:
    from moe_infinity.runtime.paged_kv_storage import PagedKVStorage


@dataclass
class AttentionMetadata:
    is_prefill: bool
    block_table: Optional[torch.Tensor]
    slot_mapping: Optional[torch.Tensor]
    seq_lens: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class LayerRegistration:
    layer_idx: int
    module_id: int


@dataclass(frozen=True)
class LayeredPagedKVPayload:
    source_block_ids: list[int]
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    fi_kv_cache: torch.Tensor | None


@dataclass(frozen=True)
class LayeredPagedKVCheckpoint:
    source_block_ids: list[int]
    k_cache_cpu: torch.Tensor
    v_cache_cpu: torch.Tensor
    fi_kv_cache_cpu: torch.Tensor | None


class LayeredPagedKVStore:
    def __init__(
        self,
        *,
        owner: "PagedAttentionBackend",
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        use_flashinfer: bool,
    ) -> None:
        self.owner = owner
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.k_cache = torch.zeros(
            num_layers,
            num_blocks,
            num_kv_heads,
            head_dim // 8,
            block_size,
            8,
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            num_layers,
            num_blocks,
            num_kv_heads,
            head_dim,
            block_size,
            dtype=dtype,
            device=device,
        )
        self.fi_kv_cache = (
            torch.zeros(
                num_layers,
                num_blocks,
                2,
                block_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
            if use_flashinfer
            else None
        )

    def _validate_ids(self, block_ids: list[int]) -> list[int]:
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("block ids must be unique")
        if any(
            block_id < 0 or block_id >= self.num_blocks
            for block_id in block_ids
        ):
            raise ValueError("block id is outside the layered store")
        return list(block_ids)

    def _validate_payload_geometry(
        self, ids: list[int], payload: LayeredPagedKVPayload
    ) -> None:
        if len(ids) != len(payload.source_block_ids):
            raise ValueError("source and destination block counts differ")
        if (
            payload.k_cache.shape != self.k_cache[:, ids].shape
            or payload.v_cache.shape != self.v_cache[:, ids].shape
        ):
            raise ValueError("K/V payload geometry mismatch")
        if (
            payload.k_cache.dtype != self.dtype
            or payload.v_cache.dtype != self.dtype
        ):
            raise ValueError("K/V payload dtype mismatch")
        if (
            payload.k_cache.device != self.device
            or payload.v_cache.device != self.device
        ):
            raise ValueError("K/V payload device mismatch")
        if (payload.fi_kv_cache is None) != (self.fi_kv_cache is None):
            raise ValueError("FlashInfer payload/store mismatch")
        if payload.fi_kv_cache is not None and self.fi_kv_cache is not None:
            if payload.fi_kv_cache.shape != self.fi_kv_cache[:, ids].shape:
                raise ValueError("FlashInfer payload geometry mismatch")
            if (
                payload.fi_kv_cache.dtype != self.dtype
                or payload.fi_kv_cache.device != self.device
            ):
                raise ValueError("FlashInfer payload dtype/device mismatch")

    def export_blocks(self, block_ids: list[int]) -> LayeredPagedKVPayload:
        ids = self._validate_ids(block_ids)
        fi = (
            None
            if self.fi_kv_cache is None
            else self.fi_kv_cache[:, ids].clone()
        )
        return LayeredPagedKVPayload(
            list(block_ids),
            self.k_cache[:, ids].clone(),
            self.v_cache[:, ids].clone(),
            fi,
        )

    def import_blocks(
        self, block_ids: list[int], payload: LayeredPagedKVPayload
    ) -> None:
        ids = self._validate_ids(block_ids)
        self._validate_payload_geometry(ids, payload)
        self.k_cache[:, ids] = payload.k_cache
        self.v_cache[:, ids] = payload.v_cache
        if payload.fi_kv_cache is not None:
            if self.fi_kv_cache is None:
                raise ValueError(
                    "FlashInfer payload cannot be imported into a "
                    "non-FlashInfer store"
                )
            self.fi_kv_cache[:, ids] = payload.fi_kv_cache

    def checkpoint(self, block_ids: list[int]) -> LayeredPagedKVCheckpoint:
        payload = self.export_blocks(block_ids)
        fi = (
            None
            if payload.fi_kv_cache is None
            else payload.fi_kv_cache.detach().cpu()
        )
        return LayeredPagedKVCheckpoint(
            payload.source_block_ids,
            payload.k_cache.detach().cpu(),
            payload.v_cache.detach().cpu(),
            fi,
        )

    def restore(
        self, block_ids: list[int], checkpoint: LayeredPagedKVCheckpoint
    ) -> None:
        payload = LayeredPagedKVPayload(
            checkpoint.source_block_ids,
            checkpoint.k_cache_cpu.to(self.device, self.dtype),
            checkpoint.v_cache_cpu.to(self.device, self.dtype),
            None
            if checkpoint.fi_kv_cache_cpu is None
            else checkpoint.fi_kv_cache_cpu.to(self.device, self.dtype),
        )
        self.import_blocks(block_ids, payload)

    def zero_blocks(self, block_ids: list[int]) -> None:
        ids = self._validate_ids(block_ids)
        self.k_cache[:, ids] = 0
        self.v_cache[:, ids] = 0
        if self.fi_kv_cache is not None:
            self.fi_kv_cache[:, ids] = 0


@dataclass(frozen=True)
class PrefixReuseCapability:
    supported: bool
    reason: str
    backend: "PagedAttentionBackend | None"
    block_store: LayeredPagedKVStore | None

    @classmethod
    def active(
        cls,
        backend: "PagedAttentionBackend",
        store: LayeredPagedKVStore,
    ) -> "PrefixReuseCapability":
        if backend.block_store is not store or store.owner is not backend:
            raise ValueError("capability backend/store ownership mismatch")
        return cls(True, "active", backend, store)

    @classmethod
    def disabled(cls, reason: str) -> "PrefixReuseCapability":
        return cls(False, reason, None, None)


@runtime_checkable
class AttentionBackend(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        scale: Optional[float] = None,
    ) -> Optional[torch.Tensor]: ...

    def get_kv_cache_shape(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]: ...

    def supports_dtype(self, dtype: torch.dtype) -> bool: ...


class PlaceholderAttentionBackend:
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        scale: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        _ = (query, key, value, kv_cache, attn_metadata, scale)
        return None

    def get_kv_cache_shape(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        _ = dtype
        return True


class PagedAttentionBackend:
    spec: KVCacheSpec
    num_gpu_blocks: int
    device: torch.device
    _use_flashinfer: bool
    _fi_workspace: Optional[torch.Tensor]
    _fi_kv_cache: Optional[torch.Tensor]
    _fi_prefill: Optional[Any]
    _fi_decode: Optional[Any]
    _block_store: Optional[LayeredPagedKVStore]
    _layer_registry: dict[int, int]
    last_flashinfer_plan: Optional[Any]

    def __init__(
        self,
        spec: Optional[KVCacheSpec] = None,
        num_gpu_blocks: Optional[int] = None,
        device: Optional[torch.device] = None,
        *,
        storage: "Optional[PagedKVStorage]" = None,
        use_flashinfer: Optional[bool] = None,
        layer_idx: Optional[int] = None,
    ) -> None:
        if storage is not None:
            self._init_from_storage(storage, layer_idx, use_flashinfer)
            return

        if spec is None or num_gpu_blocks is None or device is None:
            raise ValueError(
                "PagedAttentionBackend requires spec, num_gpu_blocks, and "
                "device when no storage is provided"
            )
        if spec.head_dim % 8 != 0:
            raise ValueError("spec.head_dim must be divisible by 8")

        self.storage = None
        self._layer_idx = None
        self.spec = spec
        self.num_gpu_blocks = int(num_gpu_blocks)
        self.device = device
        self._block_store = None
        self._layer_registry = {}
        self.last_flashinfer_plan = None

        x = 8
        self._k_cache = torch.zeros(
            self.num_gpu_blocks,
            spec.num_kv_heads,
            spec.head_dim // x,
            spec.block_size,
            x,
            dtype=spec.dtype,
            device=device,
        )
        self._v_cache = torch.zeros(
            self.num_gpu_blocks,
            spec.num_kv_heads,
            spec.head_dim,
            spec.block_size,
            dtype=spec.dtype,
            device=device,
        )

        self._use_flashinfer = False
        self._fi_workspace = None
        self._fi_kv_cache = None
        self._fi_prefill = None
        self._fi_decode = None
        if flashinfer_utils.HAS_FLASHINFER:
            flashinfer_module = cast(
                Any,
                flashinfer_utils.get_flashinfer_module(),
            )
            if flashinfer_module is not None:
                try:
                    workspace = flashinfer_utils.get_workspace(device)
                    self._fi_workspace = workspace
                    self._fi_kv_cache = torch.zeros(
                        self.num_gpu_blocks,
                        2,
                        spec.block_size,
                        spec.num_kv_heads,
                        spec.head_dim,
                        dtype=spec.dtype,
                        device=device,
                    )
                    self._fi_prefill = (
                        flashinfer_module.BatchPrefillWithPagedKVCacheWrapper(
                            workspace,
                            "NHD",
                        )
                    )
                    self._fi_decode = (
                        flashinfer_module.BatchDecodeWithPagedKVCacheWrapper(
                            workspace,
                            "NHD",
                        )
                    )
                    self._use_flashinfer = True
                except Exception:
                    self._use_flashinfer = False
                    self._fi_workspace = None
                    self._fi_kv_cache = None
                    self._fi_prefill = None
                    self._fi_decode = None

    def create_layered_store(self, *, layer_count: int) -> LayeredPagedKVStore:
        if layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if self._block_store is not None:
            raise RuntimeError("paged backend store is already initialized")
        self._block_store = LayeredPagedKVStore(
            owner=self,
            num_layers=layer_count,
            num_blocks=self.num_gpu_blocks,
            block_size=self.spec.block_size,
            num_kv_heads=self.spec.num_kv_heads,
            head_dim=self.spec.head_dim,
            dtype=self.spec.dtype,
            device=self.device,
            use_flashinfer=self._flashinfer_enabled(),
        )
        self._k_cache = self._block_store.k_cache
        self._v_cache = self._block_store.v_cache
        self._fi_kv_cache = self._block_store.fi_kv_cache
        return self._block_store

    @property
    def block_store(self) -> LayeredPagedKVStore:
        if self._block_store is None:
            raise RuntimeError("paged backend store is not initialized")
        return self._block_store

    def register_layers(self, registrations: list[LayerRegistration]) -> None:
        by_layer = {item.layer_idx: item.module_id for item in registrations}
        expected_layers = self.block_store.num_layers
        if len(by_layer) != len(registrations) or set(by_layer) != set(
            range(expected_layers)
        ):
            raise ValueError(
                "paged layer registry must contain each layer exactly once"
            )
        self._layer_registry = by_layer

    def _init_from_storage(
        self,
        storage: "PagedKVStorage",
        layer_idx: Optional[int],
        use_flashinfer: Optional[bool],
    ) -> None:
        if use_flashinfer:
            raise ValueError(
                "storage-bound PagedAttentionBackend does not support "
                "FlashInfer; the FlashInfer plan path is graph-ineligible"
            )
        self.storage = storage
        self._layer_idx = layer_idx
        self._block_store = None
        self._layer_registry = {}
        self.spec = KVCacheSpec(
            num_kv_heads=storage.num_kv_heads,
            head_dim=storage.head_dim,
            dtype=storage.spec.dtype,
            block_size=storage.block_size,
        )
        self.num_gpu_blocks = storage.num_blocks
        self.device = storage.spec.device
        self._k_cache = None
        self._v_cache = None
        self._use_flashinfer = False
        self._fi_workspace = None
        self._fi_kv_cache = None
        self._fi_prefill = None
        self._fi_decode = None

    @property
    def layer_idx(self) -> Optional[int]:
        return self._layer_idx

    @property
    def k_cache(self) -> torch.Tensor:
        if self.storage is not None:
            return self.storage.key_cache[self._layer_idx or 0]
        return self._k_cache

    @property
    def v_cache(self) -> torch.Tensor:
        if self.storage is not None:
            return self.storage.value_cache[self._layer_idx or 0]
        return self._v_cache

    def _check_owner(
        self,
        metadata: "AttentionMetadata | RuntimeAttentionMetadata",
    ) -> None:
        if self.storage is None:
            return
        owner_id = getattr(metadata, "kv_storage_owner_id", None)
        if owner_id is not None and owner_id != self.storage.owner_id:
            raise ValueError("KV storage owner mismatch")

    def write_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        if key.ndim != 3:
            raise ValueError(
                "key/value must have shape [num_tokens, num_kv_heads, head_dim]"
            )
        if slot_mapping.ndim != 1 or slot_mapping.shape[0] != key.shape[0]:
            raise ValueError("slot_mapping must have shape [num_tokens]")

        num_tokens, num_kv_heads, head_dim = key.shape
        if num_kv_heads != self.spec.num_kv_heads:
            raise ValueError("num_kv_heads mismatch with cache spec")
        if head_dim != self.spec.head_dim:
            raise ValueError("head_dim mismatch with cache spec")

        block_size = self.spec.block_size
        k_store = self._k_store(layer_idx)
        v_store = self._v_store(layer_idx)
        x = k_store.shape[-1]

        k_src = key.to(device=self.device, dtype=k_store.dtype)
        v_src = value.to(device=self.device, dtype=v_store.dtype)
        slots = slot_mapping.to(device=self.device, dtype=torch.long)

        for i in range(num_tokens):
            slot = int(slots[i].item())
            if slot < 0:
                raise ValueError("slot_mapping contains negative slot index")

            block_id = slot // block_size
            token_offset = slot % block_size
            if block_id >= self.num_gpu_blocks:
                raise ValueError(
                    "slot_mapping points past allocated GPU blocks"
                )

            k_store[block_id, :, :, token_offset, :] = k_src[i].reshape(
                self.spec.num_kv_heads,
                self.spec.head_dim // x,
                x,
            )
            v_store[block_id, :, :, token_offset] = v_src[i]

    def _k_store(self, layer_idx: int) -> torch.Tensor:
        if self._block_store is not None:
            return self._block_store.k_cache[layer_idx]
        return self.k_cache

    def _v_store(self, layer_idx: int) -> torch.Tensor:
        if self._block_store is not None:
            return self._block_store.v_cache[layer_idx]
        return self.v_cache

    def _fi_store(self, layer_idx: int) -> Optional[torch.Tensor]:
        if self._block_store is not None:
            if self._block_store.fi_kv_cache is None:
                return None
            return self._block_store.fi_kv_cache[layer_idx]
        return self._fi_kv_cache

    def write_kv_flashinfer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        fi_store = self._fi_store(layer_idx)
        if fi_store is None:
            raise RuntimeError("FlashInfer KV cache is not initialized")
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        if key.ndim != 3:
            raise ValueError(
                "key/value must have shape [num_tokens, num_kv_heads, head_dim]"
            )
        if slot_mapping.ndim != 1 or slot_mapping.shape[0] != key.shape[0]:
            raise ValueError("slot_mapping must have shape [num_tokens]")

        num_tokens, num_kv_heads, head_dim = key.shape
        if num_kv_heads != self.spec.num_kv_heads:
            raise ValueError("num_kv_heads mismatch with cache spec")
        if head_dim != self.spec.head_dim:
            raise ValueError("head_dim mismatch with cache spec")

        k_src = key.to(device=self.device, dtype=self.spec.dtype)
        v_src = value.to(device=self.device, dtype=self.spec.dtype)
        slots = slot_mapping.to(device=self.device, dtype=torch.long)

        block_size = self.spec.block_size
        for i in range(num_tokens):
            slot = int(slots[i].item())
            if slot < 0:
                raise ValueError("slot_mapping contains negative slot index")

            block_id = slot // block_size
            token_offset = slot % block_size
            if block_id >= self.num_gpu_blocks:
                raise ValueError(
                    "slot_mapping points past allocated GPU blocks"
                )

            fi_store[block_id, 0, token_offset] = k_src[i]
            fi_store[block_id, 1, token_offset] = v_src[i]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[
            AttentionMetadata | RuntimeAttentionMetadata
        ] = None,
        scale: Optional[float] = None,
        attention_metadata: Optional[
            AttentionMetadata | RuntimeAttentionMetadata
        ] = None,
        layer_idx: int = 0,
        metadata: Optional[AttentionMetadata | RuntimeAttentionMetadata] = None,
        graph_mode: bool = False,
    ) -> torch.Tensor:
        _ = (kv_cache, graph_mode)
        metadata = (
            metadata
            if metadata is not None
            else (
                attention_metadata
                if attention_metadata is not None
                else attn_metadata
            )
        )
        if metadata is None:
            raise ValueError("attention metadata is required")

        self._check_owner(metadata)
        if self.storage is not None and layer_idx is not None:
            self._layer_idx = int(layer_idx)

        if self._is_prefill(metadata):
            slot_mapping = self._get_slot_mapping(metadata)
            if slot_mapping is None:
                raise ValueError("prefill requires slot_mapping")
            self.write_kv(key, value, slot_mapping, layer_idx=layer_idx)
            if self._flashinfer_enabled():
                self.write_kv_flashinfer(
                    key, value, slot_mapping, layer_idx=layer_idx
                )
            return self._prefill_forward(
                query,
                key,
                value,
                metadata=metadata,
                scale=scale,
                layer_idx=layer_idx,
            )

        return self._decode_forward(
            query, metadata, scale=scale, layer_idx=layer_idx
        )
        self._write_decode_kv(key, value, metadata)
        return self._decode_forward(query, metadata, scale=scale)

    def _write_decode_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
    ) -> None:
        if self.storage is None:
            return
        slot_mapping = self._get_slot_mapping(metadata)
        if slot_mapping is None:
            raise ValueError("decode requires slot_mapping")
        from moe_infinity.kernel.paged_kv_write import paged_kv_write_

        paged_kv_write_(
            self.storage,
            layer_idx=self._layer_idx or 0,
            key=key,
            value=value,
            slot_mapping=slot_mapping,
        )

    def _prefill_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: Optional[AttentionMetadata | RuntimeAttentionMetadata] = None,
        scale: Optional[float] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                "prefill query/key/value must have shape [num_tokens, num_heads, head_dim]"
            )

        if self._flashinfer_enabled():
            if metadata is None:
                raise ValueError("prefill requires attention metadata")
            fi_store = self._fi_store(layer_idx)
            if self._fi_prefill is None or fi_store is None:
                raise RuntimeError(
                    "FlashInfer prefill wrappers are unavailable"
                )

            query_src = query.to(self.device, dtype=self.spec.dtype)
            num_qo_heads = int(query_src.shape[1])
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len = (
                self._build_flashinfer_metadata(metadata)
            )

            self._call_prefill_plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                query_src.dtype,
            )
            return cast(
                torch.Tensor,
                self._fi_prefill.run(query_src, fi_store),
            )

        q = (
            query.to(self.device, dtype=self.spec.dtype)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        k = (
            key.to(self.device, dtype=self.spec.dtype)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        v = (
            value.to(self.device, dtype=self.spec.dtype)
            .transpose(0, 1)
            .unsqueeze(0)
        )

        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                "query num_heads must be divisible by num_kv_heads"
            )
        head_ratio = num_heads // num_kv_heads
        if head_ratio > 1:
            k = k.repeat_interleave(head_ratio, dim=1)
            v = v.repeat_interleave(head_ratio, dim=1)

        attn_scale = (
            float(scale)
            if scale is not None
            else 1.0 / math.sqrt(float(self.spec.head_dim))
        )

        try:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=attn_scale,
                is_causal=True,
            )
        except TypeError:
            out = F.scaled_dot_product_attention(
                q * attn_scale,
                k,
                v,
                is_causal=True,
            )

        return out.squeeze(0).transpose(0, 1)

    def _decode_forward(
        self,
        query: torch.Tensor,
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
        scale: Optional[float] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if query.ndim == 2:
            query = query.unsqueeze(0)
        if query.ndim != 3:
            raise ValueError(
                "decode query must have shape [batch_size, num_heads, head_dim]"
            )

        if self._flashinfer_enabled():
            fi_store = self._fi_store(layer_idx)
            if self._fi_decode is None or fi_store is None:
                raise RuntimeError("FlashInfer decode wrappers are unavailable")

            query_src = query.to(self.device, dtype=self.spec.dtype)
            num_qo_heads = int(query_src.shape[1])
            _, kv_indptr, kv_indices, kv_last_page_len = (
                self._build_flashinfer_metadata(metadata)
            )
            self._call_decode_plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                query_src.dtype,
            )
            return cast(
                torch.Tensor,
                self._fi_decode.run(query_src, fi_store),
            )

        block_tables = self._get_block_tables(metadata)
        seq_lens = self._get_seq_lens(metadata)
        max_seq_len = self._get_max_seq_len(metadata, seq_lens)
        if block_tables is None or seq_lens is None:
            raise ValueError("decode requires block_tables and seq_lens")

        attn_scale = (
            float(scale)
            if scale is not None
            else 1.0 / math.sqrt(float(self.spec.head_dim))
        )

        return paged_attention_fwd(
            query=query.to(self.device, dtype=self.spec.dtype),
            key_cache=self._k_store(layer_idx),
            value_cache=self._v_store(layer_idx),
            block_tables=block_tables.to(self.device),
            seq_lens=seq_lens.to(self.device),
            scale=attn_scale,
            num_kv_heads=self.spec.num_kv_heads,
            block_size=self.spec.block_size,
            max_seq_len=max_seq_len,
        )

    def _flashinfer_enabled(self) -> bool:
        return bool(
            self._use_flashinfer
            and self._fi_prefill is not None
            and self._fi_decode is not None
            and self._fi_kv_cache is not None
        )

    def _resolve_query_and_kv_lengths(
        self,
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
    ) -> tuple[list[int], list[int]]:
        lengths = getattr(metadata, "lengths", None)
        if lengths is not None:
            query = [int(value) for value in lengths.query_lengths]
            kv = [int(value) for value in lengths.kv_seq_lengths]
            return query, kv

        seq_lens = self._get_seq_lens(metadata)
        if seq_lens is None:
            raise ValueError(
                "FlashInfer attention requires block_tables and seq lengths"
            )
        kv = [int(value) for value in seq_lens.reshape(-1).tolist()]
        return list(kv), kv

    def _build_flashinfer_metadata(
        self,
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        block_tables = self._get_block_tables(metadata)
        if block_tables is None:
            raise ValueError(
                "FlashInfer attention requires block_tables and seq lengths"
            )

        query_lengths, kv_seq_lengths = self._resolve_query_and_kv_lengths(
            metadata
        )
        block_tables_i32 = block_tables.to(self.device, dtype=torch.int32)
        batch_size = len(kv_seq_lengths)

        query_offsets_vals = [0]
        for query_len in query_lengths:
            query_offsets_vals.append(query_offsets_vals[-1] + query_len)
        qo_indptr = torch.tensor(
            query_offsets_vals,
            dtype=torch.int32,
            device=self.device,
        )

        block_size = int(self.spec.block_size)
        kv_indptr_vals = [0]
        kv_last_page_len = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indices_per_seq: list[torch.Tensor] = []
        total_pages = 0

        for i in range(batch_size):
            seq_len = int(kv_seq_lengths[i])
            num_pages = max((seq_len + block_size - 1) // block_size, 1)
            if num_pages > int(block_tables_i32.shape[1]):
                raise ValueError(
                    "block_tables does not have enough page indices"
                )
            kv_indices_per_seq.append(block_tables_i32[i, :num_pages])
            total_pages += num_pages
            kv_indptr_vals.append(total_pages)

            if seq_len <= 0:
                kv_last_page_len[i] = 1
            else:
                rem = seq_len % block_size
                kv_last_page_len[i] = block_size if rem == 0 else rem

        kv_indptr = torch.tensor(
            kv_indptr_vals,
            dtype=torch.int32,
            device=self.device,
        )
        if total_pages == 0:
            kv_indices = torch.empty(
                0,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            kv_indices = torch.cat(kv_indices_per_seq, dim=0).to(
                self.device,
                dtype=torch.int32,
            )

        self._record_flashinfer_plan(
            metadata, query_lengths, kv_seq_lengths, kv_indptr, kv_last_page_len
        )

        return qo_indptr, kv_indptr, kv_indices, kv_last_page_len

    def _record_flashinfer_plan(
        self,
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
        query_lengths: list[int],
        kv_seq_lengths: list[int],
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
    ) -> None:
        lengths = getattr(metadata, "lengths", None)
        if lengths is None:
            context_lengths = [
                kv - query for kv, query in zip(kv_seq_lengths, query_lengths)
            ]
            query_offsets_vals = [0]
            for query_len in query_lengths:
                query_offsets_vals.append(query_offsets_vals[-1] + query_len)
            lengths = PagedBatchLengths(
                query_lengths=list(query_lengths),
                query_offsets=query_offsets_vals,
                context_lengths=context_lengths,
                kv_seq_lengths=list(kv_seq_lengths),
            )
        self.last_flashinfer_plan = FlashInferPlanMetadata(
            lengths=lengths,
            kv_indptr=kv_indptr.detach().clone(),
            kv_last_page_len=kv_last_page_len.detach().clone(),
        )

    def _call_prefill_plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        query_dtype: torch.dtype,
    ) -> None:
        if self._fi_prefill is None:
            raise RuntimeError("FlashInfer prefill wrapper is unavailable")

        try:
            self._fi_prefill.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                self.spec.num_kv_heads,
                self.spec.head_dim,
                self.spec.block_size,
                causal=True,
                q_data_type=query_dtype,
                kv_data_type=query_dtype,
            )
        except TypeError:
            self._fi_prefill.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                self.spec.num_kv_heads,
                self.spec.head_dim,
                self.spec.block_size,
                causal=True,
            )

    def _call_decode_plan(
        self,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        query_dtype: torch.dtype,
    ) -> None:
        if self._fi_decode is None:
            raise RuntimeError("FlashInfer decode wrapper is unavailable")

        try:
            self._fi_decode.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                self.spec.num_kv_heads,
                self.spec.head_dim,
                self.spec.block_size,
                pos_encoding_mode="NONE",
                q_data_type=query_dtype,
                kv_data_type=query_dtype,
            )
        except TypeError:
            self._fi_decode.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                self.spec.num_kv_heads,
                self.spec.head_dim,
                self.spec.block_size,
                pos_encoding_mode="NONE",
                data_type=query_dtype,
            )

    @classmethod
    def get_kv_cache_shape(
        cls,
        spec: KVCacheSpec,
        num_gpu_blocks: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        _ = cls
        x = 8
        k_shape = (
            int(num_gpu_blocks),
            spec.num_kv_heads,
            spec.head_dim // x,
            spec.block_size,
            x,
        )
        v_shape = (
            int(num_gpu_blocks),
            spec.num_kv_heads,
            spec.head_dim,
            spec.block_size,
        )
        return k_shape, v_shape

    def supports_dtype(self, dtype: torch.dtype) -> bool:
        return dtype in (torch.float16, torch.bfloat16, torch.float32)

    def decode_graph_capability(self) -> "DecodeGraphCapability":
        from moe_infinity.runtime.attention_types import DecodeGraphCapability

        if self._use_flashinfer:
            return DecodeGraphCapability(False, "flashinfer_plan_path")
        if self.storage is None:
            return DecodeGraphCapability(False, "kv_storage_mismatch")
        return DecodeGraphCapability(
            True, "eligible", storage_owner_id=self.storage.owner_id
        )

    @staticmethod
    def _is_prefill(
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
    ) -> bool:
        return bool(getattr(metadata, "is_prefill", False))

    @staticmethod
    def _get_slot_mapping(
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
    ) -> Optional[torch.Tensor]:
        slot_mapping = cast(
            Optional[torch.Tensor], getattr(metadata, "slot_mapping", None)
        )
        return slot_mapping

    @staticmethod
    def _get_block_tables(
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
    ) -> Optional[torch.Tensor]:
        block_tables = getattr(metadata, "block_tables", None)
        if block_tables is not None:
            return cast(torch.Tensor, block_tables)
        block_table = getattr(metadata, "block_table", None)
        if block_table is not None:
            return cast(torch.Tensor, block_table)
        return None

    @staticmethod
    def _get_seq_lens(
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
    ) -> Optional[torch.Tensor]:
        lengths = getattr(metadata, "lengths", None)
        if lengths is not None:
            kv_seq_lengths = lengths.kv_seq_lengths
            if isinstance(kv_seq_lengths, torch.Tensor):
                return kv_seq_lengths
            return torch.tensor(
                [int(value) for value in kv_seq_lengths],
                dtype=torch.int32,
            )
        seq_lens = getattr(metadata, "seq_lens", None)
        if seq_lens is None:
            return None
        return cast(torch.Tensor, seq_lens)

    @staticmethod
    def _get_max_seq_len(
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
        seq_lens: Optional[torch.Tensor],
    ) -> int:
        max_seq_len_obj = cast(object, getattr(metadata, "max_seq_len", None))
        if isinstance(max_seq_len_obj, (int, float)):
            return int(max_seq_len_obj)
        if seq_lens is None or seq_lens.numel() == 0:
            return 0
        return int(seq_lens.max().item())
