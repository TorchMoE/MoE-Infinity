from __future__ import annotations

import math
from dataclasses import dataclass
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

from moe_infinity.kernel.paged_attention_ops import (
    paged_attention_fwd,
    probe_native_int8_binding,
)
from moe_infinity.runtime import flashinfer_utils
from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)
from moe_infinity.runtime.attention_types import (
    KVCacheSpec,
)
from moe_infinity.runtime.kv_cache_format import (
    KVCacheBackendCapabilities,
    KVCacheModelInfo,
    LayeredPagedKVStore,
    resolve_kv_cache_format,
)

if TYPE_CHECKING:
    from moe_infinity.runtime.paged_kv_storage import PagedKVStorage


@dataclass
class AttentionMetadata:
    is_prefill: bool
    block_table: Optional[torch.Tensor]
    slot_mapping: Optional[torch.Tensor]
    seq_lens: Optional[torch.Tensor] = None


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
    k_cache: Optional[torch.Tensor]
    v_cache: Optional[torch.Tensor]
    _use_flashinfer: bool
    _fi_workspace: Optional[torch.Tensor]
    _fi_kv_cache: Optional[torch.Tensor]
    _fi_prefill: Optional[Any]
    _fi_decode: Optional[Any]
    _store: Optional[LayeredPagedKVStore]
    _store_owner_id: Optional[str]
    execution_backend: Optional[str]
    effective_format: str
    format_decision_reason: Optional[str]

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

        self._store = None
        self._store_owner_id = None
        self.execution_backend = None
        self.effective_format = spec.format_name
        self.format_decision_reason = None

        self._is_int8 = spec.format_name == "int8_sym"

        x = 8
        if self._is_int8:
            self._k_cache = None
            self._v_cache = None
        else:
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
        if not self._is_int8 and flashinfer_utils.HAS_FLASHINFER:
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

    @property
    def store(self) -> LayeredPagedKVStore:
        if self._store is None:
            raise RuntimeError(
                "paged attention backend has no bound LayeredPagedKVStore"
            )
        return self._store

    def bind_store(self, store: LayeredPagedKVStore, *, owner_id: str) -> None:
        if store.owner_id != owner_id:
            raise RuntimeError("KV store owner_id does not match binding owner")
        if self._store is not None and self._store is not store:
            raise RuntimeError(
                "paged attention backend is already bound to a different KV store"
            )
        if (
            store.num_pages != self.num_gpu_blocks
            or store.num_kv_heads != self.spec.num_kv_heads
            or store.head_dim != self.spec.head_dim
            or store.block_size != self.spec.block_size
            or store.format.name.value != self.spec.format_name
        ):
            raise RuntimeError(
                "KV store shape/format does not match attention backend spec"
            )
        self._store = store
        self._store_owner_id = owner_id
        self._resolve_execution_backend()

    def _resolve_execution_backend(self) -> None:
        if self.spec.format_name == "native":
            self.execution_backend = "native"
            self.effective_format = "native"
            self.format_decision_reason = None
            return
        native_available, native_reason = probe_native_int8_binding()
        sdpa_available = callable(
            getattr(F, "scaled_dot_product_attention", None)
        )
        backend_preference = (
            "flashinfer" if flashinfer_utils.HAS_FLASHINFER else "auto"
        )
        capabilities = KVCacheBackendCapabilities(
            flashinfer_available=bool(flashinfer_utils.HAS_FLASHINFER),
            native_int8_binding_available=native_available,
            sdpa_available=sdpa_available,
            native_int8_unavailable_reason=native_reason,
        )
        decision = resolve_kv_cache_format(
            requested=self.spec.format_name,
            model=KVCacheModelInfo(
                num_attention_heads=self.spec.num_kv_heads,
                num_kv_heads=self.spec.num_kv_heads,
                head_dim=self.spec.head_dim,
                is_mla=False,
            ),
            device=self.device,
            backend_preference=backend_preference,
            capabilities=capabilities,
            allow_fallback=True,
        )
        self.execution_backend = decision.execution_backend
        self.effective_format = decision.effective_format.name.value
        self.format_decision_reason = decision.reason

    def write_chunk(
        self,
        *,
        layer_idx: int,
        key_chunk: torch.Tensor,
        value_chunk: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self.store.write_chunk(
            layer_idx=layer_idx,
            key_chunk=key_chunk,
            value_chunk=value_chunk,
            slot_mapping=slot_mapping,
        )

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
        self._is_int8 = False
        self._layer_idx = layer_idx
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

        x = self.k_cache.shape[-1]
        block_size = self.spec.block_size

        k_src = key.to(device=self.device, dtype=self.k_cache.dtype)
        v_src = value.to(device=self.device, dtype=self.v_cache.dtype)
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

            self.k_cache[block_id, :, :, token_offset, :] = k_src[i].reshape(
                self.spec.num_kv_heads,
                self.spec.head_dim // x,
                x,
            )
            self.v_cache[block_id, :, :, token_offset] = v_src[i]

    def write_kv_flashinfer(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self._fi_kv_cache is None:
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

            self._fi_kv_cache[block_id, 0, token_offset] = k_src[i]
            self._fi_kv_cache[block_id, 1, token_offset] = v_src[i]

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
        layer_idx: Optional[int] = None,
        graph_mode: bool = False,
    ) -> torch.Tensor:
        _ = (kv_cache, graph_mode)
        metadata = (
            attention_metadata
            if attention_metadata is not None
            else attn_metadata
        )
        if metadata is None:
            raise ValueError("attention metadata is required")

        if self._is_int8:
            return self._forward_int8(
                query, key, value, metadata, scale=scale, layer_idx=layer_idx
            )
        self._check_owner(metadata)
        if self.storage is not None and layer_idx is not None:
            self._layer_idx = int(layer_idx)

        if self._is_prefill(metadata):
            slot_mapping = self._get_slot_mapping(metadata)
            if slot_mapping is None:
                raise ValueError("prefill requires slot_mapping")
            self.write_kv(key, value, slot_mapping)
            if self._flashinfer_enabled():
                self.write_kv_flashinfer(key, value, slot_mapping)
            return self._prefill_forward(
                query,
                key,
                value,
                metadata=metadata,
                scale=scale,
            )

        self._write_decode_kv(key, value, metadata)
        return self._decode_forward(query, metadata, scale=scale)

    def _forward_int8(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
        scale: Optional[float],
        layer_idx: int,
    ) -> torch.Tensor:
        if self._is_prefill(metadata):
            slot_mapping = self._get_slot_mapping(metadata)
            if slot_mapping is None:
                raise ValueError("prefill requires slot_mapping")
            self.write_chunk(
                layer_idx=layer_idx,
                key_chunk=key,
                value_chunk=value,
                slot_mapping=slot_mapping,
            )
            return self._prefill_forward(
                query, key, value, metadata=metadata, scale=scale
            )

        return self._decode_forward_int8(
            query, metadata, scale=scale, layer_idx=layer_idx
        )

    def _decode_forward_int8(
        self,
        query: torch.Tensor,
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
        scale: Optional[float],
        layer_idx: int,
    ) -> torch.Tensor:
        if query.ndim == 2:
            query = query.unsqueeze(0)
        if query.ndim != 3:
            raise ValueError(
                "decode query must have shape [batch_size, num_heads, head_dim]"
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

        store = self.store
        if self.execution_backend == "native_int8":
            key_view, value_view, key_scale, value_scale = (
                store.paged_kernel_layer_view(layer_idx)
            )
            return paged_attention_fwd(
                query=query.to(self.device),
                key_cache=key_view,
                value_cache=value_view,
                block_tables=block_tables.to(self.device),
                seq_lens=seq_lens.to(self.device),
                scale=attn_scale,
                num_kv_heads=self.spec.num_kv_heads,
                block_size=self.spec.block_size,
                max_seq_len=max_seq_len,
                key_scale=key_scale,
                value_scale=value_scale,
            )

        return self._decode_sdpa_from_store(
            query, store, block_tables, seq_lens, attn_scale, layer_idx
        )

    def _decode_sdpa_from_store(
        self,
        query: torch.Tensor,
        store: LayeredPagedKVStore,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_scale: float,
        layer_idx: int,
    ) -> torch.Tensor:
        query_dtype = query.dtype
        num_heads = int(query.shape[1])
        num_kv_heads = self.spec.num_kv_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        head_ratio = num_heads // num_kv_heads
        outputs: list[torch.Tensor] = []
        for seq_idx in range(query.shape[0]):
            seq_len = int(seq_lens[seq_idx].item())
            if seq_len <= 0:
                outputs.append(torch.zeros_like(query[seq_idx]))
                continue
            k, v = store.read_prefix(
                layer_idx=layer_idx,
                block_table=block_tables[seq_idx],
                seq_len=seq_len,
                execution_dtype=torch.float32,
            )
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)
            if head_ratio > 1:
                k = k.repeat_interleave(head_ratio, dim=0)
                v = v.repeat_interleave(head_ratio, dim=0)
            q_sdpa = (
                query[seq_idx].to(dtype=torch.float32).unsqueeze(0).unsqueeze(2)
            )
            k_sdpa = k.unsqueeze(0)
            v_sdpa = v.unsqueeze(0)
            out_fp32 = self._run_sdpa_fp32(q_sdpa, k_sdpa, v_sdpa, attn_scale)
            outputs.append(out_fp32[0, :, 0, :].to(dtype=query_dtype))
        return torch.stack(outputs, dim=0)

    @staticmethod
    def _run_sdpa_fp32(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        try:
            return F.scaled_dot_product_attention(
                query, key, value, scale=scale, is_causal=False
            )
        except TypeError:
            return F.scaled_dot_product_attention(
                query * scale, key, value, is_causal=False
            )

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
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                "prefill query/key/value must have shape [num_tokens, num_heads, head_dim]"
            )

        if self._flashinfer_enabled():
            if metadata is None:
                raise ValueError("prefill requires attention metadata")
            if self._fi_prefill is None or self._fi_kv_cache is None:
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
            )
            return cast(
                torch.Tensor,
                self._fi_prefill.run(query_src, self._fi_kv_cache),
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
    ) -> torch.Tensor:
        if query.ndim == 2:
            query = query.unsqueeze(0)
        if query.ndim != 3:
            raise ValueError(
                "decode query must have shape [batch_size, num_heads, head_dim]"
            )

        if self._flashinfer_enabled():
            if self._fi_decode is None or self._fi_kv_cache is None:
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
                self._fi_decode.run(query_src, self._fi_kv_cache),
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
            key_cache=self.k_cache,
            value_cache=self.v_cache,
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

    def _build_flashinfer_metadata(
        self,
        metadata: AttentionMetadata | RuntimeAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        block_tables = self._get_block_tables(metadata)
        seq_lens = self._get_seq_lens(metadata)
        if block_tables is None or seq_lens is None:
            raise ValueError(
                "FlashInfer attention requires block_tables and seq_lens"
            )

        block_tables_i32 = block_tables.to(self.device, dtype=torch.int32)
        seq_lens_i32 = seq_lens.to(self.device, dtype=torch.int32).reshape(-1)
        batch_size = int(seq_lens_i32.shape[0])

        qo_indptr = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=self.device,
        )
        if batch_size > 0:
            qo_indptr[1:] = torch.cumsum(seq_lens_i32, dim=0)

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
            seq_len = int(seq_lens_i32[i].item())
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

        return qo_indptr, kv_indptr, kv_indices, kv_last_page_len

    def _call_prefill_plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
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
