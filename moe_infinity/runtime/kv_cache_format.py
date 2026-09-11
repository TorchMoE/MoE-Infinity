from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class KVCacheFormatName(str, Enum):
    NATIVE = "native"
    INT8_SYM = "int8_sym"


@dataclass(frozen=True)
class KVCacheFormat:
    name: KVCacheFormatName
    payload_dtype: torch.dtype | None
    scale_dtype: torch.dtype | None

    @classmethod
    def parse(cls, value: str) -> "KVCacheFormat":
        if value == "native":
            return cls(KVCacheFormatName.NATIVE, None, None)
        if value == "int8_sym":
            return cls(KVCacheFormatName.INT8_SYM, torch.int8, torch.float16)
        raise ValueError(f"unsupported KV cache format: {value}")

    def page_size_bytes(
        self,
        *,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        execution_dtype: torch.dtype = torch.float16,
    ) -> int:
        values = 2 * block_size * num_kv_heads * head_dim
        if self.name is KVCacheFormatName.NATIVE:
            return (
                values * torch.empty((), dtype=execution_dtype).element_size()
            )
        scale_values = 2 * block_size * num_kv_heads
        return values + scale_values * 2


@dataclass(frozen=True)
class KVCacheModelInfo:
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    is_mla: bool


def model_info_from_config(config: object) -> KVCacheModelInfo:
    text_config = config
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            text_config = getter()
        except Exception:
            text_config = config
    num_attention_heads = int(
        getattr(text_config, "num_attention_heads", 0) or 0
    )
    num_kv_heads = int(
        getattr(text_config, "num_key_value_heads", None)
        or num_attention_heads
        or 1
    )
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None and num_attention_heads > 0:
        hidden = int(getattr(text_config, "hidden_size", 0) or 0)
        head_dim = hidden // num_attention_heads if num_attention_heads else 0
    head_dim = int(head_dim or 0)
    kv_lora_rank = getattr(text_config, "kv_lora_rank", None)
    is_mla = (
        kv_lora_rank is not None
        or getattr(text_config, "qk_nope_head_dim", None) is not None
        or getattr(text_config, "qk_rope_head_dim", None) is not None
    )
    return KVCacheModelInfo(
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_mla=bool(is_mla),
    )


@dataclass(frozen=True)
class KVCacheFormatDecision:
    requested_format: KVCacheFormat
    effective_format: KVCacheFormat
    execution_backend: str
    reason: str | None


@dataclass(frozen=True)
class KVCacheBackendCapabilities:
    flashinfer_available: bool
    native_int8_binding_available: bool
    sdpa_available: bool
    native_int8_unavailable_reason: str | None


def resolve_kv_cache_format(
    *,
    requested: str,
    model: KVCacheModelInfo,
    device: torch.device,
    backend_preference: str,
    capabilities: KVCacheBackendCapabilities,
    allow_fallback: bool,
) -> KVCacheFormatDecision:
    requested_format = KVCacheFormat.parse(requested)
    native = KVCacheFormat.parse("native")
    if requested_format.name is KVCacheFormatName.NATIVE:
        return KVCacheFormatDecision(requested_format, native, "existing", None)
    format_failure = None
    if model.is_mla:
        format_failure = "mla_not_validated"
    elif model.num_attention_heads % model.num_kv_heads != 0:
        format_failure = "invalid_gqa_ratio"
    elif model.head_dim % 8 != 0:
        format_failure = "head_dim_not_divisible_by_8"
    if format_failure is not None:
        if not allow_fallback:
            raise RuntimeError(
                f"KV cache format int8_sym unavailable: {format_failure}"
            )
        return KVCacheFormatDecision(
            requested_format, native, "existing", format_failure
        )
    flashinfer_reason = (
        "flashinfer_no_int8_sym_contract"
        if backend_preference == "flashinfer"
        and capabilities.flashinfer_available
        else None
    )
    if device.type == "cuda" and capabilities.native_int8_binding_available:
        return KVCacheFormatDecision(
            requested_format, requested_format, "native_int8", flashinfer_reason
        )
    if capabilities.sdpa_available:
        reason = (
            "cpu_sdpa_dequant"
            if device.type != "cuda"
            else capabilities.native_int8_unavailable_reason
            or "native_int8_binding_missing"
        )
        return KVCacheFormatDecision(
            requested_format,
            requested_format,
            "sdpa_dequant",
            flashinfer_reason or reason,
        )
    if not allow_fallback:
        raise RuntimeError(
            "KV cache format int8_sym unavailable: no_int8_execution_backend"
        )
    return KVCacheFormatDecision(
        requested_format, native, "existing", "no_int8_execution_backend"
    )


def quantize_tokenwise_symmetric(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.all(torch.isfinite(x)):
        raise ValueError("KV cache input must contain only finite values")
    amax = x.float().abs().amax(dim=-1)
    raw_scale = amax / 127.0
    zero = torch.zeros((), dtype=torch.float16, device=x.device)
    inf = torch.full((), float("inf"), dtype=torch.float16, device=x.device)
    min_scale = torch.nextafter(zero, inf).float()
    raw_scale = torch.clamp(raw_scale, min=min_scale)
    candidate = raw_scale.to(torch.float16)
    scale = torch.where(
        candidate.float() < raw_scale,
        torch.nextafter(candidate, inf),
        candidate,
    )
    if not torch.all(torch.isfinite(scale)):
        raise ValueError("KV values require an unrepresentable FP16 scale")
    q = torch.clamp(
        torch.round(x.float() / scale.float().unsqueeze(-1)), -127, 127
    )
    return q.to(torch.int8), scale


@dataclass(frozen=True)
class LayeredKVPageChunk:
    page_ids: tuple[int, ...]
    format: KVCacheFormat
    payload: torch.Tensor
    scales: torch.Tensor | None = None


@dataclass
class FormatLayeredKVStore:
    owner_id: str
    format: KVCacheFormat
    num_layers: int
    num_pages: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    execution_dtype: torch.dtype
    payload: torch.Tensor
    scales: torch.Tensor | None = None

    @property
    def tensors(
        self,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        if self.scales is not None:
            return self.payload, self.scales
        return (self.payload,)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors)

    def write_chunk(
        self,
        *,
        layer_idx: int,
        key_chunk: torch.Tensor,
        value_chunk: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        expected = (key_chunk.shape[0], self.num_kv_heads, self.head_dim)
        if key_chunk.shape != expected or value_chunk.shape != expected:
            raise ValueError(f"K/V chunk must have shape {expected}")
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx outside FormatLayeredKVStore")
        slots = slot_mapping.to(self.payload.device, dtype=torch.long)
        pages, offsets = slots // self.block_size, slots % self.block_size
        if bool(torch.any(slots < 0)) or bool(
            torch.any(pages >= self.num_pages)
        ):
            raise ValueError("slot_mapping points outside allocated KV pages")
        key_chunk = key_chunk.to(
            self.payload.device, dtype=self.execution_dtype
        )
        value_chunk = value_chunk.to(
            self.payload.device, dtype=self.execution_dtype
        )
        if self.format.name is KVCacheFormatName.NATIVE:
            self.payload[layer_idx, pages, 0, offsets] = key_chunk
            self.payload[layer_idx, pages, 1, offsets] = value_chunk
            return
        if self.scales is None:
            raise RuntimeError("int8_sym storage requires K/V scales")
        key_q, key_scale = quantize_tokenwise_symmetric(key_chunk)
        value_q, value_scale = quantize_tokenwise_symmetric(value_chunk)
        self.payload[layer_idx, pages, 0, offsets] = key_q
        self.payload[layer_idx, pages, 1, offsets] = value_q
        self.scales[layer_idx, pages, 0, offsets] = key_scale
        self.scales[layer_idx, pages, 1, offsets] = value_scale

    def read_prefix(
        self,
        *,
        layer_idx: int,
        block_table: torch.Tensor,
        seq_len: int,
        execution_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _read_layer_prefix(
            self, layer_idx, block_table, seq_len, execution_dtype
        )

    def snapshot_page_chunk(
        self, page_ids: list[int], target_device: torch.device
    ) -> LayeredKVPageChunk:
        return _snapshot_page_chunk_blocking(self, page_ids, target_device)

    def restore_page_chunk(
        self, destination_page_ids: list[int], chunk: LayeredKVPageChunk
    ) -> None:
        _restore_page_chunk_blocking(self, destination_page_ids, chunk)

    def paged_kernel_layer_view(
        self, layer_idx: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        layer = self.payload[layer_idx]
        key = (
            layer[:, 0]
            .reshape(
                self.num_pages,
                self.block_size,
                self.num_kv_heads,
                self.head_dim // 8,
                8,
            )
            .permute(0, 2, 3, 1, 4)
        )
        value = layer[:, 1].permute(0, 2, 3, 1)
        if self.scales is None:
            return key, value, None, None
        return (
            key,
            value,
            self.scales[layer_idx, :, 0].permute(0, 2, 1),
            self.scales[layer_idx, :, 1].permute(0, 2, 1),
        )

    def flashinfer_layer_view(self, layer_idx: int) -> torch.Tensor:
        if self.format.name is not KVCacheFormatName.NATIVE:
            raise RuntimeError(
                "FlashInfer layer view requires native KV format"
            )
        return self.payload[layer_idx]


def _read_layer_prefix(
    store: FormatLayeredKVStore,
    layer_idx: int,
    block_table: torch.Tensor,
    seq_len: int,
    execution_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_count = (seq_len + store.block_size - 1) // store.block_size
    page_ids = block_table[:page_count].to(
        store.payload.device, dtype=torch.long
    )
    pages = store.payload[layer_idx].index_select(0, page_ids)
    key = pages[:, 0].reshape(
        page_count * store.block_size, store.num_kv_heads, store.head_dim
    )[:seq_len]
    value = pages[:, 1].reshape(
        page_count * store.block_size, store.num_kv_heads, store.head_dim
    )[:seq_len]
    if store.scales is None:
        return key.to(execution_dtype), value.to(execution_dtype)
    scales = store.scales[layer_idx].index_select(0, page_ids)
    key_scale = scales[:, 0].reshape(
        page_count * store.block_size, store.num_kv_heads
    )[:seq_len]
    value_scale = scales[:, 1].reshape(
        page_count * store.block_size, store.num_kv_heads
    )[:seq_len]
    return (
        key.to(execution_dtype) * key_scale.to(execution_dtype).unsqueeze(-1),
        value.to(execution_dtype)
        * value_scale.to(execution_dtype).unsqueeze(-1),
    )


def _snapshot_page_chunk_blocking(
    store: FormatLayeredKVStore,
    page_ids: list[int],
    target_device: torch.device,
) -> LayeredKVPageChunk:
    index = torch.tensor(
        page_ids, device=store.payload.device, dtype=torch.long
    )
    payload = (
        store.payload.index_select(1, index)
        .to(target_device, non_blocking=False)
        .clone()
    )
    scales = None
    if store.scales is not None:
        scales = (
            store.scales.index_select(1, index)
            .to(target_device, non_blocking=False)
            .clone()
        )
    return LayeredKVPageChunk(tuple(page_ids), store.format, payload, scales)


def _restore_page_chunk_blocking(
    store: FormatLayeredKVStore,
    destination_page_ids: list[int],
    chunk: LayeredKVPageChunk,
) -> None:
    if store.format != chunk.format or len(destination_page_ids) != len(
        chunk.page_ids
    ):
        raise ValueError(
            "page chunk format/count does not match destination store"
        )
    expected = (
        store.num_layers,
        len(destination_page_ids),
        2,
        store.block_size,
        store.num_kv_heads,
        store.head_dim,
    )
    if tuple(chunk.payload.shape) != expected:
        raise ValueError(f"page chunk payload must have shape {expected}")
    index = torch.tensor(
        destination_page_ids, device=store.payload.device, dtype=torch.long
    )
    store.payload.index_copy_(
        1, index, chunk.payload.to(store.payload.device, non_blocking=False)
    )
    if store.scales is not None:
        if chunk.scales is None:
            raise ValueError("INT8 page chunk is missing scales")
        store.scales.index_copy_(
            1, index, chunk.scales.to(store.scales.device, non_blocking=False)
        )


def allocate_layered_paged_kv_store(
    *,
    owner_id: str,
    format_name: str,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    execution_dtype: torch.dtype,
    device: torch.device,
) -> FormatLayeredKVStore:
    if not owner_id:
        raise ValueError("FormatLayeredKVStore requires a non-empty owner_id")
    for name, dim in (
        ("num_layers", num_layers),
        ("num_blocks", num_blocks),
        ("block_size", block_size),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        if dim <= 0:
            raise ValueError(f"{name} must be positive, got {dim}")
    fmt = KVCacheFormat.parse(format_name)
    if fmt.name is KVCacheFormatName.INT8_SYM and head_dim % 8 != 0:
        raise ValueError("int8_sym requires head_dim divisible by 8")
    payload_shape = (
        num_layers,
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        head_dim,
    )
    if fmt.name is KVCacheFormatName.NATIVE:
        payload = torch.zeros(
            payload_shape, dtype=execution_dtype, device=device
        )
        scales = None
    else:
        payload = torch.zeros(
            payload_shape, dtype=fmt.payload_dtype, device=device
        )
        scales = torch.zeros(
            (num_layers, num_blocks, 2, block_size, num_kv_heads),
            dtype=fmt.scale_dtype,
            device=device,
        )
    return FormatLayeredKVStore(
        owner_id=owner_id,
        format=fmt,
        num_layers=num_layers,
        num_pages=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        execution_dtype=execution_dtype,
        payload=payload,
        scales=scales,
    )
