from __future__ import annotations

import importlib
import logging
from typing import Optional, Protocol, cast

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _PagedAttnModule(Protocol):
    def paged_attention_v1(
        self,
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        max_seq_len: int,
    ) -> None: ...


def _load_paged_attn_module() -> tuple[_PagedAttnModule | None, bool]:
    try:
        module = importlib.import_module("moe_infinity._paged_attn")
        return cast(_PagedAttnModule, cast(object, module)), True
    except ImportError:
        return None, False


_paged_attn_ops, HAS_PAGED_ATTN = _load_paged_attn_module()
if not HAS_PAGED_ATTN:
    logger.warning(
        "PagedAttention CUDA kernel not available, falling back to torch SDPA"
    )


def probe_native_int8_binding() -> tuple[bool, str | None]:
    if _paged_attn_ops is None:
        return False, "native_int8_module_unavailable"
    binding = getattr(_paged_attn_ops, "paged_attention_int8_v1", None)
    if not callable(binding):
        return False, "native_int8_binding_missing"
    return True, None


def paged_attention_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    block_size: int = 16,
    max_seq_len: Optional[int] = None,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if query.ndim != 3:
        raise ValueError(
            "query must have shape [num_seqs, num_heads, head_dim]"
        )
    if block_tables.ndim != 2:
        raise ValueError(
            "block_tables must have shape [num_seqs, max_blocks_per_seq]"
        )
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must have shape [num_seqs]")

    if (key_scale is None) != (value_scale is None):
        raise ValueError(
            "key_scale and value_scale must both be provided or both be None"
        )

    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0

    is_int8 = key_scale is not None and value_scale is not None

    if is_int8:
        if key_cache.dtype != torch.int8 or value_cache.dtype != torch.int8:
            raise ValueError(
                "int8 paged attention requires int8 key/value payloads"
            )
        native_int8 = getattr(_paged_attn_ops, "paged_attention_int8_v1", None)
        if (
            HAS_PAGED_ATTN
            and callable(native_int8)
            and query.is_cuda
            and key_cache.is_cuda
            and value_cache.is_cuda
            and key_scale.is_cuda
            and value_scale.is_cuda
            and block_tables.is_cuda
            and seq_lens.is_cuda
        ):
            out = torch.zeros_like(query)
            native_int8(
                out,
                query,
                key_cache,
                value_cache,
                key_scale,
                value_scale,
                int(num_kv_heads),
                float(scale),
                block_tables.int(),
                seq_lens.int(),
                int(block_size),
                int(max_seq_len),
            )
            return out

        return _paged_attention_sdpa_fallback(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            scale=scale,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            key_scale=key_scale,
            value_scale=value_scale,
        )

    if (
        HAS_PAGED_ATTN
        and _paged_attn_ops is not None
        and query.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and block_tables.is_cuda
        and seq_lens.is_cuda
    ):
        out = torch.zeros_like(query)
        _paged_attn_ops.paged_attention_v1(
            out,
            query,
            key_cache,
            value_cache,
            int(num_kv_heads),
            float(scale),
            block_tables.int(),
            seq_lens.int(),
            int(block_size),
            int(max_seq_len),
        )
        return out

    return _paged_attention_sdpa_fallback(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
    )


def _run_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    try:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
            is_causal=False,
        )
    except TypeError:
        return F.scaled_dot_product_attention(
            query * scale,
            key,
            value,
            is_causal=False,
        )


def _paged_attention_sdpa_fallback(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    block_size: int,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, num_heads, _ = query.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    is_int8 = key_scale is not None and value_scale is not None
    query_dtype = query.dtype
    head_ratio = num_heads // num_kv_heads
    outputs: list[torch.Tensor] = []
    for seq_idx in range(batch_size):
        seq_len = int(seq_lens[seq_idx].item())
        q = query[seq_idx : seq_idx + 1]

        if seq_len <= 0:
            outputs.append(torch.zeros_like(q))
            continue

        num_blocks = (seq_len + block_size - 1) // block_size
        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []

        for b in range(num_blocks):
            phys_block = int(block_tables[seq_idx, b].item())
            tokens_in_block = min(block_size, seq_len - b * block_size)

            k_block = key_cache[phys_block]
            v_block = value_cache[phys_block]
            nkv, hd_x, _, x = k_block.shape
            k_block = k_block.permute(0, 2, 1, 3).reshape(
                nkv, block_size, hd_x * x
            )
            k_block = k_block[:, :tokens_in_block, :]
            v_block = v_block[:, :, :tokens_in_block].permute(0, 2, 1)

            if is_int8:
                k_block = k_block.to(torch.float32)
                v_block = v_block.to(torch.float32)
                ks = (
                    key_scale[phys_block, :, :tokens_in_block]
                    .to(torch.float32)
                    .unsqueeze(-1)
                )
                vs = (
                    value_scale[phys_block, :, :tokens_in_block]
                    .to(torch.float32)
                    .unsqueeze(-1)
                )
                k_block = k_block * ks
                v_block = v_block * vs

            k_list.append(k_block)
            v_list.append(v_block)

        k = torch.cat(k_list, dim=1)
        v = torch.cat(v_list, dim=1)
        if head_ratio > 1:
            k = k.repeat_interleave(head_ratio, dim=0)
            v = v.repeat_interleave(head_ratio, dim=0)

        if is_int8:
            q_sdpa = q.squeeze(0).to(torch.float32).unsqueeze(0).unsqueeze(2)
            k_sdpa = k.unsqueeze(0)
            v_sdpa = v.unsqueeze(0)
            out_fp32 = _run_sdpa(q_sdpa, k_sdpa, v_sdpa, float(scale))
            outputs.append(out_fp32[0, :, 0, :].to(query_dtype).unsqueeze(0))
        else:
            q_sdpa = q.squeeze(0).unsqueeze(1)
            out = _run_sdpa(q_sdpa, k, v, float(scale))
            outputs.append(out.squeeze(1).unsqueeze(0))

    return torch.cat(outputs, dim=0)
