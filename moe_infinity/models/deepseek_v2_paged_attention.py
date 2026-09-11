"""DeepSeek-V2/V3 MLA paged-attention shims (DeepSeek analogue of Qwen3PagedAttention).

MLA keys/values have different head dims: key qk_head_dim=qk_nope+qk_rope (192 for
V2-Lite), value v_head_dim (128). The symmetric paged cache needs key.shape==
value.shape, so value is zero-padded 128->192 before write_kv and the attention
output is sliced back to v_head_dim before o_proj -- the same loss-less pad/slice
HuggingFace applies on its flash-attention path.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Protocol, Union, cast

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    apply_rotary_emb,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
)

from moe_infinity.runtime.attention_backend import AttentionMetadata
from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)

_Metadata = Union[AttentionMetadata, RuntimeAttentionMetadata]


class _SupportsPagedAttention(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[_Metadata] = None,
        scale: Optional[float] = None,
        attention_metadata: Optional[_Metadata] = None,
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]: ...


# Fast paged decode/prefill kernels (FlashInfer wrappers and the compiled
# paged_attention_v1) are only instantiated for these head dims. MLA's real
# qk_head_dim is 192 (unsupported -> FlashInfer stalls, the .cu kernel is
# wrong), so the symmetric cache is zero-padded up to the next supported size
# (256). Zero dims contribute nothing to q.k scores or to the value-weighted
# output, so this is numerically identical while enabling the fast kernels.
_PAGED_SUPPORTED_HEAD_DIMS = (64, 128, 256)


def _paged_cache_head_dim(qk_head_dim: int) -> int:
    for dim in _PAGED_SUPPORTED_HEAD_DIMS:
        if dim >= qk_head_dim:
            return dim
    return qk_head_dim


def _mla_kv_cache_spec(config: object) -> dict[str, int]:
    qk_nope = getattr(config, "qk_nope_head_dim", None)
    qk_rope = getattr(config, "qk_rope_head_dim", None)
    num_heads = getattr(config, "num_attention_heads", None)
    if qk_nope is not None and qk_rope is not None and num_heads is not None:
        return {
            "num_kv_heads": int(num_heads),
            "head_dim": _paged_cache_head_dim(int(qk_nope) + int(qk_rope)),
        }

    num_kv_heads = getattr(config, "num_key_value_heads", None) or num_heads
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None or not num_heads:
            raise ValueError(
                "unable to derive MLA kv-cache head_dim from config"
            )
        head_dim = int(hidden_size) // int(num_heads)
    if num_kv_heads is None:
        raise ValueError("unable to derive num_kv_heads from config")
    return {"num_kv_heads": int(num_kv_heads), "head_dim": int(head_dim)}


def _valid_token_index(
    metadata: object,
    bsz: int,
    q_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    lengths = getattr(metadata, "lengths", None)
    query_lengths = (
        getattr(lengths, "query_lengths", None) if lengths is not None else None
    )
    if query_lengths is None:
        return None
    per_seq = [int(v) for v in query_lengths.reshape(-1).tolist()]
    if len(per_seq) != bsz or all(length == q_len for length in per_seq):
        return None
    indices = [
        row * q_len + col
        for row, length in enumerate(per_seq)
        for col in range(length)
    ]
    return torch.tensor(indices, dtype=torch.long, device=device)


def _run_paged_mla(
    paged_backend: _SupportsPagedAttention,
    attention_metadata: _Metadata,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    qk_head_dim: int,
    v_head_dim: int,
    scaling: float,
    o_proj: torch.nn.Module,
    layer_idx: int,
) -> torch.Tensor:
    batch_size, num_heads, seq_length, _ = query_states.shape
    cache_head_dim = _paged_cache_head_dim(qk_head_dim)

    query_padded = F.pad(query_states, [0, cache_head_dim - qk_head_dim])
    key_padded = F.pad(key_states, [0, cache_head_dim - qk_head_dim])
    value_padded = F.pad(value_states, [0, cache_head_dim - v_head_dim])

    query_tokens = (
        query_padded.transpose(1, 2)
        .contiguous()
        .view(-1, num_heads, cache_head_dim)
    )
    key_tokens = (
        key_padded.transpose(1, 2)
        .contiguous()
        .view(-1, num_heads, cache_head_dim)
    )
    value_tokens = (
        value_padded.transpose(1, 2)
        .contiguous()
        .view(-1, num_heads, cache_head_dim)
    )

    valid_index = _valid_token_index(
        attention_metadata, batch_size, seq_length, query_tokens.device
    )
    packed = valid_index is not None
    if packed:
        query_tokens = query_tokens.index_select(0, valid_index)
        key_tokens = key_tokens.index_select(0, valid_index)
        value_tokens = value_tokens.index_select(0, valid_index)

    attn_output_tokens = paged_backend.forward(
        query_tokens,
        key_tokens,
        value_tokens,
        attention_metadata=attention_metadata,
        scale=cast(float, scaling),
        layer_idx=layer_idx,
    )

    if attn_output_tokens is None or attn_output_tokens.ndim != 3:
        raise ValueError("paged attention backend must return rank-3 tensor")

    if packed:
        unpacked = attn_output_tokens.new_zeros(
            (batch_size * seq_length, num_heads, cache_head_dim)
        )
        unpacked.index_copy_(0, valid_index, attn_output_tokens)
        attn_output_tokens = unpacked

    attn_output = attn_output_tokens[..., :v_head_dim].reshape(
        batch_size, seq_length, num_heads * v_head_dim
    )
    return cast(torch.Tensor, o_proj(attn_output))


class DeepseekV2PagedAttention(DeepseekV2Attention):
    _paged_backend: ClassVar[Optional[_SupportsPagedAttention]] = None
    _attention_metadata: ClassVar[Optional[_Metadata]] = None

    @classmethod
    def set_paged_context(
        cls,
        backend: _SupportsPagedAttention,
        metadata: _Metadata,
    ) -> None:
        cls._paged_backend = backend
        cls._attention_metadata = metadata

    @classmethod
    def clear_paged_context(cls) -> None:
        cls._paged_backend = None
        cls._attention_metadata = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        paged_backend = self.__class__._paged_backend
        attention_metadata = self.__class__._attention_metadata
        if paged_backend is None or attention_metadata is None:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if position_embeddings is None:
            raise ValueError(
                "position_embeddings are required for paged MLA attention"
            )

        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (
            batch_size,
            seq_length,
            -1,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(query_shape).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_nope, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_nope = (
            self.kv_b_proj(self.kv_a_layernorm(k_nope))
            .view(key_shape)
            .transpose(1, 2)
        )
        k_nope, value_states = torch.split(
            k_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        q_pe, k_pe = apply_rotary_emb(
            q_pe, k_pe, position_embeddings.to(q_pe.device)
        )
        k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat((k_nope, k_pe), dim=-1)

        attn_output = _run_paged_mla(
            paged_backend,
            attention_metadata,
            query_states,
            key_states,
            value_states,
            self.qk_head_dim,
            self.v_head_dim,
            cast(float, self.scaling),
            self.o_proj,
            int(self.layer_idx or 0),
        )
        return attn_output, None

    @classmethod
    def get_kv_cache_spec_for_config(cls, config: object) -> dict[str, int]:
        return _mla_kv_cache_spec(config)


class DeepseekV3PagedAttention(DeepseekV3Attention):
    _paged_backend: ClassVar[Optional[_SupportsPagedAttention]] = None
    _attention_metadata: ClassVar[Optional[_Metadata]] = None

    @classmethod
    def set_paged_context(
        cls,
        backend: _SupportsPagedAttention,
        metadata: _Metadata,
    ) -> None:
        cls._paged_backend = backend
        cls._attention_metadata = metadata

    @classmethod
    def clear_paged_context(cls) -> None:
        cls._paged_backend = None
        cls._attention_metadata = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        paged_backend = self.__class__._paged_backend
        attention_metadata = self.__class__._attention_metadata
        if paged_backend is None or attention_metadata is None:
            return super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (
            batch_size,
            seq_length,
            -1,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pass = (
            self.kv_b_proj(self.kv_a_layernorm(k_pass))
            .view(key_shape)
            .transpose(1, 2)
        )
        k_pass, value_states = torch.split(
            k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(
                q_rot, k_rot, cos, sin
            )
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        attn_output = _run_paged_mla(
            paged_backend,
            attention_metadata,
            query_states,
            key_states,
            value_states,
            self.qk_head_dim,
            self.v_head_dim,
            cast(float, self.scaling),
            self.o_proj,
            int(self.layer_idx or 0),
        )
        return attn_output, None

    @classmethod
    def get_kv_cache_spec_for_config(cls, config: object) -> dict[str, int]:
        return _mla_kv_cache_spec(config)


__all__ = ["DeepseekV2PagedAttention", "DeepseekV3PagedAttention"]
