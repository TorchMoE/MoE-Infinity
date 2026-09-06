from __future__ import annotations

from typing import ClassVar, Optional, Protocol, Union, cast

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3_moe.configuration_qwen3_moe import (
    Qwen3MoeConfig,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    apply_rotary_pos_emb,
)
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

from moe_infinity.runtime.attention_backend import AttentionMetadata
from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)


class _SupportsPagedAttention(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[
            Union[AttentionMetadata, RuntimeAttentionMetadata]
        ] = None,
        scale: Optional[float] = None,
        attention_metadata: Optional[
            Union[AttentionMetadata, RuntimeAttentionMetadata]
        ] = None,
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]: ...


class Qwen3PagedAttention(Qwen3MoeAttention):
    _paged_backend: ClassVar[Optional[_SupportsPagedAttention]] = None
    _attention_metadata: ClassVar[
        Optional[Union[AttentionMetadata, RuntimeAttentionMetadata]]
    ] = None

    @classmethod
    def set_paged_context(
        cls,
        backend: _SupportsPagedAttention,
        metadata: Union[AttentionMetadata, RuntimeAttentionMetadata],
    ) -> None:
        cls._paged_backend = backend
        cls._attention_metadata = metadata

    @classmethod
    def clear_paged_context(cls) -> None:
        cls._paged_backend = None
        cls._attention_metadata = None

    @deprecate_kwarg(
        "past_key_value", new_name="past_key_values", version="4.58"
    )
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        paged_backend = self.__class__._paged_backend
        attention_metadata = self.__class__._attention_metadata
        if paged_backend is None or attention_metadata is None:
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = cast(
            torch.Tensor,
            self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2),
        )
        key_states = cast(
            torch.Tensor,
            self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2),
        )
        value_states = cast(
            torch.Tensor,
            self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2),
        )

        cos, sin = position_embeddings
        query_states, key_states = cast(
            tuple[torch.Tensor, torch.Tensor],
            apply_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
            ),
        )

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        bsz, q_len = int(input_shape[0]), int(input_shape[1])
        num_attention_heads = query_states.shape[1]
        num_key_value_heads = key_states.shape[1]

        query_tokens = (
            query_states.transpose(1, 2)
            .contiguous()
            .view(-1, num_attention_heads, self.head_dim)
        )
        key_tokens = (
            key_states.transpose(1, 2)
            .contiguous()
            .view(-1, num_key_value_heads, self.head_dim)
        )
        value_tokens = (
            value_states.transpose(1, 2)
            .contiguous()
            .view(-1, num_key_value_heads, self.head_dim)
        )

        attn_output_tokens = paged_backend.forward(
            query_tokens,
            key_tokens,
            value_tokens,
            attention_metadata=attention_metadata,
            scale=cast(float, self.scaling),
            layer_idx=int(self.layer_idx),
        )
        try:
            attn_output_tokens = paged_backend.forward(
                query_tokens,
                key_tokens,
                value_tokens,
                attention_metadata=attention_metadata,
                scale=cast(float, self.scaling),
                layer_idx=self.layer_idx,
            )
        except TypeError:
            attn_output_tokens = paged_backend.forward(
                query_tokens,
                key_tokens,
                value_tokens,
                attention_metadata=attention_metadata,
                scale=cast(float, self.scaling),
            )

        if attn_output_tokens is None or attn_output_tokens.ndim != 3:
            raise ValueError(
                "paged attention backend must return rank-3 tensor"
            )

        attn_output: torch.Tensor = attn_output_tokens.view(
            bsz,
            q_len,
            num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        if attn_output.size() != (
            bsz,
            num_attention_heads,
            q_len,
            self.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_attention_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            bsz,
            q_len,
            num_attention_heads * self.head_dim,
        )
        attn_output = cast(torch.Tensor, self.o_proj(attn_output))

        attn_weights = None
        return attn_output, attn_weights

    @classmethod
    def get_kv_cache_spec_for_config(
        cls, config: Qwen3MoeConfig
    ) -> dict[str, int]:
        return {
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": config.hidden_size // config.num_attention_heads,
        }
