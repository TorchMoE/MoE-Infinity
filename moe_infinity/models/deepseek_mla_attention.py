from __future__ import annotations

# ruff: noqa: I001

from typing import Any, cast

import torch
import torch.nn.functional as F

try:
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2Attention,
        apply_rotary_emb as apply_v2_rotary,
    )
except (ImportError, AttributeError):
    DeepseekV2Attention = None  # type: ignore[assignment,misc]
    apply_v2_rotary = None

try:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3Attention,
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_interleave,
    )
except (ImportError, AttributeError):
    DeepseekV3Attention = None  # type: ignore[assignment,misc]
    apply_rotary_pos_emb = None
    apply_rotary_pos_emb_interleave = None

from moe_infinity.runtime.attention_types import AttentionMetadata
from moe_infinity.serving.mla_cache import MLAPagedKVCache


class _MLAPagedAttentionMixin:
    _mla_cache: MLAPagedKVCache
    _mla_metadata: AttentionMetadata | None = None

    def _mla_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Any,
        attention_mask: torch.Tensor | None,
        *,
        version: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        metadata = self._mla_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek MLA paged context is not set")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError(
                "DeepSeek MLA paged attention supports batch size 1"
            )
        if self.layer_idx is None:
            raise ValueError("DeepSeek MLA paged attention requires layer_idx")

        batch_size, query_len, _ = hidden_states.shape
        if metadata.seq_lens.numel() != 1:
            raise ValueError(
                "DeepSeek MLA paged attention requires one seq_len"
            )
        total_len = int(metadata.seq_lens.reshape(-1)[0].item())
        if total_len < query_len:
            raise ValueError(
                f"total_len {total_len} must be >= query_len {query_len}"
            )
        if (
            metadata.slot_mapping.ndim != 1
            or metadata.slot_mapping.numel() != query_len
        ):
            raise ValueError(
                f"slot_mapping must contain query_len {query_len} entries"
            )
        if attention_mask is not None and attention_mask.shape[-1] < total_len:
            raise ValueError(
                f"attention_mask last dimension {attention_mask.shape[-1]} "
                f"must be >= total_len {total_len}"
            )
        block_table = metadata.block_tables.reshape(-1)
        if metadata.seq_id is None:
            raise ValueError(
                "engine-owned DeepSeek MLA cache access requires metadata.seq_id"
            )
        self._mla_cache.validate_owned_access(
            metadata.seq_id,
            block_table,
            metadata.slot_mapping,
            total_len,
        )
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )
        q_states = q_states.view(
            batch_size, query_len, self.num_heads, self.qk_head_dim
        ).transpose(1, 2)
        q_nope, q_rope = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed = self.kv_a_proj_with_mqa(hidden_states)
        latent, k_rope = torch.split(
            compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        latent = self.kv_a_layernorm(latent)
        k_rope = k_rope.view(batch_size, 1, query_len, self.qk_rope_head_dim)
        if version == "v2":
            if apply_v2_rotary is None:
                raise RuntimeError(
                    "Transformers DeepSeek V2 rotary helper unavailable"
                )
            q_rope, k_rope = apply_v2_rotary(
                q_rope, k_rope, position_embeddings.to(q_rope.device)
            )
        else:
            cos, sin = position_embeddings
            if self.config.rope_interleave:
                if apply_rotary_pos_emb_interleave is None:
                    raise RuntimeError(
                        "Transformers DeepSeek V3 interleaved RoPE unavailable"
                    )
                q_rope, k_rope = apply_rotary_pos_emb_interleave(
                    q_rope, k_rope, cos, sin
                )
            else:
                if apply_rotary_pos_emb is None:
                    raise RuntimeError(
                        "Transformers DeepSeek V3 RoPE unavailable"
                    )
                q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        self._mla_cache.write(
            int(self.layer_idx),
            latent.reshape(query_len, self.kv_lora_rank),
            k_rope.reshape(query_len, self.qk_rope_head_dim),
            metadata.slot_mapping,
        )
        cached_latent, cached_rope = self._mla_cache.read(
            int(self.layer_idx), block_table, total_len
        )

        expanded = (
            self.kv_b_proj(cached_latent)
            .view(
                1,
                total_len,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            .transpose(1, 2)
        )
        k_nope, values = torch.split(
            expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        keys = torch.cat(
            (
                k_nope,
                cached_rope.view(1, 1, total_len, self.qk_rope_head_dim).expand(
                    1, self.num_heads, total_len, self.qk_rope_head_dim
                ),
            ),
            dim=-1,
        )
        queries = torch.cat((q_nope, q_rope), dim=-1)
        scores = torch.matmul(queries, keys.transpose(2, 3)) * float(
            self.scaling
        )
        past_len = total_len - query_len
        causal = torch.arange(total_len, device=scores.device).view(1, 1, 1, -1)
        limits = past_len + torch.arange(query_len, device=scores.device).view(
            1, 1, -1, 1
        )
        scores = scores.masked_fill(
            causal > limits, torch.finfo(scores.dtype).min
        )
        if attention_mask is not None:
            scores = scores + attention_mask[..., :total_len]
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(
            queries.dtype
        )
        output = torch.matmul(weights, values).transpose(1, 2).contiguous()
        output = self.o_proj(output.reshape(batch_size, query_len, -1))
        return output, weights


if DeepseekV2Attention is not None:

    class DeepseekV2MLAPagedAttention(
        _MLAPagedAttentionMixin, DeepseekV2Attention
    ):
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: Any = None,
            position_embeddings: Any = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            del past_key_values, kwargs
            return self._mla_forward(
                hidden_states, position_embeddings, attention_mask, version="v2"
            )

else:
    DeepseekV2MLAPagedAttention = None  # type: ignore[misc,assignment]


if DeepseekV3Attention is not None:

    class DeepseekV3MLAPagedAttention(
        _MLAPagedAttentionMixin, DeepseekV3Attention
    ):
        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Any,
            attention_mask: torch.Tensor | None,
            past_key_values: Any = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            del past_key_values, kwargs
            return self._mla_forward(
                hidden_states, position_embeddings, attention_mask, version="v3"
            )

else:
    DeepseekV3MLAPagedAttention = None  # type: ignore[misc,assignment]


def adapt_deepseek_attention(
    module: torch.nn.Module,
    cache: MLAPagedKVCache,
    *,
    enabled: bool = False,
) -> torch.nn.Module:
    """Adapt a real upstream attention object in place, preserving parameters."""
    if not enabled:
        return module
    target: type[torch.nn.Module] | None = None
    if DeepseekV2Attention is not None and isinstance(
        module, DeepseekV2Attention
    ):
        target = cast(type[torch.nn.Module], DeepseekV2MLAPagedAttention)
    elif DeepseekV3Attention is not None and isinstance(
        module, DeepseekV3Attention
    ):
        target = cast(type[torch.nn.Module], DeepseekV3MLAPagedAttention)
    if target is None:
        raise TypeError(
            "module is not an installed upstream DeepSeek V2/V3 attention"
        )
    if getattr(module, "layer_idx", None) is None:
        raise ValueError("DeepSeek MLA adapter requires layer_idx")
    module.__class__ = target
    setattr(module, "_mla_cache", cache)
    setattr(module, "_mla_metadata", None)
    return module


def is_deepseek_mla_eligible(config: object, *, enabled: bool = False) -> bool:
    """Fail-closed structural gate for the batch-one token-KV-only path."""
    if not enabled:
        return False
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type not in {"deepseek_v2", "deepseek_v3"}:
        return False
    if not isinstance(getattr(config, "kv_lora_rank", None), int):
        return False
    for name in (
        "sliding_window",
        "sliding_window_pattern",
        "recurrent_chunk_size",
        "hybrid_attention",
    ):
        value = getattr(config, name, None)
        if value not in (None, False, 0):
            return False
    return True


def adapt_deepseek_model(
    model: torch.nn.Module,
    cache: MLAPagedKVCache,
    *,
    enabled: bool = False,
) -> list[torch.nn.Module]:
    config = getattr(model, "config", None)
    if config is None or not is_deepseek_mla_eligible(config, enabled=enabled):
        return []
    adapted: list[torch.nn.Module] = []
    for module in model.modules():
        if DeepseekV2Attention is not None and isinstance(
            module, DeepseekV2Attention
        ):
            adapted.append(
                adapt_deepseek_attention(module, cache, enabled=True)
            )
        elif DeepseekV3Attention is not None and isinstance(
            module, DeepseekV3Attention
        ):
            adapted.append(
                adapt_deepseek_attention(module, cache, enabled=True)
            )
    return adapted


def set_deepseek_mla_context(
    module: torch.nn.Module, metadata: AttentionMetadata
) -> None:
    if not isinstance(module, _MLAPagedAttentionMixin):
        raise TypeError("module is not an MLA paged attention adapter")
    module._mla_metadata = metadata


def clear_deepseek_mla_context(module: torch.nn.Module) -> None:
    if isinstance(module, _MLAPagedAttentionMixin):
        module._mla_metadata = None


__all__ = [
    "DeepseekV2MLAPagedAttention",
    "DeepseekV3MLAPagedAttention",
    "adapt_deepseek_attention",
    "adapt_deepseek_model",
    "is_deepseek_mla_eligible",
    "set_deepseek_mla_context",
    "clear_deepseek_mla_context",
]
