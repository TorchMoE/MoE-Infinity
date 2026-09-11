# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingParameterType=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from transformers.models.deepseek_v2 import modeling_deepseek_v2 as m2
from transformers.models.deepseek_v2.configuration_deepseek_v2 import (
    DeepseekV2Config,
)
from transformers.models.deepseek_v3 import modeling_deepseek_v3 as m3
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)

from moe_infinity.models import (
    DeepseekV2PagedAttention,
    DeepseekV3PagedAttention,
)
from moe_infinity.models.deepseek_v2_paged_attention import (
    _paged_cache_head_dim,
)


@dataclass
class _BackendCall:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_metadata: object
    scale: float | None


class _RecordingBackend:
    def __init__(self, fill_value: float = 0.5) -> None:
        self.fill_value = fill_value
        self.calls: list[_BackendCall] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: object = None,
        attn_metadata: object = None,
        attention_metadata: object = None,
        scale: float | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        _ = (kv_cache, attn_metadata, layer_idx)
        self.calls.append(
            _BackendCall(query, key, value, attention_metadata, scale)
        )
        return torch.full(
            (query.shape[0], query.shape[1], value.shape[2]),
            fill_value=self.fill_value,
            dtype=query.dtype,
            device=query.device,
        )


def _v2_config() -> DeepseekV2Config:
    return DeepseekV2Config(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        first_k_dense_replace=0,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        attention_dropout=0.0,
    )


def _v3_config() -> DeepseekV3Config:
    return DeepseekV3Config(
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        attention_dropout=0.0,
    )


@pytest.fixture(autouse=True)
def _clear_paged_context():
    DeepseekV2PagedAttention.clear_paged_context()
    DeepseekV3PagedAttention.clear_paged_context()
    yield
    DeepseekV2PagedAttention.clear_paged_context()
    DeepseekV3PagedAttention.clear_paged_context()


def test_v2_paged_attention_uses_backend() -> None:
    config = _v2_config()
    attn = DeepseekV2PagedAttention(config, layer_idx=0).eval()
    rotary = m2.DeepseekV2RotaryEmbedding(config=config)

    hidden_states = torch.randn(1, 3, config.hidden_size)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(hidden_states, position_ids)

    backend = _RecordingBackend(fill_value=1.5)
    metadata = object()
    DeepseekV2PagedAttention.set_paged_context(backend, metadata)

    outputs, attn_weights = attn(
        hidden_states=hidden_states,
        attention_mask=None,
        position_embeddings=position_embeddings,
    )

    cache_dim = _paged_cache_head_dim(attn.qk_head_dim)
    assert outputs.shape == hidden_states.shape
    assert attn_weights is None
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call.attention_metadata is metadata
    assert call.query.shape == (3, attn.num_heads, cache_dim)
    assert call.key.shape == (3, attn.num_heads, cache_dim)
    assert call.value.shape == (3, attn.num_heads, cache_dim)
    assert call.scale == pytest.approx(attn.scaling)


def test_v2_paged_attention_fallback() -> None:
    config = _v2_config()
    attn = DeepseekV2PagedAttention(config, layer_idx=0).eval()
    rotary = m2.DeepseekV2RotaryEmbedding(config=config)

    hidden_states = torch.randn(1, 3, config.hidden_size)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(hidden_states, position_ids)

    backend = _RecordingBackend()

    outputs, _ = attn(
        hidden_states=hidden_states,
        attention_mask=None,
        position_embeddings=position_embeddings,
    )

    assert outputs.shape == hidden_states.shape
    assert backend.calls == []


def test_v3_paged_attention_uses_backend() -> None:
    config = _v3_config()
    attn = DeepseekV3PagedAttention(config, layer_idx=0).eval()
    rotary = m3.DeepseekV3RotaryEmbedding(config=config)

    hidden_states = torch.randn(1, 3, config.hidden_size)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(hidden_states, position_ids)

    backend = _RecordingBackend(fill_value=0.25)
    metadata = object()
    DeepseekV3PagedAttention.set_paged_context(backend, metadata)

    outputs, attn_weights = attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )

    cache_dim = _paged_cache_head_dim(attn.qk_head_dim)
    assert outputs.shape == hidden_states.shape
    assert attn_weights is None
    assert len(backend.calls) == 1
    call = backend.calls[0]
    assert call.attention_metadata is metadata
    assert call.query.shape == (3, attn.num_heads, cache_dim)
    assert call.key.shape == (3, attn.num_heads, cache_dim)
    assert call.value.shape == (3, attn.num_heads, cache_dim)
    assert call.scale == pytest.approx(attn.scaling)


def test_v3_paged_attention_fallback() -> None:
    config = _v3_config()
    attn = DeepseekV3PagedAttention(config, layer_idx=0).eval()
    rotary = m3.DeepseekV3RotaryEmbedding(config=config)

    hidden_states = torch.randn(1, 3, config.hidden_size)
    position_ids = torch.arange(3, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(hidden_states, position_ids)

    backend = _RecordingBackend()

    outputs, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )

    assert outputs.shape == hidden_states.shape
    assert backend.calls == []


def test_mla_kv_cache_spec_is_symmetric_padded() -> None:
    lite_like = SimpleNamespace(
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_size=2048,
        head_dim=None,
    )
    assert DeepseekV2PagedAttention.get_kv_cache_spec_for_config(lite_like) == {
        "num_kv_heads": 16,
        "head_dim": 256,
    }
    assert DeepseekV3PagedAttention.get_kv_cache_spec_for_config(lite_like) == {
        "num_kv_heads": 16,
        "head_dim": 256,
    }

    tiny = _v2_config()
    assert DeepseekV2PagedAttention.get_kv_cache_spec_for_config(tiny) == {
        "num_kv_heads": 2,
        "head_dim": 64,
    }
