# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportUnannotatedClassAttribute=false, reportPrivateUsage=false, reportImplicitOverride=false

from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from moe_infinity.engine.generation_loop import GenerationEngine
from moe_infinity.memory.kv_cache_manager import KVCacheManager
from moe_infinity.runtime.attention_types import AttentionMetadata, KVCacheSpec


def _build_moe_with_model(model: torch.nn.Module) -> Any:
    from moe_infinity.entrypoints.big_modeling import MoE

    moe = MoE.__new__(MoE)
    moe.model = model
    moe._native_attention_backend = object()
    return moe


def test_native_model_forward_uses_attention_metadata() -> None:
    events: list[tuple[object, ...]] = []

    class DeepseekV3PagedAttention(torch.nn.Module):
        @classmethod
        def set_paged_context(cls, backend: object, metadata: object) -> None:
            events.append(("set", backend, metadata))

        @classmethod
        def clear_paged_context(cls) -> None:
            events.append(("clear",))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    class MockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = DeepseekV3PagedAttention()

        def forward(self, input_ids: torch.Tensor) -> object:
            events.append(("model",))
            batch, seq_len = input_ids.shape
            logits = torch.zeros(batch, seq_len, 32, dtype=torch.float32)
            return types.SimpleNamespace(logits=logits)

    moe = _build_moe_with_model(MockModel())
    metadata = object()
    logits = moe._native_model_forward([1, 2, 3], metadata)

    assert logits.shape == (3, 32)
    assert events[0][0] == "set"
    assert events[0][1] is moe._native_attention_backend
    assert events[0][2] is metadata
    assert events[1] == ("model",)
    assert events[2] == ("clear",)


def test_native_model_forward_ignores_metadata_for_non_paged_models() -> None:
    class MockModel(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> object:
            batch, seq_len = input_ids.shape
            logits = torch.zeros(batch, seq_len, 8, dtype=torch.float32)
            return types.SimpleNamespace(logits=logits)

    moe = _build_moe_with_model(MockModel())
    logits = moe._native_model_forward([9, 10], object())
    assert logits.shape == (2, 8)


def test_native_model_forward_clears_context_on_exception() -> None:
    events: list[tuple[str]] = []

    class DeepseekV2PagedAttention(torch.nn.Module):
        @classmethod
        def set_paged_context(cls, backend: object, metadata: object) -> None:
            _ = (backend, metadata)
            events.append(("set",))

        @classmethod
        def clear_paged_context(cls) -> None:
            events.append(("clear",))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    class FailingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = DeepseekV2PagedAttention()

        def forward(self, input_ids: torch.Tensor) -> object:
            _ = input_ids
            raise RuntimeError("boom")

    moe = _build_moe_with_model(FailingModel())

    with pytest.raises(RuntimeError, match="boom"):
        _ = moe._native_model_forward([1], object())

    assert events == [("set",), ("clear",)]


def test_generation_loop_passes_metadata_to_forward() -> None:
    observed: list[AttentionMetadata] = []

    def mock_forward(
        token_ids: list[int],
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        observed.append(metadata)
        logits = torch.full((len(token_ids), 64), -1e9, dtype=torch.float32)
        logits[:, 5] = 0.0
        return logits

    spec = KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        block_size=4,
    )
    mgr = KVCacheManager(num_gpu_blocks=16, num_cpu_blocks=32, block_size=4)
    engine = GenerationEngine(
        kv_cache_manager=mgr,
        kv_spec=spec,
        num_layers=2,
        vocab_size=64,
        model_forward_fn=mock_forward,
        eos_token_id=2,
    )

    _ = engine.generate(prompt_token_ids=[1, 3], request_id="rid")

    assert len(observed) >= 2
    assert observed[0].is_prefill is True
    assert observed[0].num_prefill_tokens == 2
    assert observed[0].num_decode_tokens == 0
    assert all(meta.num_decode_tokens == 1 for meta in observed[1:])
    assert all(meta.is_prefill is False for meta in observed[1:])


def _native_model_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        hidden_size=16,
        vocab_size=32,
        eos_token_id=2,
        max_position_embeddings=64,
        torch_dtype="float32",
    )


def _native_engine_config(*, enable_kv_cache_offload: bool) -> Any:
    from moe_infinity.utils.config import ArcherConfig

    config = ArcherConfig(
        use_native_engine=True,
        enable_kv_cache_offload=enable_kv_cache_offload,
        device_memory_ratio=0.7,
        kv_cache_memory_ratio=0.15,
    )
    return config


def _native_moe() -> Any:
    from moe_infinity.entrypoints.big_modeling import MoE

    moe = MoE.__new__(MoE)
    moe.use_native_engine = True
    moe.max_seq_length = 64
    moe.model = types.SimpleNamespace(config=_native_model_config())
    return moe


def test_native_kv_offload_stores_paged_backend_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    moe = _native_moe()
    engine_config = _native_engine_config(enable_kv_cache_offload=True)

    components = moe._build_native_components(moe.model.config, engine_config)

    coordinator = components["kv_offload_coordinator"]
    attention_backend = components["attention_backend"]
    assert coordinator is not None
    assert attention_backend is not None
    stored = getattr(coordinator, "_kv_tensors")
    assert isinstance(stored, tuple)
    assert stored[0] is attention_backend.k_cache
    assert stored[1] is attention_backend.v_cache


def test_native_kv_offload_without_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moe_infinity.runtime.attention_backend as ab_module

    def _failing_backend(*_args: Any, **_kwargs: Any) -> object:
        raise RuntimeError("no paged backend")

    monkeypatch.setattr(ab_module, "PagedAttentionBackend", _failing_backend)

    moe = _native_moe()
    engine_config = _native_engine_config(enable_kv_cache_offload=True)

    with pytest.raises(
        RuntimeError, match="KV offload requires initialized paged KV tensors"
    ):
        moe._build_native_components(moe.model.config, engine_config)
