from types import SimpleNamespace

import pytest
import torch

from moe_infinity.kernel import paged_attention_ops
from moe_infinity.runtime.kv_cache_format import (
    KVCacheBackendCapabilities,
    KVCacheFormat,
    KVCacheModelInfo,
    model_info_from_config,
    resolve_kv_cache_format,
)
from moe_infinity.serving.memory_manager import MemoryBudget, MemoryManager


def test_int8_memory_budget_counts_payload_and_scale() -> None:
    manager = MemoryManager(device=torch.device("cpu"))
    manager._last_budget = MemoryBudget(
        total_gpu_memory_bytes=1_000_000,
        model_memory_bytes=0,
        expert_cache_ratio=0.0,
        kv_cache_ratio=1.0,
        activation_reserve_ratio=0.0,
    )
    native_blocks = manager.get_max_kv_blocks(
        block_size=16,
        num_layers=2,
        num_heads=8,
        head_dim=128,
        dtype=torch.float16,
        format_name="native",
    )
    int8_blocks = manager.get_max_kv_blocks(
        block_size=16,
        num_layers=2,
        num_heads=8,
        head_dim=128,
        dtype=torch.float16,
        format_name="int8_sym",
    )
    kv_bytes = manager._last_budget.kv_cache_bytes
    native_page = KVCacheFormat.parse("native").page_size_bytes(
        block_size=16,
        num_kv_heads=8,
        head_dim=128,
        execution_dtype=torch.float16,
    )
    int8_page = KVCacheFormat.parse("int8_sym").page_size_bytes(
        block_size=16, num_kv_heads=8, head_dim=128
    )
    assert native_blocks == kv_bytes // (2 * native_page)
    assert int8_blocks == kv_bytes // (2 * int8_page)
    assert int8_page == 33280
    assert int8_blocks > native_blocks


def test_native_kv_block_count_unchanged_by_new_argument() -> None:
    manager = MemoryManager(device=torch.device("cpu"))
    manager._last_budget = MemoryBudget(
        total_gpu_memory_bytes=1_000_000,
        model_memory_bytes=0,
        expert_cache_ratio=0.0,
        kv_cache_ratio=1.0,
    )
    with_default = manager.get_max_kv_blocks(
        block_size=16,
        num_layers=2,
        num_heads=8,
        head_dim=128,
        dtype=torch.float16,
    )
    with_native = manager.get_max_kv_blocks(
        block_size=16,
        num_layers=2,
        num_heads=8,
        head_dim=128,
        dtype=torch.float16,
        format_name="native",
    )
    assert with_default == with_native


def test_mla_metadata_is_detected_from_model_config() -> None:
    info = model_info_from_config(
        SimpleNamespace(
            num_attention_heads=16,
            num_key_value_heads=1,
            head_dim=128,
            kv_lora_rank=512,
        )
    )
    assert info.is_mla is True


def test_gqa_metadata_is_detected_from_model_config() -> None:
    info = model_info_from_config(
        SimpleNamespace(
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
    )
    assert info.is_mla is False
    assert info.num_attention_heads == 32
    assert info.num_kv_heads == 8


def test_imported_native_module_without_int8_binding_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        paged_attention_ops,
        "_paged_attn_ops",
        SimpleNamespace(paged_attention_v1=lambda: None),
    )
    available, reason = paged_attention_ops.probe_native_int8_binding()
    assert available is False
    assert reason == "native_int8_binding_missing"
    decision = resolve_kv_cache_format(
        requested="int8_sym",
        model=KVCacheModelInfo(32, 8, 128, False),
        device=torch.device("cuda"),
        backend_preference="native",
        capabilities=KVCacheBackendCapabilities(False, available, True, reason),
        allow_fallback=True,
    )
    assert decision.execution_backend == "sdpa_dequant"
    assert decision.reason == "native_int8_binding_missing"


def test_missing_int8_binding_and_sdpa_strict_mode_raises() -> None:
    with pytest.raises(RuntimeError, match="no_int8_execution_backend"):
        resolve_kv_cache_format(
            requested="int8_sym",
            model=KVCacheModelInfo(32, 8, 128, False),
            device=torch.device("cuda"),
            backend_preference="native",
            capabilities=KVCacheBackendCapabilities(
                False, False, False, "native_int8_binding_missing"
            ),
            allow_fallback=False,
        )


def _build_engine(kv_cache_format: str, is_mla: bool):
    from moe_infinity.serving.engine import ContinuousBatchingEngine

    if is_mla:
        model_config = SimpleNamespace(
            num_attention_heads=16,
            num_key_value_heads=1,
            head_dim=64,
            kv_lora_rank=512,
        )
        num_kv_heads, head_dim = 1, 64
    else:
        model_config = SimpleNamespace(
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=16,
            hidden_size=128,
        )
        num_kv_heads, head_dim = 8, 16
    model = SimpleNamespace(config=model_config)
    config = {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "block_size": 16,
        "num_layers": 2,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": "float16",
        "max_batch_size": 4,
        "max_tokens_per_step": 64,
        "num_kv_blocks": 8,
        "kv_cache_format": kv_cache_format,
        "kv_cache_allow_fallback": True,
    }
    return ContinuousBatchingEngine(model, object(), config)


def test_engine_stats_report_effective_format_for_int8_request() -> None:
    if not torch.cuda.is_available():
        pytest.skip("engine store allocation targets the resolved device")
    engine = _build_engine("int8_sym", is_mla=False)
    stats = engine.get_stats()
    assert stats["requested_kv_cache_format"] == "int8_sym"
    assert stats["effective_kv_cache_format"] == "int8_sym"
    assert engine.kv_cache.store.format.name == "int8_sym"
    config = engine.get_config()
    assert config["effective_kv_cache_format"] == "int8_sym"


def test_engine_stats_report_native_fallback_for_mla_request() -> None:
    if not torch.cuda.is_available():
        pytest.skip("engine store allocation targets the resolved device")
    engine = _build_engine("int8_sym", is_mla=True)
    stats = engine.get_stats()
    assert stats["requested_kv_cache_format"] == "int8_sym"
    assert stats["effective_kv_cache_format"] == "native"
    assert stats["kv_cache_format_decision_reason"] == "mla_not_validated"
    assert engine.kv_cache.store.format.name == "native"


def test_native_request_engine_stats_are_native() -> None:
    engine = _build_engine("native", is_mla=False)
    stats = engine.get_stats()
    assert stats["effective_kv_cache_format"] == "native"
    assert engine.kv_cache.store.format.name == "native"


def test_cpu_falls_back_to_sdpa_dequant() -> None:
    decision = resolve_kv_cache_format(
        requested="int8_sym",
        model=KVCacheModelInfo(32, 8, 128, False),
        device=torch.device("cpu"),
        backend_preference="auto",
        capabilities=KVCacheBackendCapabilities(
            False, False, True, "native_int8_module_unavailable"
        ),
        allow_fallback=True,
    )
    assert decision.execution_backend == "sdpa_dequant"
    assert decision.reason == "cpu_sdpa_dequant"
