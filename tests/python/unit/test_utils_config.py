# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportAny=false, reportUnusedCallResult=false

import os
from pathlib import Path

import pytest

from moe_infinity.utils.config import ArcherConfig


def test_load_from_json_sets_paths_and_threads(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    config = ArcherConfig.load_from_json(
        {
            "offload_path": "/tmp/offload",
            "trace_capacity": 123,
            "prefetch": True,
        }
    )

    assert config.offload_path == "/tmp/offload"
    assert config.trace_capacity == 123
    assert config.prefetch is True
    assert config.device_per_node == 2
    assert config.perfect_cache_file == os.path.join(
        "/tmp/offload", "perfect_cache"
    )


def test_load_from_file_sets_trace_path(tmp_path: Path):
    config_path = tmp_path / "config.json"
    trace_file = tmp_path / "trace.json"
    config_path.write_text(
        '{"offload_path": "/tmp/offload", "trace_path": "%s"}'
        % trace_file.as_posix()
    )

    config = ArcherConfig.load_from_file(config_path)

    assert config.trace_path == os.path.abspath(trace_file)


def test_trace_path_directory_raises(tmp_path: Path):
    trace_dir = tmp_path / "trace_dir"
    trace_dir.mkdir()

    with pytest.raises(ValueError):
        ArcherConfig(offload_path=str(tmp_path), trace_path=trace_dir)


def test_kv_cache_fields_default(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.warns(UserWarning, match="auto-set to 0.15"):
        config = ArcherConfig(offload_path="/tmp")
    assert config.kv_cache_memory_ratio == 0.15
    assert config.use_native_engine is True
    assert config.enable_attention_offload is False
    assert config.enable_kv_cache_offload is False
    assert config.attention_backend == "default"


def test_kv_cache_format_defaults_native(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(offload_path="/tmp", use_native_engine=False)
    assert config.kv_cache_format == "native"
    assert config.kv_cache_allow_fallback is True


def test_kv_cache_format_int8_accepted(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.warns(UserWarning):
        config = ArcherConfig(offload_path="/tmp", kv_cache_format="int8_sym")
    assert config.kv_cache_format == "int8_sym"


def test_kv_cache_format_invalid_rejected(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match="unsupported KV cache format"):
        ArcherConfig(offload_path="/tmp", kv_cache_format="int4")


def test_kv_cache_memory_ratio_validation(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError):
        ArcherConfig(
            offload_path="/tmp",
            device_memory_ratio=0.7,
            kv_cache_memory_ratio=0.5,
        )


def test_backwards_compat_old_config(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(
        offload_path="/tmp",
        device_memory_ratio=0.75,
        use_native_engine=False,
    )
    assert config.kv_cache_memory_ratio == 0.0
    assert config.enable_kv_cache_offload is False


def test_native_engine_autocorrects_kv_cache_ratio(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.warns(UserWarning, match="auto-set to 0.15"):
        config = ArcherConfig(
            offload_path="/tmp",
            use_native_engine=True,
            kv_cache_memory_ratio=0.0,
        )
    assert config.kv_cache_memory_ratio == pytest.approx(0.15)


def test_phase_policy_defaults_are_backward_compatible(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(offload_path="/tmp", use_native_engine=False)
    assert config.phase_specific_expert_policy is False
    assert config.prefill_expert_admission == "transient_on_pressure"
    assert config.decode_expert_admission == "cache"
    assert config.prefill_expert_prefetch_top_k == 0
    assert config.decode_expert_prefetch_top_k == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prefill_expert_admission", "drop", "must be one of"),
        ("decode_expert_prefetch_top_k", -1, "must be >= 0"),
        ("decode_expert_prefetch_priority", 0, "must be in \\[1, 19\\]"),
        ("prefill_expert_eviction_weight", 0.0, "must be finite and > 0"),
        ("expert_policy_starvation_limit", 0, "must be > 0"),
    ],
)
def test_phase_policy_rejects_invalid_values(
    monkeypatch, field, value, message
):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    kwargs = {"offload_path": "/tmp", "use_native_engine": False, field: value}
    with pytest.raises(ValueError, match=message):
        ArcherConfig(**kwargs)


def test_overlap_prefetch_defaults_are_safe(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    c = ArcherConfig(offload_path="/tmp", use_native_engine=False)
    assert c.overlap_prefetch_policy == "off"
    assert c.overlap_prefetch_ewma_alpha == pytest.approx(0.2)
    assert c.overlap_prefetch_safety_factor == pytest.approx(0.8)
    assert c.overlap_prefetch_cold_start_experts == 1
    assert c.overlap_prefetch_max_window_bytes == 256 * 1024 * 1024
    assert c.overlap_prefetch_max_inflight_bytes == 512 * 1024 * 1024
    assert c.gpu_only_expert_routing is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("overlap_prefetch_policy", "fast"),
        ("overlap_prefetch_ewma_alpha", 0.0),
        ("overlap_prefetch_safety_factor", 1.1),
        ("overlap_prefetch_cold_start_experts", -1),
        ("overlap_prefetch_max_window_bytes", -1),
    ],
)
def test_overlap_prefetch_rejects_invalid_values(monkeypatch, field, value):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match=field):
        ArcherConfig(
            offload_path="/tmp",
            use_native_engine=False,
            **{field: value},
        )


@pytest.mark.parametrize("policy", ["observe", "enforce"])
def test_overlap_prefetch_rejects_gpu_only_routing(monkeypatch, policy):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match="gpu_only_expert_routing"):
        ArcherConfig.load_from_json(
            {
                "offload_path": "/tmp",
                "use_native_engine": False,
                "gpu_only_expert_routing": True,
                "overlap_prefetch_policy": policy,
            }
        )


def test_gpu_only_expert_routing_defaults_off(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(offload_path="/tmp", use_native_engine=False)
    assert config.gpu_only_expert_routing is False


def test_gpu_only_expert_routing_loads_from_json(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig.load_from_json(
        {
            "offload_path": "/tmp",
            "use_native_engine": False,
            "gpu_only_expert_routing": True,
        }
    )
    assert config.gpu_only_expert_routing is True


@pytest.mark.parametrize("policy", ["observe", "enforce"])
def test_gpu_routing_rejects_current_overlap_boolean(monkeypatch, policy):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(
        ValueError,
        match="gpu_only_expert_routing cannot be combined with overlap prefetch",
    ):
        ArcherConfig(
            offload_path="/tmp",
            use_native_engine=False,
            gpu_only_expert_routing=True,
            overlap_prefetch_policy=policy,
        )


def test_gpu_only_routing_is_independent_when_overlap_is_off(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    c = ArcherConfig(
        offload_path="/tmp",
        use_native_engine=False,
        gpu_only_expert_routing=True,
        overlap_prefetch_policy="off",
    )
    assert c.gpu_only_expert_routing is True


@pytest.mark.parametrize("mode", ["observe", "enforce"])
def test_gpu_routing_rejects_future_overlap_modes(mode):
    with pytest.raises(
        ValueError,
        match="gpu_only_expert_routing cannot be combined with overlap prefetch",
    ):
        ArcherConfig._validate_gpu_routing_overlap(True, False, mode)


def test_paged_mla_admission_guard_defaults_are_safe(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(use_native_engine=False)

    assert config.enable_deepseek_mla_paging is False
    assert config.max_resident_paged_speculative_sessions == 1
    assert config.min_free_mla_blocks_after_admission == 1


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_resident_paged_speculative_sessions", True),
        ("max_resident_paged_speculative_sessions", -1),
        ("min_free_mla_blocks_after_admission", False),
        ("min_free_mla_blocks_after_admission", 0),
    ],
)
def test_paged_mla_admission_guard_rejects_invalid_values(
    monkeypatch, field_name, value
):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)

    with pytest.raises(ValueError, match=field_name):
        ArcherConfig(use_native_engine=False, **{field_name: value})


def test_adaptive_fields_default_disabled(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(
        offload_path="/tmp",
        use_native_engine=False,
    )
    assert config.adaptive_expert_precision is False
    assert config.adaptive_hbm_budget_bytes == 0
    assert config.adaptive_variant_build is False
    assert config.adaptive_derivative_root is None


def test_adaptive_budget_must_be_positive_when_enabled(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(
        ValueError, match="adaptive_hbm_budget_bytes must be positive"
    ):
        ArcherConfig(
            offload_path="/tmp",
            use_native_engine=False,
            adaptive_expert_precision=True,
            adaptive_hbm_budget_bytes=0,
        )


def test_adaptive_threshold_ordering_validated(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError):
        ArcherConfig(
            offload_path="/tmp",
            use_native_engine=False,
            adaptive_expert_precision=True,
            adaptive_hbm_budget_bytes=1024,
            adaptive_promotion_threshold=0.2,
            adaptive_demotion_threshold=0.5,
        )


def test_adaptive_derivative_root_resolves_from_offload(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(
        offload_path="/tmp/offload",
        use_native_engine=False,
        adaptive_expert_precision=True,
        adaptive_hbm_budget_bytes=2048,
    )
    assert config.adaptive_derivative_root == os.path.join(
        "/tmp/offload", "adaptive_derivatives"
    )


def test_adaptive_defaults_disabled_and_bounded(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    config = ArcherConfig(offload_path="/tmp", use_native_engine=False)
    assert config.adaptive_memory_enabled is False
    assert config.adaptive_memory_min_expert_cache_bytes > 0
    assert config.adaptive_memory_min_kv_cache_blocks > 0
    assert config.adaptive_memory_free_reserve_bytes > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adaptive_memory_interval_steps", 0),
        ("adaptive_memory_ewma_alpha", 0.0),
        ("adaptive_memory_hysteresis_ratio", 1.1),
        ("adaptive_memory_failure_limit", 0),
    ],
)
def test_adaptive_bounds_are_validated(monkeypatch, field, value) -> None:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match="adaptive_memory"):
        ArcherConfig(
            offload_path="/tmp",
            use_native_engine=False,
            **{field: value},
        )
