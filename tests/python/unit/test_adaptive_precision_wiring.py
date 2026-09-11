from types import SimpleNamespace

import pytest

from moe_infinity.runtime import expert_variant_manifest, model_offload
from moe_infinity.runtime.adaptive_precision_allowlist import (
    ReleasedAdaptiveEntry,
)
from moe_infinity.runtime.expert_precision import ExpertFormat
from moe_infinity.runtime.model_offload import _resolve_adaptive_precision


def _archer(**overrides):
    values = {
        "adaptive_expert_precision": False,
        "adaptive_variant_build": False,
        "adaptive_derivative_root": None,
        "adaptive_hbm_budget_bytes": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_disabled_mode_returns_canonical_without_manifest_access(tmp_path):
    result = _resolve_adaptive_precision(
        SimpleNamespace(model_type="qwen3_moe", quantization_config=None),
        _archer(),
        str(tmp_path / "missing"),
        extension_names=set(),
    )
    assert result.enabled is False
    assert result.fallback_reason == "disabled"


def test_protected_model_never_builds_adaptive_variants(tmp_path):
    result = _resolve_adaptive_precision(
        SimpleNamespace(
            model_type="gpt_oss", quantization_config={"quant_method": "mxfp4"}
        ),
        _archer(
            adaptive_expert_precision=True,
            adaptive_variant_build=True,
            adaptive_hbm_budget_bytes=1024,
        ),
        str(tmp_path),
        extension_names={"_v4_fp4"},
    )
    assert result.enabled is False
    assert result.fallback_reason == "protected:gpt_oss_mxfp4"
    assert not (tmp_path / "adaptive_derivatives" / "CURRENT").exists()


def test_legacy_store_without_manifest_is_read_only_fallback(tmp_path):
    (tmp_path / "name_id_map.json").write_text("{}")
    result = _resolve_adaptive_precision(
        SimpleNamespace(model_type="qwen3_moe", quantization_config=None),
        _archer(
            adaptive_expert_precision=True,
            adaptive_variant_build=False,
            adaptive_hbm_budget_bytes=1024,
        ),
        str(tmp_path),
        extension_names=set(),
    )
    assert result.enabled is False
    assert result.fallback_reason == "manifest_missing"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["name_id_map.json"]


def test_serving_rejects_valid_but_unreleased_manifest(tmp_path, monkeypatch):
    entry = ReleasedAdaptiveEntry(
        "a" * 64, ExpertFormat.FP8_E4M3_BLOCK128, "adaptive-expert-v1", "b" * 64
    )
    manifest = SimpleNamespace(release_entries=frozenset({entry}))
    monkeypatch.setattr(
        expert_variant_manifest.ExpertVariantManifest,
        "load_current",
        lambda *args, **kwargs: manifest,
    )
    result = _resolve_adaptive_precision(
        SimpleNamespace(model_type="qwen3_moe", quantization_config=None),
        _archer(
            adaptive_expert_precision=True,
            adaptive_variant_build=False,
            adaptive_hbm_budget_bytes=1024,
        ),
        str(tmp_path),
        extension_names=set(),
        purpose="serve",
        checkpoint_fingerprint="a" * 64,
        released_entries=frozenset(),
    )
    assert result.enabled is False
    assert result.fallback_reason == "manifest_unapproved"


@pytest.mark.parametrize(
    ("model_type", "quantization", "reason"),
    [
        ("gpt_oss", {"quant_method": "mxfp4"}, "protected:gpt_oss_mxfp4"),
        ("glm_moe_dsa", {"quant_method": "fp8"}, "protected:glm_fp8"),
        ("deepseek_v4", None, "protected:deepseek_v4_fp4"),
        ("deepseek_v3", {"quant_method": "fp8"}, "protected:existing_fp8"),
        ("mixtral", {"quant_method": "gptq"}, "protected:gptq"),
        ("mixtral", {"quant_method": "awq"}, "protected:awq"),
    ],
)
def test_protected_paths_short_circuit_general_resolution(
    tmp_path, monkeypatch, model_type, quantization, reason
):
    monkeypatch.setattr(
        model_offload,
        "resolve_model_precision_capabilities",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("protected path entered general adaptive resolution")
        ),
    )
    result = _resolve_adaptive_precision(
        SimpleNamespace(
            model_type=model_type, quantization_config=quantization
        ),
        _archer(
            adaptive_expert_precision=True,
            adaptive_variant_build=True,
            adaptive_hbm_budget_bytes=1024,
        ),
        str(tmp_path),
        extension_names={"_v4_fp4"},
    )
    assert result.enabled is False
    assert result.fallback_reason == reason
    assert not (tmp_path / "adaptive_derivatives").exists()
