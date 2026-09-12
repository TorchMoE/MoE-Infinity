"""DeepSeek-V4.1-Flash (deepseek_v41) registry and config parsing.

DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, arch
`DeepseekV41ForCausalLM`, `model_type="deepseek_v41"`) is a vision-language
checkpoint whose MoE fields (`num_hidden_layers`, `n_routed_experts`, ...)
nest under `text_config`, like the Qwen3.5 / GLM-5.3-Flash families. It ships
384 routed FP4 (E2M1 + ue8m0 scale) experts per MoE layer, top-6 routing with
`scoring_func="sqrtsoftplus"` / `topk_method="noaux_tc"`.

Registration is guarded on the transformers class: mainline transformers has
NOT merged `deepseek_v41` as of 2026-09-12, so `DeepseekV41ForCausalLM` is
usually absent and the registry entry is skipped (mirrors the V4 / Qwen3.5 /
GLM guards).

SUBSTRING-DISPATCH TRAP: the registry key ``"deepseekv41"`` contains
``"deepseekv4"`` as a substring. `parse_expert_type` matches the longest
registered key first (constants.py length-sort), so a V4.1 config resolves to
`deepseekv41` before `deepseekv4` when both are registered. The config-parsing
branches in ``hf_config.py`` (``parse_moe_param`` / ``parse_expert_id``) are
string-based and independent of the registry guard, and they place the
``deepseekv41`` branch *before* both ``deepseekv4`` and the generic
``deepseek`` branch so V4.1's nested-``text_config`` layout is used regardless
of whether the HF class is importable.
"""

import json
import os

import pytest
from transformers import PretrainedConfig

from moe_infinity.common.constants import (
    MODEL_MAPPING_NAMES,
    MODEL_MAPPING_TYPES,
    parse_expert_type,
)
from moe_infinity.utils.hf_config import parse_expert_id, parse_moe_param

FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "fixtures",
    "deepseek_v41_flash",
    "config.json",
)

_HAS_V41 = "deepseekv41" in MODEL_MAPPING_NAMES
_HAS_V4 = "deepseekv4" in MODEL_MAPPING_NAMES


@pytest.fixture()
def v41_flash_config() -> PretrainedConfig:
    with open(FIXTURE) as f:
        return PretrainedConfig.from_dict(json.load(f))


@pytest.mark.skipif(
    not _HAS_V41, reason="transformers lacks DeepseekV41ForCausalLM"
)
def test_registry_maps_deepseekv41_to_expert_type_5():
    assert MODEL_MAPPING_TYPES["deepseekv41"] == 5


@pytest.mark.skipif(
    not _HAS_V41, reason="transformers lacks DeepseekV41ForCausalLM"
)
def test_registered_class_is_not_none():
    cls = MODEL_MAPPING_NAMES["deepseekv41"]
    assert cls is not None
    assert hasattr(cls, "__name__")


@pytest.mark.skipif(
    not _HAS_V41, reason="transformers lacks DeepseekV41ForCausalLM"
)
def test_parse_expert_type_v41(v41_flash_config):
    assert parse_expert_type(v41_flash_config) == 5


def test_registry_substring_dispatch_v41_precedes_v4():
    """The length-sorted match in ``parse_expert_type`` must pick the more
    specific ``deepseekv41`` key first, so a V4.1 arch never resolves to the
    ``deepseekv4`` entry when both are registered."""
    arch = "deepseekv41forcausallm"
    candidates = [
        k
        for k in sorted(MODEL_MAPPING_NAMES, key=len, reverse=True)
        if k in arch
    ]
    if not _HAS_V41:
        pytest.skip("transformers lacks DeepseekV41ForCausalLM")
    assert candidates[0] == "deepseekv41"
    if _HAS_V4:
        assert candidates.index("deepseekv41") < candidates.index("deepseekv4")


def test_length_sort_dispatch_selects_v41_over_v4(monkeypatch):
    """Environment-independent proof of the length-sort precedence: register
    both keys with distinct sentinel types and confirm a V4.1 arch resolves to
    the more specific ``deepseekv41`` entry, never the ``deepseekv4`` substring
    match."""
    import moe_infinity.common.constants as constants

    monkeypatch.setitem(constants.MODEL_MAPPING_NAMES, "deepseekv4", object)
    monkeypatch.setitem(constants.MODEL_MAPPING_NAMES, "deepseekv41", object)
    monkeypatch.setitem(constants.MODEL_MAPPING_TYPES, "deepseekv4", 5)
    monkeypatch.setitem(constants.MODEL_MAPPING_TYPES, "deepseekv41", 99)

    cfg = PretrainedConfig.from_dict(
        {"architectures": ["DeepseekV41ForCausalLM"]}
    )
    assert constants.parse_expert_type(cfg) == 99


@pytest.mark.skipif(
    _HAS_V41,
    reason="documents the fallback only when V4.1 is NOT registered",
)
def test_parse_expert_type_fallback_when_v41_unregistered(v41_flash_config):
    """Substring-dispatch caveat: when the transformers build lacks
    ``DeepseekV41ForCausalLM`` but ships ``DeepseekV4ForCausalLM``, the
    ``deepseekv4`` key is still a substring of ``deepseekv41forcausallm`` and
    is the most-specific *registered* key, so ``parse_expert_type`` resolves
    to it (also expert-type 5) rather than raising. Config parsing still routes
    V4.1 correctly via its own arch-string branches (tested below)."""
    if _HAS_V4:
        assert parse_expert_type(v41_flash_config) == 5
    else:
        with pytest.raises(RuntimeError, match="deepseekv41forcausallm"):
            parse_expert_type(v41_flash_config)


def test_parse_moe_param_reads_nested_text_config(v41_flash_config):
    num_layers, num_experts, num_encoder_layers = parse_moe_param(
        v41_flash_config
    )
    assert num_layers == 40
    assert num_experts == 384
    # Phase 1 does not model the CED encoder/decoder split; all 40 layers are
    # treated as decoder layers.
    assert num_encoder_layers == 0


def test_parse_moe_param_prefers_v41_branch_over_flat_deepseek(
    v41_flash_config,
):
    """Precedence guard at the config-parsing level: ``deepseekv41`` contains
    both ``deepseekv4`` and ``deepseek``. The fixture defines
    ``num_hidden_layers`` / ``n_routed_experts`` ONLY under ``text_config``, so
    if the elif order regressed and a V4.1 config fell through to the flat
    ``deepseek`` branch it would read the (absent) top-level fields and raise
    ``AttributeError`` instead of returning the nested counts."""
    assert not hasattr(v41_flash_config, "n_routed_experts")
    num_layers, num_experts, _ = parse_moe_param(v41_flash_config)
    assert (num_layers, num_experts) == (40, 384)


@pytest.mark.parametrize(
    "name,expected_layer,expected_expert",
    [
        # V4-native "ffn.experts" layout (see hf_config comment: exact V4.1
        # checkpoint key layout must be confirmed against real shards).
        ("model.layers.0.ffn.experts.0.w1.weight", 0, 0),
        ("model.layers.5.ffn.experts.3.w2.weight", 5, 3),
        ("model.layers.39.ffn.experts.383.w3.weight", 39, 383),
        # Flexible/unanchored: also matches a VL-style language_model prefix.
        (
            "model.language_model.layers.14.ffn.experts.42.w1.weight_scale_inv",
            14,
            42,
        ),
    ],
)
def test_parse_expert_id_routed(
    v41_flash_config, name, expected_layer, expected_expert
):
    layer_id, expert_id = parse_expert_id(name, v41_flash_config)
    assert layer_id == expected_layer
    assert expert_id == expected_expert


@pytest.mark.parametrize(
    "name",
    [
        # Shared expert (no numeric expert index) is resident, not routed.
        "model.layers.3.ffn.shared_experts.gate_proj.weight",
        "model.layers.3.ffn.gate.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "model.visual.blocks.0.attn.proj.weight",
        "lm_head.weight",
        # Generic ".mlp.experts." layout must NOT match the V4.1 ".ffn.experts."
        # branch — this discriminates the v41 branch from the generic deepseek
        # branch (which uses the mlp pattern).
        "model.layers.3.mlp.experts.0.gate_proj.weight",
    ],
)
def test_parse_expert_id_non_expert(v41_flash_config, name):
    layer_id, expert_id = parse_expert_id(name, v41_flash_config)
    assert layer_id is None
    assert expert_id is None
