"""GLM-5.3-Flash (glm5_next) registry and config parsing.

GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, arch Glm5NextForConditionalGeneration)
nests its MoE fields under text_config and stores per-expert FP8 tensors as
`model.language_model.layers.<L>.mlp.experts.<E>.<proj>.weight` (verified
against the checkpoint's safetensors index). Registration is guarded on the
transformers class (>= 5.16); on older transformers the registry must
fail fast with the standard unsupported-architecture error.
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
    "glm_5_3_flash",
    "config.json",
)

_HAS_GLM5_NEXT = "glm5next" in MODEL_MAPPING_NAMES


@pytest.fixture()
def glm53_flash_config() -> PretrainedConfig:
    with open(FIXTURE) as f:
        return PretrainedConfig.from_dict(json.load(f))


@pytest.mark.skipif(
    not _HAS_GLM5_NEXT, reason="transformers lacks Glm5Next classes"
)
def test_registry_maps_glm5next_to_expert_type_5():
    assert MODEL_MAPPING_TYPES["glm5next"] == 5


@pytest.mark.skipif(
    not _HAS_GLM5_NEXT, reason="transformers lacks Glm5Next classes"
)
def test_parse_expert_type(glm53_flash_config):
    assert parse_expert_type(glm53_flash_config) == 5


def test_fail_fast_when_unregistered(glm53_flash_config):
    if _HAS_GLM5_NEXT:
        pytest.skip("registered in this environment")
    with pytest.raises(RuntimeError, match="glm5nextforconditionalgeneration"):
        parse_expert_type(glm53_flash_config)


def test_parse_moe_param_reads_nested_text_config(glm53_flash_config):
    num_layers, num_experts, num_encoder_layers = parse_moe_param(
        glm53_flash_config
    )
    assert num_layers == 45
    assert num_experts == 288
    assert num_encoder_layers == 0


@pytest.mark.parametrize(
    "name,expected_layer,expected_expert",
    [
        (
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
            3,
            0,
        ),
        (
            "model.language_model.layers.44.mlp.experts.287.down_proj.weight_scale_inv",
            44,
            287,
        ),
    ],
)
def test_parse_expert_id_routed(
    glm53_flash_config, name, expected_layer, expected_expert
):
    layer_id, expert_id = parse_expert_id(name, glm53_flash_config)
    assert layer_id == expected_layer
    assert expert_id == expected_expert


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.language_model.layers.3.mlp.gate.weight",
        "model.language_model.layers.0.self_attn.A_log",
        "model.language_model.embed_tokens.weight",
        "model.visual.blocks.0.attn.proj.weight",
        "lm_head.weight",
    ],
)
def test_parse_expert_id_non_expert(glm53_flash_config, name):
    _, expert_id = parse_expert_id(name, glm53_flash_config)
    assert expert_id is None
