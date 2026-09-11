"""GLM-5.3 must resolve through the existing GlmMoeDsa registry path.

GLM-5.3 (zai-org/GLM-5.3) reuses the GLM-5.2 base architecture
(GlmMoeDsaForCausalLM, model_type=glm_moe_dsa); every gain over GLM-5.2 is
post-training. This test pins that the registry, expert-type parser, and
MoE-shape parser all treat the real GLM-5.3 config exactly like GLM-5.2, so
the checkpoint keeps working through the existing FP8 expert-offload path.
"""

import json
import os

import pytest

transformers = pytest.importorskip("transformers")
from transformers import PretrainedConfig  # noqa: E402

from moe_infinity.common.constants import (  # noqa: E402
    MODEL_MAPPING_NAMES,
    parse_expert_type,
)
from moe_infinity.utils.hf_config import parse_moe_param  # noqa: E402

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "fixtures", "glm_5_3", "config.json"
)


@pytest.fixture()
def glm53_config() -> PretrainedConfig:
    with open(FIXTURE) as f:
        payload = json.load(f)
    return PretrainedConfig.from_dict(payload)


def test_glm53_arch_resolves_to_glmmoedsa(glm53_config):
    if "glmmoedsa" not in MODEL_MAPPING_NAMES:
        pytest.skip("GlmMoeDsaForCausalLM not available in this transformers")
    assert parse_expert_type(glm53_config) == 5


def test_glm53_moe_shape(glm53_config):
    num_layers, num_experts, num_encoder_layers = parse_moe_param(glm53_config)
    assert num_layers == 78
    assert num_experts == 256
    assert num_encoder_layers == 0
