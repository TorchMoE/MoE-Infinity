"""CPU registration contract only; no native CUDA execution or weights."""

import re
from pathlib import Path

import pytest


def _native_weight_names(expert_type):
    header = (
        Path(__file__).resolve().parents[3] / "core/parallel/expert_module.h"
    ).read_text()
    enum = re.search(rf"\b(\w+)\s*=\s*{expert_type}\s*[,}}]", header)
    assert enum, f"Missing native expert type {expert_type}"
    traits = re.search(
        rf"struct ExpertTraits<ExpertType::{enum.group(1)}>\s*\{{"
        r"(.*?)\n\};",
        header,
        re.S,
    )
    assert traits, "Native ExpertTraits declaration changed"
    weights = re.search(r"weight_names\s*=\s*\{(.*?)\}", traits.group(1), re.S)
    assert weights, "Native weight registration declaration changed"
    return re.findall(r'"([^\"]+)"', weights.group(1))


def test_jamba_registration_matches_native_expert_layout():
    modeling = pytest.importorskip("transformers.models.jamba.modeling_jamba")
    if not hasattr(modeling, "JambaMLP"):
        pytest.skip("transformers does not expose JambaMLP")
    from transformers import JambaConfig

    from moe_infinity.common.constants import parse_expert_type
    from moe_infinity.models.jamba import SyncJambaMoEBlock

    config = JambaConfig(
        hidden_size=16,
        intermediate_size=40,
        num_experts=2,
        num_experts_per_tok=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        architectures=["JambaForCausalLM"],
    )
    block = SyncJambaMoEBlock(config)
    expected = _native_weight_names(parse_expert_type(config))
    for expert in block.experts:
        assert isinstance(expert, modeling.JambaMLP)
        parameters = list(expert.named_parameters())
        names = [name.removesuffix(".weight") for name, _ in parameters]
        assert names == expected
        # Unequal dimensions expose an up/down slot swap; these are real
        # registered weights, not a mirrored architecture-to-type table.
        assert expert.up_proj.weight.shape == (40, 16)
        assert expert.down_proj.weight.shape == (16, 40)
