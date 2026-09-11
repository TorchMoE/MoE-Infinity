"""Glm5Next Sync MoE block routing and offload wiring policy.

The block test requires transformers >= 5.16 (Glm5Next classes) and pins:
per-expert ModuleList layout (offload-indexable parameter names), the router
tuple protocol consumed by SyncGlmMoeDsaMoEBlock._route, and noaux_tc weight
normalization (per-token weights sum to routed_scaling_factor). The resident
policy test runs on any transformers version because parse_expert_id does not
need the modeling class.
"""

import json
import os

import pytest
import torch
from transformers import PretrainedConfig

from moe_infinity.runtime.model_offload import OffloadEngine

FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "fixtures",
    "glm_5_3_flash",
    "config.json",
)

_TINY_TEXT = {
    "model_type": "glm5_next_text",
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "num_hidden_layers": 4,
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
}


def _has_glm5_next() -> bool:
    try:
        import transformers.models.glm5_next  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_glm5_next(), reason="transformers lacks Glm5Next classes"
)
def test_block_layout_and_routing():
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextTextConfig,
    )

    from moe_infinity.models.glm5_next import SyncGlm5NextMoEBlock

    torch.manual_seed(0)
    block = SyncGlm5NextMoEBlock(Glm5NextTextConfig(**_TINY_TEXT))

    param_names = {name for name, _ in block.named_parameters()}
    assert "experts.0.gate_proj.weight" in param_names
    assert "experts.7.down_proj.weight" in param_names
    assert "shared_experts.up_proj.weight" in param_names
    assert "gate.weight" in param_names

    with torch.no_grad():
        block.gate.weight.normal_()
        topk_idx, topk_weights = block._route(torch.randn(4, 64))
    assert topk_idx.shape == (4, 2)
    assert topk_weights.shape == (4, 2)
    assert (topk_idx >= 0).all() and (topk_idx < 8).all()
    torch.testing.assert_close(
        topk_weights.sum(dim=-1).float(),
        torch.full((4,), 2.5),
        rtol=1e-4,
        atol=1e-4,
    )


def _resident(name: str) -> bool:
    with open(FIXTURE) as f:
        config = PretrainedConfig.from_dict(json.load(f))
    engine = object.__new__(OffloadEngine)
    engine.config = config
    return engine._is_shared_expert_param(name)


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
def test_glm5_next_non_experts_stay_resident(name):
    assert _resident(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.44.mlp.experts.287.down_proj.weight",
    ],
)
def test_glm5_next_routed_experts_are_offloaded(name):
    assert _resident(name) is False
