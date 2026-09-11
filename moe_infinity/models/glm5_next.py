"""Sync MoE block for GLM-5.3-Flash (glm5_next).

Router semantics match GlmMoeDsa exactly (sigmoid scoring, grouped noaux_tc
top-k with e_score_correction_bias, routed_scaling_factor renorm), and the
5.16 Glm5NextTextTopkRouter returns the same (logits, topk_weights,
topk_indices) tuple that SyncGlmMoeDsaMoEBlock._route already consumes, so
this subclass reuses the parent's routing/forward and only rebuilds __init__
around the glm5_next classes: MoE fields live on the nested text config, and
the HF module's batched Glm5NextTextExperts is replaced with a per-expert
ModuleList whose parameter names (experts.<i>.gate_proj/up_proj/down_proj)
match the per-expert checkpoint layout the offload engine indexes.
"""

from __future__ import annotations

import torch.nn as nn

try:
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextMLP,
        Glm5NextTextTopkRouter,
    )

    _GLM5_NEXT_AVAILABLE = True
except ImportError:
    _GLM5_NEXT_AVAILABLE = False
    Glm5NextTextMLP = Glm5NextTextTopkRouter = None

from moe_infinity.models.glm_moe_dsa import SyncGlmMoeDsaMoEBlock
from moe_infinity.utils.hf_config import moe_text_config


class SyncGlm5NextMoEBlock(SyncGlmMoeDsaMoEBlock):
    def __init__(self, config):
        if not _GLM5_NEXT_AVAILABLE:
            raise ImportError(
                "transformers >= 5.16 is required for glm5_next support"
            )
        nn.Module.__init__(self)
        text = moe_text_config(config)
        self.config = text
        self.num_experts = text.n_routed_experts
        self.n_routed_experts = text.n_routed_experts
        self.top_k = text.num_experts_per_tok
        self.n_group = text.n_group
        self.topk_group = text.topk_group
        self.norm_topk_prob = text.norm_topk_prob
        self.routed_scaling_factor = text.routed_scaling_factor

        self.gate = Glm5NextTextTopkRouter(text)
        self.experts = nn.ModuleList(
            [
                Glm5NextTextMLP(
                    config=text,
                    intermediate_size=text.moe_intermediate_size,
                )
                for _ in range(text.n_routed_experts)
            ]
        )
        self.shared_experts = Glm5NextTextMLP(
            config=text,
            intermediate_size=text.moe_intermediate_size
            * text.n_shared_experts,
        )
