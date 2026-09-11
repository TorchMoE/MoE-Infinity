# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.jamba.modeling_jamba import JambaMLP


class SyncJambaMoEBlock(nn.Module):
    layer_id: int = None

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [JambaMLP(config) for _ in range(self.num_experts)]
        )

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        router_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        routing_weights_mask = (
            routing_weights[:, :, None] * router_mask
        ).permute(0, 2, 1)
        router_mask = router_mask.permute(0, 2, 1)
        router_mask = torch.any(router_mask, dim=-1)
        routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)

        self.expert_executor.dispatch_local(
            self.layer_id,
            hidden_states,
            router_mask,
            routing_weights_mask,
            router_logits=router_logits,
        )
        final_hidden_states = self.expert_executor.wait_dispatch_local()

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        ).to(hidden_states.dtype)
        # transformers v5 decoder layers consume feed_forward(...) as a bare
        # tensor: `hidden_states = residual + hidden_states`. Returning the
        # v4 (tensor, router_logits) tuple makes that addition raise
        # `TypeError: unsupported operand type(s) for +: 'Tensor' and 'tuple'`.
        return final_hidden_states
