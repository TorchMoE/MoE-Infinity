# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import time
from typing import Dict, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBLockSparseTop2MLP,
    rotate_half,
)

from moe_infinity.utils import ArcherConfig

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    device = position_ids.device
    position_ids = position_ids.to(cos.device)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim).to(q.device)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim).to(q.device)
    # print("cos.shape", cos.device, "sin.shape", sin.device, "q.shape", q.device, "k.shape", k.device)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    position_ids = position_ids.to(device)
    return q_embed, k_embed

class SyncMixtralSparseMoeBlock(nn.Module):
    archer_config: ArcherConfig = None
    layer_id: int = None

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)])

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)


        router_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        routing_weights_mask = (routing_weights[:, :, None] * router_mask).permute(
            0, 2, 1
        )
        router_mask = router_mask.permute(0, 2, 1)
        # assume top-2 here
        router_mask = torch.logical_or(router_mask[:, :, 0], router_mask[:, :, 1])
        routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)

        # print("selected_experts", selected_experts)
        expert_index = selected_experts.reshape(batch_size, sequence_length, self.top_k)
        for i in range(batch_size):
            seq_id = self.seq_id_list[i]
            # start_time = time.time()
            expert_matrix = self.expert_predictor.predict(seq_id, expert_index[i], self.layer_id)
            # print("predict", time.time() - start_time)
            # start_time = time.time()
            self.expert_prefetcher.prefetch_experts(self.layer_id, expert_matrix)
            # print("prefetch", time.time() - start_time)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        for expert_idx in range(self.num_experts):
            # expert_layer = self.experts[expert_idx]
            token_indices = router_mask[:, expert_idx]
            current_state = hidden_states[token_indices, :]

            if token_indices.any():
                current_hidden_states = (
                    self.experts[expert_idx](current_state).to(routing_weights_mask.device)
                    * routing_weights_mask[token_indices, expert_idx][:, None]
                )
                final_hidden_states[token_indices, :] += current_hidden_states

        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

