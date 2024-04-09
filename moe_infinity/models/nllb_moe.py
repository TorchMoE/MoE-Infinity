# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import NllbMoeConfig
from transformers.models.nllb_moe.modeling_nllb_moe import (
    NllbMoeTop2Router,
    NllbMoeDenseActDense,
)

from moe_infinity.utils import ArcherConfig

GPU_IDX_COUNTER = 0


class SyncNllbMoeSparseMLP(nn.Module):

    archer_config: ArcherConfig = None
    layer_id: int = None

    def __init__(
        self,
        config: NllbMoeConfig,
        ffn_dim: int,
        expert_class: nn.Module = NllbMoeDenseActDense,
    ):
        super().__init__()
        self.router = NllbMoeTop2Router(config)
        self.moe_token_dropout = config.moe_token_dropout
        self.token_dropout = nn.Dropout(self.moe_token_dropout)

        self.num_experts = config.num_experts

        self.experts = nn.ModuleDict()
        for idx in range(self.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config, ffn_dim)

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def forward(self,
                hidden_states: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        top_1_mask, router_probs = self.router(hidden_states, padding_mask)
        combining_weights = router_probs.reshape(
            (batch_size, sequence_length, self.num_experts))
        router_mask = combining_weights.bool()

        next_states = torch.zeros_like(hidden_states)
        top_1_expert_index = torch.argmax(top_1_mask, dim=-1)

        logits_except_top_1 = router_probs.masked_fill(top_1_mask.bool(), float("-inf"))
        top_2_expert_index = torch.argmax(logits_except_top_1, dim=-1)
        # top_2_mask = torch.nn.functional.one_hot(top_2_expert_index, num_classes=self.num_experts)

        expert_index = torch.stack([top_1_expert_index, top_2_expert_index], dim=-1)
        expert_index = expert_index.reshape(batch_size, sequence_length, 2)

        for i in range(batch_size):
            seq_id = self.seq_id_list[i]
            expert_matrix = self.expert_predictor.predict(seq_id, expert_index[i], self.layer_id)
            self.expert_prefetcher.prefetch_experts(self.layer_id, expert_matrix)
        
        # self.expert_prefetcher.prefetch_tensors(self.layer_id, router_mask,
        #                                         self.expert_tensor_ids,
        #                                         n_tokens)
            
        for expert_id, expert in self.experts.items():
            idx = int(expert_id.split("_")[-1])
            token_indices = router_mask[:, :, idx].bool()
            weights = combining_weights[..., idx]

            if token_indices.any():
                expert_output = expert(hidden_states[token_indices]).to(weights.device)
                next_states[token_indices] += torch.einsum("b,be->be", weights[token_indices], expert_output)

        next_states[next_states == 0] = hidden_states[next_states == 0]
        hidden_states = next_states

        return hidden_states, (router_probs.to("cuda:0", non_blocking=True),
                               top_1_expert_index.to("cuda:0",
                                                     non_blocking=True))
