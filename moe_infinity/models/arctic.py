from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_infinity.utils import ArcherConfig

from .modeling_arctic import ArcticConfig, ArcticMLP


class SyncArcticMoeBlock(nn.Module):
    archer_config: ArcherConfig = None
    layer_id: int = None

    def __init__(self, config: ArcticConfig, layer_id: int, **kwargs):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        self.is_moe_layer = (layer_id + 1) % config.moe_layer_frequency == 0

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ArcticMLP(config) for i in range(self.num_experts)]
        )

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        expert_index = selected_experts.reshape(
            batch_size, sequence_length, self.top_k
        )
        for i in range(batch_size):
            seq_id = self.seq_id_list[i]
            expert_matrix = self.expert_predictor.predict(
                seq_id, expert_index[i], self.layer_id
            )
            self.expert_prefetcher.prefetch_experts(
                self.layer_id, expert_matrix
            )

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        return final_hidden_states, expert_mask
