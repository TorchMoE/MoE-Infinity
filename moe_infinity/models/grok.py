from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..modeling_grok.configuration_grok1 import Grok1Config
from ..modeling_grok.modeling_grok1 import MoeBlock, MoeMLP, rotate_half


from moe_infinity.utils import ArcherConfig


class SyncGrokMoeBlock(nn.Module):
    archer_config: ArcherConfig = None
    layer_id: int = None
    
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)        
        self.experts = nn.ModuleList([MoeMLP(hidden_dim, ffn_dim) for _ in range(self.num_experts)])

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

        expert_index = selected_experts.reshape(batch_size, sequence_length, self.top_k)
        for i in range(batch_size):
            seq_id = self.seq_id_list[i]
            expert_matrix = self.expert_predictor.predict(seq_id, expert_index[i], self.layer_id)
            self.expert_prefetcher.prefetch_experts(self.layer_id, expert_matrix)

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

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            # print(f"current hidden_states device: {current_state.device} expert_layer device: {expert_layer}")
            current_hidden_states = (
                expert_layer(current_state).to(routing_weights.device)
                * routing_weights[top_x_list, idx_list, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits