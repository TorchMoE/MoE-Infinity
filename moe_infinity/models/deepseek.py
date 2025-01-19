from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn

from .modeling_deepseek import DeepseekV2MLP, MoEGate


class DeepseekV2MoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                DeepseekV2MLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for i in range(config.n_routed_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                config=config, intermediate_size=intermediate_size
            )
            
            
        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        cnts = topk_idx.new_zeros((topk_idx.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = hidden_states[idxs // topk_idx.shape[1]]
        
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        
        batch_size, sequence_length, _ = orig_shape
        router_mask = F.one_hot(topk_idx, num_classes=self.config.n_routed_experts)
        
        # print("router_mask", router_mask.shape)
        
        expert_index = topk_idx.reshape(batch_size, sequence_length, self.config.num_experts_per_tok)
        for i in range(batch_size):
            seq_id = self.seq_id_list[i]
            expert_matrix = self.expert_predictor.predict(seq_id, expert_index[i], self.layer_id)
            self.expert_prefetcher.prefetch_experts(self.layer_id, expert_matrix)

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(hidden_states.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        y = (
            new_x.view(*topk_idx.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y