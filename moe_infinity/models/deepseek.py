from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepseekMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if self.config.model_type == "deepseek_v2":
            from .modeling_deepseek import DeepseekV2MLP, MoEGate

            self.mlp_cls = DeepseekV2MLP
            self.gate_cls = MoEGate
        if self.config.model_type == "deepseek_v3":
            from .modeling_deepseek_v3 import DeepseekV3MLP, MoEGate

            self.mlp_cls = DeepseekV3MLP
            self.gate_cls = MoEGate

        self.experts = nn.ModuleList(
            [
                self.mlp_cls(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for i in range(config.n_routed_experts)
            ]
        )

        self.gate = self.gate_cls(config)
        if config.n_shared_experts is not None:
            intermediate_size = (
                config.moe_intermediate_size * config.n_shared_experts
            )
            self.shared_experts = self.mlp_cls(
                config=config, intermediate_size=intermediate_size
            )

        self.archer_tracer = None
        self.archer_engine = None
        self.expert_tensor_ids: Dict[int, int] = None

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape

        gate_output = self.gate(hidden_states)
        if len(gate_output) == 3:
            topk_idx, topk_weight, aux_loss = gate_output
        else:
            topk_idx, topk_weight = gate_output
            aux_loss = None
        # topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # print("topk_idx", topk_idx.shape)
        # print("topk_weight", topk_weight.shape)
        # print(self.config.n_routed_experts, self.config.num_experts_per_tok)

        # cnts = topk_idx.new_zeros((topk_idx.shape[0], len(self.experts)))
        # cnts.scatter_(1, topk_idx, 1)
        # tokens_per_expert = cnts.sum(dim=0)
        # idxs = topk_idx.view(-1).argsort()
        # sorted_tokens = hidden_states[idxs // topk_idx.shape[1]]

        # tokens_per_expert = tokens_per_expert.cpu().numpy()

        batch_size, sequence_length, hidden_dim = orig_shape
        router_mask = F.one_hot(
            topk_idx, num_classes=self.config.n_routed_experts
        )
        routing_weights_mask = (topk_weight[:, :, None] * router_mask).permute(
            0, 2, 1
        )
        routing_weights_mask = torch.sum(routing_weights_mask, dim=-1)
        router_mask = router_mask.permute(0, 2, 1)

        # use logical or to merge last dimension
        for i in range(self.config.num_experts_per_tok):
            router_mask[:, :, 0] = torch.logical_or(
                router_mask[:, :, 0], router_mask[:, :, i]
            )
        router_mask = router_mask[:, :, 0]
        # print("router_mask", router_mask.shape)
        # print("routing_weights_mask", routing_weights_mask.shape)

        # overlap current layer with unique expert list
        # unique_expert_list = torch.unique(topk_idx).tolist()
        # self.expert_prefetcher.fetch_experts_lock_cache(
        #     self.layer_id, unique_expert_list
        # )

        # self.expert_prefetcher.prefetch_experts_list(self.layer_id, unique_expert_list)

        # expert_index = topk_idx.reshape(
        #     batch_size, sequence_length, self.config.num_experts_per_tok
        # )
        # for i in range(batch_size):
        #     seq_id = self.seq_id_list[i]
        #     expert_matrix = self.expert_predictor.predict(
        #         seq_id, expert_index[i], self.layer_id
        #     )
        #     self.expert_prefetcher.prefetch_experts(
        #         self.layer_id, expert_matrix
        #     )

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        results = self.expert_executor.dispatch_local(
            hidden_states, router_mask, self.layer_id
        )
        for output, _, idx, _ in results:
            token_indices = router_mask[:, idx].bool()
            final_hidden_states[token_indices, :] += (
                output.to(routing_weights_mask.device)
                * routing_weights_mask[token_indices, idx][:, None]
            )

        final_hidden_states = final_hidden_states.view(
            batch_size, sequence_length, hidden_dim
        )
        if self.config.n_shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(
                identity
            )
        return final_hidden_states

        # outputs = []
        # start_idx = 0
        # for i, num_tokens in enumerate(tokens_per_expert):
        #     end_idx = start_idx + num_tokens
        #     if num_tokens == 0:
        #         continue
        #     expert = self.experts[i]
        #     tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        #     expert_out = expert(tokens_for_this_expert)
        #     outputs.append(expert_out.to(hidden_states.device))
        #     start_idx = end_idx

        # outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        # new_x = torch.empty_like(outs)
        # new_x[idxs] = outs
        # y = (
        #     new_x.view(*topk_idx.shape, -1)
        #     .type(topk_weight.dtype)
        #     .mul_(topk_weight.unsqueeze(dim=-1))
        #     .sum(dim=1)
        #     .type(new_x.dtype)
        # )
        # return y
