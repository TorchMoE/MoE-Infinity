# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import time
import numpy as np
import torch
from torch.distributed import rpc
import torch.distributed as dist

from transformers import PretrainedConfig
from moe_infinity.utils import parse_moe_param

class ExpertPrefetcher(object):
    cache_file_rd = None

    def __init__(self, config: PretrainedConfig):
        print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers = parse_moe_param(
            config
        )

    def set_archer_engine(self, archer_engine):
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    def prefetch_experts(self, layer_id, expert_matrix):
        expert_list = []
        # print("expert_tensor_map", self.expert_tensor_map)
        for i in range(layer_id, self.num_layers):
            for j in range(self.num_experts):
                if expert_matrix[i, j] > 0:
                    expert_list.append(
                        (self.expert_tensor_map[(i,j)], expert_matrix[i, j])
                    )
        ordered_expert_list = sorted(expert_list, key=lambda x: x[1], reverse=True)
        tensor_ids = [x[0] for x in ordered_expert_list]
        
        self.archer_engine.replace_cache_candidates(tensor_ids)
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
