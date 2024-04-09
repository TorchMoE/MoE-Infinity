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


def _call_expert_prefetcher(method, *args, **kwargs):
    global _expert_prefetcher
    func = getattr(_expert_prefetcher, method)
    return func(*args, **kwargs)


class DistributedExpertPrefetcher(object):
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

    def set_device_map_manager(self, device_map_manager):
        self.device_map_manager = device_map_manager

    def set_archer_prefetch(self, archer_prefetch):
        self.archer_prefetch = archer_prefetch

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

        device_list = self.device_map_manager.get_target_device(tensor_ids)

        if len(tensor_ids) > 0:
            self._replace_cache_candidates(tensor_ids)
            for meta in device_list:
                rank, gpu_id, tensor_id = meta
                if rank == dist.get_rank():
                    self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)
                else:
                    rpc.rpc_async(
                        f"worker_{rank}",
                        _call_expert_prefetcher,
                        args=("enqueue_prefetch", tensor_id, gpu_id),
                    )

    def _replace_cache_candidates(self, tensor_ids):
        futures = []
        for k in range(dist.get_world_size()):
            if k == dist.get_rank():
                self.archer_engine.replace_cache_candidates(tensor_ids)
            else:
                future = rpc.rpc_async(
                    f"worker_{k}",
                    _call_expert_prefetcher,
                    args=("replace_cache_candidates", tensor_ids),
                )
                futures.append(future)
