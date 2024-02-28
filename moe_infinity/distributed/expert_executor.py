# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import torch.distributed.rpc as rpc
import torch.distributed as dist
import torch
import numpy as np
from moe_infinity.utils import ArcherConfig


def _call_expert_dispatcher(method, *args, **kwargs):
    global _expert_dispatcher
    func = getattr(_expert_dispatcher, method)
    return func(*args, **kwargs)


class DistributedExpertExecutor:

    def __init__(self, archer_config: ArcherConfig):
        self.archer_config = archer_config

    def set_expert_dispatcher(self, expert_dispatcher):
        global _expert_dispatcher
        _expert_dispatcher = expert_dispatcher
        self.expert_dispatcher = expert_dispatcher

    def set_device_map_manager(self, device_map_manager):
        self.device_map_manager = device_map_manager

    def dispatch(self, hidden_states, router_mask, layer_id):

        # if len(router_mask.shape) >= 3:
        #     router_mask = router_mask.squeeze(0)
        
        # print("router_mask", router_mask.shape, "rank", dist.get_rank())
        # print("hidden_states", hidden_states.shape, "rank", dist.get_rank())

        num_expert = router_mask.shape[-1]
        expert_count = torch.sum(router_mask.view((-1, num_expert)), dim=0).cpu().numpy().flatten()

        expert_list = np.arange(num_expert).astype(int)[
            expert_count > 0].tolist()
        
        # print("expert_list", expert_list, "rank", dist.get_rank())
        
        
        # n_tokens = hidden_states.shape[1] * hidden_states.shape[0]
        # n_experts = router_mask.shape[-1]

        # expert_counts = torch.sum(router_mask,
        #                           dim=(0, 1)).cpu().numpy().flatten()
        # expert_counts = expert_counts / n_tokens

        # # original_device = hidden_states.device

        # expert_list = np.arange(n_experts).astype(int)[
        #     expert_counts > 0].tolist()
        # expert_counts = expert_counts[expert_counts > 0].tolist()

        device_list = self.device_map_manager.get_target_device(expert_list)
        visited_ranks = set()
        rank_wait_cnt = {r: 0 for r in range(dist.get_world_size())}
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            visited_ranks.add(rank)
            rank_wait_cnt[rank] += 1

        futures = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                future = rpc.rpc_async(f"worker_{rank}",
                                       _call_expert_dispatcher,
                                       args=("set_inputs", hidden_states.cpu(),
                                             router_mask.cpu()))
                futures.append(future)
                future = rpc.rpc_async(f"worker_{rank}",
                                       _call_expert_dispatcher,
                                       args=("set_expected_queue",
                                             rank_wait_cnt[rank]))
                futures.append(future)
            else:
                self.expert_dispatcher.set_inputs(hidden_states, router_mask)
                self.expert_dispatcher.set_expected_queue(rank_wait_cnt[rank])

        # wait for all futures
        for future in futures:
            future.wait()

        futures = []
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            if rank == dist.get_rank():
                self.expert_dispatcher.enqueue_expert(layer_id, expert_id,
                                                      gpu_id, False)
            else:
                future = rpc.rpc_async(f"worker_{rank}",
                                       _call_expert_dispatcher,
                                       args=("enqueue_expert", layer_id,
                                             expert_id, gpu_id, True))
                futures.append(future)

        # wait for all futures
        for future in futures:
            future.wait()

        result_list = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                result = rpc.rpc_sync(f"worker_{rank}",
                                      _call_expert_dispatcher,
                                      args=("wait_expert", ))
                result_list += result
            else:
                result = self.expert_dispatcher.wait_expert()
                result_list += result

        return result_list
