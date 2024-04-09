# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

# The global device manager shared among all nodes, using grpc server to communicate with each other.
from typing import Tuple, List
import numpy as np
import random
from moe_infinity.utils import ArcherConfig
import torch.distributed as dist

class DeviceMapManager:
    def __init__(self, archer_config: ArcherConfig) -> None:
        world_size = dist.get_world_size()
        device_per_node = archer_config.device_per_node

        total_device = world_size * device_per_node
        if total_device > 1:
            self.num_device_plan = [1] + [
                x for x in range(2, total_device + 1, 2)
            ]
        else:
            self.num_device_plan = [1]

        self.device_per_node = device_per_node
        self.total_device = total_device
        self.world_size = world_size

    def set_expert_tensor_map(self, expert_tensor_map):
        self.expert_tensor_map = expert_tensor_map

    def set_archer_engine(self, archer_engine):
        self.archer_engine = archer_engine

    def get_target_device(self, expert_list: List[int]) -> List[Tuple[int, int, int]]:
        num_experts = len(expert_list)
        num_device = self.total_device

        # index = np.argsort(expert_counts)[::-1]
        # expert_list = np.array(expert_list)
        # expert_list = expert_list[index]

        device_list = []
        k = 0

        # scatter the experts to all GPUs
        r = num_device % self.world_size
        world_size = num_device // self.world_size

        if r > 0:
            world_size += 1

        base = 1 if self.world_size > 1 else 0

        gpu_ids = [id for id in range(self.device_per_node)]

        # scatter the experts to all GPUs
        while k < num_experts:
            random.shuffle(gpu_ids)
            for rank in range(base, min(world_size + base, self.world_size)):
                for gpu in gpu_ids:
                    if k >= num_experts:
                        break
                    expert_id = expert_list[k]
                    device_list.append((rank, gpu, expert_id))
                    k += 1

        return device_list


