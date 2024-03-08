import copy
import os
import time
from typing import Union
import numpy as np
import uuid
from collections import Counter
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
from transformers import PretrainedConfig

# from sklearn.metrics.pairwise import cosine_similarity

from moe_infinity.memory.expert_entry import ExpertTraceEntry
from moe_infinity.utils import parse_moe_param


class ExpertTracer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExpertTracer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, capacity: int, config:PretrainedConfig):
        self.num_layers, self.num_experts, self.num_encoder_layers = parse_moe_param(config)
        self.capacity = capacity

        self.trace = {}

        self.trace_collection = torch.zeros((capacity, self.num_layers, self.num_experts), device="cuda:0")
        self.collection_access = np.zeros((capacity,))

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def load_trace(self, trace: Union[os.PathLike, np.ndarray]):
        if isinstance(trace, os.PathLike):
            self.trace_collection = torch.from_numpy(np.load(trace, allow_pickle=False))
        elif isinstance(trace, np.ndarray):
            self.trace_collection = trace
        
        self.persistent_capacity = self.trace_collection.shape[0]
        assert self.persistent_capacity <= self.capacity, (
            f"loaded trace capacity {self.persistent_capacity} must be "
            f"less than or equal to capacity in config {self.capacity}"
        )


    def create_entry(self):
        seq_id = uuid.uuid4().hex
        self.trace[seq_id] = ExpertTraceEntry(
            seq_id, np.zeros((self.num_layers, self.num_experts)), 0, 0
        )
        return seq_id

    def finish_entry(self, seq_id):
        trace_sum = np.sum(self.trace_collection, axis=(1, 2))

        if np.any(trace_sum == 0):
            # find the first zero entry
            idx = np.argwhere(trace_sum == 0)[0][0]
            self.trace_collection[idx] = self.trace[seq_id].matrix
            self.collection_access[idx] = 1
        else:
            # find the first entry after self.persistent_capacity that has the least access
            collection_access_copy = self.collection_access.copy()
            collection_access_copy[: self.persistent_capacity] = 1e9

            idx = np.argmin(collection_access_copy)
            self.trace_collection[idx] = self.trace[seq_id].matrix
            self.collection_access[idx] = 1

    def update_entry(self, seq_id, expert_list, layer_idx):
        expert_counter = Counter(expert_list.flatten().tolist())
        for key, count in expert_counter.items():
            self.trace[seq_id].matrix[layer_idx, key] += count

        if layer_idx == self.num_layers - 1:
            self.trace[seq_id].num_new_tokens += 1

    def get_entry_decoder(self, seq_id):
        entry = copy.deepcopy(self.trace[seq_id])
        entry.matrix[: self.num_encoder_layers, :] = 0
        return entry

    def get_entry(self, seq_id):
        return self.trace[seq_id]

    def find_most_similar(self, matrix, layer_idx) -> np.ndarray:
        # start_time = time.time()
        trace_collection_copy = self.trace_collection.clone()
        trace_collection_copy[:, : (layer_idx + 1), :] = 1e-9
        # print("trace_collection copy", time.time() - start_time)

        trace_collection_copy /= torch.sum(trace_collection_copy, dim=2, keepdims=True)

        matrix_copy = torch.from_numpy(matrix.copy()).to("cuda:0")
        matrix_copy /= torch.sum(matrix_copy, dim=1, keepdims=True)
        replicated_matrix_copy = torch.concat(
            [matrix_copy[None, ...]] * self.capacity, dim=0
        )

        # fill nan with 0 using torch
        replicated_matrix_copy = torch.nan_to_num(replicated_matrix_copy)
        matrix_copy = torch.nan_to_num(matrix_copy)

        cos_sim = self.cos(replicated_matrix_copy, trace_collection_copy)
        # print("cos_sim", time.time() - start_time)

        # print(cos_sim.shape)

        cos_dist = 1 - torch.mean(cos_sim, dim=1)
        min_idx = torch.argmin(cos_dist).item()

        self.collection_access[min_idx] += 1

        entry = self.trace_collection[min_idx].to("cpu").numpy()
        return entry

