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

    # def __init__(
    #     self,
    #     num_encoder_layers,
    #     capacity=None,
    #     num_layers=None,
    #     num_experts=None,
    # ) -> None:
    #     # if (
    #     #     capacity is None or num_layers is None or num_experts is None
    #     # ) and trace is None:
    #     #     raise ValueError(
    #     #         "Either (capacity, num_layers, num_experts) or trace must be provided"
    #     #     )

    #     self.persistent_capacity = 0
    #     self.capacity = capacity
    #     self.num_experts = num_experts
    #     self.num_layers = num_layers
    #     self.num_encoder_layers = num_encoder_layers

    #     self.trace = {}

    #     self.trace_collection = np.zeros((capacity, num_layers, num_experts))
    #     self.collection_access = np.zeros((capacity,))

    #     # if trace is not None:
    #     #     self.load_trace(trace)

    #     #     self.persistent_capacity = self.trace_collection.shape[0]
    #     #     self.num_layers = self.trace_collection.shape[1]
    #     #     self.num_experts = self.trace_collection.shape[2]

    #     #     assert self.persistent_capacity <= self.capacity, (
    #     #         f"loaded trace capacity {self.persistent_capacity} must be "
    #     #         f"less than or equal to capacity in config {self.capacity}"
    #     #     )

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

        # assert np.allclose(replicated_matrix_copy[0], matrix_copy), (
        #     replicated_matrix_copy[0],
        #     matrix_copy,
        # )

        # print(replicated_matrix_copy.sum())
        # print(trace_collection_copy.sum())

        # print("replicated_matrix_copy", replicated_matrix_copy[0])
        # print("trace_collection_copy", trace_collection_copy[0])

        # start_time = time.time()
        cos_sim = self.cos(replicated_matrix_copy, trace_collection_copy)
        # print("cos_sim", time.time() - start_time)

        # print(cos_sim.shape)

        cos_dist = 1 - torch.mean(cos_sim, dim=1)
        min_idx = torch.argmin(cos_dist).item()

        self.collection_access[min_idx] += 1

        entry = self.trace_collection[min_idx].to("cpu").numpy()
        return entry

        # for key, entry in self.trace.items():
        #     sum_distance = 0

        #     if key == seq_id:
        #         continue

        #     for l in range(layer_idx + 1):
        #         sum_m = np.sum(matrix[l])
        #         sum_e = np.sum(entry.matrix[l])

        #         if sum_m == 0 or sum_e == 0:
        #             continue

        #         cosine_distance = cosine(
        #             matrix[l] / np.sum(matrix[l]),
        #             entry.matrix[l] / np.sum(entry.matrix[l]),
        #         )
        #         sum_distance += cosine_distance
        #     sum_distance /= layer_idx + 1
        #     if sum_distance < smallest_distance:
        #         smallest_distance = sum_distance
        #         most_similar = entry

        # return most_similar


# if __name__ == "__main__":
#     num_layers = 12
#     num_experts = 128
#     num_encoder_layers = 6
#     capacity = 100
#     expert_tracer = ExpertTracer(num_encoder_layers, capacity, num_layers, num_experts)

#     # random = np.random.randint(0, num_experts, (capacity, num_layers, num_experts)).astype(float)
#     # expert_tracer.load_trace(random)

#     for _ in range(1):
#         seq_id = expert_tracer.create_entry()
#         for k in range(num_layers):
#             experts = np.random.randint(0, num_experts, (num_experts))
#             expert_tracer.update_entry(seq_id, experts, k, np.unique(experts).shape[0])

#         expert_tracer.finish_entry(seq_id)

#     # random = np.random.randint(0, 2, (num_layers, num_experts)).astype(float)
#     random = expert_tracer.trace_collection[0]
#     # print("random", random, np.sum(random))

#     most_similar = expert_tracer.find_most_similar(random, num_layers - 1)

#     # print(most_similar, np.sum(most_similar))

#     assert np.allclose(random, most_similar)
