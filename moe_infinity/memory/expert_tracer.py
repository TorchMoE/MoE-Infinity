# pyright: reportMissingTypeArgument=false, reportUninitializedInstanceVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
import copy
import json
import os
import time
import uuid
from collections import Counter, deque
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from moe_infinity.memory.expert_entry import ExpertTraceEntry
from moe_infinity.utils import parse_moe_param

_io_profile_env_cache = None


class ExpertTracer:
    _instance = None
    _io_events = None
    _io_profiling_enabled = None
    _VALID_IO_STAGES = {
        "routing",
        "cache_lookup",
        "prefetch_predict",
        "transfer_schedule",
        "disk_to_cpu",
        "cpu_to_gpu",
        "sync_wait",
        "expert_compute",
        "eviction",
        "queue_coordination",
        "prefetch_budget",
        "prefetch_admit",
        "prefetch_complete",
        "prefetch_cancel",
        "prefetch_late",
        "prefetch_waste",
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExpertTracer, cls).__new__(cls)
        return cls._instance

    def __init__(self, capacity: int, config: PretrainedConfig):
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        self.capacity = capacity

        self.trace = {}

        self.trace_collection = torch.zeros(
            (capacity, self.num_layers, self.num_experts), device="cuda:0"
        )
        self.collection_access = np.zeros((capacity,))

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def load_trace(self, trace: Union[os.PathLike, np.ndarray]):
        if isinstance(trace, os.PathLike):
            self.trace_collection = torch.from_numpy(
                np.load(trace, allow_pickle=False)
            )
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
        trace_collection_copy = self.trace_collection.clone()
        trace_collection_copy[:, : (layer_idx + 1), :] = 1e-9

        trace_collection_copy /= torch.sum(
            trace_collection_copy, dim=2, keepdims=True
        )

        matrix_copy = torch.from_numpy(matrix.copy()).to("cuda:0")
        matrix_copy /= torch.sum(matrix_copy, dim=1, keepdims=True)
        replicated_matrix_copy = torch.concat(
            [matrix_copy[None, ...]] * self.capacity, dim=0
        )

        # fill nan with 0 using torch
        replicated_matrix_copy = torch.nan_to_num(replicated_matrix_copy)
        matrix_copy = torch.nan_to_num(matrix_copy)

        cos_sim = self.cos(replicated_matrix_copy, trace_collection_copy)

        cos_dist = 1 - torch.mean(cos_sim, dim=1)
        min_idx = torch.argmin(cos_dist).item()

        self.collection_access[min_idx] += 1

        entry = self.trace_collection[min_idx].to("cpu").numpy()
        return entry

    def _init_io_tracking(self):
        global _io_profile_env_cache

        if getattr(self, "_io_events", None) is None:
            self._io_events = deque(maxlen=10000)

        if getattr(self, "_io_profiling_enabled", None) is None:
            if _io_profile_env_cache is None:
                _io_profile_env_cache = bool(
                    os.environ.get("MOE_INFINITY_PROFILE_IO")
                )
            self._io_profiling_enabled = _io_profile_env_cache

    def record_io_event(
        self,
        layer_idx,
        expert_id,
        stage,
        duration_ns,
        bytes_transferred=0,
    ):
        if (
            getattr(self, "_io_events", None) is None
            or getattr(self, "_io_profiling_enabled", None) is None
        ):
            self._init_io_tracking()

        if self._io_profiling_enabled is None or not self._io_profiling_enabled:
            return

        if stage not in self._VALID_IO_STAGES:
            raise ValueError(f"Unsupported I/O stage: {stage}")

        io_events = self._io_events
        if io_events is None:
            return

        io_events.append(
            {
                "ts_ns": int(time.time_ns()),
                "layer_idx": int(layer_idx),
                "expert_id": int(expert_id),
                "stage": stage,
                "duration_ns": int(duration_ns),
                "bytes_transferred": int(bytes_transferred),
            }
        )

    def get_io_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        if (
            getattr(self, "_io_events", None) is None
            or getattr(self, "_io_profiling_enabled", None) is None
        ):
            self._init_io_tracking()

        if self._io_profiling_enabled is None or not self._io_profiling_enabled:
            return {}

        io_events = self._io_events
        if not io_events:
            return {}

        duration_by_stage = {}
        bytes_by_stage = Counter()

        for event in io_events:
            stage = event["stage"]
            duration_by_stage.setdefault(stage, []).append(event["duration_ns"])
            bytes_by_stage[stage] += event.get("bytes_transferred", 0)

        stats = {}
        for stage, durations in duration_by_stage.items():
            duration_arr = np.array(durations, dtype=np.float64)
            stats[stage] = {
                "count": int(duration_arr.size),
                "min_ns": int(np.min(duration_arr)),
                "max_ns": int(np.max(duration_arr)),
                "mean_ns": float(np.mean(duration_arr)),
                "p50_ns": float(np.percentile(duration_arr, 50)),
                "p95_ns": float(np.percentile(duration_arr, 95)),
                "p99_ns": float(np.percentile(duration_arr, 99)),
                "total_bytes": int(bytes_by_stage[stage]),
            }

        return stats

    def to_jsonl(self, filepath: Union[str, os.PathLike[str]]):
        if (
            getattr(self, "_io_events", None) is None
            or getattr(self, "_io_profiling_enabled", None) is None
        ):
            self._init_io_tracking()

        io_events = self._io_events
        if io_events is None:
            io_events = []

        with open(filepath, "w", encoding="utf-8") as file:
            for event in io_events:
                file.write(json.dumps(event))
                file.write("\n")
