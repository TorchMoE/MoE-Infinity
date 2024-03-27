# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

from dataclasses import dataclass, field
import os
from transformers import HfArgumentParser
import torch


@dataclass
class ArcherConfig:
    offload_path: str = field(
        default="", metadata={"help": "Path to parameter storage"}
    )
    trace_capacity: int = field(
        default=1000, metadata={"help": "Capacity of trace"}
    )
    trace_path: os.PathLike = field(
        default=None, metadata={"help": "Path to trace file"}
    )
    # master_addr: str = field(
    #     default="127.0.0.1",
    #     metadata={"help": "Hosts for running archer"},
    # )
    # master_port: str = field(
    #     default=29500,
    #     metadata={"help": "Port for running archer"},
    # )
    # device_per_node: int = field(
    #     default=1,
    #     metadata={"help": "Number of devices per node"},
    # )
    device_memory_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of device memory to use"},
    )
    host_memory_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of host memory to use"},
    )

    @classmethod
    def load_from_file(self, config_path):
        parser = HfArgumentParser(self)
        self = parser.parse_json_file(json_file=config_path)[0]
        return self

    @classmethod
    def load_from_json(self, config_json):
        parser = HfArgumentParser(self)
        self = parser.parse_dict(config_json)[0]
        return self

    def __post_init__(self):
        self.perfect_cache_file = os.path.join(self.offload_path, "perfect_cache")

        self.device_per_node = (
            torch.cuda.device_count()
        )  # always run on heterogeneous nodes

        if self.trace_path is not None:
            self.trace_path = os.path.abspath(self.trace_path)
            if os.path.isdir(self.trace_path):
                raise ValueError(
                    "The trace path should be a file, not a directory."
                )
