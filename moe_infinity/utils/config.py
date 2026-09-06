# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportMissingTypeArgument=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

# EfficientMoE Team

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from transformers import HfArgumentParser


@dataclass
class ArcherConfig:
    offload_path: str = field(
        default="", metadata={"help": "Path to parameter storage"}
    )
    trace_capacity: int = field(
        default=1000, metadata={"help": "Capacity of trace"}
    )
    trace_path: Optional[os.PathLike[str]] = field(
        default=None, metadata={"help": "Path to trace file"}
    )
    perfect_cache_file: str = field(init=False)
    device_per_node: int = field(init=False)
    prefetch: bool = field(
        default=False, metadata={"help": "Enable prefetching"}
    )
    speculative_prefetch: bool = field(
        default=False,
        metadata={
            "help": "Enable speculative expert prefetching using router logits from layer L to predict L+1 experts."
        },
    )
    speculative_prefetch_overlap: bool = field(
        default=False,
        metadata={
            "help": "When True, fire speculative prefetch BEFORE the layer-L barrier in dispatch_local so PCIe transfers overlap with layer-L compute. When False (default), prefetch fires after the barrier (legacy behavior). Requires speculative_prefetch=True. Currently exposes a cache-pressure failure mode (see .sisyphus/findings/ibp-feasibility/SUMMARY.md) when device_memory_ratio is high; lower device_memory_ratio if you enable this and observe 'All cached expert locked' warnings."
        },
    )
    device_memory_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of device memory to use"},
    )
    num_threads: int = field(
        default=4,
        metadata={
            "help": "Number of parallel expert compute threads per GPU. Higher values overlap expert forward passes on separate CUDA streams, reducing pipeline bubbles."
        },
    )
    host_memory_ratio: float = field(
        default=0.9,
        metadata={"help": "Ratio of host memory to use"},
    )
    kv_cache_memory_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of GPU memory reserved for KV cache blocks. Default 0.0 (disabled). Set > 0 alongside enable_kv_cache_offload=True. Must satisfy: device_memory_ratio + kv_cache_memory_ratio <= 1.0"
        },
    )
    use_native_engine: bool = field(
        default=True,
        metadata={
            "help": "Enable native serving engine path. Default True. Set False to keep HuggingFace generate() path."
        },
    )
    enable_attention_offload: bool = field(
        default=False,
        metadata={
            "help": "Enable attention backend offloading. Default False (uses HuggingFace attention)."
        },
    )
    enable_kv_cache_offload: bool = field(
        default=False,
        metadata={
            "help": "Enable KV cache CPU offloading. Default False. Requires C++ extension support."
        },
    )
    attention_backend: str = field(
        default="default",
        metadata={
            "help": "Attention backend name. 'default' = no-op PlaceholderAttentionBackend."
        },
    )
    phase_specific_expert_policy: bool = field(
        default=False,
        metadata={
            "help": "Master gate for phase-specific expert admission, prefetch, and eviction policy (PR #179 substrate). Default False keeps legacy behavior. Adaptive precision does not require this to be True; when False the adaptive path still uses ExpertResidencyManager with neutral, legacy-equivalent phase utility."
        },
    )
    adaptive_expert_precision: bool = field(
        default=False,
        metadata={
            "help": "Opt-in adaptive mixed-precision expert policy. Default False. Never enabled by default; validated only when True."
        },
    )
    adaptive_hbm_budget_bytes: int = field(
        default=0,
        metadata={
            "help": "Fixed HBM budget in bytes for adaptive expert representations. Must be positive when adaptive_expert_precision is True."
        },
    )
    adaptive_policy_epoch_tokens: int = field(
        default=128,
        metadata={"help": "Tokens per adaptive policy epoch. Nonnegative."},
    )
    adaptive_hotness_decay: float = field(
        default=0.95,
        metadata={
            "help": "Per-epoch hotness decay factor. Must satisfy 0 < decay <= 1."
        },
    )
    adaptive_promotion_threshold: float = field(
        default=0.70,
        metadata={
            "help": "Hotness at or above which an expert is promoted. Must satisfy demotion < promotion <= 1."
        },
    )
    adaptive_demotion_threshold: float = field(
        default=0.30,
        metadata={
            "help": "Hotness below which an expert is demoted. Must satisfy 0 <= demotion < promotion."
        },
    )
    adaptive_min_residency_epochs: int = field(
        default=2,
        metadata={
            "help": "Minimum epochs a representation stays resident before transition. Nonnegative."
        },
    )
    adaptive_transition_cooldown_epochs: int = field(
        default=2,
        metadata={
            "help": "Cooldown epochs between transitions for an expert. Nonnegative."
        },
    )
    adaptive_variant_build: bool = field(
        default=False,
        metadata={
            "help": "Enable explicit candidate-build mode for derivative variants. Default False."
        },
    )
    adaptive_derivative_root: Optional[str] = field(
        default=None,
        metadata={
            "help": "Root for adaptive derivative artifacts. Resolves to <offload_path>/adaptive_derivatives when None and adaptive precision is enabled."
        },
    )

    @classmethod
    def load_from_file(cls, config_path: Union[str, os.PathLike]):
        parser = HfArgumentParser(cls)
        config = parser.parse_json_file(json_file=config_path)[0]
        return config

    @classmethod
    def load_from_json(cls, config_json: dict):
        if "glm_fp8_in_store" in config_json:
            warnings.warn(
                "glm_fp8_in_store is deprecated and ignored: GLM-5.2-FP8 routed "
                "experts are always kept FP8 in the host store.",
                DeprecationWarning,
                stacklevel=2,
            )
            config_json = {
                k: v for k, v in config_json.items() if k != "glm_fp8_in_store"
            }
        parser = HfArgumentParser(cls)
        config = parser.parse_dict(config_json)[0]
        return config

    def __post_init__(self):
        self.perfect_cache_file = os.path.join(
            self.offload_path, "perfect_cache"
        )

        self.device_per_node = (
            torch.cuda.device_count()
        )  # always run on heterogeneous nodes

        if self.trace_path is not None:
            self.trace_path = os.path.abspath(self.trace_path)
            if os.path.isdir(self.trace_path):
                raise ValueError(
                    "The trace path should be a file, not a directory."
                )

        kv_autocorrected = False
        if self.use_native_engine and self.kv_cache_memory_ratio == 0.0:
            self.kv_cache_memory_ratio = 0.15
            kv_autocorrected = True
            warnings.warn(
                "kv_cache_memory_ratio was 0.0 with use_native_engine=True; auto-set to 0.15.",
                UserWarning,
                stacklevel=2,
            )

        if (
            kv_autocorrected
            and self.device_memory_ratio + self.kv_cache_memory_ratio > 1.0
        ):
            self.device_memory_ratio = max(
                0.0, 1.0 - self.kv_cache_memory_ratio
            )
            warnings.warn(
                f"device_memory_ratio auto-adjusted to {self.device_memory_ratio:.2f} to satisfy memory budget.",
                UserWarning,
                stacklevel=2,
            )

        if not 0.0 <= self.device_memory_ratio <= 1.0:
            raise ValueError(
                f"device_memory_ratio must be in [0, 1], got {self.device_memory_ratio}"
            )
        if not 0.0 <= self.kv_cache_memory_ratio <= 1.0:
            raise ValueError(
                f"kv_cache_memory_ratio must be in [0, 1], got {self.kv_cache_memory_ratio}"
            )
        if self.device_memory_ratio + self.kv_cache_memory_ratio > 1.0:
            raise ValueError(
                f"device_memory_ratio ({self.device_memory_ratio}) + kv_cache_memory_ratio ({self.kv_cache_memory_ratio}) > 1.0"
            )

        if self.adaptive_expert_precision:
            if self.adaptive_hbm_budget_bytes <= 0:
                raise ValueError(
                    "adaptive_hbm_budget_bytes must be positive when "
                    "adaptive_expert_precision is enabled"
                )
            if self.adaptive_policy_epoch_tokens < 0:
                raise ValueError(
                    "adaptive_policy_epoch_tokens must be a nonnegative integer"
                )
            if self.adaptive_min_residency_epochs < 0:
                raise ValueError(
                    "adaptive_min_residency_epochs must be a nonnegative integer"
                )
            if self.adaptive_transition_cooldown_epochs < 0:
                raise ValueError(
                    "adaptive_transition_cooldown_epochs must be a nonnegative integer"
                )
            if not 0.0 < self.adaptive_hotness_decay <= 1.0:
                raise ValueError(
                    "adaptive_hotness_decay must satisfy 0 < decay <= 1"
                )
            if not (
                0.0
                <= self.adaptive_demotion_threshold
                < self.adaptive_promotion_threshold
                <= 1.0
            ):
                raise ValueError(
                    "adaptive thresholds must satisfy 0 <= demotion < promotion <= 1"
                )
            if self.adaptive_derivative_root is None:
                self.adaptive_derivative_root = os.path.join(
                    self.offload_path, "adaptive_derivatives"
                )
