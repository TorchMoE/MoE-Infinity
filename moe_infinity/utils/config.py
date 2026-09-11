# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportMissingTypeArgument=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

# EfficientMoE Team

import math
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
    gpu_only_expert_routing: bool = field(
        default=False,
        metadata={
            "help": (
                "Use native CUDA active-expert discovery for single-host local "
                "dispatch. Falls back to eager Python routing when unavailable."
            )
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
    enable_deepseek_mla_paging: bool = field(
        default=False,
        metadata={
            "help": "Enable experimental batch-one DeepSeek V2/V3 MLA paging. Default False."
        },
    )
    max_resident_paged_speculative_sessions: int = field(
        default=1,
        metadata={
            "help": "Maximum concurrent resident paged-MLA speculative sessions. Default 1; set 0 to force Stage 4a fallback."
        },
    )
    min_free_mla_blocks_after_admission: int = field(
        default=1,
        metadata={
            "help": "Minimum MLA blocks that remain free after reserving all active and newly admitted requests' full declared budgets plus maximum transient DFlash verify peaks. Default 1."
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
    adaptive_memory_enabled: bool = False
    adaptive_memory_interval_steps: int = 64
    adaptive_memory_cooldown_steps: int = 256
    adaptive_memory_ewma_alpha: float = 0.20
    adaptive_memory_hysteresis_ratio: float = 0.15
    adaptive_memory_max_resize_step_bytes: int = 256 * 1024**2
    adaptive_memory_min_expert_cache_bytes: int = 512 * 1024**2
    adaptive_memory_min_kv_cache_blocks: int = 128
    adaptive_memory_free_reserve_bytes: int = 1024 * 1024**2
    adaptive_memory_failure_limit: int = 3
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
    prefill_expert_admission: str = field(
        default="transient_on_pressure",
        metadata={
            "help": "Prefill admission mode: cache or transient_on_pressure."
        },
    )
    decode_expert_admission: str = field(
        default="cache",
        metadata={
            "help": "Decode admission mode: cache or transient_on_pressure."
        },
    )
    prefill_expert_prefetch_top_k: int = field(
        default=0,
        metadata={
            "help": "Predictive prefill prefetch top-k in [0, num_experts]; zero disables."
        },
    )
    decode_expert_prefetch_top_k: int = field(
        default=2,
        metadata={
            "help": "Predictive decode prefetch top-k in [0, num_experts]."
        },
    )
    prefill_expert_prefetch_priority: int = field(
        default=2,
        metadata={"help": "Native prefill prefetch band in [1, 19]."},
    )
    decode_expert_prefetch_priority: int = field(
        default=1,
        metadata={"help": "Native decode prefetch band in [1, 19]."},
    )
    prefill_expert_eviction_weight: float = field(
        default=1.0,
        metadata={"help": "Prefill eviction weight; finite and > 0."},
    )
    decode_expert_eviction_weight: float = field(
        default=4.0,
        metadata={"help": "Decode eviction weight; finite and > 0."},
    )
    expert_policy_starvation_limit: int = field(
        default=8,
        metadata={
            "help": "Positive maximum prefetch bypasses before promotion."
        },
    )
    overlap_prefetch_policy: str = field(
        default="off",
        metadata={
            "help": "off, observe, or enforce overlap-window byte admission for speculative expert prefetch."
        },
    )
    overlap_prefetch_ewma_alpha: float = field(
        default=0.2,
        metadata={
            "help": "EWMA smoothing factor in (0, 1] for compute/bandwidth/queue/issue calibration."
        },
    )
    overlap_prefetch_safety_factor: float = field(
        default=0.8,
        metadata={
            "help": "Fraction in (0, 1] of measured compute time usable as the transfer overlap window."
        },
    )
    overlap_prefetch_cold_start_experts: int = field(
        default=1,
        metadata={
            "help": "Max experts admitted before both a compute and a transfer sample exist."
        },
    )
    overlap_prefetch_max_window_bytes: int = field(
        default=256 * 1024 * 1024,
        metadata={
            "help": "Upper bound on the per-layer admitted prefetch window in bytes."
        },
    )
    overlap_prefetch_max_inflight_bytes: int = field(
        default=512 * 1024 * 1024,
        metadata={
            "help": "Upper bound on outstanding speculative prefetch bytes."
        },
    )
    kv_cache_format: str = field(
        default="native",
        metadata={
            "help": "KV storage format: native or int8_sym; default preserves model dtype."
        },
    )
    kv_cache_allow_fallback: bool = field(
        default=True,
        metadata={
            "help": "Allow a visible native fallback when requested KV format is unsupported."
        },
    )
    kv_swap_mode: str = field(
        default="sync",
        metadata={
            "help": "Serving KV swap backend: 'sync' (blocking pageable copies, default) or 'async' (bounded event-driven pinned transfers)."
        },
    )
    kv_swap_host_memory_bytes: int = field(
        default=512 * 1024 * 1024,
        metadata={
            "help": "Hard cap on pinned host memory for async KV swap. Accepted but inactive in sync mode."
        },
    )
    kv_swap_max_inflight_bytes: int = field(
        default=256 * 1024 * 1024,
        metadata={
            "help": "Hard cap on in-flight (incomplete-event) async KV transfer bytes. Accepted but inactive in sync mode."
        },
    )
    kv_swap_checksum: bool = field(
        default=False,
        metadata={
            "help": "Enable opt-in CRC32 validation of swapped KV payloads before H2D submission."
        },
    )
    kv_swap_max_retries: int = field(
        default=2,
        metadata={
            "help": "Maximum async swap-in retries before terminal reprefill."
        },
    )
    kv_swap_allow_sync_fallback: bool = field(
        default=True,
        metadata={
            "help": "When async is requested but CUDA/pinned allocation is unavailable at init, fall back to sync instead of raising."
        },
    )

    @staticmethod
    def _validate_gpu_routing_overlap(
        gpu_only_expert_routing: bool,
        speculative_prefetch_overlap: bool,
        overlap_prefetch_mode: str = "off",
    ) -> None:
        if gpu_only_expert_routing and (
            speculative_prefetch_overlap
            or overlap_prefetch_mode in {"observe", "enforce"}
        ):
            raise ValueError(
                "gpu_only_expert_routing cannot be combined with overlap "
                "prefetch in the first release; disable "
                "speculative_prefetch_overlap and overlap_prefetch_mode"
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
        self._validate_gpu_routing_overlap(
            self.gpu_only_expert_routing,
            self.speculative_prefetch_overlap,
            getattr(self, "overlap_prefetch_mode", "off"),
        )

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
        positive_adaptive = (
            "adaptive_memory_interval_steps",
            "adaptive_memory_cooldown_steps",
            "adaptive_memory_max_resize_step_bytes",
            "adaptive_memory_min_expert_cache_bytes",
            "adaptive_memory_min_kv_cache_blocks",
            "adaptive_memory_free_reserve_bytes",
            "adaptive_memory_failure_limit",
        )
        for name in positive_adaptive:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 < self.adaptive_memory_ewma_alpha <= 1.0:
            raise ValueError("adaptive_memory_ewma_alpha must be in (0, 1]")
        if not 0.0 <= self.adaptive_memory_hysteresis_ratio <= 1.0:
            raise ValueError(
                "adaptive_memory_hysteresis_ratio must be in [0, 1]"
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
        valid_admissions = ("cache", "transient_on_pressure")
        for name in ("prefill_expert_admission", "decode_expert_admission"):
            value = getattr(self, name)
            if value not in valid_admissions:
                raise ValueError(
                    f"{name} must be one of {valid_admissions}, got {value!r}"
                )
        for name in (
            "prefill_expert_prefetch_top_k",
            "decode_expert_prefetch_top_k",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        for name in (
            "prefill_expert_prefetch_priority",
            "decode_expert_prefetch_priority",
        ):
            value = getattr(self, name)
            if not 1 <= value <= 19:
                raise ValueError(f"{name} must be in [1, 19], got {value}")
        for name in (
            "prefill_expert_eviction_weight",
            "decode_expert_eviction_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if self.expert_policy_starvation_limit <= 0:
            raise ValueError(
                f"expert_policy_starvation_limit must be > 0, got {self.expert_policy_starvation_limit}"
            )
        valid_policies = ("off", "observe", "enforce")
        if self.overlap_prefetch_policy not in valid_policies:
            raise ValueError(
                f"overlap_prefetch_policy must be one of {valid_policies}, got {self.overlap_prefetch_policy!r}"
            )
        if not 0.0 < self.overlap_prefetch_ewma_alpha <= 1.0:
            raise ValueError(
                f"overlap_prefetch_ewma_alpha must be in (0, 1], got {self.overlap_prefetch_ewma_alpha}"
            )
        if not 0.0 < self.overlap_prefetch_safety_factor <= 1.0:
            raise ValueError(
                f"overlap_prefetch_safety_factor must be in (0, 1], got {self.overlap_prefetch_safety_factor}"
            )
        if self.overlap_prefetch_cold_start_experts < 0:
            raise ValueError(
                f"overlap_prefetch_cold_start_experts must be >= 0, got {self.overlap_prefetch_cold_start_experts}"
            )
        if self.overlap_prefetch_max_window_bytes < 0:
            raise ValueError(
                f"overlap_prefetch_max_window_bytes must be >= 0, got {self.overlap_prefetch_max_window_bytes}"
            )
        if self.overlap_prefetch_max_inflight_bytes < 0:
            raise ValueError(
                f"overlap_prefetch_max_inflight_bytes must be >= 0, got {self.overlap_prefetch_max_inflight_bytes}"
            )
        if (
            self.overlap_prefetch_policy == "enforce"
            and self.overlap_prefetch_max_window_bytes
            > self.overlap_prefetch_max_inflight_bytes
        ):
            raise ValueError(
                f"overlap_prefetch_max_window_bytes ({self.overlap_prefetch_max_window_bytes}) must be <= "
                f"overlap_prefetch_max_inflight_bytes ({self.overlap_prefetch_max_inflight_bytes}) when policy is enforce"
            )
        if (
            self.gpu_only_expert_routing
            and self.overlap_prefetch_policy != "off"
        ):
            raise ValueError(
                "gpu_only_expert_routing cannot be combined with overlap "
                "prefetch (overlap_prefetch_policy=observe|enforce) in the "
                "first release"
            )
        from moe_infinity.runtime.kv_cache_format import KVCacheFormat

        KVCacheFormat.parse(self.kv_cache_format)
        if self.kv_swap_mode not in ("sync", "async"):
            raise ValueError(
                f"kv_swap_mode must be 'sync' or 'async', got {self.kv_swap_mode!r}"
            )
        if self.kv_swap_host_memory_bytes <= 0:
            raise ValueError(
                f"kv_swap_host_memory_bytes must be > 0, got {self.kv_swap_host_memory_bytes}"
            )
        if self.kv_swap_max_inflight_bytes <= 0:
            raise ValueError(
                f"kv_swap_max_inflight_bytes must be > 0, got {self.kv_swap_max_inflight_bytes}"
            )
        if self.kv_swap_max_inflight_bytes > self.kv_swap_host_memory_bytes:
            raise ValueError(
                f"kv_swap_max_inflight_bytes ({self.kv_swap_max_inflight_bytes}) must not exceed kv_swap_host_memory_bytes ({self.kv_swap_host_memory_bytes})"
            )
        if self.kv_swap_max_retries < 0:
            raise ValueError(
                f"kv_swap_max_retries must be >= 0, got {self.kv_swap_max_retries}"
            )
        if (
            type(self.max_resident_paged_speculative_sessions) is not int
            or self.max_resident_paged_speculative_sessions < 0
        ):
            raise ValueError(
                "max_resident_paged_speculative_sessions must be an integer >= 0"
            )
        if (
            type(self.min_free_mla_blocks_after_admission) is not int
            or self.min_free_mla_blocks_after_admission < 1
        ):
            raise ValueError(
                "min_free_mla_blocks_after_admission must be an integer >= 1"
            )
