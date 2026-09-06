from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class MemoryBudget:
    total_gpu_memory_bytes: int
    model_memory_bytes: int
    expert_cache_ratio: float
    kv_cache_ratio: float
    activation_reserve_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.total_gpu_memory_bytes < 0:
            raise ValueError(
                f"total_gpu_memory_bytes must be >= 0, got {self.total_gpu_memory_bytes}"
            )
        if self.model_memory_bytes < 0:
            raise ValueError(
                f"model_memory_bytes must be >= 0, got {self.model_memory_bytes}"
            )
        for name, value in (
            ("expert_cache_ratio", self.expert_cache_ratio),
            ("kv_cache_ratio", self.kv_cache_ratio),
            ("activation_reserve_ratio", self.activation_reserve_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def available_bytes(self) -> int:
        activation_reserve_bytes = int(
            self.total_gpu_memory_bytes * self.activation_reserve_ratio
        )
        available = (
            self.total_gpu_memory_bytes
            - self.model_memory_bytes
            - activation_reserve_bytes
        )
        return max(0, available)

    @property
    def expert_cache_bytes(self) -> int:
        requested = int(self.available_bytes * self.expert_cache_ratio)
        return min(self.available_bytes, max(0, requested))

    @property
    def kv_cache_bytes(self) -> int:
        requested = int(self.available_bytes * self.kv_cache_ratio)
        remaining_after_expert = max(
            0, self.available_bytes - self.expert_cache_bytes
        )
        return min(remaining_after_expert, max(0, requested))


class MemoryManager:
    device: torch.device
    device_memory_ratio: float
    kv_cache_ratio: float
    activation_reserve_ratio: float
    total_gpu_memory_bytes: int
    _last_budget: Optional[MemoryBudget]

    def __init__(
        self,
        device: Optional[torch.device] = None,
        device_memory_ratio: float = 0.75,
        kv_cache_ratio: float = 0.25,
        activation_reserve_ratio: float = 0.10,
    ) -> None:
        if not 0.0 <= device_memory_ratio <= 1.0:
            raise ValueError(
                f"device_memory_ratio must be in [0, 1], got {device_memory_ratio}"
            )
        if not 0.0 <= kv_cache_ratio <= 1.0:
            raise ValueError(
                f"kv_cache_ratio must be in [0, 1], got {kv_cache_ratio}"
            )
        if not 0.0 <= activation_reserve_ratio <= 1.0:
            raise ValueError(
                f"activation_reserve_ratio must be in [0, 1], got {activation_reserve_ratio}"
            )

        self.device = self._resolve_device(device)
        self.device_memory_ratio = device_memory_ratio
        self.kv_cache_ratio = kv_cache_ratio
        self.activation_reserve_ratio = activation_reserve_ratio
        self.total_gpu_memory_bytes = self._get_total_gpu_memory_bytes(
            self.device
        )
        self._last_budget = None
        self._cuda_graph_pool_bytes = 0
        self._cuda_graph_scratch_kv_bytes = 0

    def compute_budget(self, model_memory_bytes: int) -> MemoryBudget:
        if model_memory_bytes < 0:
            raise ValueError(
                f"model_memory_bytes must be >= 0, got {model_memory_bytes}"
            )

        budget = MemoryBudget(
            total_gpu_memory_bytes=self.total_gpu_memory_bytes,
            model_memory_bytes=model_memory_bytes,
            expert_cache_ratio=self.get_expert_cache_ratio(),
            kv_cache_ratio=self.device_memory_ratio * self.kv_cache_ratio,
            activation_reserve_ratio=self.activation_reserve_ratio,
        )
        self._last_budget = budget
        return budget

    def get_max_kv_blocks(
        self,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> int:
        for name, value in (
            ("block_size", block_size),
            ("num_layers", num_layers),
            ("num_heads", num_heads),
            ("head_dim", head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        kv_bytes_per_block = (
            2
            * block_size
            * num_layers
            * num_heads
            * head_dim
            * bytes_per_element
        )

        budget = self._last_budget
        if budget is None:
            budget = self.compute_budget(model_memory_bytes=0)

        if kv_bytes_per_block <= 0:
            return 0
        return max(0, budget.kv_cache_bytes // kv_bytes_per_block)

    def get_expert_cache_ratio(self) -> float:
        return self.device_memory_ratio - (
            self.device_memory_ratio * self.kv_cache_ratio
        )

    def set_cuda_graph_usage(
        self, *, graph_pool_bytes: int, scratch_kv_bytes: int
    ) -> None:
        if graph_pool_bytes < 0 or scratch_kv_bytes < 0:
            raise ValueError("CUDA graph memory usage must be non-negative")
        self._cuda_graph_pool_bytes = graph_pool_bytes
        self._cuda_graph_scratch_kv_bytes = scratch_kv_bytes

    def report(self) -> dict[str, Union[str, int, float]]:
        budget = self._last_budget
        if budget is None:
            budget = self.compute_budget(model_memory_bytes=0)

        return {
            "device": str(self.device),
            "total_gpu_memory_bytes": budget.total_gpu_memory_bytes,
            "model_memory_bytes": budget.model_memory_bytes,
            "available_bytes": budget.available_bytes,
            "expert_cache_ratio": budget.expert_cache_ratio,
            "kv_cache_ratio": budget.kv_cache_ratio,
            "activation_reserve_ratio": budget.activation_reserve_ratio,
            "expert_cache_bytes": budget.expert_cache_bytes,
            "kv_cache_bytes": budget.kv_cache_bytes,
            "cuda_graph_pool_bytes": self._cuda_graph_pool_bytes,
            "cuda_graph_scratch_kv_bytes": self._cuda_graph_scratch_kv_bytes,
            "cuda_graph_total_bytes": (
                self._cuda_graph_pool_bytes + self._cuda_graph_scratch_kv_bytes
            ),
        }

    @staticmethod
    def _resolve_device(device: Optional[torch.device]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    @staticmethod
    def _get_total_gpu_memory_bytes(device: torch.device) -> int:
        if device.type != "cuda" or not torch.cuda.is_available():
            return 0

        try:
            _, total = torch.cuda.mem_get_info(device)
            return int(total)
        except Exception:
            return 0


class AdaptiveKVScheduler:
    initial_max_batch_size: int
    kv_high_watermark: float
    kv_low_watermark: float
    expert_hit_target: float
    cooldown_seconds: float
    _current_max_batch_size: int
    _adjust_step: int
    _last_adjust_timestamp: Optional[float]
    _kv_utilization: float
    _expert_hit_rate: Optional[float]
    _kv_used_blocks: int
    _kv_total_blocks: int

    def __init__(
        self,
        initial_max_batch_size: int = 32,
        kv_high_watermark: float = 0.90,
        kv_low_watermark: float = 0.50,
        expert_hit_target: float = 0.70,
        cooldown_seconds: float = 10.0,
    ) -> None:
        if initial_max_batch_size <= 0:
            raise ValueError(
                f"initial_max_batch_size must be > 0, got {initial_max_batch_size}"
            )
        for name, value in (
            ("kv_high_watermark", kv_high_watermark),
            ("kv_low_watermark", kv_low_watermark),
            ("expert_hit_target", expert_hit_target),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if kv_low_watermark >= kv_high_watermark:
            raise ValueError(
                "kv_low_watermark must be < kv_high_watermark, got "
                + f"{kv_low_watermark} >= {kv_high_watermark}"
            )
        if cooldown_seconds < 0.0:
            raise ValueError(
                f"cooldown_seconds must be >= 0, got {cooldown_seconds}"
            )

        self.initial_max_batch_size = initial_max_batch_size
        self.kv_high_watermark = kv_high_watermark
        self.kv_low_watermark = kv_low_watermark
        self.expert_hit_target = expert_hit_target
        self.cooldown_seconds = cooldown_seconds

        self._current_max_batch_size = initial_max_batch_size
        self._adjust_step = max(1, initial_max_batch_size // 8)
        self._last_adjust_timestamp = None
        self._kv_utilization = 0.0
        self._expert_hit_rate = None
        self._kv_used_blocks = 0
        self._kv_total_blocks = 0

    def observe(
        self,
        kv_used_blocks: int,
        kv_total_blocks: int,
        expert_hit_rate: Optional[float] = None,
    ) -> None:
        if kv_used_blocks < 0:
            raise ValueError(
                f"kv_used_blocks must be >= 0, got {kv_used_blocks}"
            )
        if kv_total_blocks < 0:
            raise ValueError(
                f"kv_total_blocks must be >= 0, got {kv_total_blocks}"
            )
        if kv_total_blocks > 0 and kv_used_blocks > kv_total_blocks:
            raise ValueError(
                "kv_used_blocks must be <= kv_total_blocks when total > 0, "
                + f"got {kv_used_blocks} > {kv_total_blocks}"
            )
        if expert_hit_rate is not None and not 0.0 <= expert_hit_rate <= 1.0:
            raise ValueError(
                f"expert_hit_rate must be in [0, 1], got {expert_hit_rate}"
            )

        self._kv_used_blocks = kv_used_blocks
        self._kv_total_blocks = kv_total_blocks
        self._kv_utilization = (
            0.0 if kv_total_blocks == 0 else kv_used_blocks / kv_total_blocks
        )
        if expert_hit_rate is not None:
            self._expert_hit_rate = expert_hit_rate

        self._maybe_adjust()

    def get_max_batch_size(self) -> int:
        return self._current_max_batch_size

    def should_preempt_aggressively(self) -> bool:
        return self._kv_utilization >= self.kv_high_watermark

    def get_tuning_recommendations(self) -> dict[str, Union[str, float, int]]:
        if self._kv_utilization >= self.kv_high_watermark:
            return {
                "recommendation": "increase kv_cache_ratio",
                "reason": "kv_cache_near_capacity",
                "current_kv_utilization": self._kv_utilization,
                "suggested_kv_cache_ratio_delta": 0.10,
            }

        if (
            self._expert_hit_rate is not None
            and self._expert_hit_rate < self.expert_hit_target
        ):
            return {
                "recommendation": "increase expert_cache_ratio",
                "reason": "expert_cache_hit_rate_below_target",
                "current_expert_hit_rate": self._expert_hit_rate,
                "target_expert_hit_rate": self.expert_hit_target,
            }

        if (
            self._kv_utilization <= self.kv_low_watermark
            and self._current_max_batch_size < self.initial_max_batch_size
        ):
            return {
                "recommendation": "increase scheduler_batch_size",
                "reason": "kv_cache_underutilized",
                "current_max_batch_size": self._current_max_batch_size,
                "suggested_max_batch_size": min(
                    self.initial_max_batch_size,
                    self._current_max_batch_size + self._adjust_step,
                ),
            }

        return {
            "recommendation": "keep_current_policy",
            "current_kv_utilization": self._kv_utilization,
            "current_max_batch_size": self._current_max_batch_size,
        }

    def _maybe_adjust(self) -> None:
        now = time.monotonic()
        if self._last_adjust_timestamp is not None:
            elapsed = now - self._last_adjust_timestamp
            if elapsed < self.cooldown_seconds:
                return

        next_batch_size = self._current_max_batch_size
        kv_pressure_high = self._kv_utilization >= self.kv_high_watermark
        expert_pressure_high = (
            self._expert_hit_rate is not None
            and self._expert_hit_rate < self.expert_hit_target
        )
        kv_pressure_low = self._kv_utilization <= self.kv_low_watermark

        if kv_pressure_high and expert_pressure_high:
            next_batch_size -= self._adjust_step * 2
        elif kv_pressure_high or expert_pressure_high:
            next_batch_size -= self._adjust_step
        elif kv_pressure_low and (
            self._expert_hit_rate is None
            or self._expert_hit_rate >= self.expert_hit_target
        ):
            next_batch_size += self._adjust_step

        next_batch_size = max(
            1,
            min(self.initial_max_batch_size, next_batch_size),
        )

        if next_batch_size != self._current_max_batch_size:
            self._current_max_batch_size = next_batch_size
            self._last_adjust_timestamp = now


GPUMemoryBudgetManager = MemoryManager

__all__ = [
    "MemoryBudget",
    "MemoryManager",
    "AdaptiveKVScheduler",
    "GPUMemoryBudgetManager",
]
