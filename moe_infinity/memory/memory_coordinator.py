import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Union, cast

NumberLike = Union[float, int, str]


@dataclass
class MemoryCoordinator:
    device_memory_ratio: float = 0.75
    kv_cache_memory_ratio: float = 0.15

    def __post_init__(self) -> None:
        total = self.device_memory_ratio + self.kv_cache_memory_ratio
        if total > 1.0:
            raise ValueError(
                f"device_memory_ratio ({self.device_memory_ratio}) + kv_cache_memory_ratio ({self.kv_cache_memory_ratio}) = {total:.3f} > 1.0"
            )

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "MemoryCoordinator":
        device_ratio = float(
            cast(NumberLike, config.get("device_memory_ratio", 0.75))
        )
        kv_ratio = float(
            cast(NumberLike, config.get("kv_cache_memory_ratio", 0.0))
        )
        use_native = bool(cast(bool, config.get("use_native_engine", True)))

        if use_native and kv_ratio == 0.0:
            kv_ratio = 0.15
            warnings.warn(
                (
                    "kv_cache_memory_ratio was 0.0 with use_native_engine=True. "
                    "Auto-set to 0.15. Set explicitly to suppress this warning."
                ),
                UserWarning,
                stacklevel=2,
            )
            if device_ratio + kv_ratio > 1.0:
                device_ratio = max(0.0, 1.0 - kv_ratio)
                warnings.warn(
                    (
                        "device_memory_ratio adjusted to "
                        f"{device_ratio:.2f} to satisfy budget constraint."
                    ),
                    UserWarning,
                    stacklevel=2,
                )

        return cls(
            device_memory_ratio=device_ratio,
            kv_cache_memory_ratio=kv_ratio,
        )

    def num_gpu_devices(self) -> int:
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                return cuda.device_count()
        except Exception:
            pass
        return 0

    def total_gpu_memory_bytes(self, device_id: int = 0) -> int:
        """Return total GPU memory for a specific device.

        Args:
            device_id: CUDA device index (default 0).
        """
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is not None and cuda.is_available():
                count = cuda.device_count()
                if 0 <= device_id < count:
                    props = cuda.get_device_properties(device_id)
                    total = getattr(props, "total_memory", None)
                    if isinstance(total, int):
                        return total
        except Exception:
            pass
        return 24 * 1024**3

    def aggregate_gpu_memory_bytes(self) -> int:
        """Return the sum of GPU memory across all visible devices."""
        n = self.num_gpu_devices()
        if n == 0:
            return 24 * 1024**3
        return sum(self.total_gpu_memory_bytes(i) for i in range(n))

    def expert_cache_bytes(self, device_id: int = 0) -> int:
        return int(
            self.total_gpu_memory_bytes(device_id) * self.device_memory_ratio
        )

    def expert_cache_bytes_total(self) -> int:
        """Total expert cache budget across all GPUs."""
        n = max(1, self.num_gpu_devices())
        return sum(self.expert_cache_bytes(i) for i in range(n))

    def kv_cache_bytes(self, device_id: int = 0) -> int:
        return int(
            self.total_gpu_memory_bytes(device_id) * self.kv_cache_memory_ratio
        )

    def kv_cache_bytes_total(self) -> int:
        """Total KV cache budget across all GPUs."""
        n = max(1, self.num_gpu_devices())
        return sum(self.kv_cache_bytes(i) for i in range(n))

    def compute_safe_budget(
        self,
        *,
        model_bytes: int,
        activation_reserve_bytes: int,
        free_reserve_bytes: int,
        device_id: int = 0,
    ) -> int:
        return max(
            0,
            self.total_gpu_memory_bytes(device_id)
            - model_bytes
            - activation_reserve_bytes
            - free_reserve_bytes,
        )

    def validate_targets(
        self,
        *,
        device_id: int,
        expert_bytes: int,
        kv_blocks: int,
        kv_block_bytes: int,
        safe_budget_bytes: int,
    ) -> None:
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if min(expert_bytes, kv_blocks, kv_block_bytes, safe_budget_bytes) < 0:
            raise ValueError("memory targets must be non-negative")
        if expert_bytes + kv_blocks * kv_block_bytes > safe_budget_bytes:
            raise ValueError("expert and KV targets exceed safe GPU budget")

    def remaining_bytes(self, device_id: int = 0) -> int:
        total = self.total_gpu_memory_bytes(device_id)
        used = self.expert_cache_bytes(device_id) + self.kv_cache_bytes(
            device_id
        )
        return max(0, total - used)

    def can_allocate_kv_blocks(
        self, num_blocks: int, block_size_bytes: int
    ) -> bool:
        return num_blocks * block_size_bytes <= self.kv_cache_bytes()

    def can_cache_expert(self, expert_size_bytes: int) -> bool:
        return expert_size_bytes <= self.expert_cache_bytes()

    def compute_num_kv_blocks(self, block_size_bytes: int) -> int:
        if block_size_bytes <= 0:
            return 0
        return max(0, math.floor(self.kv_cache_bytes() / block_size_bytes))

    def get_budget_status(self) -> Dict[str, Union[int, float]]:
        total = self.total_gpu_memory_bytes()
        return {
            "total_gpu_bytes": total,
            "expert_cache_bytes": self.expert_cache_bytes(),
            "kv_cache_bytes": self.kv_cache_bytes(),
            "remaining_bytes": self.remaining_bytes(),
            "device_memory_ratio": self.device_memory_ratio,
            "kv_cache_memory_ratio": self.kv_cache_memory_ratio,
            "num_gpu_devices": self.num_gpu_devices(),
            "aggregate_gpu_bytes": self.aggregate_gpu_memory_bytes(),
        }
