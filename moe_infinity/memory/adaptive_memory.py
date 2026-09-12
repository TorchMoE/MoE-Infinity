"""Deterministic, bounded controller for expert/KV VRAM reallocation.

This module is pure policy: it computes proposed per-device targets from
measured expert-miss and KV-pressure signals and never imports torch, touches
CUDA, or mutates any storage. All state is keyed by ``device_id`` so no scalar
target or failure latch is ever shared across GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResizeDirection(str, Enum):
    HOLD = "hold"
    EXPERT_TO_KV = "expert_to_kv"
    KV_TO_EXPERT = "kv_to_expert"


class ResizeOutcome(str, Enum):
    COMMITTED = "committed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    PARTIAL_DONOR_COMMITTED = "partial_donor_committed"


@dataclass(frozen=True)
class AdaptiveMemoryConfig:
    enabled: bool = False
    interval_steps: int = 64
    cooldown_steps: int = 256
    ewma_alpha: float = 0.20
    hysteresis_ratio: float = 0.15
    max_resize_step_bytes: int = 256 * 1024**2
    min_expert_cache_bytes: int = 512 * 1024**2
    min_kv_cache_blocks: int = 128
    free_memory_reserve_bytes: int = 1024 * 1024**2
    failure_limit: int = 3


@dataclass(frozen=True)
class MemorySignals:
    device_id: int
    step: int
    expert_misses: int
    expert_accesses: int
    expert_fetch_stall_ms: float
    kv_used_blocks: int
    kv_total_blocks: int
    kv_swap_bytes: int
    kv_swap_stall_ms: float
    kv_preemptions: int
    free_gpu_bytes: int
    kv_supported: bool = True


@dataclass(frozen=True)
class MemoryTargets:
    device_id: int
    expert_bytes: int
    kv_blocks: int
    direction: ResizeDirection
    reason: str
    kv_supported: bool = True


@dataclass(frozen=True)
class ResizeResult:
    device_id: int
    outcome: ResizeOutcome
    expert_bytes: int
    kv_blocks: int
    reason: str
    kv_supported: bool = True

    @property
    def committed(self) -> bool:
        return self.outcome in {
            ResizeOutcome.COMMITTED,
            ResizeOutcome.PARTIAL_DONOR_COMMITTED,
        }


@dataclass
class _DeviceState:
    expert_cost_ewma: float = 0.0
    kv_pressure_cost_ewma: float = 0.0
    free_gpu_bytes: int = 0
    warm: bool = False
    last_committed_step: int | None = None
    last_expert_bytes: int | None = None
    last_kv_blocks: int | None = None
    consecutive_failures: int = 0
    resize_attempts: int = 0
    resize_failures: int = 0
    fallback_static: bool = False
    fallback_reason: str = ""
    last_reason: str = "init"
    last_expert_cost: float = 0.0
    last_kv_pressure_cost: float = 0.0


class AdaptiveMemoryController:
    def __init__(self, config: AdaptiveMemoryConfig) -> None:
        self.config = config
        self._state: dict[int, _DeviceState] = {}

    def _device(self, device_id: int) -> _DeviceState:
        state = self._state.get(device_id)
        if state is None:
            state = _DeviceState()
            self._state[device_id] = state
        return state

    def _bounded_step(self, donor_slack: int) -> int:
        return max(0, min(self.config.max_resize_step_bytes, donor_slack))

    def _hard_budget(self, total: int, model: int, activation: int) -> int:
        return max(
            0,
            total - model - activation - self.config.free_memory_reserve_bytes,
        )

    def observe(self, signals: MemorySignals) -> None:
        if signals.device_id < 0:
            raise ValueError("device_id must be non-negative")
        counters = (
            signals.expert_misses,
            signals.expert_accesses,
            signals.kv_used_blocks,
            signals.kv_total_blocks,
            signals.kv_swap_bytes,
            signals.kv_preemptions,
            signals.free_gpu_bytes,
        )
        if any(value < 0 for value in counters):
            raise ValueError("signal counters must be non-negative")
        if signals.expert_fetch_stall_ms < 0 or signals.kv_swap_stall_ms < 0:
            raise ValueError("stall milliseconds must be non-negative")
        if signals.kv_used_blocks > signals.kv_total_blocks:
            raise ValueError("kv_used_blocks cannot exceed kv_total_blocks")
        if signals.expert_misses > signals.expert_accesses:
            raise ValueError("expert_misses cannot exceed expert_accesses")

        expert_miss_rate = signals.expert_misses / max(
            1, signals.expert_accesses
        )
        expert_cost = expert_miss_rate * signals.expert_fetch_stall_ms
        kv_pressure = signals.kv_used_blocks / max(1, signals.kv_total_blocks)
        kv_cost = (
            kv_pressure * (signals.kv_swap_stall_ms + signals.kv_preemptions)
            + signals.kv_swap_stall_ms
        )

        state = self._device(signals.device_id)
        alpha = self.config.ewma_alpha
        if not state.warm:
            state.expert_cost_ewma = expert_cost
            state.kv_pressure_cost_ewma = kv_cost
            state.warm = True
        else:
            state.expert_cost_ewma = (
                alpha * expert_cost + (1.0 - alpha) * state.expert_cost_ewma
            )
            state.kv_pressure_cost_ewma = (
                alpha * kv_cost + (1.0 - alpha) * state.kv_pressure_cost_ewma
            )
        state.free_gpu_bytes = signals.free_gpu_bytes
        state.last_expert_cost = expert_cost
        state.last_kv_pressure_cost = kv_cost

    def propose(
        self,
        *,
        device_id: int,
        step: int,
        total_bytes: int,
        model_bytes: int,
        activation_reserve_bytes: int,
        kv_block_bytes: int,
        current_expert_bytes: int,
        current_kv_blocks: int,
        kv_supported: bool = True,
    ) -> MemoryTargets:
        if not kv_supported:
            return MemoryTargets(
                device_id,
                current_expert_bytes,
                0,
                ResizeDirection.HOLD,
                "kv_backend_unavailable",
                False,
            )

        state = self._device(device_id)

        def hold(reason: str) -> MemoryTargets:
            return MemoryTargets(
                device_id,
                current_expert_bytes,
                current_kv_blocks,
                ResizeDirection.HOLD,
                reason,
                True,
            )

        if not self.config.enabled:
            return hold("disabled")
        if state.fallback_static:
            return hold("fallback_static")
        if (
            self.config.interval_steps > 0
            and step % self.config.interval_steps != 0
        ):
            return hold("not_interval")
        if state.last_committed_step is not None:
            if step - state.last_committed_step < self.config.cooldown_steps:
                return hold("cooldown")
        if state.free_gpu_bytes < self.config.free_memory_reserve_bytes:
            return hold("below_reserve")

        expert_cost = state.expert_cost_ewma
        kv_cost = state.kv_pressure_cost_ewma
        margin = self.config.hysteresis_ratio * max(expert_cost, kv_cost, 1e-9)

        if abs(expert_cost - kv_cost) <= margin:
            return hold("within_hysteresis")

        hard_budget = self._hard_budget(
            total_bytes, model_bytes, activation_reserve_bytes
        )
        min_kv_bytes = self.config.min_kv_cache_blocks * kv_block_bytes

        if expert_cost > kv_cost:
            direction = ResizeDirection.KV_TO_EXPERT
            reason = "expert_miss_cost"
            donor_floor = min(min_kv_bytes, current_kv_blocks * kv_block_bytes)
            donor_slack = current_kv_blocks * kv_block_bytes - donor_floor
        else:
            direction = ResizeDirection.EXPERT_TO_KV
            reason = "kv_pressure"
            donor_floor = min(
                self.config.min_expert_cache_bytes, current_expert_bytes
            )
            donor_slack = current_expert_bytes - donor_floor

        step_bytes = self._bounded_step(donor_slack)
        delta_blocks = step_bytes // kv_block_bytes if kv_block_bytes > 0 else 0
        moved_bytes = delta_blocks * kv_block_bytes

        if direction is ResizeDirection.KV_TO_EXPERT:
            new_kv_blocks = current_kv_blocks - delta_blocks
            new_expert_bytes = current_expert_bytes + moved_bytes
        else:
            new_kv_blocks = current_kv_blocks + delta_blocks
            new_expert_bytes = current_expert_bytes - moved_bytes

        # A reallocation is footprint-invariant (bytes only move between the
        # two pools), so this clamp reverts to the current split only when the
        # device is already over its hard budget. The reported direction is
        # deliberately preserved even for a zero-byte (donor-exhausted or
        # budget-reverted) move: callers diff the returned targets against the
        # current split to decide whether any bytes actually move.
        if new_expert_bytes + new_kv_blocks * kv_block_bytes > hard_budget:
            new_kv_blocks = current_kv_blocks
            new_expert_bytes = current_expert_bytes

        return MemoryTargets(
            device_id,
            new_expert_bytes,
            new_kv_blocks,
            direction,
            reason,
            True,
        )

    def record_resize(self, result: ResizeResult, *, step: int) -> None:
        state = self._device(result.device_id)
        state.resize_attempts += 1
        state.last_reason = result.reason
        if result.outcome is ResizeOutcome.COMMITTED:
            state.consecutive_failures = 0
            state.last_committed_step = step
            state.last_expert_bytes = result.expert_bytes
            state.last_kv_blocks = result.kv_blocks
            return
        if result.outcome is ResizeOutcome.PARTIAL_DONOR_COMMITTED:
            state.last_committed_step = step
            state.last_expert_bytes = result.expert_bytes
            state.last_kv_blocks = result.kv_blocks
        state.resize_failures += 1
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.config.failure_limit:
            state.fallback_static = True
            state.fallback_reason = result.reason

    def disable_to_static(self, device_id: int, reason: str) -> None:
        state = self._device(device_id)
        state.fallback_static = True
        state.fallback_reason = reason
        state.last_reason = reason

    def report(self) -> dict[int, dict[str, int | float | str | bool]]:
        report: dict[int, dict[str, int | float | str | bool]] = {}
        for device_id, state in self._state.items():
            report[device_id] = {
                "enabled": bool(self.config.enabled),
                "fallback_static": bool(state.fallback_static),
                "fallback_reason": state.fallback_reason,
                "resize_attempts": int(state.resize_attempts),
                "resize_failures": int(state.resize_failures),
                "consecutive_failures": int(state.consecutive_failures),
                "last_reason": state.last_reason,
                "expert_target_bytes": int(state.last_expert_bytes or 0),
                "kv_target_blocks": int(state.last_kv_blocks or 0),
                "expert_miss_cost": float(state.expert_cost_ewma),
                "kv_pressure_cost": float(state.kv_pressure_cost_ewma),
            }
        return report
