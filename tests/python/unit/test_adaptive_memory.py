"""Deterministic CPU-only tests for the bounded adaptive-memory policy.

These tests never import torch or allocate CUDA memory; the controller is a
pure-Python deterministic policy keyed per device.
"""

from __future__ import annotations

import pytest

from moe_infinity.memory.adaptive_memory import (
    AdaptiveMemoryConfig,
    AdaptiveMemoryController,
    MemorySignals,
    MemoryTargets,
    ResizeDirection,
    ResizeOutcome,
    ResizeResult,
)

MIB = 1024**2


def signal(
    step: int,
    *,
    misses: int,
    fetch_ms: float,
    used: int,
    swaps: int = 0,
    swap_ms: float = 0.0,
    free: int = 4096 * MIB,
    device_id: int = 0,
):
    return MemorySignals(
        device_id,
        step,
        misses,
        100,
        fetch_ms,
        used,
        100,
        swaps * MIB,
        swap_ms,
        swaps,
        free,
    )


def test_cost_moves_one_bounded_step_toward_kv() -> None:
    ctl = AdaptiveMemoryController(
        AdaptiveMemoryConfig(
            enabled=True,
            interval_steps=1,
            cooldown_steps=0,
            max_resize_step_bytes=64 * MIB,
            min_expert_cache_bytes=128 * MIB,
            min_kv_cache_blocks=4,
            free_memory_reserve_bytes=128 * MIB,
        )
    )
    ctl.observe(
        signal(1, misses=0, fetch_ms=0.0, used=99, swaps=8, swap_ms=40.0)
    )
    target = ctl.propose(
        device_id=0,
        step=1,
        total_bytes=2048 * MIB,
        model_bytes=512 * MIB,
        activation_reserve_bytes=128 * MIB,
        kv_block_bytes=16 * MIB,
        current_expert_bytes=704 * MIB,
        current_kv_blocks=36,
    )
    assert target.direction is ResizeDirection.EXPERT_TO_KV
    assert target.expert_bytes == 640 * MIB
    assert target.kv_blocks == 40
    # One 64 MiB bounded step: 640 MiB + 40 * 16 MiB = 1280 MiB.
    assert target.expert_bytes + target.kv_blocks * 16 * MIB == 1280 * MIB


def test_hysteresis_and_cooldown_prevent_oscillation() -> None:
    ctl = AdaptiveMemoryController(
        AdaptiveMemoryConfig(
            enabled=True,
            interval_steps=1,
            cooldown_steps=8,
            hysteresis_ratio=0.25,
            max_resize_step_bytes=64 * MIB,
            min_expert_cache_bytes=128 * MIB,
            min_kv_cache_blocks=4,
            free_memory_reserve_bytes=128 * MIB,
        )
    )
    ctl.observe(signal(8, misses=80, fetch_ms=50.0, used=40))
    first = ctl.propose(
        device_id=0,
        step=8,
        total_bytes=2048 * MIB,
        model_bytes=512 * MIB,
        activation_reserve_bytes=128 * MIB,
        kv_block_bytes=16 * MIB,
        current_expert_bytes=640 * MIB,
        current_kv_blocks=40,
    )
    ctl.record_resize(
        ResizeResult(
            0,
            ResizeOutcome.COMMITTED,
            first.expert_bytes,
            first.kv_blocks,
            "committed",
        ),
        step=8,
    )
    ctl.observe(
        signal(9, misses=0, fetch_ms=0.0, used=99, swaps=8, swap_ms=50.0)
    )
    assert (
        ctl.propose(
            device_id=0,
            step=9,
            total_bytes=2048 * MIB,
            model_bytes=512 * MIB,
            activation_reserve_bytes=128 * MIB,
            kv_block_bytes=16 * MIB,
            current_expert_bytes=first.expert_bytes,
            current_kv_blocks=first.kv_blocks,
        ).direction
        is ResizeDirection.HOLD
    )


def test_three_failures_latch_static_fallback() -> None:
    ctl = AdaptiveMemoryController(
        AdaptiveMemoryConfig(
            enabled=True, interval_steps=1, cooldown_steps=0, failure_limit=3
        )
    )
    for step in range(3):
        ctl.record_resize(
            ResizeResult(1, ResizeOutcome.REJECTED, 10, 10, "pinned"), step=step
        )
    assert ctl.report()[1]["fallback_static"] is True
    assert ctl.report()[1]["fallback_reason"] == "pinned"


def test_device_state_is_independent() -> None:
    ctl = AdaptiveMemoryController(
        AdaptiveMemoryConfig(
            enabled=True, interval_steps=1, cooldown_steps=0, failure_limit=1
        )
    )
    ctl.observe(signal(1, misses=90, fetch_ms=40.0, used=10, device_id=0))
    ctl.observe(
        signal(
            1,
            misses=0,
            fetch_ms=0.0,
            used=99,
            swaps=4,
            swap_ms=30.0,
            device_id=1,
        )
    )
    zero = ctl.propose(
        device_id=0,
        step=1,
        total_bytes=2048 * MIB,
        model_bytes=512 * MIB,
        activation_reserve_bytes=128 * MIB,
        kv_block_bytes=16 * MIB,
        current_expert_bytes=640 * MIB,
        current_kv_blocks=40,
    )
    one = ctl.propose(
        device_id=1,
        step=1,
        total_bytes=1536 * MIB,
        model_bytes=384 * MIB,
        activation_reserve_bytes=128 * MIB,
        kv_block_bytes=16 * MIB,
        current_expert_bytes=512 * MIB,
        current_kv_blocks=24,
    )
    assert (
        zero.device_id == 0 and zero.direction is ResizeDirection.KV_TO_EXPERT
    )
    assert one.device_id == 1 and one.direction is ResizeDirection.EXPERT_TO_KV
    ctl.record_resize(
        ResizeResult(0, ResizeOutcome.REJECTED, 640 * MIB, 40, "pinned"), step=1
    )
    assert ctl.report()[0]["fallback_static"] is True
    assert ctl.report()[1]["fallback_static"] is False


def test_device_without_kv_backend_holds_static_expert_target() -> None:
    ctl = AdaptiveMemoryController(
        AdaptiveMemoryConfig(enabled=True, interval_steps=1, cooldown_steps=0)
    )
    target = ctl.propose(
        device_id=1,
        step=1,
        total_bytes=2048 * MIB,
        model_bytes=512 * MIB,
        activation_reserve_bytes=128 * MIB,
        kv_block_bytes=16 * MIB,
        current_expert_bytes=640 * MIB,
        current_kv_blocks=0,
        kv_supported=False,
    )
    assert target == MemoryTargets(
        1, 640 * MIB, 0, ResizeDirection.HOLD, "kv_backend_unavailable", False
    )


# ---------------------------------------------------------------------------
# Step 4: complete CPU policy matrix
# ---------------------------------------------------------------------------


def _cfg(**kw) -> AdaptiveMemoryConfig:
    base = dict(
        enabled=True,
        interval_steps=1,
        cooldown_steps=0,
        max_resize_step_bytes=64 * MIB,
        min_expert_cache_bytes=128 * MIB,
        min_kv_cache_blocks=4,
        free_memory_reserve_bytes=128 * MIB,
    )
    base.update(kw)
    return AdaptiveMemoryConfig(**base)


def _propose(ctl, **kw):
    base = dict(
        device_id=0,
        step=1,
        total_bytes=2048 * MIB,
        model_bytes=512 * MIB,
        activation_reserve_bytes=128 * MIB,
        kv_block_bytes=16 * MIB,
        current_expert_bytes=640 * MIB,
        current_kv_blocks=40,
    )
    base.update(kw)
    return ctl.propose(**base)


def test_zero_signals_hold() -> None:
    ctl = AdaptiveMemoryController(_cfg())
    ctl.observe(signal(1, misses=0, fetch_ms=0.0, used=0))
    assert _propose(ctl).direction is ResizeDirection.HOLD


def test_disabled_config_always_holds() -> None:
    ctl = AdaptiveMemoryController(_cfg(enabled=False))
    ctl.observe(signal(1, misses=80, fetch_ms=50.0, used=10))
    assert _propose(ctl).direction is ResizeDirection.HOLD


@pytest.mark.parametrize(
    "bad",
    [
        dict(device_id=-1),
        dict(expert_misses=-1),
        dict(expert_accesses=-1),
        dict(kv_used_blocks=101),  # used > total
        dict(kv_total_blocks=-1),
        dict(free_gpu_bytes=-1),
        dict(expert_misses=101),  # misses > accesses
    ],
)
def test_observe_rejects_malformed_signals(bad) -> None:
    ctl = AdaptiveMemoryController(_cfg())
    kwargs = dict(
        device_id=0,
        step=1,
        expert_misses=1,
        expert_accesses=100,
        expert_fetch_stall_ms=1.0,
        kv_used_blocks=10,
        kv_total_blocks=100,
        kv_swap_bytes=0,
        kv_swap_stall_ms=0.0,
        kv_preemptions=0,
        free_gpu_bytes=4096 * MIB,
    )
    kwargs.update(bad)
    with pytest.raises(ValueError):
        ctl.observe(MemorySignals(**kwargs))


def test_min_expert_bytes_clamped() -> None:
    # Expert donor already at minimum: pressure direction is reported but the
    # magnitude clamps to zero, so no bytes actually move.
    ctl = AdaptiveMemoryController(_cfg(min_expert_cache_bytes=640 * MIB))
    ctl.observe(
        signal(1, misses=0, fetch_ms=0.0, used=99, swaps=8, swap_ms=40.0)
    )
    target = _propose(ctl)
    assert target.direction is ResizeDirection.EXPERT_TO_KV
    assert target.expert_bytes == 640 * MIB
    assert target.kv_blocks == 40


def test_min_kv_blocks_clamped() -> None:
    # KV donor already at minimum: pressure direction reported, zero movement.
    ctl = AdaptiveMemoryController(_cfg(min_kv_cache_blocks=40))
    ctl.observe(signal(1, misses=90, fetch_ms=40.0, used=10))
    target = _propose(ctl, current_kv_blocks=40)
    assert target.direction is ResizeDirection.KV_TO_EXPERT
    assert target.expert_bytes == 640 * MIB
    assert target.kv_blocks == 40


def test_model_bytes_exceeding_total_clamps_to_zero_move() -> None:
    # Hard budget is zero: the receiver cannot grow, so targets stay put.
    ctl = AdaptiveMemoryController(_cfg())
    ctl.observe(signal(1, misses=90, fetch_ms=40.0, used=10))
    target = _propose(ctl, model_bytes=4096 * MIB)
    assert target.expert_bytes == 640 * MIB
    assert target.kv_blocks == 40


def test_reserve_breach_holds() -> None:
    # Free GPU memory below the configured reserve gates any move: the
    # controller must HOLD even when cost signals would otherwise resize.
    ctl = AdaptiveMemoryController(_cfg())
    ctl.observe(signal(1, misses=90, fetch_ms=40.0, used=10, free=64 * MIB))
    target = _propose(ctl)
    assert target.direction is ResizeDirection.HOLD
    assert target.reason == "below_reserve"


def test_non_divisible_block_size_holds_when_block_exceeds_step() -> None:
    # kv_block_bytes (128 MiB) > max step (64 MiB): no whole block can move.
    ctl = AdaptiveMemoryController(_cfg())
    ctl.observe(
        signal(1, misses=0, fetch_ms=0.0, used=99, swaps=8, swap_ms=40.0)
    )
    target = _propose(ctl, kv_block_bytes=128 * MIB)
    assert target.expert_bytes == 640 * MIB
    assert target.kv_blocks == 40


def test_deterministic_repeated_traces() -> None:
    def run():
        ctl = AdaptiveMemoryController(_cfg())
        outs = []
        for step in range(1, 6):
            ctl.observe(
                signal(
                    step, misses=0, fetch_ms=0.0, used=99, swaps=8, swap_ms=40.0
                )
            )
            outs.append(_propose(ctl, step=step))
        return outs

    assert run() == run()


def test_opposite_expert_heavy_pressure_moves_to_expert() -> None:
    ctl = AdaptiveMemoryController(_cfg())
    ctl.observe(signal(1, misses=90, fetch_ms=40.0, used=10))
    target = _propose(ctl)
    assert target.direction is ResizeDirection.KV_TO_EXPERT
    assert target.expert_bytes == 704 * MIB
    assert target.kv_blocks == 36


def test_two_gpus_unequal_capacity_independent() -> None:
    ctl = AdaptiveMemoryController(_cfg(failure_limit=1))
    ctl.observe(signal(1, misses=90, fetch_ms=40.0, used=10, device_id=0))
    ctl.observe(
        signal(
            1,
            misses=0,
            fetch_ms=0.0,
            used=99,
            swaps=8,
            swap_ms=40.0,
            device_id=1,
        )
    )
    a = _propose(ctl, device_id=0, total_bytes=2048 * MIB)
    b = _propose(
        ctl,
        device_id=1,
        total_bytes=1024 * MIB,
        current_expert_bytes=512 * MIB,
        current_kv_blocks=20,
    )
    assert a.direction is ResizeDirection.KV_TO_EXPERT
    assert b.direction is ResizeDirection.EXPERT_TO_KV


def test_failure_latch_on_one_gpu_does_not_stop_other() -> None:
    ctl = AdaptiveMemoryController(_cfg(failure_limit=1))
    ctl.record_resize(
        ResizeResult(0, ResizeOutcome.REJECTED, 10, 10, "pinned"), step=1
    )
    assert ctl.report()[0]["fallback_static"] is True
    # Device 1 still proposes.
    ctl.observe(signal(2, misses=90, fetch_ms=40.0, used=10, device_id=1))
    b = _propose(ctl, device_id=1, step=2)
    assert b.direction is ResizeDirection.KV_TO_EXPERT


def test_partial_donor_committed_records_effective_and_failure() -> None:
    ctl = AdaptiveMemoryController(_cfg(failure_limit=3))
    ctl.record_resize(
        ResizeResult(
            0,
            ResizeOutcome.PARTIAL_DONOR_COMMITTED,
            512 * MIB,
            40,
            "kv_growth_oom",
        ),
        step=1,
    )
    report = ctl.report()[0]
    assert report["resize_failures"] == 1
    assert report["fallback_static"] is False


def test_committed_resets_failures() -> None:
    ctl = AdaptiveMemoryController(_cfg(failure_limit=3))
    ctl.record_resize(
        ResizeResult(0, ResizeOutcome.REJECTED, 10, 10, "x"), step=1
    )
    ctl.record_resize(
        ResizeResult(0, ResizeOutcome.REJECTED, 10, 10, "x"), step=2
    )
    ctl.record_resize(
        ResizeResult(0, ResizeOutcome.COMMITTED, 640 * MIB, 40, "committed"),
        step=3,
    )
    assert ctl.report()[0]["consecutive_failures"] == 0
    assert ctl.report()[0]["fallback_static"] is False


def test_resize_result_committed_property() -> None:
    assert ResizeResult(0, ResizeOutcome.COMMITTED, 1, 1, "x").committed is True
    assert (
        ResizeResult(
            0, ResizeOutcome.PARTIAL_DONOR_COMMITTED, 1, 1, "x"
        ).committed
        is True
    )
    assert ResizeResult(0, ResizeOutcome.REJECTED, 1, 1, "x").committed is False
    assert (
        ResizeResult(0, ResizeOutcome.ROLLED_BACK, 1, 1, "x").committed is False
    )
