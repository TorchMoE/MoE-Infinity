import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _memory_coordinator_class() -> Any:
    module = importlib.import_module("moe_infinity.memory.memory_coordinator")
    return getattr(module, "MemoryCoordinator")


def test_valid_construction():
    memory_coordinator = _memory_coordinator_class()
    mc = memory_coordinator(device_memory_ratio=0.7, kv_cache_memory_ratio=0.15)
    assert mc.device_memory_ratio == 0.7
    assert mc.kv_cache_memory_ratio == 0.15


def test_constraint_violation_raises():
    memory_coordinator = _memory_coordinator_class()
    with pytest.raises(ValueError):
        _ = memory_coordinator(
            device_memory_ratio=0.8, kv_cache_memory_ratio=0.3
        )


def test_boundary_sum_equal_one_works():
    memory_coordinator = _memory_coordinator_class()
    mc = memory_coordinator(
        device_memory_ratio=0.85, kv_cache_memory_ratio=0.15
    )
    assert mc.device_memory_ratio + mc.kv_cache_memory_ratio == pytest.approx(
        1.0
    )


def test_from_config_autocorrects_kv_ratio_with_warning():
    memory_coordinator = _memory_coordinator_class()
    config = {
        "device_memory_ratio": 0.75,
        "kv_cache_memory_ratio": 0.0,
        "use_native_engine": True,
    }
    with pytest.warns(UserWarning, match="Auto-set to 0.15"):
        mc = memory_coordinator.from_config(config)
    assert mc.kv_cache_memory_ratio == pytest.approx(0.15)


def test_can_allocate_kv_blocks_large_count(monkeypatch: pytest.MonkeyPatch):
    memory_coordinator = _memory_coordinator_class()
    total_gpu_bytes = 24 * 1024**3
    mc = memory_coordinator(
        device_memory_ratio=0.75, kv_cache_memory_ratio=0.15
    )
    monkeypatch.setattr(
        mc, "total_gpu_memory_bytes", lambda device_id=0: total_gpu_bytes
    )
    block_size_bytes = 64 * 1024
    num_fit = mc.compute_num_kv_blocks(block_size_bytes)
    assert num_fit >= 50000
    assert mc.can_allocate_kv_blocks(num_fit, block_size_bytes)


def test_compute_num_kv_blocks_deterministic(monkeypatch: pytest.MonkeyPatch):
    memory_coordinator = _memory_coordinator_class()
    total_gpu_bytes = 24 * 1024**3
    kv_ratio = 0.15
    block_size_bytes = 128 * 1024
    mc = memory_coordinator(
        device_memory_ratio=0.75, kv_cache_memory_ratio=kv_ratio
    )
    monkeypatch.setattr(
        mc, "total_gpu_memory_bytes", lambda device_id=0: total_gpu_bytes
    )
    expected = int((total_gpu_bytes * kv_ratio) // block_size_bytes)
    assert mc.compute_num_kv_blocks(block_size_bytes) == expected


def test_get_budget_status_keys(monkeypatch: pytest.MonkeyPatch):
    memory_coordinator = _memory_coordinator_class()
    total_gpu_bytes = 24 * 1024**3
    mc = memory_coordinator(device_memory_ratio=0.7, kv_cache_memory_ratio=0.15)
    monkeypatch.setattr(
        mc, "total_gpu_memory_bytes", lambda device_id=0: total_gpu_bytes
    )
    status = mc.get_budget_status()
    expected_keys = {
        "total_gpu_bytes",
        "expert_cache_bytes",
        "kv_cache_bytes",
        "remaining_bytes",
        "device_memory_ratio",
        "kv_cache_memory_ratio",
    }
    assert expected_keys.issubset(status.keys())


def test_budget_invariant(monkeypatch: pytest.MonkeyPatch):
    memory_coordinator = _memory_coordinator_class()
    total_gpu_bytes = 24 * 1024**3
    mc = memory_coordinator(device_memory_ratio=0.7, kv_cache_memory_ratio=0.15)
    monkeypatch.setattr(
        mc, "total_gpu_memory_bytes", lambda device_id=0: total_gpu_bytes
    )
    assert (
        mc.expert_cache_bytes() + mc.kv_cache_bytes()
        <= mc.total_gpu_memory_bytes()
    )


def test_compute_safe_budget_per_device(monkeypatch: pytest.MonkeyPatch):
    memory_coordinator = _memory_coordinator_class()
    GiB = 1024**3
    mc = memory_coordinator(device_memory_ratio=0.7, kv_cache_memory_ratio=0.15)

    def fake_total(device_id: int = 0) -> int:
        return {0: 8 * GiB, 1: 12 * GiB}[device_id]

    monkeypatch.setattr(mc, "total_gpu_memory_bytes", fake_total)
    assert (
        mc.compute_safe_budget(
            device_id=0,
            model_bytes=2 * GiB,
            activation_reserve_bytes=GiB,
            free_reserve_bytes=GiB,
        )
        == 4 * GiB
    )
    assert (
        mc.compute_safe_budget(
            device_id=1,
            model_bytes=2 * GiB,
            activation_reserve_bytes=GiB,
            free_reserve_bytes=GiB,
        )
        == 8 * GiB
    )


def test_validate_targets_rejects_over_budget_per_device(
    monkeypatch: pytest.MonkeyPatch,
):
    memory_coordinator = _memory_coordinator_class()
    GiB = 1024**3
    mc = memory_coordinator(device_memory_ratio=0.7, kv_cache_memory_ratio=0.15)

    def fake_total(device_id: int = 0) -> int:
        return {0: 8 * GiB, 1: 12 * GiB}[device_id]

    monkeypatch.setattr(mc, "total_gpu_memory_bytes", fake_total)
    safe0 = mc.compute_safe_budget(
        device_id=0,
        model_bytes=2 * GiB,
        activation_reserve_bytes=GiB,
        free_reserve_bytes=GiB,
    )
    safe1 = mc.compute_safe_budget(
        device_id=1,
        model_bytes=2 * GiB,
        activation_reserve_bytes=GiB,
        free_reserve_bytes=GiB,
    )
    with pytest.raises(ValueError):
        mc.validate_targets(
            device_id=0,
            expert_bytes=5 * GiB,
            kv_blocks=0,
            kv_block_bytes=64,
            safe_budget_bytes=safe0,
        )
    mc.validate_targets(
        device_id=1,
        expert_bytes=5 * GiB,
        kv_blocks=0,
        kv_block_bytes=64,
        safe_budget_bytes=safe1,
    )
