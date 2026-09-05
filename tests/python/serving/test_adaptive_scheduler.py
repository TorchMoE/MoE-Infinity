import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Optional, Protocol, cast

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MEMORY_MANAGER_PATH = (
    Path(ROOT) / "moe_infinity" / "serving" / "memory_manager.py"
)
_MISSING_MODULE = object()


class AdaptiveKVSchedulerProtocol(Protocol):
    def __init__(
        self,
        initial_max_batch_size: int = 32,
        kv_high_watermark: float = 0.90,
        kv_low_watermark: float = 0.50,
        expert_hit_target: float = 0.70,
        cooldown_seconds: float = 10.0,
    ) -> None: ...

    def observe(
        self,
        kv_used_blocks: int,
        kv_total_blocks: int,
        expert_hit_rate: Optional[float] = None,
    ) -> None: ...

    def get_max_batch_size(self) -> int: ...

    def should_preempt_aggressively(self) -> bool: ...

    def get_tuning_recommendations(self) -> dict[str, object]: ...


def _load_adaptive_scheduler() -> type[AdaptiveKVSchedulerProtocol]:
    module_name = "task15_adaptive_scheduler"
    spec = importlib.util.spec_from_file_location(
        module_name, MEMORY_MANAGER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MEMORY_MANAGER_PATH}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name, _MISSING_MODULE)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_module is _MISSING_MODULE:
            _ = sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = cast(ModuleType, previous_module)
    return cast(
        type[AdaptiveKVSchedulerProtocol],
        getattr(module, "AdaptiveKVScheduler"),
    )


def test_reduces_batch_size_under_expert_pressure() -> None:
    AdaptiveKVScheduler = _load_adaptive_scheduler()
    scheduler = AdaptiveKVScheduler(
        initial_max_batch_size=32,
        cooldown_seconds=0.0,
        expert_hit_target=0.70,
    )

    scheduler.observe(
        kv_used_blocks=20, kv_total_blocks=100, expert_hit_rate=0.40
    )

    assert scheduler.get_max_batch_size() < 32


def test_increases_batch_size_when_idle() -> None:
    AdaptiveKVScheduler = _load_adaptive_scheduler()
    scheduler = AdaptiveKVScheduler(
        initial_max_batch_size=32,
        cooldown_seconds=0.0,
    )

    scheduler.observe(
        kv_used_blocks=95, kv_total_blocks=100, expert_hit_rate=0.90
    )
    reduced_batch_size = scheduler.get_max_batch_size()
    assert reduced_batch_size < 32

    scheduler.observe(
        kv_used_blocks=10, kv_total_blocks=100, expert_hit_rate=0.90
    )

    assert scheduler.get_max_batch_size() > reduced_batch_size
    assert scheduler.get_max_batch_size() <= 32


def test_cooldown_prevents_rapid_changes() -> None:
    AdaptiveKVScheduler = _load_adaptive_scheduler()
    scheduler = AdaptiveKVScheduler(
        initial_max_batch_size=16,
        cooldown_seconds=60.0,
    )

    scheduler.observe(
        kv_used_blocks=15, kv_total_blocks=16, expert_hit_rate=0.90
    )
    first_adjustment = scheduler.get_max_batch_size()

    scheduler.observe(
        kv_used_blocks=15, kv_total_blocks=16, expert_hit_rate=0.90
    )
    assert scheduler.get_max_batch_size() == first_adjustment

    setattr(scheduler, "_last_adjust_timestamp", time.monotonic() - 61.0)
    scheduler.observe(
        kv_used_blocks=15, kv_total_blocks=16, expert_hit_rate=0.90
    )
    assert scheduler.get_max_batch_size() < first_adjustment


def test_tuning_recommendations_kv_bound() -> None:
    AdaptiveKVScheduler = _load_adaptive_scheduler()
    scheduler = AdaptiveKVScheduler(cooldown_seconds=0.0)

    scheduler.observe(
        kv_used_blocks=95, kv_total_blocks=100, expert_hit_rate=0.90
    )
    recommendation = scheduler.get_tuning_recommendations()

    assert recommendation["recommendation"] == "increase kv_cache_ratio"


def test_preempt_aggressively_above_watermark() -> None:
    AdaptiveKVScheduler = _load_adaptive_scheduler()
    scheduler = AdaptiveKVScheduler(kv_high_watermark=0.80)

    scheduler.observe(
        kv_used_blocks=79, kv_total_blocks=100, expert_hit_rate=0.90
    )
    assert scheduler.should_preempt_aggressively() is False

    scheduler.observe(
        kv_used_blocks=80, kv_total_blocks=100, expert_hit_rate=0.90
    )
    assert scheduler.should_preempt_aggressively() is True
