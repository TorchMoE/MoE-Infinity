from types import SimpleNamespace

import torch

from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.kv_cache import PagedKVCache

SWAP_KEYS = {
    "mode",
    "fallback_reason",
    "host_capacity_bytes",
    "host_in_use_bytes",
    "host_peak_in_use_bytes",
    "inflight",
    "inflight_bytes",
    "retiring_records",
    "host_resident",
    "backpressure_total",
    "swap_out_started_total",
    "swap_out_completed_total",
    "swap_out_failed_total",
    "swap_in_started_total",
    "swap_in_completed_total",
    "swap_in_failed_total",
    "cancelled_total",
    "checksum_failures_total",
    "d2h_bytes_total",
    "h2d_bytes_total",
    "d2h_duration_ms_sum",
    "h2d_duration_ms_sum",
}


def test_engine_stats_include_complete_kv_swap_contract() -> None:
    swap_stats = {key: 0 for key in SWAP_KEYS}
    swap_stats["mode"] = "async"
    swap_stats["fallback_reason"] = None
    cache = SimpleNamespace(
        num_blocks=4,
        block_allocator=SimpleNamespace(num_free_blocks=4),
        get_swap_stats=lambda: dict(swap_stats),
    )
    engine = SimpleNamespace(
        _sequences={},
        _pending_request_ids=lambda: set(),
        _completed_request_ids=set(),
        _cancelled_request_ids=set(),
        _num_steps=0,
        _total_generated_tokens=0,
        kv_cache=cache,
        memory_manager=SimpleNamespace(report=lambda: {}),
        _kv_swap_mode="async",
        _kv_swap_fallback_reason=None,
    )

    stats = ContinuousBatchingEngine.get_stats(engine)

    assert stats["kv_swap"] == swap_stats


def test_cache_swap_stats_are_complete_and_monotonic() -> None:
    cache = PagedKVCache(
        num_blocks=1,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert SWAP_KEYS <= set(cache.get_swap_stats())

    cache.allocate_sequence(1, num_tokens=4)
    cache.swap_out(1)
    cache.free_gpu_blocks(1)
    cache.swap_in(1)
    stats = cache.get_swap_stats()
    assert stats["swap_out_started_total"] == 1
    assert stats["swap_out_completed_total"] == 1
    assert stats["swap_in_started_total"] == 1
    assert stats["swap_in_completed_total"] == 1
    assert stats["d2h_bytes_total"] > 0
    assert stats["h2d_bytes_total"] == stats["d2h_bytes_total"]
    assert stats["host_capacity_bytes"] == 0
    assert stats["host_in_use_bytes"] == 0
