"""CPU-only tests for the PD-DFlash native offloaded-expert stat accessors.

Exercises the Python surface the §8 runner probes (``_serving_measure`` and
``ExpertPrefetcher``) with fakes standing in for the native ``_store`` engine,
so the wiring is verified without a GPU or a built extension. The native C++
getters themselves are covered by the on-hardware §8 re-run.
"""

from __future__ import annotations

import torch

from benchmarks.dflash._serving_measure import (
    _count_offloaded_experts,
    _native_wasted_prefetch_bytes,
    _probe_h2d_gbps,
)
from moe_infinity.memory.expert_prefetcher import (
    ExpertPrefetcher,
    _hit_rate_from_visit_counts,
)


def _bare_prefetcher(**attrs):
    prefetcher = object.__new__(ExpertPrefetcher)
    prefetcher.archer_engine = None
    prefetcher.expert_dispatcher = None
    prefetcher.expert_tensor_map = {}
    for key, value in attrs.items():
        setattr(prefetcher, key, value)
    return prefetcher


class _FakeHandle:
    def __init__(self, *, offloaded=(), occupancy=0, wasted=0, counts=None):
        self._offloaded = set(offloaded)
        self._occupancy = occupancy
        self._wasted = wasted
        self._counts = counts

    def is_tensor_offloaded(self, tensor_id):
        return int(tensor_id) in self._offloaded

    def get_expert_occupancy_bytes(self):
        return self._occupancy

    def get_wasted_prefetch_bytes(self):
        return self._wasted

    def get_hit_rate(self):
        return self._counts


class _FakeDispatcher:
    def __init__(self, *, hit_rate=0.0, occupancy=0):
        self._hit_rate = hit_rate
        self._occupancy = occupancy

    def get_cache_hit_rate(self):
        return self._hit_rate

    def get_cache_occupancy_bytes(self):
        return self._occupancy


def test_num_offloaded_experts_counts_offloaded_tensor_ids():
    handle = _FakeHandle(offloaded=(10, 11, 13))
    prefetcher = _bare_prefetcher(
        archer_engine=handle,
        expert_tensor_map={(0, 0): 10, (0, 1): 11, (0, 2): 12, (1, 0): 13},
    )
    assert prefetcher.num_offloaded_experts == 3


def test_num_offloaded_experts_falls_back_to_map_len_without_checker():
    prefetcher = _bare_prefetcher(
        archer_engine=object(),
        expert_tensor_map={(0, 0): 1, (0, 1): 2},
    )
    assert prefetcher.num_offloaded_experts == 2


def test_hit_rate_prefers_nonzero_dispatcher_signal():
    prefetcher = _bare_prefetcher(
        archer_engine=_FakeHandle(counts=torch.tensor([[10, 0, 0, 9]])),
        expert_dispatcher=_FakeDispatcher(hit_rate=0.75),
    )
    assert prefetcher.get_hit_rate() == 0.75


def test_hit_rate_falls_back_to_topology_visit_counts():
    counts = torch.tensor([[10, 6, 4, 4], [6, 4, 2, 3]], dtype=torch.int64)
    prefetcher = _bare_prefetcher(archer_engine=_FakeHandle(counts=counts))
    assert abs(prefetcher.get_hit_rate() - (7.0 / 16.0)) < 1e-9


def test_hit_rate_zero_when_no_signal():
    prefetcher = _bare_prefetcher(archer_engine=_FakeHandle(counts=None))
    assert prefetcher.get_hit_rate() == 0.0


def test_hit_rate_from_visit_counts_handles_empty_and_no_visits():
    assert _hit_rate_from_visit_counts(None) is None
    assert _hit_rate_from_visit_counts(torch.empty((0, 4))) is None
    assert _hit_rate_from_visit_counts(torch.tensor([[0, 0, 0, 0]])) is None


def test_expert_occupancy_sums_dispatcher_and_handle():
    prefetcher = _bare_prefetcher(
        archer_engine=_FakeHandle(occupancy=2048),
        expert_dispatcher=_FakeDispatcher(occupancy=4096),
    )
    assert prefetcher.expert_occupancy_bytes() == 6144.0


def test_wasted_prefetch_bytes_reads_handle_getter():
    prefetcher = _bare_prefetcher(archer_engine=_FakeHandle(wasted=512))
    assert prefetcher.wasted_prefetch_bytes() == 512.0


class _FakeEngine:
    def __init__(self, prefetcher=None, num_offloaded=None):
        if prefetcher is not None:
            self.expert_prefetcher = prefetcher
        if num_offloaded is not None:
            self.num_offloaded_experts = num_offloaded


def test_count_offloaded_experts_reads_native_engine_attr():
    assert _count_offloaded_experts(_FakeEngine(num_offloaded=7)) == 7


def test_count_offloaded_experts_falls_back_to_nbytes_map():
    class _P:
        expert_nbytes_map = {(0, 0): 10, (0, 1): 20}

    assert _count_offloaded_experts(_FakeEngine(prefetcher=_P())) == 2


def test_native_wasted_prefetch_bytes_reads_prefetcher():
    prefetcher = _bare_prefetcher(archer_engine=_FakeHandle(wasted=768))
    engine = _FakeEngine(prefetcher=prefetcher)
    assert _native_wasted_prefetch_bytes(engine) == 768.0


def test_native_wasted_prefetch_bytes_none_without_getter():
    assert _native_wasted_prefetch_bytes(_FakeEngine()) is None


def test_probe_h2d_gbps_returns_none_or_positive():
    result = _probe_h2d_gbps(nbytes=1 << 20, iters=2)
    assert result is None or (isinstance(result, float) and result > 0.0)


def test_policy_stats_fallback_is_disabled_and_zero() -> None:
    prefetcher = _bare_prefetcher(
        archer_engine=object(), expert_dispatcher=object()
    )
    stats = prefetcher.get_policy_stats()
    assert stats["enabled"] == 0
    assert stats["resident_bytes"] == 0
    assert stats["decode_hits"] == 0
