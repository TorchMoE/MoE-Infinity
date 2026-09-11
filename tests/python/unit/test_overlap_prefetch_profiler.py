# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from collections import deque

import pytest

from moe_infinity.memory.expert_tracer import ExpertTracer
from moe_infinity.profiling.io_profiler import IOProfiler


def test_record_emits_overlap_decision_without_timing_context(monkeypatch):
    p = IOProfiler(pid=1)
    p.enabled = True
    p.record(
        "prefetch_budget",
        layer=3,
        bytes=4096,
        fields={"budget_bytes": 8192, "generation": 7},
    )
    assert p._events == [
        {
            "ts_ns": p._events[0]["ts_ns"],
            "stage": "prefetch_budget",
            "layer": 3,
            "expert": None,
            "dur_ns": 0,
            "bytes": 4096,
            "budget_bytes": 8192,
            "generation": 7,
        }
    ]


@pytest.mark.parametrize(
    "reserved", ["ts_ns", "stage", "layer", "expert", "dur_ns", "bytes"]
)
def test_record_rejects_fields_overwriting_core_keys(reserved):
    p = IOProfiler(pid=1)
    p.enabled = True
    with pytest.raises(ValueError, match=reserved):
        p.record("prefetch_admit", layer=1, fields={reserved: 1})


def test_tracer_aggregates_overlap_byte_stages():
    tracer = ExpertTracer.__new__(ExpertTracer)
    tracer._io_profiling_enabled = True
    tracer._io_events = deque(maxlen=10)
    tracer.record_io_event(3, -1, "prefetch_late", 0, 300)
    assert tracer.get_io_stats()["prefetch_late"]["total_bytes"] == 300
