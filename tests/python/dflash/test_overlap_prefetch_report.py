# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import pytest

from benchmarks.expert_io_microbench.bench_overlap_prefetch import build_report


def test_report_uses_measured_metrics_without_speedup_promise():
    report = build_report(
        policy="enforce",
        latencies_ms=[10.0, 11.0],
        stats={
            "route_bytes": 1000,
            "covered_route_bytes": 700,
            "wasted_prefetch_bytes": 100,
            "late_prefetch_bytes": 200,
            "admitted_bytes": 900,
            "budget_bytes": 1200,
            "queue_rejected_bytes": 50,
        },
        hardware={"gpu": "test", "pcie": "test"},
    )
    assert report["metrics"]["coverage"] == pytest.approx(0.7)
    assert report["metrics"]["waste_ratio"] == pytest.approx(1 / 9)
    assert report["metrics"]["late_ratio"] == pytest.approx(0.2)
    assert report["verdict"] == "MEASURED"
    assert "speedup" not in report
