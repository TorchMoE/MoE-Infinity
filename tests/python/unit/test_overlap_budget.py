# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import pytest

from moe_infinity.memory.overlap_budget import (
    Candidate,
    OverlapBudgetController,
)


def controller(**kwargs):
    return OverlapBudgetController(
        policy="enforce",
        alpha=0.5,
        safety_factor=0.8,
        max_window_bytes=10_000,
        max_inflight_bytes=10_000,
        cold_start_experts=1,
        **kwargs,
    )


def test_warm_budget_subtracts_queue_issue_and_inflight():
    c = controller()
    c.observe_compute(layer_id=3, start_ns=0, end_ns=1_000)
    c.observe_transfer(
        bytes_transferred=1000,
        transfer_ns=100,
        queue_wait_ns=100,
        issue_overhead_ns=100,
    )
    d = c.admit(
        3,
        [Candidate(2, 3.0, 3000), Candidate(1, 2.0, 2500)],
        inflight_bytes=500,
    )
    # window=(.8*1000)-100-100=600ns; 10 B/ns => 6000B; minus 500B.
    assert d.budget_bytes == 5500
    assert d.expert_ids == (2, 1)
    assert d.admitted_bytes == 5500


def test_cold_start_admits_only_one_exact_costed_expert():
    d = controller().admit(0, [Candidate(7, 9.0, 4096), Candidate(3, 8.0, 1)])
    assert d.cold_start is True
    assert d.expert_ids == (7,)


def test_missing_cost_is_never_fabricated_in_enforce():
    d = controller().admit(0, [Candidate(7, 9.0, None)])
    assert d.expert_ids == ()
    assert d.uncosted_experts == (7,)


def test_stable_whole_expert_packing_skips_non_fitting_candidate():
    c = controller()
    c.observe_compute(1, 0, 1_000)
    c.observe_transfer(1000, 100, 0, 0)  # budget=8000
    d = c.admit(
        1,
        [
            Candidate(9, 3.0, 9000),
            Candidate(4, 2.0, 4000),
            Candidate(2, 2.0, 4000),
        ],
    )
    assert d.expert_ids == (4, 2)


def test_exact_route_correction_separates_coverage_waste_late_and_uncovered():
    c = controller()
    c.record_issue(
        layer_id=2, generation=8, expert_nbytes={1: 100, 2: 200, 3: 300}
    )
    c.record_completion(generation=8, expert_id=1, bytes_transferred=100)
    c.correct_route(
        layer_id=2,
        generation=8,
        actual_expert_nbytes={1: 100, 2: 200, 4: 400},
    )
    s = c.snapshot()
    assert s["covered_route_bytes"] == 100
    assert s["late_prefetch_bytes"] == 200
    assert s["uncovered_route_bytes"] == 400
    # Expert 3 is a false positive but never completed, so it is canceled bytes,
    # not transferred waste.
    assert s["wasted_prefetch_bytes"] == 0
    assert s["canceled_prefetch_bytes"] == 300
    assert s["coverage"] == pytest.approx(1 / 7)


def test_only_completed_false_positive_counts_as_waste():
    c = controller()
    c.record_issue(layer_id=2, generation=9, expert_nbytes={1: 100, 3: 300})
    c.record_completion(generation=9, expert_id=1, bytes_transferred=100)
    c.record_completion(generation=9, expert_id=3, bytes_transferred=300)
    c.correct_route(layer_id=2, generation=9, actual_expert_nbytes={1: 100})
    s = c.snapshot()
    assert s["wasted_prefetch_bytes"] == 300
    assert s["canceled_prefetch_bytes"] == 0


def test_ewma_uses_configured_alpha():
    c = controller()
    c.observe_compute(0, 0, 100)
    c.observe_compute(0, 0, 300)
    assert c.compute_ewma_ns[0] == 200
