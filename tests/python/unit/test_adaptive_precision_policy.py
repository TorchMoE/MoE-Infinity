# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import pytest

from moe_infinity.memory.adaptive_precision_policy import (
    AdaptivePrecisionPolicy,
    ExpertKey,
)
from moe_infinity.runtime.expert_precision import (
    ExpertFormat,
    ResidentGeneration,
)


def _variants():
    return {
        ExpertKey(0, 0): {
            ExpertFormat.FP8_E4M3_BLOCK128: 100,
            ExpertFormat.BF16: 200,
        },
        ExpertKey(0, 1): {
            ExpertFormat.FP8_E4M3_BLOCK128: 100,
            ExpertFormat.BF16: 200,
        },
        ExpertKey(0, 2): {
            ExpertFormat.FP8_E4M3_BLOCK128: 100,
            ExpertFormat.BF16: 200,
        },
    }


def test_policy_is_deterministic_and_uses_lexical_tie_break():
    left = AdaptivePrecisionPolicy(300, 1.0, 0.7, 0.3, 0, 0, _variants())
    right = AdaptivePrecisionPolicy(300, 1.0, 0.7, 0.3, 0, 0, _variants())
    resident = {
        ExpertKey(0, 0): ResidentGeneration(
            ExpertFormat.FP8_E4M3_BLOCK128, 100, 1
        ),
        ExpertKey(0, 1): ResidentGeneration(
            ExpertFormat.FP8_E4M3_BLOCK128, 100, 1
        ),
    }
    for policy in (left, right):
        policy.observe(
            {ExpertKey(0, 0): 10, ExpertKey(0, 1): 10, ExpertKey(0, 2): 1},
            tokens=10,
        )
    left_plan = left.plan(resident, set(), 0, 0)
    right_plan = right.plan(resident, set(), 0, 0)
    assert left_plan == right_plan
    assert left_plan.targets[ExpertKey(0, 0)] is ExpertFormat.BF16
    assert left_plan.targets[ExpertKey(0, 1)] is ExpertFormat.FP8_E4M3_BLOCK128
    assert left_plan.accounted_bytes == 300


def test_host_ready_catalog_is_not_charged_to_hbm():
    catalog = {
        ExpertKey(0, expert_id): {
            ExpertFormat.FP8_E4M3_BLOCK128: 100,
            ExpertFormat.BF16: 200,
        }
        for expert_id in range(1000)
    }
    policy = AdaptivePrecisionPolicy(200, 1.0, 0.7, 0.3, 0, 0, catalog)
    resident = {
        ExpertKey(0, 0): ResidentGeneration(
            ExpertFormat.FP8_E4M3_BLOCK128, 100, 1
        )
    }
    plan = policy.plan(resident, set(), 0, 0)
    assert plan.accounted_bytes == 100
    assert ExpertKey(0, 999) not in plan.targets


def test_policy_never_exceeds_budget_during_transition():
    policy = AdaptivePrecisionPolicy(500, 1.0, 0.7, 0.3, 0, 0, _variants())
    policy.observe({ExpertKey(0, 0): 10}, tokens=10)
    resident = {ExpertKey(0, 0): ResidentGeneration(ExpertFormat.BF16, 200, 1)}
    plan = policy.plan(
        resident,
        {ExpertKey(0, 1)},
        transition_reserved_bytes=100,
        workspace_bytes=100,
    )
    assert plan.accounted_bytes <= 500
    assert sum(intent.reserve_bytes for intent in plan.transitions) <= 100


def test_hysteresis_prevents_epoch_to_epoch_flapping():
    policy = AdaptivePrecisionPolicy(500, 0.5, 0.7, 0.3, 2, 2, _variants())
    policy.observe({ExpertKey(0, 0): 10}, tokens=10)
    resident = {
        ExpertKey(0, 0): ResidentGeneration(
            ExpertFormat.FP8_E4M3_BLOCK128, 100, 1
        )
    }
    first = policy.plan(resident, set(), 0, 0)
    policy.commit(first)
    policy.observe({ExpertKey(0, 1): 11}, tokens=10)
    second = policy.plan(resident, {ExpertKey(0, 1)}, 0, 0)
    assert not second.transitions


def test_simulation_replay_is_byte_for_byte_reproducible():
    trace = [
        {ExpertKey(0, 0): 4, ExpertKey(0, 1): 1},
        {ExpertKey(0, 1): 5},
        {ExpertKey(0, 2): 8},
    ]
    one = AdaptivePrecisionPolicy.simulate(trace, _variants(), budget_bytes=400)
    two = AdaptivePrecisionPolicy.simulate(trace, _variants(), budget_bytes=400)
    assert one.to_json() == two.to_json()


def test_cold_resident_is_demoted_one_quality_rank():
    policy = AdaptivePrecisionPolicy(500, 1.0, 0.7, 0.3, 0, 0, _variants())
    policy.observe({ExpertKey(0, 1): 10}, tokens=10)
    resident = {ExpertKey(0, 0): ResidentGeneration(ExpertFormat.BF16, 200, 1)}

    plan = policy.plan(resident, set(), 0, 0)

    assert plan.targets[ExpertKey(0, 0)] is ExpertFormat.FP8_E4M3_BLOCK128
    assert plan.transitions[0].source_format is ExpertFormat.BF16
    assert plan.transitions[0].target_format is ExpertFormat.FP8_E4M3_BLOCK128
    assert plan.transitions[0].reserve_bytes == 0


def test_precision_plan_targets_are_immutable():
    policy = AdaptivePrecisionPolicy(200, 1.0, 0.7, 0.3, 0, 0, _variants())
    resident = {
        ExpertKey(0, 0): ResidentGeneration(
            ExpertFormat.FP8_E4M3_BLOCK128, 100, 1
        )
    }

    plan = policy.plan(resident, set(), 0, 0)

    with pytest.raises(TypeError):
        plan.targets[ExpertKey(0, 0)] = ExpertFormat.BF16
