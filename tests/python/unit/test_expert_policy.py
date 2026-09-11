# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from moe_infinity.memory.expert_policy import (
    ExpertPhase,
    current_expert_phase,
    expert_phase_scope,
)


def test_phase_scope_restores_nested_state() -> None:
    assert current_expert_phase() is ExpertPhase.MIXED
    with expert_phase_scope(ExpertPhase.PREFILL):
        assert current_expert_phase() is ExpertPhase.PREFILL
        with expert_phase_scope(ExpertPhase.DECODE):
            assert current_expert_phase() is ExpertPhase.DECODE
        assert current_expert_phase() is ExpertPhase.PREFILL
    assert current_expert_phase() is ExpertPhase.MIXED
