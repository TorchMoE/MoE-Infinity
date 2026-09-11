# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import numpy as np

from moe_infinity.memory.expert_policy import ExpertPhase, PhasePolicySettings
from moe_infinity.memory.expert_prefetcher import ExpertPrefetcher


def settings() -> PhasePolicySettings:
    return PhasePolicySettings(
        True, "transient_on_pressure", "cache", 0, 2, 2, 1, 1.0, 4.0, 8
    )


class RecordingEngine:
    def __init__(self) -> None:
        self.calls = []

    def prefetch_tensors(self, tensor_ids, priority, phase) -> None:
        self.calls.append(
            (sorted(int(t) for t in tensor_ids), int(priority), int(phase))
        )


def make_prefetcher(policy: PhasePolicySettings) -> ExpertPrefetcher:
    prefetcher = object.__new__(ExpertPrefetcher)
    prefetcher.num_layers = 4
    prefetcher.num_experts = 4
    prefetcher.phase_policy = policy
    prefetcher.route_ahead_priority = 1
    prefetcher.archer_engine = RecordingEngine()
    prefetcher.expert_tensor_map = {(1, i): 10 + i for i in range(4)}
    prefetcher._last_speculative_prediction = {}
    return prefetcher


def test_prefill_top_k_zero_issues_no_predictive_prefetch() -> None:
    prefetcher = object.__new__(ExpertPrefetcher)
    prefetcher.num_layers = 4
    prefetcher.num_experts = 4
    prefetcher.phase_policy = settings()
    prefetcher.route_ahead_priority = 1
    prefetcher.archer_engine = RecordingEngine()
    prefetcher.expert_tensor_map = {(1, i): 10 + i for i in range(4)}
    prefetcher._last_speculative_prediction = {}
    prefetcher.speculative_prefetch(
        0, np.array([[1, 4, 3, 2]]), phase=ExpertPhase.PREFILL
    )
    assert prefetcher.archer_engine.calls == []


def test_decode_uses_decode_top_k_and_priority() -> None:
    prefetcher = make_prefetcher(settings())
    prefetcher.speculative_prefetch(
        0, np.array([[1, 4, 3, 2]]), phase=ExpertPhase.DECODE
    )
    assert prefetcher.archer_engine.calls == [
        ([11, 12], 1, int(ExpertPhase.DECODE))
    ]
