# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Candidate:
    expert_id: int
    score: float
    nbytes: Optional[int]


@dataclass(frozen=True)
class AdmissionDecision:
    expert_ids: Tuple[int, ...]
    candidate_bytes: int
    budget_bytes: int
    admitted_bytes: int
    uncosted_experts: Tuple[int, ...]
    cold_start: bool


@dataclass
class _IssuedGeneration:
    layer_id: int
    expert_nbytes: Dict[int, int]
    completed: Set[int] = field(default_factory=set)


class OverlapBudgetController:
    def __init__(
        self,
        *,
        policy: str,
        alpha: float,
        safety_factor: float,
        max_window_bytes: int,
        max_inflight_bytes: int,
        cold_start_experts: int,
    ) -> None:
        self.policy = policy
        self.alpha = alpha
        self.safety_factor = safety_factor
        self.max_window_bytes = max_window_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.cold_start_experts = cold_start_experts
        self.compute_ewma_ns: Dict[int, float] = {}
        self.bandwidth_ewma_bytes_per_ns: Optional[float] = None
        self.queue_wait_ewma_ns: Optional[float] = None
        self.issue_overhead_ewma_ns: Optional[float] = None

        self._issued: Dict[int, _IssuedGeneration] = {}
        self.covered_route_bytes = 0
        self.late_prefetch_bytes = 0
        self.uncovered_route_bytes = 0
        self.wasted_prefetch_bytes = 0
        self.canceled_prefetch_bytes = 0
        self.route_bytes = 0

    def _ewma(self, old: Optional[float], sample: float) -> float:
        if old is None:
            return sample
        return self.alpha * sample + (1 - self.alpha) * old

    def observe_compute(
        self, layer_id: int, start_ns: int, end_ns: int
    ) -> None:
        sample = end_ns - start_ns
        if sample > 0:
            self.compute_ewma_ns[layer_id] = self._ewma(
                self.compute_ewma_ns.get(layer_id), float(sample)
            )

    def observe_transfer(
        self,
        bytes_transferred: int,
        transfer_ns: int,
        queue_wait_ns: int,
        issue_overhead_ns: int,
    ) -> None:
        if bytes_transferred <= 0 or transfer_ns <= 0:
            return
        self.bandwidth_ewma_bytes_per_ns = self._ewma(
            self.bandwidth_ewma_bytes_per_ns,
            bytes_transferred / transfer_ns,
        )
        self.queue_wait_ewma_ns = self._ewma(
            self.queue_wait_ewma_ns, float(max(queue_wait_ns, 0))
        )
        self.issue_overhead_ewma_ns = self._ewma(
            self.issue_overhead_ewma_ns, float(max(issue_overhead_ns, 0))
        )

    def admit(
        self,
        layer_id: int,
        candidates: Sequence[Candidate],
        inflight_bytes: int = 0,
    ) -> AdmissionDecision:
        ordered = sorted(
            enumerate(candidates),
            key=lambda item: (-item[1].score, item[0], item[1].expert_id),
        )
        uncosted = tuple(
            c.expert_id for _, c in ordered if c.nbytes is None or c.nbytes <= 0
        )
        costed = [
            c for _, c in ordered if c.nbytes is not None and c.nbytes > 0
        ]
        candidate_bytes = sum(int(c.nbytes) for c in costed)
        warm = (
            layer_id in self.compute_ewma_ns
            and self.bandwidth_ewma_bytes_per_ns is not None
        )
        if warm:
            window = max(
                0.0,
                self.safety_factor * self.compute_ewma_ns[layer_id]
                - (self.queue_wait_ewma_ns or 0.0)
                - (self.issue_overhead_ewma_ns or 0.0),
            )
            budget = min(
                self.max_window_bytes,
                floor(self.bandwidth_ewma_bytes_per_ns * window),
            )
            budget = max(
                0,
                min(
                    budget - max(inflight_bytes, 0),
                    self.max_inflight_bytes - max(inflight_bytes, 0),
                ),
            )
            limit = len(costed)
        else:
            budget = max(0, self.max_inflight_bytes - max(inflight_bytes, 0))
            limit = self.cold_start_experts
        selected: List[int] = []
        used = 0
        for c in costed:
            cost = int(c.nbytes)
            if len(selected) < limit and used + cost <= budget:
                selected.append(c.expert_id)
                used += cost
        return AdmissionDecision(
            tuple(selected),
            candidate_bytes,
            budget,
            used,
            uncosted,
            not warm,
        )

    def record_issue(
        self,
        layer_id: int,
        generation: int,
        expert_nbytes: Dict[int, int],
    ) -> None:
        self._issued[generation] = _IssuedGeneration(
            layer_id=layer_id,
            expert_nbytes={int(k): int(v) for k, v in expert_nbytes.items()},
        )

    def record_completion(
        self, generation: int, expert_id: int, bytes_transferred: int
    ) -> None:
        issued = self._issued.get(generation)
        if issued is None:
            return
        if expert_id in issued.expert_nbytes:
            issued.completed.add(int(expert_id))

    def correct_route(
        self,
        layer_id: int,
        generation: int,
        actual_expert_nbytes: Dict[int, int],
    ) -> None:
        actual = {int(k): int(v) for k, v in actual_expert_nbytes.items()}
        self.route_bytes += sum(actual.values())
        issued = self._issued.pop(generation, None)
        issued_bytes = issued.expert_nbytes if issued is not None else {}
        completed = issued.completed if issued is not None else set()

        for expert_id, nbytes in actual.items():
            if expert_id in completed:
                self.covered_route_bytes += nbytes
            elif expert_id in issued_bytes:
                self.late_prefetch_bytes += nbytes
            else:
                self.uncovered_route_bytes += nbytes

        for expert_id, nbytes in issued_bytes.items():
            if expert_id in actual:
                continue
            if expert_id in completed:
                self.wasted_prefetch_bytes += nbytes
            else:
                self.canceled_prefetch_bytes += nbytes

    def snapshot(self) -> Dict[str, Any]:
        coverage = (
            self.covered_route_bytes / self.route_bytes
            if self.route_bytes > 0
            else 1.0
        )
        bandwidth_bytes_per_second = (
            self.bandwidth_ewma_bytes_per_ns * 1e9
            if self.bandwidth_ewma_bytes_per_ns is not None
            else None
        )
        return {
            "covered_route_bytes": self.covered_route_bytes,
            "late_prefetch_bytes": self.late_prefetch_bytes,
            "uncovered_route_bytes": self.uncovered_route_bytes,
            "wasted_prefetch_bytes": self.wasted_prefetch_bytes,
            "canceled_prefetch_bytes": self.canceled_prefetch_bytes,
            "route_bytes": self.route_bytes,
            "coverage": coverage,
            "compute_ewma_ns_by_layer": dict(self.compute_ewma_ns),
            "bandwidth_ewma_bytes_per_second": bandwidth_bytes_per_second,
            "queue_wait_ewma_ns": self.queue_wait_ewma_ns,
            "issue_overhead_ewma_ns": self.issue_overhead_ewma_ns,
        }
