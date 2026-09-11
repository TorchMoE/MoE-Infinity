# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator


class ExpertPhase(IntEnum):
    PREFILL = 0
    DECODE = 1
    MIXED = 2


_CURRENT_EXPERT_PHASE: ContextVar[ExpertPhase] = ContextVar(
    "current_expert_phase", default=ExpertPhase.MIXED
)


def current_expert_phase() -> ExpertPhase:
    return _CURRENT_EXPERT_PHASE.get()


@contextmanager
def expert_phase_scope(phase: ExpertPhase) -> Iterator[None]:
    token = _CURRENT_EXPERT_PHASE.set(phase)
    try:
        yield
    finally:
        _CURRENT_EXPERT_PHASE.reset(token)


@dataclass(frozen=True)
class PhasePolicySettings:
    enabled: bool
    prefill_admission: str
    decode_admission: str
    prefill_prefetch_top_k: int
    decode_prefetch_top_k: int
    prefill_prefetch_priority: int
    decode_prefetch_priority: int
    prefill_eviction_weight: float
    decode_eviction_weight: float
    starvation_limit: int

    def effective_phase(self, phase: ExpertPhase) -> ExpertPhase:
        return ExpertPhase.DECODE if phase is ExpertPhase.MIXED else phase


__all__ = [
    "ExpertPhase",
    "PhasePolicySettings",
    "current_expert_phase",
    "expert_phase_scope",
]
