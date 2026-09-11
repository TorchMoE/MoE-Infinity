# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from moe_infinity.runtime.expert_precision import (
    ExpertFormat,
    ResidentGeneration,
)

_QUALITY_RANK = {
    ExpertFormat.BF16: 0,
    ExpertFormat.FP8_E4M3_BLOCK128: 1,
    ExpertFormat.MARLIN_INT4_GROUP128: 2,
    ExpertFormat.GLM_FP8_BLOCK128: 1,
    ExpertFormat.GPT_OSS_MXFP4: 2,
    ExpertFormat.DEEPSEEK_V4_FP4: 2,
    ExpertFormat.GPTQ: 2,
    ExpertFormat.AWQ: 2,
}


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer_id: int
    expert_id: int


@dataclass(frozen=True)
class TransitionIntent:
    key: ExpertKey
    source_format: Optional[ExpertFormat]
    target_format: ExpertFormat
    reserve_bytes: int


@dataclass(frozen=True)
class PrecisionPlan:
    epoch: int
    targets: Mapping[ExpertKey, ExpertFormat]
    transitions: Tuple[TransitionIntent, ...]
    evictions: Tuple[ExpertKey, ...]
    accounted_bytes: int

    def as_native_targets(
        self,
        generations: Mapping[tuple[ExpertKey, ExpertFormat], int] | None = None,
    ) -> list[tuple[int, int, str, int]]:
        generations = generations or {}
        return [
            (
                key.layer_id,
                key.expert_id,
                fmt.value,
                generations.get((key, fmt), 0),
            )
            for key, fmt in sorted(self.targets.items())
        ]


@dataclass(frozen=True)
class SimulationResult:
    budget_bytes: int
    epochs: Tuple[Dict[str, object], ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "budget_bytes": self.budget_bytes,
                "epochs": list(self.epochs),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


def _quality_rank(fmt: ExpertFormat) -> int:
    return _QUALITY_RANK[fmt]


def _sorted_formats_by_size(
    formats: Mapping[ExpertFormat, int],
) -> List[Tuple[ExpertFormat, int]]:
    return sorted(formats.items(), key=lambda item: (item[1], item[0].value))


class AdaptivePrecisionPolicy:
    def __init__(
        self,
        budget_bytes: int,
        decay: float,
        promotion_threshold: float,
        demotion_threshold: float,
        promote_cooldown: int,
        demote_cooldown: int,
        catalog: Mapping[ExpertKey, Mapping[ExpertFormat, int]],
        generations: Mapping[tuple[ExpertKey, ExpertFormat], int] | None = None,
        epoch_tokens: int = 1,
    ) -> None:
        self.budget_bytes = int(budget_bytes)
        self.decay = Fraction(str(decay))
        self.promotion_threshold = Fraction(str(promotion_threshold))
        self.demotion_threshold = Fraction(str(demotion_threshold))
        self.promote_cooldown = int(promote_cooldown)
        self.demote_cooldown = int(demote_cooldown)
        self.catalog: Dict[ExpertKey, Dict[ExpertFormat, int]] = {
            key: dict(formats) for key, formats in catalog.items()
        }
        self.hotness: Dict[ExpertKey, Fraction] = {}
        self.epoch = 0
        self.cooldown_until = 0
        self.generations = dict(generations or {})
        self.epoch_tokens = max(int(epoch_tokens), 1)
        self.observed_tokens = 0

    @property
    def epoch_due(self) -> bool:
        return self.observed_tokens >= self.epoch_tokens

    def as_native_targets(
        self, plan: PrecisionPlan
    ) -> list[tuple[int, int, str, int]]:
        return plan.as_native_targets(self.generations)

    def observe(self, counts: Mapping[ExpertKey, int], tokens: int) -> None:
        divisor = max(int(tokens), 1)
        self.observed_tokens += divisor
        decayed: Dict[ExpertKey, Fraction] = {
            key: value * (1 - self.decay) for key, value in self.hotness.items()
        }
        for key, count in counts.items():
            sample = self.decay * Fraction(int(count), divisor)
            decayed[key] = (
                decayed.get(
                    key, self.hotness.get(key, Fraction(0)) * (1 - self.decay)
                )
                + sample
            )
        self.hotness = decayed
        self.epoch += 1

    def _normalized_hotness(self) -> Dict[ExpertKey, Fraction]:
        if not self.hotness:
            return {}
        peak = max(self.hotness.values())
        if peak <= 0:
            return {key: Fraction(0) for key in self.hotness}
        return {key: value / peak for key, value in self.hotness.items()}

    def plan(
        self,
        resident: Mapping[ExpertKey, ResidentGeneration],
        admission_candidates: Iterable[ExpertKey],
        transition_reserved_bytes: int,
        workspace_bytes: int,
    ) -> PrecisionPlan:
        normalized = self._normalized_hotness()
        candidates = frozenset(admission_candidates)

        resident_bytes = sum(
            generation.aligned_bytes for generation in resident.values()
        )
        base_reserved = int(transition_reserved_bytes) + int(workspace_bytes)
        accounted = resident_bytes + base_reserved

        targets: Dict[ExpertKey, ExpertFormat] = {}
        for key, generation in resident.items():
            targets[key] = generation.format

        transitions: List[TransitionIntent] = []
        evictions: List[ExpertKey] = []

        if resident_bytes + base_reserved > self.budget_bytes:
            evictions = self._select_evictions(
                resident, resident_bytes + base_reserved
            )
            return PrecisionPlan(
                epoch=self.epoch,
                targets=MappingProxyType(dict(targets)),
                transitions=tuple(transitions),
                evictions=tuple(evictions),
                accounted_bytes=accounted,
            )

        cooldown_active = self.epoch < self.cooldown_until

        admissions = self._admit_candidates(
            candidates, resident, normalized, accounted, cooldown_active
        )
        for intent in admissions:
            targets[intent.key] = intent.target_format
            transitions.append(intent)
            accounted += intent.reserve_bytes

        if not cooldown_active:
            demotions = self._enumerate_demotions(resident, normalized)
            for intent in demotions:
                targets[intent.key] = intent.target_format
                transitions.append(intent)
            upgrades = self._enumerate_upgrades(
                resident, targets, normalized, accounted
            )
            for intent in upgrades:
                targets[intent.key] = intent.target_format
                transitions.append(intent)
                accounted += intent.reserve_bytes

        return PrecisionPlan(
            epoch=self.epoch,
            targets=MappingProxyType(dict(targets)),
            transitions=tuple(transitions),
            evictions=tuple(evictions),
            accounted_bytes=accounted,
        )

    def _admit_candidates(
        self,
        candidates: FrozenSet[ExpertKey],
        resident: Mapping[ExpertKey, ResidentGeneration],
        normalized: Mapping[ExpertKey, Fraction],
        accounted: int,
        cooldown_active: bool,
    ) -> List[TransitionIntent]:
        admitted: List[TransitionIntent] = []
        running = accounted
        for key in sorted(candidates):
            if key in resident or key not in self.catalog:
                continue
            if cooldown_active:
                continue
            if normalized.get(key, Fraction(0)) < self.promotion_threshold:
                continue
            chosen = None
            for fmt, size in _sorted_formats_by_size(self.catalog[key]):
                if running + size <= self.budget_bytes:
                    chosen = (fmt, size)
                    break
            if chosen is None:
                continue
            fmt, size = chosen
            admitted.append(TransitionIntent(key, None, fmt, size))
            running += size
        return admitted

    def _enumerate_demotions(
        self,
        resident: Mapping[ExpertKey, ResidentGeneration],
        normalized: Mapping[ExpertKey, Fraction],
    ) -> List[TransitionIntent]:
        demotions: List[TransitionIntent] = []
        for key in sorted(resident):
            generation = resident[key]
            if generation.state != "active" or key not in self.catalog:
                continue
            if normalized.get(key, Fraction(0)) > self.demotion_threshold:
                continue
            current_rank = _quality_rank(generation.format)
            current_bytes = self.catalog[key].get(
                generation.format, generation.aligned_bytes
            )
            choices = [
                (size, fmt.value, fmt)
                for fmt, size in self.catalog[key].items()
                if _quality_rank(fmt) == current_rank + 1
                and size < current_bytes
            ]
            if not choices:
                continue
            size, _, target = min(choices)
            demotions.append(
                TransitionIntent(
                    key=key,
                    source_format=generation.format,
                    target_format=target,
                    reserve_bytes=max(0, size - current_bytes),
                )
            )
        return demotions

    def _enumerate_upgrades(
        self,
        resident: Mapping[ExpertKey, ResidentGeneration],
        targets: Mapping[ExpertKey, ExpertFormat],
        normalized: Mapping[ExpertKey, Fraction],
        accounted: int,
    ) -> List[TransitionIntent]:
        options: List[Tuple[tuple, TransitionIntent]] = []
        for key, generation in resident.items():
            if key not in self.catalog:
                continue
            current = targets.get(key, generation.format)
            current_rank = _quality_rank(current)
            current_bytes = self.catalog[key].get(
                current, generation.aligned_bytes
            )
            hotness = normalized.get(key, Fraction(0))
            if hotness < self.promotion_threshold:
                continue
            for fmt, size in _sorted_formats_by_size(self.catalog[key]):
                rank = _quality_rank(fmt)
                if rank != current_rank - 1:
                    continue
                extra_bytes = size - current_bytes
                if extra_bytes <= 0:
                    continue
                rank_delta = current_rank - rank
                marginal = hotness * rank_delta
                order = (
                    -marginal / extra_bytes,
                    -marginal,
                    extra_bytes,
                    key.layer_id,
                    key.expert_id,
                    fmt.value,
                )
                options.append(
                    (order, TransitionIntent(key, current, fmt, extra_bytes))
                )

        options.sort(key=lambda item: item[0])

        selected: List[TransitionIntent] = []
        running = accounted
        for _, intent in options:
            if running + intent.reserve_bytes <= self.budget_bytes:
                selected.append(intent)
                running += intent.reserve_bytes
        return selected

    def _select_evictions(
        self,
        resident: Mapping[ExpertKey, ResidentGeneration],
        current_bytes: int,
    ) -> List[ExpertKey]:
        evictions: List[ExpertKey] = []
        running = current_bytes
        for key in sorted(resident):
            if running <= self.budget_bytes:
                break
            generation = resident[key]
            if generation.state != "active":
                continue
            evictions.append(key)
            running -= generation.aligned_bytes
        return evictions

    def commit(self, plan: PrecisionPlan) -> None:
        if plan.transitions:
            self.cooldown_until = self.epoch + max(
                self.promote_cooldown, self.demote_cooldown
            )
        self.observed_tokens = 0

    @classmethod
    def simulate(
        cls,
        trace: Iterable[Mapping[ExpertKey, int]],
        catalog: Mapping[ExpertKey, Mapping[ExpertFormat, int]],
        budget_bytes: int,
    ) -> SimulationResult:
        policy = cls(budget_bytes, 1.0, 0.7, 0.3, 0, 0, catalog)
        resident: Dict[ExpertKey, ResidentGeneration] = {}
        epochs: List[Dict[str, object]] = []
        generation_counter = 0
        for counts in trace:
            tokens = sum(int(value) for value in counts.values())
            policy.observe(counts, tokens=tokens)
            candidates = set(counts.keys())
            plan = policy.plan(resident, candidates, 0, 0)
            policy.commit(plan)
            for intent in plan.transitions:
                generation_counter += 1
                resident[intent.key] = ResidentGeneration(
                    intent.target_format,
                    catalog[intent.key][intent.target_format],
                    generation_counter,
                )
            for key in plan.evictions:
                resident.pop(key, None)
            epochs.append(
                {
                    "epoch": plan.epoch,
                    "accounted_bytes": plan.accounted_bytes,
                    "targets": sorted(
                        (
                            [key.layer_id, key.expert_id, fmt.value]
                            for key, fmt in plan.targets.items()
                        ),
                    ),
                    "transitions": sorted(
                        [
                            [
                                intent.key.layer_id,
                                intent.key.expert_id,
                                intent.target_format.value,
                                intent.reserve_bytes,
                            ]
                            for intent in plan.transitions
                        ],
                    ),
                    "evictions": sorted(
                        [key.layer_id, key.expert_id] for key in plan.evictions
                    ),
                }
            )
        return SimulationResult(
            budget_bytes=int(budget_bytes), epochs=tuple(epochs)
        )
