# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Deterministic CPU policy trace replay.

Replays a measured expert-routing trace through the production
:class:`AdaptivePrecisionPolicy` under a fixed HBM budget and a per-expert
byte catalog, producing the resident-format decisions and the resulting H2D
payload that a benchmark arm would incur. The replay is pure and
CUDA-free so the adaptive arm's byte accounting is reproducible and testable
without a GPU; only :mod:`bench_e2e` and :mod:`bench_transfer` touch the
device.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from benchmarks.adaptive_precision.bench_transfer import (
    CANONICAL_FORMAT,
    STATIC_LOW_FORMAT,
    aligned_bytes,
)
from moe_infinity.memory.adaptive_precision_policy import (
    AdaptivePrecisionPolicy,
    ExpertKey,
)
from moe_infinity.runtime.expert_precision import (
    ExpertFormat,
    ResidentGeneration,
)

__all__ = [
    "AdaptivePrecisionPolicy",
    "ExpertCatalogEntry",
    "ArmReplay",
    "build_catalog",
    "replay_static_arm",
    "replay_adaptive_arm",
]


@dataclass(frozen=True)
class ExpertCatalogEntry:
    """Per-expert weight count and its byte sizes for each stored format."""

    numel: int
    format_bytes: Mapping[ExpertFormat, int]


@dataclass(frozen=True)
class ArmReplay:
    """Resident/transfer accounting produced by replaying one arm."""

    h2d_payload_bytes: int
    h2d_transfers: int
    peak_accounted_bytes: int
    per_step_fetched_bytes: Tuple[int, ...]
    fallback_counts: Mapping[str, int] = field(default_factory=dict)
    promotions: int = 0
    demotions: int = 0


def build_catalog(
    expert_numel: Mapping[Tuple[int, int], int],
    formats: Sequence[ExpertFormat] = (CANONICAL_FORMAT, STATIC_LOW_FORMAT),
) -> Dict[ExpertKey, Dict[ExpertFormat, int]]:
    """Turn measured per-expert weight counts into a policy byte catalog."""
    catalog: Dict[ExpertKey, Dict[ExpertFormat, int]] = {}
    for (layer_id, expert_id), numel in expert_numel.items():
        key = ExpertKey(int(layer_id), int(expert_id))
        catalog[key] = {fmt: aligned_bytes(int(numel), fmt) for fmt in formats}
    return catalog


def _fetched_keys_per_step(
    trace: Sequence[Sequence[Tuple[int, int]]],
) -> List[List[ExpertKey]]:
    return [
        [
            ExpertKey(int(layer_id), int(expert_id))
            for layer_id, expert_id in step
        ]
        for step in trace
    ]


def _replay_lru_cache(
    trace: Sequence[Sequence[Tuple[int, int]]],
    catalog: Mapping[ExpertKey, Mapping[ExpertFormat, int]],
    budget_bytes: int,
    format_for: "Mapping[ExpertKey, ExpertFormat] | None",
    default_format: ExpertFormat,
    *,
    promotions: int = 0,
    demotions: int = 0,
) -> ArmReplay:
    """Replay one arm as a fixed-capacity LRU expert cache under the budget.

    All arms share this mechanic and differ only in which stored representation
    ``format_for`` assigns each expert (a constant format for canonical and
    static-low; the policy-derived hot/cold mix for adaptive). On a cold miss
    the expert is fetched in its assigned format and made resident, evicting the
    least-recently used experts first so the resident working set never exceeds
    the budget; an evicted-then-reactivated expert is re-fetched. The H2D
    payload sums every real fetch and peak accounted bytes is the budget-bounded
    resident high-water mark, so a smaller mix both holds more experts and
    refetches less under the same trace and budget.
    """
    from collections import OrderedDict

    steps = _fetched_keys_per_step(trace)
    resident: "OrderedDict[ExpertKey, int]" = OrderedDict()
    per_step: List[int] = []
    total_payload = 0
    transfers = 0
    peak = 0
    fallback: Dict[str, int] = {}
    for step_keys in steps:
        step_bytes = 0
        for key in step_keys:
            if key in resident:
                resident.move_to_end(key)
                continue
            sizes = catalog.get(key)
            if not sizes:
                fallback["missing_variant"] = (
                    fallback.get("missing_variant", 0) + 1
                )
                continue
            fmt = default_format
            if format_for is not None and key in format_for:
                fmt = format_for[key]
            size = sizes.get(fmt)
            if size is None:
                size = min(sizes.values())
            if size > budget_bytes:
                fallback["budget_rejected_active_expert"] = (
                    fallback.get("budget_rejected_active_expert", 0) + 1
                )
                continue
            while resident and sum(resident.values()) + size > budget_bytes:
                resident.popitem(last=False)
            resident[key] = size
            step_bytes += size
            transfers += 1
        peak = max(peak, sum(resident.values()))
        per_step.append(step_bytes)
        total_payload += step_bytes
    return ArmReplay(
        h2d_payload_bytes=total_payload,
        h2d_transfers=transfers,
        peak_accounted_bytes=peak,
        per_step_fetched_bytes=tuple(per_step),
        fallback_counts=fallback,
        promotions=promotions,
        demotions=demotions,
    )


def replay_static_arm(
    trace: Sequence[Sequence[Tuple[int, int]]],
    catalog: Mapping[ExpertKey, Mapping[ExpertFormat, int]],
    fmt: ExpertFormat,
    budget_bytes: int,
) -> ArmReplay:
    """Replay a fixed-format arm (canonical BF16 or static-low FP8)."""
    return _replay_lru_cache(trace, catalog, budget_bytes, None, fmt)


def adaptive_format_map(
    trace: Sequence[Sequence[Tuple[int, int]]],
    catalog: Mapping[ExpertKey, Mapping[ExpertFormat, int]],
    budget_bytes: int,
    *,
    decay: float = 0.95,
    promotion_threshold: float = 0.70,
    demotion_threshold: float = 0.30,
    promote_cooldown: int = 2,
    demote_cooldown: int = 2,
    epoch_tokens: int = 1,
) -> Tuple[Dict[ExpertKey, ExpertFormat], int, int]:
    """Learn each expert's resident format from the production policy.

    Drives the real :class:`AdaptivePrecisionPolicy` over the trace and records
    the highest-quality (largest) format the policy ever made resident for each
    expert; experts the policy never promoted default to the smallest format.
    Returns the format map and the observed promotion/demotion counts.
    """
    steps = _fetched_keys_per_step(trace)
    policy = AdaptivePrecisionPolicy(
        budget_bytes,
        decay,
        promotion_threshold,
        demotion_threshold,
        promote_cooldown,
        demote_cooldown,
        catalog,
        epoch_tokens=epoch_tokens,
    )
    resident: Dict[ExpertKey, ResidentGeneration] = {}
    generation = 0
    promotions = 0
    demotions = 0
    best_format: Dict[ExpertKey, ExpertFormat] = {}
    for step_keys in steps:
        counts: Dict[ExpertKey, int] = {}
        for key in step_keys:
            counts[key] = counts.get(key, 0) + 1
        tokens = sum(counts.values()) or 1
        policy.observe(counts, tokens=tokens)
        candidates = {key for key in counts if key in catalog}
        plan = policy.plan(resident, candidates, 0, 0)
        policy.commit(plan)
        for intent in plan.transitions:
            size = catalog.get(intent.key, {}).get(intent.target_format)
            if size is None:
                continue
            prior = resident.get(intent.key)
            if prior is None:
                promotions += 1
            elif intent.target_format != prior.format:
                if _rank(intent.target_format) < _rank(prior.format):
                    promotions += 1
                else:
                    demotions += 1
            generation += 1
            resident[intent.key] = ResidentGeneration(
                intent.target_format, size, generation
            )
            current = best_format.get(intent.key)
            if current is None or _rank(intent.target_format) < _rank(current):
                best_format[intent.key] = intent.target_format
        for key in plan.evictions:
            resident.pop(key, None)
    return best_format, promotions, demotions


def replay_adaptive_arm(
    trace: Sequence[Sequence[Tuple[int, int]]],
    catalog: Mapping[ExpertKey, Mapping[ExpertFormat, int]],
    budget_bytes: int,
    *,
    decay: float = 0.95,
    promotion_threshold: float = 0.70,
    demotion_threshold: float = 0.30,
    promote_cooldown: int = 2,
    demote_cooldown: int = 2,
    epoch_tokens: int = 1,
    static_low_format: ExpertFormat = STATIC_LOW_FORMAT,
) -> ArmReplay:
    """Replay the adaptive arm: policy-chosen formats in the shared LRU cache.

    The format each expert receives comes from the real policy
    (:func:`adaptive_format_map`): hot experts the policy promoted keep their
    higher-quality (larger) representation while every other expert defaults to
    the smallest format. Those assignments then run through the same
    budget-bounded LRU cache as the static arms, so the adaptive arm's H2D and
    peak bytes lie between static-low and canonical by construction and never
    exceed canonical under the same trace.
    """
    best_format, promotions, demotions = adaptive_format_map(
        trace,
        catalog,
        budget_bytes,
        decay=decay,
        promotion_threshold=promotion_threshold,
        demotion_threshold=demotion_threshold,
        promote_cooldown=promote_cooldown,
        demote_cooldown=demote_cooldown,
        epoch_tokens=epoch_tokens,
    )
    return _replay_lru_cache(
        trace,
        catalog,
        budget_bytes,
        best_format,
        static_low_format,
        promotions=promotions,
        demotions=demotions,
    )


def _rank(fmt: ExpertFormat) -> int:
    from moe_infinity.memory.adaptive_precision_policy import _quality_rank

    return _quality_rank(fmt)


def _load_trace(path: Path) -> List[List[Tuple[int, int]]]:
    trace: List[List[Tuple[int, int]]] = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        trace.append([(int(a), int(b)) for a, b in row])
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace = _load_trace(Path(args.trace))
    catalog_doc = json.loads(Path(args.catalog).read_text())
    expert_numel = {
        (int(entry["layer_id"]), int(entry["expert_id"])): int(entry["numel"])
        for entry in catalog_doc
    }
    catalog = build_catalog(expert_numel)
    canonical = replay_static_arm(
        trace, catalog, CANONICAL_FORMAT, args.budget_bytes
    )
    static_low = replay_static_arm(
        trace, catalog, STATIC_LOW_FORMAT, args.budget_bytes
    )
    adaptive = replay_adaptive_arm(trace, catalog, args.budget_bytes)
    Path(args.output).write_text(
        json.dumps(
            {
                "canonical": canonical.__dict__
                | {
                    "per_step_fetched_bytes": list(
                        canonical.per_step_fetched_bytes
                    )
                },
                "static_low": static_low.__dict__
                | {
                    "per_step_fetched_bytes": list(
                        static_low.per_step_fetched_bytes
                    )
                },
                "adaptive": adaptive.__dict__
                | {
                    "per_step_fetched_bytes": list(
                        adaptive.per_step_fetched_bytes
                    )
                },
            },
            sort_keys=True,
            default=list,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
