#!/usr/bin/env python3
"""No-download DFlash unified-execution benchmark and evidence reporter.

The ``tiny`` fixture measures CPU protocol operations. It does not estimate
real-checkpoint throughput and leaves unsupported production capabilities
explicit rather than inventing values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if os.environ.get("MKL_THREADING_LAYER") == "INTEL":
    os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moe_infinity.serving.mla_cache import MLAPagedKVCache
from moe_infinity.serving.spec_cache_adapter import PagedCacheAdapter
from moe_infinity.spec_decode._dflash_sample_ops import acceptance_sampled
from moe_infinity.spec_decode.protocols import ExecutorEvidence, PairingEvidence


def _milliseconds(start: float) -> float:
    return (time.perf_counter() - start) * 1_000.0


def _distribution(counts: Counter[int], samples: int, size: int) -> list[float]:
    return [counts[index] / samples for index in range(size)]


def _tvd(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def _kl(left: list[float], right: list[float]) -> float:
    epsilon = 1e-12
    return sum(
        a * math.log((a + epsilon) / (b + epsilon))
        for a, b in zip(left, right)
        if a > 0
    )


@dataclass(frozen=True)
class SampledLawMeasurements:
    tvd_value: float
    kl_value: float
    sample_count: int
    round_count: int
    accepted_drafts: int
    committed_tokens: int
    rollback_count: int
    elapsed_seconds: float


def _sampled_law(samples: int = 2_000) -> SampledLawMeasurements:
    proposal = torch.tensor([[0.70, 0.30]], dtype=torch.float64)
    target = torch.tensor([[0.25, 0.75], [0.60, 0.40]], dtype=torch.float64)
    speculative: Counter[int] = Counter()
    reference: Counter[int] = Counter()
    accepted = 0
    committed = 0
    rollback_count = 0
    round_count = 0
    start = time.perf_counter()
    for seed in range(samples):
        spec_generator = torch.Generator().manual_seed(seed)
        draft = torch.multinomial(
            proposal[0], 1, generator=spec_generator
        ).reshape(1)
        decision = acceptance_sampled(
            proposal, target, draft, generator=spec_generator
        )
        token = int(draft[0]) if decision.accept else int(decision.final_token)
        speculative[token] += 1
        accepted += int(decision.accept)
        committed += 1 + int(decision.accept)
        rollback_count += int(not decision.accept)
        round_count += 1

        reference_generator = torch.Generator().manual_seed(100_000 + seed)
        reference[
            int(torch.multinomial(target[0], 1, generator=reference_generator))
        ] += 1
    elapsed = time.perf_counter() - start
    spec_dist = _distribution(speculative, samples, 2)
    ref_dist = _distribution(reference, samples, 2)
    return SampledLawMeasurements(
        tvd_value=_tvd(spec_dist, ref_dist),
        kl_value=_kl(spec_dist, ref_dist),
        sample_count=sum(speculative.values()),
        round_count=round_count,
        accepted_drafts=accepted,
        committed_tokens=committed,
        rollback_count=rollback_count,
        elapsed_seconds=elapsed,
    )


def _order_invariance() -> bool:
    seeds = {"a": 17, "b": 29, "c": 41}

    def run(order: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
        generators = {
            name: torch.Generator().manual_seed(seed)
            for name, seed in seeds.items()
        }
        rows: dict[str, list[int]] = {name: [] for name in seeds}
        probabilities = torch.tensor([0.2, 0.3, 0.5])
        for _ in range(16):
            for name in order:
                rows[name].append(
                    int(
                        torch.multinomial(
                            probabilities, 1, generator=generators[name]
                        )
                    )
                )
        return {name: tuple(tokens) for name, tokens in rows.items()}

    return run(("a", "b", "c")) == run(("c", "a", "b"))


def _cache_evidence() -> dict[str, Any]:
    cache = MLAPagedKVCache(
        num_blocks=16,
        block_size=2,
        num_layers=1,
        latent_dim=2,
        rope_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    first = PagedCacheAdapter(cache, seq_id=1, initial_length=3)
    second = PagedCacheAdapter(cache, seq_id=2, initial_length=2)
    second_pages = tuple(cache.get_block_table(2))
    snapshot = first.snapshot()
    first.append(4)
    peak = len(cache.get_block_table(1)) + len(cache.get_block_table(2))
    first.truncate(5)
    first.restore(snapshot)
    isolated = tuple(cache.get_block_table(2)) == second_pages
    invariant = first.logical_length() == 3 and second.logical_length() == 2

    cancel_start = time.perf_counter()
    first.release()
    cancellation_latency = _milliseconds(cancel_start)
    try:
        cache.get_block_table(1)
        released = False
    except KeyError:
        released = True
    second.release()
    return {
        "cache_pages_peak": peak,
        "cache_invariants": invariant,
        "ownership_isolation": isolated,
        "cancellation_released_pages": released,
        "cancellation_latency_ms": cancellation_latency,
        "preemption_policy": "resident-only no-preempt for paged MLA",
        "preemption_recovery": "not exercised: swap_out/swap_in intentionally return false",
    }


def run_tiny() -> dict[str, Any]:
    prompt = torch.arange(32, dtype=torch.float32)
    weights = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
    prefill_start = time.perf_counter()
    for _ in range(64):
        _ = prompt @ weights
    prefill_ms = _milliseconds(prefill_start) / 64.0

    sampled = _sampled_law()
    verify_start = time.perf_counter()
    for seed in range(128):
        generator = torch.Generator().manual_seed(seed)
        _ = acceptance_sampled(
            torch.tensor([[0.7, 0.3]]),
            torch.tensor([[0.25, 0.75], [0.6, 0.4]]),
            torch.tensor([seed % 2]),
            generator=generator,
        )
    verify_ms = _milliseconds(verify_start) / 128.0

    cache = _cache_evidence()
    pairing = PairingEvidence(
        failure_reason="tiny fixture has no checkpoint pair"
    )
    executor = ExecutorEvidence(
        fallback_reason="tiny fixture has no expert executor"
    )
    report: dict[str, Any] = {
        "fixture": "tiny",
        "measurement_scope": "synthetic no-checkpoint CPU fixture",
        "prefill_latency_ms": prefill_ms,
        "verify_latency_ms": verify_ms,
        "decode_elapsed_seconds": sampled.elapsed_seconds,
        "decode_committed_tokens_per_second": (
            sampled.committed_tokens / sampled.elapsed_seconds
        ),
        "sample_count": sampled.sample_count,
        "round_count": sampled.round_count,
        "accepted_drafts": sampled.accepted_drafts,
        "committed_tokens": sampled.committed_tokens,
        "rollback_count": sampled.rollback_count,
        "replay_count": 0,
        "rng_order_invariant": _order_invariance(),
        "sampled_tvd_value": sampled.tvd_value,
        "sampled_kl_value": sampled.kl_value,
        "metric_units": {
            "prefill_latency_ms": "milliseconds per prefill operation",
            "verify_latency_ms": "milliseconds per verify operation",
            "decode_elapsed_seconds": "seconds",
            "decode_committed_tokens_per_second": "committed tokens per second",
            "sampled_tvd_value": "dimensionless",
            "sampled_kl_value": "nats",
            "cache_pages_peak": "pages",
            "cancellation_latency_ms": "milliseconds",
        },
        "execution_mode": "tiny_cpu_protocol_fixture",
        "pairing_evidence": pairing.as_dict(),
        "executor_evidence": executor.as_dict(),
        "route_attempted_layers": [],
        "route_fired_layers": [],
        "per_request_rich_calls": 0,
        "physical_rich_calls": 0,
        **cache,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=("tiny",), default="tiny")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_tiny()
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
