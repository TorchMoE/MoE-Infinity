# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import random

from benchmarks.adaptive_precision.bench_e2e import (
    MEASURED_REPETITIONS,
    MeasurementBundle,
    RepetitionTiming,
    derive_rows,
)
from benchmarks.adaptive_precision.bench_policy import (
    build_catalog,
    replay_adaptive_arm,
    replay_static_arm,
)
from benchmarks.adaptive_precision.bench_transfer import (
    CANONICAL_FORMAT,
    STATIC_LOW_FORMAT,
    aligned_bytes,
)
from benchmarks.adaptive_precision.report import (
    evaluate_release_gate,
    validate_run,
)
from moe_infinity.runtime.expert_precision import ExpertFormat


def _synthetic_bundle():
    rng = random.Random(1234)
    numel = 768 * 2048 * 3
    expert_numel = {
        (layer, expert): numel for layer in range(4) for expert in range(8)
    }
    hot = [(layer, expert) for layer in range(4) for expert in range(2)]
    all_keys = list(expert_numel.keys())
    routing_trace = []
    for _ in range(32):
        step_keys = []
        for layer in range(4):
            for _ in range(8):
                if rng.random() < 0.6:
                    cand = rng.choice([k for k in hot if k[0] == layer])
                else:
                    cand = rng.choice([k for k in all_keys if k[0] == layer])
                step_keys.append(cand)
        routing_trace.append(step_keys)
    bf16_all = len(all_keys) * aligned_bytes(numel, CANONICAL_FORMAT)
    budget = int(bf16_all * 0.7)
    seconds_per_byte = 1.0 / 20e9
    reps = []
    for _ in range(MEASURED_REPETITIONS):
        base = 6.0 + rng.uniform(-0.2, 0.2)
        samples = [base + rng.uniform(-0.3, 0.3) for _ in range(31)]
        reps.append(
            RepetitionTiming(
                ttft_ms=40.0,
                tpot_samples_ms=tuple(samples),
                decode_tokens=32,
                decode_wall_seconds=sum(samples) / 1000.0 + 0.006,
            )
        )
    return MeasurementBundle(
        model="synthetic-qwen3-moe",
        checkpoint_fingerprint="c" * 64,
        budget_bytes=budget,
        workload_meta={
            "prompt_tokens": 64,
            "decode_tokens": 32,
            "batch_size": 1,
            "seed": 7,
            "workload_sha256": "d" * 64,
        },
        hardware={
            "device_name": "synthetic",
            "compute_capability": [12, 0],
            "peak_torch_allocated_bytes": 123,
        },
        software={
            "torch": "x",
            "cuda": "13.0",
            "python": "3.13",
            "commit": "runtime",
        },
        expert_numel=expert_numel,
        routing_trace=routing_trace,
        per_format_seconds_per_byte={
            CANONICAL_FORMAT.value: seconds_per_byte,
            STATIC_LOW_FORMAT.value: seconds_per_byte,
        },
        repetitions=reps,
        greedy_reference_tokens=[rng.randrange(0, 1000) for _ in range(32)],
    )


def test_derive_rows_emits_five_valid_rows_per_arm():
    rows = derive_rows(_synthetic_bundle())
    assert len(rows) == 3 * MEASURED_REPETITIONS
    for row in rows:
        validate_run(row)
    for mode in ("canonical", "static_low", "adaptive"):
        assert sum(1 for row in rows if row["mode"] == mode) == 5


def test_adaptive_h2d_between_static_low_and_canonical():
    bundle = _synthetic_bundle()
    catalog = build_catalog(bundle.expert_numel)
    canonical = replay_static_arm(
        bundle.routing_trace, catalog, CANONICAL_FORMAT, bundle.budget_bytes
    )
    static_low = replay_static_arm(
        bundle.routing_trace, catalog, STATIC_LOW_FORMAT, bundle.budget_bytes
    )
    adaptive = replay_adaptive_arm(
        bundle.routing_trace, catalog, bundle.budget_bytes
    )
    assert (
        static_low.h2d_payload_bytes
        <= adaptive.h2d_payload_bytes
        <= canonical.h2d_payload_bytes
    )
    for arm in (canonical, static_low, adaptive):
        assert arm.peak_accounted_bytes <= bundle.budget_bytes
        assert not arm.fallback_counts


def test_release_gate_passes_on_derived_rows():
    rows = derive_rows(_synthetic_bundle())
    report = evaluate_release_gate(rows)
    assert report["release_gate"] == "pass"
    assert report["reasons"] == []
    assert len(report["quality_attestation_sha256"]) == 64


def test_static_low_never_charges_bf16_when_fp8_available():
    bundle = _synthetic_bundle()
    catalog = build_catalog(bundle.expert_numel)
    static_low = replay_static_arm(
        bundle.routing_trace, catalog, STATIC_LOW_FORMAT, bundle.budget_bytes
    )
    fp8_expert = aligned_bytes(
        next(iter(bundle.expert_numel.values())), ExpertFormat.FP8_E4M3_BLOCK128
    )
    assert static_low.h2d_payload_bytes % fp8_expert == 0
