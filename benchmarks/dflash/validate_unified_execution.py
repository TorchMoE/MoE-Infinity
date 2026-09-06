#!/usr/bin/env python3
"""Fail-closed local rollout gates for unified DFlash execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if os.environ.get("MKL_THREADING_LAYER") == "INTEL":
    os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.dflash.unified_execution_benchmark import run_tiny


def validate(
    *,
    require_cache_invariants: bool,
    require_order_invariance: bool,
    require_gpu: bool,
) -> tuple[dict[str, Any], bool]:
    benchmark = run_tiny()
    gpu_available = torch.cuda.is_available()
    gpu_fixture_enabled = os.environ.get("MOE_DFLASH_GPU") == "1"
    gpu_readiness_pass = gpu_available and gpu_fixture_enabled
    sampled_tvd_value = float(benchmark["sampled_tvd_value"])
    sampled_kl_value = float(benchmark["sampled_kl_value"])
    gates = {
        "cache_invariants": bool(benchmark["cache_invariants"]),
        "ownership_isolation": bool(benchmark["ownership_isolation"]),
        "order_invariance": bool(benchmark["rng_order_invariant"]),
        "sampled_tvd_value": sampled_tvd_value,
        "sampled_tvd_pass": sampled_tvd_value <= 0.10,
        "sampled_kl_value": sampled_kl_value,
        "sampled_kl_pass": sampled_kl_value <= 0.05,
        "pairing_executor_separate": set(
            benchmark["pairing_evidence"]
        ).isdisjoint({"wiring_reachable", "attempted_layers", "fired_layers"}),
        "paged_ownership_released": bool(
            benchmark["cancellation_released_pages"]
        ),
    }
    required = [
        gates["sampled_tvd_pass"],
        gates["sampled_kl_pass"],
        gates["pairing_executor_separate"],
        gates["paged_ownership_released"],
        gates["ownership_isolation"],
    ]
    if require_cache_invariants:
        required.append(gates["cache_invariants"])
    if require_order_invariance:
        required.append(gates["order_invariance"])
    if require_gpu:
        required.append(gpu_readiness_pass)

    passed = all(required)
    report: dict[str, Any] = {
        "status": "PASS" if passed else "FAIL",
        "fixture": "tiny",
        "checkpoint_downloads": False,
        "required_gpu_fixture": require_gpu,
        "gpu_readiness_required": require_gpu,
        "gpu_readiness_pass": gpu_readiness_pass,
        "gpu_gate_kind": "readiness only",
        "gpu_harness_executed": False,
        "gpu_harness_command": (
            "MOE_DFLASH_GPU=1 CUDA_VISIBLE_DEVICES=0 pytest -q "
            "tests/python/dflash/test_gpu_20b_dflash.py "
            "tests/python/dflash/test_gpu_serving_dflash.py -m gpu"
        ),
        "gpu_available": gpu_available,
        "gpu_fixture_enabled": gpu_fixture_enabled,
        **gates,
        "compatibility": {
            "pairing_evidence": benchmark["pairing_evidence"],
            "executor_evidence": benchmark["executor_evidence"],
            "execution_mode": benchmark["execution_mode"],
        },
        "trace_summary": {
            key: benchmark[key]
            for key in (
                "accepted_drafts",
                "committed_tokens",
                "sample_count",
                "round_count",
                "rollback_count",
                "replay_count",
                "per_request_rich_calls",
                "physical_rich_calls",
            )
        },
    }
    return report, passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=("tiny",), default="tiny")
    parser.add_argument("--require-cache-invariants", action="store_true")
    parser.add_argument("--require-order-invariance", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report, passed = validate(
        require_cache_invariants=args.require_cache_invariants,
        require_order_invariance=args.require_order_invariance,
        require_gpu=args.require_gpu,
    )
    print(
        json.dumps(report, sort_keys=True)
        if args.json
        else json.dumps(report, indent=2, sort_keys=True)
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
