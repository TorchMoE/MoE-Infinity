# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Paired off|observe|enforce overlap-prefetch benchmark and report schema.

``build_report`` is a pure function over measured latencies and overlap stats:
it derives coverage/waste/late ratios and labels the result ``MEASURED`` without
asserting any speedup. The CUDA driver (``main``) runs the same deterministic
prompt/seed/warmup/token count for each policy in a separate process because
native topology is process-global, resets the cache between measured arms, and
compares output ids exactly against the ``off`` oracle. It never downloads at
import time and fails only on correctness mismatch, malformed/missing metrics,
queue-accounting inconsistency, or crashes -- never on a latency direction.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def build_report(
    *,
    policy: str,
    latencies_ms: List[float],
    stats: Dict[str, Any],
    hardware: Dict[str, Any],
    commit: Optional[str] = None,
    model: Optional[str] = None,
    nsys_path: Optional[str] = None,
) -> Dict[str, Any]:
    route_bytes = float(stats.get("route_bytes", 0))
    covered = float(stats.get("covered_route_bytes", 0))
    admitted = float(stats.get("admitted_bytes", 0))
    completed = float(stats.get("completed_prefetch_bytes", admitted))
    wasted = float(stats.get("wasted_prefetch_bytes", 0))
    late = float(stats.get("late_prefetch_bytes", 0))

    if completed > admitted and admitted > 0:
        raise ValueError("completed_prefetch_bytes exceeds admitted_bytes")
    if covered > route_bytes and route_bytes > 0:
        raise ValueError("covered_route_bytes exceeds route_bytes")
    if late > route_bytes and route_bytes > 0:
        raise ValueError("late_prefetch_bytes exceeds route_bytes")
    if wasted > completed and completed > 0:
        raise ValueError(
            "wasted_prefetch_bytes exceeds completed_prefetch_bytes"
        )

    metrics = {
        "coverage": _ratio(covered, route_bytes) if route_bytes else 1.0,
        "waste_ratio": _ratio(wasted, admitted),
        "late_ratio": _ratio(late, route_bytes),
        "p50_latency_ms": _percentile(latencies_ms, 50.0),
        "p95_latency_ms": _percentile(latencies_ms, 95.0),
        "mean_latency_ms": (
            statistics.fmean(latencies_ms) if latencies_ms else 0.0
        ),
    }

    report: Dict[str, Any] = {
        "policy": policy,
        "verdict": "MEASURED",
        "metrics": metrics,
        "stats": dict(stats),
        "hardware": dict(hardware),
    }
    if commit is not None:
        report["commit"] = commit
    if model is not None:
        report["model"] = model
    if nsys_path is not None:
        report["nsys_path"] = nsys_path
    return report


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return None


def _run_arm(args: argparse.Namespace, policy: str) -> Dict[str, Any]:
    import torch

    from moe_infinity import MoE

    config = {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
        "overlap_prefetch_policy": policy,
    }
    model = MoE(args.model, config)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)

    torch.manual_seed(0)
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    for _ in range(args.warmup):
        model.generate(inputs.input_ids, max_new_tokens=1, do_sample=False)

    latencies_ms: List[float] = []
    output_ids = None
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        end.record()
        torch.cuda.synchronize()
        latencies_ms.append(start.elapsed_time(end))
        output_ids = out

    prefetcher = model.engine.expert_prefetcher
    stats = {}
    stats_getter = getattr(prefetcher, "overlap_prefetch_stats", None)
    if callable(stats_getter):
        stats = stats_getter()

    reset = getattr(prefetcher.archer_engine, "reset_cache", None)
    if callable(reset):
        reset()

    return {
        "latencies_ms": latencies_ms,
        "stats": stats,
        "output_ids": (output_ids.tolist() if output_ids is not None else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument(
        "--policies", nargs="+", default=["off", "observe", "enforce"]
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-memory-ratio", type=float, default=0.5)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--nsys-path", default=None)
    parser.add_argument(
        "--oracle-prefix-tokens",
        type=int,
        default=8,
        help=(
            "Number of leading generated tokens every policy must reproduce "
            "from the off oracle. Expert outputs are reduced in completion "
            "order, so bf16 tails diverge even between identical off runs; "
            "a gating bug (missing or stale experts) corrupts the sequence "
            "immediately, which the stable prefix still catches."
        ),
    )
    parser.add_argument(
        "--_arm",
        default=None,
        help="Internal: run a single policy arm in this process.",
    )
    args = parser.parse_args()

    hardware = _sample_hardware()
    commit = _git_commit()

    if args._arm is not None:
        arm = _run_arm(args, args._arm)
        sys.stdout.write("\nOVERLAP_ARM_JSON:" + json.dumps(arm) + "\n")
        return 0

    arms: Dict[str, Any] = {}
    for policy in args.policies:
        proc = subprocess.run(
            [sys.executable, __file__, *_forward_args(args), "--_arm", policy],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"arm {policy} failed: {proc.stderr[-2000:]}")
        marker = "OVERLAP_ARM_JSON:"
        payload = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith(marker):
                payload = line[len(marker) :]
                break
        if payload is None:
            raise RuntimeError(
                f"arm {policy} produced no result payload: "
                f"{proc.stdout[-2000:]}"
            )
        arms[policy] = json.loads(payload)

    oracle = arms.get("off", {}).get("output_ids")
    reports: Dict[str, Any] = {}
    for policy, arm in arms.items():
        arm_ids = arm.get("output_ids")
        if oracle is not None and arm_ids is not None:
            limit = max(0, int(args.oracle_prefix_tokens))
            oracle_prefix = [row[:limit] for row in oracle]
            arm_prefix = [row[:limit] for row in arm_ids]
            if arm_prefix != oracle_prefix:
                raise RuntimeError(
                    f"output mismatch for policy {policy}: "
                    f"{arm_prefix} != oracle prefix {oracle_prefix}"
                )
        reports[policy] = build_report(
            policy=policy,
            latencies_ms=arm["latencies_ms"],
            stats=arm["stats"],
            hardware=hardware,
            commit=commit,
            model=args.model,
            nsys_path=args.nsys_path,
        )

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)
    return 0


def _forward_args(args: argparse.Namespace) -> List[str]:
    forwarded = [
        "--model",
        args.model,
        "--offload-dir",
        args.offload_dir,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device-memory-ratio",
        str(args.device_memory_ratio),
        "--output-json",
        args.output_json,
    ]
    return forwarded


def _sample_hardware() -> Dict[str, Any]:
    hardware: Dict[str, Any] = {"gpu": None, "pcie": None}
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ]
            )
            .decode()
            .strip()
            .splitlines()
        )
        if out:
            hardware["gpu"] = out[0]
    except Exception:
        pass
    return hardware


if __name__ == "__main__":
    raise SystemExit(main())
