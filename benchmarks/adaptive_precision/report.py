from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

REQUIRED = {
    "mode",
    "model",
    "checkpoint_fingerprint",
    "format",
    "converter_version",
    "quality_attestation_sha256",
    "hardware",
    "software",
    "workload",
    "quality",
    "memory",
    "transfer",
    "latency",
    "throughput",
    "policy",
}


def validate_run(row: dict) -> None:
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f"missing benchmark fields: {sorted(missing)}")
    if (
        len(row["checkpoint_fingerprint"]) != 64
        or len(row["quality_attestation_sha256"]) != 64
    ):
        raise ValueError("invalid digest")
    if row["memory"]["peak_accounted_bytes"] > row["memory"]["budget_bytes"]:
        raise ValueError("adaptive budget exceeded")
    for key in ("tpot_ms_p50", "tpot_ms_p90", "tpot_ms_p99"):
        if not math.isfinite(float(row["latency"][key])):
            raise ValueError("nonfinite TPOT")


def evaluate_release_gate(rows: list[dict]) -> dict:
    for row in rows:
        validate_run(row)
    reasons = []
    by_mode = {
        mode: [row for row in rows if row["mode"] == mode]
        for mode in ("canonical", "static_low", "adaptive")
    }
    if any(len(group) != 5 for group in by_mode.values()):
        reasons.append("five_repetitions_required")
    adaptive = by_mode["adaptive"]
    if any(row["policy"].get("fallback_counts") for row in adaptive):
        reasons.append("adaptive_fallback")
    if adaptive and by_mode["canonical"] and by_mode["static_low"]:
        median_h2d = statistics.median(
            row["transfer"]["h2d_payload_bytes"] for row in adaptive
        )
        canonical_h2d = statistics.median(
            row["transfer"]["h2d_payload_bytes"] for row in by_mode["canonical"]
        )
        if median_h2d > canonical_h2d:
            reasons.append("h2d_regression")
        adaptive_tpot = statistics.median(
            row["latency"]["tpot_ms_p50"] for row in adaptive
        )
        reference_tpot = min(
            statistics.median(
                row["latency"]["tpot_ms_p50"] for row in by_mode[mode]
            )
            for mode in ("canonical", "static_low")
        )
        if adaptive_tpot > reference_tpot * 1.05:
            reasons.append("tpot_regression")
    return {
        "release_gate": "pass" if not reasons else "fail",
        "reasons": reasons,
        "quality_attestation_sha256": adaptive[0]["quality_attestation_sha256"]
        if adaptive
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text().splitlines()
        if line
    ]
    for row in rows:
        validate_run(row)
    report = evaluate_release_gate(rows)
    report["runs"] = len(rows)
    Path(args.output).write_text(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
