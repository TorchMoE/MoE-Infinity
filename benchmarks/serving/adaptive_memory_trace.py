from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from moe_infinity.memory.adaptive_memory import (
    AdaptiveMemoryConfig,
    AdaptiveMemoryController,
    MemorySignals,
    ResizeDirection,
    ResizeOutcome,
    ResizeResult,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"trace line {line_number} must be an object")
        rows.append(value)
    return rows


def replay(path: Path, *, policy: str, seed: int) -> dict[str, Any]:
    if policy not in {"fixed", "adaptive"}:
        raise ValueError("policy must be 'fixed' or 'adaptive'")
    config = AdaptiveMemoryConfig(enabled=policy == "adaptive")
    controller = AdaptiveMemoryController(config)
    targets: dict[int, tuple[int, int]] = {}
    initial_kv: dict[int, int] = {}
    decisions: list[dict[str, Any]] = []
    per_device: dict[int, dict[str, Any]] = {}
    hard_budget_violations = 0
    minimum_capacity_violations = 0

    for row in _rows(path):
        device_id = int(row["device_id"])
        current = targets.setdefault(
            device_id,
            (int(row["current_expert_bytes"]), int(row["current_kv_blocks"])),
        )
        initial_kv.setdefault(device_id, current[1])
        signals = MemorySignals(
            device_id=device_id,
            step=int(row["step"]),
            expert_misses=int(row["expert_misses"]),
            expert_accesses=int(row["expert_accesses"]),
            expert_fetch_stall_ms=float(row["expert_fetch_stall_ms"]),
            kv_used_blocks=int(row["kv_used_blocks"]),
            kv_total_blocks=int(row["kv_total_blocks"]),
            kv_swap_bytes=int(row["kv_swap_bytes"]),
            kv_swap_stall_ms=float(row["kv_swap_stall_ms"]),
            kv_preemptions=int(row["kv_preemptions"]),
            free_gpu_bytes=int(row["free_gpu_bytes"]),
            kv_supported=bool(row.get("kv_supported", True)),
        )
        controller.observe(signals)
        proposed = controller.propose(
            device_id=device_id,
            step=signals.step,
            total_bytes=int(row["total_bytes"]),
            model_bytes=int(row["model_bytes"]),
            activation_reserve_bytes=int(row["activation_reserve_bytes"]),
            kv_block_bytes=int(row["kv_block_bytes"]),
            current_expert_bytes=current[0],
            current_kv_blocks=current[1],
            kv_supported=signals.kv_supported,
        )
        changed = (proposed.expert_bytes, proposed.kv_blocks) != current
        if policy == "adaptive" and changed:
            result = ResizeResult(
                device_id,
                ResizeOutcome.COMMITTED,
                proposed.expert_bytes,
                proposed.kv_blocks,
                proposed.reason,
                proposed.kv_supported,
            )
            controller.record_resize(result, step=signals.step)
            targets[device_id] = (result.expert_bytes, result.kv_blocks)
            decisions.append(
                {
                    "device_id": device_id,
                    "step": signals.step,
                    "direction": proposed.direction.value,
                    "reason": proposed.reason,
                    "expert_bytes": result.expert_bytes,
                    "kv_blocks": result.kv_blocks,
                }
            )

        expert_bytes, kv_blocks = targets[device_id]
        block_bytes = int(row["kv_block_bytes"])
        hard_budget = max(
            0,
            int(row["total_bytes"])
            - int(row["model_bytes"])
            - int(row["activation_reserve_bytes"])
            - config.free_memory_reserve_bytes,
        )
        hard_budget_violations += int(
            expert_bytes + kv_blocks * block_bytes > hard_budget
        )
        minimum_capacity_violations += int(
            expert_bytes < min(config.min_expert_cache_bytes, current[0])
            or kv_blocks
            < min(config.min_kv_cache_blocks, initial_kv[device_id])
        )
        state = per_device.setdefault(
            device_id,
            {
                "expert_fetch_stall_ms": 0.0,
                "kv_swap_stall_ms": 0.0,
                "kv_preemptions": 0,
            },
        )
        state["expert_fetch_stall_ms"] += signals.expert_fetch_stall_ms
        state["kv_swap_stall_ms"] += signals.kv_swap_stall_ms
        state["kv_preemptions"] += signals.kv_preemptions

    for device_id, final in targets.items():
        per_device[device_id]["final_expert_bytes"] = final[0]
        per_device[device_id]["final_kv_blocks"] = final[1]
        per_device[device_id]["decisions"] = [
            item for item in decisions if item["device_id"] == device_id
        ]

    return {
        "policy": policy,
        "seed": seed,
        "cooldown_steps": config.cooldown_steps,
        "decisions": decisions,
        "resize_count": len(decisions),
        "hard_budget_violations": hard_budget_violations,
        "minimum_capacity_violations": minimum_capacity_violations,
        "per_device": per_device,
    }


def compare_policies(path: Path, *, seed: int = 7) -> dict[str, Any]:
    return {
        "seed": seed,
        "policies": {
            policy: replay(path, policy=policy, seed=seed)
            for policy in ("fixed", "adaptive")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--policies", nargs="+", choices=("fixed", "adaptive"), required=True
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "seed": args.seed,
        "policies": {
            policy: replay(args.trace, policy=policy, seed=args.seed)
            for policy in args.policies
        },
    }
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
