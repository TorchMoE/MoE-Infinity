import json
from pathlib import Path

from benchmarks.serving.adaptive_memory_trace import compare_policies, replay


def trace_row(
    step: int, *, device_id: int, expert_heavy: bool
) -> dict[str, int | float | bool]:
    return {
        "device_id": device_id,
        "step": step,
        "expert_misses": 80 if expert_heavy else 0,
        "expert_accesses": 100,
        "expert_fetch_stall_ms": 40.0 if expert_heavy else 0.0,
        "kv_used_blocks": 20 if expert_heavy else 63,
        "kv_total_blocks": 64,
        "kv_swap_bytes": 0 if expert_heavy else 64 * 1024**2,
        "kv_swap_stall_ms": 0.0 if expert_heavy else 30.0,
        "kv_preemptions": 0 if expert_heavy else 2,
        "free_gpu_bytes": 2 * 1024**3,
        "kv_supported": True,
        "total_bytes": 8 * 1024**3,
        "model_bytes": 2 * 1024**3,
        "activation_reserve_bytes": 1024**3,
        "kv_block_bytes": 16 * 1024**2,
        "current_expert_bytes": 3 * 1024**3,
        "current_kv_blocks": 64,
    }


def alternating_pressure_trace(
    *, steps: int
) -> list[dict[str, int | float | bool]]:
    return [
        trace_row(step, device_id=0, expert_heavy=(step // 128) % 2 == 0)
        for step in range(1, steps + 1)
    ]


def mixed_trace() -> list[dict[str, int | float | bool]]:
    rows: list[dict[str, int | float | bool]] = []
    for step in range(1, 257):
        rows.append(trace_row(step, device_id=0, expert_heavy=step <= 128))
        row = trace_row(step, device_id=1, expert_heavy=step > 128)
        row["total_bytes"] = 12 * 1024**3
        rows.append(row)
    return rows


def write_trace(
    tmp_path: Path, rows: list[dict[str, int | float | bool]]
) -> Path:
    path = tmp_path / "adaptive-memory.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_same_trace_produces_identical_decisions(tmp_path: Path) -> None:
    trace = write_trace(tmp_path, alternating_pressure_trace(steps=1024))
    a = replay(trace, policy="adaptive", seed=7)
    b = replay(trace, policy="adaptive", seed=7)
    assert a["decisions"] == b["decisions"]
    assert a["hard_budget_violations"] == 0


def test_stability_gate_limits_resize_frequency(tmp_path: Path) -> None:
    report = replay(
        write_trace(tmp_path, alternating_pressure_trace(steps=1024)),
        policy="adaptive",
        seed=7,
    )
    assert report["resize_count"] <= 1024 // report["cooldown_steps"] + 1
    assert report["minimum_capacity_violations"] == 0


def test_report_compares_without_claiming_gain(tmp_path: Path) -> None:
    report = compare_policies(write_trace(tmp_path, mixed_trace()))
    assert set(report["policies"]) == {"fixed", "adaptive"}
    assert "expected_improvement" not in report
