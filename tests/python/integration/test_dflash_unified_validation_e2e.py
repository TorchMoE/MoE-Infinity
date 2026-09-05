from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = ROOT / "benchmarks/dflash/validate_unified_execution.py"


def _run_validator(
    working_directory: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--fixture", "tiny", *arguments],
        cwd=working_directory,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "MOE_DFLASH_GPU": "0"},
    )


def _report(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    parsed = json.loads(completed.stdout)
    assert isinstance(parsed, dict)
    return parsed


def test_validator_passes_cpu_rollout_gates_from_non_repo_cwd(
    tmp_path: Path,
) -> None:
    completed = _run_validator(
        tmp_path,
        "--require-cache-invariants",
        "--require-order-invariance",
        "--json",
    )

    assert completed.returncode == 0, completed.stderr
    report = _report(completed)
    assert report["status"] == "PASS"
    assert (
        report["compatibility"]["execution_mode"] == "tiny_cpu_protocol_fixture"
    )
    assert report["cache_invariants"] is True
    assert report["order_invariance"] is True
    assert report["ownership_isolation"] is True
    assert report["paged_ownership_released"] is True
    assert report["pairing_executor_separate"] is True
    assert report["checkpoint_downloads"] is False
    assert report["gpu_harness_executed"] is False


def test_validator_require_gpu_fails_closed_when_fixture_is_disabled(
    tmp_path: Path,
) -> None:
    completed = _run_validator(tmp_path, "--require-gpu", "--json")

    assert completed.returncode == 1, completed.stderr
    report = _report(completed)
    assert report["status"] == "FAIL"
    assert report["gpu_readiness_required"] is True
    assert report["gpu_fixture_enabled"] is False
    assert report["gpu_readiness_pass"] is False
    assert report["checkpoint_downloads"] is False
    assert report["gpu_harness_executed"] is False
