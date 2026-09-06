from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{ROOT}{os.pathsep}{existing}" if existing else str(ROOT)
    )
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    return env


def test_dry_run_isolates_modes_and_reports_geometry(tmp_path: Path) -> None:
    output = tmp_path / "prefix.json"
    proc = subprocess.run(
        [
            sys.executable,
            "benchmarks/serving/prefix_cache_benchmark.py",
            "--dry-run",
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output.read_text())
    assert list(payload["modes"]) == [
        "disabled",
        "enabled_cold",
        "enabled_warm",
    ]
    ids = [
        payload["modes"][name]["engine_instance_id"]
        for name in payload["modes"]
    ]
    assert len(set(ids)) == 3
    assert payload["modes"]["disabled"]["prefix_cache_active"] is False
    assert payload["modes"]["enabled_cold"]["hits_total"] == 0
    assert payload["modes"]["enabled_warm"]["hits_total"] > 0
    warm = payload["modes"]["enabled_warm"]
    assert warm["geometry"]["query_offsets"][-1] == sum(
        warm["geometry"]["query_lengths"]
    )
    assert (
        warm["geometry"]["context_lengths"][0]
        + warm["geometry"]["query_lengths"][0]
        == warm["geometry"]["kv_seq_lengths"][0]
    )
    assert (
        warm["geometry"]["query_lengths"][0]
        < warm["geometry"]["kv_seq_lengths"][0]
    )
    assert warm["refcount_high_water"] >= 2
    assert payload["correctness"] == {
        "token_digests_equal": True,
        "logit_digests_equal": True,
    }


def test_dry_run_aborts_on_digest_mismatch(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "benchmarks/serving/prefix_cache_benchmark.py",
            "--dry-run",
            "--dry-run-force-mismatch",
            "--output-json",
            str(tmp_path / "bad.json"),
        ],
        cwd=ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "disabled/cold/warm digest mismatch" in proc.stderr
    assert not (tmp_path / "bad.json").exists()
