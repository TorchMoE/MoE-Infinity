from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "benchmarks" / "serving" / "kv_offload_benchmark.py"
SPEC = importlib.util.spec_from_file_location("kv_offload_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_percentiles_use_median_and_nearest_rank() -> None:
    assert benchmark.percentiles([1.0, 2.0, 3.0, 4.0]) == {
        "p50": 2.5,
        "p95": 4.0,
        "p99": 4.0,
    }


def test_parser_accepts_complete_kv_swap_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kv_offload_benchmark.py",
            "--model",
            "model",
            "--offload-dir",
            "/tmp/offload",
            "--kv-swap-mode",
            "async",
            "--warmup-requests",
            "3",
            "--trials",
            "7",
            "--host-memory-mib",
            "64",
            "--max-inflight-mib",
            "32",
            "--checksum",
            "--max-retries",
            "4",
            "--no-sync-fallback",
        ],
    )

    args = benchmark.parse_args()

    assert args.kv_swap_mode == "async"
    assert args.warmup_requests == 3
    assert args.trials == 7
    assert args.host_memory_mib == 64
    assert args.max_inflight_mib == 32
    assert args.checksum is True
    assert args.max_retries == 4
    assert args.kv_swap_allow_sync_fallback is False


def test_swap_config_converts_mib_and_preserves_all_six_values() -> None:
    args = type(
        "Args",
        (),
        {
            "kv_swap_mode": "async",
            "host_memory_mib": 64,
            "max_inflight_mib": 32,
            "checksum": True,
            "max_retries": 4,
            "kv_swap_allow_sync_fallback": False,
        },
    )()

    assert benchmark.swap_config_from_args(args) == {
        "kv_swap_mode": "async",
        "kv_swap_host_memory_bytes": 64 * 1024 * 1024,
        "kv_swap_max_inflight_bytes": 32 * 1024 * 1024,
        "kv_swap_checksum": True,
        "kv_swap_max_retries": 4,
        "kv_swap_allow_sync_fallback": False,
    }


def test_trial_summary_contains_raw_samples_and_tail_percentiles() -> None:
    trials = [
        {
            "latency_ms": 10.0,
            "swap_out_observed_ms": 1.0,
            "swap_in_observed_ms": 2.0,
        },
        {
            "latency_ms": 20.0,
            "swap_out_observed_ms": 3.0,
            "swap_in_observed_ms": 4.0,
        },
    ]

    summary = benchmark.summarize_trials(trials)

    assert summary["raw_samples"] == trials
    assert summary["latency_ms"] == {"p50": 15.0, "p95": 20.0, "p99": 20.0}
    assert summary["swap_out_observed_ms"]["p99"] == 3.0
    assert summary["swap_in_observed_ms"]["p99"] == 4.0
