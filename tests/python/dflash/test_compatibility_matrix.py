from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DFLASH_DOC = ROOT / "docs" / "dflash.md"
MODEL_MATRIX = ROOT / "docs" / "model-compatibility.md"
SERVING_DOC = ROOT / "docs" / "serving.md"
DESIGN = (
    ROOT
    / "docs"
    / "superpowers"
    / "specs"
    / "2026-08-17-dflash-unified-execution-design.md"
)
PLAN = (
    ROOT
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-08-17-dflash-unified-execution.md"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_matrix_reports_pairing_and_executor_evidence_separately() -> None:
    matrix = _text(MODEL_MATRIX)
    assert "DFlash pairing evidence" in matrix
    assert "Executor / route-ahead evidence" in matrix
    assert "GPT-OSS" in matrix
    assert "valid published pairs" in matrix
    assert "no executor route-ahead" in matrix
    assert "No real DeepSeek DFlash pair" in matrix


def test_matrix_gates_rich_batch_and_paged_claims_on_capabilities() -> None:
    matrix = _text(MODEL_MATRIX)
    assert "Rich execution capability" in matrix
    assert "Serving cache capability" in matrix
    assert "row-aware capability guard" in matrix
    assert "default-off" in matrix
    assert "DeepSeek V2/V3" in matrix
    assert "batch-1 greedy" in matrix
    assert "resident-only; no swap/preemption" in matrix
    assert "Qwen/hybrid fallback" in matrix


def test_dflash_documentation_records_unified_semantics_and_limits() -> None:
    doc = _text(DFLASH_DOC)
    required = (
        "one semantic core",
        "SessionDriver",
        "mixed greedy and sampled",
        "per-row generator",
        "correlated",
        "not bit-exact across batch shapes",
        "dense cache reconstruction",
        "last_generated_lengths",
        "right-padded",
        "grouped per-request",
        "physically batched",
        "temporary_dynamic",
        "paged_mla",
        "draft cache remains separate",
        "cancellation",
        "preemption",
    )
    for claim in required:
        assert claim in doc


def test_deprecated_generate_warning_and_serving_fallback_are_documented() -> (
    None
):
    combined = _text(DFLASH_DOC) + _text(SERVING_DOC)
    assert "MoE.generate() is deprecated" in combined
    assert "DeprecationWarning" in combined
    assert "sampled serving" in combined
    assert "temporary DynamicCache" in combined
    assert "fallback" in combined
    assert "not evidence of sampled serving" in combined


def test_trace_fields_are_stable_for_direct_and_serving() -> None:
    doc = _text(DFLASH_DOC)
    for field in (
        "request_id",
        "backend",
        "cache_kind",
        "round_count",
        "accepted",
        "committed",
        "emitted",
        "rollback",
        "replay",
        "pairing_evidence",
        "executor_evidence",
    ):
        assert f"`{field}`" in doc
    assert "direct and serving" in doc


def test_checked_in_design_and_plan_record_task_8_5_dependency_order() -> None:
    design = _text(DESIGN)
    plan = _text(PLAN)
    for text in (design, plan):
        assert "Task 8.5" in text
        assert "DeepSeek MLA prerequisite" in text
        assert "Task 8.5 -> Task 9" in text
    assert "Stage 4b is default-off" in design
    assert "actual delivered behavior" in plan


def test_tiny_benchmark_reports_measured_and_observed_fields() -> None:
    command = [
        sys.executable,
        "benchmarks/dflash/unified_execution_benchmark.py",
        "--fixture",
        "tiny",
        "--json",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    repeated = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    report = json.loads(completed.stdout)
    repeated_report = json.loads(repeated.stdout)
    required = {
        "fixture",
        "prefill_latency_ms",
        "verify_latency_ms",
        "decode_elapsed_seconds",
        "decode_committed_tokens_per_second",
        "sample_count",
        "round_count",
        "accepted_drafts",
        "committed_tokens",
        "rollback_count",
        "replay_count",
        "rng_order_invariant",
        "sampled_tvd_value",
        "sampled_kl_value",
        "metric_units",
        "cache_pages_peak",
        "execution_mode",
        "pairing_evidence",
        "executor_evidence",
        "per_request_rich_calls",
        "physical_rich_calls",
    }
    assert required <= report.keys()
    assert report["fixture"] == "tiny"
    assert report["measurement_scope"] == "synthetic no-checkpoint CPU fixture"
    assert report["prefill_latency_ms"] >= 0
    assert report["verify_latency_ms"] >= 0
    assert report["decode_committed_tokens_per_second"] > 0
    assert report["sampled_tvd_value"] >= 0
    assert report["sampled_kl_value"] >= 0
    assert report["sample_count"] == report["round_count"]
    assert (
        report["round_count"]
        == report["accepted_drafts"] + report["rollback_count"]
    )
    assert (
        report["committed_tokens"]
        == report["round_count"] + report["accepted_drafts"]
    )
    assert report["decode_committed_tokens_per_second"] == (
        report["committed_tokens"] / report["decode_elapsed_seconds"]
    )
    assert report["metric_units"] == {
        "prefill_latency_ms": "milliseconds per prefill operation",
        "verify_latency_ms": "milliseconds per verify operation",
        "decode_elapsed_seconds": "seconds",
        "decode_committed_tokens_per_second": "committed tokens per second",
        "sampled_tvd_value": "dimensionless",
        "sampled_kl_value": "nats",
        "cache_pages_peak": "pages",
        "cancellation_latency_ms": "milliseconds",
    }
    deterministic_fields = {
        "sample_count",
        "round_count",
        "accepted_drafts",
        "committed_tokens",
        "rollback_count",
        "replay_count",
        "rng_order_invariant",
        "sampled_tvd_value",
        "sampled_kl_value",
        "cache_pages_peak",
        "execution_mode",
        "pairing_evidence",
        "executor_evidence",
    }
    assert {key: report[key] for key in deterministic_fields} == {
        key: repeated_report[key] for key in deterministic_fields
    }


def test_tiny_validation_fails_closed_and_passes_all_local_gates() -> None:
    command = [
        sys.executable,
        "benchmarks/dflash/validate_unified_execution.py",
        "--fixture",
        "tiny",
        "--require-cache-invariants",
        "--require-order-invariance",
        "--json",
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS"
    assert report["checkpoint_downloads"] is False
    assert report["cache_invariants"] is True
    assert report["ownership_isolation"] is True
    assert report["order_invariance"] is True
    assert report["required_gpu_fixture"] is False
    assert isinstance(report["sampled_tvd_value"], float)
    assert isinstance(report["sampled_kl_value"], float)
    assert report["sampled_tvd_pass"] is True
    assert report["sampled_kl_pass"] is True


def test_require_gpu_is_readiness_only_and_fails_when_fixture_env_is_disabled() -> (
    None
):
    command = [
        sys.executable,
        "benchmarks/dflash/validate_unified_execution.py",
        "--fixture",
        "tiny",
        "--require-gpu",
        "--json",
    ]
    environment = dict(os.environ, MOE_DFLASH_GPU="0")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert report["status"] == "FAIL"
    assert report["gpu_readiness_required"] is True
    assert report["gpu_readiness_pass"] is False
    assert report["gpu_harness_executed"] is False
    assert report["gpu_gate_kind"] == "readiness only"


def test_docs_distinguish_gpu_readiness_from_harness_execution() -> None:
    combined = _text(DFLASH_DOC) + _text(PLAN)
    assert "--require-gpu is a readiness gate" in combined
    assert "does not execute the GPU harness" in combined
    assert "actual GPU pytest command remains separate and required" in combined
