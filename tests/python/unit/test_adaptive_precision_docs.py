from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ADAPTIVE = (ROOT / "docs/adaptive-expert-precision.md").read_text()
CONFIG = (ROOT / "docs/configuration.md").read_text()
COMPAT = (ROOT / "docs/model-compatibility.md").read_text()
BENCH = (ROOT / "docs/benchmarking.md").read_text()


def _contains_all(text: str, values: tuple[str, ...]) -> None:
    missing = [value for value in values if value not in text]
    assert not missing, f"missing documentation contracts: {missing}"


def test_documents_disabled_default_and_configuration_units():
    _contains_all(
        CONFIG + ADAPTIVE,
        (
            "adaptive_expert_precision",
            "false",
            "adaptive_hbm_budget_bytes",
            "bytes",
            "adaptive_policy_epoch_tokens",
            "adaptive_hotness_decay",
            "adaptive_promotion_threshold",
            "adaptive_demotion_threshold",
            "adaptive_min_residency_epochs",
            "adaptive_transition_cooldown_epochs",
            "adaptive_variant_build",
            "adaptive_derivative_root",
        ),
    )


def test_documents_manifest_index_and_attestation_validation():
    _contains_all(
        ADAPTIVE,
        (
            "CURRENT",
            "derivative-index.v1.json",
            "manifest.v1.json",
            "quality-attestation.v1.json",
            "checkpoint_fingerprint",
            "converter_version",
            "quality_attestation_sha256",
            "ReleasedAdaptiveEntry",
            "manifest_unapproved",
            "canonical fallback",
        ),
    )


def test_documents_support_and_four_flag_rollout_matrix():
    _contains_all(
        ADAPTIVE + COMPAT,
        (
            "GPT-OSS MXFP4",
            "GLM-5.2 FP8",
            "DeepSeek-V4-Flash FP4",
            "DeepSeek-V3 FP8",
            "GPTQ/AWQ",
            "Phase policy",
            "Adaptive precision",
            "off | off",
            "on | off",
            "off | on",
            "on | on",
            "ExpertResidencyManager",
        ),
    )


def test_documents_benchmark_and_attestation_acceptance():
    _contains_all(
        BENCH + ADAPTIVE,
        (
            "deterministic-v1",
            "canonical",
            "static-low",
            "adaptive",
            "H2D",
            "peak_accounted_bytes",
            "TTFT",
            "TPOT",
            "p50",
            "p90",
            "p99",
            "throughput",
            "release_gate",
            "quality-attestation",
            "five",
            "no speedup is guaranteed",
        ),
    )


def test_documents_executable_build_benchmark_report_and_rollback_commands():
    _contains_all(
        ADAPTIVE,
        (
            "python benchmarks/adaptive_precision/bench_e2e.py",
            "--adaptive-variant-build",
            "--build-only",
            "python benchmarks/adaptive_precision/bench_policy.py",
            "python benchmarks/adaptive_precision/report.py",
            '"adaptive_expert_precision": false',
            "restart",
            "preserve",
            "adaptive_derivatives",
            "CURRENT",
        ),
    )
