from benchmarks.adaptive_precision.report import validate_run
from benchmarks.adaptive_precision.workloads import deterministic_workload


def test_benchmark_schema_requires_quality_memory_transfer_and_tpot():
    row = {
        "mode": "adaptive",
        "model": "tiny",
        "checkpoint_fingerprint": "a" * 64,
        "format": "fp8_e4m3_block128",
        "converter_version": "adaptive-expert-v1",
        "quality_attestation_sha256": "b" * 64,
        "hardware": {},
        "software": {},
        "workload": {},
        "quality": {"perplexity": 3.0, "greedy_agreement": 1.0},
        "memory": {"budget_bytes": 1000, "peak_accounted_bytes": 900},
        "transfer": {"h2d_payload_bytes": 400, "h2d_transfers": 4},
        "latency": {
            "ttft_ms": 10.0,
            "tpot_ms_p50": 2.0,
            "tpot_ms_p90": 3.0,
            "tpot_ms_p99": 4.0,
        },
        "throughput": {"decode_tokens_per_second": 500.0},
        "policy": {"fallback_counts": {}},
    }
    validate_run(row)


def test_deterministic_workload_needs_no_prompt_file():
    one = deterministic_workload(seed=7, cases=8, min_tokens=32, max_tokens=128)
    assert one == deterministic_workload(
        seed=7, cases=8, min_tokens=32, max_tokens=128
    )
    assert len(one) == 8
    assert all(32 <= len(case.input_ids) <= 128 for case in one)
