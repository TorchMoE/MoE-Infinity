import pytest

from benchmarks.expert_io_microbench.bench_bubble import summarize_iterations
from benchmarks.expert_io_microbench.bench_routing import summarize_gpu_routing
from benchmarks.expert_io_microbench.nsys_parser import parse_cli_args
from benchmarks.expert_io_microbench.run_decision_profile import (
    build_model_config as build_profile_model_config,
)
from benchmarks.expert_io_microbench.run_decision_profile import (
    build_profile_payload,
)
from benchmarks.expert_io_microbench.run_decision_profile import (
    parse_args as parse_profile_args,
)
from benchmarks.serving.latency import (
    build_model_config as build_latency_model_config,
)
from benchmarks.serving.latency import (
    build_result_payload,
    gpu_routing_verdict,
    load_routing_baseline,
    run_sweep,
)
from benchmarks.serving.latency import (
    parse_args as parse_latency_args,
)


def test_gpu_routing_summary_exposes_submit_fallback_and_native_stats():
    events = [
        {"stage": "gpu_route_submit", "dur_ns": 100},
        {"stage": "gpu_route_submit", "dur_ns": 300},
        {"stage": "gpu_route_fallback", "dur_ns": 900},
    ]
    summary = summarize_gpu_routing(
        events,
        {
            "route_batches": 2,
            "route_failures": 0,
            "last_active_experts": 4,
            "last_route_handoff_us": 7,
            "completion_events_retired": 8,
        },
    )
    assert summary["submit_p50_ns"] == 200.0
    assert summary["fallback_count"] == 1
    assert summary["native"]["route_failures"] == 0


def test_bubble_summary_splits_route_and_completion_handoff():
    summary = summarize_iterations(
        [
            {
                "step_total_ns": 1000,
                "expert_wait_ns": 300,
                "route_submit_ns": 40,
                "completion_handoff_ns": 60,
                "bubble_ratio": 0.3,
                "layer_wait_ns": {},
                "sync_event_count": 1,
                "io_profiler_event_count": 3,
            }
        ]
    )
    assert summary["step_decomposition_mean"]["route_submit_ns"] == 40.0
    assert summary["step_decomposition_mean"]["completion_handoff_ns"] == 60.0


def test_nsys_summary_reports_activity_bitmap_d2h(monkeypatch):
    from benchmarks.expert_io_microbench import nsys_parser

    monkeypatch.setattr(
        nsys_parser,
        "parse_nsys_report",
        lambda _path: {
            "ranges": {
                "gpu_route_submit": {
                    "total_ns": 100,
                    "count": 1,
                    "mean_ns": 100,
                    "p50_ns": 100,
                }
            },
            "memcpy": {
                "h2d_bytes": 0,
                "h2d_count": 0,
                "d2h_bytes": 8,
                "d2h_count": 1,
                "d2d_bytes": 0,
                "d2d_count": 0,
            },
            "gpu_memcpy_ns": {"h2d": 0, "d2h": 10, "d2d": 0},
            "cuda_api": {
                "stream_synchronize_count": 0,
                "device_synchronize_count": 0,
            },
            "duration_ns": 100,
        },
    )
    summary = nsys_parser.summarise(
        "unused.nsys-rep", 1, {"link_width": 16, "link_gen": 4}
    )
    assert summary["routing_sync"]["device_to_host_memcpy_count"] == 1
    assert summary["routing_sync"]["device_to_host_memcpy_bytes"] == 8
    assert summary["routing_sync"]["stream_synchronize_count"] == 0


def test_gpu_routing_verdict_rolls_back_p99_regression():
    verdict = gpu_routing_verdict(
        baseline={"tpot_p50_ms": 10.0, "tpot_p99_ms": 20.0},
        candidate={"tpot_p50_ms": 9.5, "tpot_p99_ms": 21.1},
        route_failures=0,
        fallback_count=0,
    )
    assert verdict["decision"] == "ROLLBACK"
    assert "tpot_p99_regression_gt_5pct" in verdict["reasons"]


def test_gpu_routing_verdict_accepts_non_regressing_candidate():
    verdict = gpu_routing_verdict(
        baseline={"tpot_p50_ms": 10.0, "tpot_p99_ms": 20.0},
        candidate={"tpot_p50_ms": 9.0, "tpot_p99_ms": 19.0},
        route_failures=0,
        fallback_count=0,
    )
    assert verdict == {"decision": "KEEP", "reasons": []}


def test_latency_cli_wires_mode_warmups_and_paths():
    args = parse_latency_args(
        [
            "--model",
            "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "--offload-dir",
            "/tmp/moe-routing-store",
            "--gpu-only-expert-routing",
            "on",
            "--warmup-rounds",
            "3",
            "--routing-baseline-json",
            "/tmp/routing-off.json",
            "--output-json",
            "/tmp/routing-on.json",
        ]
    )
    assert args.gpu_only_expert_routing == "on"
    assert args.warmup_rounds == 3
    assert build_latency_model_config(args)["gpu_only_expert_routing"] is True
    assert (
        build_latency_model_config(args)["speculative_prefetch_overlap"]
        is False
    )


def test_decision_profile_cli_wires_mode_and_warmup_iterations():
    args = parse_profile_args(
        [
            "--model",
            "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "--offload-dir",
            "/tmp/moe-routing-store",
            "--hardware-tag",
            "single-host",
            "--mode",
            "host-only",
            "--gpu-only-expert-routing",
            "on",
            "--warmup-iters",
            "3",
            "--output-json",
            "/tmp/routing-profile.json",
        ]
    )
    assert args.gpu_only_expert_routing == "on"
    assert args.warmup_iters == 3
    assert build_profile_model_config(args)["gpu_only_expert_routing"] is True
    assert (
        build_profile_model_config(args)["speculative_prefetch_overlap"]
        is False
    )


def test_decision_profile_rejects_gpu_routing_with_overlap():
    args = parse_profile_args(
        [
            "--model",
            "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "--offload-dir",
            "/tmp/moe-routing-store",
            "--hardware-tag",
            "single-host",
            "--mode",
            "host-only",
            "--gpu-only-expert-routing",
            "on",
            "--speculative-prefetch-overlap",
            "--output-json",
            "/tmp/routing-profile.json",
        ]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        build_profile_model_config(args)


def test_nsys_cli_accepts_profile_schema_input():
    args = parse_cli_args(
        [
            "/tmp/gpu-routing-on.nsys-rep",
            "--steps",
            "96",
            "--profile-json",
            "/tmp/gpu-routing-profile.json",
        ]
    )
    assert args.steps == 96
    assert args.profile_json == "/tmp/gpu-routing-profile.json"


def test_decision_profile_result_schema():
    from types import SimpleNamespace

    payload = build_profile_payload(
        args=SimpleNamespace(
            gpu_only_expert_routing="on",
            warmup_iters=3,
            warmup_tokens=8,
            iters=3,
            max_new_tokens=32,
        ),
        decode_step_times_ns=[100, 110, 120],
        routing={
            "route_batches": 30,
            "route_failures": 0,
            "fallback_count": 0,
            "completion_events_retired": 60,
        },
        pcie={
            "link_width_pre": 16,
            "link_gen_pre": 4,
            "link_width_post": 16,
            "link_gen_post": 4,
        },
    )
    assert payload["schema_version"] == "gpu-routing-decision-profile-v1"
    assert payload["measurement"]["decode_step_count"] == 96
    assert payload["routing"]["route_failures"] == 0


def test_latency_result_schema_contains_tpot_samples_and_verdict():
    payload = build_latency_payload_for_test()
    assert payload["schema_version"] == "gpu-routing-latency-v1"
    assert payload["measurement"]["1"]["tpot_p50_ms"] == 9.0
    assert payload["measurement"]["1"]["tpot_p99_ms"] == 19.0
    assert payload["routing"]["route_failures"] == 0
    assert payload["verdict"]["decision"] == "KEEP"


def build_latency_payload_for_test():
    from types import SimpleNamespace

    args = SimpleNamespace(
        model="deepseek-ai/DeepSeek-V2-Lite-Chat",
        offload_dir="/tmp/moe-routing-store",
        gpu_only_expert_routing="on",
        concurrency=[1],
        prompt_length=128,
        max_new_tokens=64,
        warmup_rounds=3,
        num_rounds=30,
    )
    baseline = {"1": {"tpot_p50_ms": 10.0, "tpot_p99_ms": 20.0}}
    candidate = {
        "1": {
            "sample_count": 30,
            "ttft_p50_ms": 100.0,
            "ttft_p99_ms": 120.0,
            "tpot_p50_ms": 9.0,
            "tpot_p99_ms": 19.0,
            "itl_p50_ms": 9.0,
            "itl_p99_ms": 19.0,
        }
    }
    return build_result_payload(
        args=args,
        measurements=candidate,
        baseline_measurements=baseline,
        routing={
            "route_batches": 10,
            "route_failures": 0,
            "fallback_count": 0,
            "completion_events_retired": 20,
        },
    )


def test_profiled_loop_does_not_inject_stream_synchronize():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    source = (
        root / "benchmarks/expert_io_microbench/run_decision_profile.py"
    ).read_text(encoding="utf-8")
    measured = source[
        source.index("decode_step_times_ns: list[int]") : source.index(
            "cudaProfilerStop()"
        )
    ]
    assert "torch.cuda.synchronize()" not in measured


def test_latency_warmups_are_excluded(monkeypatch):
    from benchmarks.serving import latency

    samples = iter([1000.0, 1001.0, 10.0, 20.0])

    def fake_round(*args, **kwargs):
        value = next(samples)
        return [value], [value]

    monkeypatch.setattr(latency, "run_one_round", fake_round)
    result = run_sweep(
        object(),
        object(),
        concurrency_levels=[1],
        warmup_rounds=2,
        num_rounds=2,
        prompt_length=128,
        max_new_tokens=64,
    )
    assert result["1"]["sample_count"] == 2
    assert result["1"]["tpot_p50_ms"] == 15.0


def test_routing_baseline_rejects_wrong_schema(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text('{"schema_version":"baseline-performance-v1"}')
    with pytest.raises(ValueError, match="gpu-routing-latency-v1"):
        load_routing_baseline(path, expected_config={})
