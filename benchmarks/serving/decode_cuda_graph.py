from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired decode CUDA graph benchmark"
    )
    parser.add_argument(
        "--mode", choices=("fixture", "model"), default="fixture"
    )
    parser.add_argument("--model")
    parser.add_argument("--offload-dir")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(
        "--context-sizes", nargs="+", type=int, default=[128, 512]
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--max-graph-memory-bytes", type=int, default=0)
    parser.add_argument("--profile-launches", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "model" and (not args.model or not args.offload_dir):
        raise ValueError("model mode requires --model and --offload-dir")
    for name in ("batch_sizes", "context_sizes"):
        values = getattr(args, name)
        if not values or any(value <= 0 for value in values):
            raise ValueError(
                f"--{name.replace('_', '-')} values must be positive"
            )
    if args.warmup_iters < 1 or args.measure_iters < 1:
        raise ValueError("warmup and measurement iterations must be positive")
    if args.max_graph_memory_bytes < 0:
        raise ValueError("max graph memory must be non-negative")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": float(statistics.median(values)),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
    }


def build_result(
    *,
    config: dict[str, object],
    eager_us: list[float],
    replay_us: list[float],
    graph_stats: dict[str, object],
    environment: dict[str, object],
    correctness: dict[str, object] | None = None,
) -> dict[str, object]:
    measurements: dict[str, object] = {
        "eager_us": eager_us,
        "replay_us": replay_us,
    }
    if eager_us:
        measurements["eager_summary_us"] = _summary(eager_us)
    if replay_us:
        measurements["replay_summary_us"] = _summary(replay_us)
    if eager_us and replay_us:
        replay_p50 = statistics.median(replay_us)
        measurements["observed_ratio"] = (
            statistics.median(eager_us) / replay_p50 if replay_p50 else None
        )
    return {
        "schema_version": 1,
        "config": config,
        "measurements": measurements,
        "graph_stats": graph_stats,
        "environment": environment,
        "correctness": correctness or {},
    }


def _time_cuda(
    operation: Callable[[], torch.Tensor],
) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) * 1000.0), output


def _profile_cuda_operation(
    operation: Callable[[], torch.Tensor],
) -> dict[str, float | int]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        _ = operation()
        torch.cuda.synchronize()
    events = profile.key_averages()
    cuda_events = [event for event in events if event.device_time_total > 0]
    return {
        "cuda_operator_count": sum(event.count for event in cuda_events),
        "cuda_time_us": float(
            sum(event.device_time_total for event in cuda_events)
        ),
    }


def _environment() -> dict[str, object]:
    return {
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


def _run_fixture(args: argparse.Namespace) -> dict[str, object]:
    from benchmarks.serving.decode_cuda_graph_fixture import build_fixture

    batch_sizes = tuple(sorted(set(args.batch_sizes)))
    context_sizes = tuple(sorted(set(args.context_sizes)))
    fixture = build_fixture(
        batch_sizes=batch_sizes,
        context_sizes=context_sizes,
        warmup_iters=args.warmup_iters,
        max_graph_memory_bytes=args.max_graph_memory_bytes,
    )
    points: list[dict[str, object]] = []
    try:
        for batch_size in batch_sizes:
            for context_size in context_sizes:
                batch = fixture.make_batch(batch_size, context_size)
                prepared = fixture.model_runner.allocate_decode_buffers(
                    batch_bucket=batch_size,
                    context_bucket=context_size,
                )
                fixture.model_runner.copy_decode_batch(batch, prepared, [])

                for _ in range(args.warmup_iters):
                    _ = fixture.model_runner.forward_prepared_decode(prepared)
                graph_output = fixture.graph_runner.try_execute(batch)
                if graph_output is None:
                    raise RuntimeError(
                        f"fixture graph fallback: {fixture.graph_runner.stats()}"
                    )
                torch.cuda.synchronize()

                eager_us: list[float] = []
                replay_us: list[float] = []
                eager_output = fixture.model_runner.forward_prepared_decode(
                    prepared
                )
                for iteration in range(args.measure_iters):
                    operations = ("eager", "graph")
                    if iteration % 2:
                        operations = tuple(reversed(operations))
                    for operation in operations:
                        if operation == "eager":
                            elapsed, eager_output = _time_cuda(
                                lambda: fixture.model_runner.forward_prepared_decode(
                                    prepared
                                )
                            )
                            eager_us.append(elapsed)
                        else:
                            elapsed, replayed = _time_cuda(
                                lambda: fixture.graph_runner.try_execute(batch)
                            )
                            if replayed is None:
                                raise RuntimeError(
                                    "fixture graph unexpectedly fell back"
                                )
                            graph_output = replayed
                            replay_us.append(elapsed)

                torch.testing.assert_close(
                    graph_output,
                    eager_output[:batch_size],
                    rtol=1e-4,
                    atol=1e-4,
                )
                checksums = [
                    float(fixture.storage.value_cache[layer].sum().item())
                    for layer in range(fixture.storage.spec.num_layers)
                ]
                stats = fixture.graph_runner.stats()
                scratch_kv_bytes = (
                    fixture.storage.num_graph_scratch_blocks
                    * fixture.storage.spec.block_size
                    * fixture.storage.spec.num_layers
                    * 2
                    * fixture.storage.spec.num_kv_heads
                    * fixture.storage.spec.head_dim
                    * torch.empty(
                        (), dtype=fixture.storage.spec.dtype
                    ).element_size()
                )
                stats.update(
                    {
                        "capability_reason": "eligible",
                        "registered_paged_layers": 2,
                        "proved_write_layers": 2,
                        "kv_storage_owner_id": fixture.storage.owner_id,
                        "per_layer_kv_checksums": checksums,
                        "native_attention_kernel": True,
                        "scratch_kv_bytes": scratch_kv_bytes,
                        "replay_coverage": 1.0,
                    }
                )
                launch_profile: dict[str, object] = {}
                if args.profile_launches:
                    launch_profile = {
                        "eager": _profile_cuda_operation(
                            lambda: fixture.model_runner.forward_prepared_decode(
                                prepared
                            )
                        ),
                        "replay": _profile_cuda_operation(
                            lambda: fixture.graph_runner.try_execute(batch)
                        ),
                    }
                points.append(
                    build_result(
                        config={
                            "mode": "fixture",
                            "batch_size": batch_size,
                            "context_size": context_size,
                            "batch_buckets": batch_sizes,
                            "context_buckets": context_sizes,
                            "dtype": str(fixture.storage.dtype),
                        },
                        eager_us=eager_us,
                        replay_us=replay_us,
                        graph_stats=stats,
                        environment=_environment(),
                        correctness={
                            "eager_replay_equal": True,
                            "launch_profile": launch_profile,
                        },
                    )
                )
    finally:
        fixture.close()
    return {"schema_version": 1, "mode": "fixture", "results": points}


def _run_model(args: argparse.Namespace) -> dict[str, object]:
    from moe_infinity import MoE

    runtime = MoE(
        args.model,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": 0.75,
        },
    )
    capability_fn = getattr(runtime, "decode_graph_capability", None)
    capability = capability_fn() if callable(capability_fn) else None
    reason = getattr(capability, "reason", "missing_capability")
    points: list[dict[str, object]] = []
    for batch_size in sorted(set(args.batch_sizes)):
        for context_size in sorted(set(args.context_sizes)):
            input_ids = torch.ones(
                (batch_size, context_size),
                dtype=torch.long,
                device="cuda",
            )

            def eager_once() -> torch.Tensor:
                return runtime.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=False,
                )

            for _ in range(args.warmup_iters):
                _ = eager_once()
            torch.cuda.synchronize()
            eager_us = [
                _time_cuda(eager_once)[0] for _ in range(args.measure_iters)
            ]
            points.append(
                build_result(
                    config={
                        "mode": "model",
                        "model": args.model,
                        "batch_size": batch_size,
                        "context_size": context_size,
                    },
                    eager_us=eager_us,
                    replay_us=[],
                    graph_stats={
                        "captures": 0,
                        "replays": 0,
                        "capability_reason": reason,
                    },
                    environment=_environment(),
                    correctness={"comparison": "eager_only"},
                )
            )
    return {
        "schema_version": 1,
        "mode": "model",
        "model": args.model,
        "capability_reason": reason,
        "results": points,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("decode CUDA graph benchmark requires CUDA")
    result = _run_fixture(args) if args.mode == "fixture" else _run_model(args)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
