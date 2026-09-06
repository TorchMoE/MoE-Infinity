from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.expert_io_microbench.stats import TimingCollector

DEFAULT_WARMUP = 10
DEFAULT_ITERS = 100
DEFAULT_MAX_NEW_TOKENS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark routing/cache_lookup overhead via IOProfiler."
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--offload-dir",
        required=True,
        help="Directory used for MoE expert offload storage",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup generate iterations (not measured)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="Measured generate iterations",
    )
    parser.add_argument(
        "--output-json",
        default="routing_microbench_results.json",
        help="Path to write benchmark JSON",
    )
    parser.add_argument(
        "--host-only",
        action="store_true",
        help=(
            "Copy offload weights to /dev/shm (tmpfs) to eliminate disk I/O. "
            "Measures pure PCIe+sync overhead without disk reads. "
            "Requires sufficient /dev/shm space (use Docker --shm-size=32g)."
        ),
    )
    return parser.parse_args()


def _directory_size_bytes(root: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                continue
    return total


def _is_under_dev_shm(path: str) -> bool:
    try:
        resolved = os.path.realpath(path)
        return os.path.commonpath([resolved, "/dev/shm"]) == "/dev/shm"
    except (OSError, ValueError):
        return False


def setup_offload_dir(
    args: argparse.Namespace,
) -> tuple[str, str, Callable[[], None]]:
    if not args.host_only:
        return args.offload_dir, "disk", lambda: None

    if not os.path.exists(args.offload_dir):
        print(
            f"WARNING: offload dir does not exist: {args.offload_dir}",
            file=sys.stderr,
        )
        print(
            "Hint: provide a valid --offload-dir before enabling --host-only",
            file=sys.stderr,
        )
        raise RuntimeError(f"offload dir does not exist: {args.offload_dir}")

    if _is_under_dev_shm(args.offload_dir):
        print(
            f"Using existing tmpfs offload dir: {args.offload_dir}",
            flush=True,
        )
        return args.offload_dir, "host-only", lambda: None

    src_size = _directory_size_bytes(args.offload_dir)
    shm_stat = shutil.disk_usage("/dev/shm")
    required_bytes = int(src_size * 1.1)
    if shm_stat.free < required_bytes:
        print(
            "WARNING: /dev/shm has "
            f"{shm_stat.free / 1e9:.1f}GB free but offload needs {src_size / 1e9:.1f}GB",
            file=sys.stderr,
        )
        print(
            "Hint: docker run --shm-size=<size>g or increase host shm",
            file=sys.stderr,
        )
        raise RuntimeError("insufficient /dev/shm space for host-only mode")

    dst = tempfile.mkdtemp(dir="/dev/shm", prefix="moe_hostonly_")
    print(
        f"Copying {src_size / 1e9:.1f}GB offload -> {dst} (tmpfs)...",
        flush=True,
    )
    shutil.copytree(args.offload_dir, dst, dirs_exist_ok=True)
    print("Copy done. Running in host-only (RAM) mode.", flush=True)

    def cleanup() -> None:
        shutil.rmtree(dst, ignore_errors=True)

    return dst, "host-only", cleanup


def environment_info() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count()
    info: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
    }
    if cuda_available and cuda_device_count > 0:
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(idx) for idx in range(cuda_device_count)
        ]
    return info


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _repeat_to_length(token_ids: list[int], target_length: int) -> list[int]:
    if not token_ids:
        return [0] * target_length
    output: list[int] = []
    while len(output) < target_length:
        output.extend(token_ids)
    return output[:target_length]


def build_prompt_input_ids(
    tokenizer: Any, target_length: int = 64
) -> torch.Tensor:
    base_text = (
        "MoE-Infinity routing microbenchmark prompt. "
        "Measure profiler events for routing and cache lookup."
    )
    encoded = tokenizer.encode(base_text, add_special_tokens=False)
    prompt_ids = _repeat_to_length(encoded, target_length)
    return torch.tensor([prompt_ids], dtype=torch.long, device="cuda")


def ensure_model_available_locally(model_name: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub import failed while checking local model cache"
        ) from exc

    try:
        _ = snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Model is not available in local HuggingFace cache. "
            "Please pre-download it before running this benchmark "
            f"(model={model_name}). Original error: {type(exc).__name__}: {exc}"
        ) from exc


def load_model_and_tokenizer(
    model_name: str, offload_dir: str
) -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"transformers import failed: {exc}") from exc

    try:
        from moe_infinity import MoE
    except Exception as exc:
        raise RuntimeError(f"moe_infinity import failed: {exc}") from exc

    ensure_model_available_locally(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = {
        "offload_path": offload_dir,
        "device_memory_ratio": 0.75,
    }
    model = MoE(model_name, config)
    return model, tokenizer


def _snapshot_profiler_events(profiler: Any) -> list[dict[str, Any]]:
    lock = getattr(profiler, "_lock", None)
    events = getattr(profiler, "_events", None)
    if events is None:
        return []

    if lock is None:
        return list(events)

    with lock:
        return [dict(event) for event in events if isinstance(event, dict)]


def _empty_stage_stats() -> dict[str, float | int | None]:
    return {
        "min_ns": None,
        "max_ns": None,
        "mean_ns": None,
        "p50_ns": None,
        "p95_ns": None,
        "p99_ns": None,
        "count": 0,
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (q / 100.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def summarize_gpu_routing(
    events: list[Mapping[str, Any]], native_stats: Mapping[str, Any]
) -> dict[str, Any]:
    submit_values = [
        float(event["dur_ns"])
        for event in events
        if event.get("stage") == "gpu_route_submit"
    ]
    fallback_values = [
        float(event["dur_ns"])
        for event in events
        if event.get("stage") == "gpu_route_fallback"
    ]
    return {
        "submit_p50_ns": _percentile(submit_values, 50),
        "submit_p99_ns": _percentile(submit_values, 99),
        "fallback_count": len(fallback_values),
        "fallback_p50_ns": _percentile(fallback_values, 50),
        "native": {
            key: int(native_stats.get(key, 0))
            for key in (
                "route_batches",
                "route_failures",
                "last_active_experts",
                "last_route_handoff_us",
                "completion_events_retired",
            )
        },
    }


def _cache_summary(
    cache_lookup_stats: Mapping[str, float | int | None],
    transfer_events_count: int,
) -> dict[str, float | int | None | str]:
    lookup_count_raw = cache_lookup_stats.get("count")
    lookup_count = (
        int(lookup_count_raw) if isinstance(lookup_count_raw, int) else 0
    )

    miss_count = min(lookup_count, transfer_events_count)
    hit_count = max(lookup_count - miss_count, 0)
    hit_rate: Optional[float]
    if lookup_count > 0:
        hit_rate = hit_count / lookup_count
    else:
        hit_rate = None

    return {
        "hit_rate": hit_rate,
        "hit_count": hit_count,
        "miss_count_estimated": miss_count,
        "mean_lookup_ns": cache_lookup_stats.get("mean_ns"),
        "min_lookup_ns": cache_lookup_stats.get("min_ns"),
        "max_lookup_ns": cache_lookup_stats.get("max_ns"),
        "p50_lookup_ns": cache_lookup_stats.get("p50_ns"),
        "p95_lookup_ns": cache_lookup_stats.get("p95_ns"),
        "p99_lookup_ns": cache_lookup_stats.get("p99_ns"),
        "count": lookup_count,
        "note": (
            "hit_rate is estimated from transfer events (disk_to_cpu/cpu_to_gpu) "
            "because IOProfiler cache_lookup stage does not emit explicit hit/miss."
        ),
    }


def run_benchmark(
    model: Any,
    tokenizer: Any,
    profiler: Any,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    input_ids = build_prompt_input_ids(tokenizer)

    for _ in range(warmup):
        _ = model.generate(
            input_ids,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    profiler.reset()

    for _ in range(iters):
        _ = model.generate(
            input_ids,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    events = _snapshot_profiler_events(profiler)
    collector = TimingCollector()
    transfer_events_count = 0

    for event in events:
        stage_obj = event.get("stage")
        dur_obj = event.get("dur_ns")
        if not isinstance(stage_obj, str):
            continue
        if not isinstance(dur_obj, (int, float)):
            continue
        if stage_obj in {"routing", "cache_lookup"}:
            collector.record(stage_obj, int(dur_obj))
        if stage_obj in {"disk_to_cpu", "cpu_to_gpu"}:
            transfer_events_count += 1

    stats = collector.to_dict()
    routing_stats = stats.get("routing", _empty_stage_stats())
    cache_lookup_stats = stats.get("cache_lookup", _empty_stage_stats())

    return {
        "routing": routing_stats,
        "cache_lookup": cache_lookup_stats,
        "cache": _cache_summary(cache_lookup_stats, transfer_events_count),
        "io_profiler_events": len(events),
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    os.environ["MOE_INFINITY_PROFILE_IO"] = "1"
    os.environ["MOE_INFINITY_PROFILE_IO_SAMPLE"] = "1.0"

    env = environment_info()
    output_path = Path(args.output_json)

    print("=== MoE-Infinity Routing + Cache Lookup Microbenchmark ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUDA available: {env['cuda_available']}")

    mode = "host-only" if args.host_only else "disk"
    actual_offload_dir = args.offload_dir
    cleanup: Callable[[], None] = lambda: None

    if not env["cuda_available"]:
        payload = {
            "status": "BLOCKED",
            "reason": "No CUDA device",
            "environment": env,
            "routing": _empty_stage_stats(),
            "cache_lookup": _empty_stage_stats(),
            "cache": _cache_summary(
                _empty_stage_stats(), transfer_events_count=0
            ),
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "mode": mode,
            "io_mode": mode,
            "warmup": args.warmup,
            "iters": args.iters,
        }
        write_json(output_path, payload)
        return 0

    try:
        from moe_infinity.profiling.io_profiler import IOProfiler

        profiler = IOProfiler.instance()
        profiler.enabled = True
        sample = getattr(profiler, "_sample", None)
        if isinstance(sample, (int, float)):
            setattr(profiler, "_sample", 1.0)
    except Exception as exc:
        payload = {
            "status": "BLOCKED",
            "reason": f"IOProfiler init failed: {type(exc).__name__}: {exc}",
            "environment": env,
            "routing": _empty_stage_stats(),
            "cache_lookup": _empty_stage_stats(),
            "cache": _cache_summary(
                _empty_stage_stats(), transfer_events_count=0
            ),
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "mode": mode,
            "io_mode": mode,
            "warmup": args.warmup,
            "iters": args.iters,
        }
        write_json(output_path, payload)
        return 0

    try:
        actual_offload_dir, mode, cleanup = setup_offload_dir(args)
        model, tokenizer = load_model_and_tokenizer(
            args.model, actual_offload_dir
        )

        results = run_benchmark(
            model=model,
            tokenizer=tokenizer,
            profiler=profiler,
            warmup=args.warmup,
            iters=args.iters,
        )

        payload = {
            "status": "PASS",
            "environment": env,
            "requested_model": args.model,
            "offload_dir": actual_offload_dir,
            "source_offload_dir": args.offload_dir,
            "mode": mode,
            "io_mode": mode,
            "warmup": args.warmup,
            "iters": args.iters,
            **results,
        }
        write_json(output_path, payload)
        return 0
    except Exception as exc:
        payload = {
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "environment": env,
            "routing": _empty_stage_stats(),
            "cache_lookup": _empty_stage_stats(),
            "cache": _cache_summary(
                _empty_stage_stats(), transfer_events_count=0
            ),
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "mode": mode,
            "io_mode": mode,
            "warmup": args.warmup,
            "iters": args.iters,
        }
        write_json(output_path, payload)
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
