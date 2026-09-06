from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WARMUP = 10
DEFAULT_ITERS = 100
DEFAULT_MAX_NEW_TOKENS = 1
SYNC_STAGE = "sync_wait"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure pipeline bubbles from sync_wait stalls using IOProfiler."
        )
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
        default="bubble_microbench_results.json",
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
        "MoE-Infinity bubble microbenchmark prompt. "
        "Single-token decode style measurement with deterministic content."
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


def setup_profiler() -> Any:
    os.environ["MOE_INFINITY_PROFILE_IO"] = "1"
    os.environ["MOE_INFINITY_PROFILE_IO_SAMPLE"] = "1.0"

    from moe_infinity.profiling.io_profiler import IOProfiler

    profiler = IOProfiler.instance()
    profiler.enabled = True
    sample = getattr(profiler, "_sample", None)
    if isinstance(sample, (int, float)):
        setattr(profiler, "_sample", 1.0)
    return profiler


def _snapshot_profiler_events(profiler: Any) -> list[dict[str, Any]]:
    lock = getattr(profiler, "_lock", None)
    events = getattr(profiler, "_events", None)
    if events is None:
        return []

    if lock is None:
        return [dict(event) for event in events if isinstance(event, dict)]

    with lock:
        return [dict(event) for event in events if isinstance(event, dict)]


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _safe_layer(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _clamp_ratio(ratio: float) -> float:
    return min(max(float(ratio), 0.0), 1.0)


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=np.float64)))


def _mean_float(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=np.float64)))


def _empty_step_decomposition() -> dict[str, float | None]:
    return {
        "step_total_ns": None,
        "expert_wait_ns": None,
        "route_submit_ns": None,
        "completion_handoff_ns": None,
        "non_wait_ns": None,
        "bubble_ratio": None,
        "step_total_p50_ns": None,
        "step_total_p95_ns": None,
        "step_total_p99_ns": None,
        "expert_wait_p50_ns": None,
        "expert_wait_p95_ns": None,
        "expert_wait_p99_ns": None,
        "bubble_ratio_p50": None,
        "bubble_ratio_p95": None,
        "bubble_ratio_p99": None,
    }


def run_one_iteration(
    model: Any,
    tokenizer: Any,
    profiler: Any,
    input_ids: torch.Tensor,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    profiler.reset()
    start_ns = time.perf_counter_ns()
    _ = model.generate(
        input_ids,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_total_ns = time.perf_counter_ns() - start_ns

    events = _snapshot_profiler_events(profiler)
    expert_wait_ns = 0
    route_submit_ns = 0
    completion_handoff_ns = 0
    layer_wait_ns: dict[int, int] = {}
    sync_event_count = 0

    for event in events:
        stage = event.get("stage")
        dur_ns = max(_safe_int(event.get("dur_ns")), 0)
        if stage in ("gpu_route_submit", "gpu_route_fallback"):
            route_submit_ns += dur_ns
            continue
        if stage == "expert_completion_handoff":
            completion_handoff_ns += dur_ns
            continue
        if stage != SYNC_STAGE:
            continue

        expert_wait_ns += dur_ns
        sync_event_count += 1

        layer = _safe_layer(event.get("layer"))
        if layer is not None:
            layer_wait_ns[layer] = layer_wait_ns.get(layer, 0) + dur_ns

    ratio = (
        0.0
        if step_total_ns <= 0
        else _clamp_ratio(float(expert_wait_ns) / float(step_total_ns))
    )

    return {
        "step_total_ns": int(step_total_ns),
        "expert_wait_ns": int(expert_wait_ns),
        "route_submit_ns": int(route_submit_ns),
        "completion_handoff_ns": int(completion_handoff_ns),
        "bubble_ratio": ratio,
        "layer_wait_ns": layer_wait_ns,
        "sync_event_count": sync_event_count,
        "io_profiler_event_count": len(events),
    }


def summarize_iterations(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    step_total_ns_values = [
        int(item["step_total_ns"])
        for item in iterations
        if isinstance(item.get("step_total_ns"), int)
    ]
    expert_wait_ns_values = [
        int(item["expert_wait_ns"])
        for item in iterations
        if isinstance(item.get("expert_wait_ns"), int)
    ]
    bubble_ratio_values = [
        float(item["bubble_ratio"])
        for item in iterations
        if isinstance(item.get("bubble_ratio"), (int, float))
    ]
    route_submit_ns_values = [
        int(item["route_submit_ns"])
        for item in iterations
        if isinstance(item.get("route_submit_ns"), int)
    ]
    completion_handoff_ns_values = [
        int(item["completion_handoff_ns"])
        for item in iterations
        if isinstance(item.get("completion_handoff_ns"), int)
    ]

    if not step_total_ns_values:
        return {
            "overall_bubble_ratio": 0.0,
            "per_layer_bubble": None,
            "step_decomposition_mean": _empty_step_decomposition(),
            "step_decomposition_percentiles": _empty_step_decomposition(),
            "io_profiler_events": {
                "mean_events_per_step": 0.0,
                "mean_sync_wait_events_per_step": 0.0,
            },
        }

    non_wait_ns_values = [
        max(step - wait, 0)
        for step, wait in zip(step_total_ns_values, expert_wait_ns_values)
    ]

    total_step_ns = int(sum(step_total_ns_values))
    total_wait_ns = int(sum(expert_wait_ns_values))
    overall_bubble_ratio = (
        0.0
        if total_step_ns <= 0
        else _clamp_ratio(float(total_wait_ns) / float(total_step_ns))
    )

    step_decomposition_mean = {
        "step_total_ns": _mean(step_total_ns_values),
        "expert_wait_ns": _mean(expert_wait_ns_values),
        "route_submit_ns": _mean(route_submit_ns_values),
        "completion_handoff_ns": _mean(completion_handoff_ns_values),
        "non_wait_ns": _mean(non_wait_ns_values),
        "bubble_ratio": _mean_float(bubble_ratio_values),
        "step_total_p50_ns": _percentile(step_total_ns_values, 50),
        "step_total_p95_ns": _percentile(step_total_ns_values, 95),
        "step_total_p99_ns": _percentile(step_total_ns_values, 99),
        "expert_wait_p50_ns": _percentile(expert_wait_ns_values, 50),
        "expert_wait_p95_ns": _percentile(expert_wait_ns_values, 95),
        "expert_wait_p99_ns": _percentile(expert_wait_ns_values, 99),
        "bubble_ratio_p50": _percentile(
            [int(v * 1_000_000_000) for v in bubble_ratio_values], 50
        ),
        "bubble_ratio_p95": _percentile(
            [int(v * 1_000_000_000) for v in bubble_ratio_values], 95
        ),
        "bubble_ratio_p99": _percentile(
            [int(v * 1_000_000_000) for v in bubble_ratio_values], 99
        ),
    }
    for key in ("bubble_ratio_p50", "bubble_ratio_p95", "bubble_ratio_p99"):
        value = step_decomposition_mean.get(key)
        if isinstance(value, (int, float)):
            step_decomposition_mean[key] = float(value) / 1_000_000_000.0

    step_decomposition_percentiles = {
        "step_total_ns": {
            "p50": _percentile(step_total_ns_values, 50),
            "p95": _percentile(step_total_ns_values, 95),
            "p99": _percentile(step_total_ns_values, 99),
        },
        "expert_wait_ns": {
            "p50": _percentile(expert_wait_ns_values, 50),
            "p95": _percentile(expert_wait_ns_values, 95),
            "p99": _percentile(expert_wait_ns_values, 99),
        },
        "bubble_ratio": {
            "p50": _percentile(
                [int(v * 1_000_000_000) for v in bubble_ratio_values], 50
            ),
            "p95": _percentile(
                [int(v * 1_000_000_000) for v in bubble_ratio_values], 95
            ),
            "p99": _percentile(
                [int(v * 1_000_000_000) for v in bubble_ratio_values], 99
            ),
        },
    }
    bubble_percentiles = step_decomposition_percentiles.get("bubble_ratio")
    if isinstance(bubble_percentiles, dict):
        for key, value in list(bubble_percentiles.items()):
            if isinstance(value, (int, float)):
                bubble_percentiles[key] = float(value) / 1_000_000_000.0

    layers: set[int] = set()
    for item in iterations:
        layer_map = item.get("layer_wait_ns")
        if isinstance(layer_map, dict):
            for layer in layer_map:
                if isinstance(layer, int):
                    layers.add(layer)

    per_layer_bubble: dict[str, dict[str, float | int | None]] | None
    if not layers:
        per_layer_bubble = None
    else:
        per_layer_bubble = {}
        for layer in sorted(layers):
            layer_values: list[int] = []
            for item in iterations:
                layer_map_obj = item.get("layer_wait_ns")
                if not isinstance(layer_map_obj, dict):
                    layer_values.append(0)
                    continue
                raw_value = layer_map_obj.get(layer, 0)
                layer_values.append(_safe_int(raw_value))
            layer_wait_total = int(sum(layer_values))
            layer_bubble_ratio = (
                0.0
                if total_step_ns <= 0
                else _clamp_ratio(
                    float(layer_wait_total) / float(total_step_ns)
                )
            )
            per_layer_bubble[str(layer)] = {
                "total_wait_ns": layer_wait_total,
                "mean_wait_ns": _mean(layer_values),
                "p50_wait_ns": _percentile(layer_values, 50),
                "p95_wait_ns": _percentile(layer_values, 95),
                "p99_wait_ns": _percentile(layer_values, 99),
                "bubble_ratio": layer_bubble_ratio,
            }

    event_counts = [
        int(item.get("io_profiler_event_count", 0)) for item in iterations
    ]
    sync_counts = [int(item.get("sync_event_count", 0)) for item in iterations]

    return {
        "overall_bubble_ratio": overall_bubble_ratio,
        "per_layer_bubble": per_layer_bubble,
        "step_decomposition_mean": step_decomposition_mean,
        "step_decomposition_percentiles": step_decomposition_percentiles,
        "io_profiler_events": {
            "mean_events_per_step": _mean(event_counts),
            "mean_sync_wait_events_per_step": _mean(sync_counts),
        },
    }


def blocked_payload(
    *,
    reason: str,
    env: dict[str, Any],
    args: argparse.Namespace,
    mode: str,
) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": reason,
        "environment": env,
        "requested_model": args.model,
        "offload_dir": args.offload_dir,
        "mode": mode,
        "io_mode": mode,
        "warmup": args.warmup,
        "iters": args.iters,
        "measurement_mode": "max_new_tokens=1",
        "overall_bubble_ratio": 0.0,
        "per_layer_bubble": None,
        "step_decomposition_mean": _empty_step_decomposition(),
        "step_decomposition_percentiles": _empty_step_decomposition(),
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

    print("=== MoE-Infinity Pipeline Bubble Benchmark ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUDA available: {env['cuda_available']}")

    mode = "host-only" if args.host_only else "disk"
    actual_offload_dir = args.offload_dir
    cleanup: Callable[[], None] = lambda: None

    if not env["cuda_available"]:
        write_json(
            output_path,
            blocked_payload(
                reason="No CUDA device", env=env, args=args, mode=mode
            ),
        )
        return 0

    try:
        profiler = setup_profiler()
    except Exception as exc:
        write_json(
            output_path,
            blocked_payload(
                reason=f"IOProfiler init failed: {type(exc).__name__}: {exc}",
                env=env,
                args=args,
                mode=mode,
            ),
        )
        return 0

    try:
        actual_offload_dir, mode, cleanup = setup_offload_dir(args)
        model, tokenizer = load_model_and_tokenizer(
            args.model, actual_offload_dir
        )
    except Exception as exc:
        cleanup()
        write_json(
            output_path,
            blocked_payload(
                reason=f"{type(exc).__name__}: {exc}",
                env=env,
                args=args,
                mode=mode,
            ),
        )
        return 0

    try:
        input_ids = build_prompt_input_ids(tokenizer)

        for _ in range(args.warmup):
            _ = run_one_iteration(
                model=model,
                tokenizer=tokenizer,
                profiler=profiler,
                input_ids=input_ids,
            )

        iterations: list[dict[str, Any]] = []
        for _ in range(args.iters):
            result = run_one_iteration(
                model=model,
                tokenizer=tokenizer,
                profiler=profiler,
                input_ids=input_ids,
            )
            iterations.append(result)

        summary = summarize_iterations(iterations)
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
            "measurement_mode": "max_new_tokens=1",
            **summary,
        }
        write_json(output_path, payload)
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
