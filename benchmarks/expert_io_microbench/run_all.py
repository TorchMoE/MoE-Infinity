from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportUnusedCallResult=false
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DEVICE_MEMORY_RATIO = 0.5
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 100
DEFAULT_OUTPUT_JSON = "expert_io_microbench_all.json"

SCRIPT_BY_SCENARIO: dict[str, str] = {
    "routing": "bench_routing.py",
    "transfer": "bench_transfer.py",
    "compute": "bench_compute_evict.py",
    "bubble": "bench_bubble.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all expert I/O microbenchmarks and merge output with "
            "bandwidth and bottleneck analysis."
        )
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--offload-dir",
        required=True,
        help="Directory used for MoE expert offload storage",
    )
    parser.add_argument(
        "--device-memory-ratio",
        type=float,
        default=DEFAULT_DEVICE_MEMORY_RATIO,
        help="Fraction of GPU memory used for expert cache (default: 0.5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup iterations passed to each scenario benchmark",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="Measured iterations passed to each scenario benchmark",
    )
    parser.add_argument(
        "--scenario",
        choices=("all", "routing", "transfer", "compute", "bubble", "overlap"),
        default="all",
        help="Run all scenarios or a single scenario",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Path to write merged benchmark JSON",
    )
    parser.add_argument(
        "--theoretical-pcie-gbps",
        type=float,
        default=None,
        help=(
            "Optional theoretical PCIe throughput (same unit as output field). "
            "If omitted, runner tries to read from CUDA device properties."
        ),
    )
    parser.add_argument(
        "--host-only",
        action="store_true",
        help=(
            "Copy offload weights to /dev/shm (tmpfs) once in run_all and pass "
            "that tmpfs path to all scenarios to eliminate disk I/O. "
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
    *, offload_dir: str, host_only: bool
) -> tuple[str, str, Callable[[], None]]:
    if not host_only:
        return offload_dir, "disk", lambda: None

    if not os.path.exists(offload_dir):
        print(
            f"WARNING: offload dir does not exist: {offload_dir}",
            file=sys.stderr,
        )
        print(
            "Hint: provide a valid --offload-dir before enabling --host-only",
            file=sys.stderr,
        )
        raise RuntimeError(f"offload dir does not exist: {offload_dir}")

    if _is_under_dev_shm(offload_dir):
        print(
            f"Using existing tmpfs offload dir: {offload_dir}",
            file=sys.stderr,
        )
        return offload_dir, "host-only", lambda: None

    src_size = _directory_size_bytes(offload_dir)
    shm_stat = shutil.disk_usage("/dev/shm")
    required_bytes = int(src_size * 1.1)
    if shm_stat.free < required_bytes:
        free_gb = shm_stat.free / 1e9
        src_gb = src_size / 1e9
        print(
            f"WARNING: /dev/shm has {free_gb:.1f}GB free but offload needs {src_gb:.1f}GB",
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
        file=sys.stderr,
        flush=True,
    )
    shutil.copytree(offload_dir, dst, dirs_exist_ok=True)
    print("Copy done. Running in host-only (RAM) mode.", file=sys.stderr)

    def cleanup() -> None:
        shutil.rmtree(dst, ignore_errors=True)

    return dst, "host-only", cleanup


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _ns_to_ms(value: float | int | None) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value) / 1_000_000.0


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} ms"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_mean_ns(obj: Any) -> float | None:
    if not isinstance(obj, dict):
        return None
    return _safe_float(obj.get("mean_ns"))


def _extract_stage_from_scenario(
    scenarios: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    routing = scenarios.get("routing")
    if isinstance(routing, dict) and "error" not in routing:
        for stage in ("routing", "cache_lookup"):
            mean_ns = _extract_mean_ns(routing.get(stage))
            if mean_ns is not None:
                entries.append(
                    {
                        "scenario": "routing",
                        "component": stage,
                        "mean_ns": mean_ns,
                    }
                )

    transfer = scenarios.get("transfer")
    if isinstance(transfer, dict) and "error" not in transfer:
        for stage in ("disk_to_cpu", "cpu_to_gpu"):
            mean_ns = _extract_mean_ns(transfer.get(stage))
            if mean_ns is not None:
                entries.append(
                    {
                        "scenario": "transfer",
                        "component": stage,
                        "mean_ns": mean_ns,
                    }
                )

    compute = scenarios.get("compute")
    if isinstance(compute, dict) and "error" not in compute:
        for stage in ("expert_compute", "eviction", "queue_coordination"):
            mean_ns = _extract_mean_ns(compute.get(stage))
            if mean_ns is not None:
                entries.append(
                    {
                        "scenario": "compute",
                        "component": stage,
                        "mean_ns": mean_ns,
                    }
                )

    bubble = scenarios.get("bubble")
    if isinstance(bubble, dict) and "error" not in bubble:
        step = bubble.get("step_decomposition_mean")
        if isinstance(step, dict):
            for key, name in (
                ("expert_wait_ns", "expert_wait"),
                ("non_wait_ns", "non_wait"),
            ):
                mean_ns = _safe_float(step.get(key))
                if mean_ns is not None:
                    entries.append(
                        {
                            "scenario": "bubble",
                            "component": name,
                            "mean_ns": mean_ns,
                        }
                    )

    return entries


def _estimate_pcie_bandwidth_bytes_per_ns(
    theoretical_pcie_gbps: float | None,
) -> float | None:
    if (
        isinstance(theoretical_pcie_gbps, (int, float))
        and not isinstance(theoretical_pcie_gbps, bool)
        and float(theoretical_pcie_gbps) > 0.0
    ):
        return float(theoretical_pcie_gbps)

    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return None

    try:
        props = torch.cuda.get_device_properties(0)
        for attr in (
            "pcie_bandwidth_gbps",
            "pcie_link_bandwidth_gbps",
            "pcie_throughput_gbps",
            "pcie_speed_gbps",
        ):
            val = _safe_float(getattr(props, attr, None))
            if val is not None and val > 0.0:
                return val
    except Exception:
        return None
    return None


def build_bandwidth_analysis(
    scenarios: dict[str, Any],
    theoretical_pcie_gbps: float | None,
) -> dict[str, Any]:
    transfer = scenarios.get("transfer")
    if not isinstance(transfer, dict) or "error" in transfer:
        return {
            "status": "unavailable",
            "reason": "transfer scenario unavailable",
        }

    disk = transfer.get("disk_to_cpu")
    cpu = transfer.get("cpu_to_gpu")
    if not isinstance(disk, dict) or not isinstance(cpu, dict):
        return {
            "status": "unavailable",
            "reason": "transfer stats missing",
        }

    theo_bytes_per_ns = _estimate_pcie_bandwidth_bytes_per_ns(
        theoretical_pcie_gbps
    )

    out: dict[str, Any] = {
        "status": "ok",
        "theoretical_bandwidth_gbps": (
            None if theo_bytes_per_ns is None else theo_bytes_per_ns
        ),
        "note": (
            "Theoretical bandwidth comes from --theoretical-pcie-gbps when "
            "provided, otherwise runner reads CUDA device properties."
        ),
        "links": {},
    }

    for key, stats in (("disk_to_cpu", disk), ("cpu_to_gpu", cpu)):
        mean_ns = _safe_float(stats.get("mean_ns"))
        bytes_transferred = _safe_float(stats.get("bytes_transferred"))
        if mean_ns is None or mean_ns <= 0.0 or bytes_transferred is None:
            out["links"][key] = {
                "actual_bandwidth_gbps": 0.0,
                "utilization_pct": None,
                "bytes_transferred": bytes_transferred,
                "mean_ns": mean_ns,
            }
            continue

        actual_bpn = bytes_transferred / mean_ns
        util_pct = (
            None
            if theo_bytes_per_ns is None or theo_bytes_per_ns <= 0.0
            else (actual_bpn / theo_bytes_per_ns) * 100.0
        )
        out["links"][key] = {
            "actual_bandwidth_gbps": actual_bpn,
            "utilization_pct": util_pct,
            "bytes_transferred": int(bytes_transferred),
            "mean_ns": mean_ns,
        }

    return out


def build_executive_summary(scenarios: dict[str, Any]) -> dict[str, Any]:
    entries = _extract_stage_from_scenario(scenarios)
    step_ref_ns: float | None = None
    step_ref_src = "unknown"

    bubble = scenarios.get("bubble")
    if isinstance(bubble, dict) and "error" not in bubble:
        step = bubble.get("step_decomposition_mean")
        if isinstance(step, dict):
            step_ref_ns = _safe_float(step.get("step_total_ns"))
            if isinstance(step_ref_ns, float) and step_ref_ns > 0.0:
                step_ref_src = "bubble.step_decomposition_mean.step_total_ns"

    if step_ref_ns is None or step_ref_ns <= 0.0:
        transfer = scenarios.get("transfer")
        if isinstance(transfer, dict) and "error" not in transfer:
            sync = transfer.get("sync_overhead")
            if isinstance(sync, dict):
                total_sync_ms = _safe_float(sync.get("total_sync_ms"))
                sync_pct = _safe_float(sync.get("sync_pct_of_step"))
                if (
                    total_sync_ms is not None
                    and sync_pct is not None
                    and sync_pct > 0.0
                ):
                    step_ref_ns = (
                        total_sync_ms / (sync_pct / 100.0)
                    ) * 1_000_000.0
                    step_ref_src = "transfer.sync_overhead"

    if step_ref_ns is None or step_ref_ns <= 0.0:
        step_ref_ns = sum(item["mean_ns"] for item in entries)
        step_ref_src = "sum(component_mean_ns)"

    if step_ref_ns <= 0.0:
        return {
            "status": "unavailable",
            "reason": "non-positive step reference",
            "top_bottlenecks": [],
        }

    transfer = scenarios.get("transfer")
    if isinstance(transfer, dict) and "error" not in transfer:
        sync = transfer.get("sync_overhead")
        if isinstance(sync, dict):
            sync_pct = _safe_float(sync.get("sync_pct_of_step"))
            if sync_pct is not None and sync_pct > 0.0:
                sync_mean_ns = step_ref_ns * (sync_pct / 100.0)
                entries.append(
                    {
                        "scenario": "transfer",
                        "component": "sync_wait",
                        "mean_ns": sync_mean_ns,
                    }
                )

    if not entries:
        return {
            "status": "unavailable",
            "reason": "no components with mean_ns",
            "top_bottlenecks": [],
        }

    ranked = sorted(entries, key=lambda x: x["mean_ns"], reverse=True)
    top3 = ranked[:3]
    top = []
    for item in top3:
        pct = (item["mean_ns"] / step_ref_ns) * 100.0
        top.append(
            {
                "scenario": item["scenario"],
                "component": item["component"],
                "mean_ns": item["mean_ns"],
                "pct_of_step": pct,
            }
        )

    return {
        "status": "ok",
        "reference_step_ns": step_ref_ns,
        "reference_step_source": step_ref_src,
        "top_bottlenecks": top,
    }


def scenario_list(selection: str) -> list[str]:
    if selection == "all":
        return ["routing", "transfer", "compute", "bubble"]
    return [selection]


def run_overlap_scenario(
    *,
    model: str,
    offload_dir: str,
    warmup: int,
    iters: int,
    device_memory_ratio: float,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve().parent / "bench_overlap_prefetch.py"
    with tempfile.TemporaryDirectory(prefix="io_microbench_overlap_") as td:
        temp_output = Path(td) / "overlap.json"
        cmd = [
            sys.executable,
            str(script_path),
            "--model",
            model,
            "--offload-dir",
            offload_dir,
            "--policies",
            "off",
            "observe",
            "enforce",
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--device-memory-ratio",
            str(device_memory_ratio),
            "--output-json",
            str(temp_output),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return {
                "error": "overlap benchmark failed",
                "stderr": proc.stderr[-2000:],
            }
        with temp_output.open() as handle:
            return json.load(handle)


def run_one_script(
    *,
    scenario: str,
    model: str,
    offload_dir: str,
    warmup: int,
    iters: int,
    device_memory_ratio: float,
    host_only: bool,
) -> dict[str, Any]:
    script_name = SCRIPT_BY_SCENARIO[scenario]
    script_path = Path(__file__).resolve().parent / script_name

    with tempfile.TemporaryDirectory(prefix=f"io_microbench_{scenario}_") as td:
        temp_output = Path(td) / f"{scenario}.json"
        cmd = [
            sys.executable,
            str(script_path),
            "--model",
            model,
            "--offload-dir",
            offload_dir,
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--output-json",
            str(temp_output),
        ]
        if scenario == "compute":
            cmd.extend(
                [
                    "--device-memory-ratio",
                    str(device_memory_ratio),
                ]
            )
        if host_only:
            cmd.append("--host-only")
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )

        if result.returncode != 0:
            return {
                "error": (
                    f"subprocess failed rc={result.returncode}; stderr="
                    f"{result.stderr.strip()}"
                ),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        if not temp_output.exists():
            return {
                "error": "subprocess completed but output JSON missing",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        try:
            payload = json.loads(temp_output.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {
                    "error": "output JSON is not an object",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            payload.setdefault("runner_stdout", result.stdout)
            payload.setdefault("runner_stderr", result.stderr)
            return payload
        except Exception as exc:
            return {
                "error": f"failed to parse output JSON: {type(exc).__name__}: {exc}",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }


def print_human_summary(payload: dict[str, Any]) -> None:
    scenarios = payload.get("scenarios", {})
    summary = payload.get("executive_summary", {})
    bw = payload.get("bandwidth_analysis", {})

    print("=== Expert IO Microbench Integrated Summary ===", file=sys.stderr)
    print(
        f"model={payload.get('requested_model')} scenario={payload.get('scenario')}",
        file=sys.stderr,
    )

    if isinstance(scenarios, dict):
        for name in ("routing", "transfer", "compute", "bubble"):
            if name not in scenarios:
                continue
            val = scenarios[name]
            if isinstance(val, dict) and "error" in val:
                print(f"- {name}: ERROR ({val.get('error')})", file=sys.stderr)
            elif isinstance(val, dict):
                print(
                    f"- {name}: status={val.get('status', 'unknown')}",
                    file=sys.stderr,
                )

    if isinstance(summary, dict) and summary.get("status") == "ok":
        print("Top bottlenecks:", file=sys.stderr)
        top = summary.get("top_bottlenecks", [])
        if isinstance(top, list):
            for item in top:
                if not isinstance(item, dict):
                    continue
                pct = _safe_float(item.get("pct_of_step"))
                ms = _ns_to_ms(_safe_float(item.get("mean_ns")))
                print(
                    (
                        f"  * {item.get('scenario')}.{item.get('component')}: "
                        f"{_fmt_pct(pct)} of step time, {_fmt_ms(ms)}"
                    ),
                    file=sys.stderr,
                )

    if isinstance(bw, dict) and bw.get("status") == "ok":
        theo = _safe_float(bw.get("theoretical_bandwidth_gbps"))
        print(
            f"Bandwidth theoretical upper-bound: {theo if theo is not None else 'n/a'} GB/s",
            file=sys.stderr,
        )
        links = bw.get("links", {})
        if isinstance(links, dict):
            for link in ("disk_to_cpu", "cpu_to_gpu"):
                data = links.get(link)
                if not isinstance(data, dict):
                    continue
                actual = _safe_float(data.get("actual_bandwidth_gbps"))
                util = _safe_float(data.get("utilization_pct"))
                print(
                    f"  * {link}: actual={actual if actual is not None else 'n/a'} GB/s, util={_fmt_pct(util)}",
                    file=sys.stderr,
                )


def main() -> int:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if not (0.0 < args.device_memory_ratio <= 1.0):
        raise ValueError("--device-memory-ratio must be in (0, 1]")
    if (
        args.theoretical_pcie_gbps is not None
        and args.theoretical_pcie_gbps <= 0
    ):
        raise ValueError("--theoretical-pcie-gbps must be > 0 when provided")

    selected = scenario_list(args.scenario)
    scenarios: dict[str, Any] = {}

    try:
        actual_offload_dir, mode, cleanup = setup_offload_dir(
            offload_dir=args.offload_dir,
            host_only=args.host_only,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        for scenario in selected:
            try:
                if scenario == "overlap":
                    scenarios[scenario] = run_overlap_scenario(
                        model=args.model,
                        offload_dir=actual_offload_dir,
                        warmup=args.warmup,
                        iters=args.iters,
                        device_memory_ratio=args.device_memory_ratio,
                    )
                else:
                    scenarios[scenario] = run_one_script(
                        scenario=scenario,
                        model=args.model,
                        offload_dir=actual_offload_dir,
                        warmup=args.warmup,
                        iters=args.iters,
                        device_memory_ratio=args.device_memory_ratio,
                        host_only=(mode == "host-only"),
                    )
            except Exception as exc:
                scenarios[scenario] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }

        payload: dict[str, Any] = {
            "status": "PASS",
            "requested_model": args.model,
            "offload_dir": actual_offload_dir,
            "source_offload_dir": args.offload_dir,
            "mode": mode,
            "io_mode": mode,
            "host_only": mode == "host-only",
            "device_memory_ratio": args.device_memory_ratio,
            "warmup": args.warmup,
            "iters": args.iters,
            "scenario": args.scenario,
            "scenarios": scenarios,
        }

        payload["bandwidth_analysis"] = build_bandwidth_analysis(
            scenarios,
            args.theoretical_pcie_gbps,
        )
        payload["executive_summary"] = build_executive_summary(scenarios)

        write_json(Path(args.output_json), payload)
        print_human_summary(payload)
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
