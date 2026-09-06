"""Nsight Systems CLI -> IBP feasibility decision tuple.

Reads a .nsys-rep produced by `nsys profile -t cuda,nvtx ...` and computes the
per-step transfer/compute breakdown that the IBP feasibility plan
(.sisyphus/plans/ibp-feasibility-profile.md) frozen criterion expects.

This module shells out to `nsys stats` for `nvtx_pushpop_sum` and
`cuda_gpu_mem_size_sum` (both confirmed available on the pinned 2025.1.1.0
container nsys per environment.md). It NEVER hardcodes PCIe theoretical
bandwidth - the caller passes link width/gen sampled at runtime via
`nvidia-smi --query-gpu=pcie.link.width.current,pcie.link.gen.current`.

Decision criterion is FROZEN per the plan:
  1. T_transfer / T_step < 0.10  -> NO_GO
  2. Util_pcie < 0.50            -> NO_GO
  3. T_transfer/T_step >= 0.10 AND Util_pcie >= 0.70 -> PROCEED
  4. otherwise                    -> DEFER
The literal floats 0.10, 0.50, 0.70 must appear in source so a grep against
the plan's QA scenario can verify alignment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

NVTX_REPORT = "nvtx_pushpop_sum"
MEMCPY_REPORT = "cuda_gpu_mem_size_sum"
GPU_TRACE_REPORT = "cuda_gpu_sum"
CUDA_API_REPORT = "cuda_api_sum"

REQUIRED_RANGE_NAMES: tuple[str, ...] = (
    "cpu_to_gpu",
    "disk_to_cpu",
    "gpu_to_cpu",
    "cuda_stream_sync",
    "expert_compute",
    "expert_wait_barrier",
    "gpu_route_submit",
    "gpu_route_handoff",
    "gpu_route_fallback",
    "expert_completion_handoff",
)

PCIE_LANE_GBPS: dict[int, float] = {
    1: 0.250,
    2: 0.500,
    3: 0.985,
    4: 1.969,
    5: 3.938,
    6: 7.563,
}


def _resolve_nsys_binary() -> str:
    explicit = os.environ.get("NSYS_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    candidates = [
        "/opt/nvidia/nsight-systems-cli/2026.1.1/target-linux-x64/nsys",
        "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    found = shutil.which("nsys")
    if found:
        return found
    raise RuntimeError(
        "nsys not found; set NSYS_BIN or install Nsight Systems CLI"
    )


def _run_nsys_stats(rep_path: str, report: str) -> list[dict[str, Any]]:
    nsys = _resolve_nsys_binary()
    proc = subprocess.run(
        [
            nsys,
            "stats",
            "--report",
            report,
            "--format",
            "json",
            "--output",
            "-",
            rep_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = proc.stdout
    m = re.search(r"^\s*[\[\{]", raw, re.MULTILINE)
    if not m:
        return []
    body = raw[m.start() :]
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        for k in ("rows", "data"):
            if k in data and isinstance(data[k], list):
                return data[k]
        return []
    if isinstance(data, list):
        return data
    return []


def _normalize_range_name(raw: str) -> str:
    return raw.lstrip(":").strip()


def parse_nsys_report(rep_path: str) -> dict[str, Any]:
    nvtx_rows = _run_nsys_stats(rep_path, NVTX_REPORT)
    memcpy_rows = _run_nsys_stats(rep_path, MEMCPY_REPORT)
    gpu_trace_rows = _run_nsys_stats(rep_path, GPU_TRACE_REPORT)
    cuda_api_rows = _run_nsys_stats(rep_path, CUDA_API_REPORT)

    ranges: dict[str, dict[str, float | int]] = {}
    for r in nvtx_rows:
        name = _normalize_range_name(str(r.get("Range") or r.get("Name") or ""))
        if not name:
            continue
        total_ns = float(r.get("Total Time (ns)", 0) or 0)
        instances = int(r.get("Instances", 0) or 0)
        avg_ns = float(r.get("Avg (ns)", 0) or 0)
        med_ns = float(r.get("Med (ns)", 0) or 0)
        ranges[name] = {
            "total_ns": total_ns,
            "count": instances,
            "mean_ns": avg_ns,
            "p50_ns": med_ns,
        }

    memcpy: dict[str, int | float] = {
        "h2d_bytes": 0,
        "h2d_count": 0,
        "d2h_bytes": 0,
        "d2h_count": 0,
        "d2d_bytes": 0,
        "d2d_count": 0,
    }
    for r in memcpy_rows:
        op = str(r.get("Operation") or r.get("Name") or "")
        count = int(r.get("Count", 0) or 0)
        bytes_total = 0
        if "Total (MB)" in r:
            bytes_total = int(float(r["Total (MB)"]) * 1024 * 1024)
        elif "Total (KB)" in r:
            bytes_total = int(float(r["Total (KB)"]) * 1024)
        elif "Total (B)" in r:
            bytes_total = int(r["Total (B)"])
        if "Host-to-Device" in op:
            memcpy["h2d_bytes"] = bytes_total
            memcpy["h2d_count"] = count
        elif "Device-to-Host" in op:
            memcpy["d2h_bytes"] = bytes_total
            memcpy["d2h_count"] = count
        elif "Device-to-Device" in op:
            memcpy["d2d_bytes"] = bytes_total
            memcpy["d2d_count"] = count

    gpu_memcpy_ns: dict[str, int] = {"h2d": 0, "d2h": 0, "d2d": 0}
    for r in gpu_trace_rows:
        name = str(r.get("Name") or r.get("Operation") or "")
        total_ns = int(r.get("Total Time (ns)", 0) or 0)
        if "Host-to-Device" in name:
            gpu_memcpy_ns["h2d"] = total_ns
        elif "Device-to-Host" in name:
            gpu_memcpy_ns["d2h"] = total_ns
        elif "Device-to-Device" in name:
            gpu_memcpy_ns["d2d"] = total_ns

    cuda_api = {
        "stream_synchronize_count": sum(
            int(row.get("Instances", 0) or row.get("Count", 0) or 0)
            for row in cuda_api_rows
            if "cudaStreamSynchronize"
            in str(row.get("Name") or row.get("Operation") or "")
        ),
        "device_synchronize_count": sum(
            int(row.get("Instances", 0) or row.get("Count", 0) or 0)
            for row in cuda_api_rows
            if "cudaDeviceSynchronize"
            in str(row.get("Name") or row.get("Operation") or "")
        ),
    }

    duration_ns = 0
    for name, stats in ranges.items():
        duration_ns = max(duration_ns, int(stats["total_ns"]))

    return {
        "ranges": ranges,
        "memcpy": memcpy,
        "gpu_memcpy_ns": gpu_memcpy_ns,
        "cuda_api": cuda_api,
        "duration_ns": duration_ns,
    }


def pcie_theoretical_gbps(link_width: int, link_gen: int) -> float:
    if link_gen not in PCIE_LANE_GBPS:
        raise ValueError(
            f"unknown PCIe gen {link_gen}; supported: {sorted(PCIE_LANE_GBPS)}"
        )
    if link_width <= 0:
        raise ValueError(f"link_width must be positive, got {link_width}")
    return PCIE_LANE_GBPS[link_gen] * link_width


VERDICT_NO_GO = "NO_GO"
VERDICT_DEFER = "DEFER"
VERDICT_PROCEED = "PROCEED"


def _apply_criterion(
    t_transfer_ns: float, t_step_ns: float, util_pcie: float
) -> str:
    if t_step_ns <= 0:
        return VERDICT_NO_GO
    transfer_fraction = t_transfer_ns / t_step_ns
    if transfer_fraction < 0.10:
        return VERDICT_NO_GO
    if util_pcie < 0.50:
        return VERDICT_NO_GO
    if transfer_fraction >= 0.10 and util_pcie >= 0.70:
        return VERDICT_PROCEED
    return VERDICT_DEFER


def gpu_routing_trace_verdict(ranges, cuda_api, profile):
    if profile is None:
        return {
            "decision": "UNAVAILABLE",
            "reasons": ["profile_json_not_provided"],
        }
    routing = profile["routing"]
    enabled = profile["config"]["gpu_only_expert_routing"] == "on"
    reasons = []
    if int(ranges.get("gpu_route_fallback", {}).get("count", 0)) > 0:
        reasons.append("gpu_route_fallback_present")
    if int(cuda_api.get("stream_synchronize_count", 0)) > 0:
        reasons.append("stream_synchronize_present")
    if int(cuda_api.get("device_synchronize_count", 0)) > 0:
        reasons.append("device_synchronize_present")
    if enabled and routing["route_batches"] > 0:
        if int(ranges.get("gpu_route_handoff", {}).get("count", 0)) == 0:
            reasons.append("missing_route_handoff")
        if (
            int(ranges.get("expert_completion_handoff", {}).get("count", 0))
            == 0
        ):
            reasons.append("missing_completion_handoff")
    if routing["route_failures"] > 0:
        reasons.append("native_route_failure")
    if routing["fallback_count"] > 0:
        reasons.append("unexpected_eager_fallback")
    return {
        "decision": "ROLLBACK" if reasons else "KEEP",
        "reasons": reasons,
    }


def summarise(
    rep_path: str,
    step_count: int,
    hw: dict[str, Any],
    real_total_ns: int | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = parse_nsys_report(rep_path)
    ranges = report["ranges"]
    memcpy = report["memcpy"]
    gpu_memcpy_ns = report["gpu_memcpy_ns"]

    def _range_total(name: str) -> int:
        entry = ranges.get(name)
        if not entry:
            return 0
        return int(entry["total_ns"])

    t_sync = _range_total("cuda_stream_sync")
    t_expert_wait = _range_total("expert_wait_barrier")
    t_expert_compute = _range_total("expert_compute")
    t_disk = _range_total("disk_to_cpu")

    t_h2d = int(gpu_memcpy_ns["h2d"])
    t_d2h = int(gpu_memcpy_ns["d2h"])
    t_d2d = int(gpu_memcpy_ns["d2d"])

    h2d_bytes = int(memcpy["h2d_bytes"])
    d2d_bytes = int(memcpy["d2d_bytes"])

    duration_ns = report["duration_ns"]
    if step_count <= 0:
        step_count = 1
    if real_total_ns is not None and real_total_ns > 0:
        t_step = max(int(real_total_ns / step_count), 1)
    else:
        t_step = max(int(duration_ns / step_count), 1)
    t_transfer = t_h2d + t_disk + t_d2h + t_d2d
    t_compute_useful = max(t_step - t_expert_wait, 0)
    bytes_h2d_per_step = h2d_bytes / step_count

    bw_h2d_observed = 0.0
    if t_h2d > 0:
        bw_h2d_observed = h2d_bytes / (t_h2d / 1e9) / 1e9

    link_width = int(hw.get("link_width", 16))
    link_gen = int(hw.get("link_gen", 4))
    bw_pcie_theoretical = pcie_theoretical_gbps(link_width, link_gen)

    util_pcie = (
        bw_h2d_observed / bw_pcie_theoretical
        if bw_pcie_theoretical > 0
        else 0.0
    )

    bw_d2d_observed = 0.0
    if t_d2d > 0:
        bw_d2d_observed = d2d_bytes / (t_d2d / 1e9) / 1e9
    util_pcie_d2d = (
        bw_d2d_observed / bw_pcie_theoretical
        if bw_pcie_theoretical > 0
        else 0.0
    )

    verdict = _apply_criterion(
        t_transfer_ns=float(t_transfer),
        t_step_ns=float(t_step),
        util_pcie=util_pcie,
    )

    cuda_api = report.get(
        "cuda_api",
        {"stream_synchronize_count": 0, "device_synchronize_count": 0},
    )
    d2h_count = int(memcpy["d2h_count"])
    d2h_bytes = int(memcpy["d2h_bytes"])
    routing_sync = {
        "gpu_route_submit_ns": _range_total("gpu_route_submit"),
        "gpu_route_handoff_ns": _range_total("gpu_route_handoff"),
        "gpu_route_fallback_ns": _range_total("gpu_route_fallback"),
        "expert_completion_handoff_ns": _range_total(
            "expert_completion_handoff"
        ),
        "device_to_host_memcpy_count": d2h_count,
        "device_to_host_memcpy_bytes": d2h_bytes,
        "stream_synchronize_count": int(cuda_api["stream_synchronize_count"]),
        "device_synchronize_count": int(cuda_api["device_synchronize_count"]),
    }

    gpu_routing_verdict = gpu_routing_trace_verdict(ranges, cuda_api, profile)

    return {
        "routing_sync": routing_sync,
        "gpu_routing_verdict": gpu_routing_verdict,
        "T_step_ns": t_step,
        "T_h2d_ns": t_h2d,
        "T_disk_ns": t_disk,
        "T_d2h_ns": t_d2h,
        "T_d2d_ns": t_d2d,
        "T_sync_ns": t_sync,
        "T_expert_wait_ns": t_expert_wait,
        "T_expert_compute_ns": t_expert_compute,
        "T_compute_useful_ns": t_compute_useful,
        "bytes_h2d_per_step": bytes_h2d_per_step,
        "bytes_d2d_total": d2d_bytes,
        "BW_h2d_observed_GBps": bw_h2d_observed,
        "BW_d2d_observed_GBps": bw_d2d_observed,
        "BW_pcie_theoretical_GBps": bw_pcie_theoretical,
        "Util_pcie": util_pcie,
        "Util_pcie_d2d": util_pcie_d2d,
        "verdict": verdict,
        "ranges": ranges,
        "memcpy": memcpy,
        "step_count": step_count,
        "hw": {"link_width": link_width, "link_gen": link_gen},
    }


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("rep_path")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--link-width", type=int, default=16)
    parser.add_argument("--link-gen", type=int, default=4)
    parser.add_argument("--real-total-ns", type=int)
    parser.add_argument("--profile-json")
    return parser.parse_args(argv)


def _cli() -> int:
    args = parse_cli_args()
    profile = None
    if args.profile_json is not None:
        with open(args.profile_json, "r", encoding="utf-8") as handle:
            profile = json.load(handle)
        if profile.get("schema_version") != "gpu-routing-decision-profile-v1":
            print(
                "profile-json must be gpu-routing-decision-profile-v1, got "
                f"{profile.get('schema_version')!r}",
                file=sys.stderr,
            )
            return 2
    out = summarise(
        args.rep_path,
        args.steps,
        {"link_width": args.link_width, "link_gen": args.link_gen},
        args.real_total_ns,
        profile=profile,
    )
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
