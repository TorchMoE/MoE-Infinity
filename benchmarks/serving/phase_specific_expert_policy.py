#!/usr/bin/env python3
"""Measure phase-specific expert policy behavior through one OpenAI server.

Run this client once against a server started with the policy disabled and once
against the same server configuration with the policy enabled.  The client does
not mutate server configuration or launch separate prefill/decode workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import requests
import torch
import transformers


@dataclass(frozen=True, order=True)
class BenchmarkCell:
    prompt_tokens: int
    output_tokens: int
    concurrency: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def build_matrix(
    prompt_lengths: Sequence[int] = (128, 2048),
    output_lengths: Sequence[int] = (16, 256),
    concurrency_levels: Sequence[int] = (1, 8),
) -> list[BenchmarkCell]:
    """Build the four singleton cells and decode-heavy concurrent cells."""
    prompts = _positive_unique(prompt_lengths, "prompt lengths")
    outputs = _positive_unique(output_lengths, "output lengths")
    concurrencies = _positive_unique(concurrency_levels, "concurrency")
    singleton = 1 if 1 in concurrencies else concurrencies[0]
    cells = {
        BenchmarkCell(prompt, output, singleton)
        for prompt in prompts
        for output in outputs
    }
    longest_output = max(outputs)
    cells.update(
        BenchmarkCell(prompt, longest_output, concurrency)
        for prompt in prompts
        for concurrency in concurrencies
        if concurrency != singleton
    )
    return sorted(cells)


def _positive_unique(values: Sequence[int], label: str) -> list[int]:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{label} must contain positive integers")
    return list(dict.fromkeys(values))


def summarize(
    submitted: float, token_times: Sequence[float]
) -> dict[str, float | int]:
    if not token_times:
        raise ValueError(
            "token_times must contain at least one token timestamp"
        )
    if any(
        later < earlier for earlier, later in zip(token_times, token_times[1:])
    ):
        raise ValueError("token_times must be monotonic")
    gaps = [
        later - earlier for earlier, later in zip(token_times, token_times[1:])
    ]
    return {
        "ttft_s": token_times[0] - submitted,
        "tpot_s": statistics.fmean(gaps) if gaps else 0.0,
        "e2e_s": token_times[-1] - submitted,
        "output_tokens": len(token_times),
    }


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = percentile_value / 100.0 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in ("ttft_s", "tpot_s", "e2e_s"):
        values = [_number(row[metric], metric) for row in rows]
        result[metric] = {
            "p50": round(percentile(values, 50), 12),
            "p90": round(percentile(values, 90), 12),
            "p99": round(percentile(values, 99), 12),
        }
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return cast(list[Any], value)


def _request_signature(request: Mapping[str, object]) -> tuple[object, ...]:
    return (
        request.get("prompt_tokens"),
        request.get("requested_output_tokens"),
        request.get("output_tokens"),
        tuple(cast(Sequence[int], request.get("token_ids", []))),
    )


def compare_reports(
    off_report: Mapping[str, object], on_report: Mapping[str, object]
) -> list[dict[str, object]]:
    """Validate A/B parity and return signed on-minus-off metric deltas."""
    if off_report.get("policy") != "off" or on_report.get("policy") != "on":
        raise ValueError("reports must be ordered as policy off then policy on")
    if off_report.get("environment") != on_report.get("environment"):
        raise ValueError("environment fingerprints differ")

    off_cells = _list(off_report.get("cells"), "off cells")
    on_cells = _list(on_report.get("cells"), "on cells")
    if len(off_cells) != len(on_cells):
        raise ValueError("request counts differ")

    comparisons: list[dict[str, object]] = []
    for off_raw, on_raw in zip(off_cells, on_cells):
        off_cell = _object(off_raw, "off cell")
        on_cell = _object(on_raw, "on cell")
        if off_cell.get("cell") != on_cell.get("cell"):
            raise ValueError("matrix cells differ")
        off_requests = _list(off_cell.get("requests"), "off requests")
        on_requests = _list(on_cell.get("requests"), "on requests")
        if len(off_requests) != len(on_requests):
            raise ValueError("request counts differ")
        for off_request_raw, on_request_raw in zip(off_requests, on_requests):
            off_request = _object(off_request_raw, "off request")
            on_request = _object(on_request_raw, "on request")
            if off_request.get("prompt_tokens") != on_request.get(
                "prompt_tokens"
            ):
                raise ValueError("prompt length differs")
            if off_request.get("requested_output_tokens") != on_request.get(
                "requested_output_tokens"
            ) or off_request.get("output_tokens") != on_request.get(
                "output_tokens"
            ):
                raise ValueError("output length differs")
            if (
                _request_signature(off_request)[-1]
                != _request_signature(on_request)[-1]
            ):
                raise ValueError("generated token IDs differ")
        comparisons.append(
            {
                "cell": off_cell["cell"],
                "ttft_s_delta": _metric_deltas(off_cell, on_cell, "ttft_s"),
                "tpot_s_delta": _metric_deltas(off_cell, on_cell, "tpot_s"),
            }
        )
    return comparisons


def _metric_deltas(
    off_cell: Mapping[str, object], on_cell: Mapping[str, object], metric: str
) -> dict[str, float]:
    off_summary = _object(off_cell.get("summary"), "off summary")
    on_summary = _object(on_cell.get("summary"), "on summary")
    off_metric = _object(off_summary.get(metric), f"off {metric}")
    on_metric = _object(on_summary.get(metric), f"on {metric}")
    if off_metric.keys() != on_metric.keys():
        raise ValueError(f"{metric} percentiles differ")
    return {
        key: round(
            _number(on_metric[key], metric) - _number(off_metric[key], metric),
            12,
        )
        for key in off_metric
    }


def _stream_request(
    *,
    server_url: str,
    model: str,
    prompt_tokens: int,
    output_tokens: int,
    seed: int,
    tokenizer: Any,
    barrier: threading.Barrier,
) -> dict[str, object]:
    payload = {
        "model": model,
        "prompt": [1] * prompt_tokens,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
    }
    barrier.wait(timeout=60)
    submitted = time.perf_counter()
    token_times: list[float] = []
    text_parts: list[str] = []
    with requests.post(
        f"{server_url.rstrip('/')}/v1/completions",
        json=payload,
        stream=True,
        timeout=(30, 1800),
    ) as response:
        response.raise_for_status()
        for line in cast(Iterable[bytes], response.iter_lines()):
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            chunk = json.loads(body)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            text = choices[0].get("text", "")
            if text:
                token_times.append(time.perf_counter())
                text_parts.append(str(text))
    timing = summarize(submitted, token_times)
    generated_text = "".join(text_parts)
    token_ids = [
        int(token)
        for token in tokenizer.encode(generated_text, add_special_tokens=False)
    ]
    return {
        **timing,
        "prompt_tokens": prompt_tokens,
        "requested_output_tokens": output_tokens,
        "token_ids": token_ids,
        "text_sha256": hashlib.sha256(generated_text.encode()).hexdigest(),
    }


def run_cell(
    cell: BenchmarkCell,
    *,
    server_url: str,
    model: str,
    warmup: int,
    repeats: int,
    seed: int,
    tokenizer: Any,
) -> dict[str, object]:
    measured: list[dict[str, object]] = []
    for round_index in range(warmup + repeats):
        barrier = threading.Barrier(cell.concurrency)
        with ThreadPoolExecutor(max_workers=cell.concurrency) as executor:
            futures = [
                executor.submit(
                    _stream_request,
                    server_url=server_url,
                    model=model,
                    prompt_tokens=cell.prompt_tokens,
                    output_tokens=cell.output_tokens,
                    seed=seed,
                    tokenizer=tokenizer,
                    barrier=barrier,
                )
                for _ in range(cell.concurrency)
            ]
            rows = [future.result() for future in futures]
        if round_index >= warmup:
            measured.extend(rows)
    stats = _get_json(f"{server_url.rstrip('/')}/admin/stats")
    expert_policy = _object(stats.get("expert_policy"), "expert policy stats")
    return {
        "cell": cell.as_dict(),
        "requests": measured,
        "summary": summarize_rows(measured),
        "expert_policy": expert_policy,
    }


def _get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return _object(payload, url)


def _commit() -> str:
    environment = os.environ.copy()
    environment["GIT_MASTER"] = "1"
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
        env=environment,
    ).strip()


def environment_info(
    *, model: str, tokenizer: Any, server_config: Mapping[str, object]
) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    info: dict[str, object] = {
        "commit": _commit(),
        "model": model,
        "tokenizer": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_visible_devices": visible,
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "device_memory_ratio": server_config.get("device_memory_ratio"),
        "offload_path": server_config.get("offload_path"),
        "offload_medium": "operator-provided path",
    }
    fingerprint_payload = json.dumps(info, sort_keys=True, default=str).encode()
    info["fingerprint"] = hashlib.sha256(fingerprint_payload).hexdigest()
    return info


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--policy", required=True, choices=("off", "on"))
    parser.add_argument(
        "--prompt-lengths", nargs="+", type=int, default=[128, 2048]
    )
    parser.add_argument(
        "--output-lengths", nargs="+", type=int, default=[16, 256]
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare-json",
        help="Optional opposite-policy report; validates parity and prints deltas",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be >= 0 and --repeats must be > 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    server_config = _get_json(f"{args.server_url.rstrip('/')}/v1/config")
    stats = _get_json(f"{args.server_url.rstrip('/')}/admin/stats")
    policy_stats = _object(stats.get("expert_policy"), "expert policy stats")
    expected_enabled = args.policy == "on"
    if bool(policy_stats.get("enabled", 0)) is not expected_enabled:
        raise RuntimeError(
            f"server policy state does not match --policy {args.policy!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    cells = build_matrix(
        args.prompt_lengths, args.output_lengths, args.concurrency
    )
    report: dict[str, object] = {
        "policy": args.policy,
        "command": [sys.executable, *sys.argv],
        "server_url": args.server_url,
        "environment": environment_info(
            model=args.model, tokenizer=tokenizer, server_config=server_config
        ),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "cells": [
            run_cell(
                cell,
                server_url=args.server_url,
                model=args.model,
                warmup=args.warmup,
                repeats=args.repeats,
                seed=args.seed,
                tokenizer=tokenizer,
            )
            for cell in cells
        ],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.compare_json:
        other = json.loads(Path(args.compare_json).read_text())
        if args.policy == "off":
            deltas = compare_reports(report, other)
        else:
            deltas = compare_reports(other, report)
        print(json.dumps({"deltas": deltas}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
