from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = p / 100.0 * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_requests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    for row in rows:
        started = float(row["started_at"])
        token_times = [float(value) for value in row["token_times"]]
        if not token_times:
            continue
        ttft_ms.append((token_times[0] - started) * 1000.0)
        tpot_ms.extend(
            (right - left) * 1000.0
            for left, right in zip(token_times, token_times[1:])
        )
    return {
        "request_count": len(rows),
        "ttft_p50_ms": percentile(ttft_ms, 50.0),
        "ttft_p90_ms": percentile(ttft_ms, 90.0),
        "ttft_p99_ms": percentile(ttft_ms, 99.0),
        "tpot_p50_ms": percentile(tpot_ms, 50.0),
        "tpot_p90_ms": percentile(tpot_ms, 90.0),
        "tpot_p99_ms": percentile(tpot_ms, 99.0),
    }


async def stream_request(
    client: httpx.AsyncClient,
    url: str,
    prompt_token_ids: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    token_times: list[float] = []
    token_payloads: list[str] = []
    payload = {
        "model": "benchmark",
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    async with client.stream(
        "POST", f"{url}/v1/completions", json=payload
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                token_times.append(time.perf_counter())
                event = json.loads(line.removeprefix("data: "))
                choice = event["choices"][0]
                token_payloads.append(str(choice.get("text", "")))
    return {
        "started_at": started,
        "token_times": token_times,
        "token_payloads": token_payloads,
    }


def build_prompt_token_ids(tokenizer: Any, target_length: int) -> list[int]:
    if target_length <= 0:
        raise ValueError("prompt token target must be positive")
    base = tokenizer.encode(
        "MoE Infinity deterministic chunked prefill benchmark",
        add_special_tokens=False,
    )
    if not base:
        raise ValueError("tokenizer returned an empty benchmark prompt")
    prompt = [int(base[index % len(base)]) for index in range(target_length)]
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and any(
        token < 0 or token >= vocab_size for token in prompt
    ):
        raise ValueError("benchmark prompt contains token outside vocabulary")
    if len(prompt) != target_length:
        raise AssertionError("benchmark prompt length is not exact")
    return prompt


async def poll_stats(
    client: httpx.AsyncClient,
    url: str,
    stop: asyncio.Event,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while not stop.is_set():
        response = await client.get(f"{url}/admin/stats")
        response.raise_for_status()
        samples.append(response.json())
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.01)
        except asyncio.TimeoutError:
            pass
    return samples


async def run_arm(
    url: str,
    tokenizer: Any,
    *,
    short_requests: int,
    long_requests: int,
    short_prompt_tokens: int,
    long_prompt_tokens: int,
    max_tokens: int,
) -> dict[str, Any]:
    work = [
        ("long", build_prompt_token_ids(tokenizer, long_prompt_tokens))
        for _ in range(long_requests)
    ] + [
        ("short", build_prompt_token_ids(tokenizer, short_prompt_tokens))
        for _ in range(short_requests)
    ]
    work.sort(key=lambda item: (item[0] != "long", item[1]))
    timeout = httpx.Timeout(600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        config_response = await client.get(f"{url}/v1/config")
        config_response.raise_for_status()

        async def delayed(index: int, prompt: list[int]) -> dict[str, Any]:
            await asyncio.sleep(index * 0.005)
            try:
                return await stream_request(client, url, prompt, max_tokens)
            except Exception as exc:
                return {
                    "started_at": time.perf_counter(),
                    "token_times": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

        stop = asyncio.Event()
        stats_task = asyncio.create_task(poll_stats(client, url, stop))
        arm_started = time.perf_counter()
        rows = await asyncio.gather(
            *(delayed(index, prompt) for index, (_, prompt) in enumerate(work))
        )
        arm_seconds = time.perf_counter() - arm_started
        stop.set()
        stats_samples = await stats_task
    total_output_tokens = sum(
        len(row.get("token_payloads", [])) for row in rows
    )
    peak_used_blocks = max(
        (
            int(sample["kv_cache_num_blocks"])
            - int(sample["kv_cache_free_blocks"])
            for sample in stats_samples
        ),
        default=0,
    )
    return {
        "config": config_response.json(),
        "stats_samples": stats_samples,
        "raw_requests": rows,
        "summary": summarize_requests(rows),
        "error_count": sum("error" in row for row in rows),
        "wall_time_s": arm_seconds,
        "total_output_tokens": total_output_tokens,
        "throughput_tokens_per_s": (
            total_output_tokens / arm_seconds if arm_seconds > 0 else None
        ),
        "peak_kv_used_blocks": peak_used_blocks,
        "peak_kv_utilization": max(
            (
                1.0
                - int(sample["kv_cache_free_blocks"])
                / int(sample["kv_cache_num_blocks"])
                for sample in stats_samples
                if int(sample["kv_cache_num_blocks"]) > 0
            ),
            default=0.0,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--short-requests", type=int, default=64)
    parser.add_argument("--long-requests", type=int, default=16)
    parser.add_argument("--short-prompt-tokens", type=int, default=128)
    parser.add_argument("--long-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model-label", default="unspecified")
    parser.add_argument("--gpu-label", default="unspecified")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


async def run_paired(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    arms: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    kwargs = {
        "short_requests": args.short_requests,
        "long_requests": args.long_requests,
        "short_prompt_tokens": args.short_prompt_tokens,
        "long_prompt_tokens": args.long_prompt_tokens,
        "max_tokens": args.max_tokens,
    }
    for _ in range(args.rounds):
        arms["baseline"].append(
            await run_arm(args.baseline_url, tokenizer, **kwargs)
        )
        arms["candidate"].append(
            await run_arm(args.candidate_url, tokenizer, **kwargs)
        )
    baseline_rows = [
        row for run in arms["baseline"] for row in run["raw_requests"]
    ]
    candidate_rows = [
        row for run in arms["candidate"] for row in run["raw_requests"]
    ]
    baseline = summarize_requests(baseline_rows)
    candidate = summarize_requests(candidate_rows)
    output_parity = [
        row.get("token_payloads", []) for row in baseline_rows
    ] == [row.get("token_payloads", []) for row in candidate_rows]
    deltas = {
        key: (
            None
            if baseline[key] is None or candidate[key] is None
            else candidate[key] - baseline[key]
        )
        for key in (
            "ttft_p50_ms",
            "ttft_p90_ms",
            "ttft_p99_ms",
            "tpot_p50_ms",
            "tpot_p90_ms",
            "tpot_p99_ms",
        )
    }
    return {
        "model_label": args.model_label,
        "gpu_label": args.gpu_label,
        "workload": kwargs | {"rounds": args.rounds},
        "baseline": baseline,
        "candidate": candidate,
        "output_parity": output_parity,
        "baseline_error_count": sum("error" in row for row in baseline_rows),
        "candidate_error_count": sum("error" in row for row in candidate_rows),
        "baseline_throughput_tokens_per_s": [
            run["throughput_tokens_per_s"] for run in arms["baseline"]
        ],
        "candidate_throughput_tokens_per_s": [
            run["throughput_tokens_per_s"] for run in arms["candidate"]
        ],
        "baseline_peak_kv_used_blocks": max(
            run["peak_kv_used_blocks"] for run in arms["baseline"]
        ),
        "candidate_peak_kv_used_blocks": max(
            run["peak_kv_used_blocks"] for run in arms["candidate"]
        ),
        "baseline_peak_kv_utilization": max(
            run["peak_kv_utilization"] for run in arms["baseline"]
        ),
        "candidate_peak_kv_utilization": max(
            run["peak_kv_utilization"] for run in arms["candidate"]
        ),
        "candidate_minus_baseline_ms": deltas,
        "runs": arms,
    }


def main() -> int:
    args = parse_args()
    for name in (
        "short_requests",
        "long_requests",
        "short_prompt_tokens",
        "long_prompt_tokens",
        "max_tokens",
        "rounds",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    payload = asyncio.run(run_paired(args))
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
