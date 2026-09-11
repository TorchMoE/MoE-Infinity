# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Execution core for the KV-cache quantization A/B benchmark.

Boots the serving engine once per (format, context, batch) cell through the
public ``MoE`` entrypoint with ``kv_cache_format`` set, drives a fixed
prefill+decode workload, and reports the effective format decision alongside
TTFT / decode-throughput / inter-token latencies and memory counters.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from benchmarks.serving.kv_cache_quantization import (
        BenchmarkConfig,
        BenchmarkResult,
    )


def _resolve_revision(model_revision: str | None, model: str) -> str:
    if model_revision:
        return model_revision
    return f"{model}@pinned"


def _prompt_token_ids(tokenizer, context_length: int) -> list[int]:
    base = tokenizer.encode(
        "The history of computing spans mechanical calculators, "
        "electromechanical relays, vacuum tubes, transistors, and "
        "integrated circuits. "
    )
    if not base:
        base = [tokenizer.eos_token_id or 0]
    ids = (base * ((context_length // len(base)) + 1))[:context_length]
    return list(ids)


def _run_single_inprocess(
    *,
    config: "BenchmarkConfig",
    storage_format: str,
    context_length: int,
    batch_size: int,
) -> "BenchmarkResult":
    from transformers import AutoTokenizer

    from benchmarks.serving.kv_cache_quantization import (
        BenchmarkResult,
        transfer_precision_for,
    )
    from benchmarks.serving.kv_offload_benchmark import _build_engine_config
    from moe_infinity import MoE
    from moe_infinity.serving.engine import ContinuousBatchingEngine
    from moe_infinity.serving.sequence import SamplingParams

    offload_path = Path(config.offload_dir) / storage_format
    offload_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = MoE(
        config.model,
        {
            "offload_path": str(offload_path),
            "device_memory_ratio": 0.75,
        },
    )
    engine_config = _build_engine_config(model.model, kv_cache_ratio=0.05)
    engine_config["kv_cache_format"] = storage_format
    engine_config["max_batch_size"] = max(batch_size, 1)
    engine = ContinuousBatchingEngine(
        model=model.model,
        engine=model.engine,
        config=engine_config,
        tokenizer=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()

    prompt = _prompt_token_ids(tokenizer, context_length)
    sampling = SamplingParams(temperature=0.0, max_tokens=config.decode_tokens)

    for warmup in range(config.warmups):
        engine.add_request(
            request_id=f"warmup-{storage_format}-{warmup}",
            prompt_token_ids=prompt[: min(64, context_length)],
            sampling_params=sampling,
        )
    if config.warmups:
        engine.run_until_done()

    for row in range(batch_size):
        engine.add_request(
            request_id=f"bench-{storage_format}-{context_length}-{row}",
            prompt_token_ids=list(prompt),
            sampling_params=sampling,
        )

    first_token_at: float | None = None
    token_times: list[float] = []
    started = time.perf_counter()
    while engine.has_pending_requests():
        outputs = engine.step()
        now = time.perf_counter()
        if outputs:
            if first_token_at is None:
                first_token_at = now
            token_times.extend([now] * len(outputs))
    finished = time.perf_counter()

    total_tokens = len(token_times)
    decode_span = finished - (
        first_token_at if first_token_at is not None else started
    )
    decode_tokens_per_s = total_tokens / decode_span if decode_span > 0 else 0.0
    itls_ms = [
        (b - a) * 1000.0 for a, b in zip(token_times, token_times[1:])
    ] or [0.0]

    stats = engine.get_stats()
    swap = stats.get("kv_swap", {}) if isinstance(stats, dict) else {}
    fmt_stats = engine.kv_cache_format_stats()
    kv_store = getattr(engine, "kv_store", None)

    result = BenchmarkResult(
        model_revision=_resolve_revision(config.model_revision, config.model),
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        requested_format=str(fmt_stats["requested_kv_cache_format"]),
        effective_format=str(fmt_stats["effective_kv_cache_format"]),
        format_decision_reason=(
            None
            if fmt_stats["kv_cache_format_decision_reason"] is None
            else str(fmt_stats["kv_cache_format_decision_reason"])
        ),
        attention_backend=str(fmt_stats["kv_cache_execution_backend"]),
        storage_format=str(fmt_stats["effective_kv_cache_format"]),
        transfer_precision=transfer_precision_for(
            str(fmt_stats["effective_kv_cache_format"])
        ),
        execution_dtype=str(engine.dtype),
        context_length=context_length,
        batch_size=batch_size,
        ttft_ms=(((first_token_at or started) - started) * 1000.0),
        decode_tokens_per_s=decode_tokens_per_s,
        itl_p50_ms=statistics.median(itls_ms),
        itl_p99_ms=(
            statistics.quantiles(itls_ms, n=100)[98]
            if len(itls_ms) >= 100
            else max(itls_ms)
        ),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        descriptor_cache_bytes=(
            int(kv_store.payload.numel() * kv_store.payload.element_size())
            if kv_store is not None
            else 0
        ),
        measured_cache_bytes=(
            int(kv_store.payload.numel() * kv_store.payload.element_size())
            if kv_store is not None
            else 0
        ),
        d2h_swap_bytes=int(swap.get("d2h_bytes_total", 0) or 0),
        h2d_swap_bytes=int(swap.get("h2d_bytes_total", 0) or 0),
        scratch_peak_bytes=0,
    )

    engine.shutdown()
    del model
    torch.cuda.empty_cache()
    return result


def run_single(
    *,
    config: "BenchmarkConfig",
    storage_format: str,
    context_length: int,
    batch_size: int,
) -> "BenchmarkResult":
    import json
    import subprocess
    import sys

    from benchmarks.serving.kv_cache_quantization import BenchmarkResult

    # Each cell boots a full model; the native runtime supports one model
    # per process, so every cell runs in a fresh interpreter.
    payload = {
        "model": config.model,
        "offload_dir": config.offload_dir,
        "model_revision": config.model_revision,
        "decode_tokens": config.decode_tokens,
        "warmups": config.warmups,
        "storage_format": storage_format,
        "context_length": context_length,
        "batch_size": batch_size,
    }
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.serving._kv_quant_runner"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    marker = "KVQUANT_RESULT_JSON:"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            return BenchmarkResult(**json.loads(line[len(marker) :]))
    raise RuntimeError(
        f"cell subprocess failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )


def _cell_main() -> int:
    import json
    import sys
    from dataclasses import dataclass, field

    spec = json.loads(sys.stdin.read())

    @dataclass
    class _CellConfig:
        model: str
        offload_dir: str
        model_revision: str
        decode_tokens: int = 128
        warmups: int = 0
        formats: list = field(default_factory=list)
        context_lengths: list = field(default_factory=list)
        batch_sizes: list = field(default_factory=list)
        strict_fallback: bool = False
        repeats: int = 1

    config = _CellConfig(
        model=spec["model"],
        offload_dir=spec["offload_dir"],
        model_revision=spec["model_revision"],
        decode_tokens=int(spec["decode_tokens"]),
        warmups=int(spec["warmups"]),
    )
    result = _run_single_inprocess(
        config=config,
        storage_format=spec["storage_format"],
        context_length=int(spec["context_length"]),
        batch_size=int(spec["batch_size"]),
    )
    print("KVQUANT_RESULT_JSON:" + json.dumps(result.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cell_main())
