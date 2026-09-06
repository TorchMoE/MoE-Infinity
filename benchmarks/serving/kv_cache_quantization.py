"""Long-context A/B benchmark matrix for the KV-cache quantization feature.

Compares the ``native`` and ``int8_sym`` KV-cache formats across a matrix of
context lengths and batch sizes, emitting a JSON array with explicit
storage/transfer/execution precision fields plus latency, throughput, and
memory measurements. Storage/transfer/execution precisions are reported
separately so a fallback is never mistaken for a quantized run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

_TRANSFER_PRECISION = {
    "native": "model_dtype",
    "int8_sym": "int8+fp16_scale",
}


@dataclass
class BenchmarkResult:
    model_revision: str
    gpu: str
    torch_version: str
    cuda_version: str
    requested_format: str
    effective_format: str
    format_decision_reason: str | None
    attention_backend: str
    storage_format: str
    transfer_precision: str
    execution_dtype: str
    context_length: int
    batch_size: int
    ttft_ms: float
    decode_tokens_per_s: float
    itl_p50_ms: float
    itl_p99_ms: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    descriptor_cache_bytes: int
    measured_cache_bytes: int
    d2h_swap_bytes: int
    h2d_swap_bytes: int
    scratch_peak_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def example(cls) -> "BenchmarkResult":
        return cls(
            model_revision="example@0000000",
            gpu="ExampleGPU",
            torch_version="2.0.0",
            cuda_version="12.0",
            requested_format="int8_sym",
            effective_format="int8_sym",
            format_decision_reason=None,
            attention_backend="native_int8",
            storage_format="int8_sym",
            transfer_precision="int8+fp16_scale",
            execution_dtype="float16",
            context_length=2048,
            batch_size=1,
            ttft_ms=12.5,
            decode_tokens_per_s=95.0,
            itl_p50_ms=10.0,
            itl_p99_ms=15.0,
            peak_allocated_bytes=0,
            peak_reserved_bytes=0,
            descriptor_cache_bytes=33280,
            measured_cache_bytes=33280,
            d2h_swap_bytes=0,
            h2d_swap_bytes=0,
            scratch_peak_bytes=0,
        )


@dataclass
class BenchmarkConfig:
    model: str
    offload_dir: str
    model_revision: str
    formats: list[str] = field(default_factory=lambda: ["native", "int8_sym"])
    context_lengths: list[int] = field(
        default_factory=lambda: [128, 2048, 8192, 32768]
    )
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 16])
    decode_tokens: int = 128
    warmups: int = 2
    repeats: int = 5
    strict_fallback: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV cache quantization long-context A/B benchmark"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--model-revision", default="local")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["native", "int8_sym"],
        choices=["native", "int8_sym"],
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[128, 2048, 8192, 32768],
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 4, 16]
    )
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--strict-fallback", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def write_json(output_path: str, rows: list[dict[str, object]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def transfer_precision_for(storage_format: str) -> str:
    return _TRANSFER_PRECISION[storage_format]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = BenchmarkConfig(
        model=args.model,
        offload_dir=args.offload_dir,
        model_revision=args.model_revision,
        formats=args.formats,
        context_lengths=args.context_lengths,
        batch_sizes=args.batch_sizes,
        decode_tokens=args.decode_tokens,
        warmups=args.warmups,
        repeats=args.repeats,
        strict_fallback=args.strict_fallback,
    )
    rows = run_matrix(config)
    if args.output_json:
        write_json(args.output_json, rows)
    else:
        print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def run_matrix(config: BenchmarkConfig) -> list[dict[str, object]]:
    from benchmarks.serving._kv_quant_runner import run_single  # type: ignore

    rows: list[dict[str, object]] = []
    for fmt in config.formats:
        for context_length in config.context_lengths:
            for batch_size in config.batch_sizes:
                result = run_single(
                    config=config,
                    storage_format=fmt,
                    context_length=context_length,
                    batch_size=batch_size,
                )
                if (
                    config.strict_fallback
                    and result.requested_format != result.effective_format
                ):
                    raise RuntimeError(
                        "strict fallback: requested "
                        f"{result.requested_format} but effective "
                        f"{result.effective_format} "
                        f"({result.format_decision_reason})"
                    )
                rows.append(result.to_dict())
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
