from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONCURRENCY = (1, 2, 4, 8)
DEFAULT_MAX_NEW_TOKENS = 16


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure continuous-batching request latency."
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--offload-dir",
        required=True,
        help="Directory used for MoE expert offload storage",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=list(DEFAULT_CONCURRENCY),
        help="Concurrency levels to measure",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=5,
        help="Measurement rounds per concurrency level",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=128,
        help="Prompt length in tokens",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Generated tokens per request",
    )
    parser.add_argument(
        "--baseline-json",
        default="baseline_results.json",
        help="Optional baseline_performance.py output JSON for comparison",
    )
    parser.add_argument(
        "--gpu-only-expert-routing",
        choices=("off", "on"),
        default="off",
    )
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument(
        "--routing-baseline-json",
        default=None,
        help="gpu-routing-latency-v1 JSON produced by an off-mode run",
    )
    parser.add_argument(
        "--output-json",
        default="latency_results.json",
        help="Path to write the benchmark summary JSON",
    )
    return parser.parse_args(argv)


def build_model_config(args) -> dict[str, Any]:
    return {
        "offload_path": args.offload_dir,
        "device_memory_ratio": 0.75,
        "gpu_only_expert_routing": args.gpu_only_expert_routing == "on",
        "speculative_prefetch_overlap": False,
    }


def gpu_routing_verdict(baseline, candidate, route_failures, fallback_count):
    reasons = []
    base_p99 = baseline.get("tpot_p99_ms")
    candidate_p99 = candidate.get("tpot_p99_ms")
    if base_p99 and candidate_p99 and candidate_p99 > base_p99 * 1.05:
        reasons.append("tpot_p99_regression_gt_5pct")
    base_p50 = baseline.get("tpot_p50_ms")
    candidate_p50 = candidate.get("tpot_p50_ms")
    if base_p50 and candidate_p50 and candidate_p50 > base_p50 * 1.02:
        reasons.append("tpot_p50_regression_gt_2pct")
    if route_failures:
        reasons.append("native_route_failure")
    if fallback_count:
        reasons.append("unexpected_eager_fallback")
    return {
        "decision": "ROLLBACK" if reasons else "KEEP",
        "reasons": reasons,
    }


def collect_routing_stats(model):
    engine = getattr(model, "engine", None)
    executor = getattr(engine, "expert_executor", None)
    getter = getattr(executor, "get_gpu_routing_stats", None)
    if getter is None:
        return {
            "route_batches": 0,
            "route_failures": 0,
            "fallback_count": 0,
            "completion_events_retired": 0,
        }
    return {key: int(value) for key, value in getter().items()}


def _delta_pct(candidate, baseline):
    if not baseline:
        return None
    return 100.0 * (candidate / baseline - 1.0)


def build_result_payload(*, args, measurements, baseline_measurements, routing):
    comparison = {}
    reasons = []
    if baseline_measurements is not None:
        for level, candidate in measurements.items():
            baseline = baseline_measurements[level]
            comparison[level] = {
                "tpot_p50_delta_pct": _delta_pct(
                    candidate["tpot_p50_ms"], baseline["tpot_p50_ms"]
                ),
                "tpot_p99_delta_pct": _delta_pct(
                    candidate["tpot_p99_ms"], baseline["tpot_p99_ms"]
                ),
            }
            level_verdict = gpu_routing_verdict(
                baseline,
                candidate,
                routing["route_failures"],
                routing["fallback_count"],
            )
            reasons.extend(
                f"concurrency_{level}:{reason}"
                for reason in level_verdict["reasons"]
            )
        decision = "ROLLBACK" if reasons else "KEEP"
    else:
        decision = "BASELINE"
    return {
        "schema_version": "gpu-routing-latency-v1",
        "status": "PASS",
        "config": {
            "model": args.model,
            "offload_dir": args.offload_dir,
            "gpu_only_expert_routing": args.gpu_only_expert_routing,
            "speculative_prefetch_overlap": False,
            "device_memory_ratio": 0.75,
            "concurrency": list(args.concurrency),
            "prompt_length": args.prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "warmup_rounds": args.warmup_rounds,
            "num_rounds": args.num_rounds,
        },
        "measurement": measurements,
        "routing": routing,
        "comparison": comparison,
        "verdict": {"decision": decision, "reasons": reasons},
    }


def load_routing_baseline(path: Path, expected_config):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "gpu-routing-latency-v1":
        raise ValueError(
            "routing baseline must be gpu-routing-latency-v1, got "
            f"{payload.get('schema_version')!r}"
        )
    config = payload.get("config", {})
    for key in (
        "model",
        "prompt_length",
        "max_new_tokens",
        "concurrency",
        "device_memory_ratio",
    ):
        if key in expected_config and config.get(key) != expected_config[key]:
            raise ValueError(
                f"routing baseline config mismatch on {key}: "
                f"{config.get(key)!r} != {expected_config[key]!r}"
            )
    return payload.get("measurement", {})


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
    if target_length <= 0:
        raise ValueError(f"target_length must be > 0, got {target_length}")
    if not token_ids:
        return [0] * target_length

    output: list[int] = []
    while len(output) < target_length:
        output.extend(token_ids)
    return output[:target_length]


def build_prompt_input_ids(
    tokenizer: Any, target_length: int, batch_size: int
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    base_text = (
        "MoE-Infinity continuous batching latency benchmark prompt. "
        "Keep this text deterministic for stable measurements."
    )
    encoded = tokenizer.encode(base_text, add_special_tokens=False)
    prompt_ids = _repeat_to_length(encoded, target_length)
    batch_input = [list(prompt_ids) for _ in range(batch_size)]
    return torch.tensor(batch_input, dtype=torch.long, device="cuda")


def load_model_and_tokenizer(
    model_name: str, offload_dir: str, config: dict | None = None
) -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"transformers import failed: {exc}") from exc

    try:
        from moe_infinity import MoE
    except Exception as exc:
        raise RuntimeError(f"moe_infinity import failed: {exc}") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config is None:
        config = {
            "offload_path": offload_dir,
            "device_memory_ratio": 0.75,
        }
    model = MoE(model_name, config)
    return model, tokenizer


class BatchTimingStreamer:
    def __init__(self, batch_size: int) -> None:
        self._seen_prompt = False
        self.first_token_at: list[float | None] = [None] * batch_size

    def put(self, value: Any) -> None:
        _ = value
        if not self._seen_prompt:
            self._seen_prompt = True
            return

        now = time.perf_counter()
        for index, token_time in enumerate(self.first_token_at):
            if token_time is None:
                self.first_token_at[index] = now

    def end(self) -> None:
        return None


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (p / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    fraction = position - lower
    return lower_value + (upper_value - lower_value) * fraction


def run_one_round(
    model: Any,
    tokenizer: Any,
    *,
    concurrency: int,
    prompt_length: int,
    max_new_tokens: int,
) -> tuple[list[float], list[float]]:
    input_ids = build_prompt_input_ids(tokenizer, prompt_length, concurrency)
    streamer = BatchTimingStreamer(concurrency)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    total_latency_ms = (end - start) * 1000.0
    generated_per_request = max(
        int(output_ids.shape[-1] - input_ids.shape[-1]), 1
    )

    ttft_samples: list[float] = []
    itl_samples: list[float] = []
    for token_time in streamer.first_token_at:
        if token_time is None:
            ttft_ms = total_latency_ms
        else:
            ttft_ms = (token_time - start) * 1000.0
        decode_ms = max(total_latency_ms - ttft_ms, 0.0)

        if generated_per_request <= 1:
            itl_ms = decode_ms
        else:
            itl_ms = decode_ms / (generated_per_request - 1)

        ttft_samples.append(ttft_ms)
        itl_samples.append(itl_ms)

    return ttft_samples, itl_samples


def run_sweep(
    model: Any,
    tokenizer: Any,
    *,
    concurrency_levels: list[int],
    warmup_rounds: int,
    num_rounds: int,
    prompt_length: int,
    max_new_tokens: int,
) -> dict[str, dict[str, float | None]]:
    measurements: dict[str, dict[str, float | None]] = {}
    for concurrency in concurrency_levels:
        for _ in range(warmup_rounds):
            _ = run_one_round(
                model,
                tokenizer,
                concurrency=concurrency,
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens,
            )
        all_ttft: list[float] = []
        all_tpot: list[float] = []

        for _ in range(num_rounds):
            ttft_samples, tpot_samples = run_one_round(
                model,
                tokenizer,
                concurrency=concurrency,
                prompt_length=prompt_length,
                max_new_tokens=max_new_tokens,
            )
            all_ttft.extend(ttft_samples)
            all_tpot.extend(tpot_samples)

        measurements[str(concurrency)] = {
            "sample_count": len(all_tpot),
            "ttft_p50_ms": percentile(all_ttft, 50.0),
            "ttft_p90_ms": percentile(all_ttft, 90.0),
            "ttft_p99_ms": percentile(all_ttft, 99.0),
            "tpot_p50_ms": percentile(all_tpot, 50.0),
            "tpot_p90_ms": percentile(all_tpot, 90.0),
            "tpot_p99_ms": percentile(all_tpot, 99.0),
            "itl_p50_ms": percentile(all_tpot, 50.0),
            "itl_p90_ms": percentile(all_tpot, 90.0),
            "itl_p99_ms": percentile(all_tpot, 99.0),
        }
    return measurements


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def baseline_reference(
    baseline_payload: dict[str, Any] | None,
) -> dict[str, float | None]:
    if baseline_payload is None:
        return {
            "ttft_ms": None,
            "per_token_latency_ms": None,
        }

    measurement = baseline_payload.get("measurement")
    if not isinstance(measurement, dict):
        return {
            "ttft_ms": None,
            "per_token_latency_ms": None,
        }
    return {
        "ttft_ms": _to_float(measurement.get("ttft_ms")),
        "per_token_latency_ms": _to_float(
            measurement.get("per_token_latency_ms")
        ),
    }


def main() -> int:
    args = parse_args()
    if args.num_rounds <= 0:
        raise ValueError("--num-rounds must be > 0")
    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be > 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.warmup_rounds < 0:
        raise ValueError("--warmup-rounds must be >= 0")

    concurrency_levels = [value for value in args.concurrency if value > 0]
    if not concurrency_levels:
        raise ValueError(
            "--concurrency must include at least one positive value"
        )

    model_config = build_model_config(args)
    routing_baseline = None
    if args.gpu_only_expert_routing == "on":
        if not args.routing_baseline_json:
            print(
                "on-mode requires --routing-baseline-json produced by an "
                "off-mode run",
                file=sys.stderr,
            )
            return 2
        try:
            routing_baseline = load_routing_baseline(
                Path(args.routing_baseline_json),
                expected_config={
                    "model": args.model,
                    "prompt_length": args.prompt_length,
                    "max_new_tokens": args.max_new_tokens,
                    "concurrency": list(args.concurrency),
                    "device_memory_ratio": 0.75,
                },
            )
        except (ValueError, OSError) as error:
            print(f"routing baseline rejected: {error}", file=sys.stderr)
            return 2

    env = environment_info()
    output_path = Path(args.output_json)
    baseline = baseline_reference(load_baseline(Path(args.baseline_json)))

    print("=== MoE-Infinity Continuous Batching Latency ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUDA available: {env['cuda_available']}")

    if not env["cuda_available"]:
        print("BLOCKED: No CUDA. Run on GPU hardware.")
        payload = {
            "status": "BLOCKED",
            "reason": "No CUDA device",
            "environment": env,
            "measurement": {
                str(level): {
                    "ttft_p50_ms": None,
                    "ttft_p90_ms": None,
                    "ttft_p99_ms": None,
                    "itl_p50_ms": None,
                    "itl_p90_ms": None,
                    "itl_p99_ms": None,
                }
                for level in concurrency_levels
            },
            "baseline": baseline,
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "num_rounds": args.num_rounds,
        }
        write_json(output_path, payload)
        return 0

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model, args.offload_dir, model_config
        )
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}")
        payload = {
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "environment": env,
            "measurement": {
                str(level): {
                    "ttft_p50_ms": None,
                    "ttft_p90_ms": None,
                    "ttft_p99_ms": None,
                    "itl_p50_ms": None,
                    "itl_p90_ms": None,
                    "itl_p99_ms": None,
                }
                for level in concurrency_levels
            },
            "baseline": baseline,
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "num_rounds": args.num_rounds,
        }
        write_json(output_path, payload)
        return 0

    measurements = run_sweep(
        model,
        tokenizer,
        concurrency_levels=concurrency_levels,
        warmup_rounds=args.warmup_rounds,
        num_rounds=args.num_rounds,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
    )

    routing = collect_routing_stats(model)
    payload = build_result_payload(
        args=args,
        measurements=measurements,
        baseline_measurements=routing_baseline,
        routing=routing,
    )
    payload["environment"] = env
    write_json(output_path, payload)
    print(f"verdict: {payload['verdict']}")
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
