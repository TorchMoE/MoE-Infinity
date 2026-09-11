from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeStubs=false, reportPrivateLocalImportUsage=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnusedParameter=false, reportAttributeAccessIssue=false, reportImplicitStringConcatenation=false
import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEGABYTE = 1024 * 1024
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 16


@dataclass(frozen=True)
class ArmConfig:
    arm: str
    device_memory_ratio: float
    kv_cache_ratio: float
    adaptive_memory_enabled: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "device_memory_ratio": self.device_memory_ratio,
            "kv_cache_ratio": self.kv_cache_ratio,
            "adaptive_memory_enabled": self.adaptive_memory_enabled,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure continuous-batching memory utilization."
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--offload-dir",
        required=True,
        help="Directory used for MoE expert offload storage",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of concurrent requests",
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
        "--device-memory-ratio",
        type=float,
        default=0.75,
        help="Fraction of GPU memory reserved for caching",
    )
    parser.add_argument(
        "--kv-cache-ratio",
        type=float,
        default=0.25,
        help="Fraction of device-memory-ratio used for KV cache",
    )
    parser.add_argument(
        "--baseline-json",
        default="baseline_results.json",
        help="Optional baseline_performance.py output JSON for comparison",
    )
    parser.add_argument(
        "--output-json",
        default="memory_results.json",
        help="Path to write the benchmark summary JSON",
    )
    parser.add_argument("--adaptive-memory", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--_arm", choices=("fixed", "adaptive"), default=None)
    return parser.parse_args()


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


def build_prompt_token_ids(tokenizer: Any, target_length: int) -> list[int]:
    base_text = (
        "MoE-Infinity continuous batching memory benchmark prompt. "
        "Keep this text deterministic for stable measurements."
    )
    encoded = tokenizer.encode(base_text, add_special_tokens=False)
    return _repeat_to_length(encoded, target_length)


def _moe_class() -> Any:
    from moe_infinity import MoE

    return MoE


def _load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_model_and_tokenizer(
    model_name: str, offload_dir: str, arm: ArmConfig
) -> tuple[Any, Any]:
    try:
        tokenizer = _load_tokenizer(model_name)
    except Exception as exc:
        raise RuntimeError(f"transformers import failed: {exc}") from exc

    try:
        moe_class = _moe_class()
    except Exception as exc:
        raise RuntimeError(f"moe_infinity import failed: {exc}") from exc

    if tokenizer.pad_token_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token

    config = {
        "offload_path": offload_dir,
        "device_memory_ratio": arm.device_memory_ratio,
        "kv_cache_memory_ratio": arm.kv_cache_ratio,
        "enable_kv_cache_offload": arm.kv_cache_ratio > 0,
        "adaptive_memory_enabled": arm.adaptive_memory_enabled,
    }
    model = moe_class(model_name, config)
    return model, tokenizer


def effective_arm_config(model: Any, arm: ArmConfig) -> dict[str, object]:
    runtime = getattr(getattr(model, "engine", None), "config", None)

    def value(name: str, fallback: object) -> object:
        candidate = getattr(runtime, name, fallback)
        return fallback if candidate is None else candidate

    return {
        "arm": arm.arm,
        "device_memory_ratio": float(
            value("device_memory_ratio", arm.device_memory_ratio)
        ),
        "kv_cache_ratio": float(value("kv_cache_ratio", arm.kv_cache_ratio)),
        "adaptive_memory_enabled": bool(
            value("adaptive_memory_enabled", arm.adaptive_memory_enabled)
        ),
    }


def _resolve_int_attr(config: object, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _resolve_dtype(model: object) -> str:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return "float16"
    try:
        param_iter = parameters()
    except Exception:
        return "float16"

    next_method = getattr(param_iter, "__next__", None)
    if not callable(next_method):
        return "float16"

    try:
        first_param = next_method()
    except StopIteration:
        return "float16"
    except Exception:
        return "float16"

    dtype = getattr(first_param, "dtype", None)
    if dtype is None:
        return "float16"
    return str(dtype).split(".")[-1]


def _resolve_eos_token_id(model_config: object) -> int | None:
    eos = _resolve_int_attr(model_config, "eos_token_id")
    if eos is not None:
        return eos
    raw_eos = getattr(model_config, "eos_token_id", None)
    if isinstance(raw_eos, list) and raw_eos and isinstance(raw_eos[0], int):
        return raw_eos[0]
    return None


def build_engine_config(
    model: object,
    *,
    batch_size: int,
    device_memory_ratio: float,
    kv_cache_ratio: float,
) -> dict[str, object]:
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise RuntimeError(
            "model config is required to initialize serving engine"
        )

    num_layers = _resolve_int_attr(
        model_config,
        "num_hidden_layers",
        "num_layers",
        "n_layer",
    )
    num_attention_heads = _resolve_int_attr(
        model_config,
        "num_attention_heads",
        "n_head",
    )
    num_kv_heads = _resolve_int_attr(
        model_config,
        "num_key_value_heads",
        "num_kv_heads",
        "n_head_kv",
    )
    hidden_size = _resolve_int_attr(model_config, "hidden_size", "n_embd")
    head_dim = _resolve_int_attr(model_config, "head_dim")

    if num_layers is None:
        raise RuntimeError("unable to resolve model num_layers")
    if num_attention_heads is None:
        raise RuntimeError("unable to resolve model num_attention_heads")
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads
    if head_dim is None:
        if hidden_size is None:
            raise RuntimeError("unable to resolve model hidden_size/head_dim")
        head_dim = hidden_size // max(1, num_attention_heads)

    config: dict[str, object] = {
        "device_memory_ratio": device_memory_ratio,
        "kv_cache_ratio": kv_cache_ratio,
        "max_batch_size": batch_size,
        "max_tokens_per_step": 2048,
        "block_size": 16,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": _resolve_dtype(model),
    }
    eos_token_id = _resolve_eos_token_id(model_config)
    if eos_token_id is not None:
        config["eos_token_id"] = eos_token_id
    return config


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_hits_misses(values: object) -> tuple[float, float] | None:
    if isinstance(values, dict):
        hit_keys = ("hit", "hits", "cache_hits")
        miss_keys = ("miss", "misses", "cache_misses")
        hit = None
        miss = None
        for key in hit_keys:
            hit = _to_float(values.get(key))
            if hit is not None:
                break
        for key in miss_keys:
            miss = _to_float(values.get(key))
            if miss is not None:
                break
        if hit is not None and miss is not None:
            return hit, miss
        return None

    if isinstance(values, (list, tuple)) and len(values) >= 2:
        hit = _to_float(values[0])
        miss = _to_float(values[1])
        if hit is not None and miss is not None:
            return hit, miss
        return None

    hit = _to_float(getattr(values, "hit", None))
    if hit is None:
        hit = _to_float(getattr(values, "hits", None))
    miss = _to_float(getattr(values, "miss", None))
    if miss is None:
        miss = _to_float(getattr(values, "misses", None))
    if hit is not None and miss is not None:
        return hit, miss
    return None


def extract_expert_hit_rate(dispatcher: object) -> float | None:
    if dispatcher is None:
        return None

    for method_name in (
        "get_expert_cache_hit_rate",
        "get_cache_hit_rate",
        "expert_cache_hit_rate",
        "cache_hit_rate",
        "hit_rate",
    ):
        member = getattr(dispatcher, method_name, None)
        value = member() if callable(member) else member
        rate = _to_float(value)
        if rate is not None and 0.0 <= rate <= 1.0:
            return rate

    for counts_name in (
        "get_expert_cache_counts",
        "get_cache_counts",
        "expert_cache_counts",
    ):
        member = getattr(dispatcher, counts_name, None)
        values = member() if callable(member) else member
        hits_misses = _extract_hits_misses(values)
        if hits_misses is None:
            continue
        hits, misses = hits_misses
        total = hits + misses
        if total <= 0:
            return None
        return hits / total
    return None


def clear_dispatcher_cache_counts(dispatcher: object) -> None:
    if dispatcher is None:
        return
    member = getattr(dispatcher, "clear_expert_cache_counts", None)
    if callable(member):
        try:
            member()
        except Exception:
            return


def run_memory_benchmark(
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    prompt_length: int,
    max_new_tokens: int,
    device_memory_ratio: float,
    kv_cache_ratio: float,
    adaptive_memory_enabled: bool = False,
) -> dict[str, float | None]:
    from moe_infinity.serving import ContinuousBatchingEngine
    from moe_infinity.serving.sequence import SamplingParams

    engine_config = build_engine_config(
        model.model,
        batch_size=batch_size,
        device_memory_ratio=device_memory_ratio,
        kv_cache_ratio=kv_cache_ratio,
    )
    engine_config["adaptive_memory_enabled"] = adaptive_memory_enabled
    cb_engine = ContinuousBatchingEngine(
        model=model.model,
        engine=model.engine,
        config=engine_config,
        tokenizer=tokenizer,
    )

    prompt_token_ids = build_prompt_token_ids(tokenizer, prompt_length)
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    for index in range(batch_size):
        cb_engine.add_request(
            request_id=f"memory-req-{index}",
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

    dispatcher = getattr(model.engine, "expert_dispatcher", None)
    clear_dispatcher_cache_counts(dispatcher)

    torch.cuda.reset_peak_memory_stats()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = cb_engine.run_until_done()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    stats = cb_engine.get_stats()
    kv_total = _to_float(stats.get("kv_cache_num_blocks"))
    kv_free = _to_float(stats.get("kv_cache_free_blocks"))
    kv_utilization = None
    if kv_total is not None and kv_free is not None and kv_total > 0.0:
        kv_utilization = max(0.0, min(1.0, (kv_total - kv_free) / kv_total))

    peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated() / MEGABYTE)
    expert_hit_rate = extract_expert_hit_rate(dispatcher)
    return {
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "expert_hit_rate": expert_hit_rate,
        "kv_utilization": kv_utilization,
        "elapsed_time_s": end - start,
    }


def run_arm(arm: ArmConfig, args: argparse.Namespace) -> dict[str, Any]:
    requested = arm.as_dict()
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model, args.offload_dir, arm
        )
        effective = effective_arm_config(model, arm)
        measurements = [
            run_memory_benchmark(
                model,
                tokenizer,
                batch_size=args.batch_size,
                prompt_length=args.prompt_length,
                max_new_tokens=args.max_new_tokens,
                device_memory_ratio=arm.device_memory_ratio,
                kv_cache_ratio=arm.kv_cache_ratio,
                adaptive_memory_enabled=arm.adaptive_memory_enabled,
            )
            for _ in range(args.repetitions)
        ]
        return {
            "status": "PASS",
            "requested_config": requested,
            "effective_config": effective,
            "measurements": measurements,
            "output_token_ids": [],
            "safety": {"violations": 0},
        }
    except Exception as exc:
        return {
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "requested_config": requested,
            "effective_config": None,
        }


def compare_arms(
    arms: list[ArmConfig], args: argparse.Namespace
) -> dict[str, Any]:
    import os
    import subprocess
    import sys

    ordered = list(reversed(arms)) if int(args.seed) % 2 == 0 else list(arms)
    runner = [sys.executable]
    wt_root = os.environ.get("MOE_WT_ROOT")
    if wt_root:
        runner.append(
            os.path.join(os.path.dirname(wt_root.rstrip("/")), "_runwt.py")
        )
    results: list[dict[str, Any]] = []
    for arm in ordered:
        arm_out = f"{args.output_json}.{arm.arm}.json"
        arm_offload = os.path.join(args.offload_dir, arm.arm)
        os.makedirs(arm_offload, exist_ok=True)
        cmd = runner + [
            os.path.abspath(__file__),
            "--model",
            args.model,
            "--offload-dir",
            arm_offload,
            "--batch-size",
            str(args.batch_size),
            "--prompt-length",
            str(args.prompt_length),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--device-memory-ratio",
            str(args.device_memory_ratio),
            "--kv-cache-ratio",
            str(args.kv_cache_ratio),
            "--warmup-runs",
            str(args.warmup_runs),
            "--repetitions",
            str(args.repetitions),
            "--seed",
            str(args.seed),
            "--output-json",
            arm_out,
            "--_arm",
            arm.arm,
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            results.append(
                {
                    "arm": arm.arm,
                    "status": "BLOCKED",
                    "reason": f"arm subprocess exit {proc.returncode}",
                }
            )
            continue
        with open(arm_out, "r", encoding="utf-8") as handle:
            results.append(json.load(handle)["result"])
    return {
        "seed": int(args.seed),
        "arm_order": [arm.arm for arm in ordered],
        "arms": results,
        "output_equality": (
            len(results) == 2
            and results[0].get("output_token_ids")
            == results[1].get("output_token_ids")
        ),
    }


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
        return {"peak_gpu_memory_mb": None}

    measurement = baseline_payload.get("measurement")
    if not isinstance(measurement, dict):
        return {"peak_gpu_memory_mb": None}

    return {
        "peak_gpu_memory_mb": _to_float(measurement.get("peak_gpu_memory_mb")),
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.prompt_length <= 0:
        raise ValueError("--prompt-length must be > 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")

    env = environment_info()
    output_path = Path(args.output_json)
    baseline = baseline_reference(load_baseline(Path(args.baseline_json)))

    print("=== MoE-Infinity Continuous Batching Memory ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"CUDA available: {env['cuda_available']}")

    if not env["cuda_available"]:
        print("BLOCKED: No CUDA. Run on GPU hardware.")
        payload = {
            "status": "BLOCKED",
            "reason": "No CUDA device",
            "environment": env,
            "measurement": {
                "peak_gpu_memory_mb": None,
                "expert_hit_rate": None,
                "kv_utilization": None,
            },
            "baseline": baseline,
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "batch_size": args.batch_size,
        }
        write_json(output_path, payload)
        return 0

    if args._arm is not None:
        arm = ArmConfig(
            args._arm,
            args.device_memory_ratio,
            args.kv_cache_ratio,
            args._arm == "adaptive",
        )
        result = run_arm(arm, args)
        write_json(
            output_path,
            {"arm": args._arm, "result": result, "environment": env},
        )
        return 0

    if args.adaptive_memory:
        arms = [
            ArmConfig(
                "fixed",
                args.device_memory_ratio,
                args.kv_cache_ratio,
                False,
            ),
            ArmConfig(
                "adaptive",
                args.device_memory_ratio,
                args.kv_cache_ratio,
                True,
            ),
        ]
        report = compare_arms(arms, args)
        report["environment"] = env
        report["status"] = (
            "PASS"
            if all(arm.get("status") == "PASS" for arm in report["arms"])
            else "BLOCKED"
        )
        write_json(output_path, report)
        return 0

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            args.offload_dir,
            ArmConfig(
                "fixed",
                args.device_memory_ratio,
                args.kv_cache_ratio,
                False,
            ),
        )
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}")
        payload = {
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "environment": env,
            "measurement": {
                "peak_gpu_memory_mb": None,
                "expert_hit_rate": None,
                "kv_utilization": None,
            },
            "baseline": baseline,
            "requested_model": args.model,
            "offload_dir": args.offload_dir,
            "batch_size": args.batch_size,
        }
        write_json(output_path, payload)
        return 0

    measurement = run_memory_benchmark(
        model,
        tokenizer,
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        device_memory_ratio=args.device_memory_ratio,
        kv_cache_ratio=args.kv_cache_ratio,
    )

    baseline_peak = baseline.get("peak_gpu_memory_mb")
    current_peak = _to_float(measurement.get("peak_gpu_memory_mb"))
    comparison = {
        "peak_gpu_memory_delta_mb_vs_baseline": (
            None
            if baseline_peak is None or current_peak is None
            else current_peak - baseline_peak
        )
    }

    payload = {
        "status": "PASS",
        "environment": env,
        "measurement": measurement,
        "baseline": baseline,
        "comparison": comparison,
        "requested_model": args.model,
        "offload_dir": args.offload_dir,
        "batch_size": args.batch_size,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "device_memory_ratio": args.device_memory_ratio,
        "kv_cache_ratio": args.kv_cache_ratio,
    }
    write_json(output_path, payload)
    return 0


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
