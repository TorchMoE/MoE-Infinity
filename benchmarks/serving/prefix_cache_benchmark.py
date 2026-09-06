from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from moe_infinity.entrypoints.big_modeling import MoE
from moe_infinity.entrypoints.openai.api_server_v2 import _build_engine_config
from moe_infinity.runtime.attention_backend import FlashInferPlanMetadata
from moe_infinity.runtime.attention_types import PagedBatchLengths
from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.sequence import SamplingParams


@dataclass(frozen=True)
class Geometry:
    query_lengths: list[int]
    query_offsets: list[int]
    context_lengths: list[int]
    kv_seq_lengths: list[int]


@dataclass(frozen=True)
class ModeResult:
    engine_instance_id: str
    prefix_cache_active: bool
    geometry: Geometry
    hits_total: int
    matched_tokens_total: int
    open_leases: int
    refcount_high_water: int
    token_digest: str
    logit_digest: str
    ttft_ms: list[float]
    e2e_ms: list[float]


@dataclass(frozen=True)
class RunSample:
    token_ids: torch.Tensor
    last_token_logits: torch.Tensor
    plan: FlashInferPlanMetadata
    ttft_ms: list[float]
    e2e_ms: list[float]


class BenchmarkMismatch(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PrefixCapabilityError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DryBenchmarkEngine:
    def __init__(self, enabled: bool) -> None:
        self.enabled, self.primed = enabled, False
        self.instance_id = str(uuid.uuid4())
        self.refcount_high_water = 1
        self._hits = 0

    def run(self, prompt: list[int], measured: bool) -> RunSample:
        warm = measured and self.enabled and self.primed
        if warm:
            self._hits += 1
            self.refcount_high_water = 2
        query = [1] if warm else [len(prompt)]
        plan = FlashInferPlanMetadata(
            lengths=PagedBatchLengths(
                query_lengths=torch.tensor(query),
                query_offsets=torch.tensor([0, query[0]]),
                context_lengths=torch.tensor([len(prompt) - query[0]]),
                kv_seq_lengths=torch.tensor([len(prompt)]),
            ),
            kv_indptr=torch.tensor([0, 1]),
            kv_last_page_len=torch.tensor([len(prompt) % 16 or 16]),
        )
        self.primed = self.primed or not measured
        return RunSample(
            token_ids=torch.tensor([7, 8]),
            last_token_logits=torch.tensor([[0.25, 0.75]]),
            plan=plan,
            ttft_ms=[1.0],
            e2e_ms=[2.0],
        )

    def stats(self) -> dict[str, object]:
        return {
            "prefix_cache_active": self.enabled,
            "prefix_cache_hits_total": self._hits,
            "prefix_cache_matched_tokens_total": 64 if self._hits else 0,
            "prefix_cache_open_leases": 0,
        }


class RealBenchmarkEngine:
    def __init__(self, engine: ContinuousBatchingEngine) -> None:
        self.engine, self.instance_id = engine, str(uuid.uuid4())
        self.refcount_high_water = 0

    def run(self, prompt: list[int], measured: bool) -> RunSample:
        request_id, plans, logits, outputs = str(uuid.uuid4()), [], [], []
        started = time.perf_counter_ns()
        first_token_ns: int | None = None
        original = self.engine._execute_batch

        def capture(batch):
            result = original(batch)
            plans.append(
                self.engine.model_runner._get_attention_backend().last_flashinfer_plan
            )
            logits.append(
                self.engine._extract_last_token_logits(result, batch)
                .detach()
                .float()
                .cpu()
            )
            return result

        def on_token(output):
            nonlocal first_token_ns
            first_token_ns = first_token_ns or time.perf_counter_ns()
            outputs.append(output.token_id)

        self.engine._execute_batch = capture
        try:
            self.engine.add_request(
                request_id,
                prompt,
                SamplingParams(temperature=0.0, max_tokens=2),
                on_token=on_token,
            )
            while self.engine.has_pending_requests():
                self.engine.step()
                allocator = self.engine.kv_cache.block_allocator
                self.refcount_high_water = max(
                    self.refcount_high_water,
                    max(
                        (
                            allocator.ref_count(i)
                            for i in range(allocator.num_blocks)
                        ),
                        default=0,
                    ),
                )
        finally:
            self.engine._execute_batch = original
        ended = time.perf_counter_ns()
        assert first_token_ns is not None and plans and logits
        return RunSample(
            token_ids=torch.tensor(outputs),
            last_token_logits=torch.cat(logits),
            plan=plans[0],
            ttft_ms=[(first_token_ns - started) / 1e6],
            e2e_ms=[(ended - started) / 1e6],
        )

    def stats(self) -> dict[str, object]:
        return self.engine.get_stats()


def digest_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def run_mode(
    name: str,
    factory: Callable[[bool], DryBenchmarkEngine | RealBenchmarkEngine],
    prime_prompt: list[int],
    measured_prompt: list[int],
    force_mismatch: bool,
    dump_path: str | None = None,
) -> ModeResult:
    enabled = name != "disabled"
    engine = factory(enabled)
    if name == "enabled_warm":
        engine.run(prime_prompt, measured=False)
    sample = engine.run(measured_prompt, measured=True)
    if dump_path is not None:
        torch.save(
            {
                "token_ids": sample.token_ids,
                "last_token_logits": sample.last_token_logits,
            },
            dump_path,
        )
    token_digest = digest_tensor(sample.token_ids)
    if force_mismatch and name == "enabled_warm":
        token_digest = "forced-mismatch"
    stats = engine.stats()
    return ModeResult(
        engine_instance_id=engine.instance_id,
        prefix_cache_active=bool(stats["prefix_cache_active"]),
        geometry=Geometry(
            query_lengths=sample.plan.lengths.query_lengths.tolist(),
            query_offsets=sample.plan.lengths.query_offsets.tolist(),
            context_lengths=sample.plan.lengths.context_lengths.tolist(),
            kv_seq_lengths=sample.plan.lengths.kv_seq_lengths.tolist(),
        ),
        hits_total=int(stats.get("prefix_cache_hits_total", 0)),
        matched_tokens_total=int(
            stats.get("prefix_cache_matched_tokens_total", 0)
        ),
        open_leases=int(stats["prefix_cache_open_leases"]),
        refcount_high_water=engine.refcount_high_water,
        token_digest=token_digest,
        logit_digest=digest_tensor(sample.last_token_logits),
        ttft_ms=list(sample.ttft_ms),
        e2e_ms=list(sample.e2e_ms),
    )


def run_suite(
    factory, prime_prompt, measured_prompt, force_mismatch=False
) -> dict[str, object]:
    modes = {
        name: run_mode(
            name, factory, prime_prompt, measured_prompt, force_mismatch
        )
        for name in ("disabled", "enabled_cold", "enabled_warm")
    }
    if len({result.engine_instance_id for result in modes.values()}) != 3:
        raise RuntimeError("benchmark modes must use fresh engine instances")
    token_equal = len({result.token_digest for result in modes.values()}) == 1
    logit_equal = len({result.logit_digest for result in modes.values()}) == 1
    if not token_equal or not logit_equal:
        raise BenchmarkMismatch("disabled/cold/warm digest mismatch")
    if any(result.open_leases != 0 for result in modes.values()):
        raise RuntimeError("benchmark completed with open prefix leases")
    return {
        "modes": {
            name: dataclasses.asdict(result) for name, result in modes.items()
        },
        "correctness": {
            "token_digests_equal": token_equal,
            "logit_digests_equal": logit_equal,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--offload-dir", default="/tmp/moe-prefix-benchmark")
    parser.add_argument("--device-memory-ratio", type=float, default=0.5)
    parser.add_argument("--kv-cache-ratio", type=float, default=0.25)
    parser.add_argument("--shared-prefix-tokens", type=int, default=64)
    parser.add_argument("--suffix-tokens", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--parity-report",
        action="store_true",
        help=(
            "dump per-mode token/logit tensors and emit a numerical parity "
            "analysis instead of failing on digest mismatch"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-force-mismatch", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_mode",
        choices=("disabled", "enabled_cold", "enabled_warm"),
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def make_dry_factory():
    return lambda enabled: DryBenchmarkEngine(enabled)


def make_real_factory(args: argparse.Namespace, tokenizer):
    def factory(enabled: bool) -> RealBenchmarkEngine:
        owner = MoE(
            args.model,
            {
                "offload_path": str(Path(args.offload_dir) / str(uuid.uuid4())),
                "device_memory_ratio": args.device_memory_ratio,
            },
        )
        config = _build_engine_config(
            SimpleNamespace(
                device_memory_ratio=args.device_memory_ratio,
                kv_cache_ratio=args.kv_cache_ratio,
                max_batch_size=1,
                enable_prefix_caching=enabled,
                prefix_cache_max_entries=1000,
            ),
            owner.model,
        )
        engine = ContinuousBatchingEngine(
            owner.model, owner.engine, config, tokenizer=tokenizer
        )
        if enabled and not engine.get_stats()["prefix_cache_active"]:
            raise PrefixCapabilityError(
                str(engine.get_stats()["prefix_cache_disabled_reason"])
            )
        return RealBenchmarkEngine(engine)

    return factory


def exact_tokens(tokenizer, text: str, count: int) -> list[int]:
    if count <= 0:
        return []
    seed = tokenizer.encode(text, add_special_tokens=False)
    if not seed:
        raise ValueError("tokenizer returned an empty benchmark seed")
    return (seed * ((count + len(seed) - 1) // len(seed)))[:count]


def build_prompt_pair(
    args: argparse.Namespace, tokenizer=None
) -> tuple[list[int], list[int]]:
    if tokenizer is None:
        shared = [
            100 + (index % 100) for index in range(args.shared_prefix_tokens)
        ]
        prime_suffix = [200] * args.suffix_tokens
        measured_suffix = [201] * args.suffix_tokens
    else:
        shared = exact_tokens(
            tokenizer, "Exact shared prefix. ", args.shared_prefix_tokens
        )
        prime_suffix = exact_tokens(
            tokenizer, "Prime suffix. ", args.suffix_tokens
        )
        measured_suffix = exact_tokens(
            tokenizer, "Measured suffix. ", args.suffix_tokens
        )
    return shared + prime_suffix, shared + measured_suffix


def analyze_parity(dump_paths: dict[str, str]) -> dict[str, object]:
    tensors = {name: torch.load(path) for name, path in dump_paths.items()}
    report: dict[str, object] = {
        "tokens": {
            name: data["token_ids"].tolist() for name, data in tensors.items()
        }
    }
    disabled = tensors["disabled"]["last_token_logits"].float()
    cold = tensors["enabled_cold"]["last_token_logits"].float()
    warm = tensors["enabled_warm"]["last_token_logits"].float()
    steps = min(x.shape[0] for x in (disabled, cold, warm))
    report["logits_max_abs"] = {
        "disabled_vs_cold": (disabled[:steps] - cold[:steps])
        .abs()
        .max()
        .item(),
        "cold_vs_warm": (cold[:steps] - warm[:steps]).abs().max().item(),
    }
    cold_tokens = tensors["enabled_cold"]["token_ids"].tolist()
    warm_tokens = tensors["enabled_warm"]["token_ids"].tolist()
    flips = [
        index
        for index, pair in enumerate(zip(cold_tokens, warm_tokens))
        if pair[0] != pair[1]
    ]
    report["first_flip_step"] = flips[0] if flips else None
    if flips and flips[0] < steps:
        step = flips[0]
        for label, logits in (("cold", cold), ("warm", warm)):
            top2 = torch.topk(logits[step], 2)
            report[f"{label}_flip_top2"] = {
                "token_ids": top2.indices.tolist(),
                "logits": top2.values.tolist(),
                "margin": (top2.values[0] - top2.values[1]).item(),
            }
    return report


def run_suite_subprocess(args: argparse.Namespace) -> dict[str, object]:
    import os
    import subprocess

    runner = [sys.executable]
    wt_root = os.environ.get("MOE_WT_ROOT")
    if wt_root:
        runner.append(
            os.path.join(os.path.dirname(wt_root.rstrip("/")), "_runwt.py")
        )
    modes: dict[str, dict[str, object]] = {}
    for name in ("disabled", "enabled_cold", "enabled_warm"):
        mode_out = f"{args.output_json}.{name}.json"
        cmd = runner + [
            os.path.abspath(__file__),
            "--model",
            args.model,
            "--offload-dir",
            os.path.join(args.offload_dir, name),
            "--shared-prefix-tokens",
            str(args.shared_prefix_tokens),
            "--suffix-tokens",
            str(args.suffix_tokens),
            "--output-json",
            mode_out,
            "--device-memory-ratio",
            str(args.device_memory_ratio),
            "--kv-cache-ratio",
            str(args.kv_cache_ratio),
            "--_mode",
            name,
        ]
        if args.parity_report:
            cmd.append("--parity-report")
        proc = subprocess.run(cmd)
        if proc.returncode == 2:
            raise BenchmarkMismatch(
                f"mode {name} reported a capability or mismatch error"
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"mode {name} subprocess exited with {proc.returncode}"
            )
        with open(mode_out, "r", encoding="utf-8") as handle:
            modes[name] = json.load(handle)
    if len({str(m["engine_instance_id"]) for m in modes.values()}) != 3:
        raise RuntimeError("expected three distinct engine instances")
    token_equal = len({str(m["token_digest"]) for m in modes.values()}) == 1
    logit_equal = len({str(m["logit_digest"]) for m in modes.values()}) == 1
    payload: dict[str, object] = {
        "modes": modes,
        "correctness": {
            "token_digests_equal": token_equal,
            "logit_digests_equal": logit_equal,
        },
    }
    if args.parity_report:
        payload["parity"] = analyze_parity(
            {
                name: f"{args.output_json}.{name}.json.tensors.pt"
                for name in ("disabled", "enabled_cold", "enabled_warm")
            }
        )
    elif not (token_equal and logit_equal):
        raise BenchmarkMismatch("disabled/cold/warm digest mismatch")
    if any(int(m["open_leases"]) for m in modes.values()):
        raise RuntimeError("benchmark completed with open prefix leases")
    return payload


def main() -> int:
    args = parse_args()

    if args._mode is not None:
        tokenizer = (
            None
            if args.dry_run
            else AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True
            )
        )
        factory = (
            make_dry_factory()
            if args.dry_run
            else make_real_factory(args, tokenizer)
        )
        prime_prompt, measured_prompt = build_prompt_pair(args, tokenizer)
        try:
            result = run_mode(
                args._mode,
                factory,
                prime_prompt,
                measured_prompt,
                args.dry_run_force_mismatch,
                dump_path=(
                    f"{args.output_json}.tensors.pt"
                    if args.parity_report
                    else None
                ),
            )
        except (BenchmarkMismatch, PrefixCapabilityError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(dataclasses.asdict(result), indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    if args.dry_run:
        factory = make_dry_factory()
        prime_prompt, measured_prompt = build_prompt_pair(args, None)
        try:
            payload = run_suite(
                factory,
                prime_prompt,
                measured_prompt,
                force_mismatch=args.dry_run_force_mismatch,
            )
        except (BenchmarkMismatch, PrefixCapabilityError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        try:
            payload = run_suite_subprocess(args)
        except (BenchmarkMismatch, PrefixCapabilityError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
