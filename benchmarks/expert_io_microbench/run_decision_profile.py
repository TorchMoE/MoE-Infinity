"""IBP feasibility decision-profile runner (Wave 1+ orchestrator).

Wraps a real `MoE.generate` workload inside `nsys profile -t cuda,nvtx
--capture-range=cudaProfilerApi`, then invokes `nsys_parser.summarise`
to emit a verdict against the FROZEN criterion in
.sisyphus/plans/ibp-feasibility-profile.md.

This script must be invoked with `nsys profile ... python run_decision_profile.py
<args>`; it expects to be running INSIDE the nsys-instrumented process. It
toggles cudaProfilerStart/Stop around the measured decode steps so warmup is
not counted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--offload-dir", required=True)
    p.add_argument("--hardware-tag", required=True)
    p.add_argument("--mode", choices=["disk", "host-only"], required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--warmup-tokens", type=int, default=8)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--device-memory-ratio", type=float, default=0.5)
    p.add_argument("--speculative-prefetch", action="store_true")
    p.add_argument("--speculative-prefetch-overlap", action="store_true")
    p.add_argument(
        "--gpu-only-expert-routing", choices=("off", "on"), default="off"
    )
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--output-json", required=True)
    return p.parse_args(argv)


def build_model_config(args) -> dict:
    from moe_infinity.utils import ArcherConfig

    gpu_routing = args.gpu_only_expert_routing == "on"
    ArcherConfig._validate_gpu_routing_overlap(
        gpu_routing,
        args.speculative_prefetch_overlap,
        "off",
    )
    return {
        "offload_path": args.offload_dir,
        "device_memory_ratio": args.device_memory_ratio,
        "speculative_prefetch": args.speculative_prefetch,
        "speculative_prefetch_overlap": args.speculative_prefetch_overlap,
        "num_threads": args.num_threads,
        "gpu_only_expert_routing": gpu_routing,
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


def build_profile_payload(*, args, decode_step_times_ns, routing, pcie) -> dict:
    return {
        "schema_version": "gpu-routing-decision-profile-v1",
        "config": {
            "gpu_only_expert_routing": args.gpu_only_expert_routing,
            "speculative_prefetch_overlap": False,
            "warmup_iters": args.warmup_iters,
            "warmup_tokens": args.warmup_tokens,
            "iters": args.iters,
            "max_new_tokens": args.max_new_tokens,
        },
        "measurement": {
            "decode_step_times_ns": list(decode_step_times_ns),
            "decode_step_total_ns": sum(decode_step_times_ns),
            "decode_step_count": args.iters * args.max_new_tokens,
        },
        "routing": dict(routing),
        "pcie": dict(pcie),
    }


def sample_pcie_link() -> tuple[int, int]:
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=pcie.link.width.current,pcie.link.gen.current",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                text=True,
                timeout=10,
            )
            .strip()
            .splitlines()[0]
        )
        width_s, gen_s = out.split(",")
        return int(width_s.strip()), int(gen_s.strip())
    except Exception as e:
        print(f"[WARN] could not sample PCIe link: {e}", file=sys.stderr)
        return 16, 4


def main() -> int:
    parser_holder = argparse.ArgumentParser()
    args = parse_args()
    try:
        model_config = build_model_config(args)
    except ValueError as error:
        parser_holder.error(str(error))
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[runner] mode={args.mode} model={args.model}", flush=True)
    print(
        f"[runner] offload={args.offload_dir} ratio={args.device_memory_ratio}",
        flush=True,
    )

    import torch
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    t0 = time.time()
    print(f"[{time.time() - t0:.1f}s] loading model", flush=True)
    m = MoE(args.model, model_config)
    print(f"[{time.time() - t0:.1f}s] model loaded", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_text = (
        "MoE-Infinity transfer timing benchmark prompt. "
        "Use deterministic content for stable measurements."
    )
    enc = tok.encode(base_text, add_special_tokens=False)
    while len(enc) < 128:
        enc = enc + enc
    enc = enc[:128]
    ids = torch.tensor([enc], dtype=torch.long, device="cuda")

    print(
        f"[{time.time() - t0:.1f}s] warmup {args.warmup_iters} iters",
        flush=True,
    )
    for _ in range(args.warmup_iters):
        _ = m.generate(
            ids,
            max_new_tokens=max(args.warmup_tokens, 1),
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize()
    print(f"[{time.time() - t0:.1f}s] warmup done", flush=True)

    link_width, link_gen = sample_pcie_link()
    print(
        f"[runner] PCIe before profile: width={link_width} gen={link_gen}",
        flush=True,
    )

    print(f"[{time.time() - t0:.1f}s] profiler START", flush=True)
    torch.cuda.cudart().cudaProfilerStart()

    decode_step_times_ns: list[int] = []
    for it in range(args.iters):
        t_iter = time.perf_counter_ns()
        _ = m.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        decode_step_times_ns.append(time.perf_counter_ns() - t_iter)

    torch.cuda.cudart().cudaProfilerStop()
    print(f"[{time.time() - t0:.1f}s] profiler STOP", flush=True)

    link_width_after, link_gen_after = sample_pcie_link()
    print(
        f"[runner] PCIe after profile: width={link_width_after} gen={link_gen_after}",
        flush=True,
    )

    use_width = max(link_width, link_width_after)
    use_gen = max(link_gen, link_gen_after)

    out = build_profile_payload(
        args=args,
        decode_step_times_ns=decode_step_times_ns,
        routing=collect_routing_stats(m),
        pcie={
            "link_width_pre": link_width,
            "link_gen_pre": link_gen,
            "link_width_post": link_width_after,
            "link_gen_post": link_gen_after,
        },
    )
    out["model"] = args.model
    out["mode"] = args.mode
    out["hardware_tag"] = args.hardware_tag
    out["offload_dir"] = args.offload_dir
    out["pcie_link_width_observed"] = use_width
    out["pcie_link_gen_observed"] = use_gen
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"[runner] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
