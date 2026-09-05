# Expert I/O Microbench Integration Runner

`run_all.py` is the main entry point. It runs the routing, transfer, compute, and bubble scenarios, then merges their JSON into one report. Use the per-scenario scripts when you want to inspect a single stage in isolation.

## Prerequisites

- CUDA GPU and a working PyTorch build.
- `transformers`, `moe_infinity`, and a local HuggingFace cache for the model you benchmark.
- `psutil` if you want CPU RSS in the memory-style outputs.
- `nsys` for `compare_baseline.py`, `run_decision_profile.py`, and `nsys_parser.py`.
- `nvidia-smi` for PCIe link width and generation sampling in the decision profile workflow.
- Enough `/dev/shm` for `--host-only`, or Docker `--shm-size=32g`.

## Quick start

Build the benchmark image first, using the documented Docker instructions in [docs/benchmark_reproduction.md](../../docs/benchmark_reproduction.md#per-framework-setup):

```bash
docker build -t moe-infinity-bench -f docker/Dockerfile .
```

```bash
docker run --gpus all --shm-size=32g --ipc=host moe-infinity-bench \
  python benchmarks/expert_io_microbench/run_all.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /path/to/offload \
  --output-json benchmarks/expert_io_microbench/results/all.json
```

Host-only mode copies the offload tree to tmpfs first and removes disk I/O from the run:

```bash
python benchmarks/expert_io_microbench/run_all.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /path/to/offload \
  --host-only \
  --output-json host_only_results.json
```

## Supported commands

| Command | Use when | Key outputs | Notes |
| --- | --- | --- | --- |
| `python benchmarks/expert_io_microbench/run_all.py --model ... --offload-dir ... [--scenario all\|routing\|transfer\|compute\|bubble] [--host-only] [--theoretical-pcie-gbps ...] [--output-json ...]` | You want the merged report for all scenarios or one scenario. | `status`, `mode`, `scenarios`, `bandwidth_analysis`, `executive_summary`, `runner_stdout`, `runner_stderr` | Keeps going if one scenario fails and records the error under that key. |
| `python benchmarks/expert_io_microbench/bench_routing.py --model ... --offload-dir ... [--host-only] [--output-json ...]` | You want routing and cache lookup timing only. | `routing`, `cache_lookup`, `cache`, `io_profiler_events` | Sets `MOE_INFINITY_PROFILE_IO=1` and `MOE_INFINITY_PROFILE_IO_SAMPLE=1.0` internally. |
| `python benchmarks/expert_io_microbench/bench_transfer.py --model ... --offload-dir ... [--host-only] [--output-json ...]` | You want disk to CPU and CPU to GPU timing only. | `disk_to_cpu`, `cpu_to_gpu`, `sync_overhead` | Also records the number of sync events per step. |
| `python benchmarks/expert_io_microbench/bench_compute_evict.py --model ... --offload-dir ... [--device-memory-ratio ...] [--host-only] [--output-json ...]` | You want expert compute, eviction, and queue coordination timing. | `expert_compute`, `eviction`, `queue_coordination`, `observed_stages`, `non_empty_components` | Default `device_memory_ratio` is low on purpose to increase eviction pressure. |
| `python benchmarks/expert_io_microbench/bench_bubble.py --model ... --offload-dir ... [--host-only] [--output-json ...]` | You want bubble accounting and step decomposition. | `overall_bubble_ratio`, `per_layer_bubble`, `step_decomposition_mean`, `step_decomposition_percentiles` | Uses IOProfiler data to estimate how much of each step is spent waiting. |
| `python benchmarks/expert_io_microbench/compare_baseline.py --nsys-report ... --benchmark-json ... --output-json ...` | You want to compare a baseline `.nsys-rep` profile against a new benchmark JSON. | `new_instrumentation_overhead_pct`, `overhead_regression_pct`, `verdict`, `baseline_nvtx_ranges` | Requires `nsys` on PATH or through `NSYS_BIN`. |
| `nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi python benchmarks/expert_io_microbench/run_decision_profile.py --model ... --offload-dir ... --hardware-tag ... --mode disk\|host-only --output-json ...` | You want the nsys-go/no-go profile for the IBP feasibility plan. | `decode_step_times_ns`, `decode_step_total_ns`, `pcie_link_width_observed`, `pcie_link_gen_observed` | Must run inside the `nsys profile` wrapper so `cudaProfilerStart/Stop` brackets the measured decode steps. |
| `python benchmarks/expert_io_microbench/nsys_parser.py <rep.nsys-rep> [--steps N] [--link-width W] [--link-gen G] [--real-total-ns N]` | You want the raw summary tuple from a captured `.nsys-rep`. | `T_step_ns`, `T_transfer_ns`, `Util_pcie`, `verdict` | Uses `nsys stats` and a fixed PCIe link model, or the observed link width and generation if you pass them. |

## Environment variables

| Env var | Default | Meaning |
| --- | --- | --- |
| `MOE_INFINITY_PROFILE_IO` | set to `1` by the scenario scripts | Enables IOProfiler capture in the microbench scripts. |
| `MOE_INFINITY_PROFILE_IO_SAMPLE` | set to `1.0` by the scenario scripts | Forces full sampling so the stage timings are easier to compare. |
| `NSYS_BIN` | unset | Optional absolute path to the `nsys` binary for `nsys_parser.py`. |
| `CUDA_VISIBLE_DEVICES` | inherited | Standard CUDA device selection. |

## Expected outputs

- `run_all.py` writes one merged JSON object with `scenarios`, `bandwidth_analysis`, and `executive_summary`.
- `bench_routing.py` and `bench_transfer.py` return stage timing summaries and a `status` of `PASS` or `BLOCKED`.
- `bench_compute_evict.py` adds `observed_stages` and `non_empty_components` so you can see whether all expected stages were hit.
- `bench_bubble.py` reports `overall_bubble_ratio`, per-layer bubble ratios, and step decomposition percentiles.
- `compare_baseline.py` returns `verdict` together with the measured overhead regression percentage.
- `run_decision_profile.py` records the raw decode step times and the sampled PCIe link geometry used for the verdict.

## NVTX and profiler workflow

1. Capture a profile with `run_decision_profile.py` inside `nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi`.
2. Use `compare_baseline.py` to compare the new benchmark JSON against the baseline `.nsys-rep`.
3. Use `nsys_parser.py` when you want the raw transfer and compute breakdown without the comparison wrapper.

The decision-profile runner toggles `cudaProfilerStart` and `cudaProfilerStop` around the measured decode steps. Warmup runs happen before the profiler starts, so they are excluded from the timing window.

## Throughput and latency interpretation

- Routing, transfer, compute, and bubble are stage microbenchmarks. Treat them as profiler tools, not as end-to-end serving throughput.
- `--host-only` copies the offload directory to tmpfs. That removes disk I/O and exposes the lower bound when the experts live in CPU RAM.
- Compare disk mode against host-only mode to separate disk time from PCIe time.
- `overall_bubble_ratio` is the share of step time spent waiting. Lower is better.
- `sync_overhead.sync_pct_of_step` shows how much of each step disappears into synchronization.
- `bandwidth_analysis.top_bottlenecks` from `run_all.py` ranks the worst components by share of step time.

## GPU-only expert routing A/B and Nsight runbook

Run both modes with the same checkout, checkpoint, offload tree, GPU visibility, cache ratio, prompt, output length, and greedy decoding:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/serving/latency.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /tmp/moe-infinity-bench/deepseek-v2-lite-chat \
  --concurrency 1 2 4 \
  --prompt-length 128 \
  --max-new-tokens 64 \
  --warmup-rounds 3 \
  --num-rounds 30 \
  --gpu-only-expert-routing off \
  --output-json /tmp/gpu-routing-off.json

CUDA_VISIBLE_DEVICES=0 python benchmarks/serving/latency.py \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /tmp/moe-infinity-bench/deepseek-v2-lite-chat \
  --concurrency 1 2 4 \
  --prompt-length 128 \
  --max-new-tokens 64 \
  --warmup-rounds 3 \
  --num-rounds 30 \
  --gpu-only-expert-routing on \
  --routing-baseline-json /tmp/gpu-routing-off.json \
  --output-json /tmp/gpu-routing-on.json
```

```bash
MOE_INFINITY_PROFILE_IO=1 MOE_INFINITY_PROFILE_IO_SAMPLE=1.0 \
CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true \
  --output=/tmp/gpu-routing-on \
  python benchmarks/expert_io_microbench/run_decision_profile.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /tmp/moe-infinity-bench/deepseek-v2-lite-chat \
    --hardware-tag single-host --mode host-only \
    --gpu-only-expert-routing on \
    --warmup-iters 3 --warmup-tokens 8 --iters 3 --max-new-tokens 32 \
    --output-json /tmp/gpu-routing-profile.json

python benchmarks/expert_io_microbench/nsys_parser.py \
  /tmp/gpu-routing-on.nsys-rep --steps 96 \
  --profile-json /tmp/gpu-routing-profile.json \
  > /tmp/gpu-routing-nsys-summary.json
```

- TPOT is decode elapsed time divided by generated tokens after the first token; `itl_*` remains an alias for compatibility.
- Compare off/on runs only on the same commit, process environment, checkpoint, offload tree, visible GPUs, cache ratio, prompt/output lengths, and greedy mode.
- KEEP requires semantic tests to pass, zero route failures, zero unexpected eager fallbacks, TPOT p50 regression <=2%, and TPOT p99 regression <=5% at concurrency 1, 2, and 4.
- Roll back by setting `gpu_only_expert_routing=false`; do not remove eager bindings during the rollout.
- First-release runs must keep `speculative_prefetch_overlap=false` and any `overlap_prefetch_mode` at `off`; `observe`/`enforce` combinations are configuration errors, not benchmark variants.
- A later reconciliation may permit simultaneous enablement only after both plans share generation, active-list, completion, retirement, and failure ownership contracts.
- Results apply to single-host personal-machine offloading only. They are not multi-node results and are not promises of DeepEP or paper-level speedups.

## Historical results

This README does not pin a canonical result set. Any JSON or markdown you archive from a past run is historical, so label it with the commit, model, hardware, and date if you share it elsewhere.
