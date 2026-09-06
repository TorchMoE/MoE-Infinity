# Benchmarking MoE-Infinity

This guide covers the benchmark entry points under `benchmarks/`, labels each one as a stable user workflow, contributor-only spike, helper or report tool, or excluded helper module, and keeps the TTFT vs decode guidance in one place.

If you want the cross-framework comparison table, start with [Benchmark reproduction](benchmark_reproduction.md). For the expert I/O profiler flow, see [benchmarks/expert_io_microbench/README.md](../benchmarks/expert_io_microbench/README.md). For ContextPilot and DeepSeek-V4-Flash background, see [ContextPilot](contextpilot/README.md) and [DeepSeek-V4-Flash](../moe_infinity/models/deepseek_v4/README.md).

## Key terminology

| Term | Definition |
| --- | --- |
| TTFT (Time To First Token) | Time from prompt submission until the first generated token appears. This includes prefill. |
| ITL (Inter-Token Latency) | Average time between generated decode tokens. |
| Decode throughput | Tokens per second during decode only. This excludes prefill. |
| End-to-end throughput | Total generated tokens divided by total wall clock time. This includes prefill. |
| Prefill | The initial phase where the model processes the prompt and builds KV cache state. |
| Decode | The autoregressive phase where the model emits one token at a time. |
| Peak memory | Maximum GPU memory observed during the run. |

## Measurement rules

- Keep the model checkpoint, quantization, residency, offload path, and `device_memory_ratio` fixed when you compare runs.
- Keep the GPU, CPU, RAM, storage, and software stack fixed too, including CUDA, PyTorch, and Transformers versions.
- Use the same prompt length, output length, batch size, and concurrency for both baselines.
- Warm up once before you start timing.
- Separate TTFT from decode throughput. If you mix them, your numbers stop being comparable.
- Compare like with like: p50-to-p50 or average-to-average. Do not compare a percentile on one side with an average on the other.
- Record the exact commit, command line, and output file path with every result.
- For cross-framework comparisons, use the same metric on both sides. For example, compare MoE-Infinity decode throughput with llama.cpp `eval time`, not `prompt eval time`.

## Using the StopWatch utility

MoE-Infinity provides a `StopWatch` class in [`examples/interface_example.py`](../examples/interface_example.py) that separates prefill from decode timing through HuggingFace `TextStreamer`.

### How StopWatch works

```
generate() called
  |
  v
put() called (1st time) -> start_prefilling = now
  |
  v
put() called (2nd time) -> prefilling_time = now - start_prefilling
                           start_decoding = now
                           clear expert cache counts
  |
  v
put() called (3rd+ time) -> decoding_iterations++
  |
  v
end() called -> decoding_time = now - start_decoding
```

The first callback marks the start of prefill. The second callback marks the first decoded token, which is the end of prefill and the start of decode.

If you want p50-style reporting from StopWatch, run multiple trials and take the median TTFT / E2E / decode-throughput values across runs. A single StopWatch run is a per-run measurement, not a percentile.

### Standalone measurement example

This example measures the current in-process synchronous `MoE.generate()` path.
That method emits `DeprecationWarning` and is scheduled for removal, so treat
the result as path-specific transition data. `MoE.serve()` is the recommended
continuous-batching HTTP path, not a drop-in replacement for this in-process
streamer benchmark.

```python
import time
import torch
from transformers import AutoTokenizer, TextStreamer
from moe_infinity import MoE


class StopWatch(TextStreamer):
    """Separates prefill (TTFT) from decode latency."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.start_decoding = time.time()
        self.decoding_iterations += 1
        return super().put(value)

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding
        return super().end()


# --- Setup ---
model_path = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

config = {
    "offload_path": "/path/to/offload/dir",
    "device_memory_ratio": 0.75,
}
model = MoE(model_path, config)

# --- Warmup (important!) ---
warmup_ids = tokenizer("Hello", return_tensors="pt").input_ids.to("cuda:0")
with torch.no_grad():
    model.generate(warmup_ids, max_new_tokens=8, pad_token_id=tokenizer.eos_token_id)

# --- Measurement ---
prompt = "Explain the theory of relativity in simple terms."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

streamer = StopWatch(tokenizer)
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# --- Results ---
print(f"TTFT (prefill):              {streamer.prefilling_time:.3f} s")
print(f"Decode time:                 {streamer.decoding_time:.3f} s")
print(f"Decode iterations:           {streamer.decoding_iterations}")
print(f"Per-token latency (decode):  {streamer.decoding_time / streamer.decoding_iterations:.4f} s")
print(f"Decode throughput:           {streamer.decoding_iterations / streamer.decoding_time:.2f} tokens/s")
```

## Benchmark catalog

| Workflow | Entry points | Purpose | Prerequisites | Metrics / output | Guide / status |
| --- | --- | --- | --- | --- | --- |
| Serving benchmarks | `benchmarks/serving/baseline_performance.py`<br>`benchmarks/serving/throughput.py`<br>`benchmarks/serving/latency.py`<br>`benchmarks/serving/memory.py`<br>`benchmarks/serving/kv_offload_benchmark.py` | Single request TTFT, throughput sweeps, concurrency latency, memory, and KV offload checks. | CUDA GPU, `transformers`, `moe_infinity`, model checkpoint, offload dir. | `ttft_ms`, `per_token_latency_ms`, `total_time_s`, `peak_gpu_memory_mb`, throughput per batch size, TTFT and ITL percentiles, expert hit rate, KV utilization, swap count. | Stable user workflow. Command examples are below. |
| DeepSeek-V4-Flash ContextPilot A/B | `benchmarks/contextpilot/v4flash_ab.py`<br>`benchmarks/contextpilot/v4flash_ab_report.py` | A/B compare ContextPilot on DeepSeek-V4-Flash and emit a markdown GO/NO-GO report. | 4 GPUs in mp4 setup, official DeepSeek-V4 checkpoint, `torchrun`, pinned host RAM, `v4flash` image. | `ttft_p50`, `e2e_p50`, prompt token savings, decode tok/s, GO/NO-GO report. | Stable user workflow. See the DeepSeek-V4 guide and the ContextPilot guide. |
| Serving validation and FlashInfer checks | `benchmarks/serving/ab_kernel_bench.py`<br>`benchmarks/serving/validate_flashinfer.py`<br>`benchmarks/serving/validate_batched_dispatch.py` | FlashInfer vs naive SDPA, FlashInfer import and correctness checks, batched dispatch feasibility. | CUDA GPU, FlashInfer for `validate_flashinfer.py`, source tree for dispatch analysis. | Speedup, GO/NO-GO summary, correctness checks, static interface analysis. | Contributor-only, experimental. Use before changing serving kernels or dispatch wiring. |
| ContextPilot phase benchmarks | `benchmarks/contextpilot/baseline.py`<br>`benchmarks/contextpilot/phase_a_benchmark.py`<br>`benchmarks/contextpilot/phase_b_benchmark.py`<br>`benchmarks/contextpilot/phase_c_benchmark.py`<br>`benchmarks/contextpilot/compare_phases.py`<br>`benchmarks/contextpilot/memory_profile.py`<br>`benchmarks/contextpilot/reorder_overhead.py`<br>`benchmarks/contextpilot/gen_longctx_workload.py` | Baseline generation, sidecar and middleware comparisons, scheduler phase comparisons, RSS profiling, reorder overhead, and workload generation. | Local or mocked ContextPilot, optional GPU or HTTP server for real phase runs, `psutil` for `memory_profile.py`. | TTFT p50/p90/p99, E2E p50/p90/p99, token savings, KV and expert hit rates, RSS over checkpoints, reorder p50/p90/p99, generated workload JSON. | Contributor-only, experimental. Use the ContextPilot integration guide. |
| Expert I/O microbench | `benchmarks/expert_io_microbench/run_all.py`<br>`benchmarks/expert_io_microbench/bench_routing.py`<br>`benchmarks/expert_io_microbench/bench_transfer.py`<br>`benchmarks/expert_io_microbench/bench_compute_evict.py`<br>`benchmarks/expert_io_microbench/bench_bubble.py`<br>`benchmarks/expert_io_microbench/compare_baseline.py`<br>`benchmarks/expert_io_microbench/run_decision_profile.py`<br>`benchmarks/expert_io_microbench/nsys_parser.py` | Routing, transfer, compute, bubble, merged bandwidth analysis, NVTX baseline comparison, and nsys go or no-go profiling. | CUDA GPU, model cache for the scenario scripts, offload dir, `/dev/shm` for host-only mode, `nsys` for the comparison and decision tools. | Routing and cache lookup mean and percentiles, transfer timing and sync overhead, expert compute / eviction / queue coordination, bubble ratio, bandwidth utilization, overhead regression percent, transfer step times, verdict. | Contributor-only, experimental. Detailed runbook in `benchmarks/expert_io_microbench/README.md`. |
| Cross-framework comparison suite | `benchmarks/comparison/run_all.sh`<br>`benchmarks/comparison/run_moe_infinity.py`<br>`benchmarks/comparison/run_vllm.py`<br>`benchmarks/comparison/run_llamacpp.py`<br>`benchmarks/comparison/aggregate_results.py` | Reproduce the single-GPU MoE-Infinity vs vLLM vs llama.cpp table. | Docker, GPU, offload dir, HuggingFace cache or access, benchmark images. | TTFT, per-token latency, peak GPU memory, per-model JSON, markdown, CSV, and JSON comparison tables. | Contributor-only, reproducibility workflow. See `docs/benchmark_reproduction.md`. |
| Performance model and roofline | `benchmarks/performance_model/bench_glm.py`<br>`benchmarks/performance_model/report_glm.py` | Tiny GLM decode versus MTP validation and roofline report generation. | CUDA GPU, `MOE_GLM_TINY=1`, `matplotlib`, `numpy`, conference plot helper. | Decode tok/s, MTP tok/s, mean accept length, peak memory, arithmetic intensity, predicted bound, markdown report and plots. | Contributor-only, experimental. Helper modules are excluded below. |
| Kernel microbenchmarks | `benchmarks/ab_fused_kernels.py`<br>`benchmarks/ab_kernels_micro.py`<br>`benchmarks/bench_p0_topk_softmax.py`<br>`benchmarks/mxfp4_benchmark.py` | Fused kernels on or off, kernel-only A/B, gating softmax microbench, and MXFP4 versus BF16 dequant. | CUDA GPU, optional `sglang-kernel`, optional FlashInfer, optional Triton or SM120 support depending on the script. | Median, p10, p90, p99 microseconds, speedups, correctness checks, TTFT, per-token latency, peak GPU memory, expert weight size. | Contributor-only, experimental. Do not treat these as production SLA numbers. |
| Evaluation utility | `benchmarks/eval/perplexity.py` | Perplexity evaluation over Wikitext, C4, or PTB. | CUDA GPU, `datasets`, `transformers`, offload dir, local model cache. | Perplexity, NLL, sample count, elapsed seconds. | Contributor-only utility, useful for model checks, not serving validation. |
| Decode CUDA graph qualification | `benchmarks/serving/decode_cuda_graph.py` | Paired eager/replay launch-overhead and correctness qualification. | CUDA; native paged-attention kernel for fixture mode. Model mode additionally requires a checkpoint and offload directory. | Raw microseconds, p50/p90/p99, observed ratio, launch counts, replay coverage, graph/scratch bytes, capability evidence, and per-layer KV checksums. | Contributor-only, experimental; no speedup pass threshold. |

## Decode CUDA graph qualification

Use fixture mode to qualify the formally defined persistent two-layer resident
ordinary-GQA Qwen3 model. It constructs exact `Qwen3PagedAttention` layers, one
native `PagedAttentionBackend`, and one `PagedKVStorage` shared by scheduler
allocation, per-layer K/V tensors, graph scratch, and replay. It needs neither a
checkpoint nor an offload directory:

```bash
python benchmarks/serving/decode_cuda_graph.py \
  --mode fixture \
  --batch-sizes 1 2 4 \
  --context-sizes 128 512 \
  --warmup-iters 5 \
  --measure-iters 20 \
  --profile-launches \
  --output-json /tmp/decode-cuda-graph-fixture.json
```

Model mode requires both loader arguments:

```bash
python benchmarks/serving/decode_cuda_graph.py \
  --mode model \
  --model deepseek-ai/DeepSeek-V2-Lite-Chat \
  --offload-dir /tmp/moe-offload \
  --batch-sizes 1 \
  --context-sizes 128 \
  --warmup-iters 1 \
  --measure-iters 2 \
  --output-json /tmp/decode-cuda-graph-model-capability.json
```

Current offloaded MoE loaders are expected to report an explicit unsafe
capability and zero captures/replays. Treat that result as eager-fallback
capability evidence, not a graph comparison. Do not force capture or infer
resident Qwen3 utility from an offloaded or DeepSeek MLA run.

Retain the raw JSON. It records CUDA, PyTorch, GPU, dtype, bucket configuration,
storage owner, registered/proved layer counts, per-layer KV checksums, graph
private-pool bytes, authoritative scratch bytes, replay coverage, raw eager and
replay samples, and p50/p90/p99 summaries. With `--profile-launches`, it also
records one eager and one replay profiler sample. `observed_ratio` is simply
`eager_p50_us / replay_p50_us`; values below 1.0 are valid. There is no
performance pass threshold, no automatic enablement, and no claimed speedup.

Run fixture and model evidence for every resident Qwen3/GPU/dtype/bucket set
being qualified. Denser capture points reduce padding but consume more graph
memory and authoritative scratch KV capacity, which can reduce request
concurrency. The stable-pointer and eager-boundary rationale follows
[TensorRT-LLM's piecewise CUDA graph guide](https://nvidia.github.io/TensorRT-LLM/features/torch_compile_and_piecewise_cuda_graph.html),
but MoE-Infinity's first rollout is narrower: resident native-paged
ordinary-GQA Qwen3 decode only; sampling, DeepSeek MLA, FlashInfer planning, and
offloaded MoE remain eager.

## DFlash validation

There is no standalone benchmark CLI under `benchmarks/` for DFlash. Use [`docs/dflash.md`](dflash.md) and the gated tests under `tests/python/dflash/` instead.

Typical validation commands:

```bash
MOE_DFLASH_GPU=1 pytest -q tests/python/dflash/test_gpu_serving_dflash.py
MOE_DFLASH_GPU=1 pytest -q tests/python/dflash/test_gpu_20b_dflash.py
MOE_DFLASH_GPU=1 pytest -q tests/python/dflash/test_gpu_120b.py
```

## Excluded helpers and fixtures

These files are support code, not standalone benchmark entry points.

| Path group | Why excluded |
| --- | --- |
| `benchmarks/comparison/__init__.py`, `benchmarks/contextpilot/__init__.py`, `benchmarks/eval/__init__.py`, `benchmarks/expert_io_microbench/__init__.py`, `benchmarks/performance_model/__init__.py`, `benchmarks/serving/__init__.py` | Package markers only. |
| `benchmarks/comparison/common.py` | Shared result dataclasses, model metadata, and loaders for the comparison suite. |
| `benchmarks/contextpilot/benchmark_utils.py`, `benchmarks/contextpilot/dataset_utils.py`, `benchmarks/contextpilot/http_benchmark.py` | Shared ContextPilot helpers, workload builders, and HTTP utilities. |
| `benchmarks/expert_io_microbench/harness.py`, `benchmarks/expert_io_microbench/stats.py` | Shared profiler and timing primitives for the expert I/O suite. |
| `benchmarks/performance_model/model_config.py`, `benchmarks/performance_model/roofline.py`, `benchmarks/performance_model/types.py` | Pure model and roofline math, plus dataclasses. |

## Serving benchmark commands

These are the stable serving workflows referenced in the catalog above.

### Baseline performance

Measures single request TTFT, per-token latency, and peak GPU memory.

```bash
python benchmarks/serving/baseline_performance.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --num-requests 10 \
    --output-json baseline_results.json
```

### Throughput sweep

Measures decode throughput across batch sizes and can compare against the baseline JSON.

```bash
python benchmarks/serving/throughput.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --num-requests 50 \
    --batch-sizes 1 2 4 8 16 32 \
    --prompt-length 128 \
    --max-new-tokens 16 \
    --output-json throughput_results.json
```

### Latency under concurrency

Measures TTFT and ITL percentiles across concurrency levels.

```bash
python benchmarks/serving/latency.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --concurrency 1 2 4 8 \
    --num-rounds 5 \
    --prompt-length 128 \
    --max-new-tokens 16 \
    --output-json latency_results.json
```

### Memory and KV offload

```bash
python benchmarks/serving/memory.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --batch-size 8 \
    --prompt-length 128 \
    --max-new-tokens 16 \
    --output-json memory_results.json

python benchmarks/serving/kv_offload_benchmark.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --num-requests 8 \
    --prompt-length 256 \
    --max-new-tokens 32 \
    --enable-kv-offload \
    --output-json kv_offload_results.json
```

## Comparing with other frameworks

### llama.cpp

llama.cpp reports timing in its output after `Ctrl+C`:

```
llama_perf_context_print: prompt eval time = 2251.78 ms /  39 tokens (57.74 ms per token, 17.32 tokens per second)
llama_perf_context_print:        eval time = 122985.89 ms / 491 runs  (250.48 ms per token, 3.99 tokens per second)
```

- `prompt eval time` is prefill. Compare it with MoE-Infinity TTFT.
- `eval time` is decode. Compare it with MoE-Infinity decode time.
- `eval` tokens per second is decode throughput. Compare it with `decoding_iterations / decoding_time`.

### vLLM and SGLang

These frameworks report TTFT and ITL directly in their benchmark outputs. Compare p50-to-p50 or average-to-average. For MoE-Infinity `StopWatch`, collect multiple trials and compare the median of those runs when you want p50-style reporting.

### Fair comparison checklist

- Same model weights, not quantized versus full precision.
- Same GPU and same `device_memory_ratio` or memory allocation.
- Same number of GPU layers offloaded, for example llama.cpp `-ngl`.
- Prefill excluded from decode throughput in both frameworks.
- At least one warmup run before measurement.
- Same `max_new_tokens` or generation length.
- Same sampling strategy, `do_sample=False` or greedy for deterministic comparison.
- Same batch or concurrency setting and the same prompt length.
- Same baseline file or comparison table revision when you cite the result.

## Tuning `device_memory_ratio`

`device_memory_ratio` controls what fraction of GPU memory is allocated for expert caching. The remainder is used by PyTorch for activations, KV cache, and other tensors.

| Value | Effect |
| --- | --- |
| Higher, for example 0.85 | More experts cached on GPU, fewer cache misses, faster decode. Risk, OOM if model activations are large. |
| Lower, for example 0.50 | Fewer experts cached, more cache misses, slower decode. Benefit, more headroom for large prompts or batches. |
| Default 0.75 | Good starting point for most single-GPU setups. |

If you encounter CUDA OOM errors, lower this value. If decode throughput is poor, try raising it, assuming no OOM.

## Reproducibility template

Fill this in for every benchmark run:

| Field | Value |
| --- | --- |
| Commit | `git rev-parse HEAD` |
| Benchmark workflow | `benchmarks/serving/throughput.py` |
| Model / checkpoint | `deepseek-ai/DeepSeek-V2-Lite-Chat` |
| GPUs, CPU, RAM, storage | `1 x A100 80GB, 32 CPU cores, 256 GB RAM, NVMe SSD` |
| CUDA, PyTorch, Transformers | `CUDA 12.8, PyTorch 2.5.x, Transformers 5.x` |
| Residency / offload / quantization | `device_memory_ratio=0.75, offload_dir=/ssd/offload, FP16` |
| Batch / concurrency | `batch_size=8` or `concurrency=4` |
| Prompt / output lengths | `prompt_length=128, max_new_tokens=16` |
| Warmup / iterations | `warmup=1, iters=50` |
| Baseline | `baseline_results.json` or `comparison_table.md` |
| Metrics captured | `ttft_ms, itl_p50_ms, decode_toks_per_s, peak_gpu_memory_mb` |
| Notes | `host-only, nsys, FlashInfer on, sampled off` |

## GPU-only expert routing A/B and Nsight runbook

Run both modes with the same checkout, checkpoint, offload tree, GPU
visibility, cache ratio, prompt, output length, and greedy decoding:

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
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output=/tmp/gpu-routing-on \
  python benchmarks/expert_io_microbench/run_decision_profile.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /tmp/moe-infinity-bench/deepseek-v2-lite-chat \
    --hardware-tag single-host \
    --mode host-only \
    --gpu-only-expert-routing on \
    --warmup-iters 3 \
    --warmup-tokens 8 \
    --iters 3 \
    --max-new-tokens 32 \
    --output-json /tmp/gpu-routing-profile.json

python benchmarks/expert_io_microbench/nsys_parser.py \
  /tmp/gpu-routing-on.nsys-rep \
  --steps 96 \
  --profile-json /tmp/gpu-routing-profile.json \
  > /tmp/gpu-routing-nsys-summary.json
```

- TPOT is decode elapsed time divided by generated tokens after the first token;
  `itl_*` remains an alias for compatibility.
- Compare off/on runs only on the same commit, process environment, checkpoint,
  offload tree, visible GPUs, cache ratio, prompt/output lengths, and greedy mode.
- KEEP requires semantic tests to pass, zero route failures, zero unexpected
  eager fallbacks, TPOT p50 regression <=2%, and TPOT p99 regression <=5% at
  concurrency 1, 2, and 4.
- Roll back by setting `gpu_only_expert_routing=false`; do not remove eager
  bindings during the rollout.
- First-release runs must keep `speculative_prefetch_overlap=false` and any
  `overlap_prefetch_mode` at `off`; `observe`/`enforce` combinations are
  configuration errors, not benchmark variants.
- A later reconciliation may permit simultaneous enablement only after both
  plans share generation, active-list, completion, retirement, and failure
  ownership contracts.
- Results apply to single-host personal-machine offloading only. They are not
  multi-node results and are not promises of DeepEP or paper-level speedups.
