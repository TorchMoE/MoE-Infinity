# MoE-Infinity

MoE-Infinity is a cost-effective, fast, and easy-to-use library for Mixture-of-Experts (MoE) inference.

## Overview

MoE-Infinity runs large Mixture-of-Experts models on memory-constrained GPUs by offloading expert weights to host memory and SSD, then fetching them when needed. An activation-aware cache keeps hot experts resident on the GPU, while tracing and prefetching hide transfer cost. On top of the offloading runtime, MoE-Infinity ships a HuggingFace-compatible `MoE` class and an OpenAI-compatible serving engine with continuous batching, paged KV cache, and streaming.

This open-sourced version is HuggingFace-friendly and differs from the version reported in the [paper](https://arxiv.org/abs/2401.14361), which prioritized extreme performance. Start at [docs/README.md](docs/README.md) for the longer guides. Single-server multi-GPU inference is supported, with expert parameters distributed round-robin across visible GPUs, per-GPU caching, topology-qualified peer access when available, explicit host-staging fallback otherwise, and dedicated I/O threads. Multi-node distributed inference, across separate machines, is not yet supported.

## Contents
- [Documentation Hub](docs/README.md)
- [Model Compatibility](docs/model-compatibility.md)
- [DFlash](#dflash)
- [Serving](docs/serving.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Architecture](./ARCHITECTURE.md)
- [Changelog](./CHANGELOG.md)
- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from conda environment](#install-from-conda-environment)
    - [Install from PyPI](#install-from-pypi)
    - [Install from Source](#install-from-source)
    - [Enable FlashAttention (Optional)](#enable-flashattention-optional)
    - [Enable FlashInfer (Optional)](#enable-flashinfer-optional)
- [Usage and Examples](#usage-and-examples)
    - [Running Inference](#running-inference)
    - [Benchmarking](#benchmarking)
    - [OpenAI-Compatible Server (Continuous Batching)](#openai-compatible-server-continuous-batching)
- [ContextPilot Integration (Optional)](#contextpilot-integration-optional)
- [Release Plan](#release-plan)
- [Contributing and Security](#contributing-and-security)
- [Citation](#citation)

## Key Features

- **Cost-effective.** Expert offloading to host memory and SSD lets memory-constrained GPUs serve MoE models that would otherwise not fit. DeepSeek-V4-Flash additionally offloads **FP4-quantized** experts.
- **Fast.** Activation-aware expert caching, prefetching, tracing, fused CUDA kernels, CUDA graph capture, Marlin INT4 GEMM, and FP4/MXFP4 expert paths keep the hot path lean.
- **HuggingFace-native.** The `MoE` class remains the current in-process synchronous API, but `MoE.generate()` emits `DeprecationWarning` and is scheduled for removal. Use `MoE.serve()` for continuous batching; it starts an async HTTP service and is not a drop-in in-process return API.
- **Production serving.** OpenAI-compatible HTTP server with continuous batching, paged KV cache, request scheduling with preemption, streaming (SSE), runtime hot reload, watchdog/health monitoring, and crash-recovery logging. Opt-in correctness-preserving prefix KV reuse (`--enable-prefix-caching`, default off) reuses exact shared prefixes on the supported Qwen3 + FlashInfer path with cold/warm equivalence; unsupported runtimes fall back to the cold path unchanged. See [docs/serving.md](docs/serving.md#prefix-caching).
- **Acceleration-aware.** Automatically integrates with [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [FlashInfer](https://flashinfer.ai/) when installed, with graceful fallback to built-in kernels.
- **DFlash.** The experimental direct speculator API supports batch-1 greedy and sampled draft/verify without a stable API promise; deprecated `MoE.generate()` and continuous serving delegate only greedy singleton requests. Batch>1 is currently greedy-only on the bare HuggingFace target path. Route-ahead is an executor-path capability, not evidence of a validated target/drafter pair; see the model-by-model status in [docs/dflash.md](docs/dflash.md#compatibility).
- **Production serving.** OpenAI-compatible HTTP server with continuous batching, paged KV cache, request scheduling with preemption, streaming (SSE), runtime hot reload, watchdog/health monitoring, and crash-recovery logging. A prefix-cache flag and cache scaffolding exist, but the current OpenAI request path does not actively reuse cached prefixes; see [docs/serving.md](docs/serving.md#prefix-caching).
- **Acceleration-aware.** Automatically integrates with [FlashAttention](https://github.com/Dao-AILab/flash-attention) and uses FlashInfer where the selected standard paged-attention backend supports it, with graceful fallback to built-in kernels. DeepSeek MLA currently uses the correct PyTorch fallback and does not claim FlashInfer acceleration.
- **DFlash.** One session semantic core now covers direct, deprecated-sync, and serving draft/verify decisions. The experimental direct bare-HF API supports batch-1/batch>1 greedy, sampled, and mixed rows. Physical rich batching is capability-gated; unsupported MLA/hybrid wrappers run grouped per-request sessions. Serving keeps Stage 4a dynamic fallback, with default-off Stage 4b paged MLA limited to eligible greedy batch-1 DeepSeek V2/V3. Pairing and executor route-ahead evidence remain separate; see [docs/dflash.md](docs/dflash.md).
- **Multi-GPU.** Single-server multi-GPU with round-robin expert distribution, per-GPU caching, and an in-memory N-way tensor-parallel shard loader; see [docs/multi-gpu.md](docs/multi-gpu.md) and [docs/troubleshooting.md](docs/troubleshooting.md).

## Supported Models

MoE-Infinity supports HuggingFace MoE checkpoints registered in [`moe_infinity/common/constants.py`](./moe_infinity/common/constants.py). See [docs/model-compatibility.md](docs/model-compatibility.md) for the detailed compatibility matrix and model-specific notes.

| Model | Example checkpoints |
|---|---|
| [DeepSeek-V2 / V3](https://huggingface.co/collections/deepseek-ai/deepseek-v2-669a1c8b8f2dbc203fbd7746) | `deepseek-ai/DeepSeek-V2-Lite-Chat`, `deepseek-ai/DeepSeek-V3` |
| DeepSeek-V4-Flash (FP4 expert offloading) | `deepseek-ai/DeepSeek-V4-Flash` |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Mixtral-8x22B` |
| [Qwen3-MoE](https://huggingface.co/Qwen/Qwen3-30B-A3B) | `Qwen/Qwen3-30B-A3B` |
| [Qwen3.5-MoE](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | `Qwen/Qwen3.5-35B-A3B` |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2-FP8) | `zai-org/GLM-5.2-FP8` |
| [GPT-OSS](https://huggingface.co/models?search=gpt-oss) | `openai/gpt-oss-*` |
| [DBRX](https://huggingface.co/models?search=dbrx) | `databricks/dbrx-instruct` |
| [Jamba](https://huggingface.co/models?search=jamba) | `ai21labs/Jamba-*` |
| [OLMoE](https://huggingface.co/models?search=olmoe) | `allenai/OLMoE-*` |
| [Meta NLLB-MoE](https://huggingface.co/facebook/nllb-moe-54b) | `facebook/nllb-moe-54b` |

> DeepSeek-V4-Flash is only registered when your installed `transformers` provides `DeepseekV4ForCausalLM`; otherwise it is skipped automatically. Path A uses the HF-native `MoE` wrapper, and Path B uses the official FP4 offload loader. See [docs/model-compatibility.md](docs/model-compatibility.md) and [moe_infinity/models/deepseek_v4/README.md](./moe_infinity/models/deepseek_v4/README.md).

> Qwen3.5-MoE (`Qwen3_5MoeForConditionalGeneration`, requires `transformers` >= 5.12) is a vision-language checkpoint served text-only. Its 256 routed experts are offloaded while the text backbone, token embeddings, hybrid linear and full attention layers, shared expert, and `lm_head` stay resident on GPU. The v5 packed expert tensors expand to per-expert on load. Vision and MTP weights are present but unused for text generation. See [docs/model-compatibility.md](docs/model-compatibility.md).

Text-only Qwen3.5-MoE quick start:

```python
from moe_infinity import MoE

model = MoE("Qwen/Qwen3.5-35B-A3B", {
    "offload_path": "/ssd/moe-infinity/qwen3.5-35b-a3b",
    "device_memory_ratio": 0.5,
})
```

See the [model compatibility matrix](docs/model-compatibility.md) for the
validated scope and current limitations.

> GLM-5.2 (`GlmMoeDsaForCausalLM`, `model_type="glm_moe_dsa"`) requires `transformers` >= 5.12 and is registered only when that class is importable, otherwise it is skipped automatically. Its 256 routed FP8 experts are offloaded, while the 3 dense layers, shared expert, MLA attention, DSA indexer, and MTP layer stay resident. The routed experts stay FP8 in the host store, and non-routed FP8 weights are dequantized to BF16 on load. Sparse attention uses `attn_implementation="eager"`. See [docs/glm-5.2.md](docs/glm-5.2.md) and [docs/model-compatibility.md](docs/model-compatibility.md).

## Installation

We recommend installing MoE-Infinity in a virtual environment. To install MoE-Infinity, you can either install it from PyPI or build it from source.

### Prerequisites

- Python 3.10+ (3.12 recommended). Some required dependencies (e.g. `sglang-kernel`) publish wheels for Python ≥ 3.10 only, so Python 3.8/3.9 will fail to install.
- A CUDA-capable GPU. The from-source build targets compute capabilities `sm_80`/`sm_90` by default; for Blackwell (`sm_120`, e.g. RTX PRO 6000 / RTX 50-series) build with `MOE_ENABLE_SM120=1` (see [Install from Source](#install-from-source)).
- When building from source, a CUDA toolkit whose major version matches your installed PyTorch build (PyTorch enforces this at compile time).
- Recommended: isolated virtual environment (conda or venv).

### Install from conda environment

```bash
conda create -n moe-infinity python=3.12
conda activate moe-infinity
# install from either PyPI or Source will trigger requirements.txt automatically
```

### Install from PyPI

> **Note:** Official PyPI wheels are not published yet, the current `moe-infinity` entry on PyPI is a placeholder that does **not** contain the runtime. Importing `MoE` from it will fail. Until the official release, please [install from source](#install-from-source).

```bash
# (available once official wheels are published) stable release
pip install moe-infinity

# (available once official wheels are published) nightly / pre-release build
pip install --pre moe-infinity
```

### Install from Source

Building the CUDA/C++ extensions needs a few system packages, the CUTLASS headers, and PyTorch installed **before** `pip install -e .`:

```bash
# 1. System build dependencies (Debian/Ubuntu; use your distro's equivalents otherwise)
sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build git uuid-dev

# 2. Build tools + PyTorch. Match PyTorch's CUDA build to your CUDA toolkit
#    (pick the index URL for your CUDA version from https://pytorch.org).
pip install "setuptools>=78.1.1,<82" "setuptools-scm>=8" wheel ninja py-cpuinfo
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. CUTLASS headers (header-only; no separate build required)
git clone --depth 1 https://github.com/NVIDIA/cutlass.git ~/cutlass
export CUTLASS_DIR=~/cutlass

# 4. Build and install MoE-Infinity
git clone https://github.com/EfficientMoE/MoE-Infinity.git
cd MoE-Infinity
pip install --no-build-isolation -e .

# 5. Ensure a recent libstdc++ is available for the compiled extensions
conda install -c conda-forge libstdcxx-ng=12 # with conda; otherwise install libstdc++ (gcc 12+) via your package manager
```

**Building for Blackwell / SM120 GPUs** (RTX PRO 6000, RTX 50-series): the default build targets `sm_80`+`sm_90`. Enable the `sm_120` path (and the native FP4 kernel) explicitly:

```bash
MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR=~/cutlass pip install --no-build-isolation -e .
```

### Enable FlashAttention (Optional)

FlashAttention is **not** installed by default. Install it (>=2.5.2) if you want the FlashAttention path:
```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
# or, equivalently, via the optional extra:
pip install -e '.[flash_attn]'
```
Post-installation, MoE-Infinity will automatically use FlashAttention when available.

### Enable FlashInfer (Optional)

Install [FlashInfer](https://flashinfer.ai/) for optional optimized standard paged-attention kernels during prefill and decode. It does not currently accelerate DeepSeek MLA.

```bash
# Install the FlashInfer Python package (JIT-compiles kernels to match your Torch/CUDA):
pip install flashinfer-python
# or, equivalently, via the optional extra:
pip install -e '.[flashinfer]'
```

Check the [FlashInfer installation guide](https://docs.flashinfer.ai/installation.html) for prebuilt-wheel options matching specific CUDA/PyTorch versions.

Post-installation, MoE-Infinity will detect and use FlashInfer where the selected backend supports it. When FlashInfer is not installed, it falls back to built-in attention kernels with no behavior change.

## Usage and Examples

We provide a simple API for diverse setups, including single GPU and multiple GPUs. The following examples show how to use MoE-Infinity to run generation on a Huggingface LLM model.

### Important Note

- The `offload_path` must be unique for each MoE model. Reusing the same `offload_path` for different MoE models will result in unexpected behavior.
- For multi-GPU ownership, visible-device ordering, and common failure modes, see [docs/multi-gpu.md](docs/multi-gpu.md) and [docs/troubleshooting.md](docs/troubleshooting.md).


### Sample Code of Huggingface LLM Inference

> **Deprecation notice:** The examples below retain `MoE.generate()` because it
> is the current in-process synchronous path and remains covered by repository
> tests. It emits `DeprecationWarning` and is scheduled for removal. For
> continuous batching, use `MoE.serve()` or the OpenAI-compatible server; that
> HTTP serving path is not a drop-in replacement for an in-process tensor return.

```python
import torch
import os
from transformers import AutoTokenizer
from moe_infinity import MoE

user_home = os.path.expanduser('~')

checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

config = {
    "offload_path": os.path.join(user_home, "moe-infinity"),
    "device_memory_ratio": 0.75, # 75% of the device memory is used for caching, change the value according to your device memory size on OOM
}

model = MoE(checkpoint, config)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")

output_ids = model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
```

For the source-derived config reference, see [docs/configuration.md](docs/configuration.md).
Runtime and build env vars are listed in [docs/environment-variables.md](docs/environment-variables.md).

### Running Inference

Run on a single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python script.py
```

Run on multiple GPUs (expert parameters are automatically distributed across all visible devices):
```bash
CUDA_VISIBLE_DEVICES=0,1 python script.py
```

We provide ready-to-run examples under [`examples/`](./examples). The scripts download the checkpoint, run inference on the input, and print the output.

```bash
# Minimal single-prompt example (DeepSeek-V2-Lite-Chat)
CUDA_VISIBLE_DEVICES=0 python examples/readme_example.py --checkpoint deepseek-ai/DeepSeek-V2-Lite-Chat --offload_dir <your local path on SSD>

# Streaming benchmark example over GSM8K (TTFT + decode timing)
CUDA_VISIBLE_DEVICES=0 python examples/interface_example.py --model_name_or_path "deepseek-ai/DeepSeek-V2-Lite-Chat" --offload_dir <your local path on SSD>
```

**Suggested hardware (DeepSeek-V2-Lite-Chat):** a single GPU with **>= 16 GB VRAM** (e.g. RTX 4090 / A5000 / A100) plus a fast local SSD for `--offload_dir`. Lower `--device_memory_ratio` (default `0.75`) if you hit OOM on smaller GPUs.

### DeepSeek-V4-Flash (FP4 Expert Offloading)

DeepSeek-V4-Flash depends on the validated mp4/official-container setup, so this README does **not** present it as a self-contained canonical example. Use the detailed family guide for the required container, mounts, checkpoint prep, and the validated harness: [`moe_infinity/models/deepseek_v4/README.md`](./moe_infinity/models/deepseek_v4/README.md). The ContextPilot A/B benchmark entry point is [`benchmarks/contextpilot/v4flash_ab.py`](./benchmarks/contextpilot/v4flash_ab.py).

### GLM-5.2 (FP8 Expert Offloading)

GLM-5.2 (`zai-org/GLM-5.2-FP8`) runs through the drop-in `MoE` class:

```python
from moe_infinity import MoE

model = MoE("zai-org/GLM-5.2-FP8", {
    "offload_path": "/ssd/moe-infinity/glm-5.2",
    "device_memory_ratio": 0.5,
})
```

> **Memory note:** the FP8 block-scaled routed experts stay FP8 in the host store and are dequantized on-device by the expert dispatcher. Weights that run in PyTorch rather than the dispatcher, namely MLA attention, the DSA indexer, the dense-layer MLPs, and the shared expert, are dequantized to BF16 on load. Requires `transformers` >= 5.12.

## DFlash

See [docs/dflash.md](docs/dflash.md) for unified session semantics, direct
greedy/sampled/mixed batching, per-row RNG and scalar-generator correlation,
dense reconstruction, output padding and `last_generated_lengths`, grouped
versus physical rich execution, Stage 4a/4b ownership, and the separate pairing
versus executor evidence matrix. No real DeepSeek DFlash pair or GPT-OSS
executor route-ahead is implied.

No-download rollout gate:

```bash
python benchmarks/dflash/validate_unified_execution.py --fixture tiny \
  --require-cache-invariants --require-order-invariance
```

### Benchmarking

For correct throughput and latency measurement, it is critical to separate **prefill time (TTFT)** from **decode throughput**. Including prefill in your throughput calculation will produce misleadingly low numbers.

We provide a `StopWatch` utility and ready-to-use benchmark scripts. See the **[Benchmarking Guide](docs/benchmarking.md)** for:

- How to correctly measure decode throughput vs TTFT
- Common measurement pitfalls and how to avoid them
- Ready-to-use benchmark scripts (`benchmarks/serving/`)
- Fair comparison methodology with llama.cpp, vLLM, and other frameworks
- Tuning `device_memory_ratio` for optimal performance

Quick example using the benchmark scripts:
```bash
# Single-request baseline (TTFT + per-token latency + peak memory)
python benchmarks/serving/baseline_performance.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir

# Throughput sweep across batch sizes
python benchmarks/serving/throughput.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --batch-sizes 1 2 4 8 16

# Latency percentiles (TTFT + ITL at p50/p90/p99)
python benchmarks/serving/latency.py \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir /path/to/offload/dir \
    --concurrency 1 2 4 8
```

### OpenAI-Compatible Server (Continuous Batching)

MoE-Infinity includes a continuous batching OpenAI-compatible server.

> **Security:** The parser default is `--host 0.0.0.0`, and authentication is
> disabled when neither `--api-key` nor `MOE_API_KEYS` is configured. On an
> untrusted, shared, or cloud host, bind to `127.0.0.1` or configure an API key
> before exposure. Completion routes and privileged `/admin/stats`, `/v1/config`,
> and `/v1/reload` endpoints share this authentication posture.

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 --host 127.0.0.1 --model deepseek-ai/DeepSeek-V2-Lite-Chat --offload-dir ./offload_dir
```

```bash
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-ai/DeepSeek-V2-Lite-Chat","prompt":"Hello","max_tokens":32}'
```

For the full serving surface, auth, watchdogs, DFlash, and operational endpoints, see [docs/serving.md](docs/serving.md).

## ContextPilot Integration (Optional)

ContextPilot is an optional overlap-aware prompt optimization layer for shared-prefix and multi-turn workloads. You can enable it inside the OpenAI-compatible server before tokenization, or extend it into KV allocation and scheduling for deeper reuse.

Phase B quick start, in-process middleware:

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir ./offload_dir \
    --enable-contextpilot
```

Set `CONTEXTPILOT_ENABLED=0` to force-disable ContextPilot at runtime, even if the CLI flag is enabled. For setup details, CLI flags, environment variables, admin endpoints, and troubleshooting, see [docs/contextpilot/README.md](docs/contextpilot/README.md).

## Architecture

For a contributor-oriented map of the codebase, including the synchronous `engine/` path, async `serving/` path, module layout, request lifecycle, and public API surface, see **[ARCHITECTURE.md](./ARCHITECTURE.md)**.

## Release Plan

See [CHANGELOG.md](./CHANGELOG.md) for release history and unreleased notes. The current roadmap is:

- Improving vLLM runtime interoperability.
- Expert parallelism and multi-node distributed MoE inference.
- OpenAI-compatible Batch API (`/v1/batches`).

These are roadmap notes, not shipped releases.

## Contributing and Security

- See [CONTRIBUTING.md](./CONTRIBUTING.md) for development workflow, coding standards, and tests.
- See [SECURITY.md](./SECURITY.md) for vulnerability reporting and support policy.

## Citation

If you use MoE-Infinity for your research, please cite our [paper](https://arxiv.org/abs/2401.14361):
```bibtex
@misc{moe-infinity,
  author       = {Leyang Xue and
                  Yao Fu and
                  Zhan Lu and
                  Chuanhao Sun and
                  Luo Mai and
                  Mahesh Marina},
  title        = {MoE{-}Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache},
  archivePrefix= {arXiv},
  eprint       = {2401.14361},
  year         = {2024}
}
```
