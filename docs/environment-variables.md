# Environment variables

This page lists the env keys read by production code and build files in this
repo.
It excludes test harness variables and other repo-local fixtures.

Exact string comparisons matter. Several toggles only react to `"1"` or
`"0"`.

## Runtime and serving

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_API_KEYS` | unset | `moe_infinity/entrypoints/openai/api_server_v2.py` startup | Comma-separated bearer tokens for the OpenAI server. If non-empty, auth middleware enforces them. | CLI `--api-key` wins when present. Empty string means no auth. |
| `CONTEXTPILOT_ENABLED` | `"1"` | `moe_infinity/entrypoints/openai/api_server_v2.py` startup and status | If `"0"`, force ContextPilot off. | `--enable-contextpilot` (or the corresponding programmatic enable) is still required; any nonzero value only removes the disable override and does not enable ContextPilot by itself. |

## Acceleration fallbacks

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_DISABLE_FUSED_KERNELS` | `"0"` | `moe_infinity/kernel/__init__.py` import time | If `"1"`, QKV and FFN use eager/PyTorch fallback, while decode attention bypasses the fused decode wrapper and routes through `paged_attention_fwd`, which uses the CUDA paged-attention kernel when available and otherwise falls back to torch SDPA. | Read at import time, so set it before importing `moe_infinity.kernel`. |
| `MOE_DISABLE_CUDA_GRAPHS` | unset, treated as off | `moe_infinity/serving/cuda_graph.py` init | If `"1"`, disable CUDA graph warmup, capture, and replay. | Read when `CudaGraphRunner` is constructed. |

## Profiling and tracing

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_INFINITY_PROFILE_IO` | `"0"` in `IOProfiler`, truthy check in `ExpertTracer` | `moe_infinity/profiling/io_profiler.py`, `moe_infinity/memory/expert_tracer.py` | Enable I/O event capture, atexit flush, and trace recording. | The two consumers do not check it the same way. `IOProfiler` uses `== "1"`; `ExpertTracer` currently uses truthiness of the env string. |
| `MOE_INFINITY_PROFILE_IO_OUT` | unset | `moe_infinity/profiling/io_profiler.py` | Append JSONL events to this path on flush. | No output if unset. |
| `MOE_INFINITY_PROFILE_IO_SAMPLE` | `"1.0"` | `moe_infinity/profiling/io_profiler.py` | Sampling probability, clamped to `[0.0, 1.0]`. | `0` disables, `1` records all. |

With `gpu_only_expert_routing=true`, full sampling emits `gpu_route_submit`,
`gpu_route_fallback`, and the existing `sync_wait` stages. Native NVTX adds
`gpu_route_handoff` and `expert_completion_handoff`. A nonzero
`route_failures` count or any unexpected `gpu_route_fallback` event is a
rollback signal; set `gpu_only_expert_routing=false` and retain the eager path.

## Deterministic mode

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_DETERMINISTIC` | unset, treated as off | `moe_infinity/kernel/deterministic_matmul.py` import time | If `"1"`, turn on deterministic algorithms, set `CUBLAS_WORKSPACE_CONFIG=:16:8`, and set `NCCL_ALGO=Tree`. | The module snapshots and restores the previous values on disable. `CUBLAS_WORKSPACE_CONFIG` and `NCCL_ALGO` are standard CUDA/NCCL knobs, not MoE-Infinity env vars. |

## Model-specific toggles

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `MOE_INFINITY_MXFP4_DEQUANT` | unset, treated as off | `moe_infinity/runtime/model_offload.py` checkpoint load path | If `"1"`, dequantize MXFP4 weights to BF16 while loading the offload cache. | Only matters for MXFP4 checkpoints. |
| `MOE_DSV4_FORCE_NATIVE` | unset, auto | `moe_infinity/models/deepseek_v4/official_offload_adapter.py` | If `"0"`, force the non-native DeepSeek-V4 path. Otherwise auto-select native on Blackwell when the extension is available. | An explicit `use_native` argument still wins. |

For `CUDA_VISIBLE_DEVICES` ordering, expert ownership, and one-host multi-GPU behavior, see [Single-server multi-GPU](multi-gpu.md).

## Build and packaging

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `CUDA_HOME` | resolved by search, final fallback `/usr/local/cuda` | `setup.py` build | Locate the CUDA toolkit and feed it to the extension build. | The script also writes the resolved value back into `os.environ["CUDA_HOME"]` and `cpp_extension.CUDA_HOME`. |
| `CUTLASS_DIR` | `~/cutlass` in `setup.py`, `$HOME/cutlass` in root `CMakeLists.txt` | `setup.py`, `CMakeLists.txt` | Locate CUTLASS headers. | Same env key, same purpose, two build systems. |
| `NVTX_DISABLE` | `"0"` | `setup.py` build | If `"1"`, compile out NVTX instrumentation macros. | Build-time only. |
| `MOE_ENABLE_SM90` | `"1"` | `setup.py` build | Include sm_90 kernels in the compiled extensions. | Build-time only. |
| `MOE_ENABLE_SM120` | `"0"` | `setup.py` build | Include sm_120 kernels and the native FP4 extension arch flags. | Build-time only. |

The package version is not set via an environment variable; it is derived from
git tags at build time by setuptools-scm (see `pyproject.toml`).

## Standard third-party envs

| Variable | Default | Where read | Effect | Notes |
| --- | --- | --- | --- | --- |
| `TRANSFORMERS_CACHE` | unset | `moe_infinity/entrypoints/big_modeling.py` checkpoint download | HuggingFace snapshot cache directory. | Standard HuggingFace env, not project-defined. |
| `HOME` | shell home directory | `CMakeLists.txt` fallback | Used only to construct the default CUTLASS path when `CUTLASS_DIR` is unset in CMake. | Standard shell env, not project-defined. |

## Test-only and harness variables

These show up in repo tests and benchmarks, but they are not part of the
production env surface documented above.

- `MOE_GLM_TINY`, `MOE_GLM_SMOKE`, `MOE_GLM_MEDIUM`
- `GPT_OSS_E2E`, `GPT_OSS_CHECKPOINT`, `GPT_OSS_DEVICE_MEMORY_RATIO`
- `MOE_DFLASH_OFFLOAD` — DFlash GPU tests use it to override the offload root without editing the test; defaults to a model-specific cache path.
- `DSV4_FLASH_CKPT`
- `MOE_DFLASH_GPU`, `MOE_DFLASH_MEM_RATIO`
- `WORLD_SIZE`, `RANK`, `LOCAL_RANK`

Repo evidence:
- `moe_infinity/kernel/__init__.py`
- `moe_infinity/serving/cuda_graph.py`
- `moe_infinity/profiling/io_profiler.py`
- `moe_infinity/memory/expert_tracer.py`
- `moe_infinity/kernel/deterministic_matmul.py`
- `moe_infinity/runtime/model_offload.py`
- `moe_infinity/models/deepseek_v4/official_offload_adapter.py`
- `moe_infinity/entrypoints/openai/api_server_v2.py`
- `setup.py`
- `CMakeLists.txt`
