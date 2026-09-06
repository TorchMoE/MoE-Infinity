# Configuration and memory planning

This page is the source of truth for `ArcherConfig` in
`moe_infinity.utils.config`.
It documents the Python config layer only. The OpenAI server has a separate
config layer with names like `offload_dir`, `device_memory_ratio`, and
`kv_cache_ratio`, so do not copy those semantics onto
`kv_cache_memory_ratio`.

## Serving crosswalk

| Python config | Current server name | Note |
| --- | --- | --- |
| `offload_path` | `--offload-dir` / `MoE.serve(offload_dir=...)` | The server builds `<offload_dir>/<model>` and uses that as the cache root. |
| `device_memory_ratio` | `--device-memory-ratio` / `MoE.serve(device_memory_ratio=...)` | Same split, same broad meaning. |
| `kv_cache_memory_ratio` | no direct CLI | Native-engine memory split only, the serving engine uses its own `kv_cache_ratio`. |
| `use_native_engine` | none | Python-only switch between native engine and HuggingFace `generate()`. |
| `enable_kv_cache_offload` | none | Native-engine scaffold only. |
| `enable_attention_offload` | none | Currently a no-op in stock code. |

## ArcherConfig fields

### File and trace fields

| Field | Type | Default | Valid values / range | Effect | Interactions / status |
| --- | --- | --- | --- | --- | --- |
| `offload_path` | `str` | `""` | filesystem path string, required in practice | Root of the offload store. The runtime creates the directory, writes `name_id_map.json` and `model_signature.json`, and reuses them on later loads. | Must be unique per model/config pair. A reused path with a different model or fingerprint raises on load. Stable. |
| `trace_capacity` | `int` | `1000` | any integer accepted by the parser, it must be large enough to allocate the trace tensor | Sizes the expert trace collection. | Loaded traces must fit in this capacity. Stable. |
| `trace_path` | `Optional[os.PathLike[str]]` | `None` | file path or `None`, directories are rejected | Intended to load a saved trace, but current production flow normalizes it to an absolute `str` and forwards that `str` into `ExpertTracer.load_trace()`, which only accepts `os.PathLike` or `np.ndarray`, so the load is currently ineffective. | Current behavior is a type mismatch in the call chain; no stability/support claim. |
| `perfect_cache_file` | `str` | derived | `<offload_path>/perfect_cache` | Derived internal path for the cache file. | No current production consumer. Internal. |
| `device_per_node` | `int` | derived | `torch.cuda.device_count()` | Records the visible device count at config init. | Internal metadata, not a control knob. Internal. |

### Memory and execution fields

| Field | Type | Default | Valid values / range | Effect | Interactions / status |
| --- | --- | --- | --- | --- | --- |
| `prefetch` | `bool` | `False` | `True` / `False` | No current production consumer. | Legacy or reserved knob. Reserved. |
| `speculative_prefetch` | `bool` | `False` | `True` / `False` | Enables expert prefetch driven by router logits. | Used by `DistributedExpertExecutor` and `ExpertPrefetcher`. Experimental. |
| `speculative_prefetch_overlap` | `bool` | `False` | `True` / `False` | Issues speculative prefetch before the dispatch barrier so PCIe copies can overlap compute. | Requires `speculative_prefetch=True`. Can increase cache pressure and trigger `All cached expert locked` warnings when `device_memory_ratio` is high. Experimental. |
| `gpu_only_expert_routing` | `false` | Opt in to native CUDA active-expert discovery for single-host `dispatch_local`. First release: mutually exclusive with `speculative_prefetch_overlap=true` and overlap-prefetch `observe`/`enforce`; invalid combinations raise before engine construction. CPU masks, older native extensions, and active DFlash route-ahead contexts use eager routing. Routing IDs, weights, output accumulation, and RPC `dispatch()` are unchanged. |
| `device_memory_ratio` | `float` | `0.9` | `[0.0, 1.0]` | Fraction of GPU memory reserved for expert cache and native-engine budgeting. | Pairs with `kv_cache_memory_ratio`. If the native zero-KV heuristic kicks in and the sum would exceed `1.0`, `device_memory_ratio` is reduced to fit. Stable. |
| `num_threads` | `int` | `4` | any integer accepted by the parser, positive values make sense in practice | Number of expert compute threads per GPU. | Passed to the expert dispatcher. Stable. |
| `host_memory_ratio` | `float` | `0.9` | any float accepted by the parser, no explicit validation | Reserved host-memory fraction. | No current production consumer. Reserved. |
| `kv_cache_memory_ratio` | `float` | `0.0` | `[0.0, 1.0]` | Fraction of GPU memory reserved for KV cache blocks in the native path. | If `use_native_engine=True` and this is `0.0`, `__post_init__` auto-sets it to `0.15` and warns. If the final sum still exceeds `1.0`, validation raises. Stable. |
| `use_native_engine` | `bool` | `True` | `True` / `False` | Chooses the native engine vs HuggingFace generation inside the deprecated synchronous `MoE.generate()` path. | `MoE.generate()` falls back to HF when this is false. The method emits `DeprecationWarning` and is scheduled for removal; `glm_moe_dsa` forces native off in `big_modeling`. |
| `enable_attention_offload` | `bool` | `False` | `True` / `False` | Enables attention offload scaffolding. | Stock code does not yet branch on this flag. The actual backend object is created inside `big_modeling`. Experimental. |
| `enable_kv_cache_offload` | `bool` | `False` | `True` / `False` | Enables KV cache offload scaffolding in the native engine. | Registers offload handlers when a native KV manager is present, but tensor wiring is still partial. Experimental. |
| `attention_backend` | `str` | `"default"` | any string accepted by parsing, only `default` has a documented meaning today | Legacy reserved field for attention backend selection. | The stock runtime does not consume this string. The active backend object comes from `big_modeling`. Reserved. |
| `overlap_prefetch_policy` | `str` | `"off"` | `off` / `observe` / `enforce` | Selects overlap-window byte admission for speculative expert prefetch. `off` keeps the legacy path byte-for-byte; `observe` computes decisions/metrics but issues the same transfers; `enforce` applies admission, cancellation, and native queue limits. | Eager-routing-only when active in the first release. `enforce` requires the rebuilt native extension for cancellation/backpressure; otherwise it fails closed. Experimental. |
| `overlap_prefetch_ewma_alpha` | `float` | `0.2` | `(0.0, 1.0]` | EWMA smoothing for compute/bandwidth/queue/issue calibration. | Only consumed under `observe`/`enforce`. |
| `overlap_prefetch_safety_factor` | `float` | `0.8` | `(0.0, 1.0]` | Fraction of measured compute time usable as the overlap transfer window. | Conservative < 1.0 leaves compute headroom. |
| `overlap_prefetch_cold_start_experts` | `int` | `1` | `>= 0` | Max experts admitted before both a compute and a transfer sample exist. | Cold start still enforces `overlap_prefetch_max_inflight_bytes`. |
| `overlap_prefetch_max_window_bytes` | `int` | `256*1024*1024` | `>= 0` | Upper bound on the per-layer admitted prefetch window in bytes. | Must be `<= overlap_prefetch_max_inflight_bytes` when policy is `enforce`. |
| `overlap_prefetch_max_inflight_bytes` | `int` | `512*1024*1024` | `>= 0` | Upper bound on outstanding speculative prefetch bytes. | Enforced by native backpressure under `enforce`. |
| `gpu_only_expert_routing` | `bool` | `False` | `True` / `False` | Shared field for the GPU-only expert routing plan; not implemented here. | Rejected together with `overlap_prefetch_policy=observe|enforce` in the first release (see compatibility table below). |

## Overlap-aware expert prefetch

The overlap-prefetch policy budgets speculative expert transfers by measured
transfer bandwidth and the current layer's measured compute time. It never
changes routing: the native router remains the sole source of masks, weights,
and the dispatched expert set. Budgeting applies only to early cache warming.

Per layer `l`, after warm calibration:

```text
T_window_ns(l) = max(0,
    safety_factor * compute_ewma_ns[l]
    - queue_wait_ewma_ns
    - issue_overhead_ewma_ns)
B_window(l) = floor(bandwidth_ewma_bytes_per_ns * T_window_ns(l))
B_admit(l)  = clamp(B_window(l) - current_inflight_bytes,
                    0, max_prefetch_window_bytes)
```

Admission is whole-expert greedy packing over candidates stable-sorted by
`(-score, original_position, expert_id)` using exact stored expert bytes.

- **Cold start** is conservative: until both a valid compute sample for the
  target layer and a valid transfer sample exist, at most
  `overlap_prefetch_cold_start_experts` experts are admitted, still bounded by
  `overlap_prefetch_max_inflight_bytes`.
- **Fail-closed:** a missing byte map, missing native telemetry API, invalid
  sample, or non-native engine causes `enforce` to admit nothing. It never
  fabricates average sizes or guessed bandwidth. `off` and `observe` retain
  compatibility behavior.
- **Rollout:** ship `off` (default), then `observe` to verify output equality
  and complete metrics, then `enforce` for the same model/hardware pair with
  `gpu_only_expert_routing=False`. `enforce` requires the rebuilt native
  extension for cancellation and backpressure.

### Cross-plan compatibility with GPU-only expert routing

First-release compatibility is deliberately fail-closed:

| `gpu_only_expert_routing` | `overlap_prefetch_policy` | Result |
| --- | --- | --- |
| `False` | `off`, `observe`, or `enforce` | Valid eager-routing configuration |
| `True` | `off` | Valid GPU-routing configuration; this plan is inactive |
| `True` | `observe` or `enforce` | `ValueError` during `ArcherConfig` construction/loading |

### Rollout and rollback

Roll out per model/hardware pair, one stage at a time:

1. Ship code with `overlap_prefetch_policy="off"`; verify legacy latency and
   native queue behavior are unchanged.
2. Enable `observe`; require output equality and complete non-negative metrics
   for at least the benchmark's configured measured iterations. Compare decision
   overhead and budget distributions, but do not block on a predetermined
   speedup.
3. Enable `enforce` only for the same model/hardware pair with
   `gpu_only_expert_routing=False`; examine p50/p95 latency, throughput,
   coverage, waste, late bytes, cancellation, queue rejection, and cache-pressure
   warnings together.
4. Keep independent allowlisting by model, quantization/storage format, GPU,
   PCIe generation/width, and host-vs-disk mode. DFlash and non-DFlash runs
   receive separate evidence; neither gates the other's availability.

Rollback requires no code or router change: set
`overlap_prefetch_policy="off"` and restart the worker. This restores the old
`prefetch_tensors` path and disables controller calls, cancellation, bounded
admission, and telemetry polling. If the rebuilt extension itself is suspect,
deploy the prior package: old Python never calls the new binding names. Do not
use `observe` as a rollback target — it still executes controller overhead.

Rollback triggers are measured regressions or instability, not promises: output
mismatch, native queue/accounting invariant failure, crash/deadlock, sustained
cache-lock warnings, an unbounded inflight gauge, or a model/hardware-specific
latency/throughput regression judged unacceptable by the owner.

### Risks and mitigations

| Risk | Detection | Mitigation |
| --- | --- | --- |
| EWMA reacts to transient bandwidth | bandwidth/queue EWMAs and p95 late bytes | conservative safety factor; per-process warmup; switch to `off` |
| Disk and host transfer samples mix | sample source in telemetry and host-only/disk arms | treat the EWMA as conservative effective bandwidth, reset calibration when offload mode changes, keep the safety factor |
| Queue debt double-counts capacity | inflight invariant and queue-reject bytes | subtract the native gauge; native max-inflight admission under one lock |
| Cancellation races with worker pop | cancel-after-start test | only remove queued tasks under the scheduler mutex; running tasks complete |
| Queue removal bypasses byte retirement | per-reason invariant tests; zero inflight after reset/replacement | centralized terminal helpers on every removal path |
| A worker operation throws | standard/unknown-exception no-throw worker tests and failed-byte invariant | per-task RAII failure retirement; never let an exception escape a native `std::thread` |
| Timing event comes from the disabled-timing pool | CUDA elapsed-time smoke and checked return codes | dispatcher owns timing-enabled epoch/start/stop events; never pool them |
| Forward throws with timing events in an unspecified state | fake-CUDA quarantine lifecycle tests | fence the same stream; retain until query/synchronize proves completion; never sample exception tickets |
| Output/host delay inflates the compute budget | kernel offsets versus output-delay fields | time only `ForwardHelper`; exclude `OutputFunc` and host completion from the EWMA |
| Delayed sample attributed to a newer execution of the same layer | mixed-invocation unit test | stamp every sample with a nonzero invocation id; calibrate only an exact invocation/layer match |
| GPU-only routing conflicts with correction/event ownership | config matrix tests | reject `gpu_only_expert_routing=True` with `observe`/`enforce` in the first release |
| `wait_expert` throws with admitted work outstanding | wait-error/failing-cleanup tests | cancel owned generations and drain native samples in error/finally without masking the original error |
| Missing/incorrect expert sizes | uncosted candidate metric | fail closed in `enforce`; never use an average |
| Native extension version skew | capability detection | `enforce` fails closed; `off` uses the legacy API |

## Memory ratio rules

`__post_init__` does not enforce a hard invariant up front.
It applies the following sequence:

1. If `use_native_engine=True` and `kv_cache_memory_ratio == 0.0`, it sets `kv_cache_memory_ratio = 0.15` and warns.
2. If that auto-fill makes `device_memory_ratio + kv_cache_memory_ratio > 1.0`, it shrinks `device_memory_ratio` to `1.0 - kv_cache_memory_ratio` and warns.
3. If either ratio is outside `[0, 1]`, or the final sum is still above `1.0`, it raises `ValueError`.

Normal `MoE` construction passes the already normalized `ArcherConfig` ratio into `MemoryCoordinator`; `MemoryCoordinator.from_config` keeps the same `0.15` fallback for direct callers that pass zero.

For single-server multi-GPU ownership, visible-device ordering, and cache locality, see [Single-server multi-GPU](multi-gpu.md).

## Offload store layout

`offload_path` must be unique per model and config fingerprint.
On first load the runtime writes:

- `name_id_map.json`
- `model_signature.json`

On later loads it verifies both the model name and the config fingerprint.
If either differs, it raises and tells you to use a different `offload_path` or delete the cache.

If the failure looks like a serving or timeout issue instead, check [Troubleshooting](troubleshooting.md).

## Native path notes

- `use_native_engine=False` keeps the HuggingFace `generate()` path.
- `enable_attention_offload` is still scaffolded.
- `enable_kv_cache_offload` is only meaningful when the native engine is built.
- `speculative_prefetch` drives the actual router-logit based expert prefetch.
- `speculative_prefetch_overlap` moves that prefetch earlier, before the barrier.

## Deprecated inputs

`load_from_json()` still accepts `glm_fp8_in_store`, but it warns and drops the key.
It is deprecated and ignored.

Repo evidence:
- `moe_infinity/utils/config.py`
- `moe_infinity/runtime/model_offload.py`
- `moe_infinity/entrypoints/big_modeling.py`
- `moe_infinity/memory/memory_coordinator.py`
- `moe_infinity/distributed/expert_executor.py`
- `moe_infinity/memory/expert_prefetcher.py`
- `moe_infinity/runtime/attention_backend.py`
