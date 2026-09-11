# Configuration and memory planning

Adaptive expert precision is disabled by default (`adaptive_expert_precision: false`). Its `adaptive_hbm_budget_bytes` unit is bytes; see [adaptive expert precision](adaptive-expert-precision.md).

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
| `kv_cache_format` | `str` | `native` | `native` / `int8_sym` | KV-cache storage format. Default: `native` preserves the model cache dtype. | `int8_sym` is an opt-in symmetric INT8 storage path; validated for ordinary MHA/GQA only. Validated by `KVCacheFormat.parse`. Opt-in. |
| `kv_cache_allow_fallback` | `bool` | `True` | `True` / `False` | Allow a visible native fallback when the requested KV format is unsupported. | When `False`, an unsupported `int8_sym` request raises before allocation instead of falling back. Opt-in. |

## KV-cache storage format (`kv_cache_format`)

`kv_cache_format` selects how paged K/V are stored (default: `native`), which
keeps the model cache dtype and is the only path enabled by default.

`int8_sym` is an opt-in, correctness-gated symmetric INT8 storage format. It
stores each K and V element as signed INT8 with one FP16 scale per
`(layer, page, KV head, token)`, calibrated online from the per-token/per-head
absolute maximum. It does **not** claim any universal 2-bit KV support;
[KIVI](https://arxiv.org/abs/2402.02750) and
[KVQuant](https://arxiv.org/abs/2401.18079) motivate the approach and the
quality gates only.

### Precision contract

The three precisions are reported separately so a fallback is never mistaken
for a quantized run:

| Concern | `native` | `int8_sym` |
| --- | --- | --- |
| Storage precision | model cache dtype (`fp16`/`bf16`/`fp32`) | INT8 payload + FP16 scales |
| Transfer precision | same native tensor bytes | synchronously copied INT8 payload and FP16 scales; no D2H/H2D dequantization |
| Attention execution precision | model dtype, FP32 accumulator | model-dtype output, scales promoted to FP32, FP32 QK/softmax/V accumulation |

### Memory

For the canonical `(block_size=16, num_kv_heads=8, head_dim=128)` page,
`int8_sym` uses `32,768` payload bytes + `512` scale bytes = `33,280` bytes,
versus `65,536` native FP16 bytes, a ratio of `0.5078125`.

### Scope, MLA, and fallback

`int8_sym` is validated for ordinary MHA/GQA. MLA models (DeepSeek/GLM latent
attention, detected from `kv_lora_rank`/`qk_nope_head_dim`/`qk_rope_head_dim`)
are **not** validated: with `kv_cache_allow_fallback=True` an MLA request falls
back to `native` with reason `mla_not_validated`; with fallback disabled the
server raises at startup before cache allocation. When a CUDA INT8 kernel
binding is unavailable the format still runs through a validated FP32
dequantized SDPA path; FlashInfer stays active for `native` stores but an
`int8_sym` request bypasses FlashInfer (reason
`flashinfer_no_int8_sym_contract`) without changing the effective storage
format.

### Phase-specific expert policy fields

All subordinate fields in this table are inert while
`phase_specific_expert_policy=False`. That default retains legacy admission,
top-2 prefetch priority, eviction, starvation, and mixed-batch behavior.

| Field | Type | Default | Valid values / range | Effect when enabled |
| --- | --- | --- | --- | --- |
| `phase_specific_expert_policy` | `bool` | `False` | `True` / `False` | Master gate for independently tunable prefill/decode policy over one shared expert store and GPU residency set. |
| `prefill_expert_admission` | `str` | `"transient_on_pressure"` | `cache`, `transient_on_pressure` | Determines whether a prefill miss remains resident or uses the transient overflow slot under pressure. |
| `decode_expert_admission` | `str` | `"cache"` | `cache`, `transient_on_pressure` | Determines whether a decode miss remains resident or becomes transient under pressure. |
| `prefill_expert_prefetch_top_k` | `int` | `0` | `0..num_experts` | Number of predictive prefill experts; zero disables speculative prefill traffic. |
| `decode_expert_prefetch_top_k` | `int` | `2` | `0..num_experts` | Number of predictive decode experts. |
| `prefill_expert_prefetch_priority` | `int` | `2` | `1..19` | Native prefill prefetch scheduling band. |
| `decode_expert_prefetch_priority` | `int` | `1` | `1..19` | Native decode prefetch scheduling band. |
| `prefill_expert_eviction_weight` | `float` | `1.0` | finite and `> 0` | Weight applied to prefill reuse in deterministic victim scoring. |
| `decode_expert_eviction_weight` | `float` | `4.0` | finite and `> 0` | Weight applied to decode reuse in deterministic victim scoring. |
| `expert_policy_starvation_limit` | `int` | `8` | `> 0` | Maximum prefetch bypasses before one promotion opportunity; demand band 0 remains strict. |

Minimal enablement:

```json
{
  "phase_specific_expert_policy": true,
  "prefill_expert_admission": "transient_on_pressure",
  "decode_expert_admission": "cache",
  "prefill_expert_prefetch_top_k": 0,
  "decode_expert_prefetch_top_k": 2
}
```

The phase is request metadata, not part of an expert key. Enabling this policy
does not create separate phase pools or duplicate weights. See
[OpenAI-compatible serving](serving.md#phase-specific-expert-policy) for runtime
ordering and rollback, and [Benchmarking](benchmarking.md#phase-specific-expert-policy-matrix)
for the required A/B matrix.

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
# Adaptive expert/KV memory

`adaptive_memory_enabled` is an opt-in controller and defaults to `false`.
The static `device_memory_ratio` and `kv_cache_memory_ratio` remain the fallback
contract. The bounded startup-only knobs are
`adaptive_memory_interval_steps`, `adaptive_memory_cooldown_steps`,
`adaptive_memory_ewma_alpha`, `adaptive_memory_hysteresis_ratio`,
`adaptive_memory_max_resize_step_bytes`,
`adaptive_memory_min_expert_cache_bytes`,
`adaptive_memory_min_kv_cache_blocks`,
`adaptive_memory_free_reserve_bytes`, and `adaptive_memory_failure_limit`.
Only the enable flag is hot reloadable; changing another policy knob requires a
restart.

Every target leaves the configured free-memory reserve untouched and obeys the
per-device expert/KV minima and maximum resize step. A device without a KV
backend reports a hold rather than borrowing another GPU's capacity.

Serving KV resize treats `_kv_cache`, `_fi_prefill`, and `_fi_decode` as one
transactional bundle. The independent prefill and decode wrappers are rebuilt
and freshly planned for the new page count. Old storage and wrappers remain
strongly referenced until a post-publication CUDA event completes; a failed
constructor or first plan restores the complete old bundle. The native path
likewise recreates both built-in KV stores, the FlashInfer store, and both
wrappers without changing dtype or layout.

## Asynchronous hierarchical KV swap

Serving accepts the same six values through `ArcherConfig`, `MoE.serve()`, and
the OpenAI server CLI:

| Field | Default | Validation and behavior |
| --- | ---: | --- |
| `kv_swap_mode` | `"sync"` | Exactly `"sync"` or `"async"`. Async requires CUDA and sufficient locked/pinned host memory. |
| `kv_swap_host_memory_bytes` | `536870912` (512 MiB) | Positive hard cap for async pinned host leases. Inactive in effective sync mode. |
| `kv_swap_max_inflight_bytes` | `268435456` (256 MiB) | Positive async in-flight DMA cap and no greater than the host cap. Inactive in effective sync mode. |
| `kv_swap_checksum` | `False` | Enables CRC32 validation of the full padded host payload before restore. |
| `kv_swap_max_retries` | `2` | Non-negative restore retry count. |
| `kv_swap_allow_sync_fallback` | `True` | Initialization may fall back to sync when CUDA or pinned allocation is unavailable. Corrupt bytes never fall back. |

Example:

```python
model = MoE(checkpoint, {
    "offload_path": offload_path,
    "kv_swap_mode": "async",
    "kv_swap_host_memory_bytes": 2 * 1024**3,
    "kv_swap_max_inflight_bytes": 1024**3,
    "kv_swap_checksum": False,
    "kv_swap_max_retries": 2,
    "kv_swap_allow_sync_fallback": True,
})
```

The default `kv_swap_mode="sync"` deliberately retains blocking pageable
`.detach().to("cpu").clone()` and blocking restore semantics. Each host copy is
owned directly by a `PageableCPUBufferRecord`; sync mode never constructs,
acquires, releases, or accounts a pinned pool. Consequently sync reports zero
pinned capacity and in-use bytes even while pageable host copies exist.

Async mode uses bounded `PinnedBufferLease` ownership only. Pool or in-flight
cap exhaustion returns backpressure before transfer state or block ownership
changes. To roll back, drain/restart the engine with `kv_swap_mode="sync"`;
never switch transfer backends while tickets are in flight.
