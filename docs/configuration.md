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
