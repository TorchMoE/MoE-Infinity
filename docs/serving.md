# OpenAI-compatible serving guide

Authoritative reference for the async OpenAI-compatible server.

`MoE.generate()` remains the current in-process synchronous path, but it emits
`DeprecationWarning` and is scheduled for removal. `MoE.serve()` is recommended
for continuous batching; it starts an async HTTP service and is not a drop-in
replacement for an in-process return value.

## Quick Start

> **Security:** The parser defaults to `--host 0.0.0.0`, and authentication is
> disabled when neither `--api-key` nor `MOE_API_KEYS` is configured. On an
> untrusted, shared, or cloud host, bind to `127.0.0.1` or configure an API key
> before exposure. Completion routes and privileged `/admin/stats`, `/v1/config`,
> and `/v1/reload` endpoints share this authentication posture.

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --host 127.0.0.1 \
    --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --offload-dir ./offload_dir \
    --device-memory-ratio 0.5 \
    --kv-cache-ratio 0.25 \
    --max-batch-size 8
```

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V2-Lite-Chat","prompt":"Hello","max_tokens":32}'
```

For DFlash speculative decoding, use a validated pair such as `openai/gpt-oss-20b` with `z-lab/gpt-oss-20b-DFlash`, or leave `--speculative-draft` unset for non-DFlash targets.

For one-host multi-GPU ownership and `CUDA_VISIBLE_DEVICES` ordering, see [Single-server multi-GPU](multi-gpu.md).

## CLI Reference

Stable options from `api_server_v2.py`:

| Option | Default | Meaning |
|---|---:|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Listen port |
| `--model` | required | Model checkpoint or local path |
| `--offload-dir` | required | Expert offload directory |
| `--speculative-draft` | none | DFlash drafter checkpoint |
| `--device-memory-ratio` | `0.75` | GPU memory reserved for expert cache |
| `--kv-cache-ratio` | `0.25` | Remaining memory reserved for KV cache |
| `--max-batch-size` | `32` | Max concurrent sequences per batch |
| `--api-key` | none | Comma-separated Bearer keys |
| `--rate-limit` | `0` | Requests/minute/key; `0` disables |
| `--max-waiting-requests` | `0` | Queue depth backpressure threshold; `0` disables |
| `--max-n` | `16` | Cap for parallel sampling `n` / `best_of` |
| `--enable-prefix-caching` | off | Enable correctness-preserving prefix KV reuse (Qwen3 + FlashInfer) |
| `--prefix-cache-max-entries` | 1000 | Max prefix-index entries (startup-only, >= 1) |
| `--enable-prefix-caching` | off | Enable prefix-cache bookkeeping flag |
| `--enable-decode-cuda-graphs` | off | Permit decode graph qualification; unsafe runtimes still run eagerly |
| `--decode-cuda-graph-batch-sizes` | `1 2 4 8 16 32` | Positive capture/replay batch buckets |
| `--decode-cuda-graph-context-sizes` | `128 256 512 1024 2048 4096` | Positive native-paged context buckets |
| `--decode-cuda-graph-warmup-iters` | `2` | Warmup iterations before each lazy capture |
| `--decode-cuda-graph-max-memory-bytes` | `0` | Graph private-pool byte limit; `0` is unlimited |
| `--startup-timeout` | none | Startup watchdog timeout, seconds |
| `--decode-step-timeout` | none | Decode watchdog timeout, seconds |
| `--enable-pyspy-dump` | off | Scaffolded flag; currently accepted and stored, but no py-spy dump is triggered |
| `--enable-contextpilot` | off | Enable ContextPilot middleware |
| `--contextpilot-debug` | off | Enable ContextPilot fault-injection/admin hooks |

Internal / deprecated:

- `MoE.generate(...)` emits `DeprecationWarning` and is scheduled for removal;
  `MoE.serve(...)` is the continuous-batching HTTP transition path, not a
  drop-in in-process call replacement.
- `--max-waiting-requests` and `--max-n` feed internal module state used by middleware.

## Python Startup

```python
from moe_infinity import MoE

model = MoE("deepseek-ai/DeepSeek-V2-Lite-Chat", {
    "offload_path": "./offload_dir/deepseek-v2-lite",
    "device_memory_ratio": 0.5,
})
model.serve(host="127.0.0.1", port=8000, offload_dir="./offload_dir")
```

Binding the Python startup path to `0.0.0.0` has the same exposure and auth
implications as the CLI. Configure `api_key` or `MOE_API_KEYS` before using a
non-loopback bind on an untrusted network.

`MoE.serve()` accepts the same serving knobs as the CLI for host, port, memory ratios, batch sizing, the prefix-cache flag, offload path, and DFlash drafter setup. With `--enable-prefix-caching` on a supported Qwen3 + FlashInfer runtime the flag activates request-path prefix KV reuse; otherwise the cold path runs unchanged.

## Decode CUDA graphs (experimental, opt-in)

Decode CUDA graphs are disabled by default. Enabling the flag permits the
runtime to evaluate graph safety; it does not override any safety rejection.
The first rollout is deliberately limited to CUDA-resident models using the
native paged-attention backend and exact ordinary-GQA
`moe_infinity.models.qwen3_paged_attention.Qwen3PagedAttention` layers.

An eligible batch must be pure single-token decode, fit a configured batch and
context bucket, and have an explicit `eligible` runtime capability. The
scheduler cache and native backend must share one `PagedKVStorage` object,
allocator, owner ID, block count, and per-layer tensor pointers. Storage,
`ModelRunner`, every graph buffer, and the stable output must resolve to the
same indexed CUDA device. Every Qwen3 layer must have a unique in-range integer
`layer_idx`; registration creates a distinct layer-bound subclass and requires
an allocation-free `paged_kv_write_` proof that writes the current token to the
authoritative `slot_mapping` before attention reads that layer's cache.

The following always execute eagerly:

- prefill, mixed unsupported execution, and speculative verification;
- sampling and request callbacks (sampling is never captured);
- non-paged models (`native_paged_required`);
- FlashInfer planning/run paths (`flashinfer_plan_path`);
- exact `DeepseekV2PagedAttention` and `DeepseekV3PagedAttention` MLA classes
  (`mla_layout_unsupported`);
- active model hooks, Archer begin/end callbacks, transfer schedulers, expert
  dispatch, expert/KV offload, or dynamic allocation paths;
- mismatched storage ownership/device, unknown paged classes, invalid
  `layer_idx`, or incomplete per-layer write proofs;
- batches outside configured buckets, capture failures, and quarantined keys.

Graph keys are `(batch_bucket, context_bucket)`. Real rows retain scheduler
order and current sequence metadata. Padding rows are compute-only and use
unique scratch pages reserved from the same authoritative allocator; they do
not enter scheduler accounting, sampling, callbacks, or token counts. Capture
is lazy and may increase the first eligible step's latency. Denser bucket sets
reduce padding but consume more graph private-pool memory and scratch KV
capacity.

Reload invalidates graph states before Python modules reload. Hot replacement
waits for the active engine step and closes the old graph runner before
returning. Application shutdown stops and awaits the engine task before closing
the current runner. The sole lock order is application
`_engine_lifecycle_lock` → engine step ownership → `CudaGraphRunner._lock`;
shutdown never awaits a task while holding these locks. Close is idempotent and
returns scratch blocks to the authoritative allocator.

### Monitoring and rollback

`/admin/stats` reports `capability_safe`, bounded `capability_reason`, storage
owner ID, registered/proved layer counts, captures, replays, failures, replay
fallbacks, graph private-pool bytes, and scratch KV bytes. Prometheus exports:

- `moe_cuda_graph_captures_total`
- `moe_cuda_graph_replays_total`
- `moe_cuda_graph_capture_failures_total`
- `moe_cuda_graph_instances`
- `moe_cuda_graph_pool_bytes`
- `moe_cuda_graph_scratch_kv_bytes`
- `moe_cuda_graph_fallback_total{reason="..."}` over the fixed reason set

Never use request IDs, model IDs, graph keys, exception text, owner IDs, or
class names as metric labels.

Emergency rollback is immediate: set `MOE_DISABLE_CUDA_GRAPHS=1`. The gate
reads it for every batch, so replays stop without process restart. Confirm that
the replay counter stops and the eager fallback counter increases. Call
`engine.invalidate_cuda_graphs("operator_rollback")` to discard captured states,
or restart without the enable flag when graph memory must be reclaimed
immediately. `shutdown()` is reserved for replacement or process shutdown and
also releases scratch reservations.

### Staged rollout

1. **Shadow qualification:** keep production disabled. Require exact
   ordinary-GQA Qwen3, `capability_reason=eligible`, complete
   `(class_fqn, layer_idx, writer)` proof coverage, allocator/backend identity,
   exact device equality, per-layer persistence, two-step CUDA equivalence,
   cleanup tests, and paired fixture/model evidence for each
   model/GPU/dtype/bucket set.
2. **Canary:** enable one resident native-paged replica. Alert on any capture
   failure, post-capture non-`eligible` capability, request error, owner
   mismatch, memory-budget breach, or replay coverage below the workload
   target.
3. **Limited rollout:** expand only while output parity stays clean, old-engine
   scratch returns after replacement/shutdown, and observed p50/p99 ITL,
   throughput, memory, and concurrency remain acceptable.
4. **General opt-in:** retain a kill switch and per-model allowlist. DeepSeek
   MLA, non-paged, FlashInfer, and offloaded MoE remain unsupported regardless
   of benchmark results.
5. **Rollback:** set `MOE_DISABLE_CUDA_GRAPHS=1`, verify counters, then restart
   without the enable flag if immediate memory reclamation is required.

This feature makes no speedup guarantee. Its utility boundary is resident,
native-paged ordinary-GQA Qwen3 with complete write proofs. Supporting
offloaded MoE requires a separate piecewise design with explicit eager
attention, routing, dispatch, and transfer boundaries.

## Request Fields and Streaming

### `/v1/completions`

Implemented request fields:

- `model`, `prompt`
- `max_tokens`, `temperature`, `top_p`
- `n`, `best_of`
- `stream`, `stop`, `logprobs`, `response_format`

Accepted but currently no-op / unsupported:

- `echo`, `suffix`, `user`

Accepted but currently no-op:

- `presence_penalty`, `frequency_penalty`, `logit_bias`

### `/v1/chat/completions`

Implemented request fields:

- `model`, `messages`
- `max_tokens`, `temperature`, `top_p`
- `n`, `stream`, `stop`
- `logprobs`, `top_logprobs`
- `response_format`

Accepted but currently no-op / unsupported:

- `user`

Accepted but currently no-op:

- `presence_penalty`, `frequency_penalty`, `logit_bias`

### Streaming

- Streaming uses SSE.
- The terminal event is `data: [DONE]`.
- `finish_reason` is `stop` on EOS or stop-sequence termination, `length` on `max_tokens`, and `error` for JSON-object validation failures.
- Parallel sampling is constrained to `n <= --max-n` and `best_of <= --max-n`.
- Streaming with parallel sampling requires `n=1` and `best_of=1`.

## Authentication and Rate Limiting

- API key source: `--api-key` first, then `MOE_API_KEYS` if no CLI keys are provided.
- Format: `Authorization: Bearer <key>`.
- If no keys are configured, auth is disabled.
- Exempt routes: `GET /health`, `GET /metrics`, `GET /v1/models`.
- Protected routes: completions, chat completions, `/admin/stats`, `/v1/config`, `/v1/reload`.
- `--rate-limit` is per-key requests/minute. `0` disables limiting.
- Rate-limit failures return `429`.

## Backpressure and Parallel Sampling

- `--max-waiting-requests` applies queue backpressure on the scheduler.
- When the queue reaches the threshold, the server returns `503` with a queue-full error.
- `--max-n` caps `n` and `best_of`.
- `n` must be positive; `best_of` must be positive and at least `n`.

## Prefix Caching

Prefix KV reuse is opt-in and default off. It reuses the physical KV of an exact
shared prompt prefix so a warm request only recomputes its divergent suffix,
with cold/warm equivalence.

- `--enable-prefix-caching` enables reuse; `--prefix-cache-max-entries` (default
  `1000`, minimum `1`, startup-only) bounds the prefix index.
- **Supported scope.** Reuse activates only on the Qwen3 paged-attention path
  with a complete per-layer registry, matching KV geometry, and real FlashInfer
  prefill/decode. Any other runtime keeps the existing cold path and reports a
  stable `prefix_cache_disabled_reason` (for example
  `prefix-aware-prefill-unavailable`, `incomplete-paged-layer-registry`, or
  `kv-store-binding-mismatch`).
- **Exact identity.** A hit requires exact compatibility namespace (model,
  tokenizer, adapter, dtype, KV geometry, attention backend/layout, position
  config, runtime epoch), exact parent-entry path, and exact token blocks.
  SHA-256 is only a bucket accelerator and never substitutes for token equality.
- **Layer completeness.** One physical block names K/V for every layer in a
  single validated layered store; export/import, checkpoint/restore, copy-on-
  write, publication, and eviction operate across all layers or fail closed.
- **Atomic pinned admission.** All sequences in a request group (including
  `n>1`) are pinned with leases before any eviction, then admitted together or
  not at all; failed admission restores tables, free count, refcounts, statuses,
  and open-lease count.
- **Committed-range publication.** Only fully committed, block-complete prompt
  ranges are published after a successful forward; failed or partial chunks and
  DFlash verify tails are never published.
- **Copy-on-write.** Indexed blocks are immutable; any write to a shared partial
  tail first copies every layer to a private block.
- **Reload and rollback.** `/v1/reload` invalidates the prefix cache once after a
  successful module reload. Rollback is removing `--enable-prefix-caching`,
  restarting, and confirming `moe_prefix_cache_active 0`; there is no persisted
  state to migrate.
- **DFlash exclusion.** Reused-prefix and non-cold requests use ordinary paged
  execution and are excluded from DFlash delegation.
- **Observability.** `/admin/stats` and zero-safe `moe_prefix_cache_*`
  Prometheus metrics expose enabled/active/disabled-reason, entries, open leases,
  hits, matched tokens, and invalidations.

Motivated by SGLang RadixAttention
(<https://lmsys.org/blog/2024-01-17-sglang/>) and vLLM automatic prefix caching
(<https://docs.vllm.ai/en/stable/examples/features/automatic_prefix_caching>);
no universal speedup is claimed. Prefix reuse does not change expert ownership;
the multi-GPU guide covers that layout separately.

## DFlash Serving

Use DFlash with a validated pair such as:

```bash
python -m moe_infinity.entrypoints.openai.api_server_v2 \
    --model openai/gpt-oss-20b \
    --offload-dir ./offload_dir \
    --speculative-draft z-lab/gpt-oss-20b-DFlash
```

Startup validates structural pairing (hidden size, vocabulary, mask-token
bounds, target layers, block constraints, and drafter shape) separately from
executor/route-ahead reachability.

The persistent path creates one canonical session per eligible sequence. It
preserves the request's temperature, top-k, top-p, budget, EOS set, and
request-scoped generator. It does not silently turn sampled requests into
greedy requests. Unsupported grammar/guided/logit-bias metadata, penalties,
logprobs, or stop strings use the standard serving fallback before drafting.
That fallback is not evidence of sampled serving.

Two cache execution contexts are observable in `/admin/stats`:

- `temporary_dynamic` is the Stage 4a compatibility mode. It keeps a temporary
  private DynamicCache while the engine owns scheduling, callbacks,
  cancellation, and request accounting. Sampled sessions and ineligible model
  layouts remain here; this is not sampled paged-MLA serving.
- `paged_mla` is the Stage 4b default-off target enabled by
  `enable_deepseek_mla_paging=True`. It is restricted to eligible greedy
  batch-1 DeepSeek V2/V3 MLA sessions. The engine owns packed latent/rope target
  pages; the draft cache is separate. Admission is bounded by
  `max_resident_paged_speculative_sessions` (default `1`) and must leave at
  least `min_free_mla_blocks_after_admission` free blocks (default `1`) after
  reserving the block-rounded peak for the full declared
  `prompt + max_tokens` budget plus up to `DFlash block_size - 1` transient
  verify tokens. Active sessions' committed and transient headroom that is not
  yet allocated is included. A rejected eligible request immediately uses
  `temporary_dynamic`; it does not wait for a paged seat, and its sampling
  parameters are unchanged. That dense Stage 4a fallback owns a private target
  cache and can therefore increase total GPU memory use even though it consumes
  no MLA pages.

Paged MLA is currently resident-only and has no preemption/swap implementation.
The scheduler does not preempt DRAFT/VERIFY sessions. Qwen and hybrid layouts
fall back to Stage 4a; hybrid paged rollback is not claimed. Cancellation after
an in-flight backend call releases session resources, and per-sequence page
ownership prevents one cancellation or rollback from truncating another row.
Completion/cancellation frees ownership, so a later request can be admitted.
`/admin/stats` reports active paged sessions, current free blocks, configured
limits, and counters for `admitted`, `session_cap`, `free_block_reserve`, and
`ineligible` decisions. `begin_failed` is recorded only when adapter/session
construction fails; `admitted` increments only after construction succeeds.

The guard is block-based admission control, not a general fairness proof. It
does not preempt or swap admitted sessions. External cache consumers can still
invalidate reserved headroom; such allocator failures clean up the affected
request and are currently re-raised by the engine step.

There is no real DeepSeek DFlash target/drafter pair validation in the repo.
Stage 4b's tiny/local DeepSeek adapter tests establish ownership and attention
metadata only. GPT-OSS has named valid pairs, but its resident expert path has
no executor route-ahead. Qwen evidence is tiny-fixture only.

Route-ahead is observer-only. Pairing evidence, executor reachability,
prefetch-fired evidence, and cache ownership are reported as separate facts.
See [DFlash unified execution](dflash.md) for direct batching, RNG caveats,
benchmarks, and exact CPU/GPU gates.

## Operational Endpoints

| Endpoint | Method | Auth | Body | Response / scope | Production note |
|---|---|---|---|---|---|
| `/v1/models` | GET | exempt | none | loaded model list | safe for readiness checks |
| `/health` | GET | exempt | none | `STARTING` / `HEALTHY` / `UNHEALTHY` | safe to expose |
| `/metrics` | GET | exempt | none | Prometheus text metrics | safe to scrape internally |
| `/admin/stats` | GET | protected | none | queue, batch, cache, engine stats | keep private |
| `/v1/config` | GET | protected | none | current runtime config | admin-only |
| `/v1/config` | POST | protected | JSON updates | only `max_batch_size` and `max_tokens_per_step` are mutable | admin-only |
| `/v1/reload` | POST | protected | reload request | hot-reload Python modules | admin-only |

The optional `/contextpilot/toggle`, `/contextpilot/inject-fault`, and
`/contextpilot/status` routes are intentionally cataloged in
[`docs/contextpilot/README.md`](contextpilot/README.md) rather than this core
serving table. The fault-injection route is debug-only and requires
`--contextpilot-debug`.

## Watchdogs and Diagnostics

- `--startup-timeout` is disabled by default.
- When enabled, startup timeout causes a hard exit.
- `--decode-step-timeout` is disabled by default.
- When enabled, decode timeout marks the server unhealthy instead of exiting.
- `--enable-pyspy-dump` is currently a scaffolded no-op flag: it is accepted and stored in watchdog config, but the watchdog flow does not trigger py-spy dumps yet.

If you need timeout debugging, use the watchdog logs and the troubleshooting guide rather than expecting a py-spy dump.

## Failure Modes

- `401`: invalid or missing API key.
- `429`: rate limit exceeded.
- `503`: request queue full.
- `400`: invalid `n` / `best_of` / other request validation.
- `finish_reason="error"`: JSON-object validation failure.
- Penalties and `logit_bias` are accepted by the request models but are not applied in sampling.
