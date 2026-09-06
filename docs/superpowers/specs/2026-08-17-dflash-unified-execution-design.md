# DFlash Unified Execution Design

**Date:** 2026-08-17
**Status:** Approved design, updated with delivered scope on 2026-08-18
**Dependency order:** Task 8 -> Task 8.5 -> Task 9 -> Tasks 10-13

## Decision

DFlash uses one request-oriented semantic core for draft, verify, acceptance,
sampling, stop handling, rollback, and traces. Model execution is selected by
explicit capabilities. Direct generation and the deprecated sync facade use
`SessionDriver`; serving preserves the same `SpecSession` transitions through a
lifecycle-owning driver. `_generate_batched` is an adapter rather than a second
semantic loop.

## Semantic contracts

Each request owns its prompt, budget, stop set, anchor, proposal rows, target and
draft cache handles, request RNG, emitted tokens, and trace. Sampled anchor,
draft, acceptance, and correction draws consume the request stream. No sampled
decision may use another row's generator or silently become greedy.

The cache contract after commit is:

```text
target cache = prompt + committed non-bonus prefix
anchor       = emitted bonus, not duplicated in target cache
```

Rejected verify-tail cache is truncated before another round. Dense batches may
re-feed already chosen tokens only to reconstruct a shared physical rectangle;
proposal rows and RNG state remain authoritative.

## Backends and capabilities

### Bare Hugging Face

The physical bare-HF backend supports batch 1 and batch > 1, greedy, sampled,
and mixed rows. Prompts are left padded, outputs are right padded, and
`last_generated_lengths` carries true generated lengths. Scalar generators are
cloned per row and can correlate identical requests; explicit per-row generators
are recommended. Cross-batch-shape bit-exactness is not promised.

### Rich MoE

A per-request rich backend is the safe baseline. Physical rich batching requires
row-aligned input metadata, logits, hidden states, cache handles, masks,
positions, route contexts, and executor row unions. Wrappers declare this
capability explicitly. Grouped scheduling is not physical batching. MLA and
hybrid/Qwen wrappers retain per-request fallback unless their exact cache
contract is declared.

### Serving Stage 4a

`temporary_dynamic` is a compatibility context. It uses private dense target
state while `ContinuousBatchingEngine` retains scheduler, callback,
cancellation, and accounting ownership. It is temporary architecture, not
paged-cache ownership.

### Task 8.5: DeepSeek MLA prerequisite

The original ordering placed Task 9 before the model could write an engine-owned
DeepSeek MLA representation. Task 8.5 was inserted as the **DeepSeek MLA
prerequisite** and restored/adapted the historical paging foundation before
Stage 4b. It supplies packed `[kv_c_normed | k_pe]` storage, attention metadata,
DeepSeek V2/V3 eligibility, model adaptation, and the engine-owned target-cache
handle. The required order is **Task 8.5 -> Task 9**.

### Serving Stage 4b

Stage 4b is default-off. `enable_deepseek_mla_paging=True` may select
`paged_mla` only for eligible batch-1 greedy DeepSeek V2/V3 MLA requests. Each
eligible request has exactly one engine-owned target paged store, either the
standard `PagedKVCache` or packed-MLA `MLAPagedKVCache`; the drafter cache
remains separate. All in-flight DRAFT/VERIFY speculative sessions are resident
and non-preemptible. Stage 4a temporarily double-allocates target state, and
resident-only execution trades GPU memory for progress. The implemented guard
defaults to one active paged session and one free MLA block after expected
full declared prompt-plus-output allocation plus up to one DFlash verify
block's transient tail (`block_size - 1` tokens), including active sessions'
unallocated peak headroom; cap/reserve rejection records a reason and immediately uses
Stage 4a rather than waiting. Released sessions open admission for later
requests. Dense Stage 4a fallback may increase total GPU memory. This is bounded
admission, not a general fairness proof against unrelated cache consumers.
There is no swap/resume claim. Sampled,
Qwen, hybrid, and other ineligible paths remain Stage 4a. DeepSeek MLA
currently uses the correct PyTorch fallback rather than FlashInfer acceleration.
This design does not claim a real DeepSeek DFlash checkpoint pair or hybrid
paged rollback.

## Evidence dimensions

Pairing and execution are orthogonal. `PairingEvidence` covers DFlash config,
dimensions, vocabulary, mask, layers, block constraints, module validation, and
named checkpoint scope. `ExecutorEvidence` covers executor reachability,
attempted/fired route layers, actual expert unions, bytes, coverage, and
fallback. GPT-OSS named pairs can be valid while route-ahead is unreachable;
executor-wired DeepSeek/Qwen/Mixtral models can lack a validated pair.

Route-ahead is observer-only. Missing wiring or prefetch failure cannot modify
routing, acceptance, outputs, or cache state.

## Serving lifecycle

The engine owns admission, scheduling, verify demand, callbacks, cancellation,
and completion. Cancellation waits for an in-flight backend call and then
releases the session. Standard non-speculative sequences retain scheduler
preemption. In-flight DRAFT/VERIFY sessions are resident and non-preemptible.
Per-sequence ownership prevents cross-request truncation, but no speculative
swap/resume behavior is claimed. Admission never cancels or preempts an
existing session.

## Observability

The shared logical trace includes request ID, backend, cache kind, sampled mode,
rounds, accepted, committed, emitted, rollback, replay, finish reason, pairing
evidence, and executor evidence. Rich metrics separate per-request from physical
calls. Benchmarks must label fixture scope and report unavailable capabilities
as unavailable, never as zero-valued successes.
Serving stats additionally report the paged-MLA policy, active/free counts, and
decision counters.

## Delivered compatibility boundary

| Surface | Delivered behavior |
| --- | --- |
| Direct bare HF | Greedy/sampled/mixed, batch 1 and batch > 1 |
| Direct rich | Batch 1; grouped per-request fallback; physical tiny-fixture batching behind capability |
| Deprecated `MoE.generate` | Warning-preserving compatibility facade; no general sampled batch > 1 widening |
| Serving Stage 4a | Persistent canonical sessions with temporary dynamic target state |
| Serving Stage 4b | Default-off eligible greedy batch-1 DeepSeek V2/V3 MLA only |
| GPT-OSS | Named pair evidence; no executor route-ahead |
| DeepSeek | MLA paging/ownership evidence; no real DFlash pair claim |
| Qwen | Tiny/hybrid evidence only; no real pair or hybrid paged rollback |

## Non-goals

This design does not claim sampled paged serving, a real DeepSeek pair,
GPT-OSS paged MLA or executor route-ahead, hybrid paged rollback, universal rich
batching, GPU success from skipped tests, or release status from an unreleased
changelog entry.
