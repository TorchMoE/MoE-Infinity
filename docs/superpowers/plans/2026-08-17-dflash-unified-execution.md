# DFlash Unified Execution Implementation Plan

**Goal:** Deliver one DFlash semantic core with capability-selected direct,
deprecated-sync, rich, and serving execution.
**Status:** Tasks 1-12 implemented and reviewed; Task 13 records the actual
delivered behavior and rollout gates.
**Corrected dependency order:** Task 8 -> Task 8.5 -> Task 9 -> Tasks 10-13.

This checked-in plan is the canonical plan revised after implementation. It
records departures from the initial ordering without rewriting them as if they
had always been known.

## Non-negotiable guards

1. Sampling claims require retained proposal rows and request RNG.
2. Grouped rich requests are not physical rich model batching.
3. Pairing, rich execution, executor reachability, and prefetch firing are
   separate evidence dimensions.
4. Stage 4b cannot be selected without a model-owned representation that writes
   exactly one engine-owned target paged store per eligible request. The store
   is standard `PagedKVCache` or packed-MLA `MLAPagedKVCache`; the drafter cache
   is separate.
5. Missing required GPU fixtures are not successful rollout gates.

## Task record

### Task 1: Shared protocol and trace schema — delivered

Added request/sampling/result/capability/cache contracts plus `SessionTrace`,
`PairingEvidence`, and `ExecutorEvidence`.

### Task 2: Request-scoped RNG — delivered

Threaded generators through anchor, drafter, acceptance, and correction draws;
retained every sampled proposal row through verification and reconstruction.

### Task 3: Single direct session — delivered

Moved batch-1 direct semantics onto canonical sessions while retaining output,
cache, stop, and trace behavior.

### Task 4: Session driver and cohorts — delivered

Added capability-first backend selection, safe cohort splitting, fail-before-
output behavior, progress checks, cleanup, and atomic physical-cohort results.

### Task 5: Bare-HF greedy physical batching — delivered

Moved left-padding, position IDs, row commit, dense rollback/reconstruction,
right-padding, and `last_generated_lengths` into the bare-HF backend.

### Task 6: Sampled and mixed bare-HF batching — delivered

Enabled batch-1/batch>1 sampled and mixed rows with per-row policies and RNG.
One scalar generator is cloned and can correlate rows; cross-batch-shape
bit-exactness is not guaranteed.

### Task 7: Public/deprecated facade migration — delivered with compatibility limits

Direct APIs share session semantics. `MoE.generate()` retains its
`DeprecationWarning`, return rectangle, and greedy compatibility behavior. It
does not claim general sampled batch > 1 support; Qwen3.5 keeps its explicit
non-greedy rejection.

### Task 8: Stage 4a serving — delivered

Added persistent per-sequence canonical sessions with `temporary_dynamic`
execution, verify demand, streaming commits, failure records, cancellation, and
logical invariant checks.

### Task 8.5: DeepSeek MLA prerequisite — inserted and delivered before Task 9

**Why reordered:** Task 9's page adapter could not establish final target-cache
ownership while the DeepSeek model still produced ordinary dense KV. The
DeepSeek MLA prerequisite restored/adapted the paging foundation from history:

- default-off DeepSeek V2/V3 eligibility;
- engine-owned packed latent/rope `MLAPagedKVCache`;
- per-layer attention adaptation and ownership validation;
- rich-forward attention metadata and position handling;
- tiny/local parity and ownership tests.

The corrected dependency is **Task 8.5 -> Task 9**. Task 9 was blocked until
this foundation existed.

### Task 9: Stage 4b paged ownership — delivered in constrained scope

Added `PagedCacheAdapter` and `paged_mla` execution for default-off eligible
batch-1 greedy DeepSeek V2/V3. Target pages are engine-owned; the draft cache is
separate. Cancellation and completion release only the owning sequence.

The actual delivered behavior is narrower than the initial final-state wording:

- all DRAFT/VERIFY speculative sessions are resident and non-preemptible;
- Stage 4a temporarily double-allocates target state;
- resident-only execution trades GPU memory for progress; the implemented
  default-one active-session cap and block-rounded declared request-budget plus
  transient verify-peak reserve reject to Stage 4a immediately with observable
  reasons/counters; dense fallback may increase total GPU memory;
- release permits later admission, but the guard is not a general fairness
  proof against unrelated cache consumers;
- there is no speculative swap/resume claim;
- sampled and ineligible paths retain Stage 4a;
- Qwen/hybrid paths retain Stage 4a;
- there is no hybrid paged rollback claim;
- DeepSeek MLA uses the correct PyTorch fallback, not FlashInfer acceleration;
- there is no real DeepSeek DFlash target/drafter pair claim.

### Task 10: Pairing/executor orthogonality — delivered

Separated pairing evidence from executor evidence and made route-ahead failure
observer-only. GPT-OSS keeps valid named pairs but no executor route-ahead.

### Task 11: Independent loop retirement — delivered

`_generate_batched` now normalizes compatibility arguments and calls the
physical session driver/backend; it no longer owns a second acceptance loop.

### Task 12: Rich row awareness — delivered with capability gate

Added row metadata/results and physical rich execution for wrappers explicitly
declaring the full contract. Tiny standard-cache fixtures validate physical
batching. MLA and hybrid/Qwen wrappers fall back to grouped per-request
execution. Qwen evidence remains tiny-only.

### Task 13: Documentation, matrix, benchmark, and rollout gates — delivered here

Files:

- `docs/dflash.md`, `docs/model-compatibility.md`, `docs/serving.md`
- `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`
- `tests/python/dflash/test_compatibility_matrix.py`
- `benchmarks/dflash/unified_execution_benchmark.py`
- `benchmarks/dflash/validate_unified_execution.py`

The compatibility assertions gate rich and paged claims on explicit capability
language and keep pairing/executor columns separate. The tiny benchmark performs
real CPU protocol timing and sampled-law measurements; it labels its synthetic
scope and does not fabricate checkpoint metrics. The validator fails closed on
cache invariant, ownership, order, sampling, or required-GPU failures.

## Validation commands

No-download CPU gates:

```bash
pytest -q tests/python/dflash/test_compatibility_matrix.py
python benchmarks/dflash/unified_execution_benchmark.py --fixture tiny
python benchmarks/dflash/validate_unified_execution.py --fixture tiny \
  --require-cache-invariants --require-order-invariance
pytest -q tests/python/dflash
pytest -q tests/python/serving
```

GPU gates:

```bash
MOE_DFLASH_GPU=1 CUDA_VISIBLE_DEVICES=0 \
  pytest -q tests/python/dflash/test_gpu_20b_dflash.py \
           tests/python/dflash/test_gpu_serving_dflash.py -m gpu
MOE_DFLASH_GPU=1 CUDA_VISIBLE_DEVICES=0 \
  pytest -q tests/python/dflash/test_gpu_120b.py -m gpu
python benchmarks/dflash/validate_unified_execution.py --fixture tiny \
  --require-cache-invariants --require-order-invariance --require-gpu
```

`--require-gpu is a readiness gate`: the validator returns failure unless CUDA
and `MOE_DFLASH_GPU=1` are both present, but it does not execute the GPU harness.
The actual GPU pytest command remains separate and required. A skipped
checkpoint test is reported as unavailable, not passed.

## Completion boundary

Task 13 completes truthful documentation and local rollout validation. It does
not remove Stage 4a, enable Stage 4b by default, establish sampled paged serving,
claim a DeepSeek/Qwen real pair, claim GPT-OSS route-ahead, claim hybrid paged
rollback, or convert unavailable GPU evidence into success.
