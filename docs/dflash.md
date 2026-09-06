# DFlash unified execution

> **Security warning:** DFlash drafter checkpoints and tokenizer/model code may
> require `trust_remote_code=True`. That setting permits arbitrary Python code
> from the remote repository to run during loading. Use it only with a trusted,
> pinned drafter revision, and review the revision before loading it. See the
> model compatibility matrix for the pairing evidence boundary.

DFlash now has **one semantic core** for draft, verify, acceptance, rollback,
sampling, stop handling, and traces. Direct generation and the deprecated sync
facade use `SessionDriver` with request-scoped sessions; serving uses the same
canonical `SpecSession` transitions through its lifecycle-owning
`SpecSessionDriver`. `_generate_batched` is only an argument/output adapter.

This is an implementation statement, not a claim that every model, cache, or
serving mode has the same capabilities. Backend declarations gate physical
batching, sampling, rich forwarding, route-ahead, and cache ownership.

## User surfaces

| Surface | Delivered behavior | Important limits |
| --- | --- | --- |
| `DFlashSpeculator.generate` | Bare-HF batch 1 and batch > 1 support greedy, sampled, and mixed greedy and sampled rows. One session is created per row. | Experimental API. A target/drafter pair is still required. |
| Direct rich target, batch 1 | Canonical per-request rich execution. | Model-specific caches still apply. |
| Direct rich target, batch > 1 | A wrapper declaring the complete row-aware capability can use a physically batched rich forward. Otherwise rows use grouped per-request sessions. | Grouped per-request execution is not physically batched execution. DeepSeek MLA and hybrid/Qwen wrappers currently fall back per request. |
| `MoE.generate(..., speculative_draft=...)` | Compatibility facade over the same session semantics. Greedy batch 1 and eligible greedy batch > 1 retain tensor compatibility. | MoE.generate() is deprecated and emits `DeprecationWarning`. Sampled batch > 1 is not widened by this facade; Qwen3.5 rejects non-greedy speculative use explicitly. |
| Continuous serving | Persistent per-sequence sessions, streaming commits, cancellation cleanup, and scheduler-controlled verify admission. | Stage 4a and Stage 4b cache modes have different eligibility; see below. Do not infer sampled paged serving from direct sampled support. |

## Direct API

```python
import torch
from moe_infinity.spec_decode import DFlashSpeculator

# target and drafter are already constructed; no checkpoint loading is shown.
spec = DFlashSpeculator.from_models(target, drafter, config=config, device="cpu")
output = spec.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=[16, 24],
    temperature=[0.0, 0.8],
    top_p=[1.0, 0.9],
    generator=[None, torch.Generator().manual_seed(7)],
)
```

### Batched inputs and outputs

- Bare-HF batches may combine greedy and sampled rows. Budgets, stop sets,
  temperatures, top-k, top-p, and generators are row-local.
- Ragged prompts must be left-padded with a 0/1 `attention_mask`; each row must
  end in a real token. Equal-length prompts may omit the mask.
- Results form a right-padded tensor. `last_generated_lengths` records the true
  generated length of every row so callers can ignore output padding.
- Direct dense caches use lockstep physical lengths where Hugging Face requires
  a rectangle. A row ahead of the shared cache may re-feed already selected
  tokens for **dense cache reconstruction**. Re-fed tokens are neither emitted
  twice nor sampled again, and retained proposal distributions stay intact.

### Request RNG contract

Every sampled session owns a per-row generator and retains the drafter proposal
distribution for every slot until verification. Row order or the presence of an
unrelated row therefore does not change an explicitly row-seeded request.

Passing one scalar generator to batch > 1 clones the same initial state for
each row. Identical requests can consequently be **correlated**. Pass one
generator per row for independent explicit streams. Results are not bit-exact
across batch shapes: the guarantee is row order/composition invariance for a
fixed request and row-local stream, not equality between all physical batch
layouts. In short, outputs are not bit-exact across batch shapes.

## Rich execution capability

`RichBatchMetadata` carries row offsets and lengths, masks, positions, cache
handles, request contexts, and route contexts. `RichForwardResult` carries
row-aligned logits, hidden states, and cache handles. Physical rich batching is
enabled only when the wrapper's row-aware capability guard declares the full
contract. Trace/benchmark evidence distinguishes `per_request_rich_calls` from
`physical_rich_calls`.

Models with MLA or hybrid cache layouts currently keep the conservative
grouped per-request fallback unless their exact cache contract is supported.
This includes the current Qwen/hybrid fallback. Qwen evidence is tiny-fixture
only; no real Qwen target/drafter validation is claimed.

## Serving cache modes

### Stage 4a: `temporary_dynamic`

Stage 4a is the compatibility mode. Each session temporarily owns a private
target/draft temporary DynamicCache while `ContinuousBatchingEngine` continues to
own scheduling and lifecycle. Ineligible Stage 4b requests remain here. A
sampled serving request using this fallback is not evidence of sampled serving
through paged MLA, and no sampled paged-serving claim is made.

### Stage 4b: `paged_mla`

Task 8.5 restored the DeepSeek MLA prerequisite before Task 9. Stage 4b is a
default-off target selected by `enable_deepseek_mla_paging=True`, and only for
eligible greedy batch-1 DeepSeek V2/V3 MLA sessions. Each eligible request has
exactly one engine-owned target paged store: standard `PagedKVCache` for
standard attention, or packed `MLAPagedKVCache` for DeepSeek MLA.
`PagedCacheAdapter` supplies per-sequence append, snapshot, truncate, attention
metadata, and release. The **draft cache remains separate** and never owns the
target allocation.

The resident admission guard is implemented before `paged_mla` selection. It
caps active paged sessions with
`max_resident_paged_speculative_sessions=1` by default and requires prompt
plus declared output capacity, plus up to `DFlash block_size - 1` transient
verify tokens, to leave
`min_free_mla_blocks_after_admission=1` free MLA block by default. Existing
paged sessions' unallocated committed and transient headroom is included. All
demand is block-rounded using the MLA page size. Cap or reserve rejection records a structured reason/counter and
immediately starts the same request in Stage 4a `temporary_dynamic`; there is
no scheduler wait loop and no sampling downgrade. Releasing an admitted
session removes it from the active count, allowing a later request to qualify.
The dense Stage 4a target cache can increase total GPU memory use despite not
using MLA pages. Failed adapter/session construction is counted as
`begin_failed`, never `admitted`.

DeepSeek MLA currently has a resident-only, non-preemptible policy:
`MLAPagedKVCache` has no swap/resume implementation. All DRAFT/VERIFY
speculative sessions, including Stage 4a sessions using a temporary
`DynamicCache`, remain resident and are not preempted while in flight. This
uses extra GPU memory and can increase wait time for ordinary requests. The
implemented cap/reserve guard bounds new paged admission and avoids starvation
by silent waiting; it is not a general fairness proof. It reserves the declared
request budget but cannot prevent unrelated cache consumers from exhausting
pages; the affected request is cleaned up and the current engine step re-raises
that allocator failure. Stage 4a also
temporarily double-allocates target state while its private cache is live. There
is no claim that speculative sessions can be swapped out and resumed, and
preemption is not implemented for these in-flight sessions.
`swap_out()` and
`swap_in()` intentionally return false, and the scheduler does not preempt
in-flight DRAFT/VERIFY sessions. This is not hybrid paged rollback, and Qwen or
other hybrid models remain Stage 4a fallbacks. Cancellation and completion
release the sequence's pages without touching another sequence.

DeepSeek MLA does not currently use FlashInfer acceleration; it uses the
correct PyTorch fallback. Installing FlashInfer does not change that scope.

There is no real DeepSeek DFlash pair validated in this repository. Stage 4b
tests prove cache ownership and DeepSeek MLA adapter behavior with tiny/local
models; they do not establish a production target/drafter checkpoint pair.

## Pairing and executor evidence

Pairing and execution are independent dimensions:

- `pairing_evidence` describes config, dimensions, vocabulary, mask token,
  target layers, block constraints, module checks, and any named checkpoint
  scope.
- `executor_evidence` describes executor reachability, attempted/fired
  route-ahead layers, actual expert unions, bytes, coverage, and fallback.

GPT-OSS-20B and GPT-OSS-120B have valid published target/drafter pairs in the
repo's evidence, but the resident GPT-OSS expert path has no executor
route-ahead route. Conversely, executor wiring on DeepSeek/Qwen/Mixtral does not
create a valid DFlash pair. Route-ahead remains observer-only and cannot change
tokens, acceptance, or cache state.

## Trace contract

The direct and serving paths share the logical `SessionTrace` schema:
`request_id`, `backend`, `cache_kind`, sampled mode, `round_count`, `accepted`,
`committed`, `emitted`, `rollback`, `replay`, finish reason, route status,
`pairing_evidence`, and `executor_evidence`. These fields are the stable rollout
evidence schema; low-level cache objects remain internal.

## Validation and benchmarks

No-download CPU gates:

```bash
pytest -q tests/python/dflash/test_compatibility_matrix.py
python benchmarks/dflash/unified_execution_benchmark.py --fixture tiny
python benchmarks/dflash/validate_unified_execution.py --fixture tiny \
  --require-cache-invariants --require-order-invariance
pytest -q tests/python/dflash
pytest -q tests/python/serving
```

The tiny benchmark labels itself as a synthetic no-checkpoint CPU fixture. It
reports measured prefill/verify/decode work, actual sample/round/committed-token
counts, observed rollback/replay events, RNG/order invariance,
`sampled_tvd_value` (dimensionless), `sampled_kl_value` (nats), cache pages,
execution mode, separate pairing/executor evidence, and rich-call counts. It
does not project those timings onto a checkpoint.

GPU gates are opt-in and may download nothing unless checkpoints are already
cached:

```bash
MOE_DFLASH_GPU=1 CUDA_VISIBLE_DEVICES=0 \
  pytest -q tests/python/dflash/test_gpu_20b_dflash.py \
           tests/python/dflash/test_gpu_serving_dflash.py -m gpu
MOE_DFLASH_GPU=1 CUDA_VISIBLE_DEVICES=0 \
  pytest -q tests/python/dflash/test_gpu_120b.py -m gpu
python benchmarks/dflash/validate_unified_execution.py --fixture tiny \
  --require-cache-invariants --require-order-invariance --require-gpu
```

`--require-gpu is a readiness gate`: it checks only that CUDA is available and
the fixture environment is enabled. It does not execute the GPU harness. The
actual GPU pytest command remains separate and required. A skipped or
unavailable required fixture is not success; do not mark the GPU rollout gate
complete unless that command actually ran and passed.

## Evidence boundaries

- Published GPT-OSS pairs: pairing/direct evidence; no executor route-ahead.
- DeepSeek V2/V3: executor and default-off MLA paging capability; no real
  DeepSeek DFlash pair claim.
- Qwen3.5: tiny-fixture direct/hybrid evidence only; no real-pair claim.
- Hybrid paged rollback and GPT-OSS paged MLA are not implemented claims.
