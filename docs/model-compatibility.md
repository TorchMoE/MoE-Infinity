# Model compatibility matrix

This page separates general model support from DFlash evidence. `validated`
means the named scope has a repository harness; `implemented/experimental`
means code plus tiny/unit evidence; `not recorded` means no direct evidence.
Pairing validity never implies executor reachability, and a rich or paged claim
is valid only behind its corresponding capability declaration.

Loading some published drafter checkpoints requires
`trust_remote_code=True`. This executes arbitrary code supplied by the remote
repository. Only load a trusted, pinned drafter revision, and review that
revision before use. This warning applies to pairing examples below; it is not
a claim that any particular DeepSeek DFlash pair has been validated.

## General model support

| Family / HF class | General sync/offload status | Continuous serving | Notes |
| --- | --- | --- | --- |
| DeepSeek-V2 (`DeepseekV2ForCausalLM`) | validated | implemented/experimental | Eager consistency harness; Stage 4b details below. |
| DeepSeek-V3 (`DeepseekV3ForCausalLM`) | implemented/experimental | implemented/experimental | Routing and paged-attention parity evidence. |
| DeepSeek-V4 (`DeepseekV4ForCausalLM`) | validated official mp4 path | not validated | DFlash unsupported; mp1 not covered. |
| Mixtral (`MixtralForCausalLM`) | implemented/experimental | implemented/experimental | No real-model serving harness recorded. |
| Qwen3 / Qwen3.5 MoE | Qwen3 validated; Qwen3.5 tiny-fixture validated | implemented/experimental | Qwen3.5 is text-only and requires newer Transformers. |
| GLM-5.2 (`GlmMoeDsaForCausalLM`) | validated | tiny serving harness | Built-in MTP, not DFlash. |
| GPT-OSS (`GptOssForCausalLM`) | 20B validated | 20B validated | Resident expert implementation. |
| DBRX / Jamba / OLMoE / NLLB-MoE | implemented/experimental | not validated | Registry/adapter evidence only. |
| OPT (`OPTForCausalLM`) | unsupported | unsupported | Registry entry only. |

## DFlash capability and evidence matrix

| Family | DFlash pairing evidence | Direct execution | Rich execution capability | Executor / route-ahead evidence | Serving cache capability | Validation boundary |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-OSS-20B | valid published pairs: `openai/gpt-oss-20b` / `z-lab/gpt-oss-20b-DFlash` | Greedy real-pair GPU evidence; sampled direct implementation with tiny statistical evidence | Resident wrapper; no paged MLA claim | **no executor route-ahead** because the resident path does not attach `expert_executor` | Stage 4a compatibility path; no GPT-OSS paged MLA | GPU fixture required for real-pair claims. |
| GPT-OSS-120B | valid published pairs: `openai/gpt-oss-120b` / `z-lab/gpt-oss-120b-DFlash` | Greedy GPU-gated harness; sampled direct implementation not real-pair validated | Resident wrapper | **no executor route-ahead** | Stage 4a compatibility path | No board/TP claim is inferred. |
| DeepSeek V2/V3 | **No real DeepSeek DFlash pair** is recorded | Session semantics and local/tiny adapter evidence only | MLA rows use grouped per-request fallback; physical rich batching needs the row-aware capability guard | Executor seam is reachable on offloaded models; this does not validate pairing | `paged_mla` is default-off, eligible only for batch-1 greedy DeepSeek V2/V3; exactly one engine-owned target paged store per request, using `MLAPagedKVCache` for packed MLA; drafter cache separate; resident-only and no swap/preemption | DeepSeek MLA uses the correct PyTorch fallback, not FlashInfer acceleration. Stage 4b proves ownership, not a checkpoint pair. |
| Qwen3.5-MoE | No published pair recorded | Qwen tiny-only evidence; greedy hybrid rollback fixtures | **Qwen/hybrid fallback** is grouped per request, not physically batched | Executor seam has synthetic/tiny evidence | Stage 4a `temporary_dynamic`; no hybrid paged rollback | No real checkpoint/drafter or sampled serving claim. |
| Other executor-backed MoE | Not recorded | Not recorded | Physical rich batching only after the wrapper's row-aware capability guard | Wiring may exist | No DFlash paged claim | Executor evidence is not pairing evidence. |

### Capability gates

- **Rich execution capability:** `supports_batch` and `supports_rich_forward`
  must both be true, and the wrapper must preserve row-aligned logits, hidden
  states, cache handles, masks, positions, and route contexts. Otherwise
  scheduling may group rows but forwards remain per request.
- **Serving cache capability:** Stage 4b requires the default-off DeepSeek MLA
  flag, a compatible MLA module set, target-cache-adapter support, greedy mode,
  and batch 1. Every other path uses the explicit Stage 4a fallback.
- **Sampling:** direct bare-HF batch 1/batch > 1 supports greedy, sampled, and
  mixed rows with per-row RNG. This does not widen deprecated or paged-serving
  surfaces automatically.
- **Paged-store ownership:** Each eligible request has exactly one engine-owned
  target paged store, either standard `PagedKVCache` or packed-MLA
  `MLAPagedKVCache`. The drafter cache is separate. DRAFT/VERIFY speculative
  sessions are resident and non-preemptible. The implemented active-session cap
  and post-peak free-block reserve (declared budget plus transient verify
  headroom) route rejected admissions immediately to
  Stage 4a; they do not prove general fairness. No swap/resume claim is made.
- **Preemption:** ordinary serving sequences can swap/preempt. Paged MLA DFlash
  is resident-only; no swap/preemption while DRAFT/VERIFY is in flight.

## Evidence sources

- `tests/python/dflash/test_capability_orthogonality.py`
- `tests/python/dflash/test_bare_hf_backend.py`
- `tests/python/dflash/test_mixed_sampling_batch.py`
- `tests/python/dflash/test_rich_batch_forward.py`
- `tests/python/serving/test_dflash_stage4a.py`
- `tests/python/serving/test_dflash_stage4b.py`
- `tests/python/serving/test_rich_batch_runner.py`

Use `Not recorded` rather than extrapolating hardware, pairing, route-ahead,
sampling, rich batching, or paged-cache support from an adjacent capability.
