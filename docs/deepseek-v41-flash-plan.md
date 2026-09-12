# DeepSeek-V4.1-Flash support — phased implementation plan

Status: **draft / in progress**. This document tracks the incremental plan for
adding `deepseek-ai/DeepSeek-V4.1-Flash` (arch `DeepseekV41ForCausalLM`,
`model_type="deepseek_v41"`) to MoE-Infinity. Only **Phase 1** (registry +
config parsing + tests + docs) ships in the first PR; later phases are scoped
here but not yet implemented.

## Model facts

Source: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash> (config).

| Field | Value |
| --- | --- |
| `model_type` | `deepseek_v41` |
| `architectures` | `["DeepseekV41ForCausalLM"]` |
| `hidden_size` | 5120 |
| `vocab_size` | 129280 |
| `max_position_embeddings` | 1048576 (1M) |
| `torch_dtype` | bfloat16 |
| Quantization | fp8 dynamic (non-expert) + `expert_dtype="fp4"` (routed experts) |
| `weight_block_size` | `[32, 32]` |
| `num_hidden_layers` | 40 (CED: 20 encoder + 20 decoder) |
| `n_routed_experts` | 384 |
| `num_experts_per_tok` | 6 (top-6) |
| `moe_intermediate_size` | 2304 |
| `n_shared_experts` | 1 |
| `scoring_func` | `sqrtsoftplus` |
| `topk_method` | `noaux_tc` |
| `routed_scaling_factor` | 1.5 |
| Config nesting | MoE fields under `text_config` (VL wrapper, native vision tower `deepseek_v41_vision`) |

Architecture-defining extras (relevant to later phases):

- **CED** conditional encoder-decoder: 40 layers = 20 encoder + 20 decoder.
- **Engram** conditional memory: ~196B params, `engram_layer_ids=[1, 14]`,
  `engram_vocab_size=16000000` (16M).
- **CSA2** sparse attention: `sliding_window=128`,
  `kv_source_layer_ids=[2, 8, 14, 20]`, `index_topk=512`.
- **DSpark** multi-token prediction: `num_nextn_predict_layers=3`,
  `dspark_n_routed_experts=128`.
- **FP4 KV cache**: ~890 B/token.

## Transformers availability

- `transformers >= 5.6` *declares* the model, but **mainline transformers has
  NOT merged `deepseek_v41` as of 2026-09-12**. On a typical install
  `DeepseekV41ForCausalLM` is not importable, so the registry entry is skipped
  and loading a real checkpoint would go through the `trust_remote_code=True`
  path.
- All Phase-1 code is guarded exactly like the existing `DeepseekV4ForCausalLM`
  guard (class may be `None`; the registry never maps to a `None` class).

## Prior art

vLLM's implementation is the reference for later phases (vllm
`vllm/model_executor/models/deepseek_v4_1/` at commit `8065045`):

- `attention.py` — MLA + CSA2 sparse attention.
- `sparse_mla.py` — sparse MLA kernels.
- `compressor.py` — CED / conditional-memory compression path.
- `nvidia/vl_model.py` — native vision tower wiring.

## Phase 1 — registry + config parsing + tests + docs (this PR)

Touched files (registry/parsing/tests/docs only; **no** model wrapper or
monkey-patch, deferred to Phase 2):

- `moe_infinity/common/constants.py` — guarded import of
  `DeepseekV41ForCausalLM`; conditional `MODEL_MAPPING_NAMES["deepseekv41"]` and
  `MODEL_MAPPING_TYPES["deepseekv41"] = 5` (per-expert gate/up/down family, like
  V4/Qwen3/GLM).
- `moe_infinity/utils/hf_config.py` — `parse_moe_param` and `parse_expert_id`
  `deepseekv41` branches, ordered **before** `deepseekv4`/`deepseek`.
- `tests/python/unit/test_deepseekv41_registry.py` +
  `tests/fixtures/deepseek_v41_flash/config.json` — CPU-only unit tests.
- Docs: `docs/model-compatibility.md`, `README.md`, `CHANGELOG.md`, this file.

### Substring-dispatch precedence (critical)

The registry key `"deepseekv41"` contains `"deepseekv4"`, which itself contains
`"deepseek"`. Two dispatch sites must resolve the most specific key first:

1. `parse_expert_type` (constants.py) iterates keys sorted by length
   descending, so `deepseekv41` (11) is matched before `deepseekv4` (10). Both
   map to expert-type 5.
2. `parse_moe_param` / `parse_expert_id` (hf_config.py) use ordered `if/elif`
   substring checks. The `deepseekv41` branch is placed **before** the
   `deepseekv4` and generic `deepseek` branches. If that order regressed, a
   V4.1 config would fall through to the flat `deepseek` branch and read
   top-level `num_hidden_layers` / `n_routed_experts`, which V4.1 nests under
   `text_config` — raising `AttributeError`. A unit test pins this.

Caveat (documented and tested): when a build ships `DeepseekV4ForCausalLM` but
not `DeepseekV41ForCausalLM`, `parse_expert_type` on a V4.1 arch resolves to the
registered `deepseekv4` key (also type 5) rather than raising, because
`deepseekv4` is still a substring. Config parsing (`parse_moe_param` /
`parse_expert_id`) is registry-independent and still routes V4.1 via its own
arch-string branch — which is the Phase-1 correctness guarantee.

### Open question carried into Phase 2

- **Checkpoint key layout is UNCONFIRMED.** `parse_expert_id` currently reuses
  the V4 `layers\.(\d+)\.ffn\.experts\.(\d+)\.` pattern (unanchored, so it also
  matches a VL-style `language_model.layers.<L>.ffn.experts.<E>.` prefix). The
  exact V4.1 routed-expert tensor names — including any MTP/DSpark layer suffix,
  vision prefix, and FP4 packed/scale tensor naming — must be verified against
  real safetensors shards before Phase 2 offload work.

## Phase 2 — expert offload path (FP4)

Reuse the existing `moe_infinity/models/deepseek_v4/` FP4 infrastructure; the
quant format matches V4 (E2M1 + `ue8m0` block scale), so most of the host-store
+ streaming stack is reusable:

- Routed experts are FP4; 384 experts, top-6, `moe_intermediate_size=2304`.
- `weight_block_size=[32, 32]` (confirm the scale-block interpretation matches
  the V4 `FP4_SCALE_BLOCK` path; V4 uses block-32 `ue8m0` scales).
- Router: `scoring_func="sqrtsoftplus"`, `topk_method="noaux_tc"`,
  `routed_scaling_factor=1.5` — reuse `deepseek_v4/routing.py` (`sqrtsoftplus`,
  `topk_route`) where compatible.
- `n_shared_experts=1` stays resident (FP8), like V4.
- Deliverables mirror PR #201's family shape: a `SyncDeepSeekV41MoEBlock`
  wrapper (or reuse `SyncDeepSeekV4MoEBlock`), `models/__init__.py` export, and
  `runtime/model_offload.py` wiring — added only after the checkpoint key layout
  is confirmed.

Prerequisite: resolve the Phase-1 open question (real shard key layout) and add
a real-checkpoint (or tiny-fixture) loader test.

## Phase 3 — architecture gaps (NEW work, explicit unknowns)

Each item needs new design work and is flagged as research where the mapping to
existing MoE-Infinity infrastructure is not yet established:

- **CED encoder-decoder split** — 40 layers = 20 encoder + 20 decoder. Phase 1
  treats all 40 as decoder layers (`num_encoder_layers=0`). A real split needs
  `parse_moe_param` to emit encoder/decoder counts and the offload topology to
  respect the CED boundary. **Unknown:** whether routed experts differ between
  encoder and decoder stacks.
- **Engram conditional memory** — 196B, `engram_layer_ids=[1, 14]`,
  `engram_vocab_size=16M`. Very large; a **candidate for host/SSD offload**
  (not GPU-resident). **Research:** access pattern, whether it is sparse per
  token, and whether it fits the existing expert host-store abstraction or needs
  a new memory tier.
- **CSA2 sparse attention** — `sliding_window=128`,
  `kv_source_layer_ids=[2, 8, 14, 20]`, `index_topk=512`. Reference vLLM
  `attention.py` / `sparse_mla.py`. Runs resident; needs an
  `attn_implementation` selection compatible with the offload runtime.
- **DSpark 3-layer MTP** — `num_nextn_predict_layers=3`,
  `dspark_n_routed_experts=128`. Relate to the existing DFlash spec-decode path;
  MTP layers must be excluded from routed-expert offload (add an MTP layer-id
  guard in `parse_expert_id`, as GLM does).
- **FP4 KV cache** — ~890 B/token. Interaction with the existing paged KV cache
  and `kv_cache_format` options is undetermined.
- **Native vision tower** (`deepseek_v41_vision`) — served **text-only first**,
  exactly like Qwen3.5 and GLM-5.3-Flash; vision weights stay resident/unused
  for text generation.

## Non-goals for the first PR

- No download of the 552B model.
- No CED / Engram / CSA2 / DSpark / vision implementation.
- No changes to the existing `moe_infinity/models/deepseek_v4/` V4 code paths
  (V4.1 must not regress V4).
