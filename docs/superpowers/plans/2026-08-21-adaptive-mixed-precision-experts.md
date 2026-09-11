# Adaptive Mixed-Precision Experts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, deterministic, fixed-HBM adaptive precision policy for general routed experts while preserving every validated GPT-OSS MXFP4, GLM FP8, DeepSeek-V4 FP4, GPTQ, and AWQ path exactly.

**Architecture:** Keep checkpoint/storage format, cached representation format, and execution dtype as three explicit fields. Converter/build candidates are separate from a fingerprint-qualified released allowlist; serving accepts only a manifest whose checkpoint fingerprint, format, converter version, and quality attestation exactly match a released entry. Task 0 creates the sole `core/prefetch/ExpertResidencyManager` authority and its fixed-size transaction/capacity/lease contract when that exact implementation is absent, then Task 5 extends the same files with `(format, generation)` keys, variant-aware transactions, reservations, workspace, leases, and retirement; `TaskScheduler` and `ExpertDispatcher` remain clients of that one manager. Crash-safe derivative generations are journaled above canonical tensor/file IDs and become visible through one atomic `CURRENT` publication; protected model-specific paths bypass all of this.

**Tech Stack:** Python 3.10+, PyTorch, pytest, GoogleTest, safetensors, C++17, CUDA events/streams, pybind11, shared `ExpertResidencyManager`, existing Archer tensor store/topology, existing FP8/MXFP4/Marlin kernels.

---

## Scope, evidence, and non-goals

### Existing seams this plan preserves

- `moe_infinity/runtime/model_offload.py:897-1045` detects packed formats while building the store; `:1008-1024` deliberately keeps GLM routed experts in FP8 and dequantizes non-routed weights; `:1091-1233` creates the native dispatcher and delivers GLM scales.
- `moe_infinity/runtime/model_offload.py:1369-1505` keeps the high-residency GPT-OSS Python MXFP4 path resident; `:1506-1524` reconstructs GLM scales because they are not currently persisted in Archer; `:1658-1759` registers canonical expert tensor groups.
- At base `b766f8f`, `core/parallel/expert_dispatcher.cpp:190-266` owns dispatch state, `:341-359` chooses LFU victims, and `:361-586` transfers canonical nodes under `cache_sizes_`; Task 0 introduces the manager-client path before adaptive work starts. `:588-670` contains the protected GLM FP8 and GPT-OSS MXFP4 execution seams.
- `core/model/model_topology.h:40-82` and `core/model/model_topology.cpp:59-241` own node bytes, devices, pointers, and asynchronous H2D events. `model_topology.cpp:525-729` computes alignment-aware node bytes and preloads sparse host memory.
- `core/aio/archer_tensor_handle.cpp:57-100` persists dtype, shape, size, partition, and offset in `TensorStorageMeta`; adaptive derivatives therefore need a separate manifest rather than overloading tensor dtype.
- `moe_infinity/utils/fp8.py`, `moe_infinity/kernel/mxfp4_gemm.py`, `moe_infinity/kernel/marlin_gemm.py`, and `extensions/kernel/v4_fp4/v4_fp4_binding.cpp` already define validated format-specific math. They remain the only execution adapters for those formats.
- `moe_infinity/models/gpt_oss.py` keeps MXFP4 blocks/scales output-major and uses `fused_mxfp4_gemm`; `moe_infinity/models/glm_moe_dsa.py` routes GLM experts through the native executor while shared experts remain Python-resident.
- `moe_infinity/models/deepseek_v4/official_offload_adapter.py` has its own `OfficialExpertHostStore`, async copy stream, FP4 byte-preserving transfer, native/tilelang selection, and TP all-reduce. This path is protected, not generalized or replaced.

### Motivation boundary

- [SliceMoE](https://arxiv.org/abs/2512.12990) motivates treating precision granularity as cache capacity under a miss-rate constraint.
- [DynaExq](https://arxiv.org/abs/2511.15015) motivates budget-constrained online precision allocation, heavy-tailed/hot-set-changing expert use, and stable asynchronous version publication.
- These papers are motivation only. This implementation does not claim their algorithms, reported accuracy, energy, latency, or throughput, and does not import their hardware assumptions.

### Explicit non-goals

- Do not replace `fused_mxfp4_gemm`, native V4 FP4, GLM block-FP8 dequantization, Marlin, GPTQ, or AWQ kernels.
- Do not infer a new format from tensor dtype alone, rewrite checkpoint tensors, or quantize unless candidate-build mode is explicit, the model is a converter candidate, and the exact converter is registered. Serving additionally requires an exact released four-field entry.
- Do not enable adaptive mode by default or retrofit legacy offload stores in place.
- Do not promise speedup. Benchmarks record transfer, memory, latency, and quality; release gates decide whether an allowlist entry may be enabled.

### Required implementation base and ordering

This Markdown plan remains independently reviewable on `plan/adaptive-mixed-precision-experts`, and its implementation is self-contained. PR #179 and `2026-08-21-phase-specific-expert-policy.md` are design-coordination inputs only; a plan-only commit is not implementation evidence and is never a rebase prerequisite. Task 0 runs first and establishes the exact fixed-size shared-manager API, build registration, singleton/client wiring, and real-manager tests required by every adaptive task.

If PR #179 (or another predecessor) lands an implementation before execution begins, Task 0 validates the landed files against the exact API and tests in this plan and skips only duplicate file creation and duplicate registration. Any API mismatch is reconciled in the existing `core/prefetch/expert_residency.{h,cpp}` and tests within Task 0; it is not solved by adding an adapter manager or second ledger. If no matching implementation exists, Task 0 creates it. In both paths, completion means the tree contains `core/prefetch/expert_residency.h`, `core/prefetch/expert_residency.cpp`, one shared `kExpertResidencyManager`, `ExpertResidencyClient` instances for demand dispatcher and prefetch scheduler, exactly-once build registration, and passing real-manager transaction/capacity/lease tests. Tasks 1-13 are blocked only by that verified Task 0 result.

Manager routing is enabled when `phase_specific_expert_policy || adaptive_expert_precision`. With both flags false, Task 0 preserves the exact legacy fallback. With adaptive enabled and phase policy disabled, scheduler/dispatcher still use `ExpertResidencyManager`, but phase-specific scoring, decode-first splitting, and phase telemetry remain disabled; canonical generation-0 wrappers preserve legacy admission semantics. If the separately coordinated phase feature is present and both flags are enabled, the same manager applies phase utility to variant-aware transactions.

## File structure

### New files

- `core/prefetch/expert_residency.h` and `core/prefetch/expert_residency.cpp` — Task 0's sole synchronized fixed-size residency authority; created only when an exact landed implementation is absent, then extended in Task 5.
- `core/prefetch/expert_policy.h` — coordinated phase/admission types and deterministic fixed-size utility helpers consumed by the shared manager; created only when absent.
- `tests/cpp/unit/prefetch/expert_residency_test_fixture.h` and `tests/cpp/unit/prefetch/test_expert_residency.cpp` — real-manager capacity, transaction, lease, and shared-client foundation tests.
- `tests/python/unit/test_native_phase_policy_wire.py` and `tests/python/unit/test_setup_sources.py` — one-manager client identity and exactly-once source registration regressions.
- `moe_infinity/runtime/expert_precision.py` — immutable format IDs, storage/execution descriptors, model protection/converter-candidate rules, hardware probes, and registry resolution.
- `moe_infinity/runtime/adaptive_precision_allowlist.py` — checked-in released entries keyed by checkpoint fingerprint, format, converter version, and quality-attestation digest; converter candidates do not imply serving approval.
- `moe_infinity/runtime/expert_variant_manifest.py` — versioned JSON schema, released-entry validation, generation/index digests, tensor-ID ownership, and checkpoint/store compatibility checks.
- `moe_infinity/runtime/expert_variant_build.py` — durable build journal, canonical high-water marks, generation-scoped derivative ID/file ranges, retry/recovery, and atomic `CURRENT` publication.
- `moe_infinity/memory/adaptive_precision_policy.py` — deterministic hotness accounting, hysteresis, byte-feasible selection, transition intents, and pure simulation.
- `tests/cpp/unit/prefetch/test_expert_residency_variants.cpp` — real-manager variant admission, transition reservation, workspace, lease, retirement, and byte-accounting tests composed with phase policy.
- `tests/python/integration/test_phase_adaptive_residency.py` — combined enabled/disabled phase-policy and adaptive-precision wiring against one native manager identity/snapshot.
- `tests/python/unit/test_expert_precision_registry.py` — registry, model protection, hardware probe, and storage/execution separation tests.
- `tests/python/unit/test_expert_variant_manifest.py` — schema, atomic reload, fingerprint, scale ownership, and corruption tests.
- `tests/python/unit/test_expert_variant_build.py` — journal state, crash/retry, canonical high-water, and atomic publication tests.
- `tests/python/unit/test_expert_variant_conversion.py` — deterministic FP8/Marlin conversion and capability tests.
- `tests/python/unit/test_adaptive_precision_policy.py` — deterministic policy, fixed-budget, hysteresis, and tie-breaking tests.
- `tests/python/unit/test_adaptive_precision_wiring.py` — opt-in, converter candidates, released entries, old-store, and exact fallback tests.
- `tests/python/ops/test_adaptive_expert_dispatch.py` — native representation lifecycle, lease/event safety, accounting, and failure rollback tests.
- `tests/python/integration/test_adaptive_precision_parity.py` — tiny checkpoint conversion, reload, canonical fallback, and generation parity/quality gates.
- `tests/python/unit/test_adaptive_precision_benchmark_schema.py` — deterministic workload and complete result/attestation schema tests.
- `tests/python/unit/test_adaptive_precision_docs.py` — executable documentation contract for configuration, manifests/attestations, benchmarks, rollout/rollback, and operational commands.
- `benchmarks/adaptive_precision/bench_policy.py` — trace-replay policy simulation.
- `benchmarks/adaptive_precision/bench_transfer.py` — H2D payload/latency and conversion-transition measurements.
- `benchmarks/adaptive_precision/bench_e2e.py` — TTFT, TPOT, throughput, peak HBM, transfer, and quality A/B harness.
- `benchmarks/adaptive_precision/report.py` — schema validation, medians/percentiles, and release-gate report.
- `benchmarks/adaptive_precision/workloads.py` — deterministic `deterministic-v1` prompt/token workload generation; no external prompts file is required.
- `docs/adaptive-expert-precision.md` — configuration, metrics, support matrix, fallback reasons, and benchmark protocol.

### Modified files

- `moe_infinity/utils/config.py` — opt-in adaptive policy settings and strict validation.
- `moe_infinity/utils/fp8.py` — deterministic block-128 FP8 quantizer paired with the existing dequantizer.
- `moe_infinity/kernel/marlin_gemm.py` — capability predicate only; kernel math stays unchanged.
- `moe_infinity/runtime/model_offload.py` — resolve capabilities, create/load manifests, register variants, connect policy, and expose metrics without changing protected paths.
- `moe_infinity/distributed/expert_executor.py` — observe routed counts and submit a precision plan before enqueue.
- `moe_infinity/memory/expert_prefetcher.py` — expose format-aware transfer and occupancy metrics.
- `benchmarks/eval/perplexity.py` — emit checkpoint-quality measurements and raw result digests used by attestations.
- `core/model/model_topology.h` and `core/model/model_topology.cpp` — detached representation nodes and alignment-aware byte metadata; canonical `Node::SetDevice` behavior remains unchanged.
- `core/aio/archer_tensor_handle.h` and `core/aio/archer_tensor_handle.cpp` — expose canonical index snapshots and transactionally stage Python-validated derivative records without any C++ file parser or canonical-index rewrite.
- `core/prefetch/expert_residency.h` and `core/prefetch/expert_residency.cpp` — extend Task 0's `ExpertResidencyManager`, `ResidencyEntry`, lease records, and transactions for precision variants; no second residency authority is introduced.
- `core/parallel/expert_module.h` and `core/parallel/expert_module.cpp` — descriptor-driven parameter binding and existing-kernel execution dispatch.
- `core/parallel/expert_dispatcher.h` and `core/parallel/expert_dispatcher.cpp` — immutable execution descriptors and manager-client transaction/lease IDs; residency, bytes, victim selection, and retirement stay in `ExpertResidencyManager`.
- `core/prefetch/archer_prefetch_handle.h` and `core/prefetch/archer_prefetch_handle.cpp` — create detached nodes from existing tensor IDs without adding them to canonical pipeline stages.
- `core/prefetch/task_scheduler.h` and `core/prefetch/task_scheduler.cpp` — remain phase-tagged prefetch clients of `ExpertResidencyManager` and submit variant-aware transactions instead of independently evicting representation storage.
- `core/python/py_archer_prefetch.cpp` — pybind registration, target-plan, and metrics methods.
- `setup.py` — Task 0 registers `core/prefetch/expert_residency.cpp` exactly once; later tasks retain it and propagate `MOE_INFINITY_TESTING=1` when test-only bindings are requested.
- `CMakeLists.txt`, `core/CMakeLists.txt`, and `extensions/CMakeLists.txt` — retain the existing manager source exactly once, add combined variant tests, and define/propagate the same test option.
- `tests/cpp/unit/prefetch/CMakeLists.txt` and `tests/cpp/unit/prefetch/expert_residency_test_fixture.h` — Task 0 registers the real-manager fixture; Task 5 reuses it for variant tests.
- `tests/python/unit/test_setup_sources.py` and `tests/python/unit/test_native_phase_policy_wire.py` — assert one manager source, no alternate residency source, and identical manager identity for scheduler/dispatcher clients.
- `tests/python/unit/test_utils_config.py`, `tests/python/unit/test_quant_regression.py`, `tests/python/unit/test_gptq_loading.py`, and `tests/python/unit/test_awq_loading.py` — config and existing quantization fallback regressions.
- `tests/python/unit/test_glm_fp8_native_dequant.py` and `tests/test_mxfp4_kernel.py` — existing low-bit kernel correctness regressions.
- `tests/python/unit/test_glm_fp8_store.py`, `tests/python/unit/test_gpt_oss_mxfp4_dispatch.py`, `tests/python/v4/test_v4_host_offload.py`, and `tests/python/v4/test_fp4_expert_parity.py` — protected-path regression assertions.

## Exact data contracts

Use these names and values consistently across Python, JSON, pybind, and C++:

```python
class ExpertFormat(str, Enum):
    BF16 = "bf16"
    FP8_E4M3_BLOCK128 = "fp8_e4m3_block128"
    MARLIN_INT4_GROUP128 = "marlin_int4_group128"
    GPT_OSS_MXFP4 = "gpt_oss_mxfp4"
    GLM_FP8_BLOCK128 = "glm_fp8_block128"
    DEEPSEEK_V4_FP4 = "deepseek_v4_fp4"
    GPTQ = "gptq"
    AWQ = "awq"

class ExecutionKind(str, Enum):
    BF16_GEMM = "bf16_gemm"
    FP8_DEQUANT_BF16_GEMM = "fp8_dequant_bf16_gemm"
    MARLIN_W4A16 = "marlin_w4a16"
    GPT_OSS_MXFP4 = "gpt_oss_mxfp4"
    DEEPSEEK_V4_FP4 = "deepseek_v4_fp4"
    LEGACY_QUANTIZED = "legacy_quantized"

@dataclass(frozen=True)
class FormatCapability:
    format: ExpertFormat
    execution: ExecutionKind
    quality_rank: int
    tensor_roles: tuple[str, ...]
    scale_owner: Literal["inline", "manifest", "kernel"]
    supported_storage_dtypes: tuple[torch.dtype, ...]
    output_dtype: torch.dtype
    block_size: int | None
    group_size: int | None
    requires_extension: str | None
    protected: bool

@dataclass(frozen=True)
class CandidateVariantSpec:
    layer_id: int
    expert_id: int
    format: ExpertFormat
    execution: ExecutionKind
    tensor_ids: tuple[int, ...]
    tensor_roles: tuple[str, ...]
    payload_bytes: int
    aligned_bytes: int
    workspace_bytes: int
    source_format: ExpertFormat
    converter_version: str

@dataclass(frozen=True)
class ExpertVariantSpec(CandidateVariantSpec):
    quality_attestation_sha256: str

@dataclass(frozen=True)
class ResidentGeneration:
    format: ExpertFormat
    aligned_bytes: int
    generation: int
    state: Literal["active", "retiring"] = "active"
```

Converter/build eligibility and serving approval are different contracts:

```python
CONVERTER_CANDIDATE_MODEL_TYPES = frozenset({"mixtral", "qwen3_moe", "deepseek_v2"})

@dataclass(frozen=True)
class ReleasedAdaptiveEntry:
    checkpoint_fingerprint: str
    format: ExpertFormat
    converter_version: str
    quality_attestation_sha256: str

RELEASED_ADAPTIVE_ENTRIES: frozenset[ReleasedAdaptiveEntry] = frozenset()
```

The initial released allowlist is empty. A converter candidate may build and benchmark derivatives, but serving remains canonical until a reviewed attestation digest is added in a later change. Approval lookup is exact over all four fields; model type or model name alone never approves a manifest.

Derivative publication layout:

```text
{offload_path}/adaptive_derivatives/build-journal.v1.json
{offload_path}/adaptive_derivatives/canonical-checkpoint-fingerprint.v1.json
{offload_path}/adaptive_derivatives/generations/{generation}/derivative-index.v1.json
{offload_path}/adaptive_derivatives/generations/{generation}/quality-attestation.v1.json
{offload_path}/adaptive_derivatives/generations/{generation}/manifest.v1.json
{offload_path}/adaptive_derivatives/CURRENT
{offload_path}/archer_param_{reserved_derivative_file_id}
```

All four metadata files use the same canonical JSON encoder: `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"`. Every SHA-256 field is exactly 64 lowercase hexadecimal characters and hashes the complete canonical bytes, including the final newline. `CURRENT` has this strict schema (unknown or missing keys reject it):

```json
{"derivative_index_sha256":"64-lowercase-hex","generation":"g00000001","manifest_sha256":"64-lowercase-hex","quality_attestation_sha256":"64-lowercase-hex","schema_version":1}
```

Serving reads no generation that is not named by a valid `CURRENT`. It reads `CURRENT` once, validates its schema, opens only that generation, hashes the three named files before parsing any of them, and requires all three hashes to equal `CURRENT`. It then requires the manifest's `derivative_index_sha256` and `quality_attestation_sha256` to equal the same `CURRENT` values. A mismatch is `manifest_digest_mismatch`; no derivative tensor is registered.

The checkpoint fingerprint is SHA-256 over canonical JSON with strict schema `{"schema_version":1,"tensor_index":[...],"model_signature":<parsed existing model-signature JSON>,"partitions":[...]}`. `tensor_index` is sorted by `tensor_id`; each row has exactly `tensor_id`, `dtype`, `shape`, `size`, `file_id`, and `offset`. `partitions` is sorted by `file_id`; each row is `{"file_id":N,"sha256":"...","size":N}` for canonical `archer_param_{file_id}` only. `dtype` uses the stable strings `float8_e4m3fn`, `float32`, `bfloat16`, `float16`, `int32`, or `uint8`, never a process-local `c10::ScalarType` integer. Python obtains the tensor rows from the explicit `prefetch_handle.get_canonical_tensor_index_snapshot()` pybind method, canonical-encodes the envelope above, and hashes those bytes. The binding returns canonical rows only, sorted by `tensor_id`, and remains unchanged after an overlay is committed. Candidate build computes and persists this once before reserving derivative ranges. Serving recomputes it from the canonical snapshot, model signature, and canonical partitions and requires equality with both the persisted fingerprint record and manifest before release lookup; it does not substitute model type/name or config fingerprint.

`canonical-checkpoint-fingerprint.v1.json` has exactly these keys:

```json
{
  "canonical_max_file_id": 3,
  "canonical_max_tensor_id": 100,
  "checkpoint_fingerprint": "64-lowercase-hex",
  "model_signature_sha256": "64-lowercase-hex",
  "partitions": [{"file_id": 0, "sha256": "64-lowercase-hex", "size": 1073741824}],
  "schema_version": 1,
  "tensor_index_sha256": "64-lowercase-hex"
}
```

`tensor_index_sha256` hashes the canonical JSON bytes of the sorted snapshot list; `model_signature_sha256` hashes the canonical JSON bytes of the parsed model-signature object. Empty canonical stores are rejected rather than assigned maxima. Unknown/missing keys, changed partition size/digest, or disagreement among recomputed maxima, persisted maxima, derivative index, and manifest rejects the generation before native registration.

`derivative-index.v1.json` has this strict schema:

```json
{
  "canonical_max_file_id": 3,
  "canonical_max_tensor_id": 100,
  "generation": "g00000001",
  "records": [
    {
      "dtype": "float8_e4m3fn",
      "file_id": 4,
      "offset": 0,
      "sha256": "64-lowercase-hex",
      "shape": [4096, 14336],
      "size": 58720256,
      "tensor_id": 101
    }
  ],
  "schema_version": 1
}
```

Records are sorted by `tensor_id`; IDs are unique unsigned 32-bit integers strictly above their canonical maxima; `offset` is a nonnegative signed 64-bit integer aligned to `kAioAlignment`; `size` is a positive signed 64-bit integer and equals `prod(shape) * element_size(dtype)`; every shape dimension is a positive signed 64-bit integer; file intervals do not overlap; each record checksum covers exactly `size` bytes at `[offset, offset + size)` in `archer_param_{file_id}`. The dtype-to-native mapping is explicit: `float8_e4m3fn -> at::kFloat8_e4m3fn`, `float32 -> at::kFloat`, `bfloat16 -> at::kBFloat16`, `float16 -> at::kHalf`, `int32 -> at::kInt`, and `uint8 -> at::kByte`. Unknown keys, dtypes, or numeric overflows reject the whole generation.

Python alone reads and writes derivative JSON. After validating `CURRENT`, all schemas/digests, every payload interval checksum, and release entries, it opens one native overlay transaction and feeds sorted records through these explicit pybind methods; C++ never parses a derivative metadata file:

```text
prefetch_handle.get_canonical_tensor_index_snapshot() -> list[dict]
prefetch_handle.begin_derivative_overlay(generation, canonical_max_tensor_id, canonical_max_file_id)
prefetch_handle.register_derivative_tensor(generation, tensor_id, file_id, offset, size, shape, dtype)
prefetch_handle.commit_derivative_overlay(generation)
prefetch_handle.abort_derivative_overlay(generation)
```

`begin_derivative_overlay` rejects a second transaction. `register_derivative_tensor` accepts only the six dtype strings above and rechecks ranges, alignment, size, uniqueness, and disjointness against canonical and staged derivative records. It stages metadata without mutating `kTensorIndex`. `commit_derivative_overlay` atomically merges the staged generation and records its IDs as derivative-owned, so canonical snapshots continue to exclude them. Any exception calls idempotent `abort_derivative_overlay`; therefore partial overlays are never visible.

```json
{
  "schema_version": 1,
  "model_name": "Qwen/Qwen3-30B-A3B",
  "checkpoint_fingerprint": "sha256-hex",
  "store_signature_version": 1,
  "converter_version": "adaptive-expert-v1",
  "generation": "g00000001",
  "canonical_max_tensor_id": 100,
  "canonical_max_file_id": 3,
  "derivative_index_sha256": "64-lowercase-hex",
  "quality_attestation_sha256": "64-lowercase-hex",
  "variant_payload_sha256": "64-lowercase-hex",
  "complete": true,
  "variants": [
    {
      "layer_id": 0,
      "expert_id": 0,
      "format": "fp8_e4m3_block128",
      "execution": "fp8_dequant_bf16_gemm",
      "tensor_ids": [101, 102, 103, 104, 105, 106],
      "tensor_roles": ["gate.weight", "gate.scale", "up.weight", "up.scale", "down.weight", "down.scale"],
      "payload_bytes": 6303744,
      "aligned_bytes": 6328320,
      "workspace_bytes": 12582912,
      "source_format": "bf16",
      "converter_version": "adaptive-expert-v1",
      "quality_attestation_sha256": "64-lowercase-hex"
    }
  ]
}
```

`quality-attestation.v1.json` is generation-scoped and has this strict schema; `formats` is sorted lexically and contains one row for every format in the manifest:

```json
{
  "checkpoint_fingerprint": "64-lowercase-hex",
  "converter_source_commit": "40-lowercase-hex",
  "converter_version": "adaptive-expert-v1",
  "derivative_index_sha256": "64-lowercase-hex",
  "formats": [
    {
      "format": "fp8_e4m3_block128",
      "greedy_agreement": 0.995,
      "kernel_atol": 0.08,
      "kernel_rtol": 0.08,
      "perplexity_adaptive": 3.01,
      "perplexity_baseline": 3.0,
      "relative_l2_max": 0.05,
      "tensor_cosine_min": 0.995
    }
  ],
  "generation": "g00000001",
  "hardware": {
    "compute_capability": [8, 0],
    "cuda_runtime": "runtime-reported",
    "device_name": "runtime-reported"
  },
  "passed": true,
  "raw_result_sha256": "64-lowercase-hex",
  "schema_version": 1,
  "software": {
    "moe_infinity_commit": "40-lowercase-hex",
    "python": "runtime-reported",
    "torch": "runtime-reported"
  },
  "thresholds": {
    "greedy_agreement_min": 0.99,
    "perplexity_relative_increase_max": 0.01
  },
  "variant_payload_sha256": "64-lowercase-hex",
  "workload_sha256": "64-lowercase-hex"
}
```

All numeric quality values must be finite JSON numbers. The loader requires `passed` to be exactly `true`, recomputes `(perplexity_adaptive / perplexity_baseline) - 1 <= thresholds.perplexity_relative_increase_max`, requires each format's greedy agreement to meet the global minimum, and requires attestation checkpoint/generation/converter/index/payload fields to equal the manifest. `quality_attestation_sha256` is SHA-256 of these canonical bytes and is recorded in `CURRENT`, at manifest top level, and in every manifest variant.

The build journal records state (`reserved`, `writing`, `indexed`, `attested`, `published`), generation, canonical high-water marks, reserved derivative tensor-ID range, reserved derivative file-ID range, completed tensor checksums, and converter version. All derivative tensor IDs and file IDs are strictly above the canonical maxima. `variant_payload_sha256` hashes the canonical JSON array of sorted variant records with `quality_attestation_sha256` omitted; the attestation records that payload digest and the derivative-index digest, and the final manifest records the attestation digest. This avoids a manifest/attestation hash cycle. Every manifest variant uses that same generation-attestation digest; release approval remains one exact entry per included format.

`build-journal.v1.json` also rejects unknown/missing keys and has exactly this schema; range endpoints are inclusive and completed tensors are sorted by `tensor_id`:

```json
{
  "canonical_max_file_id": 3,
  "canonical_max_tensor_id": 100,
  "completed_tensors": [{"sha256": "64-lowercase-hex", "size": 58720256, "tensor_id": 101}],
  "converter_version": "adaptive-expert-v1",
  "generation": "g00000001",
  "next_file_id": 6,
  "next_tensor_id": 107,
  "published_ranges": [],
  "reserved_file_id_range": [4, 5],
  "reserved_tensor_id_range": [101, 106],
  "schema_version": 1,
  "state": "writing"
}
```

Each `published_ranges` row has exactly `generation`, `file_id_range`, and `tensor_id_range`. Every transition rewrites the complete journal durably; `next_*` values only increase, including after an aborted/corrupt generation.

Retry reuses the same unpublished generation only after checksum validation; otherwise it removes its temporary index/attestation/manifest, allocates a new generation/ranges from the durable journal, and never reuses a published range. Publication uses this exact order: write each payload `.tmp`, `flush` and `fsync` it, `os.replace` it to `archer_param_{file_id}`; after all payloads, `fsync` the offload root; canonical-write `derivative-index.v1.json.tmp`, `flush`/`fsync`, and `os.replace`; repeat for `quality-attestation.v1.json`, then `manifest.v1.json`; `fsync` the generation directory; canonical-write sibling `CURRENT.tmp`, `flush`/`fsync`, `os.replace` to `CURRENT`; finally `fsync` `adaptive_derivatives`. The journal reaches `published` only after the final directory `fsync`. The canonical index and `name_id_map.json` are never rewritten.

Native ownership is split so the manager owns mutable residency state and the dispatcher owns only immutable execution binding:

```cpp
struct ExpertExecutionDescriptor {
  ResidencyVariantKey key;
  std::uint8_t execution_kind = 0;
  std::vector<TensorID> tensor_ids;
  std::vector<std::string> tensor_roles;
};

// ExpertFormat, generation, NodePtr, bytes, state, leases, transactions,
// workspace, and retirement events live only in core/prefetch/expert_residency.h.
```

Budget invariant on each GPU:

```text
resident_generation_aligned_bytes
+ transition_reserved_bytes
+ execution_workspace_bytes
<= adaptive_hbm_budget_bytes
```

`ResidencyState::HOST_READY` representations consume zero HBM budget. Only device-resident active generations, device-resident retiring generations awaiting lease/event completion, transaction reservations, and execution workspace are counted. The policy never precharges the smallest representation for every catalogued expert; it chooses formats only for the current resident subset and demand/prefetch admission candidates. Temporary per-call dequant buffers are workspace, not transition bytes, and are released only after the execution-stream completion event. Payload bytes and alignment padding are reported separately.

Composition is transaction-ordered through Task 0's manager: Python chooses targets → scheduler/dispatcher clients call `BeginAdmission` or `BeginTransition` → the manager reserves resident/victim/workspace bytes → clients transfer/convert under manager-issued transfer leases → clients `CommitTransaction` or `AbortTransaction` → execution uses an execution lease → clients request retirement and the manager reaps only lease-free/event-complete generations. `TaskScheduler` and `ExpertDispatcher` use the same `ExpertResidencyManager`; neither maintains a representation resident map, byte counter, lease registry, or victim selector. Existing always-resident shared experts remain outside the adaptive budget and are reported as `external_shared_resident_bytes`; a routed representation generation is charged once even when both clients hold leases.

## Task 0: Establish the single shared residency foundation

**Files:**
- Create if absent, otherwise validate/modify: `core/prefetch/expert_residency.h`
- Create if absent, otherwise validate/modify: `core/prefetch/expert_residency.cpp`
- Create if absent, otherwise validate/modify: `core/prefetch/expert_policy.h`
- Modify: `setup.py:177-203`
- Modify: `core/CMakeLists.txt:1-44`
- Modify: `core/prefetch/archer_prefetch_handle.h:13-80`
- Modify: `core/prefetch/archer_prefetch_handle.cpp:20-408`
- Modify: `core/prefetch/task_scheduler.h:24-121`
- Modify: `core/prefetch/task_scheduler.cpp:47-597`
- Modify: `core/parallel/expert_dispatcher.h:40-194`
- Modify: `core/parallel/expert_dispatcher.cpp:180-586`
- Modify: `core/python/py_archer_prefetch.cpp:18-123`
- Create if absent, otherwise validate/modify: `tests/cpp/unit/prefetch/expert_residency_test_fixture.h`
- Create if absent, otherwise validate/modify: `tests/cpp/unit/prefetch/test_expert_residency.cpp`
- Modify: `tests/cpp/unit/prefetch/CMakeLists.txt`
- Create if absent, otherwise validate/modify: `tests/python/unit/test_native_phase_policy_wire.py`
- Create if absent, otherwise validate/modify: `tests/python/unit/test_setup_sources.py`

- [ ] **Step 1: Detect whether the exact foundation already landed**

PR #179 is design coordination only. Do not fetch, rebase onto, or cite a plan commit as implementation. Run this check against the implementation tree:

```bash
python - <<'PY'
from pathlib import Path

paths = [
    Path("core/prefetch/expert_residency.h"),
    Path("core/prefetch/expert_residency.cpp"),
    Path("tests/cpp/unit/prefetch/test_expert_residency.cpp"),
]
if not all(path.is_file() for path in paths):
    print("CREATE_FOUNDATION")
    raise SystemExit(0)
header = paths[0].read_text()
required = [
    "class ExpertResidencyManager", "struct ResidencyTicket",
    "struct ResidencyEntry", "struct LeaseRecord",
    "class ExpertResidencyClient", "ConfigureCapacity(",
    "BeginAdmission(", "EvictReserved(", "CommitAdmission(",
    "AbortAdmission(", "AcquireLease(", "ReleaseLease(",
    "ReplaceProtectedCandidates(", "RecordAccess(", "Snapshot(",
    "ResidentBytes(", "ResidentCount(",
]
missing = [symbol for symbol in required if symbol not in header]
print("VALIDATE_FOUNDATION" if not missing else "RECONCILE_FOUNDATION:" + ",".join(missing))
PY
```

Expected: a tree without implementation prints `CREATE_FOUNDATION`; an exact landed implementation prints `VALIDATE_FOUNDATION`; an incomplete or divergent implementation prints `RECONCILE_FOUNDATION:<symbols>`. Continue through every remaining Task 0 step in all cases. In the validate path, preserve matching definitions instead of recreating them; in the other paths, create or reconcile these same files. No path creates another manager, compatibility ledger, or wrapper authority.

- [ ] **Step 2: Write the failing build-registration and real-manager tests**

`tests/python/unit/test_setup_sources.py` must parse, without importing `setup.py`, and assert:

```python
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def read_literal_list(path: Path, name: str) -> list[str]:
    module = ast.parse(path.read_text())
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in statement.targets):
            value = ast.literal_eval(statement.value)
            assert isinstance(value, list)
            return value
    raise AssertionError(f"{name} not found in {path}")


def test_store_sources_link_one_residency_authority() -> None:
    sources = read_literal_list(ROOT / "setup.py", "_STORE_SOURCES")
    residency = [p for p in sources if "residency" in p]
    assert residency == ["core/prefetch/expert_residency.cpp"]


def test_cmake_links_one_residency_authority() -> None:
    text = (ROOT / "core/CMakeLists.txt").read_text()
    assert text.count("prefetch/expert_residency.cpp") == 1
    assert text.count("residency.cpp") == 1
```

Create the real-manager fixture with real `Node` objects and no fake accounting:

```cpp
#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "prefetch/expert_residency.h"

class RecordingTransferOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override {
    moved_ids.push_back(node->id);
    node->device = node->default_host;
    return true;
  }
  std::vector<std::size_t> moved_ids;
};

class ExpertResidencyManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    transfer_ops = std::make_shared<RecordingTransferOps>();
    manager = std::make_shared<ExpertResidencyManager>(transfer_ops);
  }

  NodePtr MakeNode(std::size_t id, std::int64_t bytes) {
    auto node = std::make_shared<Node>();
    node->id = id;
    node->corr_id = id;
    node->byte_size = bytes;
    node->device = torch::Device(torch::kCPU);
    node->default_device = torch::Device(torch::kCUDA, 0);
    node->default_host = torch::Device(torch::kCPU);
    return node;
  }

  std::shared_ptr<RecordingTransferOps> transfer_ops;
  std::shared_ptr<ExpertResidencyManager> manager;
};
```

Add these concrete tests to `test_expert_residency.cpp`:

```cpp
TEST_F(ExpertResidencyManagerTest, AdmissionRejectsUntilCapacityConfigured) {
  auto node = MakeNode(1, 40);
  auto rejected = manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
      AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.outcome, AdmissionOutcome::REJECTED);
  EXPECT_EQ(manager->ResidentBytes(0), 0);
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto admitted = manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
      AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_TRUE(admitted.valid);
  EXPECT_TRUE(manager->AbortAdmission(admitted));
}

TEST_F(ExpertResidencyManagerTest, CapacityAndVictimTransactionAreAtomic) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 60);
  auto first_ticket = manager->BeginAdmission(first, 0, ExpertPhase::DECODE,
      AdmissionMode::CACHE, AdmissionSource::DEMAND);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(first_ticket));
  auto second = MakeNode(2, 60);
  auto second_ticket = manager->BeginAdmission(second, 0, ExpertPhase::DECODE,
      AdmissionMode::CACHE, AdmissionSource::PREFETCH);
  ASSERT_EQ(second_ticket.reserved_victim, first);
  ASSERT_TRUE(manager->EvictReserved(second_ticket));
  second->device = second->default_device;
  ASSERT_TRUE(manager->CommitAdmission(second_ticket));
  EXPECT_EQ(manager->ResidentBytes(0), 60);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest, LeaseBlocksEvictionUntilReleased) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 100);
  auto ticket = manager->BeginAdmission(first, 0, ExpertPhase::DECODE,
      AdmissionMode::CACHE, AdmissionSource::DEMAND);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  const auto lease = manager->AcquireLease(first, LeaseKind::DEMAND);
  EXPECT_FALSE(manager->BeginAdmission(MakeNode(2, 100), 0,
      ExpertPhase::DECODE, AdmissionMode::CACHE,
      AdmissionSource::DEMAND).valid);
  EXPECT_TRUE(manager->ReleaseLease(lease));
  EXPECT_FALSE(manager->ReleaseLease(lease));
  auto retry = manager->BeginAdmission(MakeNode(3, 100), 0,
      ExpertPhase::DECODE, AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_EQ(retry.reserved_victim, first);
  EXPECT_TRUE(manager->AbortAdmission(retry));
}

TEST_F(ExpertResidencyManagerTest, DemandAndPrefetchClientsShareAccounting) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  ExpertResidencyClient demand(manager, AdmissionSource::DEMAND);
  ExpertResidencyClient prefetch(manager, AdmissionSource::PREFETCH);
  auto node = MakeNode(1, 40);
  auto admitted = demand.BeginAdmission(node, 0, ExpertPhase::DECODE,
                                        AdmissionMode::CACHE);
  node->device = node->default_device;
  ASSERT_TRUE(manager->CommitAdmission(admitted));
  auto duplicate = prefetch.BeginAdmission(node, 0, ExpertPhase::PREFILL,
                                            AdmissionMode::CACHE);
  EXPECT_EQ(duplicate.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_EQ(demand.manager().get(), prefetch.manager().get());
  EXPECT_EQ(manager->ResidentBytes(0), 40);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}
```

Run:

```bash
pytest -q tests/python/unit/test_setup_sources.py
cmake -S . -B build -DBUILD_TESTING=ON -DCUTLASS_DIR="$CUTLASS_DIR"
cmake --build build --target test_expert_residency -j2
```

Expected: on an absent foundation, the Python assertion and native compile fail because source registration and manager types are absent. On a landed exact foundation, they pass and protect its API before adaptive extension. On a divergent foundation, at least one assertion or compile fails and identifies what Task 0 must reconcile.

- [ ] **Step 3: Create or reconcile the exact fixed-size manager API**

Use the phase-policy plan as the naming contract, not as implementation evidence. `expert_residency.h` must expose exactly this fixed-size surface before Task 5 extends it:

```cpp
enum class AdmissionSource : std::uint8_t { DEMAND = 0, PREFETCH = 1 };
enum class AdmissionOutcome : std::uint8_t {
  ADMIT = 0, ALREADY_RESIDENT = 1, TRANSIENT = 2, REJECTED = 3
};
enum class LeaseKind : std::uint8_t { DEMAND = 0, PREFETCH = 1, TRANSFER = 2 };

class ResidencyTransferOps {
 public:
  virtual ~ResidencyTransferOps() = default;
  virtual bool MoveToHost(const NodePtr& node) = 0;
};

struct ResidencyTicket {
  std::uint64_t id = 0;
  NodePtr incoming;
  NodePtr reserved_victim;
  int gpu_id = -1;
  ExpertPhase phase = ExpertPhase::MIXED;
  AdmissionSource source = AdmissionSource::DEMAND;
  AdmissionOutcome outcome = AdmissionOutcome::REJECTED;
  bool transient = false;
  bool valid = false;
};
struct ResidencyEntry { NodePtr node; std::int64_t bytes = 0; std::uint32_t lease_count = 0; };
struct LeaseRecord { std::uint64_t id = 0; NodePtr node; LeaseKind kind = LeaseKind::DEMAND; };
using ExpertPolicyStats = std::unordered_map<std::string, std::int64_t>;

class ExpertResidencyManager {
 public:
  explicit ExpertResidencyManager(std::shared_ptr<ResidencyTransferOps> transfer_ops);
  bool ConfigureCapacity(int gpu_id, std::int64_t capacity_bytes);
  bool IsCapacityConfigured(int gpu_id) const;
  std::int64_t CapacityBytes(int gpu_id) const;
  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
      ExpertPhase phase, AdmissionMode mode, AdmissionSource source);
  bool EvictReserved(const ResidencyTicket& ticket);
  bool CommitAdmission(const ResidencyTicket& ticket);
  bool AbortAdmission(const ResidencyTicket& ticket);
  std::uint64_t AcquireLease(const NodePtr& node, LeaseKind kind);
  bool ReleaseLease(std::uint64_t lease_id);
  void ReplaceProtectedCandidates(const NodePtrList& candidates);
  void RecordAccess(const NodePtr& node, ExpertPhase phase, bool hit);
  ExpertPolicyStats Snapshot() const;
  std::int64_t ResidentBytes(int gpu_id) const;
  std::size_t ResidentCount(int gpu_id) const;
};

class ExpertResidencyClient {
 public:
  ExpertResidencyClient(std::shared_ptr<ExpertResidencyManager> manager,
                        AdmissionSource source)
      : manager_(std::move(manager)), source_(source) {}
  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
      ExpertPhase phase, AdmissionMode mode) {
    return manager_->BeginAdmission(incoming, gpu_id, phase, mode, source_);
  }
  std::shared_ptr<ExpertResidencyManager> manager() const { return manager_; }
 private:
  std::shared_ptr<ExpertResidencyManager> manager_;
  AdmissionSource source_;
};
```

`expert_policy.h` supplies the phase plan's exact `ExpertPhase`, `AdmissionMode`, `PhasePolicyConfig`, `ExpertPolicyMetadata`, stable `VictimLess`, and `EffectivePhase` definitions. Implement `expert_residency.cpp` with one mutex-protected per-GPU resident map, byte totals, pending tickets, leases, protected candidates, and optional capacities. Capacity begins unconfigured; invalid/pending-ticket reconfiguration is atomic; duplicate admission is `ALREADY_RESIDENT`; victim reservation uses one lease; commit/abort/release are idempotent Boolean operations; only `EvictReserved` removes and only `CommitAdmission` adds persistent membership.

Use these exact policy definitions required by that API:

```cpp
enum class ExpertPhase : std::uint8_t { PREFILL = 0, DECODE = 1, MIXED = 2 };
enum class AdmissionMode : std::uint8_t {
  CACHE = 0, TRANSIENT_ON_PRESSURE = 1
};
struct PhasePolicyConfig {
  bool enabled = false;
  AdmissionMode prefill_admission = AdmissionMode::TRANSIENT_ON_PRESSURE;
  AdmissionMode decode_admission = AdmissionMode::CACHE;
  double prefill_eviction_weight = 1.0;
  double decode_eviction_weight = 4.0;
  std::uint32_t starvation_limit = 8;
};
struct ExpertPolicyMetadata {
  std::uint64_t prefill_accesses = 0;
  std::uint64_t decode_accesses = 0;
  std::uint64_t last_prefill_sequence = 0;
  std::uint64_t last_decode_sequence = 0;
};
inline ExpertPhase EffectivePhase(ExpertPhase phase) {
  return phase == ExpertPhase::MIXED ? ExpertPhase::DECODE : phase;
}
```

- [ ] **Step 4: Register and wire the one authority**

Add `core/prefetch/expert_residency.cpp` exactly once to `_STORE_SOURCES` and exactly once to `ARCHER_CORE_CXX_SOURCES`. Register `test_expert_residency` as `ExpertResidencyTests` in `tests/cpp/unit/prefetch/CMakeLists.txt`. Create one shared, initially unconfigured `kExpertResidencyManager` at the native topology/task-pool lifetime; construct demand and prefetch `ExpertResidencyClient`s from that same `shared_ptr`. After each successful `SetTopology` and `SetTopologyV2`, configure each GPU from `GetSparseCacheLimit(device)`; no constructor, dispatcher, scheduler, or Python config call may infer capacity.

Route manager-enabled dispatcher and scheduler admissions through `BeginAdmission` → optional `EvictReserved` → transfer lease → `CommitAdmission`/`AbortAdmission`. Bind the shared `Snapshot()` as `get_expert_policy_stats()` in `py_archer_prefetch.cpp`; bindings expose a view only and own no map or ledger. `test_native_phase_policy_wire.py` must assert the demand and prefetch clients expose the same manager identity and one non-duplicated resident-byte snapshot.

- [ ] **Step 5: Verify the fixed-size foundation**

Run:

```bash
pytest -q tests/python/unit/test_setup_sources.py tests/python/unit/test_native_phase_policy_wire.py
cmake -S . -B build -DBUILD_TESTING=ON -DCUTLASS_DIR="$CUTLASS_DIR"
cmake --build build --target test_expert_residency -j2
ctest --test-dir build -R '^ExpertResidencyTests$' --output-on-failure
pip install --no-build-isolation -e .
python -c 'from moe_infinity import _store; assert hasattr(_store, "expert_dispatcher")'
```

Expected: source tests find exactly one residency implementation in each build graph; the four concrete real-manager tests pass; admission is rejected until topology-derived capacity is configured; transaction and lease operations are idempotent and capacity-safe; dispatcher and scheduler report the same manager identity and one byte count; `_store` links without undefined manager symbols.

- [ ] **Step 6: Commit the self-contained foundation when Task 0 changed it**

```bash
if ! git diff --quiet -- setup.py core/CMakeLists.txt core/prefetch core/parallel/expert_dispatcher.h core/parallel/expert_dispatcher.cpp core/python/py_archer_prefetch.cpp tests/cpp/unit/prefetch tests/python/unit/test_native_phase_policy_wire.py tests/python/unit/test_setup_sources.py; then
  git add setup.py core/CMakeLists.txt core/prefetch/expert_policy.h core/prefetch/expert_residency.h core/prefetch/expert_residency.cpp core/prefetch/archer_prefetch_handle.h core/prefetch/archer_prefetch_handle.cpp core/prefetch/task_scheduler.h core/prefetch/task_scheduler.cpp core/parallel/expert_dispatcher.h core/parallel/expert_dispatcher.cpp core/python/py_archer_prefetch.cpp tests/cpp/unit/prefetch/CMakeLists.txt tests/cpp/unit/prefetch/expert_residency_test_fixture.h tests/cpp/unit/prefetch/test_expert_residency.cpp tests/python/unit/test_native_phase_policy_wire.py tests/python/unit/test_setup_sources.py
  git commit -m "feat: establish shared expert residency manager"
fi
```

Expected: create/reconcile path produces the foundation commit. If an exact implementation had already landed and every Task 0 check passed without modification, the conditional makes no duplicate or empty commit.

## Task 1: Add strict opt-in configuration, converter candidates, and released entries

**Files:**
- Create: `moe_infinity/runtime/expert_precision.py`
- Create: `moe_infinity/runtime/adaptive_precision_allowlist.py`
- Modify: `moe_infinity/utils/config.py:17-162`
- Test: `tests/python/unit/test_expert_precision_registry.py`
- Test: `tests/python/unit/test_utils_config.py`

- [ ] **Step 1: Write failing registry and configuration tests**

```python
from types import SimpleNamespace

import pytest
import torch

from moe_infinity.runtime.expert_precision import (
    ExecutionKind,
    ExpertFormat,
    resolve_model_precision_capabilities,
)
from moe_infinity.runtime.adaptive_precision_allowlist import (
    ReleasedAdaptiveEntry,
    is_released,
)
from moe_infinity.utils import ArcherConfig


def test_adaptive_precision_is_disabled_by_default(tmp_path):
    cfg = ArcherConfig(
        offload_path=str(tmp_path),
        use_native_engine=False,
        kv_cache_memory_ratio=0.0,
    )
    assert cfg.adaptive_expert_precision is False
    assert cfg.adaptive_hbm_budget_bytes == 0


def test_general_converter_candidate_separates_storage_and_execution():
    model = SimpleNamespace(model_type="qwen3_moe", quantization_config=None)
    caps = resolve_model_precision_capabilities(model, extension_names=set())
    assert caps.protected_reason is None
    assert caps.formats[ExpertFormat.FP8_E4M3_BLOCK128].execution is ExecutionKind.FP8_DEQUANT_BF16_GEMM
    assert caps.formats[ExpertFormat.FP8_E4M3_BLOCK128].output_dtype is torch.bfloat16


def test_converter_candidate_is_not_implicitly_released():
    entry = ReleasedAdaptiveEntry(
        checkpoint_fingerprint="a" * 64,
        format=ExpertFormat.FP8_E4M3_BLOCK128,
        converter_version="adaptive-expert-v1",
        quality_attestation_sha256="b" * 64,
    )
    assert is_released(entry) is False


@pytest.mark.parametrize(
    ("model_type", "reason"),
    [
        ("gpt_oss", "protected:gpt_oss_mxfp4"),
        ("glm_moe_dsa", "protected:glm_fp8"),
        ("deepseek_v4", "protected:deepseek_v4_fp4"),
    ],
)
def test_model_specific_low_bit_paths_are_protected(model_type, reason):
    model = SimpleNamespace(model_type=model_type, quantization_config={"quant_method": "fp8"})
    caps = resolve_model_precision_capabilities(model, extension_names={"_v4_fp4"})
    assert caps.protected_reason == reason
    assert all(cap.protected for cap in caps.formats.values())


def test_invalid_adaptive_budget_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="adaptive_hbm_budget_bytes must be positive"):
        ArcherConfig.load_from_json(
            {
                "offload_path": str(tmp_path),
                "use_native_engine": False,
                "adaptive_expert_precision": True,
                "adaptive_hbm_budget_bytes": 0,
            }
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/python/unit/test_expert_precision_registry.py tests/python/unit/test_utils_config.py`

Expected: collection fails because `moe_infinity.runtime.expert_precision` and the adaptive config fields do not exist.

- [ ] **Step 3: Implement the registry and config contract**

Add these `ArcherConfig` fields:

```python
adaptive_expert_precision: bool = field(default=False)
adaptive_hbm_budget_bytes: int = field(default=0)
adaptive_policy_epoch_tokens: int = field(default=128)
adaptive_hotness_decay: float = field(default=0.95)
adaptive_promotion_threshold: float = field(default=0.70)
adaptive_demotion_threshold: float = field(default=0.30)
adaptive_min_residency_epochs: int = field(default=2)
adaptive_transition_cooldown_epochs: int = field(default=2)
adaptive_variant_build: bool = field(default=False)
adaptive_derivative_root: Optional[str] = field(default=None)
```

Validate that enabled mode has a positive byte budget, epoch tokens and residency/cooldown are nonnegative integers, `0 < decay <= 1`, and `0 <= demotion < promotion <= 1`. When `adaptive_derivative_root` is `None`, resolve it to `os.path.join(offload_path, "adaptive_derivatives")` without creating the directory in serving mode. Keep all old config defaults unchanged.

In `expert_precision.py`, define the exact contracts above, an immutable `ModelPrecisionCapabilities`, and:

```python
CONVERTER_CANDIDATE_MODEL_TYPES = frozenset({"mixtral", "qwen3_moe", "deepseek_v2"})
PROTECTED_MODEL_PATHS = {
    "gpt_oss": "protected:gpt_oss_mxfp4",
    "glm_moe_dsa": "protected:glm_fp8",
    "deepseek_v3": "protected:existing_fp8",
    "deepseek_v4": "protected:deepseek_v4_fp4",
}


def resolve_model_precision_capabilities(
    config: object,
    extension_names: set[str],
) -> ModelPrecisionCapabilities:
    model_type = str(getattr(config, "model_type", ""))
    quant_method = str(
        (getattr(config, "quantization_config", None) or {}).get("quant_method", "")
    ).lower()
    if model_type in PROTECTED_MODEL_PATHS:
        return protected_capabilities(model_type, PROTECTED_MODEL_PATHS[model_type])
    if quant_method in {"gptq", "awq", "mxfp4", "fp8"}:
        return protected_capabilities(model_type, f"protected:{quant_method}")
    if model_type not in CONVERTER_CANDIDATE_MODEL_TYPES:
        return ModelPrecisionCapabilities(model_type, {}, "unsupported:converter_candidate")
    formats = {ExpertFormat.BF16: bf16_capability()}
    formats[ExpertFormat.FP8_E4M3_BLOCK128] = fp8_capability()
    if "_marlin" in extension_names:
        formats[ExpertFormat.MARLIN_INT4_GROUP128] = marlin_capability()
    return ModelPrecisionCapabilities(model_type, formats, None)
```

Hardware capability is probed from installed extensions, tensor shapes, dtypes, CUDA availability, and compute capability; no GPU name or assumed bandwidth appears in registry logic.

In `adaptive_precision_allowlist.py`, define `ReleasedAdaptiveEntry`, an initially empty immutable `RELEASED_ADAPTIVE_ENTRIES`, and exact membership function `is_released(entry)`. Build/benchmark tools consult converter candidates; serving consults only released entries. Adding a released entry requires a separate reviewed source change containing the exact four-field key and its attestation artifact digest.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `pytest -q tests/python/unit/test_expert_precision_registry.py tests/python/unit/test_utils_config.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/runtime/expert_precision.py moe_infinity/runtime/adaptive_precision_allowlist.py moe_infinity/utils/config.py tests/python/unit/test_expert_precision_registry.py tests/python/unit/test_utils_config.py
git commit -m "feat: define adaptive expert precision capabilities"
```

## Task 2: Implement explicit derivative conversion and kernel capability checks

**Files:**
- Modify: `moe_infinity/utils/fp8.py:11-85`
- Modify: `moe_infinity/kernel/marlin_gemm.py:21-224`
- Create: `tests/python/unit/test_expert_variant_conversion.py`
- Modify: `tests/python/unit/test_glm_fp8_native_dequant.py:6-25`
- Modify: `tests/test_mxfp4_kernel.py:17-103`

- [ ] **Step 1: Write failing deterministic conversion tests**

```python
import pytest
import torch

from moe_infinity.kernel.marlin_gemm import marlin_supports_shape
from moe_infinity.utils.fp8 import dequant_fp8_blockwise, quant_fp8_blockwise


def test_fp8_blockwise_roundtrip_is_deterministic_and_owns_scale():
    torch.manual_seed(7)
    weight = torch.randn(257, 513, dtype=torch.bfloat16)
    q1, s1 = quant_fp8_blockwise(weight, block_size=128)
    q2, s2 = quant_fp8_blockwise(weight, block_size=128)
    assert q1.dtype == torch.float8_e4m3fn
    assert s1.dtype == torch.float32
    assert q1.shape == weight.shape
    assert s1.shape == (3, 5)
    assert torch.equal(q1.view(torch.uint8), q2.view(torch.uint8))
    assert torch.equal(s1, s2)
    restored = dequant_fp8_blockwise(q1, s1, dtype=torch.bfloat16)
    assert torch.isfinite(restored).all()


@pytest.mark.parametrize(
    ("k", "n", "available", "expected"),
    [(256, 512, True, True), (255, 512, True, False), (256, 511, True, False), (256, 512, False, False)],
)
def test_marlin_capability_requires_extension_and_layout(k, n, available, expected):
    assert marlin_supports_shape(k, n, groupsize=128, extension_available=available) is expected
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/python/unit/test_expert_variant_conversion.py tests/python/unit/test_glm_fp8_native_dequant.py tests/test_mxfp4_kernel.py`

Expected: fails because `quant_fp8_blockwise` and `marlin_supports_shape` do not exist.

- [ ] **Step 3: Implement exact converters without changing validated kernels**

Implement FP8 conversion as deterministic 128-by-128 blocks:

```python
def quant_fp8_blockwise(
    weight: torch.Tensor,
    block_size: int = FP8_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError("FP8 blockwise expert weights must be 2D")
    rows, cols = weight.shape
    padded_rows = ((rows + block_size - 1) // block_size) * block_size
    padded_cols = ((cols + block_size - 1) // block_size) * block_size
    padded = torch.zeros((padded_rows, padded_cols), dtype=torch.float32, device=weight.device)
    padded[:rows, :cols] = weight.float()
    blocks = padded.view(padded_rows // block_size, block_size, padded_cols // block_size, block_size)
    max_abs = blocks.abs().amax(dim=(1, 3))
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scales = (max_abs / fp8_max).clamp_min(torch.finfo(torch.float32).tiny)
    expanded = scales.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    quantized = (padded / expanded).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return quantized[:rows, :cols].contiguous(), scales.contiguous()
```

Add a pure predicate to `marlin_gemm.py`; do not alter `marlin_quantize`, `pack_marlin_weight`, or `marlin_gemm`:

```python
def marlin_supports_shape(
    k: int,
    n: int,
    groupsize: int,
    extension_available: bool | None = None,
) -> bool:
    available = _MARLIN_AVAILABLE if extension_available is None else extension_available
    return bool(available and groupsize in (-1, 128) and k % 128 == 0 and n % 256 == 0)
```

Conversion code always returns new tensors and never mutates source weights. FP8 scales are derivative-owned tensors listed next to their weight in the manifest. MXFP4 and GLM scale behavior stays unchanged.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `pytest -q tests/python/unit/test_expert_variant_conversion.py tests/python/unit/test_glm_fp8_native_dequant.py tests/test_mxfp4_kernel.py`

Expected: all CPU tests pass; CUDA tests pass when available or report their existing explicit skips.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/utils/fp8.py moe_infinity/kernel/marlin_gemm.py tests/python/unit/test_expert_variant_conversion.py tests/python/unit/test_glm_fp8_native_dequant.py tests/test_mxfp4_kernel.py
git commit -m "feat: add explicit expert variant converters"
```

## Task 3: Add crash-safe derivative generations, journals, and released manifest validation

**Files:**
- Create: `moe_infinity/runtime/expert_variant_manifest.py`
- Create: `moe_infinity/runtime/expert_variant_build.py`
- Test: `tests/python/unit/test_expert_variant_manifest.py`
- Test: `tests/python/unit/test_expert_variant_build.py`
- Modify: `moe_infinity/runtime/model_offload.py:354-441`
- Modify: `core/aio/archer_tensor_handle.h:17-60`
- Modify: `core/aio/archer_tensor_handle.cpp:23-221`
- Modify: `core/prefetch/archer_prefetch_handle.h:13-65`
- Modify: `core/prefetch/archer_prefetch_handle.cpp:23-84,446-458`
- Modify: `core/python/py_archer_prefetch.cpp:18-100`

- [x] **Step 1: Write failing manifest approval and build-recovery tests**

```python
import hashlib
import json

import pytest

from moe_infinity.runtime.adaptive_precision_allowlist import ReleasedAdaptiveEntry
from moe_infinity.runtime.expert_precision import ExecutionKind, ExpertFormat, ExpertVariantSpec
from moe_infinity.runtime.expert_variant_build import DerivativeBuildJournal, recover_or_reserve_generation
from moe_infinity.runtime.expert_variant_manifest import (
    ExpertVariantManifest,
    canonical_json_bytes,
    compute_checkpoint_fingerprint,
    load_derivative_overlay,
    write_derivative_index,
)


class RecordingOverlayHandle:
    def __init__(self, fail_tensor_id=None):
        self.calls = []
        self.fail_tensor_id = fail_tensor_id

    def begin_derivative_overlay(self, generation, canonical_max_tensor_id, canonical_max_file_id):
        self.calls.append(("begin", generation, canonical_max_tensor_id, canonical_max_file_id))

    def register_derivative_tensor(self, generation, tensor_id, file_id, offset, size, shape, dtype):
        if tensor_id == self.fail_tensor_id:
            raise RuntimeError("injected registration failure")
        self.calls.append(("register", generation, tensor_id, file_id, offset, size, shape, dtype))

    def commit_derivative_overlay(self, generation):
        self.calls.append(("commit", generation))

    def abort_derivative_overlay(self, generation):
        self.calls.append(("abort", generation))


@pytest.fixture
def valid_derivative_index(tmp_path):
    path = tmp_path / "derivative-index.v1.json"
    write_derivative_index(
        path,
        "g00000001",
        100,
        3,
        [
            {"tensor_id": 101, "file_id": 4, "offset": 0, "size": 2, "shape": [1], "dtype": "bfloat16", "sha256": "d" * 64},
            {"tensor_id": 102, "file_id": 4, "offset": 4096, "size": 4, "shape": [1], "dtype": "float32", "sha256": "c" * 64},
        ],
    )
    return path


def _variant():
    return ExpertVariantSpec(
        layer_id=1,
        expert_id=2,
        format=ExpertFormat.FP8_E4M3_BLOCK128,
        execution=ExecutionKind.FP8_DEQUANT_BF16_GEMM,
        tensor_ids=(101, 102, 103, 104, 105, 106),
        tensor_roles=("gate.weight", "gate.scale", "up.weight", "up.scale", "down.weight", "down.scale"),
        payload_bytes=600,
        aligned_bytes=24576,
        workspace_bytes=1200,
        source_format=ExpertFormat.BF16,
        converter_version="adaptive-expert-v1",
        quality_attestation_sha256="b" * 64,
    )


def test_serving_validation_requires_exact_released_entry(tmp_path):
    generation = tmp_path / "adaptive_derivatives" / "generations" / "g00000001"
    generation.mkdir(parents=True)
    index = generation / "derivative-index.v1.json"
    index.write_bytes(b"index")
    manifest = ExpertVariantManifest.create(
        model_name="model",
        checkpoint_fingerprint="a" * 64,
        generation="g00000001",
        canonical_max_tensor_id=100,
        canonical_max_file_id=3,
        derivative_index_sha256=hashlib.sha256(b"index").hexdigest(),
        variants=[_variant()],
    )
    manifest.write_atomic(generation / "manifest.v1.json")
    released = ReleasedAdaptiveEntry("a" * 64, ExpertFormat.FP8_E4M3_BLOCK128, "adaptive-expert-v1", "b" * 64)
    loaded = ExpertVariantManifest.load_for_serving(generation, "a" * 64, frozenset({released}))
    assert loaded.variants == (_variant(),)
    with pytest.raises(ValueError, match="manifest is not released"):
        ExpertVariantManifest.load_for_serving(generation, "a" * 64, frozenset())


def test_crash_retry_never_collides_with_canonical_ids(tmp_path):
    first = recover_or_reserve_generation(tmp_path, canonical_max_tensor_id=100, canonical_max_file_id=3, tensor_count=6, file_count=2)
    assert first.tensor_id_range == (101, 106)
    assert first.file_id_range == (4, 5)
    first.mark_writing({})
    first.generation_dir.joinpath("derivative-index.v1.json.tmp").write_bytes(b"partial")
    second = recover_or_reserve_generation(tmp_path, canonical_max_tensor_id=100, canonical_max_file_id=3, tensor_count=6, file_count=2)
    assert second.generation == first.generation
    assert second.tensor_id_range == first.tensor_id_range
    second.abort_corrupt_generation()
    third = recover_or_reserve_generation(tmp_path, canonical_max_tensor_id=100, canonical_max_file_id=3, tensor_count=6, file_count=2)
    assert third.tensor_id_range[0] > second.tensor_id_range[1]
    assert third.file_id_range[0] > second.file_id_range[1]


def test_current_is_published_only_after_index_and_manifest_are_durable(tmp_path):
    journal = DerivativeBuildJournal.reserve(tmp_path, 100, 3, 6, 2, "adaptive-expert-v1")
    with pytest.raises(RuntimeError, match="generation is not attested"):
        journal.publish_current()
    assert not (tmp_path / "adaptive_derivatives" / "CURRENT").exists()


def test_derivative_index_is_canonical_json_and_registers_explicit_records(tmp_path):
    records = [
        {
            "tensor_id": 102, "file_id": 4, "offset": 4096, "size": 4,
            "shape": [1], "dtype": "float32", "sha256": "c" * 64,
        },
        {
            "tensor_id": 101, "file_id": 4, "offset": 0, "size": 2,
            "shape": [1], "dtype": "bfloat16", "sha256": "d" * 64,
        },
    ]
    path = tmp_path / "derivative-index.v1.json"
    write_derivative_index(path, "g00000001", 100, 3, records)
    expected = {
        "schema_version": 1,
        "generation": "g00000001",
        "canonical_max_tensor_id": 100,
        "canonical_max_file_id": 3,
        "records": sorted(records, key=lambda row: row["tensor_id"]),
    }
    assert path.read_bytes() == (
        json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
    )
    native = RecordingOverlayHandle()
    load_derivative_overlay(path, native)
    assert native.calls == [
        ("begin", "g00000001", 100, 3),
        ("register", "g00000001", 101, 4, 0, 2, [1], "bfloat16"),
        ("register", "g00000001", 102, 4, 4096, 4, [1], "float32"),
        ("commit", "g00000001"),
    ]


def test_manifest_digest_mismatch_registers_nothing(tmp_path):
    root = tmp_path / "adaptive_derivatives"
    generation = root / "generations" / "g00000001"
    generation.mkdir(parents=True)
    index_bytes = b"{}\n"
    attestation_bytes = b"{}\n"
    manifest_bytes = b"{}\n"
    (generation / "derivative-index.v1.json").write_bytes(index_bytes)
    (generation / "quality-attestation.v1.json").write_bytes(attestation_bytes)
    (generation / "manifest.v1.json").write_bytes(manifest_bytes)
    (root / "CURRENT").write_bytes(canonical_json_bytes({
        "schema_version": 1,
        "generation": "g00000001",
        "derivative_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "quality_attestation_sha256": hashlib.sha256(attestation_bytes).hexdigest(),
        "manifest_sha256": "0" * 64,
    }))
    native = RecordingOverlayHandle()
    with pytest.raises(ValueError, match="manifest_digest_mismatch"):
        ExpertVariantManifest.load_current(root, native_handle=native)
    assert native.calls == []


def test_overlay_registration_failure_aborts_without_commit(valid_derivative_index):
    native = RecordingOverlayHandle(fail_tensor_id=102)
    with pytest.raises(RuntimeError, match="injected registration failure"):
        load_derivative_overlay(valid_derivative_index, native)
    assert native.calls[-1] == ("abort", "g00000001")
    assert all(call[0] != "commit" for call in native.calls)


def test_checkpoint_fingerprint_uses_canonical_snapshot_binding(tmp_path):
    class SnapshotHandle:
        def __init__(self):
            self.calls = 0

        def get_canonical_tensor_index_snapshot(self):
            self.calls += 1
            return [
                {"tensor_id": 2, "dtype": "float32", "shape": [1], "size": 4, "file_id": 0, "offset": 4096},
                {"tensor_id": 1, "dtype": "bfloat16", "shape": [1], "size": 2, "file_id": 0, "offset": 0},
            ]

    (tmp_path / "archer_param_0").write_bytes(b"partition")
    native = SnapshotHandle()
    one = compute_checkpoint_fingerprint(native, {"model_type": "qwen3_moe"}, tmp_path)
    two = compute_checkpoint_fingerprint(native, {"model_type": "qwen3_moe"}, tmp_path)
    assert native.calls == 2
    assert one == two
    assert one["canonical_max_tensor_id"] == 2
    assert one["canonical_max_file_id"] == 0
    assert set(one) == {
        "schema_version", "checkpoint_fingerprint", "tensor_index_sha256",
        "model_signature_sha256", "canonical_max_tensor_id", "canonical_max_file_id", "partitions",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update({"unknown": 1}),
        lambda doc: doc["records"][0].update({"dtype": "6"}),
        lambda doc: doc["records"][0].update({"size": 3}),
        lambda doc: doc["records"].append({**doc["records"][0], "tensor_id": 103}),
    ],
)
def test_derivative_index_schema_rejects_unknown_dtype_size_and_overlap(valid_derivative_index, mutate):
    document = json.loads(valid_derivative_index.read_text())
    mutate(document)
    valid_derivative_index.write_bytes(canonical_json_bytes(document))
    native = RecordingOverlayHandle()
    with pytest.raises(ValueError):
        load_derivative_overlay(valid_derivative_index, native)
    assert native.calls == []


def test_publication_order_puts_current_last(tmp_path, monkeypatch):
    import moe_infinity.runtime.expert_variant_build as build

    events = []
    monkeypatch.setattr(build, "_durable_replace_bytes", lambda path, data: events.append(("replace", path.name)))
    monkeypatch.setattr(build, "_fsync_directory", lambda path: events.append(("fsync_dir", path.name)))
    journal = DerivativeBuildJournal.reserve(tmp_path, 100, 3, 1, 1, "adaptive-expert-v1")
    journal.publish_attested_generation(
        payloads={4: b"payload"},
        derivative_index={"schema_version": 1},
        quality_attestation={"schema_version": 1},
        manifest={"schema_version": 1},
        current={"schema_version": 1},
    )
    assert events == [
        ("replace", "archer_param_4"),
        ("fsync_dir", tmp_path.name),
        ("replace", "derivative-index.v1.json"),
        ("replace", "quality-attestation.v1.json"),
        ("replace", "manifest.v1.json"),
        ("fsync_dir", "g00000001"),
        ("replace", "CURRENT"),
        ("fsync_dir", "adaptive_derivatives"),
    ]


def test_durable_replace_fsyncs_temp_before_replace(tmp_path, monkeypatch):
    import moe_infinity.runtime.expert_variant_build as build

    events = []
    monkeypatch.setattr(build.os, "fsync", lambda fd: events.append(("fsync", fd)))
    monkeypatch.setattr(build.os, "replace", lambda src, dst: events.append(("replace", src.name, dst.name)))
    build._durable_replace_bytes(tmp_path / "manifest.v1.json", b"{}\n")
    assert events[0][0] == "fsync"
    assert events[1] == ("replace", "manifest.v1.json.tmp", "manifest.v1.json")
```

`RecordingOverlayHandle` deliberately implements only the four mutating pybind-shaped methods used by the Python loader, so these tests fail if loading silently falls back to a C++ path parser.

- [x] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/python/unit/test_expert_variant_manifest.py tests/python/unit/test_expert_variant_build.py`

Expected: collection fails because manifest approval and derivative build modules do not exist.

- [x] **Step 3: Implement exact JSON schemas, digest chain, and released-entry validation**

Implement `canonical_json_bytes`, `compute_checkpoint_fingerprint(native_handle, model_signature, offload_path)`, `create`, `to_dict`, `from_dict`, `write_atomic`, `write_derivative_index`, `load_derivative_overlay`, `load_for_serving`, and `load_current`. Enforce the strict fingerprint, `CURRENT`, derivative-index, quality-attestation, and manifest schemas in the data-contract section; reject unknown keys as well as missing keys. Hash raw canonical file bytes before JSON parsing, validate the `CURRENT` manifest/index/attestation digest chain, then validate duplicate `(layer_id, expert_id, format)`, role/ID mismatch, derivative IDs at or below `canonical_max_tensor_id`, duplicate IDs, file-interval overlap, payload checksum/size/alignment errors, unknown enums/dtypes, wrong recomputed checkpoint fingerprint, converter mismatch, incomplete state, failed attestation, and any variant whose exact four-field `ReleasedAdaptiveEntry` is absent. A manifest containing three formats requires three released entries pointing to the same validated generation-attestation digest; partial approval rejects the entire generation rather than silently dropping variants. Do not invoke a native overlay method until all Python validation and payload hashing succeeds.

- [x] **Step 4: Implement durable generation reservation and explicit native overlay registration**

`DerivativeBuildJournal` writes canonical JSON with `flush`, `os.fsync`, atomic replace, and parent-directory `fsync` at every state transition. Implement `_durable_replace_bytes(path, data)` as open sibling `path.name + ".tmp"` in binary truncate mode → write all bytes → `flush` → `os.fsync(file.fileno())` → close → `os.replace(tmp, path)`, and `_fsync_directory(path)` as `os.open(path, os.O_RDONLY | os.O_DIRECTORY)` → `os.fsync(fd)` → close. `publish_attested_generation` uses only these helpers and the exact event order asserted above. Before reservation, call `get_canonical_tensor_index_snapshot()`, validate its exact row schema/order, hash canonical partitions in Python, and write `canonical-checkpoint-fingerprint.v1.json`. Reserve derivative tensor/file ranges strictly above persisted canonical and prior-generation high-water marks. Write derivative payloads and all four metadata files with the exact filenames and durability order in the data contract. No file named `derivative-index.v1` is created.

Extend `ArcherTensorHandle` with `GetCanonicalTensorIndexSnapshot`, `BeginDerivativeOverlay`, `RegisterDerivativeTensor`, `CommitDerivativeOverlay`, and `AbortDerivativeOverlay`; expose thin delegates with the exact snake-case names from the data contract on `ArcherPrefetchHandle` in `py_archer_prefetch.cpp`. `GetCanonicalTensorIndexSnapshot` returns sorted dictionaries with stable string dtypes. `RegisterDerivativeTensor` receives already-parsed scalar fields and never receives a path or JSON string. The transaction stages metadata and merges it only on commit after native range/alignment/size/disjointness checks. The root canonical `archer_index`, canonical partition files, `name_id_map.json`, and model signature remain unchanged. If any `CURRENT`, index, attestation, manifest, payload, release, or native registration validation fails, call `abort_derivative_overlay` and load canonical metadata only.

- [x] **Step 5: Run the tests and verify GREEN**

Run: `pytest -q tests/python/unit/test_expert_variant_manifest.py tests/python/unit/test_expert_variant_build.py`

Expected: all tests pass, including simulated crashes at every journal state.

- [x] **Step 6: Commit**

```bash
git add moe_infinity/runtime/expert_variant_manifest.py moe_infinity/runtime/expert_variant_build.py moe_infinity/runtime/model_offload.py core/aio/archer_tensor_handle.h core/aio/archer_tensor_handle.cpp core/prefetch/archer_prefetch_handle.h core/prefetch/archer_prefetch_handle.cpp core/python/py_archer_prefetch.cpp tests/python/unit/test_expert_variant_manifest.py tests/python/unit/test_expert_variant_build.py
git commit -m "feat: publish crash-safe derivative generations"
```

## Task 4: Build the deterministic fixed-budget policy simulator

**Files:**
- Create: `moe_infinity/memory/adaptive_precision_policy.py`
- Test: `tests/python/unit/test_adaptive_precision_policy.py`

- [x] **Step 1: Write failing deterministic policy tests**

```python
from moe_infinity.memory.adaptive_precision_policy import AdaptivePrecisionPolicy, ExpertKey
from moe_infinity.runtime.expert_precision import ExpertFormat, ResidentGeneration


def _variants():
    return {
        ExpertKey(0, 0): {ExpertFormat.FP8_E4M3_BLOCK128: 100, ExpertFormat.BF16: 200},
        ExpertKey(0, 1): {ExpertFormat.FP8_E4M3_BLOCK128: 100, ExpertFormat.BF16: 200},
        ExpertKey(0, 2): {ExpertFormat.FP8_E4M3_BLOCK128: 100, ExpertFormat.BF16: 200},
    }


def test_policy_is_deterministic_and_uses_lexical_tie_break():
    left = AdaptivePrecisionPolicy(300, 1.0, 0.7, 0.3, 0, 0, _variants())
    right = AdaptivePrecisionPolicy(300, 1.0, 0.7, 0.3, 0, 0, _variants())
    resident = {
        ExpertKey(0, 0): ResidentGeneration(ExpertFormat.FP8_E4M3_BLOCK128, 100, 1),
        ExpertKey(0, 1): ResidentGeneration(ExpertFormat.FP8_E4M3_BLOCK128, 100, 1),
    }
    for policy in (left, right):
        policy.observe({ExpertKey(0, 0): 10, ExpertKey(0, 1): 10, ExpertKey(0, 2): 1}, tokens=10)
    left_plan = left.plan(resident, set(), 0, 0)
    right_plan = right.plan(resident, set(), 0, 0)
    assert left_plan == right_plan
    assert left_plan.targets[ExpertKey(0, 0)] is ExpertFormat.BF16
    assert left_plan.targets[ExpertKey(0, 1)] is ExpertFormat.FP8_E4M3_BLOCK128
    assert left_plan.accounted_bytes == 300


def test_host_ready_catalog_is_not_charged_to_hbm():
    catalog = {
        ExpertKey(0, expert_id): {ExpertFormat.FP8_E4M3_BLOCK128: 100, ExpertFormat.BF16: 200}
        for expert_id in range(1000)
    }
    policy = AdaptivePrecisionPolicy(200, 1.0, 0.7, 0.3, 0, 0, catalog)
    resident = {ExpertKey(0, 0): ResidentGeneration(ExpertFormat.FP8_E4M3_BLOCK128, 100, 1)}
    plan = policy.plan(resident, set(), 0, 0)
    assert plan.accounted_bytes == 100
    assert ExpertKey(0, 999) not in plan.targets


def test_policy_never_exceeds_budget_during_transition():
    policy = AdaptivePrecisionPolicy(500, 1.0, 0.7, 0.3, 0, 0, _variants())
    policy.observe({ExpertKey(0, 0): 10}, tokens=10)
    resident = {ExpertKey(0, 0): ResidentGeneration(ExpertFormat.BF16, 200, 1)}
    plan = policy.plan(resident, {ExpertKey(0, 1)}, transition_reserved_bytes=100, workspace_bytes=100)
    assert plan.accounted_bytes <= 500
    assert sum(intent.reserve_bytes for intent in plan.transitions) <= 100


def test_hysteresis_prevents_epoch_to_epoch_flapping():
    policy = AdaptivePrecisionPolicy(500, 0.5, 0.7, 0.3, 2, 2, _variants())
    policy.observe({ExpertKey(0, 0): 10}, tokens=10)
    resident = {ExpertKey(0, 0): ResidentGeneration(ExpertFormat.FP8_E4M3_BLOCK128, 100, 1)}
    first = policy.plan(resident, set(), 0, 0)
    policy.commit(first)
    policy.observe({ExpertKey(0, 1): 11}, tokens=10)
    second = policy.plan(resident, {ExpertKey(0, 1)}, 0, 0)
    assert not second.transitions


def test_simulation_replay_is_byte_for_byte_reproducible():
    trace = [
        {ExpertKey(0, 0): 4, ExpertKey(0, 1): 1},
        {ExpertKey(0, 1): 5},
        {ExpertKey(0, 2): 8},
    ]
    one = AdaptivePrecisionPolicy.simulate(trace, _variants(), budget_bytes=400)
    two = AdaptivePrecisionPolicy.simulate(trace, _variants(), budget_bytes=400)
    assert one.to_json() == two.to_json()
```

- [x] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/python/unit/test_adaptive_precision_policy.py`

Expected: collection fails because the policy module does not exist.

- [x] **Step 3: Implement deterministic selection**

Use integer routed-token counts as observations. At each policy epoch:

1. Update `hotness[key] = decay * old + (1 - decay) * count / max(tokens, 1)`.
2. Normalize by the largest hotness; zero observations normalize to zero.
3. Seed targets only from the native `resident_generation_entries` snapshot. Host-ready catalog entries contribute zero bytes and receive no target until they appear in `admission_candidates` from current routing or approved prefetch.
4. For each admission candidate, choose its smallest representation in the caller-supplied eligible catalog that fits; the serving catalog contains released/capable formats only, while the isolated candidate-evaluation harness may supply unreleased candidates. If none fits, leave it host-ready and let the existing on-demand canonical path run without caching it.
5. Enumerate one-rank upgrades for resident/admitted experts with `marginal_utility = normalized_hotness * quality_rank_delta` and order by `(-marginal_utility / extra_bytes, -marginal_utility, extra_bytes, layer_id, expert_id, format)` using `fractions.Fraction`, not floating division.
6. Account exactly `sum(active_and_retiring_resident_generation_bytes) + transition_reserved_bytes + workspace_bytes`; never add a base representation for nonresident experts. Promotion requires normalized hotness at or above the promotion threshold. Demotion requires hotness at or below the demotion threshold and both residency/cooldown constraints.
7. Emit immutable `PrecisionPlan(epoch, targets, transitions, evictions, accounted_bytes)`. `plan()` has no side effects; only `commit(plan)` advances generations and cooldowns.

If existing resident/retiring bytes already exceed the budget, emit no admission/promotion and request eviction only for idle, unleased generations in deterministic order. Never evict an active lease to force the accounting result. `simulate()` uses no CUDA, wall clock, random number, unordered set iteration, or platform-dependent hash ordering.

- [x] **Step 4: Run the test and verify GREEN**

Run: `pytest -q tests/python/unit/test_adaptive_precision_policy.py`

Expected: all tests pass.

- [x] **Step 5: Commit**

```bash
git add moe_infinity/memory/adaptive_precision_policy.py tests/python/unit/test_adaptive_precision_policy.py
git commit -m "feat: add deterministic mixed precision cache policy"
```

## Task 5: Extend the Task 0 ExpertResidencyManager for precision variants

**Files:**
- Modify: `core/prefetch/expert_residency.h`
- Modify: `core/prefetch/expert_residency.cpp`
- Modify: `core/model/model_topology.h:40-82,215-241`
- Modify: `core/model/model_topology.cpp:59-241,525-729`
- Modify: `core/prefetch/archer_prefetch_handle.h:20-73`
- Modify: `core/prefetch/archer_prefetch_handle.cpp:239-315,447-481`
- Modify: `tests/cpp/unit/prefetch/expert_residency_test_fixture.h`
- Create: `tests/cpp/unit/prefetch/test_expert_residency_variants.cpp`
- Modify: `tests/cpp/unit/prefetch/CMakeLists.txt`
- Modify: `tests/python/unit/test_setup_sources.py`

- [ ] **Step 1: Write failing real-manager variant transaction tests**

Add `test_expert_residency_variants.cpp` using Task 0's real `ExpertResidencyManagerTest` fixture. Cover these exact cases:

```cpp
TEST_F(ExpertResidencyManagerTest, HostReadyVariantsConsumeNoCapacity) {
  auto fp8 = MakeVariant(0, 1, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 12);
  manager->RegisterVariant(fp8);
  EXPECT_EQ(manager->ResidentBytes(0), 0);
  EXPECT_EQ(manager->Snapshot().at("registered_variants"), 1);
}

TEST_F(ExpertResidencyManagerTest, TransitionReservesDestinationAndWorkspace) {
  auto low = MakeVariant(0, 1, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 12);
  auto high = MakeVariant(0, 1, ExpertFormat::BF16, 2, 80, 16);
  CommitResident(low, ExpertPhase::DECODE);
  auto tx = manager->BeginTransition(low.key, high, 0, ExpertPhase::DECODE,
                                     AdmissionSource::DEMAND);
  ASSERT_TRUE(tx.valid);
  auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("resident_bytes"), 40);
  EXPECT_EQ(snapshot.at("transition_reserved_bytes"), 80);
  EXPECT_EQ(snapshot.at("workspace_bytes"), 16);
  EXPECT_LE(40 + 80 + 16, snapshot.at("capacity_bytes"));
  EXPECT_TRUE(manager->AbortTransaction(tx));
  EXPECT_EQ(manager->Snapshot().at("workspace_bytes"), 0);
}

TEST_F(ExpertResidencyManagerTest, CommitTransitionRetiresOldGeneration) {
  auto low = MakeVariant(0, 1, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 0);
  auto high = MakeVariant(0, 1, ExpertFormat::BF16, 2, 80, 0);
  CommitResident(low, ExpertPhase::PREFILL);
  auto lease = manager->AcquireLease(low.key, LeaseKind::EXECUTION);
  auto tx = manager->BeginTransition(low.key, high, 0, ExpertPhase::DECODE,
                                     AdmissionSource::DEMAND);
  ASSERT_TRUE(manager->CommitTransaction(tx));
  EXPECT_EQ(manager->Snapshot().at("retiring_generations"), 1);
  EXPECT_EQ(manager->ReapRetired(0), 0);
  EXPECT_TRUE(manager->ReleaseLease(lease));
  EXPECT_EQ(manager->ReapRetired(0), 1);
}

TEST_F(ExpertResidencyManagerTest, DemandAndPrefetchClientsShareVariantCharge) {
  auto variant = MakeVariant(0, 2, ExpertFormat::BF16, 1, 64, 0);
  auto prefetch_tx = prefetch_client.BeginAdmission(
      variant, 0, ExpertPhase::PREFILL, AdmissionMode::CACHE);
  ASSERT_TRUE(manager->CommitTransaction(prefetch_tx));
  auto demand_tx = demand_client.BeginAdmission(
      variant, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_EQ(demand_tx.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_EQ(manager->ResidentBytes(0), 64);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest, ConcurrentTransitionForLogicalExpertIsRejected) {
  auto low = MakeVariant(0, 3, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 0);
  auto high = MakeVariant(0, 3, ExpertFormat::BF16, 2, 80, 0);
  CommitResident(low, ExpertPhase::DECODE);
  auto first = manager->BeginTransition(low.key, high, 0, ExpertPhase::DECODE,
                                        AdmissionSource::DEMAND);
  auto second = manager->BeginTransition(low.key, high, 0, ExpertPhase::DECODE,
                                         AdmissionSource::PREFETCH);
  EXPECT_TRUE(first.valid);
  EXPECT_FALSE(second.valid);
  EXPECT_TRUE(manager->AbortTransaction(first));
}

TEST_F(ExpertResidencyManagerTest, WorkspaceRemainsChargedUntilCompletion) {
  auto variant = MakeVariant(0, 4, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 16);
  auto tx = demand_client.BeginAdmission(
      variant, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(manager->CommitTransaction(tx));
  EXPECT_EQ(manager->Snapshot().at("workspace_bytes"), 16);
  ASSERT_TRUE(manager->RecordWorkspaceUse(tx.id, nullptr));
  EXPECT_EQ(manager->ReapWorkspace(0), 1);
  EXPECT_EQ(manager->Snapshot().at("workspace_bytes"), 0);
}
```

The fixture adds `MakeVariant` and `CommitResident` using real `Node` objects and the existing fake transfer operations. It does not construct another manager.

- [ ] **Step 2: Build the manager test and verify RED**

Run: `cmake --build build --target test_expert_residency_variants -j2`

Expected: compile fails because variant keys, transition transactions, workspace reservation, retirement, and `RegisterVariant` do not exist.

- [ ] **Step 3: Extend the existing manager's records and transaction API**

In `core/prefetch/expert_residency.h`, extend rather than replace the exact Task 0 API coordinated with the phase-policy plan:

```cpp
enum class ExpertFormat : std::uint8_t {
  BF16 = 0,
  FP8_E4M3_BLOCK128 = 1,
  MARLIN_INT4_GROUP128 = 2,
  GPT_OSS_MXFP4 = 3,
  GLM_FP8_BLOCK128 = 4,
  DEEPSEEK_V4_FP4 = 5,
  GPTQ = 6,
  AWQ = 7,
};

struct ResidencyVariantKey {
  std::uint64_t logical_expert_key = 0;
  ExpertFormat format = ExpertFormat::BF16;
  std::uint64_t generation = 0;
  bool operator==(const ResidencyVariantKey& other) const {
    return logical_expert_key == other.logical_expert_key &&
           format == other.format && generation == other.generation;
  }
  bool operator<(const ResidencyVariantKey& other) const {
    return std::tie(logical_expert_key, format, generation) <
           std::tie(other.logical_expert_key, other.format, other.generation);
  }
};

struct ResidencyVariant {
  ResidencyVariantKey key;
  NodePtr node;
  std::int64_t payload_bytes = 0;
  std::int64_t aligned_bytes = 0;
  std::int64_t workspace_bytes = 0;
};

enum class ResidencyState : std::uint8_t {
  HOST_READY = 0,
  ACTIVE = 1,
  RETIRING = 2,
};

struct ResidencyEntry {
  NodePtr node;
  std::int64_t bytes = 0;
  ResidencyVariantKey key;
  std::int64_t payload_bytes = 0;
  ResidencyState state = ResidencyState::HOST_READY;
  std::uint32_t lease_count = 0;
  cudaEvent_t last_use_event = nullptr;
};

enum class LeaseKind : std::uint8_t {
  DEMAND = 0,
  PREFETCH = 1,
  TRANSFER = 2,
  EXECUTION = 3,
  TRANSITION = 4,
};

struct LeaseRecord {
  std::uint64_t id = 0;
  NodePtr node;
  ResidencyVariantKey key;
  LeaseKind kind = LeaseKind::DEMAND;
};

struct WorkspaceRecord {
  std::uint64_t transaction_id = 0;
  ResidencyVariantKey key;
  int gpu_id = -1;
  std::int64_t bytes = 0;
  cudaEvent_t completion_event = nullptr;
};

enum class ResidencyTransactionKind : std::uint8_t {
  ADMISSION = 0,
  TRANSITION = 1,
};

struct ResidencyTicket {
  std::uint64_t id = 0;
  ResidencyTransactionKind kind = ResidencyTransactionKind::ADMISSION;
  NodePtr incoming;
  NodePtr reserved_victim;
  ResidencyVariant incoming_variant;
  std::optional<ResidencyVariantKey> replaced_key;
  std::optional<ResidencyVariantKey> reserved_victim_key;
  int gpu_id = -1;
  ExpertPhase phase = ExpertPhase::MIXED;
  AdmissionSource source = AdmissionSource::DEMAND;
  AdmissionOutcome outcome = AdmissionOutcome::REJECTED;
  std::int64_t reserved_aligned_bytes = 0;
  std::int64_t reserved_workspace_bytes = 0;
  bool transient = false;
  bool valid = false;
};
```

`logical_expert_key` remains the coordinated shared key `(uint64_t(layer_id) << 32) | uint32_t(expert_id)`. Phase is request metadata and is not added to identity; adaptive identity extends only with `(format, generation)`.

Extend `ExpertResidencyManager` and `ExpertResidencyClient` with:

```cpp
void RegisterVariant(const ResidencyVariant& variant);
ResidencyTicket BeginAdmission(const ResidencyVariant& incoming,
                               int gpu_id, ExpertPhase phase,
                               AdmissionMode mode,
                               AdmissionSource source);
ResidencyTicket BeginTransition(const ResidencyVariantKey& current,
                                const ResidencyVariant& incoming,
                                int gpu_id, ExpertPhase phase,
                                AdmissionSource source);
bool EvictReserved(const ResidencyTicket& transaction);
bool CommitTransaction(const ResidencyTicket& transaction);
bool AbortTransaction(const ResidencyTicket& transaction);
std::uint64_t AcquireLease(const ResidencyVariantKey& key, LeaseKind kind);
bool ReleaseLease(std::uint64_t lease_id);
bool RecordLastUse(const ResidencyVariantKey& key, cudaEvent_t event);
bool RecordWorkspaceUse(std::uint64_t transaction_id, cudaEvent_t event);
std::size_t ReapWorkspace(int gpu_id);
bool RequestRetirement(const ResidencyVariantKey& key);
std::size_t ReapRetired(int gpu_id);
std::vector<ResidencyEntry> ResidentGenerations(int gpu_id) const;
std::size_t ResidentGenerationCount(int gpu_id) const;
std::optional<ResidencyVariantKey> ActiveGeneration(
    int gpu_id, std::uint64_t logical_expert_key) const;
void ReplaceProtectedVariants(
    const std::vector<ResidencyVariantKey>& candidates);
```

Add matching client overloads:

```cpp
ResidencyTicket BeginAdmission(const ResidencyVariant& incoming, int gpu_id,
                               ExpertPhase phase, AdmissionMode mode) {
  return manager_->BeginAdmission(incoming, gpu_id, phase, mode, source_);
}
ResidencyTicket BeginTransition(const ResidencyVariantKey& current,
                                const ResidencyVariant& incoming, int gpu_id,
                                ExpertPhase phase) {
  return manager_->BeginTransition(current, incoming, gpu_id, phase, source_);
}
```

Expose `AcquireLease`, `ReleaseLease`, `RecordLastUse`, `RecordWorkspaceUse`, `ReapWorkspace`, `RequestRetirement`, `ReapRetired`, `Snapshot`, and `manager()` only as thin manager delegates. Clients store no pending transaction, lease, member, or byte state beyond IDs needed to finish their current operation.

Include `<optional>` and `<tuple>`. Change the manager's per-GPU resident map key from logical `uint64_t` to `ResidencyVariantKey`, and add manager-owned `registered_variants_` plus `workspace_records_`; do not add parallel maps in either client. Retain Task 0's fixed-size wrappers by mapping a canonical `Node` to `(logical_expert_key, ExpertFormat::BF16, generation=0, aligned_bytes=node->byte_size, workspace_bytes=0)`. Existing fixed-size manager tests and disabled behavior therefore remain valid. The manager alone owns registered variants, resident/retiring membership, pending transactions, reservations, workspace records, leases, victim reservation, and counters. `CommitTransaction` converts aligned-byte reservations to one active charge, keeps reserved workspace charged in a `WorkspaceRecord`, and marks a replaced generation retiring; `AbortTransaction` releases all reservations/transition leases; `RecordWorkspaceUse` associates completion, and `ReapWorkspace` releases bytes only after its event completes. `ReapRetired` frees only zero-lease, event-complete retiring entries.

Keep phase telemetry semantics: `resident_experts` and `ResidentCount` count unique logical experts with an `ACTIVE` generation. Add `resident_generations`/`ResidentGenerationCount` for active plus retiring copies. `resident_bytes` includes all active and retiring generation bytes until reaped.

Preserve the shared-manager steady-state invariant as: one `ACTIVE` generation per `(gpu_id, logical_expert_key)`. A transition may additionally have one reserved incoming generation and one or more `RETIRING` generations whose leases/events have not completed; every byte remains charged until reaped. Reject a second concurrent transition for the same logical expert. Fixed-size operation uses generation 0 and therefore still has exactly one copy.

- [ ] **Step 4: Implement detached variant nodes as manager-owned records**

Add `ArcherTopologyHandle::CreateDetachedNode(const std::vector<TensorID>&, int gpu_id)` with exact payload/alignment validation, but return the node to `ExpertResidencyManager::RegisterVariant`. Detached nodes do not enter canonical pipeline stages or an independent cache container. The manager requests host/device movement through its existing injected `ResidencyTransferOps`; clients never destroy detached nodes directly.

- [ ] **Step 5: Run phase and variant manager tests together**

Run:

```bash
cmake --build build --target test_expert_residency test_expert_residency_variants -j2
ctest --test-dir build -R 'ExpertResidency(Variant)?Tests' --output-on-failure
pytest -q tests/python/unit/test_setup_sources.py tests/python/unit/test_native_phase_policy_wire.py
```

Expected: Task 0's original fixed-size manager tests and new variant tests pass. Source tests report `core/prefetch/expert_residency.cpp` exactly once and reject every additional residency-authority source.

- [ ] **Step 6: Commit**

```bash
git add core/prefetch/expert_residency.h core/prefetch/expert_residency.cpp core/model/model_topology.h core/model/model_topology.cpp core/prefetch/archer_prefetch_handle.h core/prefetch/archer_prefetch_handle.cpp tests/cpp/unit/prefetch/expert_residency_test_fixture.h tests/cpp/unit/prefetch/test_expert_residency_variants.cpp tests/cpp/unit/prefetch/CMakeLists.txt tests/python/unit/test_setup_sources.py
git commit -m "feat: extend expert residency manager for precision variants"
```

## Task 6: Implement lease-safe native representation transitions and execution binding

**Files:**
- Modify: `core/parallel/expert_dispatcher.h:40-194`
- Modify: `core/parallel/expert_dispatcher.cpp:91-755`
- Modify: `core/parallel/expert_module.h:195-223`
- Modify: `core/parallel/expert_module.cpp:46-329`
- Modify: `core/prefetch/task_scheduler.h`
- Modify: `core/prefetch/task_scheduler.cpp:125-576`
- Modify: `core/prefetch/archer_prefetch_handle.h`
- Modify: `core/prefetch/archer_prefetch_handle.cpp`
- Modify: `core/python/py_archer_prefetch.cpp:109-123`
- Modify: `setup.py:153-228,298-310`
- Modify: `CMakeLists.txt:1-42`
- Modify: `core/CMakeLists.txt:1-83`
- Modify: `extensions/CMakeLists.txt:8-31`
- Test: `tests/python/ops/test_adaptive_expert_dispatch.py`
- Test: `tests/python/integration/test_phase_adaptive_residency.py`

- [ ] **Step 1: Add failing lifecycle, budget, and rollback tests**

```python
@pytest.mark.gpu
def test_dispatcher_publishes_only_ready_generation(native_adaptive_fixture):
    dispatcher = native_adaptive_fixture.dispatcher
    expert = native_adaptive_fixture.expert
    run = native_adaptive_fixture.run
    dispatcher.set_precision_targets([(0, 0, "fp8_e4m3_block128", 1)], epoch=1)
    before = dispatcher.get_precision_metrics()
    output = run(expert=expert)
    after = dispatcher.get_precision_metrics()
    assert torch.isfinite(output).all()
    assert after["published_generation"] == before["published_generation"] + 1
    assert after["active_leases"] == 0
    assert after["resident_bytes"] + after["transition_reserved_bytes"] + after["workspace_bytes"] <= after["budget_bytes"]


@pytest.mark.gpu
def test_failed_transition_keeps_canonical_generation(native_adaptive_fixture):
    dispatcher = native_adaptive_fixture.dispatcher
    expert = native_adaptive_fixture.expert
    run = native_adaptive_fixture.run
    canonical = run(expert=expert)
    dispatcher.inject_transition_failure_once_for_test(0, 0, "fp8_e4m3_block128")
    dispatcher.set_precision_targets([(0, 0, "fp8_e4m3_block128", 1)], epoch=1)
    actual = run(expert=expert)
    metrics = dispatcher.get_precision_metrics()
    assert torch.equal(actual, canonical)
    assert metrics["fallback_counts"]["transition_failed"] == 1
    assert metrics["active_format"] == "bf16"


@pytest.mark.gpu
def test_phase_and_adaptive_clients_report_one_manager(native_adaptive_fixture):
    dispatcher = native_adaptive_fixture.dispatcher
    prefetcher = native_adaptive_fixture.prefetcher
    expert = native_adaptive_fixture.expert
    run = native_adaptive_fixture.run
    assert dispatcher.get_residency_manager_id() == prefetcher.get_residency_manager_id()
    dispatcher.set_precision_targets([(0, 0, "fp8_e4m3_block128", 1)], epoch=1)
    run(expert=expert, phase="prefill")
    dispatcher.set_precision_targets([(0, 0, "bf16", 0)], epoch=2)
    run(expert=expert, phase="decode")
    dispatcher_stats = dispatcher.get_precision_metrics()
    prefetch_stats = prefetcher.get_policy_stats()
    assert dispatcher_stats["manager_instance_id"] == prefetch_stats["manager_instance_id"]
    assert dispatcher_stats["resident_bytes"] == prefetch_stats["resident_bytes"]
    assert dispatcher_stats["resident_generations"] == 1


@pytest.mark.parametrize(
    ("phase_enabled", "adaptive_enabled", "manager_enabled"),
    [(False, False, False), (True, False, True), (False, True, True), (True, True, True)],
)
def test_manager_enablement_composes_feature_flags(
    native_runtime_factory, phase_enabled, adaptive_enabled, manager_enabled
):
    runtime = native_runtime_factory(
        phase_specific_expert_policy=phase_enabled,
        adaptive_expert_precision=adaptive_enabled,
    )
    assert runtime.dispatcher.get_policy_stats()["manager_enabled"] == int(manager_enabled)
    assert runtime.dispatcher.get_residency_manager_id() == runtime.prefetcher.get_residency_manager_id()


def test_canonical_tensor_snapshot_excludes_committed_derivatives(native_adaptive_fixture):
    handle = native_adaptive_fixture.prefetcher.native_handle
    before = handle.get_canonical_tensor_index_snapshot()
    assert before == sorted(before, key=lambda row: row["tensor_id"])
    handle.begin_derivative_overlay("g00000001", max(row["tensor_id"] for row in before), max(row["file_id"] for row in before))
    handle.register_derivative_tensor("g00000001", 0xFFFFFF00, 0xFFFFFF00, 0, 2, [1], "bfloat16")
    handle.commit_derivative_overlay("g00000001")
    assert handle.get_canonical_tensor_index_snapshot() == before
```

Define `native_adaptive_fixture` as a dataclass-like object with `dispatcher`, `prefetcher`, `expert`, and `run` attributes. Its dispatcher and prefetcher are constructed by the real `_store` runtime and must expose the same nonzero `manager_instance_id`. Only transition failure injection is test-only. Lease, reservation, retirement, and shared-client correctness are tested against the real manager in Task 5 and through production snapshots here.

- [ ] **Step 2: Build and run the tests to verify RED**

Run: `python setup.py clean --all && MOE_INFINITY_TESTING=1 pip install --no-build-isolation -e . && pytest -q tests/python/ops/test_adaptive_expert_dispatch.py tests/python/integration/test_phase_adaptive_residency.py -k 'publishes or failed_transition or one_manager or composes_feature_flags'`

Expected: fails because representation registration, target setting, leases, and metrics do not exist.

- [ ] **Step 3: Wire the test-only compile definition through both build systems**

In `setup.py`, keep Task 0's existing `core/prefetch/expert_residency.cpp` entry exactly once in `_STORE_SOURCES`; add no residency source. Append `-DMOE_INFINITY_TESTING=1` to both `COMMON_CXX_ARGS` and `COMMON_NVCC_ARGS` only when `os.environ.get("MOE_INFINITY_TESTING") == "1"`. Because `_store` compiles `py_archer_prefetch.cpp`, dispatcher C++, and CUDA sources with these lists, declarations and bindings see the same definition.

In root `CMakeLists.txt`, add `option(MOE_INFINITY_TESTING "Expose native test-only bindings" OFF)`. Keep Task 0's existing `core/prefetch/expert_residency.cpp` entry exactly once in `ARCHER_CORE_CXX_SOURCES`; add no residency source. When enabled, add `MOE_INFINITY_TESTING=1` to `archer_core` and `prefetch_op` with `target_compile_definitions`; `extensions/CMakeLists.txt` must explicitly define it for `prefetch_op` because that target compiles the pybind translation unit separately. Extend `tests/python/unit/test_setup_sources.py` to collect every source path containing `residency` from `_STORE_SOURCES` and `ARCHER_CORE_CXX_SOURCES` and assert each list equals `["core/prefetch/expert_residency.cpp"]` after path normalization.

Build checks:

```bash
python setup.py clean --all
MOE_INFINITY_TESTING=1 pip install --no-build-isolation -e .
python -c 'from moe_infinity import _store; assert hasattr(_store.expert_dispatcher, "inject_transition_failure_once_for_test")'
python setup.py clean --all
env -u MOE_INFINITY_TESTING pip install --no-build-isolation -e .
python -c 'from moe_infinity import _store; assert not hasattr(_store.expert_dispatcher, "inject_transition_failure_once_for_test")'
cmake -S . -B build-testing -DMOE_INFINITY_TESTING=ON -DCUTLASS_DIR="$CUTLASS_DIR"
cmake --build build-testing --target prefetch_op -j2
```

Expected: the testing setuptools and CMake builds expose the test-only binding, while the cleaned normal setuptools build omits it.

- [ ] **Step 4: Implement representation registration and state transitions**

Add pybind methods:

```text
get_canonical_tensor_index_snapshot()
begin_derivative_overlay(generation, canonical_max_tensor_id, canonical_max_file_id)
register_derivative_tensor(generation, tensor_id, file_id, offset, size, shape, dtype)
commit_derivative_overlay(generation)
abort_derivative_overlay(generation)
register_expert_variant(layer_idx, expert_idx, format, generation, execution, tensor_ids, tensor_roles, payload_bytes, aligned_bytes, workspace_bytes)
set_precision_targets([(layer_idx, expert_idx, format, generation)], epoch)
prefetch_expert_variants([(layer_idx, expert_idx, format, generation)], priority, phase)
set_adaptive_hbm_budget_bytes(bytes)
get_precision_metrics()
get_residency_manager_id()
configure_residency_manager(manager_enabled, phase_policy_enabled)
```

Extend native `Task` with a `ResidencyVariantKey` plus only the current manager ticket/lease IDs. The scheduler client resolves a registered variant, calls `BeginAdmission`, and completes the same transaction sequence as demand; it does not maintain variant membership. When the separately coordinated phase feature is present, preserve its priority and starvation fields unchanged; without it, use Task 0's neutral `MIXED` metadata and existing scheduler priority.

Required transition sequence:

1. Reject unknown, protected, or unregistered formats, stale epochs, duplicate generations, and plans inconsistent with `ExpertResidencyManager::Snapshot`. Production registers only Python-approved released variants; candidate evaluation uses an explicit non-serving registration mode.
2. For a miss, the dispatcher or scheduler's existing `ExpertResidencyClient` calls `BeginAdmission(variant, gpu, phase, mode)`. For promotion/demotion, the demand client calls `BeginTransition(current_key, destination, gpu, phase)`; the manager atomically reserves destination aligned bytes, workspace, and any victim.
3. The client calls `EvictReserved(transaction)` when required, acquires the manager's `TRANSFER` lease, materializes on the existing nonblocking fetch stream, launches any out-of-place conversion, and records the publication event.
4. After event completion, call `CommitTransaction`; on allocation/copy/conversion/event failure call `AbortTransaction`. Both terminal calls are idempotent and release transaction reservations/leases. The old generation remains active after abort.
5. Execution obtains the current key from `ExpertResidencyManager::ActiveGeneration`, acquires its `EXECUTION` lease, records its last-use event, calls `RecordWorkspaceUse(transaction_id, completion_event)` for transaction workspace, and releases the lease after enqueueing completion. If a destination is still pending, `ActiveGeneration` continues to return the old key.
6. `CommitTransaction` marks a replaced transition generation `RETIRING`; policy-only eviction calls `RequestRetirement`. Clients may trigger `ReapWorkspace` and `ReapRetired`, but only the manager applies event/eligibility checks, releases bytes, and invokes transfer ops. Dispatcher and scheduler contain no representation victim scan or resident-byte mutation.
7. Destruction stops workers, aborts pending client transactions, releases client leases, asks the shared manager to reap retired generations, then destroys streams. The manager lifetime remains owned by Task 0's native runtime singleton.

Cache victim selection remains exclusively in `ExpertResidencyManager` and applies variant byte feasibility; when phase policy is present and enabled, it also applies the coordinated deterministic phase utility. Always-resident shared experts are excluded from adaptive transactions and reported separately.

Set `manager_enabled = phase_policy_config.enabled || adaptive_precision_enabled` once during native configuration. Dispatcher and scheduler enter manager-client paths whenever `manager_enabled`; only both-false uses legacy `cached_experts_`, `cache_sizes_`, and scheduler topology scans. When adaptive is true and phase policy false, pass `ExpertPhase::MIXED`, disable phase weighting/starvation changes, and use canonical wrapper behavior for unconverted experts. When both are true, preserve the propagated request phase and phase-policy victim utility.

- [ ] **Step 5: Bind execution descriptors to existing kernels**

Replace fixed positional assumptions with `SetRepresentation(const ExpertExecutionDescriptor&)`. This method binds tensors for a manager-leased key but owns no residency state. Preserve these exact adapters:

| Format | Stored roles | Execution |
|---|---|---|
| `bf16` | gate/up/down weights | existing fused BF16 MoE FFN |
| `fp8_e4m3_block128` | weight/scale pair for each projection | existing `fp8_dequant_blockwise_cuda`, then existing BF16 FFN |
| `marlin_int4_group128` | packed/scales pair for each projection | existing `marlin_gemm`; FP16 activations/output converted back to caller dtype |
| `gpt_oss_mxfp4` | existing six GPT-OSS roles | existing `DequantMxfp4Params` and GPT-OSS forward |
| `glm_fp8_block128` | existing three weights plus dispatcher-owned scales | existing GLM `DequantFp8Params` |

Reject shape, role, dtype, block/group size, or extension mismatch before publication. Do not route GPT-OSS, GLM, DeepSeek-V4, GPTQ, or AWQ through the new generic adapters.

- [ ] **Step 6: Build and run the tests to verify GREEN**

Run: `python setup.py clean --all && MOE_INFINITY_TESTING=1 pip install --no-build-isolation -e . && pytest -q tests/python/ops/test_adaptive_expert_dispatch.py tests/python/integration/test_phase_adaptive_residency.py -k 'publishes or failed_transition or one_manager or composes_feature_flags'`

Expected: all selected tests pass on CUDA; explicit skips without CUDA.

- [ ] **Step 7: Commit**

```bash
git add core/parallel/expert_dispatcher.h core/parallel/expert_dispatcher.cpp core/parallel/expert_module.h core/parallel/expert_module.cpp core/prefetch/task_scheduler.h core/prefetch/task_scheduler.cpp core/prefetch/archer_prefetch_handle.h core/prefetch/archer_prefetch_handle.cpp core/python/py_archer_prefetch.cpp setup.py CMakeLists.txt core/CMakeLists.txt extensions/CMakeLists.txt tests/python/ops/test_adaptive_expert_dispatch.py tests/python/integration/test_phase_adaptive_residency.py tests/python/unit/test_setup_sources.py tests/python/unit/test_native_phase_policy_wire.py
git commit -m "feat: add safe adaptive expert representation transitions"
```

## Task 7: Build/load variants and wire the policy into general model dispatch

**Files:**
- Modify: `moe_infinity/runtime/model_offload.py:519-660,835-1233,1506-1800`
- Modify: `moe_infinity/distributed/expert_executor.py:75-95,97-291`
- Modify: `moe_infinity/memory/expert_prefetcher.py:158-312`
- Create: `tests/python/unit/test_adaptive_precision_wiring.py`
- Modify: `tests/python/unit/test_quant_regression.py`
- Modify: `tests/python/unit/test_gptq_loading.py`
- Modify: `tests/python/unit/test_awq_loading.py`

- [ ] **Step 1: Write failing opt-in and exact-fallback tests**

```python
from types import SimpleNamespace

from moe_infinity.runtime.adaptive_precision_allowlist import ReleasedAdaptiveEntry
from moe_infinity.runtime.expert_precision import ExpertFormat
from moe_infinity.runtime.model_offload import _resolve_adaptive_precision


def _archer(**overrides):
    values = {
        "adaptive_expert_precision": False,
        "adaptive_variant_build": False,
        "adaptive_derivative_root": None,
        "adaptive_hbm_budget_bytes": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_disabled_mode_returns_canonical_without_manifest_access(tmp_path):
    result = _resolve_adaptive_precision(
        SimpleNamespace(model_type="qwen3_moe", quantization_config=None),
        _archer(),
        str(tmp_path / "missing"),
        extension_names=set(),
    )
    assert result.enabled is False
    assert result.fallback_reason == "disabled"


def test_protected_model_never_builds_adaptive_variants(tmp_path):
    result = _resolve_adaptive_precision(
        SimpleNamespace(model_type="gpt_oss", quantization_config={"quant_method": "mxfp4"}),
        _archer(adaptive_expert_precision=True, adaptive_variant_build=True, adaptive_hbm_budget_bytes=1024),
        str(tmp_path),
        extension_names={"_v4_fp4"},
    )
    assert result.enabled is False
    assert result.fallback_reason == "protected:gpt_oss_mxfp4"
    assert not (tmp_path / "adaptive_derivatives" / "CURRENT").exists()


def test_legacy_store_without_manifest_is_read_only_fallback(tmp_path):
    (tmp_path / "name_id_map.json").write_text("{}")
    result = _resolve_adaptive_precision(
        SimpleNamespace(model_type="qwen3_moe", quantization_config=None),
        _archer(adaptive_expert_precision=True, adaptive_variant_build=False, adaptive_hbm_budget_bytes=1024),
        str(tmp_path),
        extension_names=set(),
    )
    assert result.enabled is False
    assert result.fallback_reason == "manifest_missing"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["name_id_map.json"]


def test_serving_rejects_valid_but_unreleased_manifest(tmp_path, monkeypatch):
    entry = ReleasedAdaptiveEntry(
        "a" * 64,
        ExpertFormat.FP8_E4M3_BLOCK128,
        "adaptive-expert-v1",
        "b" * 64,
    )
    manifest = SimpleNamespace(release_entries=frozenset({entry}))
    monkeypatch.setattr(
        "moe_infinity.runtime.expert_variant_manifest.ExpertVariantManifest.load_current",
        lambda *args, **kwargs: manifest,
    )
    result = _resolve_adaptive_precision(
        SimpleNamespace(model_type="qwen3_moe", quantization_config=None),
        _archer(adaptive_expert_precision=True, adaptive_variant_build=False, adaptive_hbm_budget_bytes=1024),
        str(tmp_path),
        extension_names=set(),
        purpose="serve",
        checkpoint_fingerprint="a" * 64,
        released_entries=frozenset(),
    )
    assert result.enabled is False
    assert result.fallback_reason == "manifest_unapproved"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/python/unit/test_adaptive_precision_wiring.py tests/python/unit/test_quant_regression.py tests/python/unit/test_gptq_loading.py tests/python/unit/test_awq_loading.py`

Expected: fails because `_resolve_adaptive_precision` does not exist.

- [ ] **Step 3: Implement explicit first-run derivative building**

Run resolution after canonical quantization detection and before model construction. `_resolve_adaptive_precision(..., purpose="build" | "serve")` keeps build and serve authorization separate. The builder is entered only by `purpose="build"` when all conditions are true:

```text
adaptive_expert_precision == true
adaptive_variant_build == true
model is in CONVERTER_CANDIDATE_MODEL_TYPES
checkpoint quantization is absent
no published generation already matches checkpoint fingerprint and converter version
canonical store is being created, not reloaded
```

Build each expert independently from checkpoint keys so peak CPU memory is one expert plus one derivative. Use Task 3's journal and reserved generation-scoped ranges; never allocate IDs by scanning an in-memory map after a crash. Write derivative payloads/index, run quality gates, emit an attestation artifact and digest, then publish `CURRENT` last. BF16 canonical remains the source of truth. A conversion failure marks the journal generation aborted and leaves it unreachable; retry cannot collide with canonical or published derivative IDs/files.

For `purpose="serve"`, call `ExpertVariantManifest.load_current` in this fixed order: recompute canonical fingerprint through `get_canonical_tensor_index_snapshot` → read/validate `CURRENT` → hash index/attestation/manifest raw bytes against all three `CURRENT` digests → parse strict schemas → validate manifest-to-index/attestation digest links → checksum every indexed payload interval → require an exact `ReleasedAdaptiveEntry` for every included format → `begin_derivative_overlay` → call `register_derivative_tensor` for every sorted index record → `commit_derivative_overlay` → register approved expert variants. Any failure before commit invokes no native registration; any native failure invokes `abort_derivative_overlay`; both return a structured canonical fallback. A candidate or quality-passing manifest that has not been released returns `manifest_unapproved`. Register approved variants after canonical `register_expert` and set the adaptive byte budget. Protected models execute the original branches before adaptive resolution.

After resolving both configs, call `configure_residency_manager(manager_enabled=phase_specific_expert_policy or adaptive_enabled, phase_policy_enabled=phase_specific_expert_policy)` on the dispatcher/prefetch native seam. Do not call a separate adaptive cache constructor. Both Python adapters must report the same manager identity before adaptive policy is enabled; mismatch returns `residency_manager_mismatch` and leaves adaptive disabled.

Add `ExpertPrefetcher.get_residency_manager_id()` as a thin call to the native prefetch handle and compare it with the dispatcher's bound method. This identifier is process-local diagnostics only and is never persisted in manifests or benchmark comparisons across processes.

- [ ] **Step 4: Feed routed counts into the policy before enqueue**

In `DistributedExpertExecutor.dispatch_local`, after `expert_count` and `expert_list` are computed but before `set_inputs`, call:

```python
if self.precision_policy is not None:
    observations = {
        ExpertKey(layer_id, expert_id): int(expert_count[expert_id])
        for expert_id in expert_list
    }
    self.precision_policy.observe(observations, tokens=int(router_mask.shape[0]))
    if self.precision_policy.epoch_due:
        metrics = self.expert_dispatcher.get_precision_metrics()
        plan = self.precision_policy.plan(
            resident_generations=ResidentGeneration.from_native(metrics["resident_generation_entries"]),
            admission_candidates={ExpertKey(layer_id, expert_id) for expert_id in expert_list},
            transition_reserved_bytes=int(metrics["transition_reserved_bytes"]),
            workspace_bytes=int(metrics["workspace_bytes"]),
        )
        self.expert_dispatcher.set_precision_targets(plan.as_native_targets(), plan.epoch)
        self.precision_policy.commit(plan)
```

Composition order in `dispatch_local` is fixed: compute routed set → observe/plan → install format targets → route-ahead calls the scheduler client's `BeginAdmission` for selected generations → enqueue demand calls the dispatcher client's `BeginAdmission`/`BeginTransition` → transfer/commit → execution lease → wait barrier → `RequestRetirement`/`ReapRetired`. `_maybe_route_ahead_prefetch` and `ExpertPrefetcher.prefetch_experts_list` accept the plan's `(expert_id, format, generation)` target instead of independently choosing storage. Policy errors increment `policy_error` and leave canonical targets unchanged. Routing masks, weights, expert order, phase arguments, prefetch priority, and output accumulation do not change.

- [ ] **Step 5: Run the tests and verify GREEN**

Run: `pytest -q tests/python/unit/test_adaptive_precision_wiring.py tests/python/unit/test_quant_regression.py tests/python/unit/test_gptq_loading.py tests/python/unit/test_awq_loading.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add moe_infinity/runtime/model_offload.py moe_infinity/distributed/expert_executor.py moe_infinity/memory/expert_prefetcher.py tests/python/unit/test_adaptive_precision_wiring.py tests/python/unit/test_quant_regression.py tests/python/unit/test_gptq_loading.py tests/python/unit/test_awq_loading.py
git commit -m "feat: wire adaptive precision into general expert dispatch"
```

## Task 8: Lock down model-specific low-bit fallback behavior

**Files:**
- Modify: `tests/python/unit/test_glm_fp8_store.py`
- Modify: `tests/python/unit/test_gpt_oss_mxfp4_dispatch.py`
- Modify: `tests/python/v4/test_v4_host_offload.py`
- Modify: `tests/python/v4/test_fp4_expert_parity.py`
- Test: `tests/python/unit/test_adaptive_precision_wiring.py`

- [ ] **Step 1: Add regression assertions for protected paths**

Add tests that monkeypatch the adaptive resolver/builder to raise if invoked, then run:

- GPT-OSS packed expansion and resident/offloaded path selection;
- GLM routed FP8 scale extraction/delivery and native dequantization;
- DeepSeek-V4 `OfficialExpertHostStore.prefetch/get`, byte-identical FP4 transfer, `resolve_use_native`, and native/reference parity;
- GPTQ/AWQ packed tensor casting.

Use this common assertion pattern:

```python
def _forbid_adaptive(*args, **kwargs):
    raise AssertionError("protected path entered adaptive precision")


def test_gpt_oss_path_does_not_enter_adaptive(monkeypatch, gpt_oss_config):
    monkeypatch.setattr(
        "moe_infinity.runtime.expert_precision.resolve_model_precision_capabilities",
        _forbid_adaptive,
    )
    result = _resolve_adaptive_precision(
        gpt_oss_config,
        _archer(
            adaptive_expert_precision=True,
            adaptive_variant_build=True,
            adaptive_hbm_budget_bytes=1024,
        ),
        "/tmp/gpt-oss-protected-test",
        extension_names={"_v4_fp4"},
    )
    assert result.enabled is False
    assert result.fallback_reason == "protected:gpt_oss_mxfp4"
```

Also assert that enabling adaptive config for each protected model emits one structured fallback reason and does not create `adaptive_derivatives/CURRENT`, a derivative generation directory, or a build journal.

- [ ] **Step 2: Run protected-path tests**

Run: `pytest -q tests/python/unit/test_glm_fp8_store.py tests/python/unit/test_gpt_oss_mxfp4_dispatch.py tests/python/v4/test_v4_host_offload.py tests/python/v4/test_fp4_expert_parity.py tests/python/unit/test_adaptive_precision_wiring.py`

Expected: CPU tests pass; hardware/checkpoint-dependent tests retain their explicit skip reasons.

- [ ] **Step 3: Fix any adaptive call ordering exposed by RED tests**

Required ordering in `model_offload.py`:

```text
detect existing checkpoint quantization
select current protected model-specific branch
return protected fallback result
only then resolve general adaptive capabilities
```

Do not change protected kernel calls, tensor layouts, scale locations, dtype casts, residency thresholds, TP behavior, or environment variables.

- [ ] **Step 4: Re-run protected-path tests**

Run: `pytest -q tests/python/unit/test_glm_fp8_store.py tests/python/unit/test_gpt_oss_mxfp4_dispatch.py tests/python/v4/test_v4_host_offload.py tests/python/v4/test_fp4_expert_parity.py tests/python/unit/test_adaptive_precision_wiring.py`

Expected: all runnable tests pass.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/runtime/model_offload.py tests/python/unit/test_glm_fp8_store.py tests/python/unit/test_gpt_oss_mxfp4_dispatch.py tests/python/v4/test_v4_host_offload.py tests/python/v4/test_fp4_expert_parity.py tests/python/unit/test_adaptive_precision_wiring.py
git commit -m "test: preserve validated low bit expert paths"
```

## Task 9: Expose exact memory, transfer, policy, and fallback metrics

**Files:**
- Modify: `core/prefetch/expert_residency.h`
- Modify: `core/prefetch/expert_residency.cpp`
- Modify: `core/parallel/expert_dispatcher.h:107-194`
- Modify: `core/parallel/expert_dispatcher.cpp:292-334`
- Modify: `core/python/py_archer_prefetch.cpp:109-123`
- Modify: `moe_infinity/memory/expert_prefetcher.py:75-156`
- Modify: `moe_infinity/runtime/model_offload.py:465-517`
- Test: `tests/python/ops/test_adaptive_expert_dispatch.py`
- Test: `tests/python/unit/test_adaptive_precision_wiring.py`

- [ ] **Step 1: Write failing metrics tests**

```python
REQUIRED_METRICS = {
    "budget_bytes",
    "resident_bytes",
    "resident_payload_bytes",
    "alignment_padding_bytes",
    "transition_reserved_bytes",
    "workspace_bytes",
    "peak_accounted_bytes",
    "h2d_payload_bytes",
    "h2d_transfers",
    "conversion_input_bytes",
    "conversion_output_bytes",
    "conversion_seconds",
    "promotions",
    "demotions",
    "representation_hits",
    "representation_misses",
    "policy_epochs",
    "active_leases",
    "leases_by_kind",
    "manager_instance_id",
    "manager_enabled",
    "phase_policy_enabled",
    "pending_transactions",
    "registered_variants",
    "resident_generations",
    "resident_generation_entries",
    "retiring_generations",
    "external_shared_resident_bytes",
    "fallback_counts",
    "by_format",
}


def test_precision_metrics_are_complete_and_budget_consistent(native_adaptive_fixture):
    dispatcher = native_adaptive_fixture.dispatcher
    expert = native_adaptive_fixture.expert
    run = native_adaptive_fixture.run
    run(expert=expert)
    metrics = dispatcher.get_precision_metrics()
    assert REQUIRED_METRICS <= metrics.keys()
    assert metrics["resident_bytes"] + metrics["transition_reserved_bytes"] + metrics["workspace_bytes"] <= metrics["budget_bytes"]
    assert metrics["resident_payload_bytes"] + metrics["alignment_padding_bytes"] == metrics["resident_bytes"]
    assert all(row["state"] in {"active", "retiring"} for row in metrics["resident_generation_entries"])


def test_host_ready_variants_are_absent_from_resident_bytes(native_adaptive_fixture):
    dispatcher = native_adaptive_fixture.dispatcher
    metrics = dispatcher.get_precision_metrics()
    assert metrics["registered_variants"] > 0
    assert metrics["resident_generations"] == 0
    assert metrics["resident_generation_entries"] == []
    assert metrics["resident_bytes"] == 0
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `MOE_INFINITY_TESTING=1 pytest -q tests/python/ops/test_adaptive_expert_dispatch.py -k metrics tests/python/unit/test_adaptive_precision_wiring.py`

Expected: metrics keys are absent.

- [ ] **Step 3: Implement monotonic counters and snapshots**

Take resident bytes and lease counts only from `ExpertResidencyManager::Snapshot`; remove any additive combination of scheduler and dispatcher occupancy that can double count one generation. Increment H2D bytes from the selected representation's raw payload, not its aligned allocation. Record allocation bytes separately. Measure conversion with CUDA events on the conversion stream; synchronize only when taking a benchmark snapshot, never in the dispatch hot path. `by_format` contains resident generations/bytes, H2D bytes/transfers, hits, misses, promotions, and demotions for every registered format. `external_shared_resident_bytes` is observational and excluded from `budget_bytes` and `peak_accounted_bytes`.

Expose the native snapshot through `ExpertPrefetcher.get_precision_metrics()` and `OffloadEngine.expert_precision_metrics`. Disabled/protected mode returns:

```python
{
    "enabled": False,
    "fallback_reason": reason,
    "fallback_counts": {},
}
```

Never fabricate absent native measurements.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `MOE_INFINITY_TESTING=1 pytest -q tests/python/ops/test_adaptive_expert_dispatch.py -k metrics tests/python/unit/test_adaptive_precision_wiring.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/prefetch/expert_residency.h core/prefetch/expert_residency.cpp core/parallel/expert_dispatcher.h core/parallel/expert_dispatcher.cpp core/python/py_archer_prefetch.cpp moe_infinity/memory/expert_prefetcher.py moe_infinity/runtime/model_offload.py tests/python/ops/test_adaptive_expert_dispatch.py tests/python/unit/test_adaptive_precision_wiring.py
git commit -m "feat: report adaptive expert cache metrics"
```

## Task 10: Add kernel, checkpoint, and generation quality gates

**Files:**
- Create: `tests/python/integration/test_adaptive_precision_parity.py`
- Modify: `tests/python/unit/test_expert_variant_conversion.py`
- Modify: `tests/python/ops/test_adaptive_expert_dispatch.py`
- Modify: `benchmarks/eval/perplexity.py`

- [ ] **Step 1: Add failing quality-gate tests**

```python
def test_fp8_expert_variant_meets_tensor_and_forward_gates(tiny_expert):
    source, x = tiny_expert
    variant = build_fp8_expert_variant(source)
    restored = variant.dequantized_weights()
    for original, actual in zip(source, restored):
        relative_l2 = (actual.float() - original.float()).norm() / original.float().norm().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(original.float().flatten(), actual.float().flatten(), dim=0)
        assert relative_l2.item() <= 0.05
        assert cosine.item() >= 0.995
    canonical_output = run_bf16_expert(x, source)
    variant_output = run_fp8_expert(x, variant)
    torch.testing.assert_close(variant_output, canonical_output, rtol=0.08, atol=0.08)


def test_checkpoint_gate_rejects_nonfinite_variant(tmp_path, tiny_checkpoint):
    result = build_variants_with_quality_gate(tiny_checkpoint, tmp_path, inject_nonfinite=True)
    assert result.accepted is False
    assert result.reason == "quality_nonfinite"
    assert not (tmp_path / "adaptive_derivatives" / "CURRENT").exists()


@pytest.mark.integration
def test_candidate_evaluation_reload_matches_fresh_decisions(tiny_candidate_checkpoint, tmp_path):
    first = run_candidate_generation(tiny_candidate_checkpoint, tmp_path / "store", build_variants=True)
    second = run_candidate_generation(tiny_candidate_checkpoint, tmp_path / "store", build_variants=False)
    assert first.policy_trace_json == second.policy_trace_json
    assert first.tokens == second.tokens


def test_quality_attestation_path_schema_and_current_digest(tmp_path, tiny_checkpoint):
    import hashlib
    import json

    result = build_variants_with_quality_gate(tiny_checkpoint, tmp_path)
    assert result.accepted is True
    root = tmp_path / "adaptive_derivatives"
    current = json.loads((root / "CURRENT").read_text())
    generation = root / "generations" / current["generation"]
    attestation_path = generation / "quality-attestation.v1.json"
    attestation_bytes = attestation_path.read_bytes()
    assert hashlib.sha256(attestation_bytes).hexdigest() == current["quality_attestation_sha256"]
    attestation = json.loads(attestation_bytes)
    assert set(attestation) == {
        "schema_version", "checkpoint_fingerprint", "generation", "converter_version",
        "converter_source_commit", "variant_payload_sha256", "derivative_index_sha256",
        "thresholds", "formats", "raw_result_sha256", "workload_sha256", "hardware",
        "software", "passed",
    }
    manifest_bytes = (generation / "manifest.v1.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == current["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    assert manifest["quality_attestation_sha256"] == current["quality_attestation_sha256"]
    assert all(
        row["quality_attestation_sha256"] == current["quality_attestation_sha256"]
        for row in manifest["variants"]
    )
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/python/unit/test_expert_variant_conversion.py tests/python/integration/test_adaptive_precision_parity.py`

Expected: fails because quality-gate helpers and integration wiring do not exist.

- [ ] **Step 3: Implement staged gates**

Apply all gates before a candidate may be proposed for release:

1. **Tensor gate:** finite source/derivative/scales, exact shape/role order, FP8 relative L2 at most `0.05`, cosine at least `0.995`; Marlin uses its existing reference dequant GEMM and `rtol=0.08`, `atol=0.08`.
2. **Kernel gate:** random and real-checkpoint expert forwards for token counts `1, 8, 64, 256`; compare against canonical BF16 with format-specific checked thresholds; verify wrong dtype/shape/group/block inputs fail before launch.
3. **Checkpoint gate:** deterministic fixed prompt set and fixed seeds; no NaN/Inf; fresh-build and reload produce the same manifest and policy trace.
4. **Model quality gate:** evaluate at least 512 validation tokens with `benchmarks/eval/perplexity.py`; require relative perplexity increase no greater than `1%` for a release candidate. Also require greedy token agreement at least `99%` over the checked workload. Store raw baseline/adaptive values in benchmark JSON.
5. **Protected-path gate:** existing GPT-OSS, GLM, V4, GPTQ, and AWQ parity tests remain unchanged or stronger.

`run_candidate_generation` is implemented inside the benchmark/test harness and opens the unpublished generation with `purpose="candidate-eval"`; that purpose is not accepted by `MoE` serving construction or public configuration. Production `purpose="serve"` always performs released-entry validation.

On success, write the strict generation-scoped schema from the data-contract section to `generations/{generation}/quality-attestation.v1.json` with the shared canonical JSON encoder. Its canonical-byte SHA-256 becomes `quality_attestation_sha256` in `CURRENT`, at manifest top level, and in every manifest variant. Passing produces a release candidate only. Serving remains canonical until that exact digest is checked into `RELEASED_ADAPTIVE_ENTRIES` once per included format. If any derivative gate fails, abort the unpublished generation and use canonical execution. Never weaken a threshold automatically.

- [ ] **Step 4: Run quality tests and verify GREEN**

Run: `pytest -q tests/python/unit/test_expert_variant_conversion.py tests/python/ops/test_adaptive_expert_dispatch.py tests/python/integration/test_adaptive_precision_parity.py`

Expected: all available tests pass; checkpoint/GPU prerequisites produce explicit skips.

- [ ] **Step 5: Commit**

```bash
git add tests/python/integration/test_adaptive_precision_parity.py tests/python/unit/test_expert_variant_conversion.py tests/python/ops/test_adaptive_expert_dispatch.py benchmarks/eval/perplexity.py
git commit -m "test: gate adaptive expert precision quality"
```

## Task 11: Add transfer, memory, TPOT, and policy benchmarks

**Files:**
- Create: `benchmarks/adaptive_precision/bench_policy.py`
- Create: `benchmarks/adaptive_precision/bench_transfer.py`
- Create: `benchmarks/adaptive_precision/bench_e2e.py`
- Create: `benchmarks/adaptive_precision/report.py`
- Create: `benchmarks/adaptive_precision/workloads.py`
- Create: `tests/python/unit/test_adaptive_precision_benchmark_schema.py`

- [ ] **Step 1: Write a failing benchmark-schema test**

```python
from benchmarks.adaptive_precision.report import validate_run
from benchmarks.adaptive_precision.workloads import deterministic_workload


def test_benchmark_schema_requires_quality_memory_transfer_and_tpot():
    row = {
        "mode": "adaptive",
        "model": "tiny-qwen3-moe",
        "checkpoint_fingerprint": "a" * 64,
        "format": "fp8_e4m3_block128",
        "converter_version": "adaptive-expert-v1",
        "quality_attestation_sha256": "b" * 64,
        "hardware": {"device_name": "reported-at-runtime", "compute_capability": [8, 0]},
        "software": {"torch": "runtime", "cuda": "runtime", "commit": "runtime"},
        "workload": {"prompt_tokens": 128, "decode_tokens": 128, "batch_size": 1, "seed": 7},
        "quality": {"perplexity": 3.0, "greedy_agreement": 1.0},
        "memory": {"budget_bytes": 1000, "peak_accounted_bytes": 900, "peak_torch_allocated_bytes": 800, "external_shared_resident_bytes": 200},
        "transfer": {"h2d_payload_bytes": 400, "h2d_transfers": 4, "exposed_fetch_seconds": 0.01},
        "latency": {"ttft_ms": 10.0, "tpot_ms_p50": 2.0, "tpot_ms_p90": 3.0, "tpot_ms_p99": 4.0},
        "throughput": {"decode_tokens_per_second": 500.0},
        "policy": {"promotions": 1, "demotions": 0, "manager_instance_id": 1, "pending_transactions": 0, "fallback_counts": {}},
    }
    validate_run(row)


def test_deterministic_workload_needs_no_prompt_file():
    one = deterministic_workload(seed=7, cases=8, min_tokens=32, max_tokens=128)
    two = deterministic_workload(seed=7, cases=8, min_tokens=32, max_tokens=128)
    assert one == two
    assert len(one) == 8
    assert all(32 <= len(case.input_ids) <= 128 for case in one)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/python/unit/test_adaptive_precision_benchmark_schema.py`

Expected: collection fails because the adaptive benchmark package does not exist.

- [ ] **Step 3: Implement benchmark harnesses and schema validation**

`bench_policy.py` accepts a JSONL route trace, manifest, budget sweep, decay/threshold sweep, and emits deterministic target/transition traces plus hit rate and bytes by format.

`bench_transfer.py` runs canonical BF16, static-low, and adaptive arms with identical expert order and warmup. Record raw/aligned H2D bytes, event-timed H2D duration, transition reservation peak, conversion input/output bytes/time, and representation hit/miss counts. It probes the device at runtime and skips unavailable formats with a reason.

`workloads.py` defines `deterministic_workload` with a local `random.Random(seed)` and stable token-ID templates bounded by the loaded tokenizer vocabulary. It serializes the generated token IDs and SHA-256 workload digest into every result. No `prompts.jsonl`, network dataset, or unplanned external input is read.

`bench_e2e.py` uses that generated workload and separate phases:

```text
2 cold setup runs excluded
3 warmup generations excluded
5 measured repetitions per arm
TTFT = request start to first generated token
TPOT samples = timestamp difference between consecutive decode tokens
throughput = measured decode tokens / decode wall time, excluding prefill
peak HBM = max(native accounted bytes, torch.cuda.max_memory_allocated()) reported as separate fields
```

Required arms are `canonical`, `static_low`, and `adaptive`; all use the same HBM budget. Output one JSON object per repetition and one aggregate JSON. Never print a speedup claim; report ratios with confidence intervals only in `report.py`.

- [ ] **Step 4: Define release gates in the report**

A fingerprint/format/converter release candidate passes only if:

- quality gates from Task 10 pass and an attestation digest is emitted;
- `peak_accounted_bytes <= budget_bytes` in every repetition;
- no `policy_error`, `manifest_error`, `transition_failed`, or `budget_rejected_active_expert` fallback occurs;
- adaptive median H2D payload bytes are no greater than canonical under the same workload;
- adaptive median TPOT is no more than 5% worse than the better of canonical and static-low;
- p99 TPOT is finite and reported, not discarded;
- five raw repetitions, exact runtime hardware/software metadata, and manifest fingerprint are present.

A failed gate leaves the entry outside the released allowlist. A passing report emits a candidate `ReleasedAdaptiveEntry` snippet but does not edit `adaptive_precision_allowlist.py` and does not authorize serving. The report states measured values without extrapolating to other GPUs, interconnects, batch sizes, or models.

- [ ] **Step 5: Run the schema and policy benchmark tests**

Run: `pytest -q tests/python/unit/test_adaptive_precision_benchmark_schema.py tests/python/unit/test_adaptive_precision_policy.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/adaptive_precision tests/python/unit/test_adaptive_precision_benchmark_schema.py
git commit -m "bench: measure adaptive expert precision tradeoffs"
```

## Task 12: Document support, fallback, risks, and operations

**Files:**
- Create: `docs/adaptive-expert-precision.md`
- Modify: `docs/configuration.md`
- Modify: `docs/model-compatibility.md`
- Modify: `docs/benchmarking.md`
- Create: `tests/python/unit/test_adaptive_precision_docs.py`

- [ ] **Step 1: Write the failing executable documentation contract**

Create `tests/python/unit/test_adaptive_precision_docs.py` with exact content checks rather than a manual-review instruction:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ADAPTIVE = (ROOT / "docs/adaptive-expert-precision.md").read_text()
CONFIG = (ROOT / "docs/configuration.md").read_text()
COMPAT = (ROOT / "docs/model-compatibility.md").read_text()
BENCH = (ROOT / "docs/benchmarking.md").read_text()


def _contains_all(text: str, values: tuple[str, ...]) -> None:
    missing = [value for value in values if value not in text]
    assert not missing, f"missing documentation contracts: {missing}"


def test_documents_disabled_default_and_configuration_units() -> None:
    _contains_all(CONFIG + ADAPTIVE, (
        "adaptive_expert_precision", "false", "adaptive_hbm_budget_bytes",
        "bytes", "adaptive_policy_epoch_tokens", "adaptive_hotness_decay",
        "adaptive_promotion_threshold", "adaptive_demotion_threshold",
        "adaptive_min_residency_epochs", "adaptive_transition_cooldown_epochs",
        "adaptive_variant_build", "adaptive_derivative_root",
    ))


def test_documents_manifest_index_and_attestation_validation() -> None:
    _contains_all(ADAPTIVE, (
        "CURRENT", "derivative-index.v1.json", "manifest.v1.json",
        "quality-attestation.v1.json", "checkpoint_fingerprint",
        "converter_version", "quality_attestation_sha256",
        "ReleasedAdaptiveEntry", "manifest_unapproved", "canonical fallback",
    ))


def test_documents_support_and_four_flag_rollout_matrix() -> None:
    _contains_all(ADAPTIVE + COMPAT, (
        "GPT-OSS MXFP4", "GLM-5.2 FP8", "DeepSeek-V4-Flash FP4",
        "DeepSeek-V3 FP8", "GPTQ/AWQ", "Phase policy",
        "Adaptive precision", "off | off", "on | off", "off | on", "on | on",
        "ExpertResidencyManager",
    ))


def test_documents_benchmark_and_attestation_acceptance() -> None:
    _contains_all(BENCH + ADAPTIVE, (
        "deterministic-v1", "canonical", "static-low", "adaptive",
        "H2D", "peak_accounted_bytes", "TTFT", "TPOT", "p50", "p90", "p99",
        "throughput", "release_gate", "quality-attestation", "five",
        "no speedup is guaranteed",
    ))


def test_documents_executable_build_benchmark_report_and_rollback_commands() -> None:
    _contains_all(ADAPTIVE, (
        "python benchmarks/adaptive_precision/bench_e2e.py",
        "--adaptive-variant-build", "--build-only",
        "python benchmarks/adaptive_precision/bench_policy.py",
        "python benchmarks/adaptive_precision/report.py",
        '"adaptive_expert_precision": false', "restart",
        "preserve", "adaptive_derivatives", "CURRENT",
    ))
```

- [ ] **Step 2: Run documentation QA and verify RED**

Run: `pytest -q tests/python/unit/test_adaptive_precision_docs.py`

Expected: FAIL because `docs/adaptive-expert-precision.md` is absent or one or more required configuration, manifest/attestation, support-matrix, benchmark, rollout/rollback, or command strings are not yet documented.

- [ ] **Step 3: Write the documentation**

Document:

- Task 0's self-contained manager foundation and PR #179's design-coordination-only status;
- `ExpertResidencyManager` as the sole authority and the four feature-flag combinations;
- disabled-by-default configuration and byte units;
- the three-way distinction among checkpoint storage, cached representation, and execution dtype;
- converter/build candidates versus exact fingerprint/format/converter/attestation released entries;
- journaled derivative generation creation as an explicit candidate-build operation, separate from serving;
- all fallback reason strings;
- all metrics and benchmark fields;
- quality and release gates;
- commands below;
- statement that SliceMoE and DynaExq are motivation only;
- exact statement `no speedup is guaranteed`;
- benchmark protocol wording `five measured repetitions` so documentation QA can distinguish the release gate from a one-off run.

Use this support matrix:

| Model/checkpoint path | Adaptive policy | Existing path |
|---|---:|---|
| Candidate unquantized Mixtral/Qwen3-MoE/DeepSeek-V2 without exact released entry | build/benchmark only; serving rejects derivatives | canonical BF16 fallback |
| Exact released checkpoint fingerprint + format + converter version + attestation digest | opt-in serving after manifest/index digest validation | canonical BF16 fallback |
| GPT-OSS MXFP4 | bypassed | existing resident/native MXFP4 path |
| GLM-5.2 FP8 | bypassed | existing FP8 host store plus dispatcher dequant |
| DeepSeek-V4-Flash FP4 | bypassed | existing `OfficialExpertHostStore` plus native/tilelang adapter |
| DeepSeek-V3 FP8 | bypassed | existing model-specific FP8 behavior |
| GPTQ/AWQ | bypassed | existing quantized conversion path |
| Unknown model or unavailable extension | disabled | canonical existing path |

Also include this composition matrix:

| Phase policy | Adaptive precision | Native residency path |
|---:|---:|---|
| off | off | Task 0's preserved legacy fallback |
| on | off | `ExpertResidencyManager`, canonical generation 0, phase utility enabled |
| off | on | `ExpertResidencyManager`, variant transactions, neutral mixed-phase/legacy-equivalent utility |
| on | on | the same `ExpertResidencyManager`, variant transactions plus phase utility |

- [ ] **Step 4: Document exact operational commands and rollback**

```bash
# Build derivative candidates explicitly; serving does not build or approve them.
python benchmarks/adaptive_precision/bench_e2e.py \
  --model Qwen/Qwen3-30B-A3B \
  --offload-dir "$OFFLOAD_DIR" \
  --adaptive-variant-build \
  --adaptive-hbm-budget-bytes 17179869184 \
  --build-only \
  --workload deterministic-v1 \
  --seed 7

# Replay policy without CUDA.
python benchmarks/adaptive_precision/bench_policy.py \
  --trace routes.jsonl \
  --derivative-root "$OFFLOAD_DIR/adaptive_derivatives" \
  --budget-bytes 17179869184 \
  --output policy.json

# Run measured canonical/static-low/adaptive arms.
python benchmarks/adaptive_precision/bench_e2e.py \
  --model Qwen/Qwen3-30B-A3B \
  --offload-dir "$OFFLOAD_DIR" \
  --adaptive-hbm-budget-bytes 17179869184 \
  --workload deterministic-v1 \
  --seed 7 \
  --output results.jsonl

# Validate and summarize without claiming unmeasured gains.
python benchmarks/adaptive_precision/report.py results.jsonl --output report.json

# Roll back serving without deleting derivatives or rewriting canonical store files.
python - "$CONFIG_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
config["adaptive_expert_precision"] = False
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
# Restart the serving process with "$CONFIG_JSON", then verify fallback reason "disabled".
```

Also show the declarative rollback fragment `{"adaptive_expert_precision": false}`. State explicitly that rollback preserves `adaptive_derivatives`, `CURRENT`, canonical Archer files, and manifests for diagnosis; serving must not load or mutate them while disabled. After restart, verify fallback reason `disabled`, zero adaptive transition activity, and successful canonical requests.

- [ ] **Step 5: Run executable documentation QA**

Run:

```bash
pytest -q tests/python/unit/test_adaptive_precision_docs.py
python -m compileall benchmarks/adaptive_precision
```

Expected: the five documentation tests pass. They prove that configuration/defaults and byte units are present; manifest/index/attestation and exact release approval are explained; all protected paths and four rollout combinations are listed; benchmark arms, metrics, five repetitions, quality attestation, and release gate are specified; and copy-pastable build, policy replay, benchmark, report, and rollback commands are present. `compileall` exits 0 for every documented Python entry point.

- [ ] **Step 6: Commit**

```bash
git add docs/adaptive-expert-precision.md docs/configuration.md docs/model-compatibility.md docs/benchmarking.md tests/python/unit/test_adaptive_precision_docs.py
git commit -m "docs: describe adaptive expert precision policy"
```

## Task 13: Run complete verification and release-gate checks

**Files:**
- Verify only; no new production files.

- [ ] **Step 1: Run formatting and static checks**

Run:

```bash
ruff check moe_infinity/runtime/expert_precision.py moe_infinity/runtime/adaptive_precision_allowlist.py moe_infinity/runtime/expert_variant_manifest.py moe_infinity/runtime/expert_variant_build.py moe_infinity/memory/adaptive_precision_policy.py benchmarks/adaptive_precision/workloads.py tests/python/unit/test_expert_precision_registry.py tests/python/unit/test_expert_variant_manifest.py tests/python/unit/test_expert_variant_build.py tests/python/unit/test_adaptive_precision_policy.py
ruff format --check moe_infinity/runtime/expert_precision.py moe_infinity/runtime/adaptive_precision_allowlist.py moe_infinity/runtime/expert_variant_manifest.py moe_infinity/runtime/expert_variant_build.py moe_infinity/memory/adaptive_precision_policy.py benchmarks/adaptive_precision/workloads.py tests/python/unit/test_expert_precision_registry.py tests/python/unit/test_expert_variant_manifest.py tests/python/unit/test_expert_variant_build.py tests/python/unit/test_adaptive_precision_policy.py
```

Expected: both commands exit 0 with no changes required.

- [ ] **Step 2: Run the CPU regression suite**

Run:

```bash
pytest -q \
  tests/python/unit/test_expert_precision_registry.py \
  tests/python/unit/test_expert_variant_manifest.py \
  tests/python/unit/test_expert_variant_build.py \
  tests/python/unit/test_adaptive_precision_policy.py \
  tests/python/unit/test_adaptive_precision_wiring.py \
  tests/python/unit/test_expert_variant_conversion.py \
  tests/python/unit/test_adaptive_precision_benchmark_schema.py \
  tests/python/unit/test_adaptive_precision_docs.py \
  tests/python/unit/test_setup_sources.py \
  tests/python/unit/test_native_phase_policy_wire.py \
  tests/python/unit/test_quantization_detection.py \
  tests/python/unit/test_quant_regression.py \
  tests/python/unit/test_gptq_loading.py \
  tests/python/unit/test_awq_loading.py \
  tests/python/unit/test_glm_fp8_store.py \
  tests/test_mxfp4_kernel.py
```

Expected: all CPU-capable tests pass; CUDA-only tests show explicit skips.

- [ ] **Step 3: Build native extensions**

Run:

```bash
python setup.py clean --all
pip install --no-build-isolation -e .
python -c 'from moe_infinity import _store; assert not hasattr(_store.expert_dispatcher, "inject_transition_failure_once_for_test")'
python setup.py clean --all
MOE_INFINITY_TESTING=1 pip install --no-build-isolation -e .
python -c 'from moe_infinity import _store; assert hasattr(_store.expert_dispatcher, "inject_transition_failure_once_for_test")'
cmake -S . -B build-testing -DMOE_INFINITY_TESTING=ON -DCUTLASS_DIR="$CUTLASS_DIR"
cmake --build build-testing --target prefetch_op test_expert_residency test_expert_residency_variants -j2
ctest --test-dir build-testing -R 'ExpertResidency(Variant)?Tests' --output-on-failure
```

Expected: normal `_store` omits test bindings; setuptools and CMake testing builds expose them consistently; both build systems compile Task 0's `expert_residency.cpp` exactly once and no other residency authority; original and variant manager tests pass. `_marlin` and `_v4_fp4` availability is reported by the build; unavailable optional extensions disable only their registered formats.

- [ ] **Step 4: Run single-GPU native and integration tests**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 MOE_INFINITY_TESTING=1 pytest -q \
  tests/python/ops/test_adaptive_expert_dispatch.py \
  tests/python/integration/test_phase_adaptive_residency.py \
  tests/python/integration/test_adaptive_precision_parity.py \
  tests/python/unit/test_glm_fp8_native_dequant.py \
  tests/python/unit/test_gpt_oss_mxfp4_dispatch.py
```

Expected: all tests supported by the built extensions pass; the combined test reports one manager identity and one resident-byte snapshot for each routing combination without importing a separately landed phase-policy test module; unavailable checkpoint/extension cases are explicit skips, not silent passes.

- [ ] **Step 5: Run DeepSeek-V4 protected-path tests in its validated environment**

Run:

```bash
DSV4_FLASH_CKPT="$DSV4_FLASH_CKPT" pytest -q tests/python/v4/
torchrun --nproc-per-node 4 tests/python/v4/e2e_mp4_offload.py
```

Expected: unit tests pass and the existing mp4 golden parity harness passes. Do not substitute a different TP degree as evidence for this gate.

- [ ] **Step 6: Run measured release gates on each proposed allowlist entry**

Run the Task 12 build, benchmark, and report commands separately for each exact checkpoint fingerprint, format, and converter version. Expected: `report.json` contains `"release_gate": "pass"` and a quality-attestation digest. Serving must still return `manifest_unapproved` until a separate reviewed change adds the exact four-field `ReleasedAdaptiveEntry`; a fail result remains canonical and cannot be overridden.

- [ ] **Step 7: Verify the final diff contains no protected-kernel replacement**

Run:

```bash
git diff --check
git diff -- moe_infinity/kernel/mxfp4_gemm.py extensions/kernel/v4_fp4/v4_fp4_binding.cpp
```

Expected: `git diff --check` exits 0. The second command shows no kernel-math replacement; any test-only or comment change must be reviewed explicitly.

## Fallback matrix

| Condition | Required behavior | Metric/reason |
|---|---|---|
| Task 0 initially finds no exact `ExpertResidencyManager` API/tests | Create or reconcile the exact foundation, register it once, and pass its real-manager tests before Task 1 | foundation setup; no runtime fallback |
| Adaptive feature disabled | Do not read adaptive manifests or register variants; preserve Task 0's legacy/manager behavior selected by the phase flag | `disabled` |
| GPT-OSS/GLM/V4/DeepSeek-V3/GPTQ/AWQ | Run current model-specific path; preserve any enabled phase behavior where applicable; do not invoke general adaptive conversion | the exact `protected:gpt_oss_mxfp4`, `protected:glm_fp8`, `protected:deepseek_v4_fp4`, `protected:existing_fp8`, `protected:gptq`, or `protected:awq` reason |
| Model is not a converter candidate | Do not build derivatives; run canonical path | `unsupported:converter_candidate` |
| Candidate manifest lacks an exact released four-field entry | Reject the entire derivative generation before overlay load; run canonical | `manifest_unapproved` |
| Required extension/capability absent | Remove only that format; use next validated format or canonical | `unsupported:capability` |
| Legacy store has no published derivative `CURRENT` | Read-only canonical fallback | `manifest_missing` |
| Build journal, `CURRENT`, index, manifest, fingerprint, or digest is inconsistent | Canonical fallback; serving does not repair files | `derivative_generation_invalid` |
| Variant build or quality gate fails | Do not publish manifest; canonical fallback | `variant_build_failed`, `quality_nonfinite`, `quality_tensor`, `quality_kernel`, or `quality_model` |
| Resident/retiring generations plus reservations/workspace exceed budget | Admit/promote nothing; manager reaps only idle unleased event-complete generations | `budget_rejected` |
| Async copy/conversion/event fails | Keep old active generation and reclaim destination | `transition_failed` |
| Requested format unavailable for one expert shape | Keep that expert canonical; other validated experts may adapt | `expert_capability_miss` |
| Policy raises or emits stale epoch | Ignore plan; keep current targets | `policy_error` or `stale_epoch` |
| Dispatcher and scheduler do not expose the same manager identity | Disable adaptive registration before overlay load and retain Task 0's canonical path | `residency_manager_mismatch` |
| Manager transaction or shared-lease invariant is violated | Keep generation resident, abort the transaction, stop adaptive admission for the epoch, and use canonical on demand | `residency_transaction_error` |

No fallback path changes tensor values silently. The chosen storage and execution formats are visible in metrics and logs.

## Risk register and mitigations

| Risk | Mitigation and required test |
|---|---|
| Adaptive implementation assumes a plan-only manager exists | Task 0 creates or reconciles the exact manager files, clients, source registration, and real-manager tests; PR #179 is design coordination only. |
| A parallel merge recreates two native authorities | Adaptive modifies only `core/prefetch/expert_residency.{h,cpp}` and existing clients; source tests reject every additional residency-authority source. |
| Double-counting or undercounting HBM because Archer aligns each tensor | Track payload and aligned bytes separately; enforce the invariant with native tests. |
| Charging all host-ready experts makes adaptive mode incompatible with sparse residency | Policy inputs contain only resident/retiring generations and current admission candidates; host-ready catalog bytes are zero. |
| Prefetch and dispatcher charge or evict the same generation independently | One Task 0 `ExpertResidencyManager`, manager-issued leases/transactions, and shared-client tests. |
| Adaptive enabled while phase flag is disabled falls back into legacy private caches | `manager_enabled = phase_enabled || adaptive_enabled` matrix test forces both clients through the manager whenever adaptive is active. |
| Destination plus old generation temporarily exceeds budget | Reserve destination/workspace before allocation; retire old bytes only after lease and event completion. |
| Use-after-free between fetch, conversion, and execution streams | Stable generations, atomic active pointer, lease counts, ready/last-use events, and shutdown-order tests. |
| FP8 scale lifetime or projection order mismatch | Manifest owns generic scales and exact role order; GLM retains dispatcher-owned scales; role validation precedes publication. |
| Sharded checkpoints split expert components | Build derivatives by indexed per-expert reads, not per-shard opportunistic grouping. |
| Crash/retry reuses derivative IDs or publishes half an index | Durable high-water journal, generation-scoped tensor/file ranges above canonical maxima, checksums, and atomic `CURRENT`. |
| Existing offload store becomes unreloadable | Overlay derivative index, unchanged canonical index/IDs/signature, atomic generation publication, and read-only canonical fallback. |
| A quality-passing candidate is served without review | Empty-by-default released entries and exact fingerprint/format/converter/attestation validation before overlay load. |
| Marlin shape/dtype incompatibility | Pure capability predicate plus prelaunch validation; canonical/FP8 fallback. |
| Policy flaps under changing routes | EWMA, promotion/demotion hysteresis, minimum residency, cooldown, deterministic replay. |
| General code captures GPT-OSS, GLM, or V4 | Protected resolution occurs first and regression tests fail on any adaptive entry. |
| TP/rank-specific expert ownership differs | Adaptive keys are rank-local `(layer, expert)` entries; V4 TP remains entirely in its existing adapter. |
| Quality loss passes microbench but harms generation | Tensor, kernel, checkpoint, perplexity, and greedy-token gates are all required. |
| Performance result depends on unsupported hardware assumptions | Runtime probes and recorded metadata; no hard-coded GPU model, bandwidth, or speedup claim. |

## Completion criteria

- This plan-only branch remains independently reviewable as a Markdown-only change; implementation begins with Task 0 on the implementation branch.
- Adaptive mode is opt-in and deterministic.
- Task 0 either validates an already-landed exact manager or creates/reconciles it, then passes exactly-once build registration and concrete capacity/transaction/lease tests before adaptive changes begin.
- No second residency-authority source, singleton, resident map, byte counter, lease registry, or victim selector exists; setup/CMake register only Task 0's `expert_residency.cpp`.
- Converter/build candidates are separate from released serving entries; unapproved manifests are rejected before derivative index loading.
- Every derivative representation has explicit storage format, execution kind, roles, scale owner, payload/aligned/workspace bytes, source format, converter version, quality-attestation digest, and generation.
- Derivative tensor/file IDs are journaled strictly above canonical maxima; crash/retry cannot collide and `CURRENT` atomically names one complete index/manifest generation.
- Native accounting charges only resident/retiring generations, transition reservations, and workspace; host-ready variants charge zero, and the total never exceeds the fixed HBM budget.
- Prefetch, dispatch, execution, and retirement compose through Task 0's single `ExpertResidencyManager` transaction API; one generation is charged once regardless of client or lease count.
- `test_phase_adaptive_residency.py` exercises all four Boolean routing combinations against Task 0's manager contract without assuming a separately landed phase implementation; adaptive enabled always routes scheduler and dispatcher through the manager, while both disabled preserves the legacy fallback. When the phase feature is present, its own behavior tests also run unchanged.
- A forward pass uses only a fully published generation; failed transitions preserve the old generation.
- Protected GPT-OSS MXFP4, GLM FP8, DeepSeek-V4 FP4, DeepSeek-V3 FP8, GPTQ, and AWQ paths pass their existing parity tests and never enter general adaptive conversion.
- Legacy stores and unsupported hardware fall back exactly without file mutation.
- Policy simulation is byte-for-byte deterministic.
- Kernel, checkpoint, perplexity, and token-agreement gates produce a digest-qualified candidate; only a separate exact released entry enables serving.
- Benchmark workloads are generated deterministically by `workloads.py`; no undeclared prompts file is required.
- Benchmark output includes raw memory, transfer, TTFT, TPOT p50/p90/p99, throughput, policy, fallback, quality, hardware, software, fingerprint, format, converter version, and attestation fields without claiming unmeasured speedups.
- Executable documentation QA passes for configuration/defaults, manifests/indexes/attestations, support and four-flag rollout, benchmark/release gates, and build/replay/benchmark/report/rollback commands.
