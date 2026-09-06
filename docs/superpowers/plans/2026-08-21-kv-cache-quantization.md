# Correctness-Validated KV-Cache Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one opt-in, correctness-gated `int8_sym` paged KV-cache storage path behind a format abstraction while preserving existing attention execution dtypes and deterministic fallback to the current native/SDPA paths.

**Architecture:** Introduce a format descriptor and one `LayeredPagedKVStore` that is constructed once per serving engine and shared by `PagedKVCache`, `ModelRunner`, and the executed attention backend; owner identity is bound before the first request and mismatches fail before model execution. The first portable format stores K and V as signed symmetric INT8 with one FP16 scale per `(layer, physical page, KV head, token)` and computes attention in the model dtype; CUDA decode consumes packed pages directly, while CPU/unsupported cases dequantize the requested logical prefix into valid FP32 SDPA tensors. FlashInfer remains active for native stores, but an `int8_sym` request selects the built-in quantized path because this repository does not currently expose a compatible FlashInfer INT8 page/scale contract.

**Tech Stack:** Python 3.10+, PyTorch, CUDA/C++ PyTorch extensions, FlashInfer capability detection, pytest, Hugging Face model/config metadata.

---

## Scope and design decisions

### Repository facts that constrain the design

- `moe_infinity/serving/kv_cache.py` currently owns a dense per-layer NHD cache shaped `[layers, blocks, 2, block_size, kv_heads, head_dim]`, while `moe_infinity/runtime/attention_backend.py` separately allocates native K/V and optional FlashInfer caches. The plan must remove that duplicate serving ownership rather than wrapping both allocations.
- `ModelRunner` currently discovers an attention backend indirectly from `engine` and passes only backend/metadata to model attention classes. The new handoff must pass and validate the same store identity used by scheduling and attention execution.
- `moe_infinity/kernel/paged_attention_ops.py` dispatches to `moe_infinity._paged_attn` only when every argument is CUDA; otherwise it reconstructs pages and calls SDPA.
- `extensions/kernel/paged_attention.cu` accepts only floating K/V whose dtype matches the query. It supports MHA/GQA indexing but has no low-bit cache or scale metadata input.
- `KVCacheSpec.page_size_bytes`, `MemoryManager.get_max_kv_blocks`, and `big_modeling.py` assume all cache elements have the model dtype. They must count payload plus scale bytes instead.
- Native-engine swapping in `KVCacheOffloadCoordinator` and serving swapping in `PagedKVCache` copy cache tensors synchronously. The implementation must transfer packed INT8 plus FP16 scales as-is and must not depend on, or alter, async KV completion semantics.
- Qwen3 paged attention exposes ordinary GQA metadata. DeepSeek/GLM MLA uses compressed latent dimensions (`kv_lora_rank` and family-specific attention), so the first low-bit path must reject MLA and visibly fall back rather than infer that a one-head shape is ordinary MQA.

### Initial format: `int8_sym`

Use signed symmetric INT8, dynamically calibrated per token and KV head:

```text
amax = max(abs(x[head, :]))
raw_scale = amax / 127
stored_scale = round_up_to_fp16(max(raw_scale, smallest_positive_fp16_subnormal))
q = clamp(round(x.float32 / stored_scale.float32), -127, 127).to(int8)
x_hat = q.to(execution_dtype) * stored_scale.to(execution_dtype)
```

`round_up_to_fp16` means cast to FP16 and, when that cast is below `raw_scale`, advance once with `torch.nextafter(candidate, +inf)`. Quantization and dequantization both use that exact stored FP16 value. Inputs whose required scale exceeds finite FP16 are rejected before writing. This makes `127 * stored_scale >= amax`, prevents scale-rounding saturation, gives exact zero reconstruction (`q=0`), and preserves the element error bound `abs(x_hat - x) <= stored_scale / 2 + 1e-6`.

The sole canonical layer-major store layout is:

```text
payload: [num_layers, num_pages, 2, block_size, num_kv_heads, head_dim]
scales:  [num_layers, num_pages, 2, block_size, num_kv_heads] float16 (INT8 only)
```

No execution backend owns a second capacity tensor. `LayeredPagedKVStore.paged_kernel_layer_view(layer_idx)` derives strided views without copying:

```text
K: [num_pages, num_kv_heads, head_dim / 8, block_size, 8]
V: [num_pages, num_kv_heads, head_dim, block_size]
K/V scales: [num_pages, num_kv_heads, block_size]
```

For native FlashInfer, `flashinfer_layer_view(layer_idx)` returns the canonical `[num_pages, 2, block_size, num_kv_heads, head_dim]` layer view. It is valid only for `native`; `int8_sym` bypasses FlashInfer as specified above.

This is the initial portable choice because PyTorch and every targeted CUDA architecture support INT8 storage, FP16 metadata, deterministic quantize/dequantize, and FP32 accumulation; unlike 2/3/4-bit packing it needs no architecture-specific bit extraction contract. It also supports append-at-arbitrary-slot and whole-page swap without recalibrating older tokens. Per-token/per-head calibration is deliberately simpler than research-oriented per-channel/outlier schemes and therefore requires the quality gates below. [KIVI](https://arxiv.org/abs/2402.02750) and [KVQuant](https://arxiv.org/abs/2401.18079) motivate distribution-aware KV quantization and rigorous quality validation; they do **not** establish repository support for universal 2-bit KV cache, and this change makes no such claim.

The exact per-layer page formula is:

```text
payload_bytes = 2 * block_size * num_kv_heads * head_dim * sizeof(int8)
scale_bytes   = 2 * block_size * num_kv_heads * sizeof(float16)
page_bytes    = payload_bytes + scale_bytes
```

For `block_size=16`, `num_kv_heads=8`, and `head_dim=128`, this is `32,768 + 512 = 33,280` bytes, versus `65,536` native FP16 bytes, for a ratio of `0.5078125`.

### Precision contract

| Concern | `native` | `int8_sym` |
| --- | --- | --- |
| Storage precision | model cache dtype (`fp16`, `bf16`, or `fp32`) | INT8 payload + FP16 scales |
| Transfer precision | same native tensor bytes | synchronously copied INT8 payload and FP16 scales; no D2H/H2D dequantization |
| Attention execution precision | query/model dtype, FP32 accumulator in native CUDA kernel | query/model dtype output, scales promoted to FP32, FP32 QK/softmax/V accumulation |
| Prefill | existing FlashInfer or SDPA path; cache write remains native | quantize cache write, compute current prefill from unquantized incoming K/V in model dtype |
| Decode | existing FlashInfer/native/SDPA | quantized native CUDA kernel, or referenced-page dequantization followed by SDPA |

### Capability and fallback contract

`resolve_kv_cache_format(requested, model_info, device, backend_preference, capabilities)` returns a decision with `requested_format`, `effective_format`, `execution_backend`, and a machine-readable `reason`. `capabilities` contains `flashinfer_available`, `native_int8_binding_available`, and `sdpa_available`; the native bit is true only when the loaded `_paged_attn` module exposes a callable `paged_attention_int8_v1` binding. Module import or `HAS_PAGED_ATTN` alone is insufficient.

| Request | Conditions | Effective path |
| --- | --- | --- |
| `native` | all | current behavior; no warning |
| `int8_sym` | CUDA, callable native INT8 binding, ordinary MHA/GQA, `head_dim % 8 == 0` | effective format `int8_sym`, execution backend `native_int8` |
| `int8_sym` | CPU or callable binding absent, `sdpa_available`, ordinary MHA/GQA | effective format `int8_sym`, execution backend `sdpa_dequant`; reason is `cpu_sdpa_dequant`, `native_int8_module_unavailable`, or `native_int8_binding_missing` |
| `int8_sym` + `backend_preference=flashinfer` | ordinary MHA/GQA | bypass FlashInfer; choose `native_int8` when available, otherwise `sdpa_dequant`; emit one structured event `flashinfer_no_int8_sym_contract` without changing effective storage format |
| `int8_sym` | MLA metadata (`kv_lora_rank`, `qk_nope_head_dim`, or `qk_rope_head_dim` family contract) | `native` with warning `mla_not_validated` when fallback is allowed; startup error otherwise |
| `int8_sym` | neither native INT8 nor SDPA execution available | `native` with warning `no_int8_execution_backend` when fallback is allowed; startup error otherwise |
| unknown format | all | configuration error before allocation |

Mixed batches must not mix formats inside one backend allocation. Different model/backend instances may use different effective formats. A fallback must update exported stats/config so operators never believe a native cache is quantized.

### Canonical store protocol and vocabulary

- **Page:** fixed-size physical allocation unit addressed by block tables; existing scheduler/prefix-cache `block_id` and `block_hash` names remain unchanged at those compatibility boundaries.
- **Chunk:** K/V tensors newly written together, or a page subset snapshotted for synchronous transfer. Use `write_chunk`, `snapshot_page_chunk`, `restore_page_chunk`, and `LayeredKVPageChunk`; do not call these objects generic “storage bundles.”
- **Prefix:** logical tokens selected by `(block_table, seq_len)` for one sequence. Use `read_prefix`/`dequantize_prefix`; this aligns attention reads and DFlash’s committed-prefix terminology.
- **Store:** the sole full-capacity owner, `LayeredPagedKVStore`. `PagedKVCache` owns allocation/residency metadata but references this store; `ModelRunner` and `PagedAttentionBackend` reference exactly the same object.

### Quality and rollout gates

The feature stays opt-in (`native` default) until all gates pass on the release hardware/model matrix:

1. Quantizer calibration: the quantizer uses the stored, upward-rounded FP16 scale; scale is finite, no payload uses `-128`, zero reconstructs exactly, and adversarial values around FP16 rounding boundaries satisfy element error `<= stored_scale / 2 + 1e-6`.
2. Kernel equivalence: quantized CUDA decode versus dequantized FP32 SDPA reference with `atol=2e-2`, `rtol=2e-2`, cosine similarity `>= 0.999` for MHA and GQA.
3. Prefill/decode sequence equivalence: per-step logits cosine similarity `>= 0.999`, mean absolute logit error `<= 0.02`, and greedy token agreement `>= 99%` over at least 2,048 generated tokens. Token identity alone is not sufficient.
4. Perplexity: WikiText-2 validation perplexity increase `<= 0.10` absolute **and** `<= 1.0%` relative for each validated model. Report C4 when locally available, but do not silently waive WikiText-2.
5. Memory: allocated KV storage bytes equal the descriptor formula, and `int8_sym/native <= 0.52` for `head_dim >= 64`; peak end-to-end GPU memory must decrease at 8K and 32K context after including scratch.
6. Stability: swap-out/in, append across pages, partial/full rollback, prefill, decode, and GQA tests pass under compute-sanitizer; MLA proves visible fallback.
7. Performance is reported, not used to relax correctness: no more than 10% decode tokens/s regression at 2K context; long-context results include TTFT, ITL p50/p99, throughput, peak GPU memory, cache bytes, swap bytes, and effective backend.

## Planned file structure

- Create `moe_infinity/runtime/kv_cache_format.py`: format enum/config, callable-binding capability decision, byte accounting, `LayeredPagedKVStore`, `LayeredKVPageChunk`, chunk writes, prefix reads, page snapshots, and swap serialization contract.
- Modify `moe_infinity/runtime/attention_types.py`: make `KVCacheSpec` carry requested/effective format and delegate page-byte accounting.
- Modify `moe_infinity/runtime/attention_backend.py`: stop allocating serving KV tensors, bind one `LayeredPagedKVStore`, quantize chunk writes, read logical prefixes, select prefill/decode path, and expose effective format, execution backend, and decision reason.
- Modify `moe_infinity/kernel/paged_attention_ops.py`: Python dispatch for quantized CUDA and dequantized SDPA fallback.
- Create `extensions/kernel/paged_attention_int8.cuh`: declare the reusable INT8 paged-attention launcher consumed by both the Python extension wrapper and the standalone CUDA test.
- Create `extensions/kernel/paged_attention_int8.cu`: own the INT8+scale device kernel and launcher with FP32 accumulation; it contains no `PYBIND11_MODULE`.
- Modify `extensions/kernel/paged_attention.cu`: include the reusable INT8 declaration and register its wrapper beside the retained native binding in the single existing `PYBIND11_MODULE`.
- Modify `setup.py`: add `extensions/kernel/paged_attention_int8.cu` to `_PAGED_ATTN_SOURCES` so the extension explicitly links the reusable implementation.
- Modify `moe_infinity/serving/kv_cache.py`: accept the sole `LayeredPagedKVStore`, enforce explicit RESIDENT/SWAPPED ownership transitions, snapshot/restore page chunks, and truncate only the pages owned by the current residency state.
- Modify `moe_infinity/serving/model_runner.py`: accept the store and backend explicitly, validate owner/store identity before setting paged context, and pass layer-aware execution through the bound backend.
- Modify `moe_infinity/runtime/model_offload.py` and `moe_infinity/entrypoints/big_modeling.py`: install/get the store and backend as one owner-bound component pair for serving handoff.
- Modify `moe_infinity/serving/scheduler.py`: replace the unsafe two-call `swap_out(); free_gpu_blocks()` preemption sequence with one ownership-transferring `swap_out(release_gpu_blocks=True)` call.
- Modify `moe_infinity/engine/kv_cache_offload_coordinator.py`: install the actual format bundle, copy it synchronously without changing the scheduler completion protocol, and publish CPU state only after copy completion.
- Modify `moe_infinity/serving/memory_manager.py`, `moe_infinity/memory/memory_coordinator.py`, `moe_infinity/memory/kv_cache_manager.py`, and `moe_infinity/entrypoints/big_modeling.py`: format-aware block sizing plus prepare/copy/commit abort safety.
- Modify `moe_infinity/utils/config.py`, `moe_infinity/entrypoints/openai/api_server_v2.py`, and `moe_infinity/serving/engine.py`: opt-in configuration and model capability wiring.
- Create `tests/python/unit/test_kv_cache_format.py`: format API, calibration, bytes, and capability tests.
- Modify `tests/python/unit/test_attention_types.py`, `tests/python/integration/test_paged_attention_backend.py`, `tests/python/ops/test_paged_attention.py`, `tests/python/serving/test_kv_cache.py`, `tests/python/serving/test_model_runner.py`, `tests/python/serving/test_flashinfer_model_runner.py`, `tests/python/serving/test_flashinfer_kv_cache.py`, `tests/python/dflash/test_kv_truncate.py`, `tests/python/integration/test_kv_cache_swap.py`, `tests/python/unit/test_kv_swap_recovery.py`, and `tests/python/integration/test_flashinfer_offload_wiring.py`: focused regression, single-owner handoff, and coordinator-installation coverage.
- Create `tests/cuda/test_kv_cache_quantization.cu`: extension-level calibration and quantized decode checks under CUDA error checking.
- Modify `tests/cuda/CMakeLists.txt`: build the CUDA test through the repository's Torch/CUDA CMake path and explicitly link `extensions/kernel/paged_attention_int8.cu`, rather than relying only on `TORCH_SRC_LIST` or the broad kernel glob.
- Create `tests/python/integration/test_kv_cache_quantized_quality.py`: prefill/decode numerical and GQA gates.
- Create `tests/python/e2e/test_kv_cache_quantized_perplexity.py`: opt-in model quality gate.
- Modify `tests/python/e2e/test_kv_parity.py` and `tests/python/e2e/test_kv_memory.py`: logit/token and exact storage/peak-memory gates.
- Create `benchmarks/serving/kv_cache_quantization.py`: long-context A/B matrix and JSON output.
- Create `tests/python/unit/test_kv_cache_benchmark.py`: benchmark CLI, matrix, schema, and parent-directory creation tests.
- Modify `docs/configuration.md`, `docs/serving.md`, `docs/benchmarking.md`, and `CHANGELOG.md`: precision semantics, support matrix, rollback, benchmark runbook, and release status.
- Create `tests/python/unit/test_kv_cache_docs.py`: documentation contract assertions.
- Produce `artifacts/kv-cache-quantization/qwen3.json` only after the benchmark writer has created its parent directory and the strict matrix succeeds; otherwise retain the same JSON as a CI artifact.

### Task 1: Define the format API, calibration, and byte accounting

**Files:**
- Create: `moe_infinity/runtime/kv_cache_format.py`
- Modify: `moe_infinity/runtime/attention_types.py:8-22`
- Create: `tests/python/unit/test_kv_cache_format.py`
- Modify: `tests/python/unit/test_attention_types.py:1-35`

- [ ] **Step 1: Write failing tests for format parsing, scales, storage bytes, and fallback decisions**

```python
# tests/python/unit/test_kv_cache_format.py
import pytest
import torch

from moe_infinity.runtime.kv_cache_format import (
    KVCacheBackendCapabilities,
    KVCacheFormat,
    KVCacheModelInfo,
    quantize_tokenwise_symmetric,
    resolve_kv_cache_format,
)


def test_int8_sym_calibration_round_trip_bound() -> None:
    x = torch.tensor([[[0.0, -1.0, 0.25, 2.0, 127.03125]]], dtype=torch.float32)
    q, scale = quantize_tokenwise_symmetric(x)
    restored = q.float() * scale.float().unsqueeze(-1)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert int(q.min()) >= -127
    assert torch.all(torch.isfinite(scale))
    assert torch.all(127 * scale.float() >= x.abs().amax(dim=-1))
    assert torch.all((restored - x).abs() <= scale.float().unsqueeze(-1) / 2 + 1e-6)


def test_int8_sym_zero_and_fp16_rounding_adversaries() -> None:
    raw_scales = torch.tensor([0.0, 1.0001, 63.999, 65504.0])
    x = torch.stack((raw_scales * 127, -raw_scales * 127), dim=-1)
    q, scale = quantize_tokenwise_symmetric(x)
    restored = q.float() * scale.float().unsqueeze(-1)
    assert torch.equal(restored[0], torch.zeros_like(restored[0]))
    assert torch.all(127 * scale.float() >= x.abs().amax(dim=-1))
    assert torch.all((restored - x).abs() <= scale.float().unsqueeze(-1) / 2 + 1e-6)


def test_int8_sym_rejects_nonfinite_or_unrepresentable_scale() -> None:
    with pytest.raises(ValueError, match="finite"):
        quantize_tokenwise_symmetric(torch.tensor([[float("inf")]]))
    with pytest.raises(ValueError, match="FP16 scale"):
        quantize_tokenwise_symmetric(torch.tensor([[65504.0 * 128.0]]))


def test_int8_sym_page_bytes_include_scales() -> None:
    fmt = KVCacheFormat.parse("int8_sym")
    assert fmt.page_size_bytes(block_size=16, num_kv_heads=8, head_dim=128) == 33280


@pytest.mark.parametrize(
    "device,backend,capabilities,expected_backend,expected_reason",
    [
        (torch.device("cuda"), "native", KVCacheBackendCapabilities(False, True, True, None), "native_int8", None),
        (torch.device("cpu"), "auto", KVCacheBackendCapabilities(False, False, True, "native_int8_module_unavailable"), "sdpa_dequant", "cpu_sdpa_dequant"),
        (torch.device("cuda"), "auto", KVCacheBackendCapabilities(False, False, True, "native_int8_binding_missing"), "sdpa_dequant", "native_int8_binding_missing"),
        (torch.device("cuda"), "flashinfer", KVCacheBackendCapabilities(True, True, True, None), "native_int8", "flashinfer_no_int8_sym_contract"),
    ],
)
def test_resolver_consumes_device_kernel_and_backend_capability(
    device, backend, capabilities, expected_backend, expected_reason
) -> None:
    decision = resolve_kv_cache_format(
        requested="int8_sym",
        model=KVCacheModelInfo(32, 8, 128, False),
        device=device,
        backend_preference=backend,
        capabilities=capabilities,
        allow_fallback=True,
    )
    assert decision.effective_format.name == "int8_sym"
    assert decision.execution_backend == expected_backend
    assert decision.reason == expected_reason


def test_mla_request_falls_back_visibly() -> None:
    decision = resolve_kv_cache_format(
        requested="int8_sym",
        model=KVCacheModelInfo(num_attention_heads=16, num_kv_heads=1, head_dim=512, is_mla=True),
        device=torch.device("cuda"),
        backend_preference="auto",
        capabilities=KVCacheBackendCapabilities(False, True, True, None),
        allow_fallback=True,
    )
    assert decision.effective_format.name == "native"
    assert decision.reason == "mla_not_validated"


def test_unknown_format_fails_before_allocation() -> None:
    with pytest.raises(ValueError, match="unsupported KV cache format"):
        KVCacheFormat.parse("int4")
```

Update `tests/python/unit/test_attention_types.py` with:

```python
def test_page_size_bytes_int8_sym() -> None:
    spec = KVCacheSpec(
        num_kv_heads=8,
        head_dim=128,
        dtype=torch.float16,
        block_size=16,
        format_name="int8_sym",
    )
    assert spec.page_size_bytes == 33280
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/python/unit/test_kv_cache_format.py tests/python/unit/test_attention_types.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'moe_infinity.runtime.kv_cache_format'` or `KVCacheSpec.__init__() got an unexpected keyword argument 'format_name'`.

- [ ] **Step 3: Implement the minimal format contract and reference quantizer**

```python
# moe_infinity/runtime/kv_cache_format.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class KVCacheFormatName(str, Enum):
    NATIVE = "native"
    INT8_SYM = "int8_sym"


@dataclass(frozen=True)
class KVCacheFormat:
    name: KVCacheFormatName
    payload_dtype: torch.dtype | None
    scale_dtype: torch.dtype | None

    @classmethod
    def parse(cls, value: str) -> "KVCacheFormat":
        if value == "native":
            return cls(KVCacheFormatName.NATIVE, None, None)
        if value == "int8_sym":
            return cls(KVCacheFormatName.INT8_SYM, torch.int8, torch.float16)
        raise ValueError(f"unsupported KV cache format: {value}")

    def page_size_bytes(
        self,
        *,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        execution_dtype: torch.dtype = torch.float16,
    ) -> int:
        values = 2 * block_size * num_kv_heads * head_dim
        if self.name is KVCacheFormatName.NATIVE:
            return values * torch.empty((), dtype=execution_dtype).element_size()
        scale_values = 2 * block_size * num_kv_heads
        return values + scale_values * 2


@dataclass(frozen=True)
class KVCacheModelInfo:
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    is_mla: bool


@dataclass(frozen=True)
class KVCacheFormatDecision:
    requested_format: KVCacheFormat
    effective_format: KVCacheFormat
    execution_backend: str
    reason: str | None


@dataclass(frozen=True)
class KVCacheBackendCapabilities:
    flashinfer_available: bool
    native_int8_binding_available: bool
    sdpa_available: bool
    native_int8_unavailable_reason: str | None


def resolve_kv_cache_format(
    *,
    requested: str,
    model: KVCacheModelInfo,
    device: torch.device,
    backend_preference: str,
    capabilities: KVCacheBackendCapabilities,
    allow_fallback: bool,
) -> KVCacheFormatDecision:
    requested_format = KVCacheFormat.parse(requested)
    native = KVCacheFormat.parse("native")
    if requested_format.name is KVCacheFormatName.NATIVE:
        return KVCacheFormatDecision(requested_format, native, "existing", None)
    format_failure = None
    if model.is_mla:
        format_failure = "mla_not_validated"
    elif model.num_attention_heads % model.num_kv_heads != 0:
        format_failure = "invalid_gqa_ratio"
    elif model.head_dim % 8 != 0:
        format_failure = "head_dim_not_divisible_by_8"
    if format_failure is not None:
        if not allow_fallback:
            raise RuntimeError(f"KV cache format int8_sym unavailable: {format_failure}")
        return KVCacheFormatDecision(requested_format, native, "existing", format_failure)
    flashinfer_reason = (
        "flashinfer_no_int8_sym_contract"
        if backend_preference == "flashinfer" and capabilities.flashinfer_available
        else None
    )
    if device.type == "cuda" and capabilities.native_int8_binding_available:
        return KVCacheFormatDecision(requested_format, requested_format, "native_int8", flashinfer_reason)
    if capabilities.sdpa_available:
        reason = (
            "cpu_sdpa_dequant"
            if device.type != "cuda"
            else capabilities.native_int8_unavailable_reason
            or "native_int8_binding_missing"
        )
        return KVCacheFormatDecision(requested_format, requested_format, "sdpa_dequant", flashinfer_reason or reason)
    if not allow_fallback:
        raise RuntimeError("KV cache format int8_sym unavailable: no_int8_execution_backend")
    return KVCacheFormatDecision(requested_format, native, "existing", "no_int8_execution_backend")


def quantize_tokenwise_symmetric(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.all(torch.isfinite(x)):
        raise ValueError("KV cache input must contain only finite values")
    amax = x.float().abs().amax(dim=-1)
    raw_scale = amax / 127.0
    zero = torch.zeros((), dtype=torch.float16, device=x.device)
    inf = torch.full((), float("inf"), dtype=torch.float16, device=x.device)
    min_scale = torch.nextafter(zero, inf).float()
    raw_scale = torch.clamp(raw_scale, min=min_scale)
    candidate = raw_scale.to(torch.float16)
    scale = torch.where(
        candidate.float() < raw_scale,
        torch.nextafter(candidate, inf),
        candidate,
    )
    if not torch.all(torch.isfinite(scale)):
        raise ValueError("KV values require an unrepresentable FP16 scale")
    q = torch.clamp(torch.round(x.float() / scale.float().unsqueeze(-1)), -127, 127)
    return q.to(torch.int8), scale
```

Extend `KVCacheSpec`:

```python
@dataclass
class KVCacheSpec:
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    block_size: int
    format_name: str = "native"

    @property
    def page_size_bytes(self) -> int:
        return KVCacheFormat.parse(self.format_name).page_size_bytes(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            execution_dtype=self.dtype,
        )
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest -q tests/python/unit/test_kv_cache_format.py tests/python/unit/test_attention_types.py`

Expected: all tests pass; the existing FP16/BF16/FP32 byte tests remain unchanged.

- [ ] **Step 5: Commit the format contract**

```bash
git add moe_infinity/runtime/kv_cache_format.py moe_infinity/runtime/attention_types.py tests/python/unit/test_kv_cache_format.py tests/python/unit/test_attention_types.py
git commit -m "feat: define KV cache format abstraction"
```

### Task 2: Add the canonical layered store, chunk writes, and prefix reads

**Files:**
- Modify: `moe_infinity/runtime/kv_cache_format.py`
- Modify: `tests/python/unit/test_kv_cache_format.py`

- [ ] **Step 1: Write failing store tests for native/INT8 chunk writes and prefix reads**

```python
from moe_infinity.runtime.kv_cache_format import allocate_layered_paged_kv_store


@pytest.mark.parametrize("format_name", ["native", "int8_sym"])
def test_layered_store_writes_chunk_and_reads_logical_prefix(format_name: str) -> None:
    store = allocate_layered_paged_kv_store(
        owner_id="unit-store-prefix",
        format_name=format_name,
        num_layers=2,
        num_blocks=3,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        execution_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert store.owner_id == "unit-store-prefix"
    key = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8) / 31
    value = key + 0.5
    store.write_chunk(layer_idx=1, key_chunk=key, value_chunk=value, slot_mapping=torch.tensor([0, 5, 11]))
    k_hat, v_hat = store.read_prefix(
        layer_idx=1,
        block_table=torch.tensor([0, 1, 2]),
        seq_len=12,
        execution_dtype=torch.float32,
    )
    tolerance = 0.0 if format_name == "native" else 2e-2
    torch.testing.assert_close(k_hat[[0, 5, 11]], key, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(v_hat[[0, 5, 11]], value, atol=tolerance, rtol=tolerance)
    assert k_hat.shape == (12, 2, 8)
    assert v_hat.shape == (12, 2, 8)


def test_int8_store_bytes_match_layers_times_pages() -> None:
    store = allocate_layered_paged_kv_store(
        owner_id="unit-store-bytes", format_name="int8_sym",
        num_layers=2, num_blocks=3, block_size=4,
        num_kv_heads=2, head_dim=8, execution_dtype=torch.float16,
        device=torch.device("cpu"),
    )
    assert store.owner_id == "unit-store-bytes"
    assert store.nbytes == 2 * 3 * KVCacheFormat.parse("int8_sym").page_size_bytes(
        block_size=4, num_kv_heads=2, head_dim=8
    )


def test_page_chunk_snapshot_preserves_layer_and_page_axes() -> None:
    store = _make_layered_store(format_name="int8_sym", num_layers=2, num_blocks=4)
    chunk = store.snapshot_page_chunk([1, 3], target_device=torch.device("cpu"))
    assert chunk.page_ids == (1, 3)
    assert chunk.payload.shape[:2] == (2, 2)  # layers, selected pages
    assert chunk.scales is not None and chunk.scales.shape[:2] == (2, 2)
```

- [ ] **Step 2: Run the storage tests and verify RED**

Run: `pytest -q tests/python/unit/test_kv_cache_format.py -k 'store or page_chunk'`

Expected: import fails because `allocate_layered_paged_kv_store` is not defined.

- [ ] **Step 3: Implement `LayeredPagedKVStore` and `LayeredKVPageChunk`**

Use one layer-major canonical protocol; native and INT8 formats differ only in tensor dtype/scale presence:

```python
@dataclass(frozen=True)
class LayeredKVPageChunk:
    page_ids: tuple[int, ...]
    format: KVCacheFormat
    payload: torch.Tensor  # [layers, selected_pages, 2, block_size, kv_heads, head_dim]
    scales: torch.Tensor | None = None  # [layers, selected_pages, 2, block_size, kv_heads]


@dataclass
class LayeredPagedKVStore:
    owner_id: str
    format: KVCacheFormat
    num_layers: int
    num_pages: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    execution_dtype: torch.dtype
    payload: torch.Tensor
    scales: torch.Tensor | None = None

    @property
    def tensors(
        self,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        if self.scales is not None:
            return self.payload, self.scales
        return (self.payload,)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors)

    def write_chunk(
        self,
        *,
        layer_idx: int,
        key_chunk: torch.Tensor,
        value_chunk: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        expected = (key_chunk.shape[0], self.num_kv_heads, self.head_dim)
        if key_chunk.shape != expected or value_chunk.shape != expected:
            raise ValueError(f"K/V chunk must have shape {expected}")
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx outside LayeredPagedKVStore")
        slots = slot_mapping.to(self.payload.device, dtype=torch.long)
        pages, offsets = slots // self.block_size, slots % self.block_size
        if bool(torch.any(slots < 0)) or bool(torch.any(pages >= self.num_pages)):
            raise ValueError("slot_mapping points outside allocated KV pages")
        key_chunk = key_chunk.to(self.payload.device, dtype=self.execution_dtype)
        value_chunk = value_chunk.to(self.payload.device, dtype=self.execution_dtype)
        if self.format.name is KVCacheFormatName.NATIVE:
            self.payload[layer_idx, pages, 0, offsets] = key_chunk
            self.payload[layer_idx, pages, 1, offsets] = value_chunk
            return
        if self.scales is None:
            raise RuntimeError("int8_sym storage requires K/V scales")
        key_q, key_scale = quantize_tokenwise_symmetric(key_chunk)
        value_q, value_scale = quantize_tokenwise_symmetric(value_chunk)
        self.payload[layer_idx, pages, 0, offsets] = key_q
        self.payload[layer_idx, pages, 1, offsets] = value_q
        self.scales[layer_idx, pages, 0, offsets] = key_scale
        self.scales[layer_idx, pages, 1, offsets] = value_scale

    def read_prefix(
        self, *, layer_idx: int, block_table: torch.Tensor,
        seq_len: int, execution_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gather physical pages in logical order, reshape to
        # [pages * block_size, kv_heads, head_dim], trim to seq_len,
        # and multiply by the matching gathered FP16 scales for INT8.
        return _read_layer_prefix(self, layer_idx, block_table, seq_len, execution_dtype)

    def snapshot_page_chunk(
        self, page_ids: list[int], target_device: torch.device
    ) -> LayeredKVPageChunk:
        return _snapshot_page_chunk_blocking(self, page_ids, target_device)

    def restore_page_chunk(
        self, destination_page_ids: list[int], chunk: LayeredKVPageChunk
    ) -> None:
        _restore_page_chunk_blocking(self, destination_page_ids, chunk)

    def paged_kernel_layer_view(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        layer = self.payload[layer_idx]
        key = (
            layer[:, 0]
            .reshape(self.num_pages, self.block_size, self.num_kv_heads, self.head_dim // 8, 8)
            .permute(0, 2, 3, 1, 4)
        )
        value = layer[:, 1].permute(0, 2, 3, 1)
        if self.scales is None:
            return key, value, None, None
        return (
            key,
            value,
            self.scales[layer_idx, :, 0].permute(0, 2, 1),
            self.scales[layer_idx, :, 1].permute(0, 2, 1),
        )

    def flashinfer_layer_view(self, layer_idx: int) -> torch.Tensor:
        if self.format.name is not KVCacheFormatName.NATIVE:
            raise RuntimeError("FlashInfer layer view requires native KV format")
        return self.payload[layer_idx]


def _read_layer_prefix(
    store: LayeredPagedKVStore,
    layer_idx: int,
    block_table: torch.Tensor,
    seq_len: int,
    execution_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_count = (seq_len + store.block_size - 1) // store.block_size
    page_ids = block_table[:page_count].to(store.payload.device, dtype=torch.long)
    pages = store.payload[layer_idx].index_select(0, page_ids)
    key = pages[:, 0].reshape(page_count * store.block_size, store.num_kv_heads, store.head_dim)[:seq_len]
    value = pages[:, 1].reshape(page_count * store.block_size, store.num_kv_heads, store.head_dim)[:seq_len]
    if store.scales is None:
        return key.to(execution_dtype), value.to(execution_dtype)
    scales = store.scales[layer_idx].index_select(0, page_ids)
    key_scale = scales[:, 0].reshape(page_count * store.block_size, store.num_kv_heads)[:seq_len]
    value_scale = scales[:, 1].reshape(page_count * store.block_size, store.num_kv_heads)[:seq_len]
    return (
        key.to(execution_dtype) * key_scale.to(execution_dtype).unsqueeze(-1),
        value.to(execution_dtype) * value_scale.to(execution_dtype).unsqueeze(-1),
    )


def _snapshot_page_chunk_blocking(
    store: LayeredPagedKVStore,
    page_ids: list[int],
    target_device: torch.device,
) -> LayeredKVPageChunk:
    index = torch.tensor(page_ids, device=store.payload.device, dtype=torch.long)
    payload = store.payload.index_select(1, index).to(target_device, non_blocking=False).clone()
    scales = None
    if store.scales is not None:
        scales = store.scales.index_select(1, index).to(target_device, non_blocking=False).clone()
    return LayeredKVPageChunk(tuple(page_ids), store.format, payload, scales)


def _restore_page_chunk_blocking(
    store: LayeredPagedKVStore,
    destination_page_ids: list[int],
    chunk: LayeredKVPageChunk,
) -> None:
    if store.format != chunk.format or len(destination_page_ids) != len(chunk.page_ids):
        raise ValueError("page chunk format/count does not match destination store")
    expected = (store.num_layers, len(destination_page_ids), 2, store.block_size, store.num_kv_heads, store.head_dim)
    if tuple(chunk.payload.shape) != expected:
        raise ValueError(f"page chunk payload must have shape {expected}")
    index = torch.tensor(destination_page_ids, device=store.payload.device, dtype=torch.long)
    store.payload.index_copy_(1, index, chunk.payload.to(store.payload.device, non_blocking=False))
    if store.scales is not None:
        if chunk.scales is None:
            raise ValueError("INT8 page chunk is missing scales")
        store.scales.index_copy_(1, index, chunk.scales.to(store.scales.device, non_blocking=False))
```

Implement `allocate_layered_paged_kv_store(owner_id, format_name, num_layers, num_blocks, block_size, num_kv_heads, head_dim, execution_dtype, device)` to construct exactly one layer-major store and validate all dimensions before allocation. The functions above return `[seq_len, kv_heads, head_dim]` K/V prefixes and synchronously clone/restore all layers and scales.

- [ ] **Step 4: Run storage and existing backend tests**

Run: `pytest -q tests/python/unit/test_kv_cache_format.py tests/python/integration/test_paged_attention_backend.py`

Expected: all tests pass.

- [ ] **Step 5: Commit storage semantics**

```bash
git add moe_infinity/runtime/kv_cache_format.py tests/python/unit/test_kv_cache_format.py
git commit -m "feat: add canonical layered KV store"
```

### Task 3: Integrate quantized writes, prefill, and decode fallback in the runtime backend

**Files:**
- Modify: `moe_infinity/runtime/attention_backend.py:78-449`
- Modify: `moe_infinity/kernel/paged_attention_ops.py:13-181`
- Modify: `moe_infinity/models/qwen3_paged_attention.py:24-155`
- Modify: `tests/python/integration/test_paged_attention_backend.py`
- Modify: `tests/python/unit/test_flashinfer_attention_backend.py`

- [ ] **Step 1: Write failing backend tests for INT8 write, GQA prefill, decode, and FlashInfer fallback**

```python
def _make_int8_backend() -> PagedAttentionBackend:
    return PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2, head_dim=8, dtype=torch.float32,
            block_size=4, format_name="int8_sym",
        ),
        num_gpu_blocks=10,
        device=torch.device("cpu"),
    )


def test_int8_write_uses_payload_and_scales() -> None:
    backend = _make_int8_backend()
    store = _make_layered_store(format_name="int8_sym", num_layers=2, num_blocks=10)
    backend.bind_store(store, owner_id="serving-engine-1")
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    backend.write_chunk(layer_idx=1, key_chunk=key, value_chunk=value, slot_mapping=torch.arange(4))
    assert backend.store is store
    assert store.payload.dtype == torch.int8
    assert store.scales is not None
    assert store.scales.shape == (2, 10, 2, 4, 2)


def test_int8_gqa_prefill_and_decode_match_reference() -> None:
    torch.manual_seed(7)
    backend = _make_int8_backend()
    backend.bind_store(
        _make_layered_store(format_name="int8_sym", num_layers=2, num_blocks=10),
        owner_id="serving-engine-1",
    )
    query = torch.randn(4, 4, 8)
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    prefill = backend.forward(query, key, value, layer_idx=1, attention_metadata=_prefill_metadata(4))
    decode = backend.forward(query[-1:], key[-1:], value[-1:], layer_idx=1, attention_metadata=_decode_metadata(4))
    assert prefill.shape == (4, 4, 8)
    assert decode.shape == (1, 4, 8)
    assert backend.execution_backend == "sdpa_dequant"
```

In `test_flashinfer_attention_backend.py` add:

```python
def test_int8_request_does_not_allocate_duplicate_flashinfer_cache(monkeypatch) -> None:
    _enable_fake_flashinfer(monkeypatch)
    spec = _spec()
    spec.format_name = "int8_sym"
    backend = attention_backend_module.PagedAttentionBackend(spec, 4, torch.device("cpu"))
    store = _make_layered_store(format_name="int8_sym", num_layers=2, num_blocks=4)
    backend.bind_store(store, owner_id="serving-engine-1")
    assert backend._fi_kv_cache is None
    assert backend.store is store
    assert backend.effective_format == "int8_sym"
    assert backend.format_decision_reason == "flashinfer_no_int8_sym_contract"
```

- [ ] **Step 2: Run the backend tests and verify RED**

Run: `pytest -q tests/python/integration/test_paged_attention_backend.py tests/python/unit/test_flashinfer_attention_backend.py`

Expected: failures report missing `storage`, `execution_backend`, or INT8 dtype rejection in the existing path.

- [ ] **Step 3: Make the backend store-bound and layer-aware**

Remove direct `k_cache`, `v_cache`, and `_fi_kv_cache` capacity allocation from the serving backend. Add a one-time identity binding:

```python
@property
def store(self) -> LayeredPagedKVStore:
    if self._store is None:
        raise RuntimeError("paged attention backend has no bound LayeredPagedKVStore")
    return self._store

def bind_store(self, store: LayeredPagedKVStore, *, owner_id: str) -> None:
    if store.owner_id != owner_id:
        raise RuntimeError("KV store owner_id does not match binding owner")
    if self._store is not None and self._store is not store:
        raise RuntimeError("paged attention backend is already bound to a different KV store")
    if (
        store.num_pages != self.num_gpu_blocks
        or store.num_kv_heads != self.spec.num_kv_heads
        or store.head_dim != self.spec.head_dim
        or store.block_size != self.spec.block_size
        or store.format.name.value != self.spec.format_name
    ):
        raise RuntimeError("KV store shape/format does not match attention backend spec")
    self._store = store
    self._store_owner_id = owner_id
```

Add required `layer_idx: int` to `forward` and expose `write_chunk(layer_idx, key_chunk, value_chunk, slot_mapping)`; it delegates to `store.write_chunk`. Prefill computes with incoming unquantized K/V while persisting the chunk in the shared store. Decode takes the layer’s payload/scales from `store`; SDPA fallback calls `store.read_prefix`. Update `Qwen3PagedAttention.forward` and every paged model adapter to pass `self.layer_idx`. Copy `decision.execution_backend` and `decision.reason` to `execution_backend` and `format_decision_reason`. For `int8_sym`, do not allocate a duplicate FlashInfer cache; `flashinfer_no_int8_sym_contract` is informational, not a storage-format fallback.

- [ ] **Step 4: Add INT8 fallback dispatch to `paged_attention_ops.py`**

Extend the protocol and function signature:

```python
def paged_attention_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    block_size: int = 16,
    max_seq_len: Optional[int] = None,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
) -> torch.Tensor:
```

Dispatch `paged_attention_int8_v1` only when both scale tensors exist and all tensors are CUDA. Reject exactly-one-scale and non-INT8 payload combinations. Otherwise, `_paged_attention_sdpa_fallback` must dequantize selected blocks in FP32, expand MHA/GQA heads, form rank-4 SDPA inputs, execute FP32 SDPA, and cast only the result back to the original query dtype:

```python
query_dtype = query.dtype
outputs: list[torch.Tensor] = []
for seq_idx in range(batch_size):
    seq_len = int(seq_lens[seq_idx].item())
    if seq_len <= 0:
        outputs.append(torch.zeros_like(query[seq_idx]))
        continue

    # Existing page selection restores K/V as [Hkv, L, D]. For INT8,
    # multiply by the matching restored [Hkv, L, 1] FP16 scales in FP32.
    k = k.to(dtype=torch.float32)
    v = v.to(dtype=torch.float32)
    if key_scale is not None and value_scale is not None:
        k = k * selected_key_scale.to(dtype=torch.float32).unsqueeze(-1)
        v = v * selected_value_scale.to(dtype=torch.float32).unsqueeze(-1)
    head_ratio = num_heads // num_kv_heads
    if head_ratio > 1:
        k = k.repeat_interleave(head_ratio, dim=0)
        v = v.repeat_interleave(head_ratio, dim=0)

    q_sdpa = query[seq_idx].to(dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    k_sdpa = k.unsqueeze(0)
    v_sdpa = v.unsqueeze(0)
    assert q_sdpa.shape == (1, num_heads, 1, query.shape[-1])
    assert k_sdpa.shape == (1, num_heads, seq_len, query.shape[-1])
    assert v_sdpa.shape == (1, num_heads, seq_len, query.shape[-1])
    assert q_sdpa.dtype == k_sdpa.dtype == v_sdpa.dtype == torch.float32
    out_fp32 = _run_sdpa(q_sdpa, k_sdpa, v_sdpa, float(scale))
    outputs.append(out_fp32[0, :, 0, :].to(dtype=query_dtype))
return torch.stack(outputs, dim=0)
```

Do not pass rank-3 tensors or model-dtype Q/K/V into the fallback SDPA call. Keep `_run_sdpa`'s compatibility handling for PyTorch versions without the `scale=` keyword, but both branches must receive the FP32 rank-4 tensors above.

- [ ] **Step 5: Run runtime tests and verify GREEN**

Run: `pytest -q tests/python/integration/test_paged_attention_backend.py tests/python/unit/test_flashinfer_attention_backend.py tests/python/ops/test_paged_attention.py`

Expected: all tests pass on CPU; CUDA-only tests skip when CUDA is absent.

- [ ] **Step 6: Commit runtime integration**

```bash
git add moe_infinity/runtime/attention_backend.py moe_infinity/kernel/paged_attention_ops.py moe_infinity/models/qwen3_paged_attention.py tests/python/integration/test_paged_attention_backend.py tests/python/unit/test_flashinfer_attention_backend.py tests/python/ops/test_paged_attention.py
git commit -m "feat: integrate INT8 KV cache runtime fallback"
```

### Task 4: Add the native INT8 paged decode CUDA kernel

**Files:**
- Modify: `extensions/kernel/paged_attention.cu:13-253`
- Create: `extensions/kernel/paged_attention_int8.cuh`
- Create: `extensions/kernel/paged_attention_int8.cu`
- Modify: `setup.py:278-345`
- Modify: `tests/python/ops/test_paged_attention.py`
- Create: `tests/cuda/test_kv_cache_quantization.cu`
- Modify: `tests/cuda/CMakeLists.txt:162-195`

- [ ] **Step 1: Add failing Python CUDA parity cases**

```python
@pytest.mark.gpu
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (32, 8)])
def test_int8_paged_attention_matches_dequantized_sdpa(num_heads, num_kv_heads):
    torch.manual_seed(19)
    batch, head_dim, seq_len, block_size = 2, 128, 33, 16
    query = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k, v, block_tables, seq_lens = make_paged_kv(batch, num_kv_heads, head_dim, seq_len, block_size)
    k = k.cuda().half()
    v = v.cuda().half()
    k_vectors = k.permute(0, 3, 1, 2, 4).reshape(-1, num_kv_heads, head_dim)
    v_vectors = v.permute(0, 3, 1, 2).reshape(-1, num_kv_heads, head_dim)
    k_q_vectors, k_scale_vectors = quantize_tokenwise_symmetric(k_vectors)
    v_q_vectors, v_scale_vectors = quantize_tokenwise_symmetric(v_vectors)
    num_blocks = k.shape[0]
    k_q = (
        k_q_vectors.reshape(num_blocks, block_size, num_kv_heads, head_dim // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_q = v_q_vectors.reshape(num_blocks, block_size, num_kv_heads, head_dim).permute(0, 2, 3, 1)
    k_scale = k_scale_vectors.reshape(num_blocks, block_size, num_kv_heads).permute(0, 2, 1).contiguous()
    v_scale = v_scale_vectors.reshape(num_blocks, block_size, num_kv_heads).permute(0, 2, 1).contiguous()
    actual = paged_attention_fwd(
        query, k_q, v_q, block_tables.cuda(), seq_lens.cuda(),
        1.0 / math.sqrt(head_dim), num_kv_heads, block_size,
        key_scale=k_scale, value_scale=v_scale,
    )
    references = []
    for seq_idx in range(batch):
        logical_len = int(seq_lens[seq_idx])
        page_count = math.ceil(logical_len / block_size)
        page_ids = block_tables[seq_idx, :page_count].to(k_q.device, dtype=torch.long)
        k_tokens = (
            k_q.index_select(0, page_ids)
            .permute(0, 3, 1, 2, 4)
            .reshape(page_count * block_size, num_kv_heads, head_dim)[:logical_len]
            .float()
        )
        v_tokens = (
            v_q.index_select(0, page_ids)
            .permute(0, 3, 1, 2)
            .reshape(page_count * block_size, num_kv_heads, head_dim)[:logical_len]
            .float()
        )
        k_scales = (
            k_scale.index_select(0, page_ids)
            .permute(0, 2, 1)
            .reshape(page_count * block_size, num_kv_heads)[:logical_len]
            .float()
        )
        v_scales = (
            v_scale.index_select(0, page_ids)
            .permute(0, 2, 1)
            .reshape(page_count * block_size, num_kv_heads)[:logical_len]
            .float()
        )
        k_tokens = k_tokens * k_scales.unsqueeze(-1)
        v_tokens = v_tokens * v_scales.unsqueeze(-1)
        repeat = num_heads // num_kv_heads
        q_sdpa = query[seq_idx].to(dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        k_sdpa = (
            k_tokens.repeat_interleave(repeat, dim=1)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .to(dtype=torch.float32)
        )
        v_sdpa = (
            v_tokens.repeat_interleave(repeat, dim=1)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .to(dtype=torch.float32)
        )
        assert q_sdpa.shape == (1, num_heads, 1, head_dim)
        assert k_sdpa.shape == (1, num_heads, logical_len, head_dim)
        assert v_sdpa.shape == (1, num_heads, logical_len, head_dim)
        assert q_sdpa.dtype == k_sdpa.dtype == v_sdpa.dtype == torch.float32
        out_sdpa_fp32 = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            scale=1.0 / math.sqrt(head_dim), is_causal=False,
        )
        assert out_sdpa_fp32.shape == (1, num_heads, 1, head_dim)
        references.append(out_sdpa_fp32[0, :, 0, :].to(dtype=query.dtype))
    reference = torch.stack(references)
    assert reference.shape == actual.shape == (batch, num_heads, head_dim)
    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)
```

This exact test runs for MHA `(num_heads=num_kv_heads=8)` and GQA `(num_heads=32, num_kv_heads=8)`. The independent oracle uses rank-4 SDPA inputs `[batch=1, heads, query_len=1, head_dim]` and `[1, heads, prefix_len, head_dim]`; it must not call `paged_attention_fwd`, `_paged_attn`, or the CUDA kernel under test.

- [ ] **Step 2: Rebuild and verify RED reaches the missing binding**

Run: `pip install --no-build-isolation -e . && pytest -q tests/python/ops/test_paged_attention.py -k int8`

Expected: fail with `_paged_attn` missing `paged_attention_int8_v1`.

- [ ] **Step 3: Extract and implement the reusable INT8 kernel helper, then register it in the existing extension**

Create `extensions/kernel/paged_attention_int8.cuh` with the declaration of `paged_attention_int8_v1(torch::Tensor& out, const torch::Tensor& query, const torch::Tensor& key_cache, const torch::Tensor& value_cache, const torch::Tensor& key_scale, const torch::Tensor& value_scale, int num_kv_heads, float scale, const torch::Tensor& block_tables, const torch::Tensor& seq_lens, int block_size, int max_seq_len)`. Create `extensions/kernel/paged_attention_int8.cu` with that function and the device kernel whose signature includes `const int8_t* k_cache`, `const int8_t* v_cache`, `const at::Half* k_scale`, and `const at::Half* v_scale`. This reusable source must not define `PYBIND11_MODULE`. At each K load use:

```cpp
const float k = static_cast<float>(k_cache[k_idx]) *
                static_cast<float>(k_scale[scale_idx]);
qk += static_cast<float>(q[q_idx]) * k;
```

At each V load use:

```cpp
const float v = static_cast<float>(v_cache[v_idx]) *
                static_cast<float>(v_scale[scale_idx]);
acc += weight * v;
```

The reusable C++ launcher must enforce: query/output are FP16/BF16/FP32 and same dtype; payloads are INT8; scales are FP16; scales have `[blocks, kv_heads, block_size]`; `num_heads % num_kv_heads == 0`; head size and block layouts match; all tensors are contiguous CUDA tensors on one device. Include `paged_attention_int8.cuh` from `paged_attention.cu` and register both bindings in the single existing module:

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention_v1", &paged_attention_v1, "PagedAttention V1 forward pass");
  m.def("paged_attention_int8_v1", &paged_attention_int8_v1,
        "INT8 symmetric paged attention forward pass");
}
```

Add the reusable implementation to the production extension explicitly:

```python
_PAGED_ATTN_SOURCES = [
    "extensions/kernel/paged_attention.cu",
    "extensions/kernel/paged_attention_int8.cu",
]
```

`paged_attention.cu` remains the only file containing `PYBIND11_MODULE`, so linking the helper does not create duplicate module definitions.

- [ ] **Step 4: Add the CUDA launch/error test to the repository CMake target list**

`tests/cuda/test_kv_cache_quantization.cu` must include `../../extensions/kernel/paged_attention_int8.cuh`, allocate one MHA and one GQA case, call the same `paged_attention_int8_v1` launcher linked into the extension, call `cudaDeviceSynchronize()`, assert `cudaGetLastError() == cudaSuccess`, and compare against an independent host FP32 dequantized attention implementation with `2e-2` absolute/relative tolerance. Add a dedicated target that explicitly compiles and links the shared implementation; merely appending the test to `TORCH_SRC_LIST` is insufficient:

```cmake
add_executable(
  test_kv_cache_quantization
  test_kv_cache_quantization.cu
  ${CMAKE_SOURCE_DIR}/../../extensions/kernel/paged_attention_int8.cu
)
target_link_libraries(
  test_kv_cache_quantization
  cutlass ${CUDA_LIBRARIES} ${torch_python_LIBRARY} ${Python3_LIBRARIES} ${TORCH_LIBRARIES}
)
target_include_directories(
  test_kv_cache_quantization PRIVATE
  ${CONDA_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/../../extensions/kernel
)
```

Keep this target outside the `TORCH_SRC_LIST` loop so it links only the intended helper source and never the broad `${KERNEL_SRC}` glob (which includes `paged_attention.cu` and its `PYBIND11_MODULE`). Then use the repository CMake convention:

Run: `cmake -S tests/cuda -B build/tests-cuda -DCMAKE_BUILD_TYPE=Release && cmake --build build/tests-cuda --target test_kv_cache_quantization -j2 && compute-sanitizer --tool memcheck build/tests-cuda/test_kv_cache_quantization`

Expected: exit 0, output includes `PASS int8_sym MHA` and `PASS int8_sym GQA`, and compute-sanitizer reports `ERROR SUMMARY: 0 errors`.

- [ ] **Step 5: Rebuild once and verify GREEN**

Run: `pip install --no-build-isolation -e . && pytest -q tests/python/ops/test_paged_attention.py -k 'int8 or gqa or paged_vs_sdpa'`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the CUDA kernel**

```bash
git add extensions/kernel/paged_attention.cu extensions/kernel/paged_attention_int8.cuh extensions/kernel/paged_attention_int8.cu setup.py tests/python/ops/test_paged_attention.py tests/cuda/test_kv_cache_quantization.cu tests/cuda/CMakeLists.txt
git commit -m "feat: add native INT8 paged KV decode"
```

### Task 5: Make serving cache lifecycle and swap format-aware

**Files:**
- Modify: `moe_infinity/serving/kv_cache.py:140-440`
- Modify: `moe_infinity/serving/engine.py:84-150`
- Modify: `moe_infinity/serving/model_runner.py:27-154`
- Modify: `moe_infinity/runtime/model_offload.py:523-579`
- Modify: `moe_infinity/entrypoints/big_modeling.py:401-446`
- Modify: `moe_infinity/serving/scheduler.py:474-523`
- Modify: `moe_infinity/engine/kv_cache_offload_coordinator.py:15-146`
- Modify: `moe_infinity/memory/kv_cache_manager.py:144-223`
- Modify: `tests/python/serving/test_kv_cache.py`
- Modify: `tests/python/dflash/test_kv_truncate.py`
- Modify: `tests/python/integration/test_kv_cache_swap.py`
- Modify: `tests/python/unit/test_kv_swap_recovery.py`
- Modify: `tests/python/serving/test_model_runner.py`
- Modify: `tests/python/serving/test_flashinfer_model_runner.py`
- Modify: `tests/python/serving/test_flashinfer_kv_cache.py`

- [ ] **Step 1: Write failing serving lifecycle tests**

Add a cache factory that creates one `LayeredPagedKVStore` and passes it into `PagedKVCache`; write deterministic K/V through `store.write_chunk`, then assert:

```python
def test_int8_swap_round_trip_preserves_payload_and_scales() -> None:
    cache = _make_int8_cache(num_blocks=4)
    cache.allocate_sequence(1, 8)
    slots = torch.arange(8)
    key = torch.randn(8, 2, 8)
    value = torch.randn(8, 2, 8)
    cache.store.write_chunk(layer_idx=0, key_chunk=key, value_chunk=value, slot_mapping=slots)
    before = cache.store.snapshot_page_chunk(cache.get_block_table(1), torch.device("cpu"))
    cache.swap_out(1, release_gpu_blocks=True)
    assert cache.sequence_residency(1) == "swapped"
    assert cache.get_block_table(1) == []
    cache.swap_in(1)
    after = cache.store.snapshot_page_chunk(cache.get_block_table(1), torch.device("cpu"))
    torch.testing.assert_close(after.payload, before.payload)
    torch.testing.assert_close(after.scales, before.scales)


def test_int8_truncate_swapped_cache_slices_scale_pages() -> None:
    cache = _make_int8_cache(num_blocks=4)
    cache.allocate_sequence(1, 8)
    cache.swap_out(1, release_gpu_blocks=True)
    cache.truncate_tokens(1, 4)
    swapped = cache._swapped_cpu_buffers[1]
    assert swapped.payload.shape[1] == 1
    assert swapped.scales is not None and swapped.scales.shape[1] == 1


def test_swap_then_truncate_releases_each_gpu_block_once(monkeypatch) -> None:
    cache = _make_int8_cache(num_blocks=4)
    cache.allocate_sequence(1, 8)
    released: list[int] = []
    original_free = cache.block_allocator.free
    def recording_free(block_ids: list[int]) -> None:
        released.extend(block_ids)
        original_free(block_ids)
    monkeypatch.setattr(cache.block_allocator, "free", recording_free)
    cache.swap_out(1, release_gpu_blocks=True)
    cache.truncate_tokens(1, 4)
    assert sorted(released) == [0, 1]
    assert len(released) == len(set(released))


def test_free_gpu_blocks_rejects_swapped_sequence() -> None:
    cache = _make_int8_cache(num_blocks=4)
    cache.allocate_sequence(1, 8)
    cache.swap_out(1, release_gpu_blocks=True)
    with pytest.raises(RuntimeError, match="does not own GPU blocks"):
        cache.free_gpu_blocks(1)


def test_serving_engine_shares_one_store_across_cache_runner_and_backend() -> None:
    engine = _build_test_serving_engine(kv_cache_format="int8_sym")
    backend = engine.model_runner.attention_backend
    assert engine.kv_cache.store is engine.model_runner.kv_store
    assert backend.store is engine.kv_cache.store
    assert engine.kv_cache.store.owner_id == engine.kv_store_owner_id


def test_model_runner_rejects_store_identity_mismatch_before_forward() -> None:
    model = CountingModel()
    store_a = _make_layered_store(owner_id="engine-a")
    store_b = _make_layered_store(owner_id="engine-b")
    backend = _make_backend()
    backend.bind_store(store_a, owner_id="engine-a")
    with pytest.raises(RuntimeError, match="KV store identity mismatch"):
        ModelRunner(
            model=model,
            engine=object(),
            kv_store=store_b,
            attention_backend=backend,
            owner_id="engine-b",
            device=torch.device("cpu"),
        )
    assert model.forward_calls == 0


def test_backend_rejects_rebind_to_equal_shape_different_owner() -> None:
    backend = _make_backend()
    store_a = _make_layered_store(owner_id="engine-a")
    store_b = _make_layered_store(owner_id="engine-b")
    backend.bind_store(store_a, owner_id="engine-a")
    with pytest.raises(RuntimeError, match="already bound to a different KV store"):
        backend.bind_store(store_b, owner_id="engine-b")
```

- [ ] **Step 2: Run lifecycle tests and verify RED**

Run: `pytest -q tests/python/serving/test_kv_cache.py tests/python/dflash/test_kv_truncate.py tests/python/unit/test_kv_swap_recovery.py`

Expected: construction fails because `PagedKVCache`/`ModelRunner` do not yet accept the canonical shared store, or backend/store identity is not enforced.

- [ ] **Step 3: Install or adopt exactly one serving store before any request can run**

Make `big_modeling._build_native_components` construct the native components as one ownership unit: `LayeredPagedKVStore`, `PagedAttentionBackend`, and `KVCacheOffloadCoordinator` receive one `owner_id`; bind the backend and coordinator to that store before returning the component map. Pass `kv_store` into `OffloadEngine` beside `attention_backend`, expose `get_paged_kv_store()`, and never let either component allocate payload tensors internally.

Make `ContinuousBatchingEngine` the serving handoff validator. Adopt `engine.get_paged_kv_store()` and `engine.get_attention_backend()` when present. For lightweight test engines with neither installed, construct both once and call `engine.install_paged_kv_components(store, backend, owner_id)`; reject the partial state where only one exists. Then hand the same objects to all serving consumers:

```python
store = engine.get_paged_kv_store()
attention_backend = engine.get_attention_backend()
if (store is None) != (attention_backend is None):
    raise RuntimeError("engine must install KV store and attention backend together")
if store is None:
    owner_id = f"serving-engine:{id(self)}"
    store = allocate_layered_paged_kv_store(
        owner_id=owner_id,
        format_name=format_decision.effective_format.name.value,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        execution_dtype=self.dtype,
        device=self.device,
    )
    assert store.owner_id == owner_id
    attention_backend = PagedAttentionBackend(spec, num_blocks, self.device)
    attention_backend.bind_store(store, owner_id=owner_id)
    engine.install_paged_kv_components(store, attention_backend, owner_id)
self.kv_store = store
self.kv_store_owner_id = store.owner_id
self.attention_backend = attention_backend
self._validate_store_dimensions_against_config(store)
self.kv_cache = PagedKVCache(
    store=self.kv_store,
    owner_id=self.kv_store_owner_id,
)
self.model_runner = ModelRunner(
    model,
    engine,
    kv_store=self.kv_store,
    attention_backend=attention_backend,
    owner_id=self.kv_store_owner_id,
    device=self.device,
)
```

`PagedKVCache.__init__` must not allocate payloads; validate `store.owner_id == owner_id` and derive all dimensions/dtype/format from `store`. `ModelRunner.__init__` must reject unless `attention_backend.store is kv_store` and owner IDs match; repeat the object-identity assertion immediately before `set_paged_context` so later accidental rebinding fails before model forward. Remove `_get_attention_backend` discovery for this serving path. For `int8_sym`, prefill uses incoming K/V and decode uses the bound store. Native FlashInfer must consume a layer view owned by `LayeredPagedKVStore`, not allocate `_fi_kv_cache` separately. The serving `kv_cache_ratio` must participate when the store is first constructed; when adopting an installed store, report its actual `num_pages` and reject a conflicting configured capacity instead of creating a second store.

Remove attention execution ownership from `PagedKVCache`: delete `_fi_workspace`, `_fi_prefill`, `_fi_decode`, `_use_flashinfer`, and `_compute_attention`. `ModelRunner` invokes only the explicitly supplied `PagedAttentionBackend`. Update `tests/python/serving/test_flashinfer_kv_cache.py` to assert native FlashInfer receives `store.flashinfer_layer_view(layer_idx)` and that no second capacity tensor is allocated.

- [ ] **Step 4: Make residency ownership and rollback transitions explicit**

Add `SequenceResidency.RESIDENT` and `SequenceResidency.SWAPPED` plus a per-sequence state record containing `num_tokens`, `owned_gpu_block_ids`, and `swapped_storage`. Only `owned_gpu_block_ids` may be returned to `BlockAllocator`. Implement transitions as follows:

```text
RESIDENT --swap_out(release_gpu_blocks=True)--> SWAPPED:
  synchronously clone payload/scales; validate all owned IDs; free each ID once;
  clear owned_gpu_block_ids; publish SWAPPED only after copy and release succeed.

SWAPPED --truncate_tokens--> SWAPPED:
  update num_tokens and slice swapped payload/scales to blocks_needed;
  never call BlockAllocator.free because SWAPPED owns no GPU IDs.

SWAPPED --swap_in--> RESIDENT:
  allocate a complete replacement ID list transactionally; on allocation/copy
  failure return all newly allocated IDs and remain SWAPPED; after copy succeeds,
  install IDs, clear swapped_storage, and publish RESIDENT.

RESIDENT --truncate_tokens--> RESIDENT:
  validate the tail IDs, free only that tail once, then atomically install the
  kept ID list and new token count.
```

Keep `free_gpu_blocks` for native callers but require RESIDENT state and nonempty ownership; reject SWAPPED state instead of silently freeing stale IDs. Update `Scheduler._preempt_oldest_running_group` to call only `swap_out(seq_id, release_gpu_blocks=True)` and remove its following `free_gpu_blocks` call. Add scheduler recovery tests proving a failed swap-in leaves SWAPPED ownership and allocator counts unchanged.

For the native `KVCacheManager`/`BlockPool` path, preserve the existing prepare/copy/commit boundary: `prepare_swap_out` reserves CPU IDs without touching GPU `KVCacheBlock.ref_cnt`; the synchronous coordinator copy runs next; only a successful `commit_swap_out` calls `release_gpu_block` once per GPU block. Add `abort_swap_out(pairs)` to return reserved CPU blocks after copy failure without decrementing GPU refs. Mirror this for swap-in: copy into newly reserved GPU IDs, then commit CPU release; on failure return only the new GPU reservations. Extend `tests/python/integration/test_kv_cache_swap.py` with a shared block whose `ref_cnt=2`, inject a copy failure, and assert refcount/sequence table are unchanged; then run a successful swap and assert exactly one decrement at commit.

- [ ] **Step 5: Keep packed transfers explicitly synchronous and lifetime-safe**

Change serving `_swapped_cpu_buffers` and native coordinator `_cpu_cache` to hold `LayeredKVPageChunk`. For this first path, do not use `async_d2h`, `async_h2d`, nonblocking copies, transfer streams, pinned-buffer reuse, or persistent CUDA events. `snapshot_page_chunk` uses blocking `.to("cpu", non_blocking=False).clone()` for every payload/scale tensor; `restore_page_chunk` uses blocking `.to(target_device, non_blocking=False)`. Publish `_cpu_cache[transfer_id]` only after the complete page chunk reaches CPU; remove it only after every H2D copy and destination-page assignment finish. The page chunk remains strongly referenced for the whole call. This deliberately synchronous lifetime contract is independent of async KV completion and requires no event handoff to `UnifiedTransferScheduler`.

- [ ] **Step 6: Run lifecycle tests and verify GREEN**

Run: `pytest -q tests/python/serving/test_kv_cache.py tests/python/dflash/test_kv_truncate.py tests/python/integration/test_kv_cache_swap.py tests/python/unit/test_kv_swap_recovery.py tests/python/unit/test_kv_cache_free.py`

Expected: all pass; native tests retain exact tensor equality, INT8 tests retain exact payload/scale equality.

- [ ] **Step 7: Commit lifecycle integration**

```bash
git add moe_infinity/serving/kv_cache.py moe_infinity/serving/engine.py moe_infinity/serving/model_runner.py moe_infinity/serving/scheduler.py moe_infinity/runtime/model_offload.py moe_infinity/entrypoints/big_modeling.py moe_infinity/engine/kv_cache_offload_coordinator.py moe_infinity/memory/kv_cache_manager.py tests/python/serving/test_kv_cache.py tests/python/serving/test_model_runner.py tests/python/serving/test_flashinfer_model_runner.py tests/python/serving/test_flashinfer_kv_cache.py tests/python/dflash/test_kv_truncate.py tests/python/integration/test_kv_cache_swap.py tests/python/unit/test_kv_swap_recovery.py tests/python/unit/test_kv_cache_free.py
git commit -m "feat: preserve quantized KV metadata across lifecycle"
```

### Task 6: Wire capability detection, configuration, and memory calculations

**Files:**
- Modify: `moe_infinity/utils/config.py:17-162`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py:1778-1889`
- Modify: `moe_infinity/entrypoints/big_modeling.py:340-427`
- Modify: `moe_infinity/serving/engine.py:84-130`
- Modify: `moe_infinity/serving/memory_manager.py:114-147`
- Modify: `moe_infinity/memory/memory_coordinator.py:124-135`
- Modify: `tests/python/unit/test_utils_config.py`
- Modify: `tests/python/unit/test_memory_coordinator.py`
- Create: `tests/python/unit/test_kv_cache_capability.py`
- Modify: `tests/python/integration/test_flashinfer_offload_wiring.py`
- Modify: `tests/python/unit/test_kv_handler_registration.py`

- [ ] **Step 1: Write failing config, memory, GQA, and MLA tests**

```python
def test_kv_cache_format_defaults_native() -> None:
    config = ArcherConfig(offload_path="/tmp/x", use_native_engine=False)
    assert config.kv_cache_format == "native"
    assert config.kv_cache_allow_fallback is True


def test_int8_memory_budget_counts_payload_and_scale() -> None:
    manager = MemoryManager(device=torch.device("cpu"))
    manager._last_budget = MemoryBudget(
        total_gpu_memory_bytes=1_000_000, model_memory_bytes=0,
        expert_cache_ratio=0.0, kv_cache_ratio=1.0,
    )
    blocks = manager.get_max_kv_blocks(
        block_size=16, num_layers=2, num_heads=8, head_dim=128,
        dtype=torch.float16, format_name="int8_sym",
    )
    assert blocks == 1_000_000 // (2 * 33280)


def test_mla_metadata_is_detected_from_model_config() -> None:
    info = model_info_from_config(SimpleNamespace(
        num_attention_heads=16, num_key_value_heads=1,
        head_dim=128, kv_lora_rank=512,
    ))
    assert info.is_mla is True


def test_native_coordinator_installs_backend_storage_before_registration(monkeypatch) -> None:
    components = build_native_components_for_test(
        kv_cache_format="int8_sym", enable_kv_cache_offload=True
    )
    coordinator = components["kv_offload_coordinator"]
    backend = components["attention_backend"]
    assert coordinator._kv_store is backend.store
    assert coordinator._kv_store.format.name == "int8_sym"
    assert coordinator._transfer_scheduler is components["transfer_scheduler"]
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/python/unit/test_utils_config.py tests/python/unit/test_memory_coordinator.py tests/python/unit/test_kv_cache_capability.py`

Expected: missing config fields, function, or `format_name` argument.

- [ ] **Step 3: Add opt-in fields and CLI**

Add to `ArcherConfig`:

```python
kv_cache_format: str = field(
    default="native",
    metadata={"help": "KV storage format: native or int8_sym; default preserves model dtype."},
)
kv_cache_allow_fallback: bool = field(
    default=True,
    metadata={"help": "Allow a visible native fallback when requested KV format is unsupported."},
)
```

Validate by `KVCacheFormat.parse`. Create the OpenAI flags in `parse_args` before any task runs a server or benchmark command:

```python
parser.add_argument(
    "--kv-cache-format",
    choices=("native", "int8_sym"),
    default="native",
)
parser.add_argument(
    "--no-kv-cache-format-fallback",
    dest="kv_cache_allow_fallback",
    action="store_false",
    default=True,
)
```

Copy both into `_build_engine_config`, and pass the effective decision to `PagedKVCache`/`PagedAttentionBackend`. Include `requested_kv_cache_format`, `effective_kv_cache_format`, `kv_cache_execution_backend`, and `kv_cache_format_decision_reason` in engine stats and `/v1/config`.

- [ ] **Step 4: Detect model constraints before allocation**

`model_info_from_config` must call `get_text_config()` when available, detect GQA from `num_attention_heads`/`num_key_value_heads`, and set `is_mla` when `kv_lora_rank` is non-null or the family exposes `qk_nope_head_dim`/`qk_rope_head_dim`. Probe the native INT8 binding directly in `paged_attention_ops.py`:

```python
def probe_native_int8_binding() -> tuple[bool, str | None]:
    if _paged_attn_ops is None:
        return False, "native_int8_module_unavailable"
    binding = getattr(_paged_attn_ops, "paged_attention_int8_v1", None)
    if not callable(binding):
        return False, "native_int8_binding_missing"
    return True, None
```

Build `KVCacheBackendCapabilities` from `flashinfer_utils.HAS_FLASHINFER`, this probe result, and `callable(getattr(F, "scaled_dot_product_attention", None))`. Pass model info, resolved device, configured `attention_backend`, and these capabilities to `resolve_kv_cache_format` before calculating `num_gpu_blocks`. Do not use `HAS_PAGED_ATTN` as evidence of INT8 support.

Add fail-closed tests:

```python
def test_imported_native_module_without_int8_binding_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(paged_attention_ops, "_paged_attn_ops", SimpleNamespace(paged_attention_v1=lambda: None))
    available, reason = paged_attention_ops.probe_native_int8_binding()
    assert available is False
    assert reason == "native_int8_binding_missing"
    decision = resolve_kv_cache_format(
        requested="int8_sym",
        model=KVCacheModelInfo(32, 8, 128, False),
        device=torch.device("cuda"),
        backend_preference="native",
        capabilities=KVCacheBackendCapabilities(False, available, True, reason),
        allow_fallback=True,
    )
    assert decision.execution_backend == "sdpa_dequant"
    assert decision.reason == "native_int8_binding_missing"


def test_missing_int8_binding_and_sdpa_strict_mode_raises(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="no_int8_execution_backend"):
        resolve_kv_cache_format(
            requested="int8_sym",
            model=KVCacheModelInfo(32, 8, 128, False),
            device=torch.device("cuda"),
            backend_preference="native",
            capabilities=KVCacheBackendCapabilities(False, False, False, "native_int8_binding_missing"),
            allow_fallback=False,
        )
```

Tests must also cover CUDA/callable-native, CPU, FlashInfer preference, MLA fallback, and strict failure. Emit one warning only when effective storage falls back to native; emit an info event for backend-only reasons such as FlashInfer bypass. Never catch the strict-mode startup error in the broad backend-construction `except Exception`.

- [ ] **Step 5: Make both memory calculators descriptor-driven**

Add `format_name` to `MemoryManager.get_max_kv_blocks`; calculate `num_layers * KVCacheFormat.parse(format_name).page_size_bytes(block_size=block_size, num_kv_heads=num_heads, head_dim=head_dim, execution_dtype=dtype)`. Keep `MemoryCoordinator.compute_num_kv_blocks` generic but ensure every caller passes `KVCacheSpec.page_size_bytes * num_layers`. Tests must assert native counts are unchanged and INT8 counts include scale metadata.

- [ ] **Step 6: Install the real format bundle in the offload coordinator**

After the single `LayeredPagedKVStore` is constructed and bound to `PagedAttentionBackend`, install that same store before handler registration:

```python
kv_offload_coordinator = KVCacheOffloadCoordinator(
    kv_store=None,
    block_pool=kv_cache_manager,
    config=engine_config,
)
kv_offload_coordinator.set_kv_store(attention_backend.store)
kv_offload_coordinator.register_with_scheduler(transfer_scheduler)
```

Make `LayeredPagedKVStore`/`set_kv_store` the canonical coordinator contract. Keep a private legacy tensor adapter only if an existing non-serving caller still requires it; do not expose two public ownership protocols. If backend construction returns `None`, do not register KV swap handlers. Remove the current comment that leaves tensor installation for an unspecified later phase. `test_flashinfer_offload_wiring.py` and `test_kv_handler_registration.py` must assert store object identity, owner identity, format identity, and installation-before-registration ordering.

- [ ] **Step 7: Run focused and construction tests**

Run: `pytest -q tests/python/unit/test_utils_config.py tests/python/unit/test_memory_coordinator.py tests/python/unit/test_kv_cache_capability.py tests/python/unit/test_attention_types.py tests/python/serving/test_kv_cache.py tests/python/integration/test_flashinfer_offload_wiring.py tests/python/unit/test_kv_handler_registration.py`

Expected: all pass, including default-native regression assertions.

- [ ] **Step 8: Commit configuration, accounting, and coordinator wiring**

```bash
git add moe_infinity/utils/config.py moe_infinity/entrypoints/openai/api_server_v2.py moe_infinity/entrypoints/big_modeling.py moe_infinity/serving/engine.py moe_infinity/serving/memory_manager.py moe_infinity/memory/memory_coordinator.py moe_infinity/engine/kv_cache_offload_coordinator.py tests/python/unit/test_utils_config.py tests/python/unit/test_memory_coordinator.py tests/python/unit/test_kv_cache_capability.py tests/python/integration/test_flashinfer_offload_wiring.py tests/python/unit/test_kv_handler_registration.py
git commit -m "feat: wire opt-in KV cache format selection"
```

### Task 7: Add numerical equivalence and perplexity gates

**Files:**
- Create: `tests/python/integration/test_kv_cache_quantized_quality.py`
- Create: `tests/python/e2e/test_kv_cache_quantized_perplexity.py`
- Modify: `tests/python/e2e/test_kv_parity.py`
- Modify: `tests/python/e2e/test_kv_memory.py`

- [ ] **Step 1: Add deterministic tensor-level quality tests**

In `test_kv_cache_quantized_quality.py`, parameterize `dtype in [float16, bfloat16]`, `(q_heads, kv_heads) in [(8, 8), (32, 8)]`, `seq_len in [1, 15, 16, 17, 257]`, and prefill/decode. Build native and INT8 backends from identical tensors and assert:

```python
torch.testing.assert_close(int8_out, native_out, atol=2e-2, rtol=2e-2)
cosine = torch.nn.functional.cosine_similarity(
    int8_out.float().flatten(), native_out.float().flatten(), dim=0
)
assert float(cosine) >= 0.999
```

Include non-contiguous block tables, zero vectors, exact page boundaries, and repeated append to a partially filled page.

- [ ] **Step 2: Run integration quality tests and verify they initially expose missing edge handling**

Run: `pytest -q tests/python/integration/test_kv_cache_quantized_quality.py`

Expected before edge fixes: at least one failure identifies zero-scale, partial-page, or non-contiguous-page handling; after fixing only the identified storage/kernel defect, all cases pass.

- [ ] **Step 3: Add model logit and greedy agreement gate**

Extend `test_kv_parity.py` to run the same local GQA model twice with `kv_cache_format=native/int8_sym`, capture logits for each decode step, and require:

```python
assert mean_logit_cosine >= 0.999
assert mean_absolute_logit_error <= 0.02
assert matching_tokens / total_tokens >= 0.99
assert total_tokens >= 2048
```

Mark it `integration`, `slow`, `gpu`, and skip only when the configured local model is unavailable. Record the effective format and fail if the INT8 run fell back.

- [ ] **Step 4: Add the explicit perplexity release gate**

`test_kv_cache_quantized_perplexity.py` must load a pinned local GQA checkpoint and a pinned local WikiText-2 validation snapshot, evaluate fixed 2,048-token windows with stride 1,024, and calculate NLL from shifted logits. Require:

```python
absolute_delta = int8_ppl - native_ppl
relative_delta = absolute_delta / native_ppl
assert absolute_delta <= 0.10
assert relative_delta <= 0.01
```

Gate execution with `MOE_KV_QUANT_QUALITY=1`; a release qualification job must set the variable and may not convert threshold failures to skips. Save model revision, dataset revision, native/int8 perplexity, and effective backend in pytest output.

- [ ] **Step 5: Add measured storage and peak-memory assertions**

Extend `test_kv_memory.py` to compare native/INT8 at the same cache budget and workload. For the canonical `(block_size=16, kv_heads=8, head_dim=128)` page, assert `int8_storage_bytes == 33_280`, `native_fp16_storage_bytes == 65_536`, and ratio `== pytest.approx(0.5078125)`; retain the general `<= 0.52` gate for `head_dim >= 64`. For 8K context, require `int8_peak_delta < native_peak_delta`; synchronize and clear allocator state exactly as the existing helper does.

- [ ] **Step 6: Run the non-network gate subset**

Run: `pytest -q tests/python/integration/test_kv_cache_quantized_quality.py tests/python/e2e/test_kv_memory.py -m 'not network'`

Expected: quality tests pass; model-dependent cases either pass with configured local assets or skip with the existing explicit asset reason.

- [ ] **Step 7: Run release quality gates on the qualification host**

Run: `MOE_KV_QUANT_QUALITY=1 MOE_TEST_MODEL=Qwen/Qwen3-30B-A3B pytest -q tests/python/e2e/test_kv_cache_quantized_perplexity.py tests/python/e2e/test_kv_parity.py`

Expected: both threshold suites pass, effective format is `int8_sym`, and output reports at least 2,048 compared generated tokens.

- [ ] **Step 8: Commit quality gates**

```bash
git add tests/python/integration/test_kv_cache_quantized_quality.py tests/python/e2e/test_kv_cache_quantized_perplexity.py tests/python/e2e/test_kv_parity.py tests/python/e2e/test_kv_memory.py
git commit -m "test: gate quantized KV cache quality"
```

### Task 8: Add the long-context benchmark matrix and rollout evidence

**Files:**
- Create: `benchmarks/serving/kv_cache_quantization.py`
- Modify: `docs/benchmarking.md`

- [ ] **Step 1: Write CLI parsing and matrix tests before the benchmark implementation**

Create `tests/python/unit/test_kv_cache_benchmark.py` with:

```python
def test_default_matrix_covers_long_context_and_concurrency() -> None:
    args = parse_args([
        "--model", "/models/qwen3", "--offload-dir", "/tmp/offload",
    ])
    assert args.context_lengths == [128, 2048, 8192, 32768]
    assert args.batch_sizes == [1, 4, 16]
    assert args.formats == ["native", "int8_sym"]


def test_result_schema_separates_precisions() -> None:
    result = BenchmarkResult.example()
    payload = result.to_dict()
    assert payload["storage_format"] == "int8_sym"
    assert payload["transfer_precision"] == "int8+fp16_scale"
    assert payload["execution_dtype"] == "float16"
```

- [ ] **Step 2: Run the benchmark unit test and verify RED**

Run: `pytest -q tests/python/unit/test_kv_cache_benchmark.py`

Expected: import fails because `benchmarks.serving.kv_cache_quantization` does not exist.

- [ ] **Step 3: Implement deterministic A/B benchmark output**

The script must accept model/revision, offload dir, formats, context lengths, batch sizes, decode tokens (default 128), warmups (2), repeats (5), output JSON, and strict-fallback. Its `write_json` must call `output_path.parent.mkdir(parents=True, exist_ok=True)` before opening the artifact, so the documented commands do not rely on a pre-existing directory. For each run record:

```text
model_revision, GPU, torch/CUDA versions, requested/effective format,
format_decision_reason, attention backend, storage precision, transfer precision,
execution dtype, context length, batch size, TTFT ms, decode tok/s,
ITL p50/p99 ms, peak allocated/reserved bytes, descriptor cache bytes,
measured cache bytes, D2H/H2D swap bytes, scratch peak bytes
```

Use fixed prompt token IDs and seeds. Construct a fresh engine per format, synchronize around timers, run warmups outside measurement, and fail under `--strict-fallback` if requested and effective formats differ.

- [ ] **Step 4: Verify the benchmark CLI and artifact writer exist before invoking them**

Run: `pytest -q tests/python/unit/test_kv_cache_benchmark.py`

Expected: all pass.

Run: `python benchmarks/serving/kv_cache_quantization.py --help`

Expected: exit 0 and help lists `--formats`, `--context-lengths`, `--batch-sizes`, `--strict-fallback`, and `--output-json`.

- [ ] **Step 5: Run a smoke benchmark after the CLI/artifact writer test passes**

Run on a qualification GPU: `python benchmarks/serving/kv_cache_quantization.py --model Qwen/Qwen3-30B-A3B --offload-dir /local/ssd/qwen3-kv --context-lengths 128 2048 --batch-sizes 1 --decode-tokens 16 --repeats 1 --strict-fallback --output-json /tmp/kv-cache-smoke.json`

Expected: JSON contains four successful rows (two formats by two lengths), with no INT8 fallback and explicit precision fields.

- [ ] **Step 6: Run the full matrix and evaluate fixed gates**

Run: `python benchmarks/serving/kv_cache_quantization.py --model Qwen/Qwen3-30B-A3B --offload-dir /local/ssd/qwen3-kv --context-lengths 128 2048 8192 32768 --batch-sizes 1 4 16 --decode-tokens 128 --warmups 2 --repeats 5 --strict-fallback --output-json artifacts/kv-cache-quantization/qwen3.json`

Expected: INT8 measured storage ratio is `<= 0.52`, peak memory is lower at 8K/32K, and 2K decode throughput is at least 90% of native. Performance misses block rollout but never alter quality tolerances.

- [ ] **Step 7: Commit benchmark tooling**

```bash
git add benchmarks/serving/kv_cache_quantization.py tests/python/unit/test_kv_cache_benchmark.py docs/benchmarking.md
git commit -m "bench: add KV cache quantization matrix"
```

### Task 9: Document opt-in rollout, fallback, and rollback

**Files:**
- Modify: `docs/configuration.md`
- Modify: `docs/serving.md`
- Modify: `docs/benchmarking.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add documentation assertions**

Create `tests/python/unit/test_kv_cache_docs.py`:

```python
from pathlib import Path


def test_kv_cache_docs_state_opt_in_and_precision_contract() -> None:
    text = Path("docs/configuration.md").read_text()
    assert "`kv_cache_format`" in text
    assert "default: `native`" in text
    assert "Storage precision" in text
    assert "Transfer precision" in text
    assert "Attention execution precision" in text
    assert "MLA" in text and "fallback" in text


def test_serving_docs_include_one_command_rollback() -> None:
    text = Path("docs/serving.md").read_text()
    assert "--kv-cache-format native" in text
    assert "effective_kv_cache_format" in text
```

- [ ] **Step 2: Run docs tests and verify RED**

Run: `pytest -q tests/python/unit/test_kv_cache_docs.py`

Expected: assertions fail because the new fields and rollback command are undocumented.

- [ ] **Step 3: Document exact operational behavior**

Document:

```bash
# Opt in
python -m moe_infinity.entrypoints.openai.api_server_v2 \
  --model Qwen/Qwen3-30B-A3B --offload-dir /local/ssd/qwen3-kv \
  --host 127.0.0.1 --kv-cache-format int8_sym

# Strict qualification: refuse fallback
python -m moe_infinity.entrypoints.openai.api_server_v2 \
  --model Qwen/Qwen3-30B-A3B --offload-dir /local/ssd/qwen3-kv \
  --host 127.0.0.1 --kv-cache-format int8_sym \
  --no-kv-cache-format-fallback

# Immediate rollback; native remains the default
python -m moe_infinity.entrypoints.openai.api_server_v2 \
  --model Qwen/Qwen3-30B-A3B --offload-dir /local/ssd/qwen3-kv \
  --host 127.0.0.1 --kv-cache-format native
```

Explain the precision table, scale layout, online calibration, exact memory formula, supported MHA/GQA scope, MLA fallback, FlashInfer-to-built-in mixed backend choice, synchronous packed swap, quality thresholds, and metrics used to verify effective format. State that KIVI/KVQuant are motivation only and that no universal 2-bit support is claimed.

- [ ] **Step 4: Run docs tests and verify GREEN**

Run: `pytest -q tests/python/unit/test_kv_cache_docs.py`

Expected: all pass.

- [ ] **Step 5: Commit docs and release note**

```bash
git add docs/configuration.md docs/serving.md docs/benchmarking.md CHANGELOG.md tests/python/unit/test_kv_cache_docs.py
git commit -m "docs: describe opt-in quantized KV cache"
```

### Task 10: Final regression and release qualification

**Files:**
- No production file changes expected
- Test evidence: `artifacts/kv-cache-quantization/`

- [ ] **Step 1: Run static checks on every changed Python file**

Run: `ruff check moe_infinity/runtime/kv_cache_format.py moe_infinity/runtime/attention_types.py moe_infinity/runtime/attention_backend.py moe_infinity/runtime/model_offload.py moe_infinity/kernel/paged_attention_ops.py moe_infinity/models/qwen3_paged_attention.py moe_infinity/serving/kv_cache.py moe_infinity/serving/model_runner.py moe_infinity/serving/scheduler.py moe_infinity/engine/kv_cache_offload_coordinator.py moe_infinity/serving/memory_manager.py moe_infinity/memory/memory_coordinator.py moe_infinity/memory/kv_cache_manager.py moe_infinity/utils/config.py moe_infinity/entrypoints/big_modeling.py moe_infinity/entrypoints/openai/api_server_v2.py moe_infinity/serving/engine.py benchmarks/serving/kv_cache_quantization.py tests/python/unit/test_kv_cache_format.py tests/python/integration/test_kv_cache_quantized_quality.py tests/python/e2e/test_kv_cache_quantized_perplexity.py`

Expected: exit 0 with no diagnostics.

- [ ] **Step 2: Run the complete non-slow regression suite**

Run: `pytest -q -m 'not slow and not network'`

Expected: all tests pass; GPU tests may skip only for an unavailable GPU/optional dependency, not because INT8 was requested and silently fell back.

- [ ] **Step 3: Run CUDA correctness under sanitizer**

Run: `cmake -S tests/cuda -B build/tests-cuda -DCMAKE_BUILD_TYPE=Release && cmake --build build/tests-cuda --target test_kv_cache_quantization -j2 && compute-sanitizer --tool memcheck build/tests-cuda/test_kv_cache_quantization`

Expected: CMake builds the registered repository target, MHA/GQA checks pass, and compute-sanitizer reports `ERROR SUMMARY: 0 errors`.

- [ ] **Step 4: Run strict quality and long-context qualification**

Run: `MOE_KV_QUANT_QUALITY=1 MOE_TEST_MODEL=Qwen/Qwen3-30B-A3B pytest -q tests/python/e2e/test_kv_cache_quantized_perplexity.py tests/python/e2e/test_kv_parity.py tests/python/e2e/test_kv_memory.py`

Expected: perplexity, logit, token agreement, and memory gates all pass with `effective_kv_cache_format=int8_sym`.

Run: `python benchmarks/serving/kv_cache_quantization.py --model Qwen/Qwen3-30B-A3B --offload-dir /local/ssd/qwen3-kv --context-lengths 128 2048 8192 32768 --batch-sizes 1 4 16 --decode-tokens 128 --warmups 2 --repeats 5 --strict-fallback --output-json artifacts/kv-cache-quantization/qwen3.json`

Expected: complete 24-row matrix, no fallback, fixed memory/performance gates satisfied.

- [ ] **Step 5: Prove fallback and rollback separately**

Run the capability tests and one MLA startup probe:

```bash
pytest -q tests/python/unit/test_kv_cache_capability.py
python -m moe_infinity.entrypoints.openai.api_server_v2 \
  --model zai-org/GLM-5.2-FP8 --offload-dir /local/ssd/glm \
  --kv-cache-format int8_sym --host 127.0.0.1
```

Expected: capability tests pass; the MLA probe logs `requested=int8_sym effective=native reason=mla_not_validated`. Re-run with `--no-kv-cache-format-fallback`; expected startup failure contains `mla_not_validated` before cache allocation.

- [ ] **Step 6: Record rollout decision**

Keep the default `native`. Enable `int8_sym` only for canary deployments with the strict qualification matrix attached. Promote beyond canary only after two supported GQA checkpoints on each supported GPU architecture pass all quality gates. Roll back by setting `kv_cache_format=native`; no cache migration or async-drain protocol is needed because cache allocations are process-local and format-homogeneous.

- [ ] **Step 7: Commit only generated qualification metadata if repository policy permits it**

```bash
git add artifacts/kv-cache-quantization/qwen3.json
git commit -m "test: record quantized KV qualification"
```

If benchmark artifacts are intentionally ignored by repository policy, retain them in CI artifacts and do not create this commit.

## Completion checklist

- [ ] Default behavior and native cache byte counts are unchanged.
- [ ] `int8_sym` stores INT8 payload plus FP16 per-token/per-head scales.
- [ ] Canonical page accounting is `32,768` payload bytes + `512` scale bytes = `33,280` bytes (`0.5078125` of native FP16).
- [ ] Quantization uses the exact upward-rounded stored FP16 scale and passes zero, rounding-boundary, and unrepresentable-scale tests.
- [ ] Storage, transfer, and execution precision are separately visible in API/stats/benchmarks.
- [ ] Device, native-kernel, FlashInfer preference, and SDPA capability all affect the resolver exactly as the capability table specifies.
- [ ] Native INT8 capability requires callable `paged_attention_int8_v1`; an imported module without that binding fails closed with `native_int8_binding_missing`.
- [ ] Every `allocate_layered_paged_kv_store(...)` call supplies `owner_id`, and its caller/test immediately asserts the returned store retained that identity.
- [ ] One `LayeredPagedKVStore` object and owner ID are shared by `PagedKVCache`, `ModelRunner`, the executed attention backend, and the coordinator before serving; equal-shape different-owner stores are rejected before model forward.
- [ ] Store protocol uses page for physical allocation, chunk for writes/transfers, and prefix for logical attention/DFlash reads while preserving existing block-hash compatibility names.
- [ ] Append, page boundary, partial rollback, full rollback, free, swap-out, and swap-in preserve payload/scale consistency without double release or premature refcount changes.
- [ ] Packed payload/scales use the documented blocking transfer lifetime; no pinned buffer or CUDA event outlives a coordinator call.
- [ ] `KVCacheOffloadCoordinator.set_kv_store(attention_backend.store)` runs before handler registration and preserves store/owner object identity.
- [ ] Prefill uses incoming model-dtype K/V; decode uses INT8 native CUDA or explicit SDPA dequantization, where Q/K/V are rank-4 FP32 for the SDPA call and the result is cast back to the query dtype.
- [ ] CUDA parity uses rank-valid `[1, Hq, 1, D]` queries and `[1, Hq, L, D]` dequantized FP32 SDPA K/V for both MHA and GQA, casts the FP32 oracle output to query dtype, and remains independent of paged-attention code.
- [ ] The INT8 device kernel/launcher lives in reusable `paged_attention_int8.cuh`/`.cu` files explicitly linked by both `_PAGED_ATTN_SOURCES` and the dedicated `tests/cuda/CMakeLists.txt` target; adding only the test to `TORCH_SRC_LIST` is not accepted.
- [ ] MHA and GQA are validated; MLA visibly falls back or fails under strict mode.
- [ ] FlashInfer native behavior is unchanged; INT8 selects the built-in path without duplicate native cache allocation.
- [ ] Numerical, logit, perplexity, memory, sanitizer, and long-context gates pass.
- [ ] Server CLI flags and benchmark artifact writer exist and are tested before any documented server/benchmark invocation.
- [ ] The feature remains opt-in with a one-setting rollback and no dependency on async KV completion.
