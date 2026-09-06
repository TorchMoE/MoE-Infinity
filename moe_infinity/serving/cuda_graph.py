from __future__ import annotations

import math
import os
import threading
from collections import Counter
from dataclasses import dataclass, field

import torch

from moe_infinity.runtime.paged_kv_storage import (
    PagedKVStorage,
    canonical_device,
)

from .batch import BatchMetadata
from .model_runner import ModelRunner, PreparedDecodeBuffers

FALLBACK_REASONS = (
    "disabled",
    "kill_switch",
    "non_cuda",
    "not_decode",
    "no_batch_bucket",
    "no_context_bucket",
    "block_table_too_wide",
    "missing_capability",
    "active_model_hooks",
    "archer_callbacks",
    "transfer_scheduler",
    "expert_dispatcher",
    "kv_offload",
    "flashinfer_plan_path",
    "dynamic_allocations",
    "native_paged_required",
    "mla_layout_unsupported",
    "kv_storage_mismatch",
    "paged_class_unregistered",
    "layer_idx_invalid",
    "layer_write_unproven",
    "insufficient_memory",
    "capture_failed",
    "quarantined",
    "invalidated",
)


@dataclass(frozen=True, order=True)
class GraphKey:
    batch_size: int
    context_size: int


@dataclass(frozen=True)
class GraphDecision:
    eligible: bool
    reason: str
    key: GraphKey | None = None


@dataclass
class GraphExecutionStats:
    captures: int = 0
    replays: int = 0
    capture_failures: int = 0
    graph_pool_bytes: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)


class _GraphMemoryUnavailable(RuntimeError):
    pass


@dataclass
class _CapturedGraphState:
    graph: object
    buffers: PreparedDecodeBuffers
    output_logits: torch.Tensor
    generation: int


class _DefaultCudaOps:
    """Injectable indirection over ``torch.cuda`` so gate/lifecycle logic runs
    deterministically on hosts without a GPU during tests."""

    @property
    def available(self) -> bool:
        return torch.cuda.is_available()

    def memory_allocated(self, device: torch.device) -> int:
        return int(torch.cuda.memory_allocated(device))


def _normalize_buckets(buckets: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted({value for value in buckets if value > 0}))


def _is_cuda_device(device: torch.device) -> bool:
    return getattr(device, "type", None) == "cuda"


def _devices_match(a: torch.device, b: torch.device) -> bool:
    """Compare two devices.

    When both are CUDA-typed, compare ``(type, index)`` exactly so that
    ``cuda:0`` and ``cuda:1`` differ even on hosts where CUDA is unavailable
    (``canonical_device`` would otherwise collapse both to CPU). Otherwise fall
    back to canonical comparison.
    """
    if _is_cuda_device(a) and _is_cuda_device(b):
        a_index = a.index if a.index is not None else 0
        b_index = b.index if b.index is not None else 0
        return a_index == b_index
    return canonical_device(a) == canonical_device(b)


class CudaGraphRunner:
    def __init__(
        self,
        model_runner: ModelRunner,
        storage: PagedKVStorage | None,
        *,
        enabled: bool = False,
        batch_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        context_buckets: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096),
        warmup_iters: int = 2,
        max_graph_memory_bytes: int = 0,
    ) -> None:
        self.model_runner = model_runner
        self.storage = storage
        self.enabled = bool(enabled)
        self.batch_buckets = _normalize_buckets(batch_buckets)
        self.context_buckets = _normalize_buckets(context_buckets)
        if self.enabled and (
            not self.batch_buckets or not self.context_buckets
        ):
            raise ValueError(
                "enabled CUDA graph runner requires non-empty bucket sets"
            )
        if warmup_iters < 1:
            raise ValueError("warmup_iters must be at least 1")
        if max_graph_memory_bytes < 0:
            raise ValueError("max_graph_memory_bytes must be non-negative")
        self.warmup_iters = warmup_iters
        self.max_graph_memory_bytes = max_graph_memory_bytes

        self.generation = 0
        self._graphs: dict[GraphKey, _CapturedGraphState] = {}
        self._quarantined: dict[GraphKey, str] = {}
        self._stats = GraphExecutionStats()
        self._scratch_block_ids: list[int] | None = None
        self._closed = False
        self._lock = threading.RLock()
        self._cuda_ops: object = _DefaultCudaOps()

    def check_eligibility(self, batch: BatchMetadata) -> GraphDecision:
        if self._closed or not self.enabled:
            return GraphDecision(False, "disabled")
        if os.environ.get("MOE_DISABLE_CUDA_GRAPHS") == "1":
            return GraphDecision(False, "kill_switch")
        if not getattr(self._cuda_ops, "available", False):
            return GraphDecision(False, "non_cuda")

        if not self._is_decode_batch(batch):
            return GraphDecision(False, "not_decode")

        batch_bucket = self._select_bucket(
            self.batch_buckets, len(batch.seq_ids)
        )
        if batch_bucket is None:
            return GraphDecision(False, "no_batch_bucket")

        required_context = max(
            (context_len + 1 for context_len in batch.context_lengths),
            default=1,
        )
        context_bucket = self._select_bucket(
            self.context_buckets, required_context
        )
        if context_bucket is None:
            return GraphDecision(False, "no_context_bucket")

        if self.storage is not None:
            max_blocks = math.ceil(context_bucket / self.storage.block_size)
            if any(len(row) > max_blocks for row in batch.block_tables):
                return GraphDecision(False, "block_table_too_wide")

        key = GraphKey(batch_size=batch_bucket, context_size=context_bucket)
        if key in self._quarantined:
            return GraphDecision(False, "quarantined")

        capability_reason = self._check_capability(key)
        if capability_reason != "eligible":
            return GraphDecision(False, capability_reason)

        return GraphDecision(True, "eligible", key=key)

    def _check_capability(self, key: GraphKey) -> str:
        capability = self.model_runner.decode_graph_capability()

        if self.storage is None:
            if not capability.safe:
                return capability.reason
            return "native_paged_required"

        if not capability.safe:
            return capability.reason
        if capability.storage_owner_id != self.storage.owner_id:
            return "kv_storage_mismatch"

        registry = self.model_runner.paged_attention_registry
        if registry is None:
            return "paged_class_unregistered"
        registry_reason = getattr(registry, "reason", "eligible")
        if registry_reason != "eligible":
            return registry_reason
        bindings = list(getattr(registry, "bindings", []))
        if not bindings:
            return "paged_class_unregistered"

        proof_set = {
            (proof.class_fqn, proof.layer_idx)
            for proof in capability.layer_write_proofs
        }
        binding_set = {
            (binding.class_fqn, binding.layer_idx) for binding in bindings
        }
        if proof_set != binding_set:
            return "layer_write_unproven"

        if not _devices_match(
            self.storage.spec.device, self.model_runner.device
        ):
            return "kv_storage_mismatch"

        return "eligible"

    def check_state_eligibility(
        self, state: _CapturedGraphState
    ) -> GraphDecision:
        if self._validate_state_devices(state):
            return GraphDecision(True, "eligible")
        return GraphDecision(False, "kv_storage_mismatch")

    def _validate_state_devices(self, state: _CapturedGraphState) -> bool:
        if self.storage is None:
            return False
        target = self.storage.spec.device
        tensors = list(state.buffers.tensor_values()) + [state.output_logits]
        return all(tensor.device == target for tensor in tensors)

    @staticmethod
    def _select_bucket(buckets: tuple[int, ...], required: int) -> int | None:
        for bucket in buckets:
            if bucket >= required:
                return bucket
        return None

    @staticmethod
    def _is_decode_batch(batch: BatchMetadata) -> bool:
        if len(batch.seq_ids) == 0 or batch.total_tokens == 0:
            return False
        if any(batch.is_prefill):
            return False
        if any(seq_len != 1 for seq_len in batch.seq_lengths):
            return False
        return batch.total_tokens == len(batch.seq_ids)

    def _ensure_scratch(self) -> None:
        if self._scratch_block_ids is not None:
            return
        if self.storage is None:
            raise _GraphMemoryUnavailable("no storage bound for scratch")
        try:
            self._scratch_block_ids = self.storage.reserve_graph_scratch_blocks(
                max(self.batch_buckets)
            )
        except RuntimeError as exc:
            raise _GraphMemoryUnavailable(str(exc)) from exc

    def _scratch_rows(
        self, *, start: int, stop: int, context_size: int
    ) -> list[int]:
        self._ensure_scratch()
        assert self._scratch_block_ids is not None
        return self._scratch_block_ids[start:stop]

    def _capture_key(self, key: GraphKey) -> _CapturedGraphState:
        assert self.storage is not None
        self._ensure_scratch()
        assert self._scratch_block_ids is not None

        buffers = self.model_runner.allocate_decode_buffers(
            batch_bucket=key.batch_size, context_bucket=key.context_size
        )
        self._fill_scratch_metadata(buffers)
        captured_generation = self.generation

        device = self.storage.spec.device
        stream = torch.cuda.Stream(device=device)
        warmup_logits: torch.Tensor | None = None
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_iters):
                warmup_logits = self.model_runner.forward_prepared_decode(
                    buffers
                )
        stream.synchronize()

        before = self._cuda_ops.memory_allocated(device)
        vocab = self.model_runner._resolve_vocab_size()
        logits_dtype = (
            warmup_logits.dtype if warmup_logits is not None else torch.float32
        )
        static_output_logits = torch.empty(
            (key.batch_size, vocab), device=device, dtype=logits_dtype
        )

        for tensor in buffers.tensor_values():
            if tensor.device != device:
                raise ValueError(
                    "prepared buffer device does not match storage device"
                )
        if static_output_logits.device != device:
            raise ValueError("output logits device does not match storage")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_logits = self.model_runner.forward_prepared_decode(buffers)
            static_output_logits.copy_(captured_logits)

        after = self._cuda_ops.memory_allocated(device)
        pool = max(0, after - before)
        if (
            self.max_graph_memory_bytes > 0
            and self._stats.graph_pool_bytes + pool
            > self.max_graph_memory_bytes
        ):
            del graph
            raise _GraphMemoryUnavailable(
                "captured graph exceeds max_graph_memory_bytes"
            )

        state = _CapturedGraphState(
            graph=graph,
            buffers=buffers,
            output_logits=static_output_logits,
            generation=captured_generation,
        )
        if captured_generation == self.generation:
            self._graphs[key] = state
            self._stats.captures += 1
            self._stats.graph_pool_bytes += pool
        return state

    def _fill_scratch_metadata(self, buffers: PreparedDecodeBuffers) -> None:
        assert self.storage is not None
        assert self._scratch_block_ids is not None
        block_size = self.storage.spec.block_size
        metadata = buffers.attention_metadata

        buffers.input_ids.zero_()
        buffers.position_ids.zero_()
        buffers.attention_mask.fill_(1)
        buffers.active_rows.zero_()
        metadata.block_tables.zero_()
        metadata.seq_lens.fill_(1)
        metadata.slot_mapping.zero_()

        for row in range(buffers.batch_bucket):
            scratch_id = self._scratch_block_ids[row]
            metadata.block_tables[row, 0] = scratch_id
            metadata.slot_mapping[row] = scratch_id * block_size
        buffers.real_batch_size = 0

    def try_execute(self, batch: BatchMetadata) -> torch.Tensor | None:
        decision = self.check_eligibility(batch)
        if not decision.eligible or decision.key is None:
            self._record_fallback(decision.reason)
            return None
        with self._lock:
            state = self._graphs.get(decision.key)
            if state is None:
                try:
                    state = self._capture_key(decision.key)
                except _GraphMemoryUnavailable:
                    self._record_fallback("insufficient_memory")
                    return None
                except (RuntimeError, ValueError) as exc:
                    self._quarantined[decision.key] = type(exc).__name__
                    self._stats.capture_failures += 1
                    self._record_fallback("capture_failed")
                    return None
            if state.generation != self.generation:
                self._record_fallback("invalidated")
                return None
            if not self._validate_state_devices(state):
                self._record_fallback("kv_storage_mismatch")
                return None
            scratch_rows = self._scratch_rows(
                start=len(batch.seq_ids),
                stop=decision.key.batch_size,
                context_size=decision.key.context_size,
            )
            self.model_runner.copy_decode_batch(
                batch, state.buffers, scratch_rows
            )
            self.model_runner.prepare_batch_side_effects(batch)
            try:
                state.graph.replay()
            except Exception:
                self._quarantined[decision.key] = "replay_failed"
                self._graphs.pop(decision.key, None)
                raise
            self._stats.replays += 1
            return state.output_logits[: len(batch.seq_ids)].clone()

    def _record_fallback(self, reason: str) -> None:
        assert reason in FALLBACK_REASONS, f"unknown fallback reason: {reason}"
        self._stats.fallback_reasons[reason] += 1

    def invalidate(self, reason: str) -> None:
        with self._lock:
            if self._graphs and self.storage is not None:
                if getattr(self._cuda_ops, "available", False):
                    torch.cuda.synchronize(self.storage.spec.device)
            self._graphs.clear()
            self._quarantined.clear()
            self.generation += 1

    def close(self) -> None:
        with self._lock:
            self.invalidate("close")
            if self._scratch_block_ids is not None and self.storage is not None:
                self.storage.release_graph_scratch_blocks(
                    self._scratch_block_ids
                )
            self._scratch_block_ids = None
            self._closed = True

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "captures": self._stats.captures,
                "replays": self._stats.replays,
                "capture_failures": self._stats.capture_failures,
                "graph_pool_bytes": self._stats.graph_pool_bytes,
                "fallback_reasons": dict(self._stats.fallback_reasons),
                "graphs": len(self._graphs),
            }


__all__ = [
    "CudaGraphRunner",
    "GraphDecision",
    "GraphExecutionStats",
    "GraphKey",
    "FALLBACK_REASONS",
]
