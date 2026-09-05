"""Shared contracts for DFlash execution backends and request sessions.

The types in this module are deliberately model- and scheduler-independent.
They define the values exchanged by native and serving implementations without
requiring either implementation to own a second sampling, cache, or trace
schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import torch

CacheKind = Literal["dense_dynamic", "paged", "other"]


@dataclass(frozen=True)
class PairingEvidence:
    """Structural target/drafter compatibility, independent of execution.

    The fields are limited to the authoritative DFlash config, shape, vocab,
    mask, layer, block, and drafter-module checks.  Executor wiring and
    route-ahead observations deliberately live in :class:`ExecutorEvidence`.
    ``validated_checkpoint_scope`` is empty unless the caller explicitly knows
    the checkpoint identities covered by the validation.
    """

    valid: bool = False
    config_valid: bool = False
    dimensions_valid: bool = False
    vocab_valid: bool = False
    mask_valid: bool = False
    layers_valid: bool = False
    block_valid: bool = False
    module_valid: bool | None = None
    validated_checkpoint_scope: tuple[str, ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.valid and not all(
            (
                self.config_valid,
                self.dimensions_valid,
                self.vocab_valid,
                self.mask_valid,
                self.layers_valid,
                self.block_valid,
                self.module_valid is not False,
            )
        ):
            raise ValueError(
                "valid pairing evidence requires all checked fields"
            )
        if any(not item for item in self.validated_checkpoint_scope):
            raise ValueError(
                "validated_checkpoint_scope entries must be non-empty"
            )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "valid": self.valid,
            "config_valid": self.config_valid,
            "dimensions_valid": self.dimensions_valid,
            "vocab_valid": self.vocab_valid,
            "mask_valid": self.mask_valid,
            "layers_valid": self.layers_valid,
            "block_valid": self.block_valid,
            "module_valid": self.module_valid,
            "validated_checkpoint_scope": self.validated_checkpoint_scope,
            "failure_reason": self.failure_reason,
        }
        return result


@dataclass(frozen=True)
class ExecutorEvidence:
    """Observed executor reachability and route-ahead behavior.

    This is an immutable snapshot.  It never establishes target/drafter
    compatibility and may therefore report a reachable executor for an invalid
    pair, or an unreachable executor for a valid published pair.
    """

    wiring_reachable: bool = False
    prefetcher_present: bool = False
    attempted_layers: tuple[int, ...] = ()
    fired_layers: tuple[int, ...] = ()
    actual_expert_union: frozenset[tuple[int, int]] = frozenset()
    actual_expert_union_by_row: frozenset[tuple[int, int, int]] = frozenset()
    prefetched_bytes: int = 0
    coverage: float | None = None
    wasted_prefetch_bytes: int | None = None
    cache_hit_rate: float | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if any(
            layer < 0 for layer in self.attempted_layers + self.fired_layers
        ):
            raise ValueError("executor evidence layer ids must be non-negative")
        if not set(self.fired_layers).issubset(self.attempted_layers):
            raise ValueError(
                "fired_layers must be a subset of attempted_layers"
            )
        if any(
            layer < 0 or expert < 0
            for layer, expert in self.actual_expert_union
        ):
            raise ValueError("actual expert ids must be non-negative")
        if any(
            row < 0 or layer < 0 or expert < 0
            for row, layer, expert in self.actual_expert_union_by_row
        ):
            raise ValueError("row-aware actual expert ids must be non-negative")
        if self.prefetched_bytes < 0:
            raise ValueError("prefetched_bytes must be >= 0")
        if (
            self.wasted_prefetch_bytes is not None
            and self.wasted_prefetch_bytes < 0
        ):
            raise ValueError("wasted_prefetch_bytes must be >= 0")
        for name, value in (
            ("coverage", self.coverage),
            ("cache_hit_rate", self.cache_hit_rate),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "wiring_reachable": self.wiring_reachable,
            "prefetcher_present": self.prefetcher_present,
            "attempted_layers": self.attempted_layers,
            "fired_layers": self.fired_layers,
            "actual_expert_union": tuple(sorted(self.actual_expert_union)),
            "prefetched_bytes": self.prefetched_bytes,
            "coverage": self.coverage,
            "wasted_prefetch_bytes": self.wasted_prefetch_bytes,
            "cache_hit_rate": self.cache_hit_rate,
            "fallback_reason": self.fallback_reason,
        }
        if self.actual_expert_union_by_row:
            result["actual_expert_union_by_row"] = tuple(
                sorted(self.actual_expert_union_by_row)
            )
        return result


@dataclass(frozen=True)
class SamplingContext:
    """Request-scoped sampling policy and random-number stream."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    generator: torch.Generator | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0

    @property
    def is_sampled(self) -> bool:
        return not self.is_greedy


@dataclass(frozen=True)
class RequestSpec:
    """Backend-neutral inputs needed to start one speculative session."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    stop_token_ids: frozenset[int] = frozenset()
    sampling: SamplingContext = field(default_factory=SamplingContext)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def is_sampled(self) -> bool:
        return self.sampling.is_sampled


class NativeStepTrace(NamedTuple):
    """Canonical per-round cache and emission accounting.

    ``accept`` counts accepted draft tokens.  The anchor plus those drafts are
    cached, while the emitted bonus remains uncached for the next round.
    """

    prev_start: int
    accept: int
    start: int
    emitted_len: int
    target_cache_len: int
    draft_cache_len: int | None

    @property
    def committed_count(self) -> int:
        """Number of block tokens retained in cache by this round."""

        return self.accept + 1

    @property
    def cache_advance(self) -> int:
        return self.start - self.prev_start


@dataclass(frozen=True)
class SessionRoundResult:
    """Backend-neutral result of one draft/verify/commit round.

    ``accepted_draft_count`` records the verify rule's untruncated acceptance.
    A finished stop/budget-truncated round may emit fewer tokens, including no
    tokens for a defensive no-op entry.
    """

    accepted_draft_count: int
    committed_token_ids: tuple[int, ...]
    next_anchor: int | None
    target_cache_length: int
    emitted_length: int
    finished: bool
    finish_reason: str | None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.accepted_draft_count < 0:
            raise ValueError("accepted_draft_count must be >= 0")
        if self.target_cache_length < 0:
            raise ValueError("target_cache_length must be >= 0")
        if self.emitted_length < 0:
            raise ValueError("emitted_length must be >= 0")
        committed_count = len(self.committed_token_ids)
        complete_count = self.accepted_draft_count + 1
        if committed_count == 0:
            if not self.finished or self.next_anchor is not None:
                raise ValueError(
                    "an empty committed_token_ids result must be a finished no-op "
                    "with no next_anchor"
                )
        elif committed_count != complete_count and (
            not self.finished or committed_count > complete_count
        ):
            raise ValueError(
                "committed_token_ids must contain at most the accepted drafts "
                "plus one target bonus token; only finished rounds may be "
                "stop/budget-truncated"
            )

    @property
    def cached_token_count(self) -> int:
        """Anchor plus accepted drafts retained by the target cache."""

        if not self.committed_token_ids:
            return 0
        return min(len(self.committed_token_ids), self.accepted_draft_count) + 1

    @property
    def emitted_token_count(self) -> int:
        return len(self.committed_token_ids)

    @property
    def commit_block_token_ids(self) -> tuple[int, ...]:
        """Tokens emitted by this commit block after stop/budget truncation."""

        return self.committed_token_ids

    @property
    def accepted_drafts(self) -> int:
        """Compatibility alias for the verify rule's accepted-draft count."""

        return self.accepted_draft_count

    @property
    def bonus_token_id(self) -> int | None:
        return self.next_anchor


@dataclass(frozen=True)
class BackendCapabilities:
    """Static execution features advertised by a speculative backend."""

    supports_batch: bool
    supports_sampling: bool
    supports_ragged_rows: bool
    cache_kind: CacheKind
    supports_route_ahead: bool
    supports_rich_forward: bool
    pairing_evidence: PairingEvidence = field(default_factory=PairingEvidence)
    executor_evidence: ExecutorEvidence = field(
        default_factory=ExecutorEvidence
    )

    def __post_init__(self) -> None:
        if self.cache_kind not in ("dense_dynamic", "paged", "other"):
            raise ValueError(
                "cache_kind must be 'dense_dynamic', 'paged', or 'other'"
            )

    @property
    def is_dense_cache(self) -> bool:
        return self.cache_kind == "dense_dynamic"

    @property
    def is_paged_cache(self) -> bool:
        return self.cache_kind == "paged"


@dataclass(frozen=True)
class CacheSnapshot:
    """Rollback point shared by cache adapters."""

    logical_length: int

    def __post_init__(self) -> None:
        if self.logical_length < 0:
            raise ValueError("logical_length must be >= 0")


@dataclass(frozen=True)
class RichBatchMetadata:
    """Row layout and model inputs for one physical rich forward."""

    row_offsets: tuple[int, ...]
    row_lengths: tuple[int, ...]
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    cache_handles: tuple[object, ...] = ()
    request_contexts: tuple[object, ...] = ()
    route_contexts: tuple[object, ...] = ()
    block_tables: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    is_prefill: bool = True

    def __post_init__(self) -> None:
        rows = len(self.row_lengths)
        if len(self.row_offsets) != rows + 1 or self.row_offsets[:1] != (0,):
            raise ValueError(
                "row_offsets must start at zero and have one sentinel"
            )
        expected = 0
        for row, length in enumerate(self.row_lengths):
            if length < 0:
                raise ValueError("row_lengths must be non-negative")
            expected += length
            if self.row_offsets[row + 1] != expected:
                raise ValueError("row_offsets must be cumulative row_lengths")
        for name, values in (
            ("cache_handles", self.cache_handles),
            ("request_contexts", self.request_contexts),
            ("route_contexts", self.route_contexts),
        ):
            if values and len(values) != rows:
                raise ValueError(f"{name} must have one entry per row")
        for name, value in (
            ("attention_mask", self.attention_mask),
            ("position_ids", self.position_ids),
        ):
            if value is not None and (
                value.ndim != 2 or value.shape[0] != rows
            ):
                raise ValueError(f"{name} must have shape [rows, sequence]")
        if self.block_tables is not None and self.block_tables.shape[0] != rows:
            raise ValueError("block_tables must have one row per request")
        if self.seq_lens is not None and self.seq_lens.numel() != rows:
            raise ValueError("seq_lens must have one entry per row")

    @property
    def row_count(self) -> int:
        return len(self.row_lengths)


@dataclass(frozen=True)
class RichForwardResult:
    """Target forward payload whose cache handle may be engine-owned."""

    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    cache_handle: object
    cache_handles: tuple[object, ...] = ()
    row_offsets: tuple[int, ...] = ()
    row_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.cache_handles:
            object.__setattr__(self, "cache_handles", (self.cache_handle,))
        if self.row_lengths:
            if len(self.row_offsets) != len(self.row_lengths) + 1:
                raise ValueError(
                    "row_offsets must have one sentinel per rich row"
                )
            expected = 0
            for row, length in enumerate(self.row_lengths):
                expected += length
                if self.row_offsets[row + 1] != expected:
                    raise ValueError(
                        "row_offsets must be cumulative row_lengths"
                    )
            if len(self.cache_handles) not in (1, len(self.row_lengths)):
                raise ValueError("cache_handles must be shared or row-aligned")

    def __iter__(self):
        yield self.logits
        yield self.hidden_states
        yield self.cache_handle


@runtime_checkable
class CacheAdapter(Protocol):
    """Structural cache lifecycle required by speculative sessions."""

    def snapshot(self) -> CacheSnapshot:
        """Capture the cache position before a tentative verify."""
        ...

    def restore(self, snapshot: CacheSnapshot) -> None:
        """Restore an earlier logical cache position."""
        ...

    def append(self, token_count: int) -> None:
        """Record tokens appended by a completed model forward."""
        ...

    def truncate(self, logical_length: int) -> None:
        """Discard cache state beyond ``logical_length``."""
        ...

    def logical_length(self) -> int:
        """Return the current logical token count."""
        ...

    def release(self) -> None:
        """Release all cache state owned by this adapter."""
        ...


@runtime_checkable
class _DenseCache(Protocol):
    def get_seq_length(self) -> int: ...

    def crop(self, length: int) -> None: ...


class DenseCacheAdapter:
    """Adapter for transformers-style dense caches with ``crop`` support."""

    cache_kind: CacheKind = "dense_dynamic"

    def __init__(self, cache: object) -> None:
        if not isinstance(cache, _DenseCache):
            raise TypeError(
                "dense cache must provide get_seq_length() and crop()"
            )
        self.cache: _DenseCache = cache
        self._logical_length: int = int(cache.get_seq_length())
        self._released: bool = False

    def _ensure_active(self) -> None:
        if self._released:
            raise RuntimeError("dense cache adapter has been released")

    def snapshot(self) -> CacheSnapshot:
        self._ensure_active()
        return CacheSnapshot(logical_length=self._logical_length)

    def restore(self, snapshot: CacheSnapshot) -> None:
        self.truncate(snapshot.logical_length)

    def append(self, token_count: int) -> None:
        self._ensure_active()
        if token_count < 0:
            raise ValueError("token_count must be >= 0")
        expected_length = self._logical_length + token_count
        physical_length = int(self.cache.get_seq_length())
        if physical_length != expected_length:
            raise RuntimeError(
                "dense cache physical length does not match the reported append: "
                + f"expected {expected_length}, got {physical_length}"
            )
        self._logical_length = expected_length

    def truncate(self, logical_length: int) -> None:
        self._ensure_active()
        if logical_length < 0 or logical_length > self._logical_length:
            raise ValueError(
                "logical_length must be between 0 and the current logical length"
            )
        self.cache.crop(logical_length)
        self._logical_length = logical_length

    def logical_length(self) -> int:
        self._ensure_active()
        return self._logical_length

    def release(self) -> None:
        if self._released:
            return
        self.cache.crop(0)
        self._logical_length = 0
        self._released = True


@dataclass
class SessionTrace:
    """Serializable session accounting built from ``NativeStepTrace`` rows."""

    request_id: str
    backend: str
    cache_kind: CacheKind
    sampled: bool
    round_count: int = 0
    accepted: int = 0
    committed: int = 0
    emitted: int = 0
    rollback: int = 0
    replay: int = 0
    finish_reason: str | None = None
    route_ahead_status: str | None = None
    pairing_evidence: PairingEvidence = field(default_factory=PairingEvidence)
    executor_evidence: ExecutorEvidence = field(
        default_factory=ExecutorEvidence
    )

    def append(self, step: NativeStepTrace) -> None:
        self.round_count += 1
        self.accepted += step.accept
        self.committed += step.committed_count
        self.emitted = step.emitted_len

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "backend": self.backend,
            "cache_kind": self.cache_kind,
            "sampled": self.sampled,
            "round_count": self.round_count,
            "accepted": self.accepted,
            "committed": self.committed,
            "emitted": self.emitted,
            "rollback": self.rollback,
            "replay": self.replay,
            "finish_reason": self.finish_reason,
            "route_ahead_status": self.route_ahead_status,
            "pairing_evidence": self.pairing_evidence.as_dict(),
            "executor_evidence": self.executor_evidence.as_dict(),
        }


__all__ = [
    "BackendCapabilities",
    "CacheAdapter",
    "CacheKind",
    "CacheSnapshot",
    "DenseCacheAdapter",
    "ExecutorEvidence",
    "NativeStepTrace",
    "PairingEvidence",
    "RichBatchMetadata",
    "RichForwardResult",
    "RequestSpec",
    "SamplingContext",
    "SessionRoundResult",
    "SessionTrace",
]
