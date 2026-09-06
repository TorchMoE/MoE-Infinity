"""Temporary DynamicCache execution records for continuous speculative serving.

Stage 4a keeps lifecycle and scheduling in ``ContinuousBatchingEngine`` while
each eligible sequence owns a persistent canonical ``SpecSession``.  The cache
context here is intentionally private and temporary; it is not the serving
``PagedKVCache`` ownership contract planned for Stage 4b.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, cast

import torch

from .mla_cache import MLAPagedKVCache
from .spec_cache_adapter import (
    EXECUTION_CONTEXT_PAGED_MLA,
    PagedCacheAdapter,
)
from .spec_state import SpecDecodeState

if TYPE_CHECKING:
    from moe_infinity.spec_decode.dflash import SpecSession


EXECUTION_CONTEXT_TEMPORARY_DYNAMIC = "temporary_dynamic"


@dataclass
class TemporaryDynamicCacheContext:
    """Explicit Stage 4a bridge around one session's private dense caches."""

    owner: object | None
    target_cache: object | None
    draft_cache: object | None
    mode: str = EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    released: bool = False
    owned_caches: list[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.refresh(self.target_cache, self.draft_cache)

    def refresh(
        self, target_cache: object | None, draft_cache: object | None
    ) -> None:
        """Release superseded caches and retain only currently active objects."""
        active = [
            cache for cache in (target_cache, draft_cache) if cache is not None
        ]
        self.target_cache = target_cache
        self.draft_cache = draft_cache
        errors: list[BaseException] = []
        retained: list[object] = []
        for cache in self.owned_caches:
            if any(cache is current for current in active):
                retained.append(cache)
                continue
            crop = getattr(cache, "crop", None)
            if callable(crop):
                try:
                    crop(0)
                except BaseException as exc:
                    errors.append(exc)
                    retained.append(cache)
        self.owned_caches = retained
        for cache in active:
            if cache is not None and not any(
                cache is owned for owned in self.owned_caches
            ):
                self.owned_caches.append(cache)
        if errors:
            raise errors[0]

    @contextmanager
    def activate(self) -> Iterator[None]:
        if self.released:
            raise RuntimeError(
                "temporary DynamicCache context has been released"
            )
        previous: object | None = None
        if self.owner is not None:
            previous = getattr(self.owner, "_cached_past_key_values", None)
            setattr(self.owner, "_cached_past_key_values", self.target_cache)
        try:
            yield
        finally:
            if self.owner is not None:
                setattr(self.owner, "_cached_past_key_values", previous)

    def release(self) -> None:
        if self.released:
            return
        errors: list[BaseException] = []
        for cache in self.owned_caches:
            crop = getattr(cache, "crop", None)
            if callable(crop):
                try:
                    crop(0)
                except BaseException as exc:
                    errors.append(exc)
        if self.owner is not None:
            current = getattr(self.owner, "_cached_past_key_values", None)
            if current is self.target_cache:
                setattr(self.owner, "_cached_past_key_values", None)
        self.released = True
        if errors:
            raise errors[0]


@dataclass
class PagedMLAExecutionContext:
    """No-swap execution context for one engine-owned MLA sequence."""

    target_cache: PagedCacheAdapter
    draft_cache: object | None
    mode: str = EXECUTION_CONTEXT_PAGED_MLA
    released: bool = False
    _owned_draft_caches: list[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.refresh(self.target_cache, self.draft_cache)

    def refresh(
        self, target_cache: object | None, draft_cache: object | None
    ) -> None:
        if target_cache is not self.target_cache:
            raise RuntimeError("paged MLA target cache handle was replaced")
        active = [] if draft_cache is None else [draft_cache]
        for cache in tuple(self._owned_draft_caches):
            if cache in active:
                continue
            crop = getattr(cache, "crop", None)
            if callable(crop):
                crop(0)
            self._owned_draft_caches.remove(cache)
        if (
            draft_cache is not None
            and draft_cache not in self._owned_draft_caches
        ):
            self._owned_draft_caches.append(draft_cache)
        self.draft_cache = draft_cache

    @contextmanager
    def activate(self) -> Iterator[None]:
        if self.released:
            raise RuntimeError("paged MLA execution context has been released")
        yield

    def release(self) -> None:
        if self.released:
            return
        errors: list[BaseException] = []
        for cache in self._owned_draft_caches:
            crop = getattr(cache, "crop", None)
            if callable(crop):
                try:
                    crop(0)
                except BaseException as exc:
                    errors.append(exc)
        try:
            self.target_cache.release()
        except BaseException as exc:
            errors.append(exc)
        self.released = True
        if errors:
            raise errors[0]


@dataclass
class ServingSpecSession:
    """Persistent serving record for exactly one request sequence."""

    spec_session: SpecSession | object
    request_id: str
    seq_id: int
    execution_context: TemporaryDynamicCacheContext | PagedMLAExecutionContext
    decode_state: SpecDecodeState
    callbacks: tuple[Callable[[object], None], ...] = ()
    cancelled: bool = False
    in_flight: bool = False
    pending_draft: object | None = None
    streamed_count: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    released: bool = False
    failure_reason: dict[str, str] | None = None
    paged_mla_admission: dict[str, object] = field(default_factory=dict)
    paged_mla_block_budget: int = 0

    @property
    def finished(self) -> bool:
        return bool(getattr(self.spec_session, "finished", False))

    def diagnostics(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "seq_id": self.seq_id,
            "execution_context": self.execution_context.mode,
            "cancelled": self.cancelled,
            "in_flight": self.in_flight,
            "released": self.released,
            "failure_reason": self.failure_reason,
            "paged_mla_admission": dict(self.paged_mla_admission),
            "cached_len": self.decode_state.cached_len,
            "emitted_len": self.decode_state.emitted_len,
        }


class SpecSessionDriver:
    """Step canonical speculative sessions without taking scheduler ownership."""

    def __init__(
        self,
        speculator: object,
        *,
        enable_paged_mla: bool = False,
        max_resident_paged_speculative_sessions: int = 1,
        min_free_mla_blocks_after_admission: int = 1,
    ) -> None:
        if (
            type(max_resident_paged_speculative_sessions) is not int
            or max_resident_paged_speculative_sessions < 0
        ):
            raise ValueError(
                "max_resident_paged_speculative_sessions must be an integer >= 0"
            )
        if (
            type(min_free_mla_blocks_after_admission) is not int
            or min_free_mla_blocks_after_admission < 1
        ):
            raise ValueError(
                "min_free_mla_blocks_after_admission must be an integer >= 1"
            )
        self.speculator = speculator
        self.sessions: dict[int, ServingSpecSession] = {}
        self.enable_paged_mla = enable_paged_mla
        self.max_resident_paged_speculative_sessions = (
            max_resident_paged_speculative_sessions
        )
        self.min_free_mla_blocks_after_admission = (
            min_free_mla_blocks_after_admission
        )
        self._admission_counters = {
            "admitted": 0,
            "session_cap": 0,
            "free_block_reserve": 0,
            "ineligible": 0,
            "begin_failed": 0,
        }
        self._last_admission_rejection: dict[str, object] | None = None
        begin_session = getattr(speculator, "begin_session")
        begin_parameters = signature(begin_session).parameters.values()
        self.supports_request_generator = any(
            parameter.kind is Parameter.VAR_KEYWORD
            or parameter.name == "generator"
            for parameter in begin_parameters
        )
        self.supports_target_cache_adapter = any(
            parameter.kind is Parameter.VAR_KEYWORD
            or parameter.name == "target_cache_adapter"
            for parameter in begin_parameters
        )

    @property
    def execution_context_mode(self) -> str:
        modes = {
            record.execution_context.mode for record in self.sessions.values()
        }
        if not modes:
            return EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
        if len(modes) == 1:
            return next(iter(modes))
        return "mixed"

    @property
    def admission_stats(self) -> dict[str, object]:
        owner = getattr(self.speculator, "moe", None)
        cache = getattr(owner, "_native_mla_cache", None)
        return {
            "active_sessions": self._active_paged_mla_sessions(),
            "enabled": self.enable_paged_mla,
            "max_resident_sessions": self.max_resident_paged_speculative_sessions,
            "min_free_blocks_after_admission": (
                self.min_free_mla_blocks_after_admission
            ),
            "free_blocks": (
                cache.free_block_count
                if isinstance(cache, MLAPagedKVCache)
                else None
            ),
            "counters": dict(self._admission_counters),
            "last_rejection": (
                dict(self._last_admission_rejection)
                if self._last_admission_rejection is not None
                else None
            ),
        }

    def begin(
        self,
        *,
        request_id: str,
        seq_id: int,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        stop_token_ids: Sequence[int],
        callbacks: Sequence[Callable[[object], None]],
        generator: torch.Generator | None = None,
    ) -> ServingSpecSession:
        if seq_id in self.sessions:
            raise ValueError(
                f"speculative session already exists for seq_id {seq_id}"
            )
        begin_session = getattr(self.speculator, "begin_session")
        owner = getattr(self.speculator, "moe", None)
        admission = self._paged_mla_admission(
            owner, temperature, len(prompt_token_ids), max_new_tokens
        )
        use_paged_mla = bool(admission["admitted"])
        if owner is not None and not use_paged_mla:
            setattr(owner, "_cached_past_key_values", None)
        begin_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stop_token_ids": list(stop_token_ids),
            "top_k": top_k,
            "top_p": top_p,
            "collect_route_union": True,
        }
        if self.supports_request_generator:
            begin_kwargs["generator"] = generator
        paged_adapter: PagedCacheAdapter | None = None
        try:
            paged_block_budget = 0
            if use_paged_mla:
                cache = getattr(owner, "_native_mla_cache")
                paged_block_budget = self._peak_blocks_for_request(
                    len(prompt_token_ids), max_new_tokens, cache.block_size
                )
                paged_adapter = PagedCacheAdapter(
                    cache, seq_id=seq_id, initial_length=len(prompt_token_ids)
                )
                begin_kwargs["target_cache_adapter"] = paged_adapter
            session = begin_session(
                torch.tensor([list(prompt_token_ids)], dtype=torch.long),
                **begin_kwargs,
            )
            if paged_adapter is None:
                context: (
                    TemporaryDynamicCacheContext | PagedMLAExecutionContext
                ) = TemporaryDynamicCacheContext(
                    owner=owner,
                    target_cache=getattr(session, "target_kv", None),
                    draft_cache=getattr(session, "draft_kv", None),
                )
            else:
                if getattr(session, "target_kv", None) is not paged_adapter:
                    raise RuntimeError(
                        "paged MLA session did not retain its cache adapter"
                    )
                context = PagedMLAExecutionContext(
                    target_cache=paged_adapter,
                    draft_cache=getattr(session, "draft_kv", None),
                )
            record = ServingSpecSession(
                spec_session=session,
                request_id=request_id,
                seq_id=seq_id,
                execution_context=context,
                decode_state=SpecDecodeState(
                    seq_id=seq_id, prompt_len=len(prompt_token_ids)
                ),
                callbacks=tuple(callbacks),
                paged_mla_admission=admission,
                paged_mla_block_budget=paged_block_budget,
            )
        except BaseException:
            if paged_adapter is not None:
                paged_adapter.release()
            failed_decision = {
                "eligible": bool(admission["eligible"]),
                "admitted": False,
                "reason": "begin_failed",
            }
            self._last_admission_rejection = failed_decision
            self._admission_counters["begin_failed"] += 1
            raise
        self.sessions[seq_id] = record
        reason = str(admission["reason"])
        self._admission_counters[reason] += 1
        if not bool(admission["admitted"]):
            self._last_admission_rejection = dict(admission)
        return record

    def draft(self, record: ServingSpecSession) -> object:
        self._ensure_runnable(record)
        if record.pending_draft is not None:
            return record.pending_draft
        self._refresh_context(record)
        record.in_flight = True
        primary: BaseException | None = None
        try:
            with record.execution_context.activate():
                draft = getattr(self.speculator, "draft_round")(
                    record.spec_session
                )
            record.pending_draft = draft
            return draft
        except BaseException as exc:
            primary = exc
            raise
        finally:
            self._finish_inflight(record, primary)

    def verify(self, record: ServingSpecSession) -> object | None:
        self._ensure_runnable(record)
        if record.pending_draft is None:
            raise RuntimeError("verify requires a pending draft")
        self._refresh_context(record)
        record.in_flight = True
        primary: BaseException | None = None
        try:
            with record.execution_context.activate():
                return getattr(self.speculator, "verify_round")(
                    record.spec_session
                )
        except BaseException as exc:
            primary = exc
            raise
        finally:
            record.pending_draft = None
            self._finish_inflight(record, primary)

    def commit(self, record: ServingSpecSession) -> tuple[int, ...]:
        """Publish unseen canonical emissions and advance logical accounting."""
        if record.cancelled or record.released:
            return ()
        emitted_obj = getattr(record.spec_session, "emitted", None)
        if emitted_obj is None:
            emitted_obj = getattr(record.spec_session, "output_ids")
        emitted = tuple(
            int(token) for token in cast(Sequence[int], emitted_obj)
        )
        committed = emitted[record.streamed_count :]
        if not committed:
            return ()
        record.streamed_count += len(committed)
        record.output_token_ids.extend(committed)
        record.decode_state.record_commit(len(committed))
        if not record.decode_state.invariant_holds():
            raise RuntimeError(
                "speculative serving logical cache invariant violated for "
                f"seq_id {record.seq_id}"
            )
        return committed

    def cancel(self, seq_id: int) -> None:
        record = self.sessions.get(seq_id)
        if record is None:
            return
        record.cancelled = True
        if not record.in_flight:
            self.release(record)

    def release(self, record: ServingSpecSession) -> None:
        if record.released:
            return
        errors: list[BaseException] = []
        clear_pending = getattr(record.spec_session, "clear_pending", None)
        if callable(clear_pending):
            try:
                clear_pending()
            except BaseException as exc:
                errors.append(exc)
        record.pending_draft = None
        try:
            self._refresh_context(record)
        except BaseException as exc:
            errors.append(exc)
        try:
            try:
                record.execution_context.release()
            except BaseException as exc:
                errors.append(exc)
        finally:
            record.released = True
            self.sessions.pop(record.seq_id, None)
        if errors:
            for cleanup_error in errors[1:]:
                add_note = getattr(errors[0], "add_note", None)
                if callable(add_note):
                    add_note(f"additional cleanup failure: {cleanup_error}")
            raise errors[0]

    def fail(
        self, record: ServingSpecSession, failure_reason: dict[str, str]
    ) -> None:
        record.failure_reason = dict(failure_reason)
        record.cancelled = True
        self.release(record)

    def _release_if_cancelled(self, record: ServingSpecSession) -> None:
        if record.cancelled and not record.released:
            self.release(record)

    def _finish_inflight(
        self,
        record: ServingSpecSession,
        primary: BaseException | None,
    ) -> None:
        cleanup_errors: list[BaseException] = []
        try:
            self._refresh_context(record)
        except BaseException as exc:
            cleanup_errors.append(exc)
        record.in_flight = False
        try:
            self._release_if_cancelled(record)
        except BaseException as exc:
            cleanup_errors.append(exc)
        if primary is not None:
            add_note = getattr(primary, "add_note", None)
            if callable(add_note):
                for cleanup_error in cleanup_errors:
                    add_note(f"session cleanup failed: {cleanup_error}")
            return
        if cleanup_errors:
            raise cleanup_errors[0]

    @staticmethod
    def _refresh_context(record: ServingSpecSession) -> None:
        record.execution_context.refresh(
            getattr(record.spec_session, "target_kv", None),
            getattr(record.spec_session, "draft_kv", None),
        )

    @staticmethod
    def _ensure_runnable(record: ServingSpecSession) -> None:
        if record.cancelled:
            raise RuntimeError("speculative serving session is cancelled")
        if record.released:
            raise RuntimeError("speculative serving session is released")
        if record.finished:
            raise RuntimeError("speculative serving session is finished")

    def _paged_mla_admission(
        self,
        owner: object | None,
        temperature: float,
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> dict[str, object]:
        if (
            not self.enable_paged_mla
            or owner is None
            or temperature != 0.0
            or not self.supports_target_cache_adapter
        ):
            return {
                "eligible": False,
                "admitted": False,
                "reason": "ineligible",
            }
        cache = getattr(owner, "_native_mla_cache", None)
        modules_fn = getattr(owner, "_get_mla_attention_modules", None)
        if not (
            isinstance(cache, MLAPagedKVCache)
            and callable(modules_fn)
            and bool(modules_fn())
        ):
            return {
                "eligible": False,
                "admitted": False,
                "reason": "ineligible",
            }
        if self._transient_verify_tokens() is None:
            return {
                "eligible": False,
                "admitted": False,
                "reason": "ineligible",
            }
        if (
            self._active_paged_mla_sessions()
            >= self.max_resident_paged_speculative_sessions
        ):
            return {
                "eligible": True,
                "admitted": False,
                "reason": "session_cap",
            }
        declared_blocks = self._peak_blocks_for_request(
            prompt_tokens, max_new_tokens, cache.block_size
        )
        available_after_active_headroom = (
            cache.free_block_count - self._active_unallocated_headroom(cache)
        )
        if (
            available_after_active_headroom - declared_blocks
            < self.min_free_mla_blocks_after_admission
        ):
            return {
                "eligible": True,
                "admitted": False,
                "reason": "free_block_reserve",
            }
        return {"eligible": True, "admitted": True, "reason": "admitted"}

    @staticmethod
    def _blocks_for_tokens(token_count: int, block_size: int) -> int:
        return (token_count + block_size - 1) // block_size

    def _transient_verify_tokens(self) -> int | None:
        config = getattr(self.speculator, "config", None)
        block_size = getattr(config, "block_size", None)
        if type(block_size) is not int or block_size < 2:
            return None
        return block_size - 1

    def _peak_blocks_for_request(
        self, prompt_tokens: int, max_new_tokens: int, mla_block_size: int
    ) -> int:
        transient_tokens = self._transient_verify_tokens()
        if transient_tokens is None:
            raise RuntimeError("paged MLA requires a valid DFlash block_size")
        return self._blocks_for_tokens(
            prompt_tokens + max_new_tokens + transient_tokens,
            mla_block_size,
        )

    def _active_unallocated_headroom(self, cache: MLAPagedKVCache) -> int:
        headroom = 0
        for record in self.sessions.values():
            if (
                record.released
                or record.execution_context.mode != EXECUTION_CONTEXT_PAGED_MLA
            ):
                continue
            allocated = len(cache.get_block_table(record.seq_id))
            headroom += max(0, record.paged_mla_block_budget - allocated)
        return headroom

    def _active_paged_mla_sessions(self) -> int:
        return sum(
            record.execution_context.mode == EXECUTION_CONTEXT_PAGED_MLA
            for record in self.sessions.values()
            if not record.released
        )


__all__ = [
    "EXECUTION_CONTEXT_TEMPORARY_DYNAMIC",
    "PagedMLAExecutionContext",
    "ServingSpecSession",
    "SpecSessionDriver",
    "TemporaryDynamicCacheContext",
]
