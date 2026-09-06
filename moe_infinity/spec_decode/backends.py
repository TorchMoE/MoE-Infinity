"""Execution backend contract and singleton DFlash adapter.

Task 4 deliberately keeps physical model execution per request.  Backends may
advertise cohort compatibility now, but ``DFlashExecutionBackend`` always
drives the canonical ``SpecSession`` one row at a time; a later backend can
replace that physical execution without changing the driver contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Hashable,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import torch

from moe_infinity.spec_decode.protocols import (
    BackendCapabilities,
    ExecutorEvidence,
    NativeStepTrace,
    PairingEvidence,
    RequestSpec,
    SamplingContext,
    SessionRoundResult,
    SessionTrace,
)

if TYPE_CHECKING:
    from moe_infinity.spec_decode.dflash import DFlashSpeculator, SpecSession


SessionT = TypeVar("SessionT")
SnapshotT = TypeVar("SnapshotT")


@runtime_checkable
class ExecutionBackend(Protocol[SessionT, SnapshotT]):
    """Request-scoped speculative execution lifecycle.

    ``supports`` and ``cohort_key`` are pure capability checks: the driver
    invokes them for every row before any backend receives ``prefill``.
    ``snapshot``/``restore`` form an abort-only boundary. A restored session is
    invalid for further execution and must be released without being resumed.
    Every successful ``draft`` + ``verify`` pair must increase observable
    output/round progress or report the session finished.
    """

    name: str
    capabilities: BackendCapabilities

    def supports(self, request: RequestSpec) -> bool: ...

    def cohort_key(self, request: RequestSpec) -> Hashable: ...

    def prefill(self, request: RequestSpec) -> SessionT: ...

    def draft(self, session: SessionT) -> object: ...

    def verify(self, session: SessionT) -> SessionRoundResult: ...

    def snapshot(self, session: SessionT) -> SnapshotT:
        """Capture state solely for abort-only restoration."""
        ...

    def restore(self, session: SessionT, snapshot: SnapshotT) -> None:
        """Abort to ``snapshot``; the driver must not resume this session."""
        ...

    def is_finished(self, session: SessionT) -> bool: ...

    def output_token_ids(self, session: SessionT) -> tuple[int, ...]: ...

    def trace(self, session: SessionT) -> SessionTrace: ...

    def release(self, session: SessionT) -> None: ...


class PhysicalCohortResult(Protocol):
    """Validated result surface shared by physical cohort backends."""

    generated_token_ids: tuple[tuple[int, ...], ...]
    step_trace: tuple[NativeStepTrace, ...]
    target_cache: object
    draft_cache: object
    session_traces: tuple[SessionTrace, ...]


@runtime_checkable
class PhysicalCohortBackend(Protocol):
    """Backend that executes one compatible cohort as a physical batch.

    This contract is deliberately separate from ``ExecutionBackend`` so the
    Task 4 request-scoped lifecycle remains valid.  Tensor compatibility
    adapters can use this seam when dense-cache mechanics require rows to be
    drafted, verified, and rolled back together.
    """

    name: str
    capabilities: BackendCapabilities

    def supports(self, request: RequestSpec) -> bool: ...
    def cohort_key(self, request: RequestSpec) -> Hashable: ...
    def execute_cohort(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: tuple[int, ...],
        stop_token_ids: tuple[int, ...],
        attention_mask: torch.Tensor,
        sampling_contexts: tuple[SamplingContext, ...] | None = None,
        stop_token_ids_by_row: tuple[tuple[int, ...], ...] | None = None,
    ) -> PhysicalCohortResult: ...


@dataclass(frozen=True)
class _DFlashSnapshot:
    emitted: tuple[int, ...]
    finished: bool
    round_index: int
    trace_length: int


class DFlashExecutionBackend:
    """Drive canonical ``SpecSession`` objects one request at a time."""

    name = "dflash-per-request"

    def __init__(
        self,
        speculator: DFlashSpeculator,
        *,
        retain_diagnostics: bool = False,
        collect_route_union: bool = True,
    ) -> None:
        self.speculator = speculator
        self.retain_diagnostics = retain_diagnostics
        self.collect_route_union = collect_route_union
        self._restored_sessions: set[int] = set()
        rich_forward = callable(
            getattr(speculator.moe, "_native_model_forward_rich", None)
        )
        pairing_evidence = getattr(
            speculator, "pairing_evidence", PairingEvidence()
        )
        executor_evidence = getattr(
            speculator, "executor_evidence", ExecutorEvidence()
        )
        self.capabilities = BackendCapabilities(
            supports_batch=False,
            supports_sampling=True,
            supports_ragged_rows=True,
            cache_kind="dense_dynamic",
            supports_route_ahead=executor_evidence.wiring_reachable,
            supports_rich_forward=rich_forward,
            pairing_evidence=pairing_evidence,
            executor_evidence=executor_evidence,
        )

    def supports(self, request: RequestSpec) -> bool:
        del request
        return True

    def cohort_key(self, request: RequestSpec) -> Hashable:
        return (request.is_sampled, self.capabilities.cache_kind)

    def prefill(self, request: RequestSpec) -> SpecSession:
        sampling = request.sampling
        session = self.speculator.begin_session(
            torch.tensor([request.prompt_token_ids], dtype=torch.long),
            max_new_tokens=request.max_new_tokens,
            temperature=sampling.temperature,
            stop_token_ids=list(request.stop_token_ids),
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            generator=sampling.generator,
            collect_route_union=self.collect_route_union,
        )
        if len(session.output_ids) >= request.max_new_tokens:
            session.finished = True
        return session

    def draft(self, session: SpecSession) -> object:
        return self.speculator.draft_round(session)

    def verify(self, session: SpecSession) -> SessionRoundResult:
        verified = self.speculator.verify_round(session)
        tokens = tuple(int(token) for token in verified.accepted_token_ids)
        includes_bonus = len(tokens) == verified.verified_accept + 1
        return SessionRoundResult(
            accepted_draft_count=verified.verified_accept,
            committed_token_ids=tokens,
            next_anchor=tokens[-1] if includes_bonus else None,
            target_cache_length=session.start,
            emitted_length=len(session.output_ids),
            finished=verified.finished,
            finish_reason=(
                self._finish_reason(session) if verified.finished else None
            ),
        )

    def snapshot(self, session: SpecSession) -> object:
        return _DFlashSnapshot(
            emitted=tuple(session.emitted),
            finished=session.finished,
            round_index=session.round_index,
            trace_length=len(session.step_trace),
        )

    def restore(self, session: SpecSession, snapshot: object) -> None:
        if not isinstance(snapshot, _DFlashSnapshot):
            raise TypeError("invalid DFlash session snapshot")
        session.emitted[:] = snapshot.emitted
        session.finished = snapshot.finished
        session.round_index = snapshot.round_index
        del session.step_trace[snapshot.trace_length :]
        session.clear_pending()
        self._restored_sessions.add(id(session))

    def is_finished(self, session: SpecSession) -> bool:
        return session.finished

    def output_token_ids(self, session: SpecSession) -> tuple[int, ...]:
        return tuple(session.output_ids[: session.max_new_tokens])

    def trace(self, session: SpecSession) -> SessionTrace:
        stats = getattr(self.speculator, "route_ahead_stats", None)
        executor_evidence = (
            stats.executor_evidence
            if stats is not None and stats.executor_evidence.attempted_layers
            else self.capabilities.executor_evidence
        )
        trace = SessionTrace(
            request_id="",
            backend=self.name,
            cache_kind=self.capabilities.cache_kind,
            sampled=session.sampled,
            finish_reason=self._finish_reason(session),
            route_ahead_status=(
                "enabled"
                if self.capabilities.supports_route_ahead
                else "disabled"
            ),
            pairing_evidence=self.capabilities.pairing_evidence,
            executor_evidence=executor_evidence,
        )
        for step in session.step_trace:
            trace.append(step)
        trace.finish_reason = self._finish_reason(session)
        return trace

    def release(self, session: SpecSession) -> None:
        restored = id(session) in self._restored_sessions
        self._restored_sessions.discard(id(session))
        if self.retain_diagnostics and not restored:
            self.speculator.last_target_cache = session.target_kv
            self.speculator.last_draft_cache = session.draft_kv
            return
        errors: list[BaseException] = []
        for cache in (session.target_kv, session.draft_kv):
            crop = getattr(cache, "crop", None)
            if callable(crop):
                try:
                    crop(0)
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]

    @staticmethod
    def _finish_reason(session: SpecSession) -> str:
        output = session.output_ids
        if output and output[-1] in session.stop_ids:
            return "stop"
        return "length"


__all__ = [
    "DFlashExecutionBackend",
    "ExecutionBackend",
    "PhysicalCohortBackend",
    "PhysicalCohortResult",
]
