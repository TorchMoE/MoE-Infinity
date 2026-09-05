"""Deterministic request cohort planning and singleton session execution."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Hashable, cast

import torch

from .backends import (
    ExecutionBackend,
    PhysicalCohortBackend,
    PhysicalCohortResult,
)
from .protocols import (
    BackendCapabilities,
    RequestSpec,
    SamplingContext,
    SessionRoundResult,
    SessionTrace,
)


class UnsupportedRequestError(RuntimeError):
    """No configured backend can preserve a request's semantics."""


class BackendProgressError(RuntimeError):
    """A backend returned an unfinished round without observable progress."""


class SessionCleanupError(RuntimeError):
    """One or more cleanup operations failed after execution completed."""

    def __init__(self, errors: Sequence[BaseException]) -> None:
        self.errors = tuple(errors)
        super().__init__(
            f"{len(self.errors)} session cleanup operation(s) failed: "
            + "; ".join(str(error) for error in self.errors)
        )


def _attach_cleanup_context(
    primary: BaseException, errors: Sequence[BaseException]
) -> None:
    """Retain cleanup failures without replacing ``primary``.

    Python 3.11+ exceptions normally expose ``add_note``. Python 3.10 and
    custom exceptions may not, so the fallback chains an aggregate cleanup
    error through ``__context__`` and exposes the original errors as an
    attribute when possible. Cleanup reporting itself must never mask the
    execution failure.
    """
    if not errors:
        return

    try:
        add_note = getattr(primary, "add_note", None)
    except BaseException:
        add_note = None
    if callable(add_note):
        try:
            for error in errors:
                add_note(f"session cleanup failed: {error}")
            return
        except BaseException:
            pass

    cleanup_context = SessionCleanupError(errors)
    try:
        primary.__context__ = cleanup_context
    except BaseException:
        pass
    try:
        setattr(primary, "session_cleanup_errors", tuple(errors))
    except BaseException:
        pass


@dataclass(frozen=True)
class CohortPlan:
    """Stable row indices sharing one selected backend compatibility key."""

    backend: str
    capabilities: BackendCapabilities
    compatibility_key: Hashable
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class DriverResult:
    """One completed request result; returned in original row order."""

    request_id: str
    output_token_ids: tuple[int, ...]
    finish_reason: str
    backend: str
    trace: SessionTrace
    rounds: tuple[SessionRoundResult, ...]
    fallback_reason: str | None = None


@dataclass(frozen=True)
class PhysicalCohortDriverResult:
    """Atomically published row results plus backend-owned diagnostics."""

    results: tuple[DriverResult, ...]
    backend_result: PhysicalCohortResult


@dataclass(frozen=True)
class _Selection:
    row_index: int
    request: RequestSpec
    backend: ExecutionBackend[Any, Any]
    compatibility_key: Hashable
    fallback_reason: str | None


@dataclass(frozen=True)
class _PhysicalSelection:
    row_index: int
    request: RequestSpec
    backend: PhysicalCohortBackend
    compatibility_key: Hashable
    fallback_reason: str | None


@dataclass
class _ActiveSession:
    selection: _Selection
    session: object
    initial_snapshot: object
    rounds: list[SessionRoundResult]
    observable_progress: int


class SessionDriver:
    """Plan compatible cohorts, then drive one backend session per request.

    Backend order is the documented fallback policy: the first backend that
    both advertises the required capability and accepts the request wins.
    Sampling requests skip non-sampling backends; they are never rewritten as
    greedy requests.  Selection for every row completes before the first
    prefill, making unsupported input a pre-output failure.
    """

    def __init__(
        self,
        backends: Sequence[ExecutionBackend[Any, Any] | PhysicalCohortBackend],
    ) -> None:
        if not backends:
            raise ValueError("SessionDriver requires at least one backend")
        self.backends = tuple(backends)
        self.last_cohorts: tuple[CohortPlan, ...] = ()
        self.last_results: tuple[DriverResult, ...] = ()

    def run(
        self, requests: RequestSpec | Iterable[RequestSpec]
    ) -> tuple[DriverResult, ...]:
        normalized = self._normalize_requests(requests)
        selections = tuple(
            self._select_backend(row_index, request)
            for row_index, request in enumerate(normalized)
        )
        cohorts = self._cohort(selections)
        self.last_cohorts = tuple(
            CohortPlan(
                backend=rows[0].backend.name,
                capabilities=rows[0].backend.capabilities,
                compatibility_key=rows[0].compatibility_key,
                row_indices=tuple(row.row_index for row in rows),
            )
            for rows in cohorts
        )
        self.last_results = ()

        active: list[_ActiveSession] = []
        pending_results: tuple[DriverResult, ...] | None = None
        primary: BaseException | None = None
        primary_traceback: TracebackType | None = None
        try:
            # Prefill remains singleton, but happens only after the complete
            # capability plan has succeeded for every row.
            for selection in selections:
                session = selection.backend.prefill(selection.request)
                initial_progress = len(
                    selection.backend.output_token_ids(session)
                )
                active.append(
                    _ActiveSession(
                        selection=selection,
                        session=session,
                        initial_snapshot=selection.backend.snapshot(session),
                        rounds=[],
                        observable_progress=initial_progress,
                    )
                )

            by_row = {item.selection.row_index: item for item in active}
            for cohort in cohorts:
                rows = [by_row[selection.row_index] for selection in cohort]
                self._run_cohort(rows)

            pending_results = tuple(
                self._result(item)
                for item in sorted(
                    active, key=lambda item: item.selection.row_index
                )
            )
        except BaseException as exc:
            primary = exc
            primary_traceback = exc.__traceback__

        cleanup_errors = self._cleanup(active, restore=primary is not None)
        if primary is not None:
            _attach_cleanup_context(primary, cleanup_errors)
            raise primary.with_traceback(primary_traceback)
        if cleanup_errors:
            raise SessionCleanupError(cleanup_errors)

        assert pending_results is not None
        self.last_results = pending_results
        return pending_results

    def run_physical_cohort(
        self,
        input_ids: torch.Tensor,
        *,
        requests: RequestSpec | Iterable[RequestSpec],
        attention_mask: torch.Tensor,
    ) -> PhysicalCohortDriverResult:
        """Select and execute one physical backend cohort atomically.

        All rows are capability-selected before the backend receives the
        cohort. A physical entry never splits rows or rewrites sampling policy;
        callers that need semantic per-request fallback must use :meth:`run`.
        """
        self.last_results = ()
        self.last_cohorts = ()
        normalized = self._normalize_requests(requests)
        self._validate_physical_inputs(
            input_ids, attention_mask, len(normalized)
        )
        if not normalized:
            raise ValueError("a physical cohort requires at least one request")

        selections = tuple(
            self._select_physical_backend(row_index, request)
            for row_index, request in enumerate(normalized)
        )
        first = selections[0]
        if any(
            selection.backend is not first.backend
            or selection.compatibility_key != first.compatibility_key
            for selection in selections[1:]
        ):
            raise UnsupportedRequestError(
                "requests cannot execute as one physical cohort without "
                "splitting backends or compatibility keys"
            )

        self.last_cohorts = (
            CohortPlan(
                backend=first.backend.name,
                capabilities=first.backend.capabilities,
                compatibility_key=first.compatibility_key,
                row_indices=tuple(range(len(normalized))),
            ),
        )
        stop_rows = tuple(
            tuple(sorted(request.stop_token_ids)) for request in normalized
        )
        shared_stops = stop_rows[0]
        common_stops = all(row == shared_stops for row in stop_rows)
        budgets = tuple(request.max_new_tokens for request in normalized)
        sampling_contexts = tuple(request.sampling for request in normalized)
        sampled = any(context.is_sampled for context in sampling_contexts)
        backend_result = self._execute_physical_backend(
            first.backend,
            input_ids,
            budgets=budgets,
            shared_stops=shared_stops if common_stops else (),
            attention_mask=attention_mask,
            sampling_contexts=sampling_contexts if sampled else None,
            stop_rows=None if common_stops else stop_rows,
        )
        results = self._physical_results(selections, backend_result)
        self.last_results = results
        return PhysicalCohortDriverResult(
            results=results,
            backend_result=backend_result,
        )

    @staticmethod
    def _normalize_requests(
        requests: RequestSpec | Iterable[RequestSpec],
    ) -> tuple[RequestSpec, ...]:
        rows = (
            (requests,)
            if isinstance(requests, RequestSpec)
            else tuple(requests)
        )
        if not rows:
            return ()
        if any(not isinstance(row, RequestSpec) for row in rows):
            raise TypeError("SessionDriver.run expects RequestSpec rows")
        request_ids = [row.request_id for row in rows]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError(
                "request_id values must be unique within a driver run"
            )
        return rows

    def _select_backend(
        self, row_index: int, request: RequestSpec
    ) -> _Selection:
        for backend_index, candidate in enumerate(self.backends):
            if not isinstance(candidate, ExecutionBackend):
                continue
            backend = cast(ExecutionBackend[Any, Any], candidate)
            if (
                request.is_sampled
                and not backend.capabilities.supports_sampling
            ):
                continue
            if not backend.supports(request):
                continue
            compatibility_key = backend.cohort_key(request)
            try:
                hash(compatibility_key)
            except TypeError as exc:
                raise TypeError("backend cohort_key must be hashable") from exc
            fallback_reason = (
                None
                if backend_index == 0
                else f"selected compatible fallback backend {backend.name}"
            )
            return _Selection(
                row_index=row_index,
                request=request,
                backend=backend,
                compatibility_key=compatibility_key,
                fallback_reason=fallback_reason,
            )
        mode = "sampled" if request.is_sampled else "greedy"
        raise UnsupportedRequestError(
            f"request {request.request_id!r} has no compatible {mode} backend"
        )

    def _select_physical_backend(
        self, row_index: int, request: RequestSpec
    ) -> _PhysicalSelection:
        physical_index = 0
        for candidate in self.backends:
            if not isinstance(candidate, PhysicalCohortBackend):
                continue
            backend = cast(PhysicalCohortBackend, candidate)
            if not backend.capabilities.supports_batch:
                physical_index += 1
                continue
            if (
                request.is_sampled
                and not backend.capabilities.supports_sampling
            ):
                physical_index += 1
                continue
            if not backend.supports(request):
                physical_index += 1
                continue
            compatibility_key = backend.cohort_key(request)
            try:
                hash(compatibility_key)
            except TypeError as exc:
                raise TypeError("backend cohort_key must be hashable") from exc
            return _PhysicalSelection(
                row_index=row_index,
                request=request,
                backend=backend,
                compatibility_key=compatibility_key,
                fallback_reason=(
                    None
                    if physical_index == 0
                    else f"selected compatible fallback backend {backend.name}"
                ),
            )
        mode = "sampled" if request.is_sampled else "greedy"
        raise UnsupportedRequestError(
            f"request {request.request_id!r} has no compatible physical {mode} backend"
        )

    @staticmethod
    def _validate_physical_inputs(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        request_count: int,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "physical cohort input_ids must have shape [batch, seq]"
            )
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} != "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        if int(input_ids.shape[0]) != request_count:
            raise ValueError(
                f"physical cohort has {int(input_ids.shape[0])} tensor rows "
                f"for {request_count} requests"
            )
        binary = (attention_mask == 0) | (attention_mask == 1)
        if not bool(torch.all(binary).item()):
            raise ValueError("attention_mask must be 0/1 valued")

    @staticmethod
    def _execute_physical_backend(
        backend: PhysicalCohortBackend,
        input_ids: torch.Tensor,
        *,
        budgets: tuple[int, ...],
        shared_stops: tuple[int, ...],
        attention_mask: torch.Tensor,
        sampling_contexts: tuple[SamplingContext, ...] | None,
        stop_rows: tuple[tuple[int, ...], ...] | None,
    ) -> PhysicalCohortResult:
        return backend.execute_cohort(
            input_ids,
            max_new_tokens=budgets,
            stop_token_ids=shared_stops,
            attention_mask=attention_mask,
            sampling_contexts=sampling_contexts,
            stop_token_ids_by_row=stop_rows,
        )

    @staticmethod
    def _physical_results(
        selections: Sequence[_PhysicalSelection],
        backend_result: PhysicalCohortResult,
    ) -> tuple[DriverResult, ...]:
        generated_value = getattr(backend_result, "generated_token_ids", None)
        if generated_value is None:
            raise TypeError(
                "physical backend result must expose generated_token_ids"
            )
        generated = tuple(
            tuple(int(token) for token in row) for row in generated_value
        )
        if len(generated) != len(selections):
            raise ValueError(
                "physical backend returned a different number of generated rows"
            )
        traces_value = tuple(getattr(backend_result, "session_traces", ()))
        if traces_value and len(traces_value) != len(selections):
            raise ValueError(
                "physical backend returned a different number of session traces"
            )

        results: list[DriverResult] = []
        for row, selection in enumerate(selections):
            tokens = generated[row]
            if len(tokens) > selection.request.max_new_tokens:
                raise ValueError(
                    f"physical backend exceeded max_new_tokens for request "
                    f"{selection.request.request_id!r}"
                )
            if traces_value:
                trace = traces_value[row]
                if not isinstance(trace, SessionTrace):
                    raise TypeError(
                        "physical backend session_traces must contain SessionTrace"
                    )
            else:
                stopped = (
                    bool(tokens)
                    and tokens[-1] in selection.request.stop_token_ids
                )
                trace = SessionTrace(
                    request_id=selection.request.request_id,
                    backend=selection.backend.name,
                    cache_kind=selection.backend.capabilities.cache_kind,
                    sampled=selection.request.is_sampled,
                    emitted=len(tokens),
                    finish_reason="stop" if stopped else "length",
                    route_ahead_status=(
                        "enabled"
                        if selection.backend.capabilities.supports_route_ahead
                        else "disabled"
                    ),
                    pairing_evidence=selection.backend.capabilities.pairing_evidence,
                    executor_evidence=selection.backend.capabilities.executor_evidence,
                )
            trace.request_id = selection.request.request_id
            finish_reason = trace.finish_reason or (
                "stop"
                if tokens and tokens[-1] in selection.request.stop_token_ids
                else "length"
            )
            results.append(
                DriverResult(
                    request_id=selection.request.request_id,
                    output_token_ids=tokens,
                    finish_reason=finish_reason,
                    backend=selection.backend.name,
                    trace=trace,
                    rounds=(),
                    fallback_reason=selection.fallback_reason,
                )
            )
        return tuple(results)

    @staticmethod
    def _cohort(
        selections: Sequence[_Selection],
    ) -> tuple[tuple[_Selection, ...], ...]:
        grouped: OrderedDict[tuple[int, Hashable], list[_Selection]] = (
            OrderedDict()
        )
        for selection in selections:
            key = (id(selection.backend), selection.compatibility_key)
            grouped.setdefault(key, []).append(selection)
        return tuple(tuple(rows) for rows in grouped.values())

    @staticmethod
    def _run_cohort(rows: Sequence[_ActiveSession]) -> None:
        while True:
            unfinished = [
                row
                for row in rows
                if not row.selection.backend.is_finished(row.session)
            ]
            if not unfinished:
                return
            for row in unfinished:
                backend = row.selection.backend
                backend.draft(row.session)
                round_result = backend.verify(row.session)
                output_progress = len(backend.output_token_ids(row.session))
                progress = max(output_progress, round_result.emitted_length)
                if (
                    not round_result.finished
                    and progress <= row.observable_progress
                ):
                    raise BackendProgressError(
                        f"backend {backend.name} made no progress for request "
                        f"{row.selection.request.request_id!r}"
                    )
                row.observable_progress = max(row.observable_progress, progress)
                row.rounds.append(round_result)

    @staticmethod
    def _cleanup(
        rows: Sequence[_ActiveSession], *, restore: bool
    ) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        if restore:
            for row in rows:
                try:
                    row.selection.backend.restore(
                        row.session, row.initial_snapshot
                    )
                except BaseException as exc:
                    errors.append(exc)
        for row in rows:
            try:
                row.selection.backend.release(row.session)
            except BaseException as exc:
                errors.append(exc)
        return tuple(errors)

    @staticmethod
    def _result(item: _ActiveSession) -> DriverResult:
        backend = item.selection.backend
        trace = backend.trace(item.session)
        trace.request_id = item.selection.request.request_id
        finish_reason = trace.finish_reason or "length"
        return DriverResult(
            request_id=item.selection.request.request_id,
            output_token_ids=backend.output_token_ids(item.session),
            finish_reason=finish_reason,
            backend=backend.name,
            trace=trace,
            rounds=tuple(item.rounds),
            fallback_reason=item.selection.fallback_reason,
        )


__all__ = [
    "BackendProgressError",
    "CohortPlan",
    "DriverResult",
    "PhysicalCohortDriverResult",
    "SessionDriver",
    "SessionCleanupError",
    "UnsupportedRequestError",
]
