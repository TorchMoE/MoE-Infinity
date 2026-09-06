from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable

import pytest
import torch

import moe_infinity.spec_decode.session_driver as driver_module
from moe_infinity.spec_decode import DFlashSpeculator, read_dflash_config
from moe_infinity.spec_decode.backends import (
    DFlashExecutionBackend,
    ExecutionBackend,
)
from moe_infinity.spec_decode.dflash import VerifyResult
from moe_infinity.spec_decode.protocols import (
    BackendCapabilities,
    RequestSpec,
    SamplingContext,
    SessionRoundResult,
    SessionTrace,
)
from moe_infinity.spec_decode.session_driver import (
    SessionDriver,
    UnsupportedRequestError,
)
from tests.python.dflash.fixtures_tiny import (
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
)


@dataclass
class _State:
    request: RequestSpec
    output: list[int] = field(default_factory=list)
    pending: bool = False
    finished: bool = False
    released: bool = False
    restored: bool = False
    rounds: int = 0


class _FakeBackend:
    def __init__(
        self,
        name: str,
        events: list[tuple[str, str, str]],
        *,
        supports_sampling: bool = True,
        accepts: Callable[[RequestSpec], bool] = lambda _request: True,
        cohort: Callable[[RequestSpec], Hashable] = lambda _request: "default",
        fail_draft_for: frozenset[str] = frozenset(),
        draft_error: BaseException | None = None,
        no_progress_for: frozenset[str] = frozenset(),
        fail_restore_for: frozenset[str] = frozenset(),
        fail_release_for: frozenset[str] = frozenset(),
    ) -> None:
        self.name = name
        self.events = events
        self.accepts = accepts
        self.cohort = cohort
        self.fail_draft_for = fail_draft_for
        self.draft_error = draft_error
        self.no_progress_for = no_progress_for
        self.fail_restore_for = fail_restore_for
        self.fail_release_for = fail_release_for
        self.sessions: list[_State] = []
        self.capabilities = BackendCapabilities(
            supports_batch=False,
            supports_sampling=supports_sampling,
            supports_ragged_rows=True,
            cache_kind="dense_dynamic",
            supports_route_ahead=False,
            supports_rich_forward=True,
        )

    def supports(self, request: RequestSpec) -> bool:
        self.events.append(("supports", self.name, request.request_id))
        return self.accepts(request)

    def cohort_key(self, request: RequestSpec) -> Hashable:
        return self.cohort(request)

    def prefill(self, request: RequestSpec) -> _State:
        self.events.append(("prefill", self.name, request.request_id))
        state = _State(request=request, finished=request.max_new_tokens == 0)
        self.sessions.append(state)
        return state

    def draft(self, session: _State) -> None:
        assert not session.restored, "driver resumed an abort-restored session"
        self.events.append(("draft", self.name, session.request.request_id))
        if session.finished:
            raise AssertionError("finished row was drafted")
        session.pending = True
        if session.request.request_id in self.fail_draft_for:
            # Deliberately corrupt tentative state. The driver must restore it.
            session.output.append(999)
            raise self.draft_error or RuntimeError("draft failed")

    def verify(self, session: _State) -> SessionRoundResult:
        assert not session.restored, "driver resumed an abort-restored session"
        self.events.append(("verify", self.name, session.request.request_id))
        assert session.pending
        if session.request.request_id in self.no_progress_for:
            session.pending = False
            return SessionRoundResult(
                accepted_draft_count=0,
                committed_token_ids=(777,),
                next_anchor=777,
                target_cache_length=len(session.request.prompt_token_ids),
                emitted_length=len(session.output),
                finished=False,
                finish_reason=None,
            )
        token = 10 + session.rounds
        session.output.append(token)
        session.rounds += 1
        session.pending = False
        stop = token in session.request.stop_token_ids
        session.finished = (
            stop or len(session.output) >= session.request.max_new_tokens
        )
        return SessionRoundResult(
            accepted_draft_count=0,
            committed_token_ids=(token,),
            next_anchor=token,
            target_cache_length=len(session.request.prompt_token_ids)
            + len(session.output),
            emitted_length=len(session.output),
            finished=session.finished,
            finish_reason=("stop" if stop else "length")
            if session.finished
            else None,
        )

    def snapshot(self, session: _State) -> tuple[list[int], bool, bool, int]:
        return (
            list(session.output),
            session.pending,
            session.finished,
            session.rounds,
        )

    def restore(
        self,
        session: _State,
        snapshot: tuple[list[int], bool, bool, int],
    ) -> None:
        assert not session.restored, "abort-only restore called more than once"
        self.events.append(("restore", self.name, session.request.request_id))
        output, pending, finished, rounds = snapshot
        session.output = output
        session.pending = pending
        session.finished = finished
        session.rounds = rounds
        session.restored = True
        if session.request.request_id in self.fail_restore_for:
            raise RuntimeError(f"restore failed: {session.request.request_id}")

    def is_finished(self, session: _State) -> bool:
        assert not session.restored, "driver resumed an abort-restored session"
        return session.finished

    def output_token_ids(self, session: _State) -> tuple[int, ...]:
        assert not session.restored, "driver resumed an abort-restored session"
        return tuple(session.output)

    def trace(self, session: _State) -> SessionTrace:
        assert not session.restored, "driver resumed an abort-restored session"
        return SessionTrace(
            request_id=session.request.request_id,
            backend=self.name,
            cache_kind=self.capabilities.cache_kind,
            sampled=session.request.is_sampled,
            round_count=session.rounds,
            emitted=len(session.output),
            finish_reason=(
                "stop"
                if session.output
                and session.output[-1] in session.request.stop_token_ids
                else "length"
            ),
        )

    def release(self, session: _State) -> None:
        self.events.append(("release", self.name, session.request.request_id))
        session.released = True
        if session.request.request_id in self.fail_release_for:
            raise RuntimeError(f"release failed: {session.request.request_id}")


def _request(
    request_id: str,
    *,
    budget: int = 2,
    sampling: SamplingContext | None = None,
    stops: frozenset[int] = frozenset(),
) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        prompt_token_ids=(1, 2, 3),
        max_new_tokens=budget,
        stop_token_ids=stops,
        sampling=sampling or SamplingContext(),
    )


def test_driver_creates_one_session_per_row() -> None:
    events: list[tuple[str, str, str]] = []
    backend = _FakeBackend("native", events)
    requests = [_request("r0", budget=1), _request("r1", budget=2)]

    results = SessionDriver([backend]).run(requests)

    assert isinstance(backend, ExecutionBackend)
    assert [state.request for state in backend.sessions] == requests
    assert [result.request_id for result in results] == ["r0", "r1"]
    assert [result.output_token_ids for result in results] == [(10,), (10, 11)]
    assert all(state.released for state in backend.sessions)


def test_backend_selected_before_prefill() -> None:
    events: list[tuple[str, str, str]] = []
    backend = _FakeBackend("native", events)

    SessionDriver([backend]).run([_request("r0"), _request("r1")])

    first_prefill = next(
        i for i, event in enumerate(events) if event[0] == "prefill"
    )
    assert [event[0] for event in events[:first_prefill]] == [
        "supports",
        "supports",
    ]


def test_incompatible_rows_split() -> None:
    events: list[tuple[str, str, str]] = []
    even = _FakeBackend(
        "even", events, accepts=lambda request: request.request_id.endswith("0")
    )
    odd = _FakeBackend(
        "odd", events, accepts=lambda request: request.request_id.endswith("1")
    )
    driver = SessionDriver([even, odd])

    results = driver.run([_request("r0", budget=1), _request("r1", budget=1)])

    assert [result.backend for result in results] == ["even", "odd"]
    assert [
        (cohort.backend, cohort.row_indices) for cohort in driver.last_cohorts
    ] == [
        ("even", (0,)),
        ("odd", (1,)),
    ]


def test_unsupported_sampling_no_greedy_downgrade() -> None:
    events: list[tuple[str, str, str]] = []
    greedy = _FakeBackend("greedy", events, supports_sampling=False)
    sampled = _FakeBackend("sampled", events, supports_sampling=True)
    sampling = SamplingContext(
        temperature=0.8,
        top_p=0.9,
        generator=torch.Generator().manual_seed(17),
    )

    result = SessionDriver([greedy, sampled]).run(
        [_request("sampled-row", budget=1, sampling=sampling)]
    )[0]

    assert result.backend == "sampled"
    assert sampled.sessions[0].request.sampling is sampling
    assert greedy.sessions == []


def test_unsupported_sampling_fails_before_output() -> None:
    events: list[tuple[str, str, str]] = []
    greedy = _FakeBackend("greedy", events, supports_sampling=False)
    sampled = SamplingContext(temperature=0.7)

    with pytest.raises(UnsupportedRequestError, match="sampled-row"):
        SessionDriver([greedy]).run(
            [_request("greedy-row"), _request("sampled-row", sampling=sampled)]
        )

    assert greedy.sessions == []
    assert all(event[0] == "supports" for event in events)


def test_failure_before_verify_no_partial_output() -> None:
    events: list[tuple[str, str, str]] = []
    backend = _FakeBackend(
        "native", events, fail_draft_for=frozenset({"broken"})
    )
    driver = SessionDriver([backend])

    with pytest.raises(RuntimeError, match="draft failed"):
        driver.run([_request("healthy"), _request("broken")])

    assert driver.last_results == ()
    assert [state.output for state in backend.sessions] == [[], []]
    assert all(state.released for state in backend.sessions)
    assert not any(
        event[0] == "verify" and event[2] == "broken" for event in events
    )


def test_snapshot_restore_contract_is_abort_only() -> None:
    assert "abort-only" in (ExecutionBackend.snapshot.__doc__ or "")
    assert "must not resume" in (ExecutionBackend.restore.__doc__ or "")


def test_driver_never_resumes_or_restores_a_failed_session_twice() -> None:
    events: list[tuple[str, str, str]] = []
    backend = _FakeBackend(
        "native", events, fail_draft_for=frozenset({"broken"})
    )

    with pytest.raises(RuntimeError, match="draft failed"):
        SessionDriver([backend]).run([_request("broken")])

    assert [event[0] for event in events].count("restore") == 1
    restore_index = next(
        i for i, event in enumerate(events) if event[0] == "restore"
    )
    assert all(event[0] == "release" for event in events[restore_index + 1 :])


def test_no_progress_round_aborts_all_sessions_and_publishes_no_results() -> (
    None
):
    events: list[tuple[str, str, str]] = []
    backend = _FakeBackend(
        "native", events, no_progress_for=frozenset({"stuck"})
    )
    driver = SessionDriver([backend])

    with pytest.raises(
        driver_module.BackendProgressError,
        match="backend native made no progress for request 'stuck'",
    ):
        driver.run([_request("stuck"), _request("other")])

    assert driver.last_results == ()
    assert all(state.restored and state.released for state in backend.sessions)
    assert [state.output for state in backend.sessions] == [[], []]


def test_primary_execution_error_survives_all_cleanup_failures() -> None:
    events: list[tuple[str, str, str]] = []
    backend = _FakeBackend(
        "native",
        events,
        fail_draft_for=frozenset({"broken"}),
        fail_restore_for=frozenset({"healthy"}),
        fail_release_for=frozenset({"healthy"}),
    )
    driver = SessionDriver([backend])

    with pytest.raises(RuntimeError) as caught:
        driver.run([_request("healthy"), _request("broken")])

    assert str(caught.value) == "draft failed"
    assert driver.last_results == ()
    assert [event[2] for event in events if event[0] == "restore"] == [
        "healthy",
        "broken",
    ]
    assert [event[2] for event in events if event[0] == "release"] == [
        "healthy",
        "broken",
    ]
    assert any(
        "restore failed: healthy" in note for note in caught.value.__notes__
    )
    assert any(
        "release failed: healthy" in note for note in caught.value.__notes__
    )


def test_primary_error_survives_cleanup_when_add_note_is_unavailable() -> None:
    class LegacyRuntimeError(RuntimeError):
        def __getattribute__(self, name: str) -> object:
            if name == "add_note":
                return None
            return super().__getattribute__(name)

    events: list[tuple[str, str, str]] = []
    primary = LegacyRuntimeError("legacy primary")
    backend = _FakeBackend(
        "native",
        events,
        fail_draft_for=frozenset({"broken"}),
        draft_error=primary,
        fail_release_for=frozenset({"broken"}),
    )

    with pytest.raises(LegacyRuntimeError, match="legacy primary") as caught:
        SessionDriver([backend]).run([_request("broken")])

    assert caught.value is primary
    assert backend.sessions[0].released
    cleanup_context = caught.value.__context__
    assert isinstance(cleanup_context, driver_module.SessionCleanupError)
    assert [str(error) for error in cleanup_context.errors] == [
        "release failed: broken"
    ]
    assert "release failed: broken" in str(cleanup_context)


def test_successful_execution_surfaces_cleanup_errors_after_all_releases() -> (
    None
):
    events: list[tuple[str, str, str]] = []
    backend = _FakeBackend(
        "native",
        events,
        fail_release_for=frozenset({"r0", "r1"}),
    )
    driver = SessionDriver([backend])

    with pytest.raises(driver_module.SessionCleanupError) as caught:
        driver.run([_request("r0", budget=1), _request("r1", budget=1)])

    assert driver.last_results == ()
    assert [event[2] for event in events if event[0] == "release"] == [
        "r0",
        "r1",
    ]
    assert [str(error) for error in caught.value.errors] == [
        "release failed: r0",
        "release failed: r1",
    ]


def test_finished_rows_inactive() -> None:
    events: list[tuple[str, str, str]] = []
    backend = _FakeBackend("native", events)

    results = SessionDriver([backend]).run(
        [_request("short", budget=1), _request("long", budget=3)]
    )

    drafted = [event[2] for event in events if event[0] == "draft"]
    assert drafted.count("short") == 1
    assert drafted.count("long") == 3
    assert [len(result.output_token_ids) for result in results] == [1, 3]


def test_dflash_backend_is_a_per_request_session_adapter() -> None:
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    speculator = DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )
    backend = DFlashExecutionBackend(speculator)

    results = SessionDriver([backend]).run(
        [_request("r0", budget=3), _request("r1", budget=5)]
    )

    assert [len(result.output_token_ids) for result in results] == [3, 5]
    assert [result.request_id for result in results] == ["r0", "r1"]
    assert all(result.backend == "dflash-per-request" for result in results)
    assert all(
        result.trace.round_count == len(result.rounds) for result in results
    )


def test_dflash_backend_empty_verify_is_a_finished_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    speculator = DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )
    backend = DFlashExecutionBackend(speculator)
    session = backend.prefill(_request("empty", budget=0))
    monkeypatch.setattr(
        speculator,
        "verify_round",
        lambda _session: VerifyResult(
            accepted_token_ids=[],
            accept=0,
            verified_accept=5,
            committed_count=0,
            finished=True,
        ),
    )

    result = backend.verify(session)

    assert result.accepted_draft_count == 5
    assert result.committed_token_ids == ()
    assert result.cached_token_count == 0
    assert result.emitted_length == 0
    assert result.next_anchor is None
    assert result.finished


def test_spec_session_clear_pending_encapsulates_abort_state() -> None:
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    speculator = DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )
    session = speculator.begin_session(
        torch.tensor([[1, 2, 3]], dtype=torch.long), max_new_tokens=4
    )
    speculator.draft_round(session)

    assert session.has_pending_draft
    session.clear_pending()

    assert not session.has_pending_draft
    with pytest.raises(RuntimeError, match="without a pending draft"):
        speculator.verify_round(session)


def test_verify_round_clamps_over_budget_defensive_entry() -> None:
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    speculator = DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )
    session = speculator.begin_session(
        torch.tensor([[1, 2, 3]], dtype=torch.long), max_new_tokens=1
    )
    speculator.draft_round(session)
    output_before = session.output_ids
    session.emitted.extend([55, 56])

    result = speculator.verify_round(session)

    assert result.accepted_token_ids == []
    assert result.committed_count == 0
    assert result.finished
    assert session.output_ids == output_before
