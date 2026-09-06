from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field
from typing import Callable

import pytest
import torch

from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.sequence import SamplingParams, SequenceStatus
from moe_infinity.serving.spec_session_driver import (
    EXECUTION_CONTEXT_TEMPORARY_DYNAMIC,
    SpecSessionDriver,
)


class _Cache:
    def __init__(self, *, fail_crop: bool = False) -> None:
        self.released = False
        self.fail_crop = fail_crop
        self.crop_calls = 0
        self.crop_error = RuntimeError("cache cleanup boom")

    def crop(self, length: int) -> None:
        self.crop_calls += 1
        if self.fail_crop:
            raise self.crop_error
        if length == 0:
            self.released = True


@dataclass
class _Session:
    emitted: list[int]
    max_new_tokens: int
    target_kv: _Cache = field(default_factory=_Cache)
    draft_kv: _Cache = field(default_factory=_Cache)
    finished: bool = False
    pending: bool = False

    @property
    def output_ids(self) -> list[int]:
        return self.emitted[: self.max_new_tokens]

    def clear_pending(self) -> None:
        self.pending = False


class _Speculator:
    def __init__(self) -> None:
        self.moe = types.SimpleNamespace(_cached_past_key_values=None)
        self.begin_calls: list[dict[str, object]] = []
        self.sessions: list[_Session] = []
        self.events: list[tuple[str, int]] = []
        self.verify_hook: Callable[[], None] | None = None
        self.replace_target_cache = False
        self.draft_error: BaseException | None = None
        self.verify_error: BaseException | None = None

    def begin_session(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> _Session:
        self.begin_calls.append(dict(kwargs))
        anchor = int(input_ids[0, -1]) + 1
        session = _Session(
            emitted=[anchor], max_new_tokens=int(kwargs["max_new_tokens"])
        )
        session.finished = session.max_new_tokens <= 1
        self.sessions.append(session)
        self.events.append(("begin", anchor))
        return session

    def draft_round(self, session: _Session) -> object:
        assert self.moe._cached_past_key_values is session.target_kv
        if self.draft_error is not None:
            raise self.draft_error
        session.pending = True
        self.events.append(("draft", session.emitted[-1]))
        return types.SimpleNamespace(tokens=4, expert_bytes=64)

    def verify_round(self, session: _Session) -> object:
        assert self.moe._cached_past_key_values is session.target_kv
        assert session.pending
        if self.verify_error is not None:
            raise self.verify_error
        self.events.append(("verify", session.emitted[-1]))
        if self.verify_hook is not None:
            self.verify_hook()
        if self.replace_target_cache:
            session.target_kv = _Cache()
        session.pending = False
        remaining = session.max_new_tokens - len(session.emitted)
        for _ in range(min(2, remaining)):
            session.emitted.append(session.emitted[-1] + 1)
        session.finished = len(session.emitted) >= session.max_new_tokens
        return types.SimpleNamespace(
            accepted_token_ids=session.emitted[-min(2, remaining) :],
            committed_count=min(2, remaining),
            finished=session.finished,
        )


class _Model:
    config = types.SimpleNamespace(vocab_size=128, eos_token_id=99)

    def eval(self) -> None:
        pass

    def forward(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> types.SimpleNamespace:
        del kwargs
        logits = torch.full((*input_ids.shape, 128), -1e9)
        for row in range(input_ids.shape[0]):
            for col in range(input_ids.shape[1]):
                logits[row, col, int(input_ids[row, col]) + 1] = 0
        return types.SimpleNamespace(logits=logits)


class _Offload:
    def __init__(self) -> None:
        self.request_id = 0
        self.expert_tracer = types.SimpleNamespace(create_entry=lambda: 0)
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]

    def _generate_request_id(self) -> int:
        value = self.request_id
        self.request_id += 1
        return value


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": 8,
        "max_tokens_per_step": 16,
        "block_size": 4,
        "num_layers": 1,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype": "float32",
        "eos_token_id": 99,
        "num_kv_blocks": 32,
        "verify_token_budget": 8,
        "verify_expert_byte_budget": 128,
        "verify_token_deficit_cap": 32,
        "verify_expert_byte_deficit_cap": 512,
    }
    config.update(overrides)
    return config


def _engine(speculator: object, **config: object) -> ContinuousBatchingEngine:
    return ContinuousBatchingEngine(
        model=_Model(),
        engine=_Offload(),
        config=_config(**config),
        speculative_draft=speculator,
    )


def test_driver_record_owns_temporary_context_and_logical_commit_state() -> (
    None
):
    speculator = _Speculator()
    driver = SpecSessionDriver(speculator)

    record = driver.begin(
        request_id="r0",
        seq_id=7,
        prompt_token_ids=[3, 4],
        max_new_tokens=3,
        temperature=0.7,
        top_k=5,
        top_p=0.8,
        stop_token_ids=[99],
        callbacks=(),
    )

    assert record.request_id == "r0" and record.seq_id == 7
    assert record.spec_session is speculator.sessions[0]
    assert record.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    assert record.decode_state.invariant_holds()
    assert driver.commit(record) == (5,)
    assert record.decode_state.cached_len == 3
    assert record.decode_state.invariant_holds()


def test_driver_tracks_replaced_private_target_cache_for_cleanup() -> None:
    speculator = _Speculator()
    speculator.replace_target_cache = True
    driver = SpecSessionDriver(speculator)
    record = driver.begin(
        request_id="replace",
        seq_id=8,
        prompt_token_ids=[3],
        max_new_tokens=3,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        stop_token_ids=[99],
        callbacks=(),
    )
    original = record.spec_session.target_kv
    _ = driver.commit(record)
    _ = driver.draft(record)
    _ = driver.verify(record)
    replacement = record.spec_session.target_kv

    assert record.execution_context.target_cache is replacement
    assert original.released
    assert not replacement.released
    assert not record.spec_session.draft_kv.released
    assert len(record.execution_context.owned_caches) == 2
    driver.release(record)
    assert replacement.released


def test_engine_creates_and_persists_one_session_per_sequence() -> None:
    speculator = _Speculator()
    engine = _engine(speculator)
    engine.add_request(
        "multi",
        [10],
        SamplingParams(temperature=0.7, top_p=0.9, max_tokens=5),
        n=2,
    )

    first = engine.step()

    assert [output.token_id for output in first] == [11, 11]
    assert len(speculator.sessions) == 2
    assert len(engine.speculative_sessions) == 2
    assert all(
        engine._sequences[seq_id].status is SequenceStatus.DRAFT
        for seq_id in engine._request_to_seq_ids["multi"]
    )

    second = engine.step()
    assert [output.token_id for output in second] == [12, 13, 12, 13]
    assert len(engine.speculative_sessions) == 2


def test_sampled_parameters_stops_and_callbacks_are_preserved() -> None:
    speculator = _Speculator()
    engine = _engine(speculator)
    streamed: list[int] = []
    engine.add_request(
        "sampled",
        [20],
        SamplingParams(temperature=0.75, top_k=7, top_p=0.85, max_tokens=4),
        on_token=lambda output: streamed.append(output.token_id),
    )

    result = engine.run_until_done()

    assert result == {"sampled": [21, 22, 23, 24]}
    assert streamed == [21, 22, 23, 24]
    assert speculator.begin_calls[0]["temperature"] == 0.75
    assert speculator.begin_calls[0]["top_k"] == 7
    assert speculator.begin_calls[0]["top_p"] == 0.85
    assert speculator.begin_calls[0]["stop_token_ids"] == [99]


def test_cancellation_during_verify_suppresses_output_and_releases_caches() -> (
    None
):
    speculator = _Speculator()
    engine = _engine(speculator)
    streamed: list[int] = []
    engine.add_request(
        "cancel",
        [30],
        SamplingParams(temperature=0.6, max_tokens=5),
        on_token=lambda output: streamed.append(output.token_id),
    )
    assert [output.token_id for output in engine.step()] == [31]
    session = speculator.sessions[0]
    speculator.verify_hook = lambda: engine.abort_request("cancel")

    assert engine.step() == []
    assert streamed == [31]
    assert session.target_kv.released and session.draft_kv.released
    assert engine.speculative_sessions == {}
    assert engine.has_pending_requests() is False


def test_draft_failure_cleans_request_without_streaming_unverified_tokens() -> (
    None
):
    speculator = _Speculator()
    engine = _engine(speculator)
    streamed: list[int] = []
    engine.add_request(
        "draft-fail",
        [30],
        SamplingParams(temperature=0.5, max_tokens=5),
        on_token=lambda output: streamed.append(output.token_id),
    )
    assert [output.token_id for output in engine.step()] == [31]
    session = speculator.sessions[0]
    original = RuntimeError("draft secret at /srv/private/model.bin")
    speculator.draft_error = original

    with pytest.raises(RuntimeError, match="draft secret") as caught:
        engine.step()

    assert caught.value is original
    assert streamed == [31]
    assert session.target_kv.released and session.draft_kv.released
    assert engine.speculative_sessions == {}
    assert engine.scheduler._verify_demands == {}
    assert engine.has_pending_requests() is False
    assert engine.get_request_failure("draft-fail") == {
        "phase": "draft",
        "failure_type": "RuntimeError",
        "code": "speculative_draft_failed",
    }


def test_verify_failure_cleans_pending_draft_without_streaming_it() -> None:
    speculator = _Speculator()
    engine = _engine(speculator)
    streamed: list[int] = []
    engine.add_request(
        "verify-fail",
        [40],
        SamplingParams(temperature=0.5, max_tokens=5),
        on_token=lambda output: streamed.append(output.token_id),
    )
    assert [output.token_id for output in engine.step()] == [41]
    session = speculator.sessions[0]
    original = ValueError("verify boom")
    speculator.verify_error = original

    with pytest.raises(ValueError, match="verify boom") as caught:
        engine.step()

    assert caught.value is original
    assert streamed == [41]
    assert session.pending is False
    assert session.target_kv.released and session.draft_kv.released
    assert engine.speculative_sessions == {}
    assert engine.scheduler._verify_demands == {}
    assert engine.has_pending_requests() is False
    assert engine.get_request_failure("verify-fail")["phase"] == "verify"


def test_backend_failure_survives_private_cache_cleanup_failure() -> None:
    speculator = _Speculator()
    engine = _engine(speculator)
    engine.add_request(
        "cleanup-fail",
        [50],
        SamplingParams(temperature=0.5, max_tokens=5),
    )
    _ = engine.step()
    session = speculator.sessions[0]
    session.target_kv.fail_crop = True
    original = RuntimeError("draft primary")
    speculator.draft_error = original

    with pytest.raises(RuntimeError, match="draft primary") as caught:
        engine.step()

    assert caught.value is original
    assert session.target_kv.crop_calls == 1
    assert session.draft_kv.released
    assert engine.speculative_sessions == {}
    assert engine.has_pending_requests() is False
    assert caught.value.session_cleanup_errors == (
        session.target_kv.crop_error,
    )
    notes = getattr(caught.value, "__notes__", None)
    if notes is not None:
        assert any("cache cleanup boom" in note for note in notes)
    assert engine.get_request_failure("cleanup-fail") == {
        "phase": "draft",
        "failure_type": "RuntimeError",
        "code": "speculative_draft_failed",
    }


def test_cleanup_metadata_does_not_depend_on_add_note() -> None:
    class LegacyRuntimeError(RuntimeError):
        def __getattribute__(self, name: str) -> object:
            if name == "add_note":
                return None
            return super().__getattribute__(name)

    speculator = _Speculator()
    engine = _engine(speculator)
    engine.add_request(
        "legacy-cleanup-fail",
        [50],
        SamplingParams(temperature=0.5, max_tokens=5),
    )
    _ = engine.step()
    session = speculator.sessions[0]
    session.target_kv.fail_crop = True
    original = LegacyRuntimeError("draft primary")
    speculator.draft_error = original

    with pytest.raises(LegacyRuntimeError, match="draft primary") as caught:
        engine.step()

    assert caught.value is original
    assert caught.value.session_cleanup_errors == (
        session.target_kv.crop_error,
    )


def test_cleanup_reporting_failures_are_logged_without_masking_primary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class HostileRuntimeError(RuntimeError):
        def __setattr__(self, name: str, value: object) -> None:
            if name == "session_cleanup_errors":
                raise TypeError("metadata blocked")
            super().__setattr__(name, value)

        def add_note(self, note: str) -> None:
            del note
            raise RuntimeError("note blocked")

    speculator = _Speculator()
    engine = _engine(speculator)
    engine.add_request(
        "hostile-cleanup-fail",
        [50],
        SamplingParams(temperature=0.5, max_tokens=5),
    )
    _ = engine.step()
    session = speculator.sessions[0]
    session.target_kv.fail_crop = True
    original = HostileRuntimeError("draft primary")
    speculator.draft_error = original

    with (
        caplog.at_level(logging.DEBUG, logger="moe_infinity.serving.engine"),
        pytest.raises(HostileRuntimeError, match="draft primary") as caught,
    ):
        engine.step()

    assert caught.value is original
    assert session.target_kv.crop_calls == 1
    assert session.draft_kv.released
    assert engine.speculative_sessions == {}
    assert engine.has_pending_requests() is False
    assert any(
        "cleanup metadata attachment failed for request "
        "hostile-cleanup-fail during draft; "
        "primary=HostileRuntimeError reporting=TypeError" in message
        for message in caplog.messages
    )
    assert any(
        "cleanup note attachment failed for request "
        "hostile-cleanup-fail during draft; "
        "primary=HostileRuntimeError reporting=RuntimeError" in message
        for message in caplog.messages
    )


def test_cleanup_note_lookup_failure_is_logged_without_masking_primary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class LookupHostileRuntimeError(RuntimeError):
        def __getattribute__(self, name: str) -> object:
            if name == "add_note":
                raise LookupError("note lookup blocked")
            return super().__getattribute__(name)

    speculator = _Speculator()
    engine = _engine(speculator)
    engine.add_request(
        "lookup-cleanup-fail",
        [50],
        SamplingParams(temperature=0.5, max_tokens=5),
    )
    _ = engine.step()
    session = speculator.sessions[0]
    session.target_kv.fail_crop = True
    original = LookupHostileRuntimeError("draft primary")
    record = next(iter(engine.speculative_sessions.values()))

    with (
        caplog.at_level(logging.DEBUG, logger="moe_infinity.serving.engine"),
        pytest.raises(
            LookupHostileRuntimeError, match="draft primary"
        ) as caught,
    ):
        engine._fail_speculative_request(record, "draft", original)
        raise original

    assert caught.value is original
    assert caught.value.session_cleanup_errors == (
        session.target_kv.crop_error,
    )
    assert session.draft_kv.released
    assert engine.speculative_sessions == {}
    assert engine.has_pending_requests() is False
    assert any(
        "cleanup note-lookup attachment failed for request "
        "lookup-cleanup-fail during draft; "
        "primary=LookupHostileRuntimeError reporting=LookupError" in message
        for message in caplog.messages
    )


def test_mixed_scheduled_work_splits_eligible_from_normal_fallback() -> None:
    speculator = _Speculator()
    engine = _engine(speculator)
    engine.add_request(
        "eligible",
        [40],
        SamplingParams(temperature=0.7, max_tokens=3),
    )
    engine.add_request(
        "fallback",
        [50],
        SamplingParams(temperature=0.0, repetition_penalty=1.1, max_tokens=1),
    )

    outputs = engine.step()

    assert [(row.request_id, row.token_id) for row in outputs] == [
        ("eligible", 41),
        ("fallback", 51),
    ]
    assert len(speculator.sessions) == 1
    assert (
        engine._sequences[engine._request_to_seq_ids["eligible"][0]].status
        is SequenceStatus.DRAFT
    )


def test_grammar_metadata_keeps_request_on_normal_serving_fallback() -> None:
    speculator = _Speculator()
    engine = _engine(speculator)
    params = SamplingParams(temperature=0.0, max_tokens=1)
    params.grammar = "root ::= 'x'"  # type: ignore[attr-defined]
    engine.add_request("grammar", [70], params)

    outputs = engine.step()

    assert [output.token_id for output in outputs] == [71]
    assert speculator.sessions == []


def test_verify_admission_and_diagnostics_report_temporary_dynamic_mode() -> (
    None
):
    speculator = _Speculator()
    engine = _engine(
        speculator,
        verify_token_budget=2,
        verify_expert_byte_budget=32,
    )
    engine.add_request(
        "admit",
        [60],
        SamplingParams(temperature=0.5, max_tokens=3),
    )
    assert [output.token_id for output in engine.step()] == [61]

    assert engine.step() == []
    record = next(iter(engine.speculative_sessions.values()))
    assert record.pending_draft is not None
    assert engine.get_stats()["speculative_execution_context"] == (
        EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    )

    outputs = engine.step()
    assert [output.token_id for output in outputs] == [62, 63]
    assert outputs[-1].finished
    assert record.decode_state.invariant_holds()
    assert record.decode_state.cached_len == record.decode_state.prompt_len + 3
