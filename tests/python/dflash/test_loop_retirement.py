from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Hashable

import pytest
import torch

from moe_infinity.spec_decode.backends_bare_hf import BareHFCohortResult
from moe_infinity.spec_decode.dflash import DFlashSpeculator
from moe_infinity.spec_decode.protocols import (
    BackendCapabilities,
    NativeStepTrace,
    RequestSpec,
    SamplingContext,
    SessionTrace,
)
from moe_infinity.spec_decode.session_driver import (
    DriverResult,
    SessionDriver,
    UnsupportedRequestError,
)
from tests.python.dflash import test_batched_spec as batched


def _call_tail(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def test_legacy_batch_adapter_has_no_semantic_decode_path() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(DFlashSpeculator._generate_batched))
    )
    calls = {
        _call_tail(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    forbidden = {
        "acceptance_lengths",
        "acceptance_sampled",
        "committed_tokens_ragged",
        "committed_tokens_sampled",
        "warped_probs",
        "_forward_target",
        "_verify_target_block",
        "_run_drafter",
        "_snapshot_target_cache",
        "_rollback_target_cache",
        "commit_step",
        "execute_cohort",
    }

    assert not any(isinstance(node, ast.While) for node in ast.walk(tree))
    assert calls.isdisjoint(forbidden)
    assert {"RequestSpec", "BatchedBareHFBackend", "SessionDriver"} <= calls
    assert "run_physical_cohort" in calls


def _capabilities(*, sampling: bool = True) -> BackendCapabilities:
    return BackendCapabilities(
        supports_batch=True,
        supports_sampling=sampling,
        supports_ragged_rows=True,
        cache_kind="dense_dynamic",
        supports_route_ahead=False,
        supports_rich_forward=False,
    )


def _request(
    request_id: str,
    *,
    prompt: tuple[int, ...] = (1, 2, 3),
    budget: int = 2,
    sampling: SamplingContext | None = None,
    stops: frozenset[int] = frozenset(),
) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        prompt_token_ids=prompt,
        max_new_tokens=budget,
        sampling=sampling or SamplingContext(),
        stop_token_ids=stops,
    )


@dataclass
class _PhysicalBackend:
    name: str
    events: list[tuple[str, str]]
    accepts: frozenset[str] | None = None
    sampling: bool = True
    malformed: bool = False
    failure: BaseException | None = None
    capabilities: BackendCapabilities = field(init=False)
    execute_calls: list[dict[str, Any]] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:
        self.capabilities = _capabilities(sampling=self.sampling)

    def supports(self, request: RequestSpec) -> bool:
        self.events.append(("supports", request.request_id))
        return self.accepts is None or request.request_id in self.accepts

    def cohort_key(self, request: RequestSpec) -> Hashable:
        del request
        return "dense-mixed"

    def execute_cohort(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        self.events.append(("execute", self.name))
        self.execute_calls.append(dict(kwargs))
        if self.failure is not None:
            raise self.failure
        rows = int(input_ids.shape[0]) - int(self.malformed)
        generated = tuple((30 + row,) for row in range(rows))
        sampling_contexts = kwargs.get(
            "sampling_contexts",
            tuple(SamplingContext() for _ in range(int(input_ids.shape[0]))),
        )
        if sampling_contexts is None:
            sampling_contexts = tuple(
                SamplingContext() for _ in range(int(input_ids.shape[0]))
            )
        traces = tuple(
            SessionTrace(
                request_id=f"backend-{row}",
                backend=self.name,
                cache_kind="dense_dynamic",
                sampled=sampling_contexts[row].is_sampled,
                emitted=1,
                finish_reason="length",
            )
            for row in range(rows)
        )
        return SimpleNamespace(
            generated_token_ids=generated,
            generated_lengths=tuple(len(row) for row in generated),
            session_traces=traces,
            step_trace=(),
            target_cache=None,
            draft_cache=None,
        )


def test_physical_driver_selects_every_row_before_one_execution() -> None:
    events: list[tuple[str, str]] = []
    backend = _PhysicalBackend("physical", events)
    sampled = SamplingContext(
        temperature=0.7, generator=torch.Generator().manual_seed(9)
    )
    requests = (
        _request("r0", budget=1, stops=frozenset({8})),
        _request("r1", budget=1, sampling=sampled, stops=frozenset({9})),
    )
    driver = SessionDriver([backend])  # type: ignore[list-item]

    run = driver.run_physical_cohort(
        torch.tensor([[1, 2, 3], [0, 2, 3]]),
        requests=requests,
        attention_mask=torch.tensor([[1, 1, 1], [0, 1, 1]]),
    )

    assert events == [
        ("supports", "r0"),
        ("supports", "r1"),
        ("execute", "physical"),
    ]
    assert run.backend_result.generated_token_ids == ((30,), (31,))
    assert [result.request_id for result in run.results] == ["r0", "r1"]
    assert [result.output_token_ids for result in run.results] == [(30,), (31,)]
    assert [result.trace.request_id for result in run.results] == ["r0", "r1"]
    assert [result.trace.sampled for result in run.results] == [False, True]
    assert driver.last_results == run.results
    assert driver.last_cohorts[0].row_indices == (0, 1)


def test_physical_driver_never_downgrades_sampling() -> None:
    events: list[tuple[str, str]] = []
    greedy = _PhysicalBackend("greedy", events, sampling=False)
    sampled = _PhysicalBackend("sampled", events)
    request = _request("sampled-row", sampling=SamplingContext(temperature=0.8))

    run = SessionDriver([greedy, sampled]).run_physical_cohort(  # type: ignore[list-item]
        torch.tensor([[1, 2, 3]]),
        requests=(request,),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    assert run.results[0].backend == "sampled"
    assert ("execute", "greedy") not in events


def test_physical_driver_passes_optional_metadata_in_one_backend_call() -> None:
    events: list[tuple[str, str]] = []
    backend = _PhysicalBackend("physical", events)

    SessionDriver([backend]).run_physical_cohort(  # type: ignore[list-item]
        torch.tensor([[1, 2, 3]]),
        requests=(_request("r0", budget=1),),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )

    assert len(backend.execute_calls) == 1
    assert backend.execute_calls[0]["sampling_contexts"] is None
    assert backend.execute_calls[0]["stop_token_ids_by_row"] is None


def test_physical_driver_rejects_split_cohort_before_output() -> None:
    events: list[tuple[str, str]] = []
    first = _PhysicalBackend("first", events, accepts=frozenset({"r0"}))
    second = _PhysicalBackend("second", events, accepts=frozenset({"r1"}))
    driver = SessionDriver([first, second])  # type: ignore[list-item]

    with pytest.raises(UnsupportedRequestError, match="one physical cohort"):
        driver.run_physical_cohort(
            torch.tensor([[1, 2, 3], [1, 2, 3]]),
            requests=(_request("r0"), _request("r1")),
            attention_mask=torch.ones(2, 3, dtype=torch.long),
        )

    assert not any(event[0] == "execute" for event in events)
    assert driver.last_results == ()


@pytest.mark.parametrize("mode", ["malformed", "failure"])
def test_physical_driver_publishes_nothing_on_failure(mode: str) -> None:
    events: list[tuple[str, str]] = []
    backend = _PhysicalBackend(
        "physical",
        events,
        malformed=mode == "malformed",
        failure=RuntimeError("verify failed") if mode == "failure" else None,
    )
    driver = SessionDriver([backend])  # type: ignore[list-item]

    with pytest.raises((RuntimeError, ValueError)):
        driver.run_physical_cohort(
            torch.tensor([[1, 2, 3], [1, 2, 3]]),
            requests=(_request("r0"), _request("r1")),
            attention_mask=torch.ones(2, 3, dtype=torch.long),
        )

    assert driver.last_results == ()


def test_physical_driver_clears_published_state_before_normalization() -> None:
    events: list[tuple[str, str]] = []
    backend = _PhysicalBackend("physical", events)
    driver = SessionDriver([backend])  # type: ignore[list-item]
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    driver.run_physical_cohort(
        input_ids,
        requests=(_request("valid", budget=1),),
        attention_mask=attention_mask,
    )
    assert driver.last_results
    assert driver.last_cohorts

    with pytest.raises(TypeError, match="RequestSpec"):
        driver.run_physical_cohort(
            input_ids,
            requests=(object(),),  # pyright: ignore[reportArgumentType]
            attention_mask=attention_mask,
        )

    assert driver.last_results == ()
    assert driver.last_cohorts == ()


def test_legacy_adapter_preserves_requests_results_traces_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _ = batched._tiny_spec()
    input_ids = torch.tensor(
        [[0, 0, *batched.PROMPT_B], batched.PROMPT_A], dtype=torch.int32
    )
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    sampling = (
        SamplingContext(),
        SamplingContext(
            temperature=0.8,
            top_k=4,
            top_p=0.9,
            generator=torch.Generator().manual_seed(17),
        ),
    )
    target_cache = object()
    draft_cache = object()
    step = NativeStepTrace(5, 0, 6, 1, 6, None)
    traces = (
        SessionTrace(
            "direct-0",
            "physical",
            "dense_dynamic",
            False,
            emitted=1,
            finish_reason="stop",
        ),
        SessionTrace(
            "direct-1",
            "physical",
            "dense_dynamic",
            True,
            emitted=2,
            finish_reason="length",
        ),
    )
    backend_result = BareHFCohortResult(
        generated_token_ids=((41,), (51, 52)),
        step_trace=(step,),
        target_cache=target_cache,
        draft_cache=draft_cache,
        session_traces=traces,
    )
    captured: dict[str, Any] = {}

    def run_physical(
        self: SessionDriver,
        cohort_input_ids: torch.Tensor,
        *,
        requests: tuple[RequestSpec, ...],
        attention_mask: torch.Tensor,
    ) -> Any:
        captured.update(
            driver=self,
            input_ids=cohort_input_ids,
            requests=requests,
            attention_mask=attention_mask,
        )
        results = tuple(
            DriverResult(
                request_id=request.request_id,
                output_token_ids=backend_result.generated_token_ids[row],
                finish_reason=traces[row].finish_reason or "length",
                backend="physical",
                trace=traces[row],
                rounds=(),
            )
            for row, request in enumerate(requests)
        )
        return SimpleNamespace(results=results, backend_result=backend_result)

    monkeypatch.setattr(
        SessionDriver, "run_physical_cohort", run_physical, raising=False
    )

    output = spec._generate_batched(
        input_ids,
        max_new_tokens=[1, 2],
        stop_token_ids=None,
        attention_mask=attention_mask,
        sampling_contexts=sampling,
        stop_token_ids_by_row=((41,), ()),
    )

    requests = captured["requests"]
    assert [request.prompt_token_ids for request in requests] == [
        tuple(batched.PROMPT_B),
        tuple(batched.PROMPT_A),
    ]
    assert [request.max_new_tokens for request in requests] == [1, 2]
    assert [request.stop_token_ids for request in requests] == [
        frozenset({41}),
        frozenset(),
    ]
    assert [request.sampling for request in requests] == list(sampling)
    assert torch.equal(captured["attention_mask"], attention_mask)
    assert output[:, input_ids.shape[1] :].tolist() == [[41, 0], [51, 52]]
    assert output.dtype == input_ids.dtype
    assert spec.last_generated_lengths == [1, 2]
    assert spec.step_trace == [step]
    assert spec.last_target_cache is target_cache
    assert spec.last_draft_cache is draft_cache
    assert spec.last_session_results == captured[
        "driver"
    ].last_results or tuple(
        result.request_id for result in spec.last_session_results
    ) == ("direct-0", "direct-1")
    assert spec.last_session_traces == traces
