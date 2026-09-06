from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from moe_infinity.entrypoints.big_modeling import MoE
from moe_infinity.spec_decode import SessionTrace
from moe_infinity.spec_decode.dflash import _normalize_stop_rows
from tests.python.dflash import test_batched_spec as batched
from tests.python.dflash.test_engine_wire import (
    PROMPT,
    _tiny_moe_shell,
    _tiny_speculator,
)

WARNING_TEXT = (
    "MoE.generate() is deprecated. Use MoE.serve() for continuous batching "
    "with higher throughput. MoE.generate() will be removed in a future version."
)


@pytest.mark.parametrize(
    ("value", "expected", "per_row"),
    [
        (17, ((17,), (17,)), False),
        ([17, 19], ((17, 19), (17, 19)), False),
        ([[17], [19, 23]], ((17,), (19, 23)), True),
    ],
)
def test_stop_rows_accept_scalar_shared_and_per_row_forms(
    value: object,
    expected: tuple[tuple[int, ...], ...],
    per_row: bool,
) -> None:
    target = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))

    rows, actual_per_row = _normalize_stop_rows(target, value, batch=2)

    assert rows == expected
    assert actual_per_row is per_row


@pytest.mark.parametrize(
    "value",
    [True, [1, [2]], [[1], 2], [None, 2]],
)
def test_stop_rows_reject_malformed_or_boolean_values(value: object) -> None:
    target = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))

    with pytest.raises(ValueError, match="stop_token_ids"):
        _normalize_stop_rows(target, value, batch=2)


def test_direct_bare_hf_batch_accepts_scalar_stop_id() -> None:
    spec, target = batched._tiny_spec()
    stop_id = batched._plain_new(target, batched.PROMPT_A, 1)[0]
    input_ids = torch.tensor([batched.PROMPT_A, batched.PROMPT_A])

    output = spec.generate(
        input_ids,
        max_new_tokens=8,
        stop_token_ids=stop_id,
    )

    assert spec.last_generated_lengths == [1, 1]
    assert output[:, input_ids.shape[1]].tolist() == [stop_id, stop_id]


def test_moe_generate_preserves_deprecation_warning_text_and_stacklevel() -> (
    None
):
    shell, _ = _tiny_moe_shell()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        shell.generate(
            cast(torch.LongTensor, torch.tensor([PROMPT])),
            do_sample=False,
            max_new_tokens=1,
        )

    warning = next(
        item for item in caught if item.category is DeprecationWarning
    )
    assert str(warning.message) == WARNING_TEXT
    assert warning.filename == __file__


def test_direct_rich_batch_uses_independent_sessions_and_shared_trace_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell, target = _tiny_moe_shell()
    spec = _tiny_speculator(shell, target)
    begin_shapes: list[tuple[int, ...]] = []
    real_begin = spec.begin_session

    def begin(input_ids: torch.Tensor, **kwargs: Any):
        begin_shapes.append(tuple(input_ids.shape))
        return real_begin(input_ids, **kwargs)

    monkeypatch.setattr(spec, "begin_session", begin)
    input_ids = torch.tensor([PROMPT, PROMPT], dtype=torch.long)

    output = spec.generate(input_ids, max_new_tokens=[2, 5])

    assert begin_shapes == [(1, len(PROMPT)), (1, len(PROMPT))]
    assert output.shape == (2, len(PROMPT) + 5)
    assert torch.equal(output[:, : len(PROMPT)], input_ids)
    assert spec.last_generated_lengths == [2, 5]
    assert len(spec.last_session_traces) == 2
    assert all(
        isinstance(trace, SessionTrace) for trace in spec.last_session_traces
    )
    assert [trace.request_id for trace in spec.last_session_traces] == [
        "direct-0",
        "direct-1",
    ]
    assert all(
        trace.backend == "dflash-per-request"
        for trace in spec.last_session_traces
    )


def test_moe_generate_rich_batch_delegates_once_without_engine_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell, target = _tiny_moe_shell()
    spec = _tiny_speculator(shell, target)
    input_ids = torch.tensor([PROMPT, PROMPT], dtype=torch.long)
    expected = torch.cat(
        [input_ids, torch.tensor([[31, 0, 0], [41, 42, 43]])], dim=1
    )
    calls: list[dict[str, Any]] = []

    def direct_generate(ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls.append({"input_ids": ids, **kwargs})
        spec.last_generated_lengths = [1, 3]
        return expected

    monkeypatch.setattr(spec, "generate", direct_generate)
    engine_calls: list[dict[str, Any]] = []
    real_engine_generate = shell._native_generation_engine.generate

    def engine_generate(**kwargs: Any):
        engine_calls.append(kwargs)
        return real_engine_generate(**kwargs)

    monkeypatch.setattr(
        shell._native_generation_engine, "generate", engine_generate
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        actual = shell.generate(
            cast(torch.LongTensor, input_ids),
            do_sample=False,
            max_new_tokens=[1, 3],
            speculative_draft=spec,
        )

    assert torch.equal(actual, expected)
    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == [1, 3]
    assert engine_calls == []
    assert shell._native_generation_engine.spec_strategy is None


def test_real_moe_rich_batch_accepts_scalar_eos_and_detaches_strategy() -> None:
    shell, target = _tiny_moe_shell()
    spec = _tiny_speculator(shell, target)
    input_ids = torch.tensor([PROMPT, PROMPT], dtype=torch.long)
    probe = spec.generate(input_ids[:1], max_new_tokens=1)
    stop_id = int(probe[0, len(PROMPT)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        output = shell.generate(
            cast(torch.LongTensor, input_ids),
            do_sample=False,
            max_new_tokens=8,
            eos_token_id=stop_id,
            speculative_draft=spec,
        )

    assert output.shape == (2, len(PROMPT) + 1)
    assert output[:, len(PROMPT)].tolist() == [stop_id, stop_id]
    assert spec.last_generated_lengths == [1, 1]
    assert shell._native_generation_engine.spec_strategy is None


def test_moe_rich_batch_detaches_strategy_when_speculator_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell, target = _tiny_moe_shell()
    spec = _tiny_speculator(shell, target)
    input_ids = torch.tensor([PROMPT, PROMPT], dtype=torch.long)

    def fail(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise RuntimeError("direct batch failed")

    monkeypatch.setattr(spec, "generate", fail)

    with warnings.catch_warnings(), pytest.raises(
        RuntimeError, match="direct batch failed"
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        shell.generate(
            cast(torch.LongTensor, input_ids),
            do_sample=False,
            speculative_draft=spec,
        )

    assert shell._native_generation_engine.spec_strategy is None


def test_generation_config_is_normalized_for_selected_dflash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell, target = _tiny_moe_shell()
    spec = _tiny_speculator(shell, target)
    input_ids = torch.tensor([PROMPT, PROMPT], dtype=torch.long)
    calls: list[dict[str, Any]] = []

    def direct_generate(ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls.append({"input_ids": ids, **kwargs})
        return torch.cat([ids, torch.full((2, 4), 7, dtype=ids.dtype)], dim=1)

    monkeypatch.setattr(spec, "generate", direct_generate)
    generation_config = SimpleNamespace(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=4,
        eos_token_id=2,
        pad_token_id=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shell.generate(
            cast(torch.LongTensor, input_ids),
            generation_config=generation_config,
            speculative_draft=spec,
        )

    assert len(calls) == 1
    assert calls[0]["max_new_tokens"] == 4
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["top_k"] == 0
    assert calls[0]["top_p"] == 1.0
    assert calls[0]["stop_token_ids"] == [2]


def test_qwen35_sampled_dflash_rejects_before_speculator_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = MoE.__new__(MoE)
    shell.model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_moe"),
        generate=lambda *_args, **_kwargs: pytest.fail(
            "HF fallback was called"
        ),
    )
    shell.use_native_engine = True
    shell._native_generation_engine = SimpleNamespace(
        generate=lambda **_kwargs: None
    )
    shell.max_seq_length = 64
    resolve_calls: list[object] = []

    def resolve(value: object) -> None:
        resolve_calls.append(value)

    monkeypatch.setattr(shell, "_resolve_spec_strategy", resolve)

    with warnings.catch_warnings(), pytest.raises(ValueError, match="greedy"):
        warnings.simplefilter("ignore", DeprecationWarning)
        shell.generate(
            cast(torch.LongTensor, torch.tensor([[1, 2]])),
            do_sample=True,
            temperature=0.7,
            speculative_draft=object(),
        )

    assert resolve_calls == []
