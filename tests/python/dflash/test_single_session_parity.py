from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import moe_infinity.spec_decode.dflash as dflash_module
from moe_infinity.spec_decode import (
    DFlashExecutionBackend,
    DFlashSpeculator,
    RequestSpec,
    read_dflash_config,
)
from tests.python.dflash.fixtures_tiny import (
    TINY_BLOCK_SIZE,
    TINY_HIDDEN,
    TINY_VOCAB,
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
    plain_greedy_decode,
)

PROMPT = torch.tensor([[3, 7, 11, 2, 5]], dtype=torch.long)
PROMPT_LEN = int(PROMPT.shape[1])
EOS_ID = 62


def _tiny_spec(seed: int = 0) -> DFlashSpeculator:
    target = build_tiny_target(seed=seed)
    drafter = build_tiny_drafter(target, seed=seed + 1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    return DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )


def _observe_session_driver(
    monkeypatch: pytest.MonkeyPatch, spec: DFlashSpeculator
) -> dict[str, int]:
    calls = {"begin": 0, "draft": 0, "verify": 0}
    real_begin = spec.begin_session
    real_draft = spec.draft_round
    real_verify = spec.verify_round

    def begin(*args: Any, **kwargs: Any):
        calls["begin"] += 1
        return real_begin(*args, **kwargs)

    def draft(*args: Any, **kwargs: Any):
        calls["draft"] += 1
        return real_draft(*args, **kwargs)

    def verify(*args: Any, **kwargs: Any):
        calls["verify"] += 1
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(spec, "begin_session", begin)
    monkeypatch.setattr(spec, "draft_round", draft)
    monkeypatch.setattr(spec, "verify_round", verify)
    return calls


def _run_direct_session(
    spec: DFlashSpeculator,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[int]:
    session = spec.begin_session(
        PROMPT.clone(),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=generator,
    )
    while not session.finished and len(session.output_ids) < max_new_tokens:
        spec.draft_round(session)
        spec.verify_round(session)
    return session.output_ids


def _cache_length(cache: Any) -> int | None:
    return None if cache is None else int(cache.get_seq_length())


def test_dense_cache_length_mismatch_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_spec()
    original_rollback = dflash_module.rollback_target_cache

    def truncate_too_far(cache: Any, snapshot: Any, **kwargs: Any) -> None:
        original_rollback(cache, snapshot, **kwargs)
        cache.crop(int(kwargs["prev_start"]))

    monkeypatch.setattr(
        dflash_module, "rollback_target_cache", truncate_too_far
    )
    session = spec.begin_session(PROMPT.clone(), max_new_tokens=2)
    spec.draft_round(session)

    with pytest.raises(RuntimeError, match="target cache length invariant"):
        spec.verify_round(session)


class _ScriptedHead:
    def __init__(self, draft_fn: Callable[[int], list[int]]) -> None:
        self.draft_fn = draft_fn
        self.calls = 0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        drafts = [int(token) for token in self.draft_fn(self.calls)]
        self.calls += 1
        assert len(drafts) == TINY_BLOCK_SIZE - 1
        logits = torch.zeros(
            1,
            hidden.shape[1],
            TINY_VOCAB,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        offset = hidden.shape[1] - (TINY_BLOCK_SIZE - 1)
        for index, token in enumerate(drafts):
            logits[0, offset + index, token] = 1.0
        return logits


def _install_scripted_drafter(
    monkeypatch: pytest.MonkeyPatch,
    spec: DFlashSpeculator,
    draft_fn: Callable[[int], list[int]],
) -> None:
    monkeypatch.setattr(
        spec,
        "_run_drafter",
        lambda block, context_feature, start, draft_kv: torch.zeros(
            1, TINY_BLOCK_SIZE, TINY_HIDDEN
        ),
    )
    monkeypatch.setattr(spec, "lm_head", _ScriptedHead(draft_fn))


def _force_target_argmax(
    monkeypatch: pytest.MonkeyPatch,
    spec: DFlashSpeculator,
    *,
    prefill: dict[int, int] | None = None,
    verify: dict[int, int] | None = None,
) -> None:
    original = spec._forward_target

    def wrapped(input_ids, past_key_values=None, logits_to_keep=0):
        logits, hidden, cache = original(
            input_ids,
            past_key_values=past_key_values,
            logits_to_keep=logits_to_keep,
        )
        overrides = prefill if int(logits_to_keep) == 1 else verify
        if overrides:
            logits = logits.clone()
            for row, token in overrides.items():
                logits[0, row, :] = -1e9
                logits[0, row, int(token)] = 1e9
        return logits, hidden, cache

    monkeypatch.setattr(spec, "_forward_target", wrapped)


def test_generate_single_delegates_through_real_session_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_spec()
    calls = _observe_session_driver(monkeypatch, spec)

    output = spec.generate(PROMPT.clone(), max_new_tokens=14)

    assert calls["begin"] == 1
    assert calls["draft"] == calls["verify"] == len(spec.step_trace)
    assert calls["verify"] > 0
    assert output.shape == (1, PROMPT_LEN + 14)


@pytest.mark.parametrize("budget", [0, 1])
def test_zero_and_one_token_budgets_prefill_without_unverified_rounds(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    spec = _tiny_spec()
    calls = _observe_session_driver(monkeypatch, spec)

    output = spec.generate(PROMPT.clone(), max_new_tokens=budget)

    assert calls == {"begin": 1, "draft": 0, "verify": 0}
    assert torch.equal(output[:, :PROMPT_LEN], PROMPT)
    assert output.shape == (1, PROMPT_LEN + budget)
    assert output.dtype == PROMPT.dtype
    assert output.device == PROMPT.device
    assert spec.step_trace == []
    assert spec.last_generated_lengths is None
    assert _cache_length(spec.last_target_cache) == PROMPT_LEN
    assert spec.last_draft_cache is None


def test_greedy_generate_matches_direct_session_diagnostics_exactly() -> None:
    budget = 17
    generated_spec = _tiny_spec()
    generated_stats = generated_spec.enable_route_ahead_stats()
    generated = generated_spec.generate(PROMPT.clone(), max_new_tokens=budget)

    session_spec = _tiny_spec()
    session_stats = session_spec.enable_route_ahead_stats()
    session_ids = _run_direct_session(
        session_spec, max_new_tokens=budget, temperature=0.0
    )

    assert generated[0, PROMPT_LEN:].tolist() == session_ids
    assert generated_spec.step_trace == session_spec.step_trace
    assert _cache_length(generated_spec.last_target_cache) == _cache_length(
        session_spec.last_target_cache
    )
    assert _cache_length(generated_spec.last_draft_cache) == _cache_length(
        session_spec.last_draft_cache
    )
    assert generated_stats.as_dict() == session_stats.as_dict()
    assert generated_spec.last_generated_lengths is None
    assert generated.shape == (1, PROMPT_LEN + budget)
    assert generated.dtype == PROMPT.dtype
    assert generated.device == PROMPT.device


def test_sampled_generate_matches_direct_session_with_explicit_generator() -> (
    None
):
    budget = 18
    generate_rng = torch.Generator().manual_seed(917)
    session_rng = torch.Generator().manual_seed(917)

    generated_spec = _tiny_spec()
    generated = generated_spec.generate(
        PROMPT.clone(),
        max_new_tokens=budget,
        temperature=0.8,
        top_k=-1,
        top_p=0.9,
        generator=generate_rng,
    )
    session_spec = _tiny_spec()
    session_ids = _run_direct_session(
        session_spec,
        max_new_tokens=budget,
        temperature=0.8,
        top_k=-1,
        top_p=0.9,
        generator=session_rng,
    )

    assert generated[0, PROMPT_LEN:].tolist() == session_ids
    assert torch.equal(generate_rng.get_state(), session_rng.get_state())
    assert generated_spec.step_trace == session_spec.step_trace
    assert _cache_length(generated_spec.last_target_cache) == _cache_length(
        session_spec.last_target_cache
    )
    assert generated_spec.last_generated_lengths is None


def test_immediate_eos_delegates_to_begin_without_a_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_spec()
    _force_target_argmax(monkeypatch, spec, prefill={-1: EOS_ID})
    calls = _observe_session_driver(monkeypatch, spec)

    output = spec.generate(
        PROMPT.clone(), max_new_tokens=16, stop_token_ids=[EOS_ID]
    )

    assert output[0, PROMPT_LEN:].tolist() == [EOS_ID]
    assert calls == {"begin": 1, "draft": 0, "verify": 0}
    assert spec.step_trace == []
    assert _cache_length(spec.last_target_cache) == PROMPT_LEN


@pytest.mark.parametrize("eos_kind", ["accepted_draft", "bonus"])
def test_eos_inside_verified_commit_preserves_cache_and_trace(
    monkeypatch: pytest.MonkeyPatch, eos_kind: str
) -> None:
    spec = _tiny_spec()
    target = spec.target
    greedy = plain_greedy_decode(target, PROMPT, max_new_tokens=16)[0].tolist()
    if eos_kind == "accepted_draft":
        other = 40
        drafts = [
            greedy[PROMPT_LEN + 1],
            greedy[PROMPT_LEN + 2],
            EOS_ID,
            (other + 1) % TINY_VOCAB,
        ] + [0] * (TINY_BLOCK_SIZE - 5)
        verify = {2: EOS_ID, 3: other}
        expected_cache_length = PROMPT_LEN + 4
    else:
        drafts = greedy[PROMPT_LEN + 1 : PROMPT_LEN + TINY_BLOCK_SIZE]
        verify = {2: EOS_ID}
        expected_cache_length = PROMPT_LEN + 3
    _install_scripted_drafter(monkeypatch, spec, lambda _step: drafts)
    _force_target_argmax(monkeypatch, spec, verify=verify)
    calls = _observe_session_driver(monkeypatch, spec)

    output = spec.generate(
        PROMPT.clone(), max_new_tokens=40, stop_token_ids=[EOS_ID]
    )

    assert output[0, -1].item() == EOS_ID
    assert calls == {"begin": 1, "draft": 1, "verify": 1}
    assert len(spec.step_trace) == 1
    assert _cache_length(spec.last_target_cache) == expected_cache_length
    trace = spec.step_trace[0]
    assert trace.start == trace.prev_start + trace.accept + 1
    assert trace.target_cache_len == trace.start


def test_backend_reports_true_acceptance_for_eos_truncated_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_spec()
    greedy = plain_greedy_decode(spec.target, PROMPT, max_new_tokens=16)[
        0
    ].tolist()
    drafts = [
        greedy[PROMPT_LEN + 1],
        greedy[PROMPT_LEN + 2],
        EOS_ID,
    ] + [0] * (TINY_BLOCK_SIZE - 4)
    _install_scripted_drafter(monkeypatch, spec, lambda _step: drafts)
    _force_target_argmax(monkeypatch, spec, verify={2: EOS_ID, 3: 40})
    backend = DFlashExecutionBackend(spec)
    session = backend.prefill(
        RequestSpec(
            request_id="eos-truncated",
            prompt_token_ids=tuple(PROMPT[0].tolist()),
            max_new_tokens=40,
            stop_token_ids=frozenset({EOS_ID}),
        )
    )

    backend.draft(session)
    result = backend.verify(session)

    assert result.committed_token_ids[-1] == EOS_ID
    assert result.accepted_draft_count == 3
    assert len(result.committed_token_ids) == 3
    assert result.next_anchor is None
    assert result.finished


def test_backend_reports_true_acceptance_for_budget_truncated_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_spec()
    greedy = plain_greedy_decode(spec.target, PROMPT, max_new_tokens=16)[
        0
    ].tolist()
    drafts = greedy[PROMPT_LEN + 1 : PROMPT_LEN + TINY_BLOCK_SIZE]
    _install_scripted_drafter(monkeypatch, spec, lambda _step: drafts)
    backend = DFlashExecutionBackend(spec)
    session = backend.prefill(
        RequestSpec(
            request_id="budget-truncated",
            prompt_token_ids=tuple(PROMPT[0].tolist()),
            max_new_tokens=2,
        )
    )

    backend.draft(session)
    result = backend.verify(session)

    assert result.accepted_draft_count == TINY_BLOCK_SIZE - 1
    assert len(result.committed_token_ids) == 1
    assert result.next_anchor is None
    assert result.finished


def test_partial_final_block_delegates_and_crops_to_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_spec()
    target = spec.target
    budget = TINY_BLOCK_SIZE + 5
    greedy = plain_greedy_decode(target, PROMPT, max_new_tokens=25)[0].tolist()

    def drafts(step: int) -> list[int]:
        anchor_index = PROMPT_LEN + TINY_BLOCK_SIZE * step
        return greedy[anchor_index + 1 : anchor_index + TINY_BLOCK_SIZE]

    _install_scripted_drafter(monkeypatch, spec, drafts)
    calls = _observe_session_driver(monkeypatch, spec)

    output = spec.generate(PROMPT.clone(), max_new_tokens=budget)

    assert torch.equal(
        output, plain_greedy_decode(target, PROMPT, max_new_tokens=budget)
    )
    assert calls == {"begin": 1, "draft": 2, "verify": 2}
    assert spec.step_trace[-1].accept == 4
    assert _cache_length(spec.last_target_cache) == PROMPT_LEN + budget


def _install_malformed_native_backend(spec: DFlashSpeculator) -> None:
    target = spec.target

    class Backend:
        def _native_model_forward_rich(
            self, token_ids, metadata, logits_to_keep=0
        ):
            if metadata is not None:
                return (torch.zeros(1), torch.zeros(1))
            ids = torch.tensor([token_ids], dtype=torch.long)
            outputs = target(ids, use_cache=True, output_hidden_states=True)
            logits = (
                outputs.logits[:, -1:, :] if logits_to_keep else outputs.logits
            )
            return logits, outputs.hidden_states, outputs.past_key_values

    spec.moe = Backend()


def test_malformed_backend_tuple_preserves_existing_error_and_no_trace() -> (
    None
):
    spec = _tiny_spec()
    _install_malformed_native_backend(spec)

    with pytest.raises(
        RuntimeError,
        match=r"_native_model_forward_rich must return \(logits, hidden_states, past_key_values\)",
    ):
        spec.generate(PROMPT.clone(), max_new_tokens=8)

    assert spec.step_trace == []
    assert spec.last_target_cache is None
    assert spec.last_draft_cache is None


def test_drafted_tokens_are_not_exposed_before_successful_verification() -> (
    None
):
    spec = _tiny_spec()
    _install_malformed_native_backend(spec)
    session = spec.begin_session(PROMPT.clone(), max_new_tokens=8)
    verified_output = session.output_ids

    spec.draft_round(session)

    assert session.output_ids == verified_output
    with pytest.raises(RuntimeError, match="must return"):
        spec.verify_round(session)
    assert session.output_ids == verified_output
    assert spec.step_trace == []


def test_hybrid_cache_rollback_runs_through_session_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qwen = pytest.importorskip(
        "tests.python.dflash.test_qwen35_hybrid_rollback"
    )
    target = qwen._tiny_qwen35_target(seed=9)
    drafter = build_tiny_drafter(
        target,
        seed=10,
        block_size=4,
        target_layer_ids=(0, 1, 2, 3),
    )
    config = read_dflash_config(
        make_tiny_drafter_config(
            target.config,
            block_size=4,
            target_layer_ids=(0, 1, 2, 3),
        )
    )
    spec = DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )
    calls = _observe_session_driver(monkeypatch, spec)

    output = spec.generate(PROMPT.clone(), max_new_tokens=12)

    assert calls["begin"] == 1
    assert calls["draft"] == calls["verify"] == len(spec.step_trace)
    assert any(
        trace.accept + 1 < config.block_size for trace in spec.step_trace
    )
    cached_length = _cache_length(spec.last_target_cache)
    assert cached_length is not None
    with torch.no_grad():
        expected = target(
            output[:, :cached_length], use_cache=True
        ).past_key_values
    assert torch.allclose(
        spec.last_target_cache.layers[0].conv_states,
        expected.layers[0].conv_states,
    )
    assert torch.allclose(
        spec.last_target_cache.layers[0].recurrent_states,
        expected.layers[0].recurrent_states,
    )
