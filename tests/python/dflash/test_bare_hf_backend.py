from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

from moe_infinity.spec_decode.backends import PhysicalCohortBackend
from moe_infinity.spec_decode.backends_bare_hf import (
    BareHFCohortResult,
    BatchedBareHFBackend,
)
from moe_infinity.spec_decode.protocols import (
    NativeStepTrace,
    RequestSpec,
    SamplingContext,
)
from tests.python.dflash import test_batched_spec as batched


def _request(
    request_id: str,
    prompt: Sequence[int] = batched.PROMPT_A,
    *,
    budget: int = 8,
    temperature: float = 0.0,
) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        prompt_token_ids=tuple(prompt),
        max_new_tokens=budget,
        sampling=SamplingContext(temperature=temperature),
    )


def _new_tokens(result: BareHFCohortResult) -> list[list[int]]:
    return [list(row) for row in result.generated_token_ids]


def test_backend_declares_physical_mixed_bare_hf_capabilities() -> None:
    spec, _ = batched._tiny_spec()
    backend = BatchedBareHFBackend(spec)

    assert isinstance(backend, PhysicalCohortBackend)
    assert backend.name == "dflash-batched-bare-hf"
    assert backend.capabilities.supports_batch
    assert backend.capabilities.supports_sampling
    assert backend.capabilities.supports_ragged_rows
    assert backend.capabilities.cache_kind == "dense_dynamic"
    assert not backend.capabilities.supports_route_ahead
    assert not backend.capabilities.executor_evidence.wiring_reachable
    assert not backend.capabilities.supports_rich_forward
    assert backend.supports(_request("greedy"))
    assert backend.supports(_request("sampled", temperature=0.7))
    assert backend.cohort_key(_request("greedy")) == backend.cohort_key(
        _request("sampled", temperature=0.7)
    )


def test_backend_direct_entry_preserves_padding_positions_budgets_and_diagnostics() -> (
    None
):
    spec, target = batched._tiny_spec()
    backend = BatchedBareHFBackend(spec)
    prompts = [batched.PROMPT_A, batched.PROMPT_B]
    input_ids, attention_mask, _ = batched._left_pad(prompts)
    budgets = (6, 14)

    result = backend.execute_cohort(
        input_ids,
        max_new_tokens=budgets,
        stop_token_ids=(),
        attention_mask=attention_mask,
    )

    assert result.generated_lengths == budgets
    assert _new_tokens(result) == [
        batched._plain_new(target, prompt, budget)
        for prompt, budget in zip(prompts, budgets)
    ]
    assert result.target_cache is not None
    assert result.step_trace
    assert all(
        step.target_cache_len == step.start for step in result.step_trace
    )


def test_backend_mixed_accept_lengths_refeeds_without_double_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, target = batched._tiny_spec()
    backend = BatchedBareHFBackend(spec)
    prompts = [batched.PROMPT_A, batched.PROMPT_B]
    pads = [0, len(batched.PROMPT_A) - len(batched.PROMPT_B)]
    budget = 21
    streams = batched._greedy_streams(target, prompts, budget)
    true_drafts = batched._true_continuation_drafts(streams, pads)

    def draft_fn(start: int, row: int) -> list[int]:
        if row == 0:
            return true_drafts(start, row)
        base = start - pads[row]
        wrong = (streams[row][base + 1] + 1) % batched.TINY_VOCAB
        return [wrong] * (batched.TINY_BLOCK_SIZE - 1)

    batched._install_scripted_batched_drafter(
        monkeypatch, spec, draft_fn, batch=2
    )
    input_ids, attention_mask, _ = batched._left_pad(prompts)

    result = backend.execute_cohort(
        input_ids,
        max_new_tokens=(budget, budget),
        stop_token_ids=(),
        attention_mask=attention_mask,
    )

    assert _new_tokens(result) == [
        batched._plain_new(target, prompt, budget) for prompt in prompts
    ]
    by_step: dict[int, list[NativeStepTrace]] = {}
    for step in result.step_trace:
        by_step.setdefault(step.prev_start, []).append(step)
    shared_steps = [steps for steps in by_step.values() if len(steps) == 2]
    assert shared_steps
    assert all(
        sorted(step.accept for step in steps)
        == [0, batched.TINY_BLOCK_SIZE - 1]
        for steps in shared_steps
    )
    assert all(
        steps[0].start - previous == 1 for previous, steps in by_step.items()
    )


def test_backend_eos_bonus_finishes_only_its_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, target = batched._tiny_spec()
    backend = BatchedBareHFBackend(spec)
    prompts = [batched.PROMPT_A, batched.PROMPT_B]
    pads = [0, len(batched.PROMPT_A) - len(batched.PROMPT_B)]
    budget = 20
    streams = batched._greedy_streams(target, prompts, budget)
    batched._install_scripted_batched_drafter(
        monkeypatch,
        spec,
        batched._true_continuation_drafts(streams, pads),
        batch=2,
    )
    batched._force_target_argmax_rows(
        monkeypatch, spec, {0: {2: batched.EOS_ID}}
    )
    input_ids, attention_mask, _ = batched._left_pad(prompts)

    result = backend.execute_cohort(
        input_ids,
        max_new_tokens=(budget, budget),
        stop_token_ids=(batched.EOS_ID,),
        attention_mask=attention_mask,
    )

    assert result.generated_token_ids[0][-1] == batched.EOS_ID
    assert result.generated_lengths == (4, budget)
    assert list(result.generated_token_ids[1]) == batched._plain_new(
        target, prompts[1], budget
    )


def test_legacy_adapter_delegates_and_only_adapts_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _ = batched._tiny_spec()
    input_ids = torch.tensor([batched.PROMPT_A, batched.PROMPT_A])
    target_cache = object()
    draft_cache = object()
    trace = NativeStepTrace(5, 0, 6, 1, 6, None)
    calls: list[dict[str, Any]] = []

    def execute(
        self: BatchedBareHFBackend,
        cohort_input_ids: torch.Tensor,
        *,
        max_new_tokens: tuple[int, ...],
        stop_token_ids: tuple[int, ...],
        attention_mask: torch.Tensor,
        sampling_contexts: tuple[SamplingContext, ...] | None = None,
        stop_token_ids_by_row: tuple[tuple[int, ...], ...] | None = None,
    ) -> BareHFCohortResult:
        calls.append(
            {
                "backend": self,
                "input_ids": cohort_input_ids,
                "budgets": max_new_tokens,
                "stops": stop_token_ids,
                "mask": attention_mask,
                "sampling_contexts": sampling_contexts,
                "stop_rows": stop_token_ids_by_row,
            }
        )
        return BareHFCohortResult(
            generated_token_ids=((41,), (51, 52)),
            step_trace=(trace,),
            target_cache=target_cache,
            draft_cache=draft_cache,
        )

    monkeypatch.setattr(BatchedBareHFBackend, "execute_cohort", execute)

    output = spec._generate_batched(
        input_ids,
        max_new_tokens=[1, 2],
        stop_token_ids=[7],
        attention_mask=None,
    )

    assert len(calls) == 1
    assert calls[0]["budgets"] == (1, 2)
    assert calls[0]["stops"] == (7,)
    assert torch.equal(calls[0]["mask"], torch.ones_like(input_ids))
    assert calls[0]["sampling_contexts"] is None
    assert calls[0]["stop_rows"] is None
    assert output[:, input_ids.shape[1] :].tolist() == [[41, 0], [51, 52]]
    assert spec.last_generated_lengths == [1, 2]
    assert spec.step_trace == [trace]
    assert spec.last_target_cache is target_cache
    assert spec.last_draft_cache is draft_cache


def test_legacy_adapter_rejects_non_binary_attention_mask() -> None:
    spec, _ = batched._tiny_spec()
    input_ids = torch.tensor([batched.PROMPT_A, batched.PROMPT_A])
    attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
    attention_mask[0, 0] = 0.5

    with pytest.raises(ValueError, match="0/1 valued"):
        spec._generate_batched(
            input_ids,
            max_new_tokens=2,
            stop_token_ids=None,
            attention_mask=attention_mask,
        )
