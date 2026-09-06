from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch

import moe_infinity.spec_decode.backends_bare_hf as bare_backend_module
from moe_infinity.spec_decode.backends_bare_hf import BatchedBareHFBackend
from moe_infinity.spec_decode.protocols import RequestSpec, SamplingContext
from tests.python.dflash import test_batched_spec as batched
from tests.python.dflash import test_sampled_spec as sampled


def _request(
    request_id: str,
    *,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        prompt_token_ids=tuple(batched.PROMPT_A),
        max_new_tokens=8,
        sampling=SamplingContext(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=torch.Generator().manual_seed(17),
        ),
    )


def _new_tokens(
    output: torch.Tensor, spec: Any, prompt_width: int
) -> list[list[int]]:
    lengths = spec.last_generated_lengths
    assert lengths is not None
    return [
        output[row, prompt_width : prompt_width + length].tolist()
        for row, length in enumerate(lengths)
    ]


def test_bare_hf_backend_advertises_sampled_and_mixed_physical_batches() -> (
    None
):
    spec, _ = batched._tiny_spec()
    backend = BatchedBareHFBackend(spec)

    greedy = _request("greedy", temperature=0.0)
    sampled = _request("sampled", temperature=0.8, top_k=7, top_p=0.9)

    assert backend.capabilities.supports_batch
    assert backend.capabilities.supports_sampling
    assert backend.supports(greedy)
    assert backend.supports(sampled)
    assert backend.cohort_key(greedy) == backend.cohort_key(sampled)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("temperature", [0.0]),
        ("top_k", [0]),
        ("top_p", [1.0]),
        ("generator", [torch.Generator().manual_seed(1)]),
    ],
)
def test_per_row_sampling_lengths_are_validated_before_prefill(
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: Sequence[Any],
) -> None:
    spec, _ = batched._tiny_spec()
    input_ids = torch.tensor([batched.PROMPT_A, batched.PROMPT_A])
    called = False

    def forbidden_prefill(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("prefill must not run")

    monkeypatch.setattr(spec, "_forward_target", forbidden_prefill)
    kwargs: dict[str, Any] = {
        "temperature": [0.0, 0.8],
        "top_k": [0, 5],
        "top_p": [1.0, 0.9],
        "generator": [None, torch.Generator().manual_seed(2)],
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match=f"{argument}.*batch size 2"):
        spec.generate(input_ids, max_new_tokens=4, **kwargs)

    assert not called


def test_same_explicit_generator_for_sampled_rows_rejected_before_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _ = batched._tiny_spec()
    input_ids = torch.tensor([batched.PROMPT_A, batched.PROMPT_A])
    generator = torch.Generator().manual_seed(31)
    initial_state = generator.get_state().clone()
    called = False

    def forbidden_prefill(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("prefill must not run")

    monkeypatch.setattr(spec, "_forward_target", forbidden_prefill)
    with pytest.raises(ValueError, match="same explicit generator object"):
        spec.generate(
            input_ids,
            max_new_tokens=4,
            temperature=[0.7, 0.9],
            generator=[generator, generator],
        )

    assert not called
    assert torch.equal(generator.get_state(), initial_state)


def test_mixed_rows_use_one_physical_target_verify_and_keep_row_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, target = batched._tiny_spec()
    prompts = [batched.PROMPT_A, batched.PROMPT_B]
    input_ids, attention_mask, width = batched._left_pad(prompts)
    verify_batches: list[int] = []
    original_verify = spec._verify_target_block

    def wrapped_verify(
        block: torch.Tensor, target_kv: Any, **kwargs: Any
    ) -> Any:
        verify_batches.append(int(block.shape[0]))
        return original_verify(block, target_kv, **kwargs)

    monkeypatch.setattr(spec, "_verify_target_block", wrapped_verify)
    sampled_generator = torch.Generator().manual_seed(91)
    output = spec.generate(
        input_ids,
        max_new_tokens=[8, 11],
        temperature=[0.0, 0.75],
        top_k=[0, 5],
        top_p=[1.0, 0.85],
        generator=[None, sampled_generator],
        attention_mask=attention_mask,
    )
    rows = _new_tokens(output, spec, width)

    assert verify_batches and all(
        batch_size == 2 for batch_size in verify_batches
    )
    assert rows[0] == batched._plain_new(target, prompts[0], 8)
    assert len(rows[1]) == 11
    assert rows[0] != rows[1][: len(rows[0])]


def test_scalar_sampling_generator_is_cloned_per_sampled_row() -> None:
    spec, _ = batched._tiny_spec()
    input_ids = torch.tensor([batched.PROMPT_A, batched.PROMPT_A])
    generator = torch.Generator().manual_seed(123)
    initial_state = generator.get_state().clone()

    first = spec.generate(
        input_ids,
        max_new_tokens=12,
        temperature=0.8,
        top_p=0.9,
        generator=generator,
    )
    second = spec.generate(
        input_ids,
        max_new_tokens=12,
        temperature=0.8,
        top_p=0.9,
        generator=generator,
    )

    assert torch.equal(first, second)
    assert torch.equal(generator.get_state(), initial_state)
    assert first[0].tolist() == first[1].tolist()


def test_omitted_batch_generators_become_independent_request_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _ = batched._tiny_spec()
    input_ids = torch.tensor([batched.PROMPT_A, batched.PROMPT_A])
    captured: list[SamplingContext] = []
    original = BatchedBareHFBackend.execute_cohort

    def wrapped(
        self: BatchedBareHFBackend,
        cohort_input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        captured.extend(kwargs["sampling_contexts"])
        return original(self, cohort_input_ids, **kwargs)

    monkeypatch.setattr(BatchedBareHFBackend, "execute_cohort", wrapped)
    spec.generate(input_ids, max_new_tokens=4, temperature=0.8)

    generators = [context.generator for context in captured]
    assert all(
        isinstance(generator, torch.Generator) for generator in generators
    )
    assert len({id(generator) for generator in generators}) == len(generators)


def _run_named_rows(names: Sequence[str]) -> dict[str, list[int]]:
    prompts = {
        "a": batched.PROMPT_A,
        "b": batched.PROMPT_B,
        "c": batched.PROMPT_C,
    }
    policies = {
        "a": (0.8, 0, 0.9, 101),
        "b": (0.0, 0, 1.0, 202),
        "c": (1.1, 7, 1.0, 303),
    }
    spec, _ = batched._tiny_spec()
    input_ids, attention_mask, width = batched._left_pad(
        [prompts[name] for name in names]
    )
    output = spec.generate(
        input_ids,
        max_new_tokens=[18] * len(names),
        temperature=[policies[name][0] for name in names],
        top_k=[policies[name][1] for name in names],
        top_p=[policies[name][2] for name in names],
        generator=[
            torch.Generator().manual_seed(policies[name][3]) for name in names
        ],
        attention_mask=attention_mask,
    )
    rows = _new_tokens(output, spec, width)
    return {name: rows[row] for row, name in enumerate(names)}


def test_row_order_and_unrelated_composition_do_not_change_request_stream() -> (
    None
):
    forward = _run_named_rows(["a", "b", "c"])
    reverse = _run_named_rows(["c", "b", "a"])
    composed = _run_named_rows(["a", "c"])

    assert forward == reverse
    assert forward["a"] == composed["a"]
    assert forward["c"] == composed["c"]


def test_sampled_backend_retains_every_slot_proposal_and_row_warp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _ = batched._tiny_spec()
    prompts = [batched.PROMPT_A, batched.PROMPT_B]
    input_ids, attention_mask, _ = batched._left_pad(prompts)
    backend = BatchedBareHFBackend(spec)
    seen_proposals: list[torch.Tensor] = []
    seen_warps: list[tuple[float, int, float]] = []
    original_acceptance = bare_backend_module.acceptance_sampled
    original_warp = bare_backend_module.warped_probs

    def capture_warp(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        seen_warps.append((temperature, top_k, top_p))
        return original_warp(logits, temperature, top_k, top_p)

    def capture_acceptance(
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        drafts: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> Any:
        seen_proposals.append(draft_probs.clone())
        return original_acceptance(
            draft_probs, target_probs, drafts, generator=generator
        )

    monkeypatch.setattr(bare_backend_module, "warped_probs", capture_warp)
    monkeypatch.setattr(
        bare_backend_module, "acceptance_sampled", capture_acceptance
    )
    backend.execute_cohort(
        input_ids,
        max_new_tokens=(12, 12),
        stop_token_ids=(),
        attention_mask=attention_mask,
        sampling_contexts=(
            SamplingContext(
                temperature=0.65,
                top_k=4,
                top_p=0.8,
                generator=torch.Generator().manual_seed(11),
            ),
            SamplingContext(
                temperature=1.2,
                top_k=9,
                top_p=0.95,
                generator=torch.Generator().manual_seed(22),
            ),
        ),
    )

    assert seen_proposals
    assert all(
        proposal.shape == (batched.TINY_BLOCK_SIZE - 1, batched.TINY_VOCAB)
        for proposal in seen_proposals
    )
    assert all(
        torch.allclose(
            proposal.sum(dim=-1),
            torch.ones(batched.TINY_BLOCK_SIZE - 1),
        )
        for proposal in seen_proposals
    )
    assert (0.65, 4, 0.8) in seen_warps
    assert (1.2, 9, 0.95) in seen_warps


def test_budget_zero_and_greedy_rows_consume_no_request_rng() -> None:
    spec, _ = batched._tiny_spec()
    input_ids = torch.tensor([batched.PROMPT_A] * 3)
    zero = torch.Generator().manual_seed(1)
    greedy = torch.Generator().manual_seed(2)
    sampled_generator = torch.Generator().manual_seed(3)
    zero_before = zero.get_state().clone()
    greedy_before = greedy.get_state().clone()
    sampled_before = sampled_generator.get_state().clone()

    spec.generate(
        input_ids,
        max_new_tokens=[0, 8, 8],
        temperature=[0.8, 0.0, 0.8],
        generator=[zero, greedy, sampled_generator],
    )

    assert torch.equal(zero_before, zero.get_state())
    assert torch.equal(greedy_before, greedy.get_state())
    assert not torch.equal(sampled_before, sampled_generator.get_state())


def test_per_row_stop_sets_and_budgets_are_independent() -> None:
    spec, target = batched._tiny_spec()
    prompts = [batched.PROMPT_A, batched.PROMPT_B]
    input_ids, attention_mask, width = batched._left_pad(prompts)
    first = batched._plain_new(target, prompts[0], 1)[0]

    output = spec.generate(
        input_ids,
        max_new_tokens=[8, 13],
        temperature=[0.0, 0.0],
        stop_token_ids=[[first], []],
        attention_mask=attention_mask,
    )
    rows = _new_tokens(output, spec, width)

    assert rows[0] == [first]
    assert rows[1] == batched._plain_new(target, prompts[1], 13)
    assert spec.last_generated_lengths == [1, 13]


def test_sampled_batch_distribution_matches_looped_singletons_and_plain_target() -> (
    None
):
    spec, target = sampled._build_spec(
        vocab_size=sampled.PARITY_VOCAB,
        mask_token_id=sampled.PARITY_MASK_ID,
    )
    input_ids = sampled.PROMPT.repeat(2, 1)
    batched_tokens: list[list[int]] = []
    singleton_tokens: list[list[int]] = []
    plain_tokens: list[list[int]] = []
    runs = 120
    for run in range(runs):
        output = spec.generate(
            input_ids,
            max_new_tokens=sampled.PARITY_MAX_NEW,
            temperature=0.9,
            top_k=5,
            top_p=0.9,
            generator=[
                torch.Generator().manual_seed(10_000 + 2 * run),
                torch.Generator().manual_seed(10_001 + 2 * run),
            ],
        )
        batched_tokens.extend(_new_tokens(output, spec, sampled.PROMPT_LEN))
        torch.manual_seed(20_000 + run)
        singleton = spec.generate(
            sampled.PROMPT,
            max_new_tokens=sampled.PARITY_MAX_NEW,
            temperature=0.9,
            top_k=5,
            top_p=0.9,
        )
        singleton_tokens.append(singleton[0, sampled.PROMPT_LEN :].tolist())
        torch.manual_seed(30_000 + run)
        plain_tokens.append(
            sampled._plain_sampled_decode(
                target,
                sampled.PROMPT,
                sampled.PARITY_MAX_NEW,
                0.9,
                top_k=5,
                top_p=0.9,
            )
        )

    batch_hist = sampled._pooled_histogram(batched_tokens, sampled.PARITY_VOCAB)
    singleton_hist = sampled._pooled_histogram(
        singleton_tokens, sampled.PARITY_VOCAB
    )
    plain_hist = sampled._pooled_histogram(plain_tokens, sampled.PARITY_VOCAB)
    assert sampled._tvd(batch_hist, singleton_hist) <= 0.13
    assert sampled._tvd(batch_hist, plain_hist) <= 0.13
    assert sampled._kl(batch_hist, singleton_hist) <= 0.08
    assert sampled._kl(batch_hist, plain_hist) <= 0.08


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_generator_device_mismatch_is_rejected_before_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _ = batched._tiny_spec()
    spec.device = "cuda"
    called = False

    def forbidden_prefill(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("prefill must not run")

    monkeypatch.setattr(spec, "_forward_target", forbidden_prefill)
    with pytest.raises(ValueError, match="generator device cpu.*cuda"):
        spec.generate(
            torch.tensor([batched.PROMPT_A, batched.PROMPT_A]),
            max_new_tokens=4,
            temperature=[0.8, 0.8],
            generator=[
                torch.Generator().manual_seed(1),
                torch.Generator().manual_seed(2),
            ],
        )
    assert not called
