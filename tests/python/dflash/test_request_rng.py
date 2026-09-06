from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch

from moe_infinity.spec_decode import DFlashSpeculator, read_dflash_config
from moe_infinity.spec_decode._dflash_sample_ops import acceptance_sampled
from moe_infinity.spec_decode.dflash import SpecSession
from tests.python.dflash.fixtures_tiny import (
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
)

PROMPTS = {
    "a": torch.tensor([[3, 7, 11, 2, 5]], dtype=torch.long),
    "b": torch.tensor([[13, 1, 9, 4, 6]], dtype=torch.long),
}


def _tiny_spec() -> DFlashSpeculator:
    target = build_tiny_target(seed=0)
    drafter = build_tiny_drafter(target, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    return DFlashSpeculator.from_models(
        target, drafter, config=config, device="cpu"
    )


def _run_to_completion(
    spec: DFlashSpeculator, session: SpecSession
) -> list[int]:
    while not session.finished:
        spec.draft_round(session)
        spec.verify_round(session)
    return session.output_ids


def _run_requests(
    request_order: Iterable[str], *, include: Iterable[str] = ("a", "b")
) -> dict[str, list[int]]:
    spec = _tiny_spec()
    seeds = {"a": 101, "b": 202}
    sessions = {
        name: spec.begin_session(
            PROMPTS[name],
            max_new_tokens=18,
            temperature=0.8,
            top_p=0.9,
            generator=torch.Generator().manual_seed(seeds[name]),
        )
        for name in include
    }
    order = list(request_order)
    while any(not session.finished for session in sessions.values()):
        for name in order:
            session = sessions.get(name)
            if session is None or session.finished:
                continue
            spec.draft_round(session)
            spec.verify_round(session)
    return {name: session.output_ids for name, session in sessions.items()}


def test_explicit_session_generator_isolated_from_ambient_rng() -> None:
    def generate(ambient_seed: int) -> list[int]:
        spec = _tiny_spec()
        torch.manual_seed(ambient_seed)
        session = spec.begin_session(
            PROMPTS["a"],
            max_new_tokens=18,
            temperature=0.8,
            top_p=0.9,
            generator=torch.Generator().manual_seed(77),
        )
        return _run_to_completion(spec, session)

    assert generate(11) == generate(999)


def test_explicit_session_generator_does_not_advance_ambient_rng() -> None:
    spec = _tiny_spec()
    torch.manual_seed(314159)
    before = torch.random.get_rng_state().clone()

    session = spec.begin_session(
        PROMPTS["a"],
        max_new_tokens=18,
        temperature=0.8,
        generator=torch.Generator().manual_seed(77),
    )
    _run_to_completion(spec, session)

    assert torch.equal(before, torch.random.get_rng_state())


def test_session_outputs_are_invariant_to_request_order() -> None:
    forward = _run_requests(("a", "b"))
    reverse = _run_requests(("b", "a"))

    assert forward == reverse


def test_session_output_is_invariant_to_unrelated_request_composition() -> None:
    alone = _run_requests(("a",), include=("a",))["a"]
    composed = _run_requests(("a", "b"))["a"]

    assert alone == composed


def test_generate_accepts_an_isolated_request_generator() -> None:
    def generate(ambient_seed: int) -> list[int]:
        spec = _tiny_spec()
        torch.manual_seed(ambient_seed)
        return spec.generate(
            PROMPTS["a"],
            max_new_tokens=18,
            temperature=0.8,
            top_p=0.9,
            generator=torch.Generator().manual_seed(77),
        )[0].tolist()

    assert generate(11) == generate(999)


def test_session_retains_complete_proposal_probs_until_verification() -> None:
    spec = _tiny_spec()
    session = spec.begin_session(
        PROMPTS["a"],
        max_new_tokens=18,
        temperature=0.8,
        generator=torch.Generator().manual_seed(77),
    )

    spec.draft_round(session)

    assert session.pending_draft_probs is not None
    assert session.pending_draft_probs.shape == (
        spec.config.block_size - 1,
        spec.config.vocab_size,
    )
    assert torch.equal(
        session.pending_draft_probs.sum(dim=-1),
        torch.ones(spec.config.block_size - 1),
    )

    spec.verify_round(session)
    assert session.pending_draft_probs is None


def test_sampled_verify_refeeds_final_draw_as_next_anchor() -> None:
    spec = _tiny_spec()
    session = spec.begin_session(
        PROMPTS["a"],
        max_new_tokens=18,
        temperature=0.8,
        generator=torch.Generator().manual_seed(77),
    )
    spec.draft_round(session)

    result = spec.verify_round(session)

    assert not result.finished
    assert session.anchor == result.accepted_token_ids[-1]
    spec.draft_round(session)
    assert session._pending_block is not None
    assert int(session._pending_block[0, 0]) == session.anchor


def test_greedy_session_does_not_consume_request_generator() -> None:
    spec = _tiny_spec()
    generator = torch.Generator().manual_seed(77)
    before = generator.get_state().clone()

    session = spec.begin_session(
        PROMPTS["a"],
        max_new_tokens=18,
        temperature=0.0,
        generator=generator,
    )
    _run_to_completion(spec, session)

    assert torch.equal(before, generator.get_state())


def test_finished_session_consumes_no_additional_rng() -> None:
    spec = _tiny_spec()
    generator = torch.Generator().manual_seed(77)
    session = spec.begin_session(
        PROMPTS["a"],
        max_new_tokens=8,
        temperature=0.8,
        generator=generator,
    )
    _run_to_completion(spec, session)
    after_finish = generator.get_state().clone()

    with pytest.raises(RuntimeError, match="finished session"):
        spec.draft_round(session)

    assert torch.equal(after_finish, generator.get_state())


def test_omitted_generator_keeps_global_rng_compatibility() -> None:
    def generate() -> list[int]:
        spec = _tiny_spec()
        torch.manual_seed(77)
        session = spec.begin_session(
            PROMPTS["a"], max_new_tokens=18, temperature=0.8
        )
        return _run_to_completion(spec, session)

    assert generate() == generate()


def test_begin_session_normalizes_negative_top_k_to_disabled() -> None:
    spec = _tiny_spec()

    session = spec.begin_session(
        PROMPTS["a"], max_new_tokens=8, temperature=0.0, top_k=-1
    )

    assert session.sampling.top_k == 0
    assert session.sampling.is_greedy


def test_acceptance_sampled_rejects_generator_device_mismatch_before_draw() -> (
    None
):
    generator = torch.Generator().manual_seed(17)
    before = generator.get_state().clone()
    draft_probs = torch.empty((1, 2), device="meta")
    target_probs = torch.empty((2, 2), device="meta")
    drafts = torch.empty((1,), dtype=torch.long, device="meta")

    with pytest.raises(
        ValueError,
        match="generator device cpu does not match probability device meta",
    ):
        acceptance_sampled(
            draft_probs, target_probs, drafts, generator=generator
        )

    assert torch.equal(before, generator.get_state())


def test_sampled_session_advances_its_request_generator() -> None:
    spec = _tiny_spec()
    generator = torch.Generator().manual_seed(77)
    before = generator.get_state().clone()

    session = spec.begin_session(
        PROMPTS["a"],
        max_new_tokens=18,
        temperature=0.8,
        generator=generator,
    )
    _run_to_completion(spec, session)

    assert not torch.equal(before, generator.get_state())


def test_distinct_request_seeds_produce_non_degenerate_token_streams() -> None:
    streams: list[tuple[int, ...]] = []
    states: list[torch.Tensor] = []
    for seed in (7, 17, 27, 37):
        spec = _tiny_spec()
        generator = torch.Generator().manual_seed(seed)
        session = spec.begin_session(
            PROMPTS["a"],
            max_new_tokens=18,
            temperature=0.8,
            generator=generator,
        )
        streams.append(tuple(_run_to_completion(spec, session)))
        states.append(generator.get_state().clone())

    assert len(set(streams)) > 1
    assert any(not torch.equal(states[0], state) for state in states[1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_acceptance_sampled_supports_matching_cuda_generator() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(17)
    draft_probs = torch.tensor([[0.5, 0.5]], device=device)
    target_probs = torch.tensor([[0.2, 0.8], [0.7, 0.3]], device=device)
    drafts = torch.tensor([0], dtype=torch.long, device=device)

    decisions = [
        acceptance_sampled(
            draft_probs, target_probs, drafts, generator=generator
        )
        for _ in range(16)
    ]

    assert len(decisions) == 16
