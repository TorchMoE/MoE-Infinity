from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import cast, final

import pytest
import torch

from moe_infinity.spec_decode._dflash_sample_ops import acceptance_sampled
from moe_infinity.spec_decode.protocols import (
    BackendCapabilities,
    CacheAdapter,
    CacheKind,
    CacheSnapshot,
    DenseCacheAdapter,
    NativeStepTrace,
    RequestSpec,
    SamplingContext,
    SessionRoundResult,
    SessionTrace,
)


@final
class _FakeDenseCache:
    def __init__(self, length: int) -> None:
        self.length = length
        self.crop_calls: list[int] = []

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, length: int) -> None:
        self.crop_calls.append(length)
        self.length = length


def _step(*, accept: int = 2, emitted_len: int = 3) -> NativeStepTrace:
    return NativeStepTrace(
        prev_start=5,
        accept=accept,
        start=5 + accept + 1,
        emitted_len=emitted_len,
        target_cache_len=5 + accept + 1,
        draft_cache_len=5,
    )


def _seed_ambient(seed: int) -> None:
    manual_seed = cast(Callable[[int], torch.Generator], torch.manual_seed)
    _ = manual_seed(seed)


def test_sampling_context_exposes_request_scoped_sampling_semantics() -> None:
    generator = torch.Generator().manual_seed(17)
    sampled = SamplingContext(
        temperature=0.7, top_k=8, top_p=0.9, generator=generator
    )
    greedy = SamplingContext(temperature=0.0)

    assert sampled.is_sampled
    assert not sampled.is_greedy
    assert sampled.generator is generator
    assert greedy.is_greedy
    assert not greedy.is_sampled


def test_sampling_context_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _ = SamplingContext(temperature=-0.1)
    with pytest.raises(ValueError, match="top_k"):
        _ = SamplingContext(top_k=-1)
    with pytest.raises(ValueError, match="top_p"):
        _ = SamplingContext(top_p=0.0)
    with pytest.raises(ValueError, match="top_p"):
        _ = SamplingContext(top_p=1.1)


def test_request_spec_rejects_invalid_identity_prompt_and_budget() -> None:
    with pytest.raises(ValueError, match="request_id"):
        _ = RequestSpec("", (1,), 1)
    with pytest.raises(ValueError, match="prompt_token_ids"):
        _ = RequestSpec("req", (), 1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        _ = RequestSpec("req", (1,), -1)


def test_request_spec_sampling_uses_a_default_factory() -> None:
    sampling_field = next(
        field for field in fields(RequestSpec) if field.name == "sampling"
    )
    assert sampling_field.default_factory is SamplingContext


def test_request_and_round_contracts_have_stable_derived_counts() -> None:
    request = RequestSpec(
        request_id="req-7",
        prompt_token_ids=(4, 5, 6),
        max_new_tokens=12,
        stop_token_ids=frozenset({2}),
        sampling=SamplingContext(temperature=0.0),
    )
    result = SessionRoundResult(
        accepted_draft_count=2,
        committed_token_ids=(7, 8, 9),
        next_anchor=9,
        target_cache_length=8,
        emitted_length=3,
        finished=False,
        finish_reason=None,
    )

    assert request.prompt_length == 3
    assert request.is_sampled is False
    assert result.cached_token_count == 3
    assert result.emitted_token_count == 3
    assert result.commit_block_token_ids == (7, 8, 9)
    assert not hasattr(result, "emitted_token_ids")


def test_session_round_result_enforces_accept_plus_one_commit_invariant() -> (
    None
):
    with pytest.raises(ValueError, match="accepted drafts plus one"):
        _ = SessionRoundResult(
            accepted_draft_count=2,
            committed_token_ids=(7, 8),
            next_anchor=8,
            target_cache_length=8,
            emitted_length=2,
            finished=False,
            finish_reason=None,
        )


def test_session_round_result_allows_a_finished_empty_noop() -> None:
    result = SessionRoundResult(
        accepted_draft_count=4,
        committed_token_ids=(),
        next_anchor=None,
        target_cache_length=8,
        emitted_length=0,
        finished=True,
        finish_reason="length",
    )

    assert result.cached_token_count == 0
    assert result.emitted_token_count == 0
    assert result.next_anchor is None


def test_session_round_result_rejects_an_unfinished_empty_commit() -> None:
    with pytest.raises(ValueError, match="finished no-op"):
        _ = SessionRoundResult(
            accepted_draft_count=0,
            committed_token_ids=(),
            next_anchor=None,
            target_cache_length=8,
            emitted_length=0,
            finished=False,
            finish_reason=None,
        )


def test_session_round_result_exposes_the_approved_signature() -> None:
    assert tuple(field.name for field in fields(SessionRoundResult)) == (
        "accepted_draft_count",
        "committed_token_ids",
        "next_anchor",
        "target_cache_length",
        "emitted_length",
        "finished",
        "finish_reason",
        "fallback_reason",
    )


@pytest.mark.parametrize("cache_kind", ["dense_dynamic", "paged", "other"])
def test_backend_capabilities_expose_the_approved_signature(
    cache_kind: CacheKind,
) -> None:
    capabilities = BackendCapabilities(
        supports_batch=True,
        supports_sampling=True,
        supports_ragged_rows=False,
        cache_kind=cache_kind,
        supports_route_ahead=True,
        supports_rich_forward=False,
    )

    assert tuple(field.name for field in fields(BackendCapabilities)) == (
        "supports_batch",
        "supports_sampling",
        "supports_ragged_rows",
        "cache_kind",
        "supports_route_ahead",
        "supports_rich_forward",
        "pairing_evidence",
        "executor_evidence",
    )
    assert capabilities.cache_kind == cache_kind


def test_backend_capabilities_reject_invalid_cache_kind() -> None:
    invalid_kind = cast(CacheKind, cast(object, "dense"))
    with pytest.raises(ValueError, match="cache_kind"):
        _ = BackendCapabilities(
            supports_batch=False,
            supports_sampling=False,
            supports_ragged_rows=False,
            cache_kind=invalid_kind,
            supports_route_ahead=False,
            supports_rich_forward=False,
        )


@final
class _StructuralAdapter:
    def snapshot(self) -> CacheSnapshot:
        return CacheSnapshot(logical_length=0)

    def restore(self, snapshot: CacheSnapshot) -> None:
        del snapshot

    def append(self, token_count: int) -> None:
        del token_count

    def truncate(self, logical_length: int) -> None:
        del logical_length

    def logical_length(self) -> int:
        return 0

    def release(self) -> None:
        return None


def test_cache_adapter_is_a_runtime_structural_protocol() -> None:
    assert getattr(CacheAdapter, "_is_protocol", False)
    assert isinstance(_StructuralAdapter(), CacheAdapter)


def test_cache_snapshot_rejects_negative_logical_length() -> None:
    with pytest.raises(ValueError, match="logical_length"):
        _ = CacheSnapshot(logical_length=-1)


def test_dense_cache_adapter_implements_the_complete_cache_lifecycle() -> None:
    cache = _FakeDenseCache(length=5)
    adapter = DenseCacheAdapter(cache)
    snapshot = adapter.snapshot()

    assert snapshot == CacheSnapshot(logical_length=5)
    assert adapter.logical_length() == 5

    cache.length = 8
    adapter.append(3)
    assert adapter.logical_length() == 8

    adapter.truncate(6)
    assert adapter.logical_length() == 6
    assert cache.crop_calls == [6]

    snapshot = adapter.snapshot()
    cache.length = 9
    adapter.append(3)
    adapter.restore(snapshot)
    assert adapter.logical_length() == 6
    assert cache.crop_calls == [6, 6]

    adapter.release()
    assert cache.crop_calls == [6, 6, 0]
    with pytest.raises(RuntimeError, match="released"):
        _ = adapter.logical_length()


def test_dense_cache_adapter_rejects_invalid_length_transitions() -> None:
    adapter = DenseCacheAdapter(_FakeDenseCache(length=3))

    with pytest.raises(ValueError, match="token_count"):
        adapter.append(-1)
    with pytest.raises(ValueError, match="logical_length"):
        adapter.truncate(4)


def test_session_trace_aggregates_native_step_trace_without_a_second_schema() -> (
    None
):
    first = _step(accept=2, emitted_len=3)
    second = NativeStepTrace(
        prev_start=8,
        accept=0,
        start=9,
        emitted_len=4,
        target_cache_len=9,
        draft_cache_len=8,
    )
    trace = SessionTrace(
        request_id="req-7",
        backend="native",
        cache_kind="dense_dynamic",
        sampled=False,
        route_ahead_status="disabled",
    )
    trace.append(first)
    trace.append(second)
    trace.rollback = 1
    trace.replay = 1
    trace.finish_reason = "length"

    assert trace.round_count == 2
    assert trace.accepted == 2
    assert trace.committed == 4
    assert trace.emitted == 4
    assert first.committed_count == 3
    assert trace.as_dict() == {
        "request_id": "req-7",
        "backend": "native",
        "cache_kind": "dense_dynamic",
        "sampled": False,
        "round_count": 2,
        "accepted": 2,
        "committed": 4,
        "emitted": 4,
        "rollback": 1,
        "replay": 1,
        "finish_reason": "length",
        "route_ahead_status": "disabled",
        "pairing_evidence": {
            "valid": False,
            "config_valid": False,
            "dimensions_valid": False,
            "vocab_valid": False,
            "mask_valid": False,
            "layers_valid": False,
            "block_valid": False,
            "module_valid": None,
            "validated_checkpoint_scope": (),
            "failure_reason": None,
        },
        "executor_evidence": {
            "wiring_reachable": False,
            "prefetcher_present": False,
            "attempted_layers": (),
            "fired_layers": (),
            "actual_expert_union": (),
            "prefetched_bytes": 0,
            "coverage": None,
            "wasted_prefetch_bytes": None,
            "cache_hit_rate": None,
            "fallback_reason": None,
        },
    }


def test_acceptance_sampled_uses_only_the_explicit_generator() -> None:
    draft_probs = torch.tensor([[0.5, 0.5]])
    target_probs = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    drafts = torch.tensor([0])

    _seed_ambient(999)
    ambient_before = torch.random.get_rng_state().clone()
    first = acceptance_sampled(
        draft_probs,
        target_probs,
        drafts,
        generator=torch.Generator().manual_seed(123),
    )
    ambient_after = torch.random.get_rng_state()
    second = acceptance_sampled(
        draft_probs,
        target_probs,
        drafts,
        generator=torch.Generator().manual_seed(123),
    )

    assert first == second
    assert torch.equal(ambient_before, ambient_after)


def test_acceptance_sampled_keeps_ambient_rng_compatibility() -> None:
    draft_probs = torch.tensor([[0.5, 0.5]])
    target_probs = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    drafts = torch.tensor([0])

    _seed_ambient(321)
    first = acceptance_sampled(draft_probs, target_probs, drafts)
    _seed_ambient(321)
    second = acceptance_sampled(draft_probs, target_probs, drafts)

    assert first == second
