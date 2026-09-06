from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from moe_infinity.distributed.expert_executor import DistributedExpertExecutor
from moe_infinity.spec_decode._route_ahead_ctx import route_ahead_context
from moe_infinity.spec_decode._route_ahead_stats import RouteAheadStats
from moe_infinity.spec_decode.backends import DFlashExecutionBackend
from moe_infinity.spec_decode.dflash import DFlashConfig, validate_pairing
from moe_infinity.spec_decode.protocols import (
    ExecutorEvidence,
    PairingEvidence,
    SessionTrace,
)
from moe_infinity.utils import ArcherConfig


def _valid_pairing(
    *, checkpoint_scope: tuple[str, ...] = ()
) -> PairingEvidence:
    return PairingEvidence(
        valid=True,
        config_valid=True,
        dimensions_valid=True,
        vocab_valid=True,
        mask_valid=True,
        layers_valid=True,
        block_valid=True,
        module_valid=True,
        validated_checkpoint_scope=checkpoint_scope,
    )


def _executor(*, prefetcher: object | None = None) -> DistributedExpertExecutor:
    config = ArcherConfig.load_from_json(
        {
            "offload_path": "/tmp/moe-infinity-capability-orthogonality",
            "trace_capacity": 16,
            "prefetch": False,
            "speculative_prefetch": True,
        }
    )
    executor = DistributedExpertExecutor(config)
    executor.set_expert_dispatcher(MagicMock(name="ExpertDispatcher"))
    if prefetcher is not None:
        executor.set_prefetcher(prefetcher)
    return executor


def _dispatch(executor: DistributedExpertExecutor) -> None:
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    executor.dispatch_local(
        4,
        torch.zeros(2, 4),
        mask,
        mask.to(torch.float32),
        router_logits=torch.zeros(2, 3),
    )


def test_pairing_and_executor_evidence_are_frozen_and_orthogonal() -> None:
    pairing = _valid_pairing(
        checkpoint_scope=(
            "openai/gpt-oss-120b",
            "z-lab/gpt-oss-120b-DFlash",
        )
    )
    executor = ExecutorEvidence(
        wiring_reachable=False,
        fallback_reason="executor_unreachable",
    )

    assert pairing.valid
    assert not executor.wiring_reachable
    assert not hasattr(pairing, "wiring_reachable")
    assert not hasattr(executor, "valid")
    with pytest.raises(FrozenInstanceError):
        pairing.valid = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        executor.wiring_reachable = True  # type: ignore[misc]


def test_valid_published_pair_with_no_executor_reports_valid_unreachable() -> (
    None
):
    pairing = _valid_pairing(
        checkpoint_scope=(
            "openai/gpt-oss-20b",
            "z-lab/gpt-oss-20b-DFlash",
        )
    )
    speculator = SimpleNamespace(
        moe=SimpleNamespace(_native_model_forward_rich=lambda *_args: None),
        pairing_evidence=pairing,
        executor_evidence=ExecutorEvidence(
            wiring_reachable=False,
            fallback_reason="executor_unreachable",
        ),
    )

    backend = DFlashExecutionBackend(speculator)

    assert backend.capabilities.pairing_evidence.valid
    assert not backend.capabilities.executor_evidence.wiring_reachable
    assert not backend.capabilities.supports_route_ahead


def test_executor_wiring_does_not_make_an_invalid_pair_valid() -> None:
    invalid = DFlashConfig(
        block_size=8,
        mask_token_id=63,
        target_layer_ids=[1, 3],
        num_target_layers=4,
        hidden_size=16,
        vocab_size=64,
    )
    wired_target = SimpleNamespace(
        hidden_size=32,
        vocab_size=64,
        num_hidden_layers=4,
        expert_executor=object(),
    )

    with pytest.raises(ValueError, match="hidden_size"):
        validate_pairing(invalid, wired_target)


def test_prefetch_exception_records_fallback_and_preserves_legacy_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    prefetcher = MagicMock(name="ExpertPrefetcher")
    prefetcher.fetch_experts_lock_cache.side_effect = RuntimeError(
        "prefetch boom"
    )
    executor = _executor(prefetcher=prefetcher)
    stats = RouteAheadStats()
    stats.begin_step()

    with route_ahead_context(prefetcher=prefetcher, stats=stats):
        _dispatch(executor)

    evidence = stats.executor_evidence
    assert evidence.wiring_reachable
    assert evidence.prefetcher_present
    assert evidence.attempted_layers == (4,)
    assert evidence.fired_layers == ()
    assert evidence.actual_expert_union == frozenset({(4, 0), (4, 1), (4, 2)})
    assert evidence.fallback_reason == "prefetch_exception:RuntimeError"
    assert executor._pending_prefetch is not None
    assert executor._pending_prefetch[3] is not None
    assert sorted(
        call.args[1]
        for call in executor.expert_dispatcher.enqueue_expert.call_args_list
    ) == [0, 1, 2]


def test_prefetch_exceptions_cannot_change_waited_expert_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    prefetcher = MagicMock(name="ExpertPrefetcher")
    prefetcher.fetch_experts_lock_cache.side_effect = RuntimeError("lock boom")
    prefetcher.correct_prefetch.side_effect = RuntimeError("correct boom")
    prefetcher.speculative_prefetch.side_effect = RuntimeError("legacy boom")
    executor = _executor(prefetcher=prefetcher)
    executor.expert_dispatcher.wait_expert.return_value = "legacy-output"
    stats = RouteAheadStats()
    stats.begin_step()

    with route_ahead_context(prefetcher=prefetcher, stats=stats):
        _dispatch(executor)
        result = executor.wait_dispatch_local()

    assert result == "legacy-output"
    assert stats.executor_evidence.fallback_reason == (
        "prefetch_exception:RuntimeError"
    )


@pytest.mark.parametrize(
    ("with_context", "mask", "reason"),
    [
        (False, torch.tensor([[1, 0]], dtype=torch.bool), "context_inactive"),
        (True, torch.tensor([[1, 0]], dtype=torch.bool), "prefetcher_absent"),
        (True, torch.zeros(1, 2, dtype=torch.bool), "empty_actual_union"),
    ],
)
def test_capability_misses_fall_back_without_changing_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    with_context: bool,
    mask: torch.Tensor,
    reason: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    executor = _executor()
    stats = RouteAheadStats()
    stats.begin_step()

    def run() -> None:
        executor.dispatch_local(
            2,
            torch.zeros(mask.shape[0], 4),
            mask,
            mask.to(torch.float32),
            router_logits=torch.zeros(mask.shape),
        )

    if with_context:
        with route_ahead_context(stats=stats):
            run()
        evidence = stats.executor_evidence
    else:
        run()
        evidence = executor.last_executor_evidence

    assert evidence.fallback_reason == reason
    assert executor._pending_prefetch is not None
    assert executor._pending_prefetch[3] is not None


def test_common_trace_serializes_pairing_and_executor_evidence_separately() -> (
    None
):
    pairing = _valid_pairing()
    executor = ExecutorEvidence(
        wiring_reachable=True,
        prefetcher_present=True,
        attempted_layers=(1,),
        fired_layers=(1,),
        actual_expert_union=frozenset({(1, 3)}),
        prefetched_bytes=4096,
        coverage=1.0,
        wasted_prefetch_bytes=0,
        cache_hit_rate=0.75,
    )
    trace = SessionTrace(
        request_id="req",
        backend="native",
        cache_kind="dense_dynamic",
        sampled=False,
        pairing_evidence=pairing,
        executor_evidence=executor,
    )

    payload = trace.as_dict()
    assert payload["pairing_evidence"]["valid"] is True
    assert payload["executor_evidence"]["wiring_reachable"] is True
    assert "wiring_reachable" not in payload["pairing_evidence"]
    assert "valid" not in payload["executor_evidence"]
