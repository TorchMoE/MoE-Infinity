"""Unit tests for the Track A3 route-ahead wire (DFlash verify prefetch).

Autonomous correctness gate for Track A3 of
``.sisyphus/plans/dflash-deferred-tracks-plan.md``. Covers:

(a) context ACTIVE -> ``dispatch_local`` derives the ACTUAL routed union via
    the A1 helper and invokes the A2 explicit-set
    ``speculative_prefetch(layer_id, expert_ids=union, prefetch_layer_id=
    layer_id)``, pinned via ``fetch_experts_lock_cache`` BEFORE any
    ``enqueue_expert`` read (A0 section 2 ordering), with the legacy
    mean/topk pooled prefetch suppressed (A0 section 3);
(b) context INACTIVE -> no pin, no explicit prefetch, and the legacy
    overlap/deferred paths behave exactly as pre-A3 (byte-identical);
(c) the context manager resets even when the wrapped verify call raises
    (try/finally token reset), restores nested handles, and isolates threads;
(d) ``DFlashSpeculator._verify_target_block`` activates the context around
    the verify forward and resolves the prefetcher from the MoE offload
    engine, defaulting to ``None`` (executor fallback / resident no-op);
(e) Track A4 offload coupling: consecutive dispatches pin exactly ONE layer
    each (the global ``ReplaceCacheCandidates`` guard), while gpt-oss's
    force-resident ``SyncGptOssMLP`` loop reports read-only route-ahead
    metrics without pinning or prefetching resident experts.

Construction mirrors ``test_speculative_prefetch.py`` (A2) and
``tests/python/unit/test_distributed_smoke.py``: real Python objects with
mocked dispatcher/archer engine, CPU-only, no native extension.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from moe_infinity.distributed.expert_executor import DistributedExpertExecutor
from moe_infinity.memory.expert_prefetcher import ExpertPrefetcher
from moe_infinity.spec_decode._route_ahead_ctx import (
    current_prefetcher,
    is_active,
    route_ahead_context,
)
from moe_infinity.spec_decode.dflash import DFlashSpeculator
from moe_infinity.utils import ArcherConfig

LAYER_ID = 3
# [3 tokens, 8 experts]; the union routed by ANY token is {0, 1, 2, 5, 7}.
ROUTER_MASK = torch.tensor(
    [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=torch.bool,
)
UNION = [0, 1, 2, 5, 7]
HIDDEN = torch.zeros(3, 4)
WEIGHTS = torch.zeros(3, 8)
LOGITS = torch.zeros(3, 8)


@pytest.fixture(autouse=True)
def _cpu_dispatch_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "moe_infinity.distributed.expert_executor.IOProfiler", None
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


def _make_executor(*, overlap: bool = False, prefetcher=None, policy=None):
    config = ArcherConfig.load_from_json(
        {
            "offload_path": "/tmp/moe-infinity-route-ahead-test",
            "trace_capacity": 16,
            "prefetch": False,
            "speculative_prefetch": True,
            "speculative_prefetch_overlap": overlap,
        }
    )
    executor = DistributedExpertExecutor(config)
    executor.set_expert_dispatcher(MagicMock(name="ExpertDispatcher"))
    if prefetcher is not None:
        if policy in ("observe", "enforce"):
            prefetcher._overlap_active.return_value = True
            prefetcher._overlap_caps = SimpleNamespace(
                scheduler_ready=True, dispatcher_timing_ready=True
            )
        executor.set_prefetcher(prefetcher)
    return executor


def _compute_sample(*, invocation_id, layer_id):
    return SimpleNamespace(invocation_id=invocation_id, layer_id=layer_id)


def _make_real_prefetcher(num_layers: int = 8, num_experts: int = 8):
    """A2-style real ExpertPrefetcher; tensor id = layer * 100 + expert."""
    prefetcher = ExpertPrefetcher.__new__(ExpertPrefetcher)
    prefetcher.num_layers = num_layers
    prefetcher.num_experts = num_experts
    engine = MagicMock(name="ArcherEngine")
    engine.get_node_default_device.return_value = 0
    prefetcher.archer_engine = engine
    prefetcher.expert_tensor_map = {
        (layer, expert): layer * 100 + expert
        for layer in range(num_layers)
        for expert in range(num_experts)
    }
    prefetcher._last_speculative_prediction = set()
    return prefetcher, engine


def _dispatch(executor, *, router_logits=None, prefetcher=None):
    executor.dispatch_local(
        LAYER_ID,
        HIDDEN,
        ROUTER_MASK,
        WEIGHTS,
        router_logits=router_logits,
        prefetcher=prefetcher,
    )


def _enqueued_experts(executor) -> list[int]:
    return sorted(
        c.args[1]
        for c in executor.expert_dispatcher.enqueue_expert.call_args_list
    )


def _issued_tensor_ids(engine) -> list[int]:
    # Mechanism-agnostic route-ahead issuance readout: a batched
    # ``prefetch_tensors([...])`` call carries the same ordered tensor ids the
    # per-expert ``enqueue_prefetch`` fallback would, so flatten the batched
    # calls in order when present and fall back otherwise.
    batched = getattr(engine, "prefetch_tensors", None)
    batched_calls = getattr(batched, "call_args_list", None)
    if batched_calls:
        issued: list[int] = []
        for call in batched_calls:
            issued.extend(call.args[0])
        return issued
    return [call.args[0] for call in engine.enqueue_prefetch.call_args_list]


# ---------------------------------------------------------------------------
# (a) context active -> exact-union pin + prefetch for the current layer
# ---------------------------------------------------------------------------


def test_active_context_prefetches_exact_union_for_current_layer():
    prefetcher = MagicMock(name="ExpertPrefetcher")
    executor = _make_executor(overlap=True)
    events = []
    prefetcher.fetch_experts_lock_cache.side_effect = (
        lambda layer, ids: events.append(("lock", layer, list(ids)))
    )
    prefetcher.speculative_prefetch.side_effect = (
        lambda layer, **kw: events.append(("prefetch", layer, kw))
    )
    executor.expert_dispatcher.enqueue_expert.side_effect = (
        lambda layer, expert, gpu, remote: events.append(("enqueue", expert))
    )

    with route_ahead_context(prefetcher=prefetcher):
        _dispatch(executor, router_logits=LOGITS)
    assert not is_active()

    prefetcher.fetch_experts_lock_cache.assert_called_once_with(LAYER_ID, UNION)
    prefetcher.speculative_prefetch.assert_called_once_with(
        LAYER_ID, expert_ids=UNION, prefetch_layer_id=LAYER_ID
    )
    first_read = next(i for i, e in enumerate(events) if e[0] == "enqueue")
    assert events.index(("lock", LAYER_ID, UNION)) < first_read
    assert (
        next(i for i, e in enumerate(events) if e[0] == "prefetch") < first_read
    )
    # Routing untouched: exactly the union experts are dispatched to compute.
    assert [e[1] for e in events if e[0] == "enqueue"] == UNION
    # A0 section 3: legacy pooled prediction suppressed for this dispatch.
    assert executor._pending_prefetch == (
        None,
        LAYER_ID,
        UNION,
        None,
        [],
        None,
    )


def test_active_context_falls_back_to_executor_prefetcher():
    prefetcher, engine = _make_real_prefetcher()
    executor = _make_executor(overlap=True, prefetcher=prefetcher)
    trigger_spy = MagicMock(wraps=executor.trigger_speculative_prefetch)
    executor.trigger_speculative_prefetch = trigger_spy

    with route_ahead_context():
        _dispatch(executor, router_logits=LOGITS)

    engine.replace_cache_candidates.assert_called_once_with(
        [300, 301, 302, 305, 307]
    )
    assert _issued_tensor_ids(engine) == [300, 301, 302, 305, 307]
    assert prefetcher._last_speculative_prediction == set(UNION)
    trigger_spy.assert_not_called()
    assert executor._pending_prefetch == (
        prefetcher,
        LAYER_ID,
        UNION,
        None,
        [],
        None,
    )

    # The pending correct_prefetch(layer+1, expert_list) no-ops because the
    # recorded prediction IS the actual union; nothing further is enqueued.
    executor.wait_dispatch_local()
    engine.replace_cache_candidates.assert_called_once()
    assert _issued_tensor_ids(engine) == [300, 301, 302, 305, 307]
    assert prefetcher._last_speculative_prediction == set()


def test_active_context_prefetcher_arg_wins_over_context_handle():
    arg_prefetcher = MagicMock(name="ArgPrefetcher")
    ctx_prefetcher = MagicMock(name="CtxPrefetcher")
    self_prefetcher = MagicMock(name="SelfPrefetcher")
    executor = _make_executor(prefetcher=self_prefetcher)

    with route_ahead_context(prefetcher=ctx_prefetcher):
        _dispatch(executor, prefetcher=arg_prefetcher)

    arg_prefetcher.fetch_experts_lock_cache.assert_called_once_with(
        LAYER_ID, UNION
    )
    arg_prefetcher.speculative_prefetch.assert_called_once_with(
        LAYER_ID, expert_ids=UNION, prefetch_layer_id=LAYER_ID
    )
    ctx_prefetcher.fetch_experts_lock_cache.assert_not_called()
    self_prefetcher.fetch_experts_lock_cache.assert_not_called()


def test_active_context_without_any_prefetcher_is_noop():
    executor = _make_executor(overlap=True)

    with route_ahead_context():
        _dispatch(executor, router_logits=LOGITS)

    # Resident mode: nothing pinned/prefetched, legacy flow untouched.
    assert _enqueued_experts(executor) == UNION
    assert executor._pending_prefetch is not None
    assert executor._pending_prefetch[3] is LOGITS


def test_active_context_empty_union_skips_pin():
    prefetcher = MagicMock(name="ExpertPrefetcher")
    executor = _make_executor(prefetcher=prefetcher)

    with route_ahead_context():
        executor.dispatch_local(
            LAYER_ID,
            HIDDEN[:1],
            torch.zeros(1, 8, dtype=torch.bool),
            WEIGHTS[:1],
        )

    prefetcher.fetch_experts_lock_cache.assert_not_called()
    prefetcher.speculative_prefetch.assert_not_called()


# ---------------------------------------------------------------------------
# (b) context inactive -> legacy paths byte-identical to pre-A3
# ---------------------------------------------------------------------------


def test_inactive_context_overlap_path_byte_identical():
    prefetcher = MagicMock(name="ExpertPrefetcher")
    executor = _make_executor(overlap=True, prefetcher=prefetcher)
    trigger_spy = MagicMock(wraps=executor.trigger_speculative_prefetch)
    executor.trigger_speculative_prefetch = trigger_spy

    _dispatch(executor, router_logits=LOGITS)

    prefetcher.fetch_experts_lock_cache.assert_not_called()
    assert prefetcher.speculative_prefetch.call_count == 1
    args, kwargs = prefetcher.speculative_prefetch.call_args
    assert args[0] == LAYER_ID and args[1] is LOGITS and not kwargs
    trigger_spy.assert_called_once()
    assert executor._pending_prefetch == (
        prefetcher,
        LAYER_ID,
        UNION,
        None,
        [],
        None,
    )


def test_inactive_context_deferred_legacy_path_byte_identical():
    prefetcher = MagicMock(name="ExpertPrefetcher")
    executor = _make_executor(overlap=False, prefetcher=prefetcher)

    _dispatch(executor, router_logits=LOGITS)

    prefetcher.fetch_experts_lock_cache.assert_not_called()
    prefetcher.speculative_prefetch.assert_not_called()
    pending = executor._pending_prefetch
    assert pending is not None and pending[3] is LOGITS

    executor.wait_dispatch_local()
    prefetcher.correct_prefetch.assert_called_once_with(LAYER_ID + 1, UNION)
    assert prefetcher.speculative_prefetch.call_count == 1
    args, kwargs = prefetcher.speculative_prefetch.call_args
    assert args[0] == LAYER_ID and args[1] is LOGITS and not kwargs


def test_inactive_context_without_router_logits_makes_no_prefetch_calls():
    prefetcher = MagicMock(name="ExpertPrefetcher")
    executor = _make_executor(overlap=False, prefetcher=prefetcher)

    _dispatch(executor)
    executor.wait_dispatch_local()

    prefetcher.fetch_experts_lock_cache.assert_not_called()
    prefetcher.speculative_prefetch.assert_not_called()
    # Pre-A3 behavior: correction still fires from the pending tuple.
    prefetcher.correct_prefetch.assert_called_once_with(LAYER_ID + 1, UNION)


# ---------------------------------------------------------------------------
# (c) context manager semantics: default off, exception-safe, scoped
# ---------------------------------------------------------------------------


def test_context_default_inactive_and_set_clear():
    handle = object()
    assert not is_active() and current_prefetcher() is None
    with route_ahead_context(prefetcher=handle):
        assert is_active() and current_prefetcher() is handle
    assert not is_active() and current_prefetcher() is None


def test_context_resets_after_exception():
    with pytest.raises(RuntimeError, match="verify boom"):
        with route_ahead_context(prefetcher=object()):
            assert is_active()
            raise RuntimeError("verify boom")
    assert not is_active() and current_prefetcher() is None


def test_context_nested_restores_outer_handle():
    outer, inner = object(), object()
    with route_ahead_context(prefetcher=outer):
        with route_ahead_context(prefetcher=inner):
            assert current_prefetcher() is inner
        assert is_active() and current_prefetcher() is outer
    assert not is_active() and current_prefetcher() is None


def test_context_does_not_leak_into_new_thread():
    observed = []

    def worker():
        observed.append(is_active())

    with route_ahead_context():
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert observed == [False]


# ---------------------------------------------------------------------------
# (d) speculator seam: verify forward runs under the context
# ---------------------------------------------------------------------------


def _make_speculator(moe) -> DFlashSpeculator:
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.moe = moe
    return speculator


def test_verify_target_block_activates_context_with_engine_prefetcher():
    handle = object()
    speculator = _make_speculator(
        SimpleNamespace(engine=SimpleNamespace(expert_prefetcher=handle))
    )
    observed = []
    logits = torch.zeros(1, 2, 5)

    def fake_forward(
        input_ids: torch.Tensor,
        past_key_values: object = None,
        logits_to_keep: int = 0,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object, object]:
        observed.append((is_active(), current_prefetcher(), logits_to_keep))
        return logits, "hidden", past_key_values

    speculator._forward_target = fake_forward
    block = torch.zeros(1, 2, dtype=torch.long)
    out = speculator._verify_target_block(block, "kv")
    assert observed == [(True, handle, 0)]
    assert out[0] is logits and out[1:] == ("hidden", "kv")
    assert not is_active() and current_prefetcher() is None


def test_verify_target_block_resets_context_on_forward_error():
    speculator = _make_speculator(SimpleNamespace())

    def boom(
        input_ids: torch.Tensor,
        past_key_values: object = None,
        logits_to_keep: int = 0,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object, object]:
        assert is_active()
        raise RuntimeError("verify boom")

    speculator._forward_target = boom
    with pytest.raises(RuntimeError, match="verify boom"):
        speculator._verify_target_block(
            torch.zeros(1, 2, dtype=torch.long), "kv"
        )
    assert not is_active() and current_prefetcher() is None


def test_resolve_route_ahead_prefetcher_without_engine_is_none():
    speculator = _make_speculator(SimpleNamespace())
    assert speculator._resolve_route_ahead_prefetcher() is None


# ---------------------------------------------------------------------------
# (e) Track A4 offload coupling: single-layer pin guard + gpt-oss exclusion
# ---------------------------------------------------------------------------


def test_consecutive_dispatches_each_pin_exactly_one_layer():
    """A4 guard: pins must never batch layers -- ``ReplaceCacheCandidates``
    is global and clears the background prefetch queues, so a cross-layer pin
    would evict candidates the next layer's dispatch still needs. Two
    dispatches in one verify context -> two single-layer pins, in order."""
    prefetcher, engine = _make_real_prefetcher()
    executor = _make_executor(prefetcher=prefetcher)

    with route_ahead_context():
        _dispatch(executor)
        executor.wait_dispatch_local()
        executor.dispatch_local(LAYER_ID + 1, HIDDEN, ROUTER_MASK, WEIGHTS)
        executor.wait_dispatch_local()

    pin_calls = engine.replace_cache_candidates.call_args_list
    assert len(pin_calls) == 2
    assert pin_calls[0].args[0] == [300, 301, 302, 305, 307]
    assert pin_calls[1].args[0] == [400, 401, 402, 405, 407]
    # No call ever mixes tensor ids from two layers (id = layer * 100 + e).
    for call in pin_calls:
        assert len({tensor_id // 100 for tensor_id in call.args[0]}) == 1
    # Issuances stay per-layer single-layered as well (one batched call/layer).
    assert _issued_tensor_ids(engine) == [
        300,
        301,
        302,
        305,
        307,
        400,
        401,
        402,
        405,
        407,
    ]


def _make_gpt_oss_mlp():
    from moe_infinity.models.gpt_oss import SyncGptOssMLP

    torch.manual_seed(0)
    module = SyncGptOssMLP(
        SimpleNamespace(
            hidden_size=16,
            intermediate_size=8,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
    )
    for param in module.parameters():
        torch.nn.init.normal_(param, std=0.02)
    module.eval()
    module.layer_id = LAYER_ID
    assert module.expert_executor is None
    return module


def test_gpt_oss_resident_loop_observes_route_ahead_metrics():
    """A2: resident gpt-oss reports its exact routed union as already covered."""
    from moe_infinity.spec_decode._route_ahead_stats import RouteAheadStats

    module = _make_gpt_oss_mlp()

    prefetcher = MagicMock(name="ExpertPrefetcher")
    stats = RouteAheadStats()
    stats.begin_step()
    with torch.no_grad():
        with route_ahead_context(prefetcher=prefetcher, stats=stats):
            final_hidden, router_logits = module(torch.randn(1, 3, 16))

    assert final_hidden.shape == (1, 3, 16)
    assert router_logits.shape == (3, 4)
    prefetcher.fetch_experts_lock_cache.assert_not_called()
    prefetcher.speculative_prefetch.assert_not_called()
    summary = stats.commit_step(kept_rows=1)
    assert summary.layers == 1
    assert stats.steps > 0
    assert stats.layers_observed > 0
    assert stats.coverage == 1.0


def test_gpt_oss_resident_loop_does_not_observe_when_context_inactive(
    monkeypatch: pytest.MonkeyPatch,
):
    """Spec-off decode must return before resolving or updating metrics."""
    from moe_infinity.spec_decode import _route_ahead_ctx as route_ahead_ctx
    from moe_infinity.spec_decode._route_ahead_stats import RouteAheadStats

    module = _make_gpt_oss_mlp()
    stats = RouteAheadStats()
    stats.begin_step()
    current_stats = MagicMock(return_value=stats)
    monkeypatch.setattr(route_ahead_ctx, "current_stats", current_stats)

    with torch.no_grad():
        module(torch.randn(1, 3, 16))

    current_stats.assert_not_called()
    assert stats.commit_step(kept_rows=1).layers == 0
    assert stats.as_dict() == RouteAheadStats().as_dict()


# ---------------------------------------------------------------------------
# overlap-aware dispatch integration (plan Task 6)
# ---------------------------------------------------------------------------


def test_partial_exact_route_prefetch_still_dispatches_full_native_union():
    prefetcher = MagicMock()
    prefetcher.plan_candidates.return_value = (41, [0, 2])
    executor = _make_executor(prefetcher=prefetcher, policy="enforce")
    with route_ahead_context(prefetcher=prefetcher):
        _dispatch(executor, router_logits=LOGITS)
    prefetcher.fetch_experts_lock_cache.assert_called_once_with(
        LAYER_ID, [0, 2]
    )
    assert _enqueued_experts(executor) == UNION


def test_empty_budget_does_not_replace_pin_or_suppress_dispatch():
    prefetcher = MagicMock()
    prefetcher.plan_candidates.return_value = (42, [])
    executor = _make_executor(prefetcher=prefetcher, policy="enforce")
    with route_ahead_context(prefetcher=prefetcher):
        _dispatch(executor)
    prefetcher.fetch_experts_lock_cache.assert_not_called()
    assert _enqueued_experts(executor) == UNION


def test_exact_route_correction_precedes_every_expert_enqueue():
    events = []
    prefetcher = MagicMock()
    prefetcher._overlap_active.return_value = True
    prefetcher.correct_to_native_route.side_effect = (
        lambda layer, ids: events.append("correct") or ids
    )
    executor = _make_executor(prefetcher=prefetcher, policy="enforce")
    executor.expert_dispatcher.enqueue_expert.side_effect = (
        lambda *a: events.append("enqueue")
    )
    _dispatch(executor)
    assert events[0] == "correct"


def test_wait_error_cancels_owned_generations_drains_and_reraises():
    prefetcher = MagicMock()
    executor = _make_executor(prefetcher=prefetcher)
    executor._pending_prefetch = (
        prefetcher,
        LAYER_ID,
        UNION,
        None,
        [41, 42],
        None,
    )
    executor.expert_dispatcher.wait_expert.side_effect = RuntimeError(
        "wait failed"
    )
    with pytest.raises(RuntimeError, match="wait failed"):
        executor.wait_dispatch_local()
    prefetcher.abort_prefetch_generations.assert_called_once_with(
        [41, 42], reason="wait_expert_error"
    )
    prefetcher.drain_native_prefetch_samples.assert_called()
    assert executor._pending_prefetch is None


def test_cleanup_error_does_not_mask_wait_error():
    prefetcher = MagicMock()
    prefetcher.abort_prefetch_generations.side_effect = RuntimeError(
        "cleanup failed"
    )
    executor = _make_executor(prefetcher=prefetcher)
    executor._pending_prefetch = (
        prefetcher,
        LAYER_ID,
        UNION,
        None,
        [41],
        None,
    )
    executor.expert_dispatcher.wait_expert.side_effect = RuntimeError(
        "wait failed"
    )
    with pytest.raises(RuntimeError, match="wait failed"):
        executor.wait_dispatch_local()
    prefetcher.drain_native_prefetch_samples.assert_called()


def test_delayed_compute_sample_from_prior_invocation_is_not_calibration():
    prefetcher = MagicMock()
    executor = _make_executor(prefetcher=prefetcher, policy="observe")
    executor._pending_prefetch = (
        prefetcher,
        LAYER_ID,
        UNION,
        None,
        [],
        102,
    )
    executor.expert_dispatcher.drain_compute_samples.return_value = [
        _compute_sample(invocation_id=101, layer_id=LAYER_ID),
        _compute_sample(invocation_id=102, layer_id=LAYER_ID),
    ]
    executor.wait_dispatch_local()
    prefetcher.observe_compute_samples.assert_called_once()
    observed = prefetcher.observe_compute_samples.call_args.args[0]
    assert [s.invocation_id for s in observed] == [102]
    prefetcher.record_stale_compute_samples.assert_called_once_with(1)


def test_same_layer_sample_without_matching_invocation_is_rejected():
    prefetcher = MagicMock()
    executor = _make_executor(prefetcher=prefetcher, policy="enforce")
    executor._pending_prefetch = (
        prefetcher,
        LAYER_ID,
        UNION,
        None,
        [],
        202,
    )
    executor.expert_dispatcher.drain_compute_samples.return_value = [
        _compute_sample(invocation_id=201, layer_id=LAYER_ID),
    ]
    executor.wait_dispatch_local()
    prefetcher.observe_compute_samples.assert_not_called()
