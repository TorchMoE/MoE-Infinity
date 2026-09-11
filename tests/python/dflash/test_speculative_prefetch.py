"""Unit tests for the Track A2 explicit-set extension of ``speculative_prefetch``.

Autonomous correctness gate for Track A2 of
``.sisyphus/plans/dflash-deferred-tracks-plan.md``. Covers:

(a) characterization: the legacy positional call
    ``speculative_prefetch(layer_idx, router_logits)`` still pools via
    ``mean(0)`` and enqueues exactly ``topk(min(2, E))`` for ``layer_idx + 1``;
(b) the new explicit mode (``expert_ids=...``, ``prefetch_layer_id=...``)
    enqueues exactly the given set for the requested layer and never runs
    the mean/topk path -- this is the seam A3's verify loop will call with
    the model-exact routed union;
(c) empty ``expert_ids`` is a safe no-op;
(d) both arguments ``None`` raises ``ValueError``.

Construction is intentionally light: ``ExpertPrefetcher.__new__`` plus the
four attributes ``speculative_prefetch`` touches (``num_layers``,
``num_experts``, ``archer_engine``, ``expert_tensor_map``), with a mocked
``archer_engine`` -- no config parsing, no native extension, CPU-only.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from moe_infinity.memory.expert_prefetcher import (
    BACKGROUND_PREFETCH_PRIORITY,
    ON_DEMAND_PRIORITY,
    ROUTE_AHEAD_PRIORITY,
    ExpertPrefetcher,
)

# Hand-checked legacy fixture: [2 tokens, 4 experts]
#   mean over tokens = [2.0, 3.5, 3.0, 0.5] -> top-2 = experts [1, 2]
LOGITS = torch.tensor(
    [
        [1.0, 5.0, 2.0, 0.0],
        [3.0, 2.0, 4.0, 1.0],
    ]
)
LEGACY_TOP2 = [1, 2]


_LEGACY_ENGINE_SPEC = [
    "get_node_default_device",
    "enqueue_prefetch",
    "replace_cache_candidates",
    "prefetch_tensors",
    "is_tensor_on_device",
]

_NATIVE_ENGINE_SPEC = _LEGACY_ENGINE_SPEC + [
    "schedule_prefetch_tensors",
    "cancel_prefetch_generation",
    "drain_prefetch_samples",
    "get_inflight_prefetch_bytes",
]


def _make_prefetcher(
    num_layers: int = 8, num_experts: int = 8, extension: str = "native"
):
    prefetcher = ExpertPrefetcher.__new__(ExpertPrefetcher)
    prefetcher.num_layers = num_layers
    prefetcher.num_experts = num_experts
    spec = _LEGACY_ENGINE_SPEC if extension == "legacy" else _NATIVE_ENGINE_SPEC
    engine = MagicMock(spec=spec)
    engine.get_node_default_device.return_value = 0
    engine.prefetch_tensors.return_value = None
    engine.is_tensor_on_device.return_value = False
    if extension != "legacy":
        engine.get_inflight_prefetch_bytes.return_value = 0
        engine.drain_prefetch_samples.return_value = []
        admission = MagicMock()
        admission.accepted_tensor_ids = []
        engine.schedule_prefetch_tensors.return_value = admission
    prefetcher.archer_engine = engine
    dispatcher = MagicMock(
        spec=[
            "set_inputs_with_invocation",
            "set_overlap_compute_timing_enabled",
            "drain_compute_samples",
        ]
    )
    dispatcher.drain_compute_samples.return_value = []
    prefetcher.expert_dispatcher = dispatcher
    # Readable tensor ids: (layer, expert) -> layer * 100 + expert.
    prefetcher.expert_tensor_map = {
        (layer, expert): layer * 100 + expert
        for layer in range(num_layers)
        for expert in range(num_experts)
    }
    prefetcher.expert_nbytes_map = {}
    prefetcher.overlap_controller = None
    prefetcher._last_speculative_prediction = set()
    return prefetcher, engine


class _OverlapConfig:
    def __init__(self, policy: str):
        self.overlap_prefetch_policy = policy
        self.overlap_prefetch_ewma_alpha = 0.5
        self.overlap_prefetch_safety_factor = 0.8
        self.overlap_prefetch_cold_start_experts = 1
        self.overlap_prefetch_max_window_bytes = 10_000_000
        self.overlap_prefetch_max_inflight_bytes = 10_000_000


def _config(policy: str) -> _OverlapConfig:
    return _OverlapConfig(policy)


def _enqueued_tensor_ids(engine: MagicMock) -> list[int]:
    # Mechanism-agnostic: a batched ``prefetch_tensors([...])`` issuance carries
    # the same ordered ids as the per-expert ``enqueue_prefetch`` fallback, so
    # flatten the batched calls when present and fall back otherwise.
    scheduled = getattr(engine, "schedule_prefetch_tensors", None)
    scheduled_calls = getattr(scheduled, "call_args_list", None)
    if scheduled_calls:
        issued: list[int] = []
        for call in scheduled_calls:
            issued.extend(
                call.args[0] if call.args else call.kwargs["tensor_ids"]
            )
        return issued
    batched = getattr(engine, "prefetch_tensors", None)
    batched_calls = getattr(batched, "call_args_list", None)
    if batched_calls:
        issued = []
        for call in batched_calls:
            issued.extend(call.args[0])
        return issued
    return [call.args[0] for call in engine.enqueue_prefetch.call_args_list]


# ---------------------------------------------------------------------------
# (a) legacy characterization -- old call style must behave exactly as before
# ---------------------------------------------------------------------------


def test_legacy_positional_call_enqueues_mean_topk_for_next_layer():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.speculative_prefetch(2, LOGITS)
    assert _enqueued_tensor_ids(engine) == [301, 302]
    assert prefetcher._last_speculative_prediction == set(LEGACY_TOP2)


def test_legacy_numpy_logits_enqueue_same_experts_as_torch_path():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.speculative_prefetch(2, LOGITS.numpy())
    assert _enqueued_tensor_ids(engine) == [301, 302]
    assert prefetcher._last_speculative_prediction == set(LEGACY_TOP2)


def test_legacy_last_layer_is_noop():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.speculative_prefetch(7, LOGITS)
    engine.enqueue_prefetch.assert_not_called()


def test_legacy_router_logits_still_accepted_as_keyword():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.speculative_prefetch(2, router_logits=LOGITS)
    assert _enqueued_tensor_ids(engine) == [301, 302]


# ---------------------------------------------------------------------------
# (b) explicit route-ahead mode -- exact set, exact layer, no mean/topk
# ---------------------------------------------------------------------------


def test_explicit_expert_ids_enqueue_exact_set_for_target_layer(
    monkeypatch: pytest.MonkeyPatch,
):
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=8)
    topk_spy = MagicMock(wraps=torch.topk)
    monkeypatch.setattr(torch, "topk", topk_spy)
    # Logits that would route to expert 0 if the legacy topk path ran.
    logits = torch.full((3, 8), -100.0)
    logits[:, 0] = 100.0
    prefetcher.speculative_prefetch(
        1, logits, expert_ids=[3, 1, 7], prefetch_layer_id=5
    )
    topk_spy.assert_not_called()
    assert _enqueued_tensor_ids(engine) == [503, 501, 507]
    assert prefetcher._last_speculative_prediction == {3, 1, 7}


def test_explicit_defaults_to_next_layer():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=8)
    prefetcher.speculative_prefetch(2, expert_ids=[0, 2])
    assert _enqueued_tensor_ids(engine) == [300, 302]


def test_explicit_out_of_range_target_layer_is_noop():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=8)
    # Default target = 7 + 1 = 8 >= num_layers.
    prefetcher.speculative_prefetch(7, expert_ids=[1, 2])
    engine.enqueue_prefetch.assert_not_called()


# ---------------------------------------------------------------------------
# (c) empty explicit set is a safe no-op
# ---------------------------------------------------------------------------


def test_explicit_empty_list_is_noop():
    prefetcher, engine = _make_prefetcher()
    prefetcher.speculative_prefetch(0, expert_ids=[])
    engine.enqueue_prefetch.assert_not_called()
    engine.get_node_default_device.assert_not_called()
    assert prefetcher._last_speculative_prediction == set()


# ---------------------------------------------------------------------------
# (d) neither argument -> ValueError
# ---------------------------------------------------------------------------


def test_both_none_raises_value_error():
    prefetcher, _engine = _make_prefetcher()
    with pytest.raises(ValueError, match="router_logits"):
        prefetcher.speculative_prefetch(0)


def _make_prefetcher_without_batch(num_layers: int = 8, num_experts: int = 8):
    prefetcher = ExpertPrefetcher.__new__(ExpertPrefetcher)
    prefetcher.num_layers = num_layers
    prefetcher.num_experts = num_experts
    engine = MagicMock(
        spec=[
            "get_node_default_device",
            "enqueue_prefetch",
            "replace_cache_candidates",
        ]
    )
    engine.get_node_default_device.return_value = 0
    prefetcher.archer_engine = engine
    prefetcher.expert_tensor_map = {
        (layer, expert): layer * 100 + expert
        for layer in range(num_layers)
        for expert in range(num_experts)
    }
    prefetcher._last_speculative_prediction = set()
    return prefetcher, engine


def test_prefetch_experts_list_batches_one_native_call_when_available():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=8)
    prefetcher.prefetch_experts_list(3, [3, 1, 7])
    engine.prefetch_tensors.assert_called_once_with(
        [303, 301, 307], priority=ROUTE_AHEAD_PRIORITY
    )
    engine.enqueue_prefetch.assert_not_called()


def test_prefetch_experts_list_falls_back_to_per_expert_without_batch_api():
    prefetcher, engine = _make_prefetcher_without_batch(
        num_layers=8, num_experts=8
    )
    prefetcher.prefetch_experts_list(3, [3, 1, 7])
    assert _enqueued_tensor_ids(engine) == [303, 301, 307]


def test_prefetch_experts_list_empty_batch_calls_neither_path():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=8)
    prefetcher.prefetch_experts_list(3, [])
    engine.prefetch_tensors.assert_not_called()
    engine.enqueue_prefetch.assert_not_called()


# ---------------------------------------------------------------------------
# priority bands (plan Task 9 Step 5): explicit route-ahead vs legacy background
# ---------------------------------------------------------------------------


def _issued_priority(engine: MagicMock) -> int:
    return engine.prefetch_tensors.call_args.kwargs["priority"]


def test_explicit_route_ahead_issues_on_the_route_ahead_band():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=8)
    prefetcher.speculative_prefetch(
        1, expert_ids=[3, 1, 7], prefetch_layer_id=5
    )
    assert _enqueued_tensor_ids(engine) == [503, 501, 507]
    assert _issued_priority(engine) == ROUTE_AHEAD_PRIORITY


def test_legacy_router_logits_issues_on_the_background_band():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.speculative_prefetch(2, LOGITS)
    assert _enqueued_tensor_ids(engine) == [301, 302]
    assert _issued_priority(engine) == BACKGROUND_PREFETCH_PRIORITY


def test_correct_prefetch_issues_on_the_route_ahead_band():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=8)
    prefetcher._last_speculative_prediction = {1}
    prefetcher.correct_prefetch(3, [3, 1, 7])
    assert _enqueued_tensor_ids(engine) == [303, 307]
    assert _issued_priority(engine) == ROUTE_AHEAD_PRIORITY


def test_route_ahead_priority_knob_sweeps_the_issued_band():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=8)
    prefetcher.route_ahead_priority = ON_DEMAND_PRIORITY
    prefetcher.prefetch_experts_list(3, [3, 1, 7])
    assert _issued_priority(engine) == ON_DEMAND_PRIORITY


# ---------------------------------------------------------------------------
# overlap-aware byte admission (plan Task 5)
# ---------------------------------------------------------------------------


class StrictLegacyExtension:
    def __init__(self, forbidden: set[str]):
        self._forbidden = forbidden
        self.forbidden_accesses: list[str] = []
        self.prefetch_tensors = MagicMock()

    def __getattr__(self, name: str):
        forbidden = object.__getattribute__(self, "_forbidden")
        if name in forbidden:
            object.__getattribute__(self, "forbidden_accesses").append(name)
            raise AssertionError(
                f"new binding accessed while unavailable: {name}"
            )
        raise AttributeError(name)


def test_enforce_issues_only_whole_experts_inside_budget():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.configure_overlap_policy(_config(policy="enforce"))
    prefetcher.expert_nbytes_map = {
        (3, 0): 4000,
        (3, 1): 3000,
        (3, 2): 3000,
        (3, 3): 1000,
    }
    prefetcher.overlap_controller.observe_compute(3, 0, 1000)
    prefetcher.overlap_controller.observe_transfer(1000, 100, 0, 0)
    accepted = MagicMock()
    accepted.accepted_tensor_ids = [301, 302]
    engine.schedule_prefetch_tensors.return_value = accepted
    prefetcher.speculative_prefetch(2, LOGITS)
    assert _enqueued_tensor_ids(engine) == [301, 302]
    assert engine.schedule_prefetch_tensors.call_args.kwargs["layer_id"] == 3


def test_enforce_missing_byte_map_fails_closed():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.configure_overlap_policy(_config(policy="enforce"))
    prefetcher.expert_nbytes_map = {}
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.schedule_prefetch_tensors.assert_not_called()
    engine.prefetch_tensors.assert_not_called()


def test_observe_records_decision_but_preserves_legacy_issue():
    prefetcher, engine = _make_prefetcher(num_layers=8, num_experts=4)
    prefetcher.configure_overlap_policy(_config(policy="observe"))
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.prefetch_tensors.assert_called_once_with([301, 302], priority=2)
    assert prefetcher.overlap_prefetch_stats()["decisions"] == 1


def test_off_never_probes_calls_or_drains_new_bindings():
    prefetcher, _ = _make_prefetcher(num_layers=8, num_experts=4)
    engine = StrictLegacyExtension(
        {
            "schedule_prefetch_tensors",
            "cancel_prefetch_generation",
            "drain_prefetch_samples",
            "get_inflight_prefetch_bytes",
        }
    )
    dispatcher = StrictLegacyExtension(
        forbidden={
            "set_inputs_with_invocation",
            "set_overlap_compute_timing_enabled",
            "drain_compute_samples",
        }
    )
    prefetcher.archer_engine = engine
    prefetcher.expert_dispatcher = dispatcher
    prefetcher.configure_overlap_policy(_config(policy="off"))
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.prefetch_tensors.assert_called_once()
    assert prefetcher.overlap_controller is None
    assert dispatcher.forbidden_accesses == []
    assert engine.forbidden_accesses == []


def test_observe_with_legacy_extension_records_capability_miss_and_issues_legacy():
    prefetcher, engine = _make_prefetcher(
        num_layers=8, num_experts=4, extension="legacy"
    )
    prefetcher.configure_overlap_policy(_config(policy="observe"))
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.prefetch_tensors.assert_called_once()
    assert prefetcher.overlap_prefetch_stats()["native_capability_misses"] == 1


def test_enforce_with_legacy_extension_fails_closed_without_new_or_legacy_issue():
    prefetcher, engine = _make_prefetcher(
        num_layers=8, num_experts=4, extension="legacy"
    )
    prefetcher.configure_overlap_policy(_config(policy="enforce"))
    prefetcher.speculative_prefetch(2, LOGITS)
    engine.prefetch_tensors.assert_not_called()
    assert prefetcher.overlap_prefetch_stats()["native_capability_misses"] == 1
