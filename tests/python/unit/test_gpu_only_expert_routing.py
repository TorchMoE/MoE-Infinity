from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from moe_infinity.distributed.expert_executor import DistributedExpertExecutor


class FakeDispatcher:
    def __init__(self):
        self.calls = []
        self.active = [0, 2]

    def set_inputs(self, hidden, mask, weights):
        self.calls.append(("set_inputs", hidden, mask, weights))

    def set_expected_queue(self, count):
        self.calls.append(("set_expected_queue", count))

    def enqueue_expert(self, layer, expert, gpu, remote):
        self.calls.append(("enqueue_expert", layer, expert, gpu, remote))

    def notify_fetch_start(self):
        self.calls.append(("notify_fetch_start",))

    def dispatch_experts(self, layer):
        self.calls.append(("dispatch_experts", layer))

    def wait_expert(self):
        set_inputs_call = next(
            (call for call in self.calls if call[0] == "set_inputs"), None
        )
        self.calls.append(("wait_expert",))
        hidden = (
            set_inputs_call[1]
            if set_inputs_call is not None
            else torch.zeros(1, 1)
        )
        return torch.zeros_like(hidden, dtype=torch.float32)

    def take_last_active_experts(self):
        self.calls.append(("take_last_active_experts",))
        return list(self.active)

    def get_routing_stats(self):
        return {
            "route_batches": 1,
            "route_failures": 0,
            "last_active_experts": len(self.active),
            "last_route_handoff_us": 1,
            "completion_events_retired": 2,
        }


class FakePrefetcher:
    def __init__(self):
        self.corrected = []
        self.speculative = []

    def correct_prefetch(self, layer, experts):
        self.corrected.append((layer, experts))

    def speculative_prefetch(self, layer, router_logits):
        self.speculative.append((layer, router_logits))


def make_executor(enabled=True):
    config = SimpleNamespace(
        gpu_only_expert_routing=enabled,
        speculative_prefetch_overlap=False,
    )
    executor = DistributedExpertExecutor(config)
    dispatcher = FakeDispatcher()
    executor.set_expert_dispatcher(dispatcher)
    return executor, dispatcher


def test_cpu_mask_uses_eager_fallback_in_ascending_order():
    executor, dispatcher = make_executor(enabled=True)
    hidden = torch.ones(2, 4)
    mask = torch.tensor([[True, False, True], [False, False, True]])
    weights = mask.float()

    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(3, hidden, mask, weights)

    assert [call[0] for call in dispatcher.calls] == [
        "set_inputs",
        "set_expected_queue",
        "enqueue_expert",
        "enqueue_expert",
        "notify_fetch_start",
    ]
    enqueued = [
        call[2] for call in dispatcher.calls if call[0] == "enqueue_expert"
    ]
    assert enqueued == [0, 2]


def test_disabled_flag_uses_eager_fallback():
    executor, dispatcher = make_executor(enabled=False)
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[False, True]])
    with patch("torch.cuda.device_count", return_value=1):
        executor.dispatch_local(0, hidden, mask, mask.float())
    assert not any(call[0] == "dispatch_experts" for call in dispatcher.calls)


def test_native_active_list_drives_existing_prefetch_correction():
    executor, dispatcher = make_executor(enabled=True)
    prefetcher = FakePrefetcher()
    executor._last_dispatch_used_native_routing = True
    executor._pending_prefetch = (prefetcher, 7, None, None)

    result = executor.wait_dispatch_local()

    assert result.dtype == torch.float32
    assert prefetcher.corrected == [(8, [0, 2])]
    assert [call[0] for call in dispatcher.calls] == [
        "wait_expert",
        "take_last_active_experts",
    ]


def test_missing_native_binding_uses_eager_fallback():
    executor, dispatcher = make_executor(enabled=True)
    delattr(FakeDispatcher, "dispatch_experts")
    try:
        hidden = torch.ones(1, 2)
        mask = torch.tensor([[True, False]])
        with patch("torch.cuda.device_count", return_value=1):
            executor.dispatch_local(0, hidden, mask, mask.float())
        assert any(call[0] == "enqueue_expert" for call in dispatcher.calls)
    finally:
        FakeDispatcher.dispatch_experts = lambda self, layer: self.calls.append(
            ("dispatch_experts", layer)
        )


def test_synchronous_native_submission_error_is_not_replayed_eagerly():
    executor, dispatcher = make_executor(enabled=True)
    executor._can_use_gpu_only_routing = lambda mask: True
    dispatcher.dispatch_experts = lambda layer: (_ for _ in ()).throw(
        RuntimeError("DispatchExperts: invalid mask")
    )
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[True, False]])
    with (
        patch("torch.cuda.device_count", return_value=1),
        pytest.raises(RuntimeError, match="DispatchExperts: invalid mask"),
    ):
        executor.dispatch_local(0, hidden, mask, mask.float())
    assert not any(call[0] == "enqueue_expert" for call in dispatcher.calls)


def test_representative_adapters_keep_dispatch_wait_contract():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    for relative in (
        "moe_infinity/models/qwen.py",
        "moe_infinity/models/mixtral.py",
        "moe_infinity/models/deepseek.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        dispatch_at = source.index("dispatch_local(")
        wait_at = source.index("wait_dispatch_local()", dispatch_at)
        assert dispatch_at < wait_at
        # deepseek names the argument ``routing_mask`` while qwen/mixtral use
        # ``router_mask``; both carry the routing mask through the boundary.
        assert "mask" in source[dispatch_at:wait_at]


def test_gpu_routing_exposes_host_list_only_after_wait():
    executor, dispatcher = make_executor(enabled=True)
    executor._can_use_gpu_only_routing = lambda mask: True
    prefetcher = FakePrefetcher()
    executor.set_prefetcher(prefetcher)
    hidden = torch.ones(1, 2)
    mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[3.0, 1.0, 2.0]])

    executor.dispatch_local(4, hidden, mask, mask.float(), router_logits=logits)

    assert prefetcher.speculative == []
    assert not any(
        call[0] == "take_last_active_experts" for call in dispatcher.calls
    )
    _ = executor.wait_dispatch_local()
    assert prefetcher.corrected == [(5, [0, 2])]
    assert prefetcher.speculative == [(4, logits)]
    assert dispatcher.calls[-1][0] == "take_last_active_experts"
