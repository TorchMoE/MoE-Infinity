import importlib
from collections.abc import Callable
from typing import Any

import pytest

from moe_infinity.distributed.expert_executor import DistributedExpertExecutor
from moe_infinity.utils import ArcherConfig
from tests.python.ops.conftest import BF16_ATOL, BF16_RTOL, requires_cuda

torch = pytest.importorskip("torch")


class _FakeExpertDispatcher:
    def __init__(self, expert_fns: dict[int, Callable[[Any], Any]]):
        self.expert_fns = expert_fns
        self.hidden_states = None
        self.router_mask = None
        self.router_weights = None
        self.expected_wait_cnt = 0
        self.enqueued_expert_ids: list[int] = []
        self.enqueued_phases: list[int] = []
        self.notify_fetch_start_called = False
        self.token_contribution_counts = None

    def set_inputs(self, hidden_states, router_mask, router_weights):
        self.hidden_states = hidden_states
        self.router_mask = router_mask
        self.router_weights = router_weights

    def set_expected_queue(self, expected_wait_cnt):
        self.expected_wait_cnt = int(expected_wait_cnt)

    def enqueue_expert(self, layer_id, expert_id, gpu_id, remote, phase=2):
        self.enqueued_expert_ids.append(int(expert_id))
        self.enqueued_phases.append(int(phase))

    def notify_fetch_start(self):
        self.notify_fetch_start_called = True

    def wait_expert(self):
        assert self.hidden_states is not None
        assert self.router_mask is not None
        assert self.router_weights is not None
        assert self.notify_fetch_start_called

        output = torch.zeros_like(self.hidden_states, dtype=torch.float32)
        contribution_counts = torch.zeros(
            self.hidden_states.size(0), dtype=torch.int32, device=output.device
        )

        for expert_id in self.enqueued_expert_ids:
            token_mask = self.router_mask[:, expert_id].bool()
            if not token_mask.any():
                continue
            expert_out = self.expert_fns[expert_id](
                self.hidden_states[token_mask]
            ).to(torch.float32)
            weights = (
                self.router_weights[token_mask, expert_id]
                .unsqueeze(1)
                .to(torch.float32)
            )
            output[token_mask] += expert_out * weights
            contribution_counts[token_mask] += 1

        self.token_contribution_counts = contribution_counts
        return output


def _reference_weighted_sum(
    hidden_states,
    router_mask,
    router_weights,
    expert_fns: dict[int, Callable[[Any], Any]],
):
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    contribution_counts = torch.zeros(
        hidden_states.size(0), dtype=torch.int32, device=hidden_states.device
    )

    for expert_id, expert_fn in expert_fns.items():
        token_mask = router_mask[:, expert_id].bool()
        if not token_mask.any():
            continue
        output[token_mask] += expert_fn(hidden_states[token_mask]).to(
            torch.float32
        ) * router_weights[token_mask, expert_id].unsqueeze(1).to(torch.float32)
        contribution_counts[token_mask] += 1

    return output, contribution_counts


@requires_cuda
def test_expert_executor_import_contract_or_documented_skip():
    module = importlib.import_module("moe_infinity.distributed.expert_executor")
    assert hasattr(module, "DistributedExpertExecutor")

    if not hasattr(module, "ExpertExecutor"):
        pytest.skip(
            "`ExpertExecutor` alias is not exported in this revision; "
            + "production path uses `DistributedExpertExecutor`."
        )

    expert_executor_cls = getattr(module, "ExpertExecutor")
    assert expert_executor_cls is not None


@requires_cuda
def test_dispatch_local_routing_weighted_accumulation_and_no_drops():
    hidden_states = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -0.5, 1.5],
            [2.0, 1.0, 0.0],
            [-1.0, 3.0, 0.25],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    router_mask = torch.tensor(
        [
            [True, True, False],
            [False, True, False],
            [True, False, True],
            [False, False, True],
        ],
        dtype=torch.bool,
        device="cuda",
    )
    router_weights = torch.tensor(
        [
            [0.30, 0.70, 0.00],
            [0.00, 1.00, 0.00],
            [0.40, 0.00, 0.60],
            [0.00, 0.00, 1.00],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    expert_fns: dict[int, Callable[[Any], Any]] = {
        0: lambda x: x * 2.0,
        1: lambda x: x + 1.0,
        2: lambda x: -x,
    }

    fake_dispatcher = _FakeExpertDispatcher(expert_fns)
    executor = DistributedExpertExecutor(
        archer_config=ArcherConfig(offload_path="")
    )
    executor.set_expert_dispatcher(fake_dispatcher)

    executor.dispatch_local(
        layer_id=0,
        hidden_states=hidden_states,
        router_mask=router_mask,
        router_weights=router_weights,
    )
    dispatched_output = executor.wait_dispatch_local()

    expected_output, expected_counts = _reference_weighted_sum(
        hidden_states=hidden_states,
        router_mask=router_mask,
        router_weights=router_weights,
        expert_fns=expert_fns,
    )

    active_experts = torch.where(router_mask.any(dim=0))[0].tolist()

    assert fake_dispatcher.notify_fetch_start_called
    assert fake_dispatcher.expected_wait_cnt == len(active_experts)
    assert fake_dispatcher.enqueued_expert_ids == active_experts
    assert torch.equal(
        fake_dispatcher.token_contribution_counts, expected_counts
    )
    assert torch.all(expected_counts == router_mask.sum(dim=1).to(torch.int32))

    torch.testing.assert_close(
        dispatched_output,
        expected_output,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )


@requires_cuda
def test_dispatch_local_passes_current_phase_to_enqueue_expert():
    from moe_infinity.memory.expert_policy import (
        ExpertPhase,
        expert_phase_scope,
    )

    hidden_states = torch.tensor(
        [[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]],
        dtype=torch.float32,
        device="cuda",
    )
    router_mask = torch.tensor(
        [[True, False], [False, True]],
        dtype=torch.bool,
        device="cuda",
    )
    router_weights = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    expert_fns: dict[int, Callable[[Any], Any]] = {
        0: lambda x: x,
        1: lambda x: x,
    }
    fake_dispatcher = _FakeExpertDispatcher(expert_fns)
    executor = DistributedExpertExecutor(
        archer_config=ArcherConfig(offload_path="")
    )
    executor.set_expert_dispatcher(fake_dispatcher)

    with expert_phase_scope(ExpertPhase.DECODE):
        executor.dispatch_local(
            layer_id=0,
            hidden_states=hidden_states,
            router_mask=router_mask,
            router_weights=router_weights,
        )
        _ = executor.wait_dispatch_local()

    assert fake_dispatcher.enqueued_phases == [int(ExpertPhase.DECODE)] * len(
        fake_dispatcher.enqueued_expert_ids
    )
    assert fake_dispatcher.enqueued_expert_ids


@requires_cuda
def test_cpp_expert_dispatcher_full_async_flow_documented_skip():
    store = pytest.importorskip("moe_infinity._store")
    if not hasattr(store, "expert_dispatcher"):
        pytest.skip("moe_infinity._store has no `expert_dispatcher` binding")

    pytest.skip(
        "C++ expert_dispatcher constructor requires full engine initialization; "
        "calling it without proper setup causes a segfault. "
        "See tests/python/integration/test_model_consistency.py for full "
        "integration coverage."
    )
