# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch

from moe_infinity import _store


def _wait_for_lease_drain(dispatcher, timeout_s: float = 5.0):
    import time

    deadline = time.monotonic() + timeout_s
    metrics = dispatcher.get_precision_metrics()
    while metrics["active_leases"] != 0 and time.monotonic() < deadline:
        time.sleep(0.01)
        metrics = dispatcher.get_precision_metrics()
    return metrics


@dataclass
class NativeAdaptiveFixture:
    dispatcher: object
    prefetcher: object
    hidden: torch.Tensor

    def run(self):
        mask = torch.ones((1, 1), dtype=torch.bool, device="cuda:0")
        weights = torch.ones((1, 1), dtype=torch.bfloat16, device="cuda:0")
        self.dispatcher.set_inputs(self.hidden, mask, weights)
        self.dispatcher.set_expected_queue(1)
        self.dispatcher.enqueue_expert(0, 0, 0, False)
        self.dispatcher.notify_fetch_start()
        result = self.dispatcher.wait_expert()
        _wait_for_lease_drain(self.dispatcher)
        return result


@pytest.fixture(scope="module")
def native_adaptive_fixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("adaptive_dispatch")
    handle = _store.prefetch_handle(str(tmp_path), 0.5)
    tensors = [torch.randn(128, 128, dtype=torch.bfloat16) for _ in range(3)]
    for tensor_id, tensor in enumerate(tensors):
        handle.offload(tensor, tensor_id)
    dense_ids = list(range(3, 15))
    for tensor_id in dense_ids:
        handle.offload(torch.zeros(1, dtype=torch.bfloat16), tensor_id)
    topology = [
        (f"dense.before.{i}", [[tensor_id]])
        for i, tensor_id in enumerate(dense_ids[:6])
    ]
    topology += [("model.layers.0.mlp.experts", [[0, 1, 2]])]
    topology += [
        (f"dense.after.{i}", [[tensor_id]])
        for i, tensor_id in enumerate(dense_ids[6:])
    ]
    handle.set_topology(topology)
    dispatcher = _store.expert_dispatcher(1, 1, 0, 5, 1)
    dispatcher.register_expert(0, 0, [0, 1, 2], "")
    handle.configure_residency_manager(True, False)
    dispatcher.configure_residency_manager(True, False)
    assert (
        handle.get_residency_manager_id()
        == dispatcher.get_residency_manager_id()
    )
    assert dispatcher.set_adaptive_hbm_budget_bytes(1 << 20)
    payload = sum(tensor.nbytes for tensor in tensors)
    aligned = sum((tensor.nbytes + 4095) // 4096 * 4096 for tensor in tensors)
    assert dispatcher.register_expert_variant(
        0,
        0,
        "bf16",
        1,
        "bf16_gemm",
        [0, 1, 2],
        ["gate.weight", "up.weight", "down.weight"],
        payload,
        aligned,
        0,
    )
    fixture = NativeAdaptiveFixture(
        dispatcher,
        handle,
        torch.randn(1, 128, dtype=torch.bfloat16, device="cuda:0"),
    )
    yield fixture
    del dispatcher
    handle.clean_up_resources()


@pytest.mark.gpu
def test_dispatcher_publishes_adaptive_control_surface():
    required = {
        "register_expert_variant",
        "set_precision_targets",
        "set_adaptive_hbm_budget_bytes",
        "get_precision_metrics",
        "get_residency_manager_id",
        "configure_residency_manager",
    }
    assert required <= set(dir(_store.expert_dispatcher))


@pytest.mark.gpu
def test_failed_transition_binding_is_testing_only():
    assert hasattr(
        _store.expert_dispatcher, "inject_transition_failure_once_for_test"
    )


def test_precision_metrics_schema_is_complete(native_adaptive_fixture):
    required = {
        "budget_bytes",
        "resident_bytes",
        "resident_payload_bytes",
        "alignment_padding_bytes",
        "transition_reserved_bytes",
        "workspace_bytes",
        "peak_accounted_bytes",
        "h2d_payload_bytes",
        "h2d_transfers",
        "conversion_input_bytes",
        "conversion_output_bytes",
        "conversion_seconds",
        "promotions",
        "demotions",
        "representation_hits",
        "representation_misses",
        "policy_epochs",
        "active_leases",
        "leases_by_kind",
        "manager_instance_id",
        "manager_enabled",
        "phase_policy_enabled",
        "pending_transactions",
        "registered_variants",
        "resident_generations",
        "resident_generation_entries",
        "retiring_generations",
        "external_shared_resident_bytes",
        "fallback_counts",
        "by_format",
    }
    metrics = native_adaptive_fixture.dispatcher.get_precision_metrics()
    assert required <= metrics.keys()
    assert (
        metrics["resident_payload_bytes"] + metrics["alignment_padding_bytes"]
        == metrics["resident_bytes"]
    )


@pytest.mark.gpu
def test_dispatcher_publishes_only_ready_generation(native_adaptive_fixture):
    dispatcher = native_adaptive_fixture.dispatcher
    before = dispatcher.get_precision_metrics()
    assert dispatcher.set_precision_targets([(0, 0, "bf16", 1)], epoch=1)
    output = native_adaptive_fixture.run()
    after = _wait_for_lease_drain(dispatcher)
    assert torch.isfinite(output).all()
    assert after["published_generation"] == before["published_generation"] + 1
    assert after["active_leases"] == 0
    assert (
        after["resident_bytes"]
        + after["transition_reserved_bytes"]
        + after["workspace_bytes"]
        <= after["budget_bytes"]
    )


@pytest.mark.gpu
def test_failed_transition_keeps_canonical_generation(native_adaptive_fixture):
    dispatcher = native_adaptive_fixture.dispatcher
    canonical = native_adaptive_fixture.run()
    dispatcher.inject_transition_failure_once_for_test(0, 0, "bf16")
    assert dispatcher.set_precision_targets([(0, 0, "bf16", 1)], epoch=2)
    actual = native_adaptive_fixture.run()
    metrics = _wait_for_lease_drain(dispatcher)
    assert torch.equal(actual, canonical)
    assert metrics["fallback_counts"]["transition_failed"] == 1
    assert metrics["active_format"] == "bf16"
