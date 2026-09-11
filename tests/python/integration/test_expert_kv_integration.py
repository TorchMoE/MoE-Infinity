from importlib import import_module

from moe_infinity.engine.expert_offload_coordinator import (
    ExpertOffloadCoordinator,
)
from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)
from moe_infinity.engine.unified_transfer_scheduler import (
    UnifiedTransferScheduler,
)


def _kv_swap_in_request(transfer_id: str) -> TransferRequest:
    return TransferRequest(
        transfer_id=transfer_id,
        transfer_type=TransferType.KV_SWAP_IN,
        priority=TransferPriority.NORMAL,
        source_device="cpu",
        target_device="cuda:0",
        device_id=0,
        block_ids=[0, 1],
    )


def test_expert_and_kv_transfers_no_crash() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=2)
    coordinator = ExpertOffloadCoordinator()
    kv_seen: list[str] = []

    def kv_handler(req: TransferRequest) -> None:
        kv_seen.append(req.transfer_id)

    coordinator.register_with_scheduler(scheduler)
    scheduler.register_handler(TransferType.KV_SWAP_IN, kv_handler)

    expert_transfer_id = coordinator.prefetch_experts(
        layer_id=1,
        expert_ids=[2, 3],
        priority=TransferPriority.HIGH,
    )
    kv_transfer_id = scheduler.enqueue(_kv_swap_in_request("kv-1"))

    try:
        assert scheduler.wait(expert_transfer_id, timeout_ms=2000)
        assert scheduler.wait(kv_transfer_id, timeout_ms=2000)

        expert_result = scheduler.get_result(expert_transfer_id)
        kv_result = scheduler.get_result(kv_transfer_id)
        assert expert_result is not None
        assert kv_result is not None
        assert expert_result.status == "COMPLETED"
        assert kv_result.status == "COMPLETED"
        assert "kv-1" in kv_seen

        metrics = scheduler.get_metrics()
        assert metrics[TransferType.EXPERT_FETCH.name]["count"] >= 1
        assert metrics[TransferType.KV_SWAP_IN.name]["count"] >= 1
    finally:
        scheduler.shutdown()


def test_coordinator_registers_handler() -> None:
    class _RecordingPrefetcher:
        def __init__(self) -> None:
            self.fetch_calls: list[tuple[int, list[int]]] = []
            self.evict_calls: list[tuple[int, list[int]]] = []

        def prefetch_experts_list(
            self, layer_id: int, expert_list: list[int]
        ) -> None:
            self.fetch_calls.append((layer_id, list(expert_list)))

        def fetch_experts_lock_cache(
            self, layer_id: int, expert_list: list[int]
        ) -> None:
            self.evict_calls.append((layer_id, list(expert_list)))

    scheduler = UnifiedTransferScheduler(max_workers=1)
    prefetcher = _RecordingPrefetcher()
    coordinator = ExpertOffloadCoordinator(expert_prefetcher=prefetcher)

    try:
        coordinator.register_with_scheduler(scheduler)

        fetch_id = scheduler.enqueue(
            TransferRequest(
                transfer_id="fetch-1",
                transfer_type=TransferType.EXPERT_FETCH,
                priority=TransferPriority.HIGH,
                source_device="cpu",
                target_device="cuda:0",
                device_id=0,
                tensor_id="3",
                block_ids=[10, 11],
            )
        )
        evict_id = scheduler.enqueue(
            TransferRequest(
                transfer_id="evict-1",
                transfer_type=TransferType.EXPERT_EVICT,
                priority=TransferPriority.NORMAL,
                source_device="cuda:0",
                target_device="cpu",
                device_id=0,
                tensor_id="4",
                block_ids=[12],
            )
        )

        assert scheduler.wait(fetch_id, timeout_ms=2000)
        assert scheduler.wait(evict_id, timeout_ms=2000)
        assert prefetcher.fetch_calls == [(3, [10, 11])]
        assert prefetcher.evict_calls == [(4, [12])]
    finally:
        scheduler.shutdown()


def test_expert_offload_coordinator_import() -> None:
    module = import_module("moe_infinity.engine.expert_offload_coordinator")
    assert hasattr(module, "ExpertOffloadCoordinator")
