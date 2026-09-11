import pytest

from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)
from moe_infinity.engine.unified_transfer_scheduler import (
    TransferScheduler,  # pyright: ignore[reportMissingImports, reportUnknownVariableType]
)


def test_transfer_scheduler_is_abstract():
    with pytest.raises(TypeError):
        TransferScheduler()


def test_incomplete_subclass_cannot_instantiate():
    class IncompleteScheduler(TransferScheduler):  # pyright: ignore[reportUntypedBaseClass]
        def enqueue(self, request: TransferRequest) -> str:
            return request.transfer_id

        def cancel(self, _transfer_id: str) -> bool:
            return False

        def wait(self, _transfer_id: str, _timeout_ms: float = 5000.0) -> bool:
            return False

        def get_pending_count(self) -> dict[TransferType, int]:
            return {}

    with pytest.raises(TypeError):
        _ = IncompleteScheduler()


def test_complete_subclass_can_instantiate():
    class CompleteScheduler(TransferScheduler):  # pyright: ignore[reportUntypedBaseClass]
        def enqueue(self, request: TransferRequest) -> str:
            return request.transfer_id

        def cancel(self, _transfer_id: str) -> bool:
            return True

        def wait(self, _transfer_id: str, _timeout_ms: float = 5000.0) -> bool:
            return True

        def wait_for_device(self, device_id: int, timeout_ms: float) -> bool:
            _ = (device_id, timeout_ms)
            return True

        def get_pending_count(self) -> dict[TransferType, int]:
            return {TransferType.KV_SWAP_IN: 0}

        def set_bandwidth_budget(
            self, expert_ratio: float, kv_ratio: float
        ) -> None:
            if expert_ratio + kv_ratio > 1.0:
                raise ValueError(
                    "bandwidth budget must satisfy expert_ratio + kv_ratio <= 1.0"
                )

    scheduler = CompleteScheduler()
    assert (
        scheduler.enqueue(
            TransferRequest(
                transfer_id="t1",
                transfer_type=TransferType.KV_SWAP_IN,
                priority=TransferPriority.NORMAL,
                source_device="cpu",
                target_device="cuda:0",
                device_id=0,
            )
        )
        == "t1"
    )


def test_transfer_priority_values():
    assert TransferPriority.URGENT.value == 0
    assert TransferPriority.NORMAL.value == 10
    assert TransferPriority.BACKGROUND.value == 19


def test_bandwidth_budget_validation_in_subclass():
    class BudgetCheckingScheduler(TransferScheduler):  # pyright: ignore[reportUntypedBaseClass]
        def enqueue(self, request: TransferRequest) -> str:
            return request.transfer_id

        def cancel(self, _transfer_id: str) -> bool:
            return True

        def wait(self, _transfer_id: str, _timeout_ms: float = 5000.0) -> bool:
            return True

        def wait_for_device(self, _device_id: int, _timeout_ms: float) -> bool:
            return True

        def get_pending_count(self) -> dict[TransferType, int]:
            return {}

        def set_bandwidth_budget(
            self, expert_ratio: float, kv_ratio: float
        ) -> None:
            if expert_ratio + kv_ratio > 1.0:
                raise ValueError("expert_ratio + kv_ratio must be <= 1.0")

    scheduler = BudgetCheckingScheduler()
    scheduler.set_bandwidth_budget(0.4, 0.6)

    with pytest.raises(ValueError):
        scheduler.set_bandwidth_budget(0.7, 0.4)
