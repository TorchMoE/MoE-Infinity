import threading
import time
from typing import Optional

from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferResult,
    TransferType,
)
from moe_infinity.engine.unified_transfer_scheduler import (
    UnifiedTransferScheduler,
)


def _request(
    transfer_id: str,
    transfer_type: TransferType,
    priority: TransferPriority,
    block_ids: Optional[list[int]] = None,
) -> TransferRequest:
    return TransferRequest(
        transfer_id=transfer_id,
        transfer_type=transfer_type,
        priority=priority,
        source_device="cpu",
        target_device="cuda:0",
        block_ids=[] if block_ids is None else block_ids,
    )


def test_priority_ordering() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)
    dispatch_order: list[str] = []
    lock = threading.Lock()

    def handler(req: TransferRequest) -> None:
        with lock:
            dispatch_order.append(req.transfer_id)

    scheduler.register_handler(TransferType.KV_SWAP_OUT, handler)

    low_id = scheduler.enqueue(
        _request("low", TransferType.KV_SWAP_OUT, TransferPriority.LOW)
    )
    urgent_id = scheduler.enqueue(
        _request("urgent", TransferType.KV_SWAP_OUT, TransferPriority.URGENT)
    )

    try:
        assert scheduler.wait(urgent_id, timeout_ms=2000)
        assert scheduler.wait(low_id, timeout_ms=2000)
        assert dispatch_order[:2] == ["urgent", "low"]
    finally:
        scheduler.shutdown()


def test_expert_kv_interleave() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)
    progress = {"expert": 0, "kv": 0}
    lock = threading.Lock()

    def expert_handler(_req: TransferRequest) -> None:
        with lock:
            progress["expert"] += 1

    def kv_handler(_req: TransferRequest) -> None:
        with lock:
            progress["kv"] += 1

    scheduler.register_handler(TransferType.EXPERT_FETCH, expert_handler)
    scheduler.register_handler(TransferType.KV_SWAP_IN, kv_handler)

    tids = [
        scheduler.enqueue(
            _request(
                "expert-1", TransferType.EXPERT_FETCH, TransferPriority.NORMAL
            )
        ),
        scheduler.enqueue(
            _request("kv-1", TransferType.KV_SWAP_IN, TransferPriority.NORMAL)
        ),
        scheduler.enqueue(
            _request(
                "expert-2", TransferType.EXPERT_FETCH, TransferPriority.NORMAL
            )
        ),
        scheduler.enqueue(
            _request("kv-2", TransferType.KV_SWAP_IN, TransferPriority.NORMAL)
        ),
    ]

    try:
        for tid in tids:
            assert scheduler.wait(tid, timeout_ms=2000)
        assert progress["expert"] > 0
        assert progress["kv"] > 0
    finally:
        scheduler.shutdown()


def test_cancel_pending() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)
    release = threading.Event()

    def handler(req: TransferRequest) -> None:
        if req.transfer_id == "first":
            _ = release.wait(timeout=2.0)

    scheduler.register_handler(TransferType.KV_SWAP_OUT, handler)

    first_id = scheduler.enqueue(
        _request("first", TransferType.KV_SWAP_OUT, TransferPriority.NORMAL)
    )
    second_id = scheduler.enqueue(
        _request("second", TransferType.KV_SWAP_OUT, TransferPriority.NORMAL)
    )

    try:
        time.sleep(0.05)
        assert scheduler.cancel(second_id)
        release.set()
        assert scheduler.wait(first_id, timeout_ms=2000)
        assert scheduler.wait(second_id, timeout_ms=2000)
        result = scheduler.get_result(second_id)
        assert result is not None
        assert result.status == "CANCELLED"
    finally:
        scheduler.shutdown()


def test_wait_timeout() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)

    def slow_handler(_req: TransferRequest) -> None:
        time.sleep(0.25)

    scheduler.register_handler(TransferType.EXPERT_FETCH, slow_handler)
    transfer_id = scheduler.enqueue(
        _request("slow", TransferType.EXPERT_FETCH, TransferPriority.NORMAL)
    )

    try:
        assert not scheduler.wait(transfer_id, timeout_ms=10)
        assert scheduler.wait(transfer_id, timeout_ms=2000)
    finally:
        scheduler.shutdown()


def test_metrics_tracked() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)

    def expert_handler(_req: TransferRequest) -> int:
        return 0

    def kv_handler(_req: TransferRequest) -> int:
        return 4096

    scheduler.register_handler(TransferType.EXPERT_FETCH, expert_handler)
    scheduler.register_handler(TransferType.KV_SWAP_OUT, kv_handler)

    expert_id = scheduler.enqueue(
        _request("exp", TransferType.EXPERT_FETCH, TransferPriority.NORMAL)
    )
    kv_id = scheduler.enqueue(
        _request(
            "kv",
            TransferType.KV_SWAP_OUT,
            TransferPriority.NORMAL,
            block_ids=[1, 2, 3],
        )
    )

    try:
        assert scheduler.wait(expert_id, timeout_ms=2000)
        assert scheduler.wait(kv_id, timeout_ms=2000)
        metrics = scheduler.get_metrics()
        assert metrics["EXPERT_FETCH"]["count"] == 1
        assert metrics["KV_SWAP_OUT"]["count"] == 1
        assert metrics["KV_SWAP_OUT"]["bytes"] == 4096

        result = scheduler.get_result(kv_id)
        assert result is not None
        assert result.status == "COMPLETED"
        assert result.bytes_transferred == 4096
        assert result.error is None
    finally:
        scheduler.shutdown()


def test_handler_failure_records_error_text_and_failure_metric() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)

    def failing_handler(_req: TransferRequest) -> int:
        raise RuntimeError("missing host KV for transfer missing")

    scheduler.register_handler(TransferType.KV_SWAP_IN, failing_handler)
    transfer_id = scheduler.enqueue(
        _request("missing", TransferType.KV_SWAP_IN, TransferPriority.NORMAL)
    )

    try:
        assert scheduler.wait(transfer_id, timeout_ms=2000)
        result = scheduler.get_result(transfer_id)
        assert result is not None
        assert result.status == "FAILED"
        assert result.error == (
            "RuntimeError: missing host KV for transfer missing"
        )
        assert result.bytes_transferred == 0
        metrics = scheduler.get_metrics()
        assert metrics["KV_SWAP_IN"]["count"] == 1
        assert metrics["KV_SWAP_IN"]["failures"] == 1
        assert metrics["KV_SWAP_IN"]["bytes"] == 0
    finally:
        scheduler.shutdown()


def test_pending_cancellation_records_cancelled_metric_and_zero_bytes() -> None:
    scheduler = UnifiedTransferScheduler(max_workers=1)
    release = threading.Event()
    ran_after_cancel = {"count": 0}

    def handler(req: TransferRequest) -> int:
        if req.transfer_id == "first":
            _ = release.wait(timeout=2.0)
            return 10
        ran_after_cancel["count"] += 1
        return 999

    scheduler.register_handler(TransferType.KV_SWAP_OUT, handler)

    first_id = scheduler.enqueue(
        _request("first", TransferType.KV_SWAP_OUT, TransferPriority.NORMAL)
    )
    second_id = scheduler.enqueue(
        _request("second", TransferType.KV_SWAP_OUT, TransferPriority.NORMAL)
    )

    try:
        time.sleep(0.05)
        assert scheduler.cancel(second_id)
        release.set()
        assert scheduler.wait(first_id, timeout_ms=2000)
        assert scheduler.wait(second_id, timeout_ms=2000)

        result = scheduler.get_result(second_id)
        assert result is not None
        assert result.status == "CANCELLED"
        assert result.bytes_transferred == 0

        metrics = scheduler.get_metrics()
        assert metrics["KV_SWAP_OUT"]["cancelled"] == 1
        assert ran_after_cancel["count"] == 0
        assert metrics["KV_SWAP_OUT"]["bytes"] == 10
        assert metrics["KV_SWAP_OUT"]["count"] == 1
        assert metrics["KV_SWAP_OUT"]["failures"] == 0

        for stored in list(scheduler._results.values()):
            assert isinstance(stored, TransferResult)
            assert not hasattr(stored, "query")
            assert not hasattr(stored, "record_event")
    finally:
        scheduler.shutdown()
