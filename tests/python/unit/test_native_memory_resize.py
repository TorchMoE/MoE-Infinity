from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from moe_infinity.engine.memory_resize import NativeMemoryResizer, ResizeReceipt
from moe_infinity.engine.scheduler import Scheduler
from moe_infinity.engine.transfer_types import TransferRequest, TransferType
from moe_infinity.engine.types import Request, SamplingParams, SequenceStatus
from moe_infinity.engine.unified_transfer_scheduler import TransferScheduler
from moe_infinity.memory.adaptive_memory import (
    MemoryTargets,
    ResizeDirection,
    ResizeOutcome,
)
from moe_infinity.memory.block_pool import BlockPool
from moe_infinity.memory.kv_cache_manager import KVCacheManager


class FakeEvent:
    def query(self) -> bool:
        return True


def completed_receipt(*, device_id: int) -> ResizeReceipt:
    return ResizeReceipt(
        device_id=device_id,
        request_queues_drained=True,
        dispatch_queues_drained=True,
        cuda_events=(FakeEvent(),),
        admissions_paused=True,
    )


def targets(*, device_id: int, expert: int, kv: int) -> MemoryTargets:
    direction = (
        ResizeDirection.EXPERT_TO_KV if kv > 8 else ResizeDirection.KV_TO_EXPERT
    )
    return MemoryTargets(device_id, expert, kv, direction, "test")


def make_request(request_id: str, *, tokens: int = 8) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(tokens)),
        sampling_params=SamplingParams(),
        arrival_time=time.time(),
    )


def make_running_request(request_id: str) -> Request:
    request = make_request(request_id)
    request.status = SequenceStatus.RUNNING
    return request


class RecordingTransferScheduler(TransferScheduler):
    def __init__(self) -> None:
        self.requests: list[TransferRequest] = []
        self._by_id: dict[str, TransferRequest] = {}

    def enqueue(self, request: TransferRequest) -> str:
        self.requests.append(request)
        self._by_id[request.transfer_id] = request
        return request.transfer_id

    def wait(self, transfer_id: str, timeout_ms: float = 5000.0) -> bool:
        _ = timeout_ms
        return transfer_id in self._by_id

    def wait_for_device(self, device_id: int, timeout_ms: float) -> bool:
        _ = timeout_ms
        return all(
            request.device_id != device_id or request.transfer_id in self._by_id
            for request in self.requests
        )

    def cancel(self, transfer_id: str) -> bool:
        return self._by_id.pop(transfer_id, None) is not None

    def shutdown(self, wait: bool = True) -> None:
        _ = wait

    def get_pending_count(self) -> dict[TransferType, int]:
        return {}

    def set_bandwidth_budget(
        self, expert_ratio: float, kv_ratio: float
    ) -> None:
        _ = (expert_ratio, kv_ratio)


class FakeDispatcher:
    def __init__(self) -> None:
        self._paused: set[int] = set()
        self._fetches: dict[int, int] = {}

    def enqueue_fetch(self, *, device_id: int) -> None:
        self._fetches[device_id] = self._fetches.get(device_id, 0) + 1

    def begin_memory_resize(self, device_id: int, timeout_ms: int) -> object:
        _ = timeout_ms
        self._paused.add(device_id)
        self._fetches[device_id] = 0
        return SimpleNamespace(device_id=device_id, ready=True)

    def end_memory_resize(self, token: object) -> None:
        self._paused.remove(int(token.device_id))

    def admissions_paused(self, device_id: int) -> bool:
        return device_id in self._paused


class FakeExpertCache:
    def __init__(self, device_id: int, resident_bytes: int) -> None:
        self.device_id = device_id
        self._limit = resident_bytes

    def reserve_victims(self, device_id: int, target_bytes: int) -> object:
        assert device_id == self.device_id
        return SimpleNamespace(device_id=device_id, target_bytes=target_bytes)

    def commit_reserved_victims(self, reservation: object) -> int:
        self._limit = int(reservation.target_bytes)
        return self._limit

    def cancel_reservation(self, reservation: object) -> None:
        _ = reservation

    def limit_bytes(self, device_id: int) -> int:
        assert device_id == self.device_id
        return self._limit


class FakeAttentionBackend:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks

    def resize_num_blocks(
        self, device_id: int, target_blocks: int, receipt: ResizeReceipt
    ) -> None:
        assert receipt.device_id == device_id
        self.num_blocks = target_blocks


def make_native_bundle(
    *, device_id: int, gpu_blocks: int, expert_bytes: int = 1024
) -> SimpleNamespace:
    transfer = RecordingTransferScheduler()
    manager = KVCacheManager(
        num_gpu_blocks=gpu_blocks,
        num_cpu_blocks=32,
        block_size=4,
        device_id=device_id,
    )
    scheduler = Scheduler(
        kv_cache_manager=manager,
        transfer_scheduler=transfer,
        device_id=device_id,
    )
    dispatcher = FakeDispatcher()
    expert = FakeExpertCache(device_id, expert_bytes)
    attention = FakeAttentionBackend(gpu_blocks)
    resizer = NativeMemoryResizer(
        device_id=device_id,
        scheduler=scheduler,
        dispatcher=dispatcher,
        expert_cache=expert,
        kv_manager=manager,
        attention_backend=attention,
        reserve_probe=lambda _: 2**40,
    )
    return SimpleNamespace(
        transfer=transfer,
        scheduler=scheduler,
        dispatcher=dispatcher,
        expert=expert,
        kv_mgr=manager,
        attention=attention,
        resizer=resizer,
    )


def test_native_pool_resize_rejects_referenced_blocks() -> None:
    mgr = KVCacheManager(8, 16, block_size=4)
    assert mgr.allocate_blocks_for_sequence("r1", 8)
    with pytest.raises(RuntimeError, match="referenced KV blocks"):
        mgr.resize_gpu_blocks(0, 4, completed_receipt(device_id=0))


def test_native_transaction_rolls_back_when_attention_allocation_ooms() -> None:
    bundle = make_native_bundle(device_id=0, gpu_blocks=8)
    running = make_running_request("resident")
    assert bundle.kv_mgr.allocate_blocks_for_sequence(running.request_id, 8)
    bundle.scheduler._running.append(running)
    receipt = bundle.resizer.quiesce(device_id=0)
    bundle.attention.resize_num_blocks = Mock(
        side_effect=torch.OutOfMemoryError()
    )
    result = bundle.resizer.apply(
        targets(device_id=0, expert=1536, kv=4), receipt=receipt
    )
    assert result.outcome is ResizeOutcome.ROLLED_BACK
    assert bundle.kv_mgr.num_gpu_blocks == 8
    assert bundle.scheduler.num_swapped == 1
    assert bundle.scheduler.admissions_paused is False


def test_native_shrink_never_removes_cached_block_with_reference() -> None:
    pool = BlockPool(4)
    block = pool.allocate_block()
    assert block is not None and block.ref_cnt == 1
    assert pool.removable_tail_ids(2) == []


def test_expert_donor_failure_after_eviction_commits_reduced_state() -> None:
    bundle = make_native_bundle(device_id=1, gpu_blocks=8, expert_bytes=1024)
    bundle.attention.resize_num_blocks = Mock(
        side_effect=torch.OutOfMemoryError()
    )
    result = bundle.resizer.apply(targets(device_id=1, expert=512, kv=12))
    assert result.outcome is ResizeOutcome.PARTIAL_DONOR_COMMITTED
    assert result.device_id == 1
    assert result.expert_bytes == 512
    assert result.kv_blocks == 8
    assert bundle.expert.limit_bytes(1) == 512
    assert bundle.scheduler.admissions_paused is False


def test_native_quiescence_drains_transfers_and_synchronizes_streams() -> None:
    bundle = make_native_bundle(device_id=1, gpu_blocks=8)
    bundle.scheduler.add_request(make_request("r1"))
    bundle.dispatcher.enqueue_fetch(device_id=1)
    receipt = bundle.resizer.quiesce(device_id=1)
    assert bundle.scheduler.admissions_paused is True
    assert bundle.dispatcher.admissions_paused(1) is True
    assert receipt.request_queues_drained
    assert receipt.dispatch_queues_drained
    assert all(event.query() for event in receipt.cuda_events)
    bundle.resizer.resume(receipt)
    assert bundle.scheduler.admissions_paused is False
    assert bundle.dispatcher.admissions_paused(1) is False
