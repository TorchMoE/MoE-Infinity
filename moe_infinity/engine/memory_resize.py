from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from moe_infinity.memory.adaptive_memory import (
    MemoryTargets,
    ResizeDirection,
    ResizeOutcome,
    ResizeResult,
)
from moe_infinity.memory.block_pool import BlockPool


class _Event(Protocol):
    def query(self) -> bool: ...


class _CompleteEvent:
    def query(self) -> bool:
        return True


@dataclass
class ResizeReceipt:
    device_id: int
    request_queues_drained: bool
    dispatch_queues_drained: bool
    cuda_events: tuple[_Event, ...]
    admissions_paused: bool
    scheduler_token: object | None = None
    dispatcher_token: object | None = None
    retained_objects: list[object] = field(default_factory=list)
    consumed: bool = False

    def validate(self, device_id: int) -> None:
        if self.consumed:
            raise RuntimeError("resize receipt has already been consumed")
        if self.device_id != device_id:
            raise ValueError(
                "resize receipt device_id does not match transaction"
            )
        if not self.admissions_paused:
            raise RuntimeError("resize receipt requires paused admissions")
        if not self.request_queues_drained or not self.dispatch_queues_drained:
            raise RuntimeError("resize receipt requires drained queues")
        if not self.cuda_events or not all(
            event.query() for event in self.cuda_events
        ):
            raise RuntimeError(
                "resize receipt requires synchronized CUDA events"
            )

    def retain(self, value: object) -> None:
        if self.consumed:
            raise RuntimeError("cannot retain objects on a consumed receipt")
        self.retained_objects.append(value)

    def consume(self) -> None:
        if self.consumed:
            raise RuntimeError("resize receipt has already been consumed")
        self.consumed = True
        self.retained_objects.clear()


class NativeMemoryResizer:
    def __init__(
        self,
        *,
        device_id: int,
        scheduler: object,
        dispatcher: object,
        expert_cache: object,
        kv_manager: object,
        attention_backend: object,
        reserve_probe: Callable[[int], int],
        controller: object | None = None,
        timeout_ms: int = 5000,
    ) -> None:
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        self.device_id = device_id
        self.scheduler = scheduler
        self.dispatcher = dispatcher
        self.expert_cache = expert_cache
        self.kv_manager = kv_manager
        self.attention_backend = attention_backend
        self.reserve_probe = reserve_probe
        self.controller = controller
        self.timeout_ms = timeout_ms

    def quiesce(self, *, device_id: int) -> ResizeReceipt:
        if device_id != self.device_id:
            raise ValueError("resize device_id does not match adapter")
        scheduler_token = None
        dispatcher_token = None
        try:
            scheduler_token = self.scheduler.begin_memory_resize(
                device_id, timeout_ms=self.timeout_ms
            )
            dispatcher_token = self.dispatcher.begin_memory_resize(
                device_id, self.timeout_ms
            )
            ready = bool(getattr(dispatcher_token, "ready", False))
            if not ready:
                raise RuntimeError("dispatcher queues did not drain")
            event = _CompleteEvent()
            return ResizeReceipt(
                device_id=device_id,
                request_queues_drained=True,
                dispatch_queues_drained=True,
                cuda_events=(event,),
                admissions_paused=True,
                scheduler_token=scheduler_token,
                dispatcher_token=dispatcher_token,
            )
        except Exception:
            if dispatcher_token is not None:
                self.dispatcher.end_memory_resize(dispatcher_token)
            if scheduler_token is not None:
                self.scheduler.end_memory_resize(scheduler_token)
            raise

    def resume(self, receipt: ResizeReceipt) -> None:
        receipt.validate(self.device_id)
        if receipt.dispatcher_token is not None:
            self.dispatcher.end_memory_resize(receipt.dispatcher_token)
        if receipt.scheduler_token is not None:
            self.scheduler.end_memory_resize(receipt.scheduler_token)
        receipt.admissions_paused = False
        receipt.consume()

    def _current_expert_bytes(self) -> int:
        getter = getattr(self.expert_cache, "limit_bytes", None)
        if callable(getter):
            return int(getter(self.device_id))
        return 0

    def _record(self, result: ResizeResult) -> None:
        recorder = getattr(self.controller, "record_resize", None)
        if callable(recorder):
            recorder(result, step=0)

    def apply(
        self, targets: MemoryTargets, *, receipt: ResizeReceipt | None = None
    ) -> ResizeResult:
        if targets.device_id != self.device_id:
            raise ValueError("target device_id does not match adapter")
        active = receipt or self.quiesce(device_id=self.device_id)
        active.validate(self.device_id)
        old_blocks = int(self.kv_manager.num_gpu_blocks)
        old_expert = self._current_expert_bytes()
        old_pool: BlockPool | None = None
        reservation: object | None = None
        expert_committed = False
        result: ResizeResult
        try:
            if targets.direction is ResizeDirection.HOLD:
                result = ResizeResult(
                    self.device_id,
                    ResizeOutcome.REJECTED,
                    old_expert,
                    old_blocks,
                    targets.reason,
                    targets.kv_supported,
                )
            elif targets.direction is ResizeDirection.KV_TO_EXPERT:
                old_pool = self.kv_manager._gpu_pool
                self.kv_manager.resize_gpu_blocks(
                    self.device_id, targets.kv_blocks, active
                )
                self.attention_backend.resize_num_blocks(
                    self.device_id, targets.kv_blocks, active
                )
                result = ResizeResult(
                    self.device_id,
                    ResizeOutcome.COMMITTED,
                    targets.expert_bytes,
                    targets.kv_blocks,
                    targets.reason,
                    targets.kv_supported,
                )
            else:
                reservation = self.expert_cache.reserve_victims(
                    self.device_id, targets.expert_bytes
                )
                measured = int(
                    self.expert_cache.commit_reserved_victims(reservation)
                )
                expert_committed = True
                old_pool = self.kv_manager._gpu_pool
                self.kv_manager.resize_gpu_blocks(
                    self.device_id, targets.kv_blocks, active
                )
                self.attention_backend.resize_num_blocks(
                    self.device_id, targets.kv_blocks, active
                )
                result = ResizeResult(
                    self.device_id,
                    ResizeOutcome.COMMITTED,
                    measured,
                    targets.kv_blocks,
                    targets.reason,
                    targets.kv_supported,
                )
        except Exception as error:
            if old_pool is not None:
                self.kv_manager.restore_gpu_pool(old_pool)
            if reservation is not None and not expert_committed:
                self.expert_cache.cancel_reservation(reservation)
            if expert_committed:
                result = ResizeResult(
                    self.device_id,
                    ResizeOutcome.PARTIAL_DONOR_COMMITTED,
                    self._current_expert_bytes(),
                    old_blocks,
                    str(error),
                    targets.kv_supported,
                )
            else:
                result = ResizeResult(
                    self.device_id,
                    ResizeOutcome.ROLLED_BACK,
                    old_expert,
                    old_blocks,
                    str(error),
                    targets.kv_supported,
                )
        self._record(result)
        self.resume(active)
        return result


__all__ = ["NativeMemoryResizer", "ResizeReceipt"]
