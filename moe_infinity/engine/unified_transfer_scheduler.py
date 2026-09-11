import heapq
import threading
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Callable, Optional

from typing_extensions import override

from moe_infinity.engine.transfer_types import (
    TransferRequest,
    TransferResult,
    TransferType,
)

try:
    import nvtx  # type: ignore[reportMissingTypeStubs]
except ImportError:
    nvtx = None

HAS_NVTX = nvtx is not None

try:
    from moe_infinity.profiling.io_profiler import (  # pyright: ignore[reportMissingImports]
        IOProfiler,
    )
except Exception:
    IOProfiler = None


class TransferScheduler(ABC):
    @abstractmethod
    def enqueue(self, request: TransferRequest) -> str: ...

    @abstractmethod
    def cancel(self, transfer_id: str) -> bool: ...

    @abstractmethod
    def wait(self, transfer_id: str, timeout_ms: float = 5000.0) -> bool: ...

    @abstractmethod
    def get_pending_count(self) -> dict[TransferType, int]: ...

    @abstractmethod
    def set_bandwidth_budget(
        self, expert_ratio: float, kv_ratio: float
    ) -> None: ...


class UnifiedTransferScheduler(TransferScheduler):
    def __init__(
        self,
        max_workers: int = 2,
        expert_bandwidth_ratio: float = 0.6,
        kv_bandwidth_ratio: float = 0.4,
    ):
        self._queue: list[tuple[int, int, TransferRequest]] = []
        self._seq_counter: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._condition: threading.Condition = threading.Condition(self._lock)

        self._pending: dict[str, threading.Event] = {}
        self._requests: dict[str, TransferRequest] = {}
        self._results: dict[str, TransferResult] = {}

        self._cancelled: set[str] = set()

        self._metrics: dict[TransferType, dict[str, int]] = {
            t: {"count": 0, "bytes": 0, "failures": 0, "cancelled": 0}
            for t in TransferType
        }

        self._bandwidth_budgets: dict[str, float] = {
            "expert": expert_bandwidth_ratio,
            "kv": kv_bandwidth_ratio,
        }

        self._handlers: dict[
            TransferType, Callable[[TransferRequest], Optional[int]]
        ] = {}

        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=max_workers
        )
        self._running: bool = True
        self._worker_thread: threading.Thread = threading.Thread(
            target=self._worker_loop, daemon=True
        )
        self._worker_thread.start()

    def register_handler(
        self,
        transfer_type: TransferType,
        handler: Callable[[TransferRequest], Optional[int]],
    ) -> None:
        with self._lock:
            self._handlers[transfer_type] = handler

    @override
    def enqueue(self, request: TransferRequest) -> str:
        profiler = IOProfiler.instance() if IOProfiler is not None else None
        nvtx_cm = nullcontext()
        if HAS_NVTX and nvtx is not None:
            nvtx_cm = nvtx.annotate("transfer_schedule", color="yellow")
        profiler_cm = (
            profiler.time("transfer_schedule")
            if profiler is not None
            else nullcontext()
        )

        with profiler_cm:
            with nvtx_cm:
                transfer_id = request.transfer_id or str(uuid.uuid4())
                normalized_request = TransferRequest(
                    transfer_id=transfer_id,
                    transfer_type=request.transfer_type,
                    priority=request.priority,
                    source_device=request.source_device,
                    target_device=request.target_device,
                    tensor_id=request.tensor_id,
                    block_ids=list(request.block_ids),
                )
                event = threading.Event()
                with self._condition:
                    self._pending[transfer_id] = event
                    self._requests[transfer_id] = normalized_request
                    self._seq_counter += 1
                    heapq.heappush(
                        self._queue,
                        (
                            normalized_request.priority.value,
                            self._seq_counter,
                            normalized_request,
                        ),
                    )
                    self._condition.notify()
                return transfer_id

    @override
    def cancel(self, transfer_id: str) -> bool:
        with self._condition:
            if transfer_id not in self._pending:
                return False
            request = self._requests[transfer_id]
            if not any(
                queued.transfer_id == transfer_id
                for _, _, queued in self._queue
            ):
                return False
            self._cancelled.add(transfer_id)
            event = self._pending.pop(transfer_id, None)
            if event:
                self._results[transfer_id] = TransferResult(
                    transfer_id=transfer_id,
                    status="CANCELLED",
                    duration_ms=0.0,
                )
                self._metrics[request.transfer_type]["cancelled"] += 1
                event.set()
        return True

    @override
    def wait(self, transfer_id: str, timeout_ms: float = 5000.0) -> bool:
        with self._lock:
            event = self._pending.get(transfer_id)
            if event is None:
                return transfer_id in self._results
        return event.wait(timeout=timeout_ms / 1000.0)

    @override
    def get_pending_count(self) -> dict[TransferType, int]:
        with self._lock:
            counts: dict[TransferType, int] = {t: 0 for t in TransferType}
            for _, _, req in self._queue:
                counts[req.transfer_type] += 1
        return counts

    def get_result(self, transfer_id: str) -> Optional[TransferResult]:
        with self._lock:
            return self._results.get(transfer_id)

    @override
    def set_bandwidth_budget(
        self, expert_ratio: float, kv_ratio: float
    ) -> None:
        assert 0.0 <= expert_ratio + kv_ratio <= 1.0
        with self._lock:
            self._bandwidth_budgets["expert"] = expert_ratio
            self._bandwidth_budgets["kv"] = kv_ratio

    def get_metrics(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {t.name: dict(v) for t, v in self._metrics.items()}

    def shutdown(self, wait: bool = True) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if wait:
            self._worker_thread.join(timeout=5.0)
        self._executor.shutdown(wait=wait)

    def _run_request(self, request: TransferRequest, start_time: float) -> None:
        profiler = IOProfiler.instance() if IOProfiler is not None else None
        nvtx_cm = nullcontext()
        if HAS_NVTX and nvtx is not None:
            nvtx_cm = nvtx.annotate("transfer_schedule", color="yellow")
        profiler_cm = (
            profiler.time("transfer_schedule")
            if profiler is not None
            else nullcontext()
        )

        with profiler_cm:
            with nvtx_cm:
                transfer_id = request.transfer_id
                bytes_transferred = 0
                error: Optional[str] = None
                try:
                    handler = self._handlers.get(request.transfer_type)
                    if handler is not None:
                        handler_bytes = handler(request)
                        if handler_bytes is not None:
                            bytes_transferred = int(handler_bytes)
                    status = "COMPLETED"
                except Exception as exc:
                    status = "FAILED"
                    bytes_transferred = 0
                    error = f"{type(exc).__name__}: {exc}"

                duration_ms = (time.monotonic() - start_time) * 1000.0
                with self._lock:
                    self._results[transfer_id] = TransferResult(
                        transfer_id=transfer_id,
                        status=status,
                        duration_ms=duration_ms,
                        bytes_transferred=bytes_transferred,
                        error=error,
                    )
                    metrics = self._metrics[request.transfer_type]
                    metrics["count"] += 1
                    if status == "FAILED":
                        metrics["failures"] += 1
                    else:
                        metrics["bytes"] += bytes_transferred
                    event = self._pending.pop(transfer_id, None)
                    self._requests.pop(transfer_id, None)
                    if event:
                        event.set()

    def _worker_loop(self) -> None:
        while self._running:
            with self._condition:
                while self._running and not self._queue:
                    _ = self._condition.wait(timeout=0.1)
                if not self._running:
                    break
                if not self._queue:
                    continue
                _, _, request = heapq.heappop(self._queue)

            transfer_id = request.transfer_id
            with self._lock:
                if transfer_id in self._cancelled:
                    self._cancelled.discard(transfer_id)
                    self._requests.pop(transfer_id, None)
                    continue

            start_time = time.monotonic()
            future = self._executor.submit(
                self._run_request, request, start_time
            )
            future.result()
