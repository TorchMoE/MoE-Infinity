# pyright: reportMissingImports=false, reportUnknownVariableType=false

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol, cast

from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)
from moe_infinity.engine.types import (
    Request,
    SchedulerOutput,
    Sequence,
    SequenceStatus,
)
from moe_infinity.engine.unified_transfer_scheduler import (
    TransferScheduler,
)
from moe_infinity.memory.kv_cache_manager import KVCacheManager

try:
    from moe_infinity.serving.cp_kv_interface import (
        CPAwareKVManager as _RuntimeCPAwareKVManager,
    )
except ImportError:
    _RuntimeCPAwareKVManager = None

_ = _RuntimeCPAwareKVManager


class CPAwareKVManager(Protocol):
    def predict_prefix_reuse(
        self, request_id: str, token_ids: list[int]
    ) -> float: ...

    def notify_blocks_allocated(
        self, request_id: str, block_hashes: list[int]
    ) -> None: ...


@dataclass(frozen=True)
class MemoryResizeToken:
    device_id: int
    waiting_ids: tuple[str, ...]
    running_ids: tuple[str, ...]
    swapped_ids: tuple[str, ...]

    def notify_blocks_freed(
        self, request_id: str, block_hashes: list[int]
    ) -> None: ...


class Scheduler:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        transfer_scheduler: Optional[TransferScheduler] = None,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
        device_id: int = 0,
    ):
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if kv_cache_manager.device_id != device_id:
            raise ValueError("scheduler device_id must match KV cache owner")
        self.device_id = device_id
        self.kv_mgr: KVCacheManager = kv_cache_manager
        self._transfer_scheduler: Optional[TransferScheduler] = (
            transfer_scheduler
        )
        self.max_num_seqs: int = max_num_seqs
        self.max_num_batched_tokens: int = max_num_batched_tokens

        self._waiting: deque[Request] = deque()
        self._running: deque[Request] = deque()
        self._swapped: deque[Request] = deque()
        self._swapped_gpu_ids: dict[str, list[int]] = {}
        self._swap_needs_reprefill: set[str] = set()
        self._seq_generated_tokens: dict[str, int] = {}
        self._cp_kv_manager: Optional[CPAwareKVManager] = None
        self._cp_request_block_hashes: dict[str, list[int]] = {}
        self._admissions_paused = False
        self._maintenance_backlog: deque[Request] = deque()
        self._resize_preemptions = 0

    def set_cp_kv_manager(self, manager: CPAwareKVManager) -> None:
        self._cp_kv_manager = manager

    def add_request(self, request: Request) -> None:
        if self._admissions_paused:
            self._maintenance_backlog.append(request)
        else:
            self._waiting.append(request)

    def schedule(self) -> SchedulerOutput:
        if self._admissions_paused:
            return SchedulerOutput([], [], [], 0)
        scheduled: list[Request] = []
        preempted: list[Request] = []
        swapped_in: list[Request] = []
        num_tokens = 0

        previously_swapped = deque(self._swapped)
        self._swapped = deque()

        new_running: list[Request] = []

        if self._cp_kv_manager is not None and len(self._waiting) > 1:
            scored_waiting: list[tuple[float, Request]] = []
            for req in self._waiting:
                try:
                    score = self._cp_kv_manager.predict_prefix_reuse(
                        req.request_id,
                        req.prompt_token_ids,
                    )
                except Exception:
                    score = 0.0
                scored_waiting.append((float(score), req))

            scored_waiting.sort(key=lambda x: x[0], reverse=True)
            self._waiting = deque(req for _, req in scored_waiting)

        while self._waiting:
            req = self._waiting[0]
            prompt_tokens = len(req.prompt_token_ids)

            if len(self._running) + len(new_running) >= self.max_num_seqs:
                break

            _ = self._waiting.popleft()

            allocated, num_cached_tokens = self._allocate_with_prefix_cache(req)
            while not allocated and self._running:
                victim = self._running.popleft()
                self._preempt_with_transfer(victim)
                preempted.append(victim)

                allocated, num_cached_tokens = self._allocate_with_prefix_cache(
                    req
                )

            if not allocated:
                self._waiting.appendleft(req)
                break

            uncached_prompt_tokens = prompt_tokens - num_cached_tokens
            if (
                num_tokens + uncached_prompt_tokens
                > self.max_num_batched_tokens
            ):
                self._notify_cp_blocks_freed(req.request_id)
                self.kv_mgr.free_sequence(req.request_id)
                self._waiting.appendleft(req)
                break

            req.transition_to(SequenceStatus.RUNNING)
            new_running.append(req)
            scheduled.append(req)
            num_tokens += uncached_prompt_tokens
            self._seq_generated_tokens[req.request_id] = 0

        self._running.extend(new_running)

        if previously_swapped:
            for req in previously_swapped:
                if len(self._running) >= self.max_num_seqs:
                    self._swapped.append(req)
                    continue
                if self._swap_in_request(req):
                    swapped_in.append(req)
                else:
                    self._swapped.append(req)

        scheduled_ids = {req.request_id for req in scheduled}
        for req in self._running:
            if req.request_id in scheduled_ids:
                continue
            if num_tokens + 1 > self.max_num_batched_tokens:
                break
            scheduled.append(req)
            num_tokens += 1
            self._seq_generated_tokens[req.request_id] = (
                self._seq_generated_tokens.get(req.request_id, 0) + 1
            )

        return SchedulerOutput(
            scheduled_seqs=cast(list[Sequence], scheduled),
            preempted_seqs=cast(list[Sequence], preempted),
            swapped_in_seqs=cast(list[Sequence], swapped_in),
            num_batched_tokens=num_tokens,
        )

    def finish_request(self, request_id: str) -> None:
        request: Optional[Request] = None
        for req in [*self._waiting, *self._running, *self._swapped]:
            if req.request_id == request_id:
                request = req
                break

        if request is not None and self.kv_mgr.get_block_table(request_id):
            self._register_completed_blocks_in_cache(request)

        self._waiting = deque(
            req for req in self._waiting if req.request_id != request_id
        )
        self._running = deque(
            req for req in self._running if req.request_id != request_id
        )
        self._swapped = deque(
            req for req in self._swapped if req.request_id != request_id
        )
        _ = self._swapped_gpu_ids.pop(request_id, None)
        self._swap_needs_reprefill.discard(request_id)
        _ = self._seq_generated_tokens.pop(request_id, None)
        self._notify_cp_blocks_freed(request_id)
        self.kv_mgr.free_sequence(request_id)

    def _allocate_with_prefix_cache(self, req: Request) -> tuple[bool, int]:
        from moe_infinity.memory.block_pool import (
            KVCacheBlock,
            hash_block_tokens,
        )

        prompt = req.prompt_token_ids
        block_size = self.kv_mgr.block_size

        num_full_blocks = len(prompt) // block_size
        cached_blocks: list[KVCacheBlock] = []
        num_cached_tokens = 0
        parent_hash = 0

        for i in range(num_full_blocks):
            token_slice = tuple(prompt[i * block_size : (i + 1) * block_size])
            block_hash = hash_block_tokens(parent_hash, token_slice)
            cached_block = self.kv_mgr.get_cached_gpu_block(block_hash)
            if cached_block is None:
                break
            cached_blocks.append(cached_block)
            num_cached_tokens += block_size
            parent_hash = block_hash

        remaining_tokens = len(prompt) - num_cached_tokens
        remaining_blocks_needed = (
            (remaining_tokens + block_size - 1) // block_size
            if remaining_tokens > 0
            else 0
        )

        new_blocks: list[KVCacheBlock] = []
        for _ in range(remaining_blocks_needed):
            block = self.kv_mgr.allocate_gpu_block()
            if block is None:
                for cached_block in cached_blocks:
                    self.kv_mgr.release_gpu_block(cached_block)
                for new_block in new_blocks:
                    self.kv_mgr.release_gpu_block(new_block)
                return False, 0

            new_blocks.append(block)

        all_block_ids = [b.block_id for b in cached_blocks] + [
            b.block_id for b in new_blocks
        ]
        self.kv_mgr.set_block_table(req.request_id, all_block_ids)

        cp_block_hashes = self._compute_prompt_block_hashes(
            req.prompt_token_ids
        )
        self._cp_request_block_hashes[req.request_id] = cp_block_hashes
        if self._cp_kv_manager is not None and cp_block_hashes:
            try:
                self._cp_kv_manager.notify_blocks_allocated(
                    req.request_id,
                    list(cp_block_hashes),
                )
            except Exception:
                pass

        return True, num_cached_tokens

    def _register_completed_blocks_in_cache(self, req: Request) -> None:
        from moe_infinity.memory.block_pool import hash_block_tokens

        prompt = req.prompt_token_ids
        block_size = self.kv_mgr.block_size
        block_ids = self.kv_mgr.get_block_table(req.request_id)

        parent_hash = 0
        num_full_blocks = len(prompt) // block_size

        for i in range(num_full_blocks):
            if i >= len(block_ids):
                break
            token_slice = tuple(prompt[i * block_size : (i + 1) * block_size])
            block_hash = hash_block_tokens(parent_hash, token_slice)
            block = self.kv_mgr.get_gpu_block(block_ids[i])
            if block is None:
                break
            self.kv_mgr.cache_gpu_block(block, block_hash)
            parent_hash = block_hash

    def _notify_cp_blocks_freed(self, request_id: str) -> None:
        cp_block_hashes = self._cp_request_block_hashes.pop(request_id, [])
        if self._cp_kv_manager is None or not cp_block_hashes:
            return
        try:
            self._cp_kv_manager.notify_blocks_freed(
                request_id,
                list(cp_block_hashes),
            )
        except Exception:
            pass

    def abort_request(self, request_id: str) -> None:
        self.finish_request(request_id)

    def _can_schedule(self, req: Request) -> bool:
        prompt_tokens = len(req.prompt_token_ids)
        blocks_needed = (
            prompt_tokens + self.kv_mgr.block_size - 1
        ) // self.kv_mgr.block_size
        return self.kv_mgr.num_free_gpu_blocks >= blocks_needed

    def _preempt_with_transfer(self, victim: Request) -> None:
        pairs = self.kv_mgr.prepare_swap_out(self.device_id, victim.request_id)
        if not pairs:
            self._notify_cp_blocks_freed(victim.request_id)
            self.kv_mgr.free_sequence(victim.request_id)
            victim.transition_to(SequenceStatus.SWAPPED)
            self._swapped.append(victim)
            self._swap_needs_reprefill.add(victim.request_id)
            return

        orig_gpu_ids = [gpu_id for gpu_id, _ in pairs]
        self._swapped_gpu_ids[victim.request_id] = orig_gpu_ids

        if self._transfer_scheduler is not None:
            transfer_req = TransferRequest(
                transfer_id=f"swap_out_{victim.request_id}",
                transfer_type=TransferType.KV_SWAP_OUT,
                priority=TransferPriority.HIGH,
                source_device=f"cuda:{self.device_id}",
                device_id=self.device_id,
                target_device="cpu",
                block_ids=orig_gpu_ids,
            )
            transfer_id = self._transfer_scheduler.enqueue(transfer_req)
            completed = self._transfer_scheduler.wait(
                transfer_id, timeout_ms=5000.0
            )
            if completed:
                self.kv_mgr.commit_swap_out(
                    self.device_id, victim.request_id, pairs
                )
            else:
                _ = self._transfer_scheduler.cancel(transfer_id)
                self.kv_mgr.commit_swap_out(
                    self.device_id, victim.request_id, pairs
                )
                self._notify_cp_blocks_freed(victim.request_id)
                self.kv_mgr.free_sequence(victim.request_id)
                _ = self._swapped_gpu_ids.pop(victim.request_id, None)
                self._swap_needs_reprefill.add(victim.request_id)
        else:
            self.kv_mgr.commit_swap_out(
                self.device_id, victim.request_id, pairs
            )

        victim.transition_to(SequenceStatus.SWAPPED)
        self._swapped.append(victim)

    def _swap_in_request(self, req: Request) -> bool:
        if req.request_id in self._swap_needs_reprefill:
            self._swap_needs_reprefill.discard(req.request_id)
            req.transition_to(SequenceStatus.WAITING)
            self._waiting.appendleft(req)
            return True

        orig_gpu_ids = self._swapped_gpu_ids.get(req.request_id, [])
        pairs = self.kv_mgr.prepare_swap_in(
            self.device_id, req.request_id, orig_gpu_ids
        )
        if not pairs:
            return False

        if self._transfer_scheduler is not None:
            transfer_req = TransferRequest(
                transfer_id=f"swap_in_{req.request_id}",
                transfer_type=TransferType.KV_SWAP_IN,
                priority=TransferPriority.NORMAL,
                source_device="cpu",
                target_device=f"cuda:{self.device_id}",
                device_id=self.device_id,
                block_ids=[cpu_id for cpu_id, _ in pairs],
            )
            transfer_id = self._transfer_scheduler.enqueue(transfer_req)
            completed = self._transfer_scheduler.wait(
                transfer_id, timeout_ms=5000.0
            )
            if completed:
                self.kv_mgr.commit_swap_in(
                    self.device_id, req.request_id, orig_gpu_ids, pairs
                )
                _ = self._swapped_gpu_ids.pop(req.request_id, None)
            else:
                _ = self._transfer_scheduler.cancel(transfer_id)
                self.kv_mgr.commit_swap_in(
                    self.device_id, req.request_id, orig_gpu_ids, pairs
                )
                self._notify_cp_blocks_freed(req.request_id)
                self.kv_mgr.free_sequence(req.request_id)
                _ = self._swapped_gpu_ids.pop(req.request_id, None)
                self._swap_needs_reprefill.add(req.request_id)

                req.transition_to(SequenceStatus.WAITING)
                self._waiting.appendleft(req)
                return True
        else:
            self.kv_mgr.commit_swap_in(
                self.device_id, req.request_id, orig_gpu_ids, pairs
            )
            _ = self._swapped_gpu_ids.pop(req.request_id, None)

        req.transition_to(SequenceStatus.WAITING)
        req.transition_to(SequenceStatus.RUNNING)
        self._running.append(req)
        return True

    def _compute_prompt_block_hashes(
        self, prompt_token_ids: list[int]
    ) -> list[int]:
        from moe_infinity.memory.block_pool import hash_block_tokens

        block_size = self.kv_mgr.block_size
        num_full_blocks = len(prompt_token_ids) // block_size
        parent_hash = 0
        block_hashes: list[int] = []

        for i in range(num_full_blocks):
            token_slice = tuple(
                prompt_token_ids[i * block_size : (i + 1) * block_size]
            )
            block_hash = hash_block_tokens(parent_hash, token_slice)
            block_hashes.append(block_hash)
            parent_hash = block_hash

        return block_hashes

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_swapped(self) -> int:
        return len(self._swapped)

    @property
    def admissions_paused(self) -> bool:
        return self._admissions_paused

    def begin_memory_resize(
        self, device_id: int, timeout_ms: float = 5000.0
    ) -> MemoryResizeToken:
        if device_id != self.device_id:
            raise ValueError("resize device_id must match scheduler owner")
        if self._admissions_paused:
            raise RuntimeError("memory resize is already active")
        token = MemoryResizeToken(
            device_id=device_id,
            waiting_ids=tuple(req.request_id for req in self._waiting),
            running_ids=tuple(req.request_id for req in self._running),
            swapped_ids=tuple(req.request_id for req in self._swapped),
        )
        self._admissions_paused = True
        try:
            while self._running:
                request = self._running.popleft()
                self._preempt_with_transfer(request)
                self._resize_preemptions += 1
            if (
                self._transfer_scheduler is not None
                and not self._transfer_scheduler.wait_for_device(
                    device_id, timeout_ms
                )
            ):
                raise TimeoutError("timed out draining device transfers")
            return token
        except Exception:
            self._admissions_paused = False
            while self._maintenance_backlog:
                self._waiting.append(self._maintenance_backlog.popleft())
            raise

    def end_memory_resize(self, token: MemoryResizeToken) -> None:
        if token.device_id != self.device_id:
            raise ValueError(
                "resize token device_id must match scheduler owner"
            )
        while self._maintenance_backlog:
            self._waiting.append(self._maintenance_backlog.popleft())
        self._admissions_paused = False

    def memory_pressure_snapshot(self) -> dict[str, int | float]:
        used = self.kv_mgr.num_gpu_blocks - self.kv_mgr.num_free_gpu_blocks
        return {
            "kv_used_blocks": used,
            "kv_total_blocks": self.kv_mgr.num_gpu_blocks,
            "kv_preemptions": self._resize_preemptions,
        }
