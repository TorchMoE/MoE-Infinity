from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING, Optional, Protocol

logger = logging.getLogger(__name__)

from .batch import PrefillChunk, SchedulerOutput
from .kv_cache import PagedKVCache
from .sequence import SequenceData, SequenceGroup, SequenceStatus

if TYPE_CHECKING:
    from moe_infinity.runtime.attention_backend import (
        LayeredPagedKVCheckpoint,
    )


@dataclass
class InFlightPrefill:
    transaction_id: int
    seq_id: int
    chunk: PrefillChunk
    prior_table_existed: bool
    prior_reserved_tokens: int
    prior_block_ids: tuple[int, ...]
    checkpoint_block_ids: list[int] = field(default_factory=list)
    write_checkpoint: "LayeredPagedKVCheckpoint | None" = None
    cow_block_ids: tuple[int, ...] = ()
    was_removed_from_ready_queue: bool = False
    prior_num_computed_tokens: int = 0
    prior_committed_kv_tokens: int = 0
    prior_status: SequenceStatus = SequenceStatus.WAITING


class _PrefillRowCommit:
    def __init__(self, scheduler: "Scheduler", lease: InFlightPrefill) -> None:
        self._scheduler = scheduler
        self._lease = lease
        self._prepared = False

    @property
    def seq_id(self) -> int:
        return self._lease.seq_id

    @property
    def chunk(self) -> PrefillChunk:
        return self._lease.chunk

    def prepare_commit(self) -> None:
        lease = self._lease
        sequence = self._scheduler._sequence_map.get(lease.seq_id)
        if sequence is None:
            raise RuntimeError(
                f"prefill row for seq_id={lease.seq_id} has no sequence"
            )
        if lease.chunk.start_pos != sequence.num_computed_tokens:
            raise RuntimeError(
                "prefill chunk start does not match committed progress"
            )
        end = lease.chunk.start_pos + lease.chunk.num_tokens
        if end > sequence.prompt_length:
            raise RuntimeError("prefill chunk exceeds prompt length")
        if self._scheduler.kv_cache.get_num_reserved_tokens(lease.seq_id) < end:
            raise RuntimeError("prefill chunk is not fully reserved")
        lease.prior_num_computed_tokens = sequence.num_computed_tokens
        lease.prior_committed_kv_tokens = sequence.committed_kv_tokens
        self._prepared = True

    def commit(self) -> None:
        lease = self._lease
        sequence = self._scheduler._sequence_map[lease.seq_id]
        sequence.advance_prefill(lease.chunk.num_tokens)

    def abort(self) -> None:
        if not self._prepared:
            return
        lease = self._lease
        sequence = self._scheduler._sequence_map.get(lease.seq_id)
        if sequence is not None:
            sequence.num_computed_tokens = lease.prior_num_computed_tokens
            sequence.committed_kv_tokens = lease.prior_committed_kv_tokens
        self._prepared = False


class CPAwareKVManager(Protocol):
    def predict_prefix_reuse(
        self, request_id: str, token_ids: list[int]
    ) -> float: ...


@dataclass(frozen=True)
class Deficit2D:
    tokens: int
    expert_bytes: int


@dataclass(frozen=True)
class VerifyDemand:
    seq_id: int
    tokens: int
    expert_bytes: int
    in_flight: bool = False


@dataclass(frozen=True)
class VerifyAdmission:
    seated_ids: tuple[int, ...]
    admitted_ids: tuple[int, ...]
    carried: Deficit2D


def _reject_negative(label: str, deficit: Deficit2D) -> None:
    if deficit.tokens < 0 or deficit.expert_bytes < 0:
        raise ValueError(
            f"{label} dimensions must be non-negative; got {deficit}"
        )


def admit_verify_demands(
    demands: list[VerifyDemand],
    budget: Deficit2D,
    carried: Deficit2D,
    deficit_cap: Deficit2D,
) -> VerifyAdmission:
    _reject_negative("budget", budget)
    _reject_negative("carried", carried)
    _reject_negative("deficit_cap", deficit_cap)
    for demand in demands:
        if demand.tokens < 0 or demand.expert_bytes < 0:
            raise ValueError(
                f"demand {demand.seq_id} dimensions must be non-negative; "
                f"got tokens={demand.tokens}, "
                f"expert_bytes={demand.expert_bytes}"
            )

    if demands:
        largest_tokens = max(demand.tokens for demand in demands)
        largest_bytes = max(demand.expert_bytes for demand in demands)
        if (
            deficit_cap.tokens < largest_tokens
            or deficit_cap.expert_bytes < largest_bytes
        ):
            raise ValueError(
                "deficit_cap must be >= the largest single demand in each "
                f"dimension; cap={deficit_cap}, largest_tokens="
                f"{largest_tokens}, largest_expert_bytes={largest_bytes}"
            )

    seated_ids = tuple(demand.seq_id for demand in demands if demand.in_flight)
    seated_tokens = sum(demand.tokens for demand in demands if demand.in_flight)
    seated_bytes = sum(
        demand.expert_bytes for demand in demands if demand.in_flight
    )

    pool_tokens = budget.tokens + carried.tokens - seated_tokens
    pool_bytes = budget.expert_bytes + carried.expert_bytes - seated_bytes

    admitted_ids: list[int] = []
    for demand in demands:
        if demand.in_flight:
            continue
        if demand.tokens <= pool_tokens and demand.expert_bytes <= pool_bytes:
            pool_tokens -= demand.tokens
            pool_bytes -= demand.expert_bytes
            admitted_ids.append(demand.seq_id)

    carried_out = Deficit2D(
        tokens=min(max(0, pool_tokens), deficit_cap.tokens),
        expert_bytes=min(max(0, pool_bytes), deficit_cap.expert_bytes),
    )
    return VerifyAdmission(
        seated_ids=seated_ids,
        admitted_ids=tuple(admitted_ids),
        carried=carried_out,
    )


@dataclass(frozen=True)
class _VerifyConfig:
    enabled: bool
    budget: Deficit2D
    deficit_cap: Deficit2D


def _resolve_verify_config(
    *,
    token_budget: Optional[int],
    expert_byte_budget: Optional[int],
    token_deficit_cap: Optional[int],
    expert_byte_deficit_cap: Optional[int],
) -> _VerifyConfig:
    provided = [
        token_budget,
        expert_byte_budget,
        token_deficit_cap,
        expert_byte_deficit_cap,
    ]
    if all(value is None for value in provided):
        zero = Deficit2D(tokens=0, expert_bytes=0)
        return _VerifyConfig(enabled=False, budget=zero, deficit_cap=zero)
    if any(value is None for value in provided):
        raise ValueError(
            "verify scheduling requires all four of token_budget, "
            "expert_byte_budget, token_deficit_cap, and "
            "expert_byte_deficit_cap, or none of them"
        )
    for name, value in (
        ("verify_token_budget", token_budget),
        ("verify_expert_byte_budget", expert_byte_budget),
        ("verify_token_deficit_cap", token_deficit_cap),
        ("verify_expert_byte_deficit_cap", expert_byte_deficit_cap),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    return _VerifyConfig(
        enabled=True,
        budget=Deficit2D(
            tokens=int(token_budget), expert_bytes=int(expert_byte_budget)
        ),
        deficit_cap=Deficit2D(
            tokens=int(token_deficit_cap),
            expert_bytes=int(expert_byte_deficit_cap),
        ),
    )


class Scheduler:
    kv_cache: PagedKVCache
    max_batch_size: int
    max_tokens_per_step: int

    def __init__(
        self,
        kv_cache: PagedKVCache,
        max_batch_size: int = 32,
        max_tokens_per_step: int = 2048,
        *,
        enable_chunked_prefill: bool = False,
        prefill_chunk_size: int = 512,
        prefill_starvation_threshold_steps: int = 8,
        verify_token_budget: Optional[int] = None,
        verify_expert_byte_budget: Optional[int] = None,
        verify_token_deficit_cap: Optional[int] = None,
        verify_expert_byte_deficit_cap: Optional[int] = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be > 0, got {max_batch_size}"
            )
        if max_tokens_per_step <= 0:
            raise ValueError(
                f"max_tokens_per_step must be > 0, got {max_tokens_per_step}"
            )
        if prefill_chunk_size <= 0:
            raise ValueError(
                f"prefill_chunk_size must be > 0, got {prefill_chunk_size}"
            )
        if prefill_starvation_threshold_steps <= 0:
            raise ValueError(
                "prefill_starvation_threshold_steps must be > 0, got "
                f"{prefill_starvation_threshold_steps}"
            )

        self.kv_cache = kv_cache
        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step

        self.chunked_prefill_requested = bool(enable_chunked_prefill)
        self.chunked_prefill_enabled = bool(enable_chunked_prefill)
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_starvation_threshold_steps = (
            prefill_starvation_threshold_steps
        )
        self._prefill_queue: deque[int] = deque()
        self._prefill_wait_steps: dict[int, int] = {}
        self._swapped_resume_status: dict[int, SequenceStatus] = {}
        self._inflight_prefill: dict[int, InFlightPrefill] = {}
        self._next_prefill_transaction_id = 0
        self._schedule_steps = 0
        self._completed_prefill_transactions: set[int] = set()

        self._waiting: deque[SequenceGroup] = deque()
        self._running: deque[SequenceGroup] = deque()
        self._swapped: deque[SequenceGroup] = deque()

        self._sequence_map: dict[int, SequenceData] = {}
        self._request_map: dict[str, SequenceGroup] = {}
        self._cp_kv_manager: Optional[CPAwareKVManager] = None

        self._verify_config = _resolve_verify_config(
            token_budget=verify_token_budget,
            expert_byte_budget=verify_expert_byte_budget,
            token_deficit_cap=verify_token_deficit_cap,
            expert_byte_deficit_cap=verify_expert_byte_deficit_cap,
        )
        self._verify_demands: dict[int, VerifyDemand] = {}
        self._carried_verify_deficit = Deficit2D(tokens=0, expert_bytes=0)

    def set_cp_kv_manager(self, manager: CPAwareKVManager) -> None:
        self._cp_kv_manager = manager

    @property
    def verify_scheduling_enabled(self) -> bool:
        """True iff all four verify budgets were configured (Step-5 opt-in)."""
        return self._verify_config.enabled

    def add_request(self, seq_group: SequenceGroup) -> None:
        if seq_group.request_id in self._request_map:
            raise ValueError(
                f"request_id '{seq_group.request_id}' already exists"
            )

        for sequence in seq_group.sequences:
            if sequence.seq_id in self._sequence_map:
                raise ValueError(
                    f"sequence id {sequence.seq_id} already exists"
                )
            if sequence.status is not SequenceStatus.WAITING:
                sequence.set_status(SequenceStatus.WAITING)
            self._sequence_map[sequence.seq_id] = sequence

        self._request_map[seq_group.request_id] = seq_group
        self._waiting.append(seq_group)
        for sequence in seq_group.sequences:
            self._enqueue_prefill_once(sequence.seq_id)

    @property
    def inflight_prefill_seq_ids(self) -> list[int]:
        return list(self._inflight_prefill)

    def set_chunked_prefill_runtime_enabled(self, enabled: bool) -> None:
        self.chunked_prefill_enabled = self.chunked_prefill_requested and (
            bool(enabled)
        )

    def schedule(self) -> SchedulerOutput:
        if not self.chunked_prefill_enabled:
            return self._schedule_whole_prefill()
        return self._schedule_chunked_prefill()

    def _schedule_whole_prefill(self) -> SchedulerOutput:
        output = SchedulerOutput()
        swapped_snapshot = list(self._swapped)

        scheduled_seqs = 0
        scheduled_tokens = 0

        self._recover_swapped_groups(swapped_snapshot)

        if self._cp_kv_manager is not None and len(self._waiting) > 1:
            scored_waiting = [
                (
                    self._cp_kv_manager.predict_prefix_reuse(
                        group.request_id,
                        self._group_token_ids(group),
                    ),
                    group,
                )
                for group in self._waiting
            ]
            scored_waiting.sort(key=lambda x: x[0], reverse=True)
            self._waiting = deque(group for _, group in scored_waiting)

        waiting_blocked = False
        while self._waiting and not waiting_blocked:
            next_group = self._waiting[0]
            next_seqs = [
                sequence
                for sequence in next_group.sequences
                if sequence.status is SequenceStatus.WAITING
            ]

            if not next_seqs:
                _ = self._waiting.popleft()
                continue

            prefill_tokens = sum(
                self._num_prefill_tokens(sequence) for sequence in next_seqs
            )

            if scheduled_seqs + len(next_seqs) > self.max_batch_size:
                break
            if scheduled_tokens + prefill_tokens > self.max_tokens_per_step:
                break

            required_blocks = sum(
                self._required_blocks(sequence) for sequence in next_seqs
            )

            if self.kv_cache.block_allocator.num_free_blocks < required_blocks:
                preempted_seq_ids = self._preempt_oldest_running_group()
                if not preempted_seq_ids:
                    waiting_blocked = True
                    continue
                output.preempted_seq_ids.extend(preempted_seq_ids)

                if (
                    self.kv_cache.block_allocator.num_free_blocks
                    < required_blocks
                ):
                    waiting_blocked = True
                    continue

            _ = self._waiting.popleft()
            self._running.append(next_group)

            for sequence in next_seqs:
                self.kv_cache.allocate_sequence(
                    sequence.seq_id,
                    num_tokens=sequence.prompt_length,
                )
                sequence.set_status(SequenceStatus.PREFILL)
                output.prefill_seq_ids.append(sequence.seq_id)

            scheduled_seqs += len(next_seqs)
            scheduled_tokens += prefill_tokens
            output.num_prefill_tokens += prefill_tokens

        capacity_reached = False
        for group in self._running:
            if capacity_reached:
                break
            for sequence in group.sequences:
                if sequence.status is not SequenceStatus.DECODE:
                    continue
                if (
                    scheduled_seqs >= self.max_batch_size
                    or scheduled_tokens >= self.max_tokens_per_step
                ):
                    capacity_reached = True
                    break

                output.decode_seq_ids.append(sequence.seq_id)
                output.num_decode_tokens += 1
                scheduled_seqs += 1
                scheduled_tokens += 1

        self._apply_verify_scheduling(output)
        return output

    def _schedule_chunked_prefill(self) -> SchedulerOutput:
        output = SchedulerOutput()
        self._schedule_steps += 1
        self._recover_swapped_groups(list(self._swapped))

        scheduled_rows = 0
        scheduled_tokens = 0
        for group in self._running:
            for sequence in group.sequences:
                if sequence.status is not SequenceStatus.DECODE:
                    continue
                if scheduled_rows >= self.max_batch_size:
                    break
                if scheduled_tokens + 1 > self.max_tokens_per_step:
                    break
                output.decode_seq_ids.append(sequence.seq_id)
                output.num_decode_tokens += 1
                scheduled_rows += 1
                scheduled_tokens += 1

        available_rows = self.max_batch_size - scheduled_rows
        available_tokens = self.max_tokens_per_step - scheduled_tokens
        if available_rows <= 0 or available_tokens <= 0:
            self._age_unscheduled_prefills(set())
            self._apply_verify_scheduling(output)
            return output

        ordered = self._ordered_prefill_candidates()
        selected: set[int] = set()
        transaction_id = self._next_prefill_transaction_id
        self._next_prefill_transaction_id += 1
        try:
            for seq_id in ordered:
                if available_rows <= 0 or available_tokens <= 0:
                    break
                sequence = self._sequence_map.get(seq_id)
                if sequence is None or sequence.status not in {
                    SequenceStatus.WAITING,
                    SequenceStatus.PREFILL,
                }:
                    continue
                count = min(
                    self.prefill_chunk_size,
                    sequence.remaining_prefill_tokens,
                    available_tokens,
                )
                if count <= 0:
                    continue
                end = sequence.num_computed_tokens + count
                required_blocks = self._incremental_required_blocks(
                    sequence, end
                )
                if (
                    self.kv_cache.block_allocator.num_free_blocks
                    < required_blocks
                ):
                    continue
                chunk = PrefillChunk(
                    start_pos=sequence.num_computed_tokens,
                    num_tokens=count,
                    is_terminal=end == sequence.prompt_length,
                )
                existed = self.kv_cache.has_sequence(seq_id)
                prior_reserved = (
                    self.kv_cache.get_num_reserved_tokens(seq_id)
                    if existed
                    else 0
                )
                prior_block_ids = (
                    tuple(self.kv_cache.get_block_table(seq_id))
                    if existed
                    else ()
                )

                self.kv_cache.ensure_sequence_capacity(seq_id, end)
                lease = InFlightPrefill(
                    transaction_id=transaction_id,
                    seq_id=seq_id,
                    chunk=chunk,
                    prior_table_existed=existed,
                    prior_reserved_tokens=prior_reserved,
                    prior_block_ids=prior_block_ids,
                    prior_status=sequence.status,
                )
                self._inflight_prefill[seq_id] = lease
                cow = self.kv_cache.ensure_writable_range(
                    seq_id, chunk.start_pos, end
                )
                lease.cow_block_ids = cow.new_block_ids
                touched = self.kv_cache.get_block_ids_for_range(
                    seq_id, chunk.start_pos, end
                )
                lease.checkpoint_block_ids = touched
                lease.write_checkpoint = self.kv_cache.block_store.checkpoint(
                    touched
                )

                output.prefill_seq_ids.append(seq_id)
                output.prefill_chunks[seq_id] = chunk
                output.num_prefill_tokens += count
                selected.add(seq_id)
                available_rows -= 1
                available_tokens -= count
        except BaseException:
            self._rollback_prepared_transaction(transaction_id)
            raise

        if output.prefill_chunks:
            for seq_id in output.prefill_seq_ids:
                sequence = self._sequence_map[seq_id]
                if sequence.status is SequenceStatus.WAITING:
                    sequence.set_status(SequenceStatus.PREFILL)
                    self._move_request_to_running(seq_id)
            output.prefill_transaction_id = transaction_id
        self._remove_inflight_from_ready_queue(output.prefill_chunks)
        for seq_id in output.prefill_seq_ids:
            self._inflight_prefill[seq_id].was_removed_from_ready_queue = True
        self._age_unscheduled_prefills(selected)
        self._apply_verify_scheduling(output)
        return output

    def commit_prefill_step(self, transaction_id: int | None) -> None:
        rows = self._leases_for_transaction(transaction_id)
        row_commits = [_PrefillRowCommit(self, row) for row in rows]
        try:
            for participant in row_commits:
                participant.prepare_commit()
        except BaseException as original:
            rollback_errors: list[BaseException] = []
            for participant in reversed(row_commits):
                try:
                    participant.abort()
                except BaseException as rollback_error:  # noqa: BLE001
                    rollback_errors.append(rollback_error)
            try:
                self._rollback_prepared_transaction(transaction_id)
            except BaseException as rollback_error:  # noqa: BLE001
                rollback_errors.append(rollback_error)
            if rollback_errors:
                details = "; ".join(repr(error) for error in rollback_errors)
                raise RuntimeError(
                    f"{original}; transaction rollback errors: {details}"
                ) from original
            raise
        for participant in row_commits:
            participant.commit()
        for row in rows:
            self._inflight_prefill.pop(row.seq_id, None)
            if not row.chunk.is_terminal:
                self._enqueue_prefill_once(row.seq_id)
        self._completed_prefill_transactions.add(int(transaction_id))
        if len(self._completed_prefill_transactions) > 4096:
            self._completed_prefill_transactions = {
                tid
                for tid in self._completed_prefill_transactions
                if tid > self._next_prefill_transaction_id - 2048
            }

    def rollback_prefill_step(self, transaction_id: int | None) -> None:
        leases = self._leases_for_transaction(transaction_id)
        self._restore_leases(leases)

    def _restore_leases(self, leases: list[InFlightPrefill]) -> None:
        for lease in reversed(leases):
            if lease.write_checkpoint is not None:
                self.kv_cache.block_store.restore(
                    lease.checkpoint_block_ids, lease.write_checkpoint
                )
            self.kv_cache.rollback_sequence_reservation(
                seq_id=lease.seq_id,
                prior_table_existed=lease.prior_table_existed,
                prior_block_ids=lease.prior_block_ids,
                prior_reserved_tokens=lease.prior_reserved_tokens,
                cow_block_ids=lease.cow_block_ids,
            )
            self._inflight_prefill.pop(lease.seq_id, None)
            sequence = self._sequence_map.get(lease.seq_id)
            if sequence is not None and lease.prior_status is (
                SequenceStatus.WAITING
            ):
                if sequence.status is SequenceStatus.PREFILL:
                    sequence.set_status(SequenceStatus.WAITING)
            if (
                lease.was_removed_from_ready_queue
                and sequence is not None
                and lease.seq_id not in self._prefill_queue
            ):
                self._prefill_queue.appendleft(lease.seq_id)

    def _rollback_prepared_transaction(
        self, transaction_id: int | None
    ) -> None:
        leases = self._leases_for_transaction(transaction_id)
        self._restore_leases(leases)

    def _leases_for_transaction(
        self, transaction_id: int | None
    ) -> list[InFlightPrefill]:
        if transaction_id is None or (
            int(transaction_id) in self._completed_prefill_transactions
        ):
            raise RuntimeError(f"unknown prefill transaction {transaction_id}")
        leases = [
            lease
            for lease in self._inflight_prefill.values()
            if lease.transaction_id == int(transaction_id)
        ]
        if not leases:
            raise RuntimeError(f"unknown prefill transaction {transaction_id}")
        return leases

    def _ordered_prefill_candidates(self) -> list[int]:
        unique = list(dict.fromkeys(self._prefill_queue))
        live: list[int] = []
        for seq_id in unique:
            sequence = self._sequence_map.get(seq_id)
            if (
                sequence is not None
                and sequence.status
                in {SequenceStatus.WAITING, SequenceStatus.PREFILL}
                and sequence.remaining_prefill_tokens > 0
                and seq_id not in self._inflight_prefill
            ):
                live.append(seq_id)
        aged = [
            seq_id
            for seq_id in live
            if self._prefill_wait_steps.get(seq_id, 0)
            >= self.prefill_starvation_threshold_steps
        ]
        aged_set = set(aged)
        return [*aged, *(seq_id for seq_id in live if seq_id not in aged_set)]

    def _remove_inflight_from_ready_queue(
        self, chunks: dict[int, PrefillChunk]
    ) -> None:
        selected = set(chunks)
        self._prefill_queue = deque(
            seq_id
            for seq_id in dict.fromkeys(self._prefill_queue)
            if seq_id not in selected and seq_id in self._sequence_map
        )

    def _age_unscheduled_prefills(self, selected: set[int]) -> None:
        for seq_id in self._ordered_prefill_candidates():
            if seq_id in selected:
                self._prefill_wait_steps[seq_id] = 0
            else:
                self._prefill_wait_steps[seq_id] = (
                    self._prefill_wait_steps.get(seq_id, 0) + 1
                )

    def _move_request_to_running(self, seq_id: int) -> None:
        request_id = next(
            request_id
            for request_id, group in self._request_map.items()
            if seq_id in group.sequence_ids
        )
        group = self._request_map[request_id]
        if group in self._waiting:
            self._waiting.remove(group)
        if group not in self._running:
            self._running.append(group)

    def _incremental_required_blocks(
        self, sequence: SequenceData, end: int
    ) -> int:
        current = (
            len(self.kv_cache.get_block_table(sequence.seq_id))
            if self.kv_cache.has_sequence(sequence.seq_id)
            else 0
        )
        return max(0, ceil(end / self.kv_cache.block_size) - current)

    def _enqueue_prefill_once(self, seq_id: int) -> None:
        if seq_id not in self._prefill_queue:
            self._prefill_queue.append(seq_id)
        self._prefill_wait_steps.setdefault(seq_id, 0)

    def set_verify_demand(
        self,
        seq_id: int,
        tokens: int,
        expert_bytes: int,
        in_flight: bool,
    ) -> None:
        self._verify_demands[seq_id] = VerifyDemand(
            seq_id=seq_id,
            tokens=tokens,
            expert_bytes=expert_bytes,
            in_flight=in_flight,
        )

    def clear_verify_demand(self, seq_id: int) -> None:
        _ = self._verify_demands.pop(seq_id, None)

    @property
    def carried_verify_deficit(self) -> Deficit2D:
        return self._carried_verify_deficit

    def _apply_verify_scheduling(self, output: SchedulerOutput) -> None:
        if not self._verify_config.enabled or not self._verify_demands:
            return

        demands = list(self._verify_demands.values())
        admission = admit_verify_demands(
            demands,
            self._verify_config.budget,
            self._carried_verify_deficit,
            self._verify_config.deficit_cap,
        )
        self._carried_verify_deficit = admission.carried

        verify_ids = [*admission.seated_ids, *admission.admitted_ids]
        verify_set = set(verify_ids)
        output.verify_seq_ids = list(verify_ids)
        output.draft_seq_ids = [
            demand.seq_id
            for demand in demands
            if demand.seq_id not in verify_set
        ]
        output.num_verify_tokens = sum(
            demand.tokens for demand in demands if demand.seq_id in verify_set
        )
        output.num_verify_expert_bytes = sum(
            demand.expert_bytes
            for demand in demands
            if demand.seq_id in verify_set
        )

    def update_after_step(
        self,
        completed_seq_ids: list[int],
        new_decode_seq_ids: list[int],
        committed_counts: dict[int, int] | None = None,
    ) -> None:
        completed = set(completed_seq_ids)

        for seq_id in dict.fromkeys(new_decode_seq_ids):
            sequence = self._sequence_map.get(seq_id)
            if sequence is None or seq_id in completed:
                continue

            if sequence.status is SequenceStatus.PREFILL:
                sequence.set_status(SequenceStatus.DECODE)

            if sequence.status is SequenceStatus.DECODE:
                num_new = (
                    1
                    if committed_counts is None
                    else committed_counts.get(seq_id, 1)
                )
                try:
                    self.kv_cache.append_tokens(seq_id, num_new_tokens=num_new)
                    target_tokens = sequence.num_computed_tokens + 1
                    if target_tokens > self.kv_cache.get_num_reserved_tokens(
                        seq_id
                    ):
                        self.kv_cache.ensure_sequence_capacity(
                            seq_id, target_tokens
                        )
                except KeyError:
                    pass

        for seq_id in completed:
            sequence = self._sequence_map.get(seq_id)
            if sequence is None:
                continue

            if sequence.status in (
                SequenceStatus.WAITING,
                SequenceStatus.PREFILL,
                SequenceStatus.DECODE,
                SequenceStatus.DRAFT,
                SequenceStatus.VERIFY,
                SequenceStatus.SWAPPED,
            ):
                sequence.set_status(SequenceStatus.FINISHED)
            self.kv_cache.free_sequence(seq_id)

        self._prune_finished_and_cancelled_requests()

    def abort_request(self, request_id: str) -> None:
        group = self._request_map.get(request_id)
        if group is None:
            return

        for sequence in group.sequences:
            lease = self._inflight_prefill.get(sequence.seq_id)
            if lease is not None:
                self._restore_leases([lease])
            if sequence.status not in (
                SequenceStatus.FINISHED,
                SequenceStatus.CANCELLED,
            ):
                sequence.set_status(SequenceStatus.CANCELLED)
            self.kv_cache.free_sequence(sequence.seq_id)

        self._waiting = deque(
            queued
            for queued in self._waiting
            if queued.request_id != request_id
        )
        self._running = deque(
            queued
            for queued in self._running
            if queued.request_id != request_id
        )
        self._swapped = deque(
            queued
            for queued in self._swapped
            if queued.request_id != request_id
        )

        self._drop_request_metadata(group)
        self._prune_finished_and_cancelled_requests()

    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    def get_running_seq_ids(self) -> list[int]:
        running_seq_ids: list[int] = []
        for group in self._running:
            for sequence in group.sequences:
                if sequence.status in (
                    SequenceStatus.PREFILL,
                    SequenceStatus.DECODE,
                    SequenceStatus.DRAFT,
                    SequenceStatus.VERIFY,
                ):
                    running_seq_ids.append(sequence.seq_id)
        return running_seq_ids

    def _preempt_oldest_running_group(self) -> list[int]:
        preserved: list[SequenceGroup] = []
        while self._running:
            group = self._running.popleft()
            active = [
                sequence
                for sequence in group.sequences
                if sequence.status
                not in (SequenceStatus.FINISHED, SequenceStatus.CANCELLED)
            ]
            if not active or any(
                sequence.status
                not in (SequenceStatus.PREFILL, SequenceStatus.DECODE)
                for sequence in active
            ):
                preserved.append(group)
                continue
            preempted_seq_ids: list[int] = []

            for sequence in group.sequences:
                if sequence.status not in (
                    SequenceStatus.PREFILL,
                    SequenceStatus.DECODE,
                ):
                    continue

                if self.chunked_prefill_enabled:
                    self._swapped_resume_status[sequence.seq_id] = (
                        sequence.status
                    )
            for sequence in active:
                try:
                    self.kv_cache.swap_out(sequence.seq_id)
                except KeyError:
                    pass
                self.kv_cache.free_gpu_blocks(sequence.seq_id)
                sequence.set_status(SequenceStatus.SWAPPED)
                preempted_seq_ids.append(sequence.seq_id)

            if preempted_seq_ids:
                self._swapped.append(group)
                self._running.extendleft(reversed(preserved))
                return preempted_seq_ids

        self._running.extend(preserved)
        return []

    def _recover_swapped_groups(
        self, swapped_groups: list[SequenceGroup]
    ) -> None:
        if self.kv_cache.block_allocator.num_free_blocks <= 0:
            return

        for group in swapped_groups:
            if group not in self._swapped:
                continue

            swapped_sequences = [
                sequence
                for sequence in group.sequences
                if sequence.status is SequenceStatus.SWAPPED
            ]
            if not swapped_sequences:
                _ = self._swapped.remove(group)
                continue

            recovered = True
            for sequence in swapped_sequences:
                try:
                    self.kv_cache.swap_in(sequence.seq_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "swap_in failed for seq_id=%s: %s",
                        sequence.seq_id,
                        exc,
                    )
                    recovered = False
                    break

            if not recovered:
                continue

            for sequence in swapped_sequences:
                resume = self._swapped_resume_status.pop(
                    sequence.seq_id, SequenceStatus.DECODE
                )
                sequence.set_status(resume)
                if resume is SequenceStatus.PREFILL:
                    self._enqueue_prefill_once(sequence.seq_id)

            _ = self._swapped.remove(group)
            self._running.appendleft(group)

    def _prune_finished_and_cancelled_requests(self) -> None:
        active_requests: dict[str, SequenceGroup] = {}
        finished_groups: list[SequenceGroup] = []

        for group in list(self._request_map.values()):
            has_active = False
            for sequence in group.sequences:
                if sequence.status not in (
                    SequenceStatus.FINISHED,
                    SequenceStatus.CANCELLED,
                ):
                    has_active = True
                    break

            if has_active:
                active_requests[group.request_id] = group
            else:
                finished_groups.append(group)

        for group in finished_groups:
            self._drop_request_metadata(group)

        self._request_map = active_requests
        self._waiting = deque(
            group
            for group in self._waiting
            if group.request_id in self._request_map
        )
        self._running = deque(
            group
            for group in self._running
            if group.request_id in self._request_map
        )
        self._swapped = deque(
            group
            for group in self._swapped
            if group.request_id in self._request_map
        )

    def _drop_request_metadata(self, group: SequenceGroup) -> None:
        _ = self._request_map.pop(group.request_id, None)
        for sequence in group.sequences:
            seq_id = sequence.seq_id
            _ = self._sequence_map.pop(seq_id, None)
            _ = self._prefill_wait_steps.pop(seq_id, None)
            _ = self._swapped_resume_status.pop(seq_id, None)
            _ = self._inflight_prefill.pop(seq_id, None)
            _ = self._verify_demands.pop(seq_id, None)
            if seq_id in self._prefill_queue:
                self._prefill_queue = deque(
                    queued for queued in self._prefill_queue if queued != seq_id
                )

    def _required_blocks(self, sequence: SequenceData) -> int:
        if sequence.prompt_length == 0:
            return 0
        return ceil(sequence.prompt_length / self.kv_cache.block_size)

    @staticmethod
    def _num_prefill_tokens(sequence: SequenceData) -> int:
        return max(0, sequence.prompt_length - sequence.num_computed_tokens)

    @staticmethod
    def _group_token_ids(group: SequenceGroup) -> list[int]:
        token_ids: list[int] = []
        for sequence in group.sequences:
            token_ids.extend(sequence.prompt_token_ids)
        return token_ids


RequestScheduler = Scheduler


__all__ = [
    "Deficit2D",
    "InFlightPrefill",
    "RequestScheduler",
    "Scheduler",
    "VerifyAdmission",
    "VerifyDemand",
    "admit_verify_demands",
]
