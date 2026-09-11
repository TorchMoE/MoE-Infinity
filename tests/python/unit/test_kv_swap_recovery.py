from typing import cast

import torch

from moe_infinity.engine.kv_transfer import (
    CopyTicket,
    KVTransferState,
    PinnedBufferPool,
)
from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.scheduler import Scheduler
from moe_infinity.serving.sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
    SequenceStatus,
)


class _FakeEvent:
    def __init__(self) -> None:
        self.done = False

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.done = True


class _FakeAsyncBackend:
    """Async KV transfer backend with manual per-transfer completion.

    ``complete_next`` flips the oldest not-yet-done event, letting a test
    advance one D2H/H2D at a time. ``fail_seq_ids`` forces the matching
    ``submit_d2h`` to raise, exercising partial-failure rollback.
    """

    asynchronous = True

    def __init__(
        self,
        fail_d2h_calls: set[int] | None = None,
        fail_all_h2d: bool = False,
    ) -> None:
        self.events: list[_FakeEvent] = []
        self.fail_d2h_calls: set[int] = set(fail_d2h_calls or ())
        self.fail_all_h2d = fail_all_h2d
        self._d2h_calls = 0

    def submit_d2h(
        self,
        source_cache: torch.Tensor,
        destination: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        self._d2h_calls += 1
        if self._d2h_calls in self.fail_d2h_calls:
            raise RuntimeError(f"forced D2H failure on call {self._d2h_calls}")
        if block_ids:
            source = source_cache.index_select(
                block_dim, torch.tensor(block_ids, dtype=torch.long)
            )
            destination.copy_(source)
            nbytes = source.numel() * source.element_size()
        else:
            nbytes = 0
        event = _FakeEvent()
        self.events.append(event)
        return CopyTicket(
            device=source_cache.device,
            stream=None,
            event=event,
            owned_staging_tensors=(),
            submitted_ns=1,
            nbytes=nbytes,
        )

    def submit_h2d(
        self,
        source: torch.Tensor,
        destination_cache: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        if self.fail_all_h2d:
            raise RuntimeError("forced H2D failure")
        if block_ids:
            destination_cache.index_copy_(
                block_dim,
                torch.tensor(block_ids, dtype=torch.long),
                source,
            )
            nbytes = source.numel() * source.element_size()
        else:
            nbytes = 0
        event = _FakeEvent()
        self.events.append(event)
        return CopyTicket(
            device=destination_cache.device,
            stream=None,
            event=event,
            owned_staging_tensors=(),
            submitted_ns=1,
            nbytes=nbytes,
        )

    def complete_next(self) -> bool:
        for event in self.events:
            if not event.done:
                event.done = True
                return True
        return False

    def complete_all(self) -> int:
        completed = 0
        for event in self.events:
            if not event.done:
                event.done = True
                completed += 1
        return completed

    def close(self) -> None:
        return None


def _make_async_cache(
    num_blocks: int, backend: _FakeAsyncBackend
) -> PagedKVCache:
    pool = PinnedBufferPool(
        capacity_bytes=1 << 20,
        allocator=lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
        transfer_backend=backend,
        pinned_pool=pool,
    )


def _make_async_scheduler(
    num_blocks: int, backend: _FakeAsyncBackend
) -> Scheduler:
    return Scheduler(
        kv_cache=_make_async_cache(num_blocks, backend),
        max_batch_size=4,
        max_tokens_per_step=64,
    )


def _multi_group(
    request_id: str, seq_ids: list[int], prompt_len: int = 8
) -> SequenceGroup:
    return SequenceGroup(
        request_id=request_id,
        sequences=[
            SequenceData(
                seq_id=seq_id,
                prompt_token_ids=list(range(prompt_len)),
                sampling_params=SamplingParams(),
            )
            for seq_id in seq_ids
        ],
    )


def _only_queue(scheduler: Scheduler, request_id: str) -> str:
    queues = {
        "waiting": any(g.request_id == request_id for g in scheduler._waiting),
        "running": any(g.request_id == request_id for g in scheduler._running),
        "swapped": any(g.request_id == request_id for g in scheduler._swapped),
    }
    present = [name for name, hit in queues.items() if hit]
    assert (
        len(present) == 1
    ), f"request {request_id} must be in exactly one queue, got {present}"
    return present[0]


def _make_kv_cache(num_blocks: int) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )


def _make_scheduler(num_blocks: int) -> Scheduler:
    return Scheduler(
        kv_cache=_make_kv_cache(num_blocks),
        max_batch_size=4,
        max_tokens_per_step=64,
    )


def _make_group(
    request_id: str, seq_id: int, prompt_len: int = 8
) -> SequenceGroup:
    return SequenceGroup(
        request_id=request_id,
        sequences=[
            SequenceData(
                seq_id=seq_id,
                prompt_token_ids=list(range(prompt_len)),
                sampling_params=SamplingParams(),
            )
        ],
    )


def test_swap_recovery_lifecycle() -> None:
    scheduler = _make_scheduler(num_blocks=4)
    group1 = _make_group("req-1", seq_id=1, prompt_len=4)
    scheduler.add_request(group1)

    first = scheduler.schedule()
    assert first.prefill_seq_ids == [1]
    scheduler.update_after_step(completed_seq_ids=[], new_decode_seq_ids=[1])
    assert group1.sequences[0].status is SequenceStatus.DECODE

    group2 = _make_group("req-2", seq_id=2, prompt_len=12)
    scheduler.add_request(group2)

    preempt_cycle = scheduler.schedule()
    assert 1 in preempt_cycle.preempted_seq_ids
    assert group1.sequences[0].status is SequenceStatus.SWAPPED

    scheduler.update_after_step(completed_seq_ids=[2], new_decode_seq_ids=[])

    recovery_cycle = scheduler.schedule()
    assert group1.sequences[0].status is SequenceStatus.DECODE
    assert 1 in recovery_cycle.decode_seq_ids


def test_no_recovery_when_gpu_full() -> None:
    scheduler = _make_scheduler(num_blocks=2)
    group1 = _make_group("req-1", seq_id=10)
    scheduler.add_request(group1)

    first = scheduler.schedule()
    assert first.prefill_seq_ids == [10]
    assert group1.sequences[0].status is SequenceStatus.PREFILL

    group2 = _make_group("req-2", seq_id=20)
    scheduler.add_request(group2)

    preempt_cycle = scheduler.schedule()
    assert 10 in preempt_cycle.preempted_seq_ids
    assert group1.sequences[0].status is SequenceStatus.SWAPPED

    no_recovery_cycle = scheduler.schedule()
    assert no_recovery_cycle.decode_seq_ids == []
    assert group1.sequences[0].status is SequenceStatus.SWAPPED


def _build_pressure_case(
    backend: _FakeAsyncBackend,
) -> tuple[Scheduler, SequenceGroup, SequenceGroup]:
    scheduler = _make_async_scheduler(num_blocks=2, backend=backend)
    first = _make_group("req-1", seq_id=1, prompt_len=4)
    second = _make_group("req-2", seq_id=2, prompt_len=8)
    return scheduler, first, second


def test_preemption_does_not_reuse_evicting_blocks() -> None:
    backend = _FakeAsyncBackend()
    scheduler, first, second = _build_pressure_case(backend)
    scheduler.add_request(first)
    scheduler.schedule()
    scheduler.update_after_step([], [first.sequence_ids[0]])
    scheduler.add_request(second)

    output = scheduler.schedule()
    assert output.preempted_seq_ids == [first.sequence_ids[0]]
    assert output.prefill_seq_ids == []
    assert scheduler.kv_cache.block_allocator.num_free_blocks == 0

    backend.complete_next()
    output = scheduler.schedule()
    assert output.prefill_seq_ids == [second.sequence_ids[0]]


def _build_host_resident_case(
    backend: _FakeAsyncBackend,
) -> tuple[Scheduler, SequenceGroup]:
    scheduler = _make_async_scheduler(num_blocks=2, backend=backend)
    first = _make_group("req-1", seq_id=1, prompt_len=4)
    second = _make_group("req-2", seq_id=2, prompt_len=8)
    scheduler.add_request(first)
    scheduler.schedule()
    scheduler.update_after_step([], [first.sequence_ids[0]])
    scheduler.add_request(second)
    scheduler.schedule()
    backend.complete_all()
    scheduler.schedule()
    scheduler.abort_request("req-2")
    return scheduler, first


def test_swapped_sequence_not_decoded_before_h2d_completion() -> None:
    backend = _FakeAsyncBackend()
    scheduler, group = _build_host_resident_case(backend)

    output = scheduler.schedule()
    assert output.decode_seq_ids == []
    assert group.sequences[0].status is SequenceStatus.SWAPPED

    backend.complete_next()
    output = scheduler.schedule()
    assert output.decode_seq_ids == [group.sequence_ids[0]]
    assert group.sequences[0].status is SequenceStatus.DECODE


def test_reservation_backpressure_leaves_group_running() -> None:
    backend = _FakeAsyncBackend()
    scheduler = _make_async_scheduler(num_blocks=2, backend=backend)
    victim = _make_group("req-1", seq_id=1, prompt_len=8)
    scheduler.add_request(victim)
    scheduler.schedule()
    prior = victim.sequences[0].status

    scheduler.kv_cache.request_swap_out(1)
    intruder = _make_group("req-2", seq_id=2, prompt_len=8)
    scheduler.add_request(intruder)

    output = scheduler.schedule()

    assert output.preempted_seq_ids == []
    assert victim.sequences[0].status is prior
    assert _only_queue(scheduler, "req-1") == "running"


def test_accepted_preemption_moves_group_and_marks_swapped() -> None:
    backend = _FakeAsyncBackend()
    scheduler = _make_async_scheduler(num_blocks=2, backend=backend)
    group = _multi_group("req-1", [1, 2], prompt_len=4)
    scheduler.add_request(group)
    scheduler.schedule()

    other = _make_group("req-2", seq_id=3, prompt_len=8)
    scheduler.add_request(other)
    output = scheduler.schedule()

    assert set(output.preempted_seq_ids) == {1, 2}
    assert all(seq.status is SequenceStatus.SWAPPED for seq in group.sequences)
    assert _only_queue(scheduler, "req-1") == "swapped"


def test_partial_completion_holds_queues_and_statuses() -> None:
    backend = _FakeAsyncBackend()
    scheduler = _make_async_scheduler(num_blocks=2, backend=backend)
    group = _multi_group("req-1", [1, 2], prompt_len=4)
    scheduler.add_request(group)
    scheduler.schedule()
    other = _make_group("req-2", seq_id=3, prompt_len=8)
    scheduler.add_request(other)
    scheduler.schedule()

    backend.complete_next()
    scheduler.schedule()

    assert all(seq.status is SequenceStatus.SWAPPED for seq in group.sequences)
    assert _only_queue(scheduler, "req-1") == "swapped"


def test_all_h2d_completions_restore_mixed_prior_statuses() -> None:
    backend = _FakeAsyncBackend()
    scheduler = _make_async_scheduler(num_blocks=2, backend=backend)
    group = _multi_group("req-1", [1, 2], prompt_len=4)
    scheduler.add_request(group)
    scheduler.schedule()
    group.sequences[0].set_status(SequenceStatus.DECODE)
    group.sequences[1].set_status(SequenceStatus.DRAFT)
    prior = {seq.seq_id: seq.status for seq in group.sequences}

    other = _make_group("req-2", seq_id=3, prompt_len=8)
    scheduler.add_request(other)
    scheduler.schedule()
    backend.complete_all()
    scheduler.schedule()
    scheduler.abort_request("req-2")
    scheduler.schedule()
    backend.complete_all()
    scheduler.schedule()

    for seq in group.sequences:
        assert seq.status is prior[seq.seq_id]
    assert _only_queue(scheduler, "req-1") == "running"


def test_partial_d2h_failure_rolls_back_without_partial_publish() -> None:
    backend = _FakeAsyncBackend(fail_d2h_calls={2})
    scheduler = _make_async_scheduler(num_blocks=2, backend=backend)
    group = _multi_group("req-1", [1, 2], prompt_len=4)
    scheduler.add_request(group)
    scheduler.schedule()
    prior = {seq.seq_id: seq.status for seq in group.sequences}

    other = _make_group("req-2", seq_id=3, prompt_len=8)
    scheduler.add_request(other)
    scheduler.schedule()
    backend.complete_all()
    scheduler.schedule()
    backend.complete_all()
    scheduler.schedule()

    queue = _only_queue(scheduler, "req-1")
    if queue == "running":
        for seq in group.sequences:
            assert seq.status is prior[seq.seq_id]
    else:
        assert all(
            seq.status is SequenceStatus.SWAPPED for seq in group.sequences
        )


def test_retry_exhaustion_moves_group_to_waiting() -> None:
    backend = _FakeAsyncBackend(fail_all_h2d=True)
    scheduler = Scheduler(
        kv_cache=_make_async_cache(2, backend),
        max_batch_size=4,
        max_tokens_per_step=64,
        kv_swap_max_retries=1,
    )
    group = _multi_group("req-1", [1, 2], prompt_len=4)
    scheduler.add_request(group)
    scheduler.schedule()

    other = _make_group("req-2", seq_id=3, prompt_len=8)
    scheduler.add_request(other)
    scheduler.schedule()
    backend.complete_all()
    scheduler.schedule()
    scheduler.abort_request("req-2")

    reached_waiting = False
    for _ in range(6):
        backend.complete_all()
        scheduler.schedule()
        if (
            all(seq.status is SequenceStatus.WAITING for seq in group.sequences)
            and not scheduler._swapped_groups
        ):
            reached_waiting = True
            break

    assert reached_waiting
    assert all(seq.status is SequenceStatus.WAITING for seq in group.sequences)
    assert "req-1" not in scheduler._swapped_groups


def test_free_gpu_blocks_preserves_cpu_buffer() -> None:
    from moe_infinity.engine.kv_transfer import KVTransferState

    kv_cache = _make_kv_cache(num_blocks=2)
    kv_cache.allocate_sequence(seq_id=33, num_tokens=8)
    kv_cache.swap_out(seq_id=33)
    kv_records = cast(dict[int, object], getattr(kv_cache, "_kv_records"))
    sequence_tables = cast(
        dict[int, object],
        getattr(kv_cache, "_sequence_tables"),
    )

    assert kv_cache.transfer_state(33) is KVTransferState.HOST_RESIDENT
    assert kv_records[33].pageable_buffer is not None
    blocks_before = kv_cache.get_block_table(33)
    assert len(blocks_before) == 2

    kv_cache.free_gpu_blocks(seq_id=33)

    assert kv_records[33].pageable_buffer is not None
    assert 33 in sequence_tables
    assert kv_cache.get_block_table(33) == []
    assert kv_cache.block_allocator.num_free_blocks == 2


def _make_paged_backend(num_blocks: int):
    from moe_infinity.runtime.attention_backend import PagedAttentionBackend
    from moe_infinity.runtime.attention_types import KVCacheSpec

    backend = PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=1, head_dim=8, dtype=torch.float16, block_size=4
        ),
        num_gpu_blocks=num_blocks,
        device=torch.device("cpu"),
    )
    backend.create_layered_store(layer_count=1)
    return backend


def test_partial_prefill_recovers_to_prefill_at_same_offset() -> None:
    cache = _make_kv_cache(4)
    backend = _make_paged_backend(num_blocks=4)
    cache.set_block_store(backend.block_store, owner=backend)
    scheduler = Scheduler(
        kv_cache=cache,
        max_batch_size=1,
        max_tokens_per_step=4,
        enable_chunked_prefill=True,
        prefill_chunk_size=4,
    )
    group = _make_group("partial", seq_id=41, prompt_len=8)
    scheduler.add_request(group)
    first = scheduler.schedule()
    scheduler.commit_prefill_step(first.prefill_transaction_id)
    assert group.sequences[0].num_computed_tokens == 4

    preempted = scheduler._preempt_oldest_running_group()
    assert preempted == [41]
    assert group.sequences[0].status is SequenceStatus.SWAPPED
    scheduler._restored_this_pass = set()
    scheduler._advance_swapped_groups(scheduler.kv_cache.poll_transfers())
    scheduler._recover_host_resident_groups()
    scheduler._advance_swapped_groups(scheduler.kv_cache.poll_transfers())

    assert group.sequences[0].status is SequenceStatus.PREFILL
    resumed = scheduler.schedule()
    assert resumed.prefill_chunks[41].start_pos == 4
