import torch

from moe_infinity.engine.kv_transfer import (
    CopyTicket,
    KVTransferState,
    PinnedBufferPool,
    SyncKVTransferBackend,
)
from moe_infinity.serving.kv_cache import PagedKVCache


class FakeEvent:
    def __init__(self) -> None:
        self.done = False

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.done = True


class FakeBackend:
    asynchronous = True

    def __init__(self) -> None:
        self.events: list[FakeEvent] = []

    def submit_d2h(
        self,
        source_cache: torch.Tensor,
        destination: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        source = source_cache.index_select(
            block_dim, torch.tensor(block_ids, dtype=torch.long)
        )
        destination.copy_(source)
        event = FakeEvent()
        self.events.append(event)
        return CopyTicket(
            device=source.device,
            stream=None,
            event=event,
            owned_staging_tensors=(),
            submitted_ns=1,
            nbytes=source.numel() * source.element_size(),
        )

    def submit_h2d(
        self,
        source: torch.Tensor,
        destination_cache: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        destination_cache.index_copy_(
            block_dim,
            torch.tensor(block_ids, dtype=torch.long),
            source,
        )
        event = FakeEvent()
        self.events.append(event)
        return CopyTicket(
            device=destination_cache.device,
            stream=None,
            event=event,
            owned_staging_tensors=(),
            submitted_ns=1,
            nbytes=source.numel() * source.element_size(),
        )

    def close(self) -> None:
        return None


def _cpu_test_pool(capacity_bytes: int) -> PinnedBufferPool:
    return PinnedBufferPool(
        capacity_bytes=capacity_bytes,
        allocator=lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )


def make_cache(backend: FakeBackend) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=2,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
        transfer_backend=backend,
        pinned_pool=_cpu_test_pool(1024),
    )


def test_swap_out_holds_blocks_until_event_completion() -> None:
    backend = FakeBackend()
    cache = make_cache(backend)
    cache.allocate_sequence(7, num_tokens=8)
    assert cache.request_swap_out(7)
    assert cache.transfer_state(7) is KVTransferState.SWAP_OUT_IN_FLIGHT
    assert cache.block_allocator.num_free_blocks == 0
    assert cache.poll_transfers() == []
    backend.events[0].done = True
    completion = cache.poll_transfers()
    assert completion[0].success
    assert cache.transfer_state(7) is KVTransferState.HOST_RESIDENT
    assert cache.block_allocator.num_free_blocks == 2
    assert cache.get_block_table(7) == []


def test_swap_in_publishes_blocks_only_after_event() -> None:
    backend = FakeBackend()
    cache = make_cache(backend)
    cache.allocate_sequence(9, num_tokens=8)
    assert cache.request_swap_out(9)
    backend.events[-1].done = True
    cache.poll_transfers()
    assert cache.request_swap_in(9)
    assert cache.get_block_table(9) == []
    assert not cache.is_gpu_ready(9)
    backend.events[-1].done = True
    cache.poll_transfers()
    assert cache.is_gpu_ready(9)
    assert len(cache.get_block_table(9)) == 2


def test_sync_round_trip_is_blocking_pageable_and_never_allocates_pool() -> (
    None
):
    pool_factory_calls: list[int] = []

    def forbidden_pool_factory(capacity_bytes: int):
        pool_factory_calls.append(capacity_bytes)
        raise AssertionError("sync mode must not construct a pinned pool")

    cache = PagedKVCache(
        num_blocks=2,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        transfer_backend=SyncKVTransferBackend(),
        host_pool_bytes=1,
        host_pool_factory=forbidden_pool_factory,
    )
    cache.allocate_sequence(12, num_tokens=8)
    original_ids = cache.get_block_table(12)
    expected = torch.arange(32, dtype=torch.float32).reshape(1, 2, 2, 4, 1, 2)
    cache.get_kv_cache_tensors()[:, original_ids, ...].copy_(expected)

    cache.swap_out(12)

    record = cache._kv_records[12]
    assert record.state is KVTransferState.HOST_RESIDENT
    assert record.host_lease is None
    assert record.pageable_buffer is not None
    assert not record.pageable_buffer.require_tensor().is_pinned()
    assert pool_factory_calls == []
    assert cache.get_swap_stats()["host_capacity_bytes"] == 0
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0

    cache.free_gpu_blocks(12)
    cache.get_kv_cache_tensors().fill_(-1)
    cache.swap_in(12)

    restored_ids = cache.get_block_table(12)
    torch.testing.assert_close(
        cache.get_kv_cache_tensors()[:, restored_ids, ...], expected
    )
    assert record.pageable_buffer is None
    assert cache.is_gpu_ready(12)
    assert pool_factory_calls == []


def test_sync_host_resident_free_releases_pageable_record_without_pool() -> (
    None
):
    cache = PagedKVCache(
        num_blocks=1,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        transfer_backend=SyncKVTransferBackend(),
        host_pool_bytes=1,
        host_pool_factory=lambda _capacity: (_ for _ in ()).throw(
            AssertionError("sync mode must not construct a pinned pool")
        ),
    )
    cache.allocate_sequence(13, num_tokens=4)
    cache.swap_out(13)
    pageable = cache._kv_records[13].pageable_buffer
    assert pageable is not None

    cache.free_sequence(13)

    assert pageable.tensor is None
    assert cache.block_allocator.num_free_blocks == 1
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0


class FailingBackend(FakeBackend):
    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self._fail_on = fail_on

    def submit_d2h(self, *args, **kwargs) -> CopyTicket:
        if self._fail_on == "d2h":
            raise RuntimeError("simulated d2h failure")
        return super().submit_d2h(*args, **kwargs)

    def submit_h2d(self, *args, **kwargs) -> CopyTicket:
        if self._fail_on == "h2d":
            raise RuntimeError("simulated h2d failure")
        return super().submit_h2d(*args, **kwargs)


def make_async_cache(
    backend: FakeBackend,
    *,
    num_blocks: int = 2,
    host_pool_bytes: int = 4096,
) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
        transfer_backend=backend,
        pinned_pool=_cpu_test_pool(host_pool_bytes),
    )


def test_pool_backpressure_leaves_sequence_gpu_resident() -> None:
    backend = FakeBackend()
    cache = make_async_cache(backend, host_pool_bytes=1)
    cache.allocate_sequence(7, num_tokens=8)

    assert not cache.request_swap_out(7)

    assert cache.transfer_state(7) is KVTransferState.GPU_RESIDENT
    assert cache.block_allocator.num_free_blocks == 0
    assert len(cache.get_block_table(7)) == 2
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0


def test_swap_out_failure_keeps_source_blocks_allocated() -> None:
    backend = FailingBackend(fail_on="d2h")
    cache = make_async_cache(backend)
    cache.allocate_sequence(7, num_tokens=8)

    assert not cache.request_swap_out(7)

    assert cache.transfer_state(7) is KVTransferState.GPU_RESIDENT
    assert cache.block_allocator.num_free_blocks == 0
    assert len(cache.get_block_table(7)) == 2
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0


def test_swap_in_failure_releases_restoring_blocks_and_keeps_host_copy() -> (
    None
):
    backend = FailingBackend(fail_on="h2d")
    cache = make_async_cache(backend)
    cache.allocate_sequence(7, num_tokens=8)
    assert cache.request_swap_out(7)
    backend.events[-1].done = True
    cache.poll_transfers()
    assert cache.transfer_state(7) is KVTransferState.HOST_RESIDENT
    cache.free_gpu_blocks(7)
    assert cache.block_allocator.num_free_blocks == 2

    assert not cache.request_swap_in(7)

    assert cache.transfer_state(7) is KVTransferState.HOST_RESIDENT
    assert cache.block_allocator.num_free_blocks == 2
    assert cache._kv_records[7].host_lease is not None


def test_checksum_mismatch_never_submits_h2d() -> None:
    backend = FakeBackend()
    cache = make_async_cache(backend)
    cache.allocate_sequence(7, num_tokens=8)
    assert cache.request_swap_out(7)
    backend.events[-1].done = True
    cache.poll_transfers()
    cache.free_gpu_blocks(7)

    record = cache._kv_records[7]
    corrupt = record.metadata
    assert corrupt is not None
    from dataclasses import replace

    record.metadata = replace(corrupt, num_tokens=corrupt.num_tokens + 1)

    submissions_before = len(backend.events)
    assert not cache.request_swap_in(7)

    assert len(backend.events) == submissions_before
    assert cache.transfer_state(7) is KVTransferState.FAILED


def test_free_sequence_during_d2h_retains_blocks_lease_and_ticket_until_retired() -> (
    None
):  # noqa: E501
    backend = FakeBackend()
    cache = make_async_cache(backend)
    cache.allocate_sequence(7, num_tokens=8)
    assert cache.request_swap_out(7)

    record = cache._kv_records[7]
    ticket = record.ticket
    lease = record.host_lease
    assert ticket is not None and lease is not None
    block_count = len(record.active_block_ids)
    key = (record.key.seq_id, record.key.generation)

    cache.free_sequence(7)

    assert key in cache._retiring_records
    assert not backend.events[-1].query()
    assert cache.block_allocator.num_free_blocks == 2 - block_count
    assert cache._pinned_pool.in_use_bytes == lease.nbytes

    assert cache.poll_transfers() == []

    backend.events[-1].done = True
    cache.poll_transfers()

    assert ticket.retired
    assert key not in cache._retiring_records
    assert cache.block_allocator.num_free_blocks == 2
    assert cache._pinned_pool.in_use_bytes == 0


def test_free_sequence_during_h2d_retains_blocks_lease_and_ticket_until_retired() -> (
    None
):  # noqa: E501
    backend = FakeBackend()
    cache = make_async_cache(backend)
    cache.allocate_sequence(7, num_tokens=8)
    assert cache.request_swap_out(7)
    backend.events[-1].done = True
    cache.poll_transfers()
    cache.free_gpu_blocks(7)
    assert cache.request_swap_in(7)

    record = cache._kv_records[7]
    ticket = record.ticket
    lease = record.host_lease
    assert ticket is not None and lease is not None
    restoring = list(record.restoring_block_ids)
    key = (record.key.seq_id, record.key.generation)

    cache.free_sequence(7)

    assert key in cache._retiring_records
    assert not backend.events[-1].query()
    assert cache.block_allocator.num_free_blocks == 2 - len(restoring)
    assert cache._pinned_pool.in_use_bytes == lease.nbytes

    backend.events[-1].done = True
    cache.poll_transfers()

    assert ticket.retired
    assert key not in cache._retiring_records
    assert cache.block_allocator.num_free_blocks == 2
    assert cache._pinned_pool.in_use_bytes == 0


def test_cancel_sequence_is_idempotent_while_ticket_is_pending() -> None:
    backend = FakeBackend()
    cache = make_async_cache(backend)
    cache.allocate_sequence(7, num_tokens=8)
    assert cache.request_swap_out(7)
    key = (
        cache._kv_records[7].key.seq_id,
        cache._kv_records[7].key.generation,
    )

    cache.cancel_sequence(7)
    cache.cancel_sequence(7)
    cache.free_sequence(7)

    assert key in cache._retiring_records
    assert cache.block_allocator.num_free_blocks == 0

    backend.events[-1].done = True
    cache.poll_transfers()

    assert key not in cache._retiring_records
    assert cache.block_allocator.num_free_blocks == 2


def test_stale_generation_completion_cannot_mutate_reused_seq_id() -> None:
    backend = FakeBackend()
    cache = make_async_cache(backend, num_blocks=4)
    cache.allocate_sequence(7, num_tokens=8)
    old_generation = cache._kv_records[7].key.generation
    assert cache.request_swap_out(7)

    cache.free_sequence(7)

    cache.allocate_sequence(7, num_tokens=4)
    new_generation = cache._kv_records[7].key.generation
    assert new_generation != old_generation
    new_blocks = list(cache.get_block_table(7))

    backend.events[-1].done = True
    cache.poll_transfers()

    assert cache._kv_records[7].key.generation == new_generation
    assert cache.transfer_state(7) is KVTransferState.GPU_RESIDENT
    assert cache.get_block_table(7) == new_blocks


def test_free_sequence_reclaims_host_resident_lease() -> None:
    backend = FakeBackend()
    cache = make_async_cache(backend)
    cache.allocate_sequence(7, num_tokens=8)
    assert cache.request_swap_out(7)
    backend.events[-1].done = True
    cache.poll_transfers()
    assert cache.transfer_state(7) is KVTransferState.HOST_RESIDENT
    assert cache._pinned_pool.in_use_bytes > 0

    cache.free_sequence(7)

    assert cache._pinned_pool.in_use_bytes == 0
    assert cache.block_allocator.num_free_blocks == 2
