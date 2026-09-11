import time
import weakref

import pytest
import torch

from moe_infinity.engine.kv_transfer import CudaKVTransferBackend
from moe_infinity.serving.kv_cache import PagedKVCache

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _cache(*, dtype: torch.dtype = torch.float16, num_blocks: int = 4):
    device = torch.device("cuda:0")
    backend = CudaKVTransferBackend(device)
    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=dtype,
        device=device,
        transfer_backend=backend,
        host_pool_bytes=1 << 20,
    )
    return cache, backend


def _poll(cache: PagedKVCache, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while cache.has_pending_transfers() and time.monotonic() < deadline:
        cache.poll_transfers()
        time.sleep(0.001)
    cache.poll_transfers()
    assert not cache.has_pending_transfers()


def _delay(stream: torch.cuda.Stream) -> None:
    sleep = getattr(torch.cuda, "_sleep", None)
    if sleep is None:
        pytest.skip("torch.cuda._sleep unavailable")
    with torch.cuda.stream(stream):
        sleep(5_000_000_000)


def _warm_transfer_path(cache: PagedKVCache, block_count: int) -> None:
    pool = cache._pinned_pool
    backend = cache._backend
    assert pool is not None
    assert isinstance(backend, CudaKVTransferBackend)
    lease = pool.acquire(
        (
            cache.num_layers,
            block_count,
            2,
            cache.block_size,
            cache.num_heads,
            cache.head_dim,
        ),
        cache.dtype,
    )
    assert lease is not None
    ticket = backend.submit_d2h(
        cache.get_kv_cache_tensors(),
        lease.tensor,
        block_ids=list(range(block_count)),
        block_dim=1,
    )
    assert ticket.retire(synchronize=True)
    pool.release(lease)


def _delay_recorded_events(monkeypatch: pytest.MonkeyPatch) -> None:
    original_event = torch.cuda.Event

    class DelayedEvent:
        def __init__(self) -> None:
            self._event = original_event()

        def record(self, stream: torch.cuda.Stream) -> None:
            _delay(stream)
            self._event.record(stream)

        def query(self) -> bool:
            return self._event.query()

        def synchronize(self) -> None:
            self._event.synchronize()

    monkeypatch.setattr(torch.cuda, "Event", DelayedEvent)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_swap_round_trip_is_exact_and_uses_pinned_host(
    dtype: torch.dtype,
) -> None:
    cache, _ = _cache(dtype=dtype)
    cache.allocate_sequence(1, num_tokens=8)
    original_ids = cache.get_block_table(1)
    expected = (
        torch.arange(1 * 2 * 2 * 4 * 1 * 8, dtype=torch.float32, device="cuda")
        .reshape(1, 2, 2, 4, 1, 8)
        .to(dtype)
    )
    cache.get_kv_cache_tensors()[:, original_ids, ...].copy_(expected)

    assert cache.request_swap_out(1)
    lease = cache._kv_records[1].host_lease
    assert lease is not None
    assert lease.tensor.device.type == "cpu"
    assert lease.tensor.is_pinned()
    _poll(cache)

    cache.get_kv_cache_tensors().fill_(-1)
    assert cache.request_swap_in(1)
    _poll(cache)
    restored_ids = cache.get_block_table(1)
    torch.testing.assert_close(
        cache.get_kv_cache_tensors()[:, restored_ids, ...],
        expected,
        rtol=0,
        atol=0,
    )
    cache.free_sequence(1)
    cache.shutdown()


def test_async_submission_returns_before_transfer_event_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, backend = _cache()
    cache.allocate_sequence(1, num_tokens=8)
    _warm_transfer_path(cache, block_count=2)
    _delay_recorded_events(monkeypatch)

    assert cache.request_swap_out(1)

    ticket = cache._kv_records[1].ticket
    assert ticket is not None
    assert not ticket.query()
    cache.shutdown()


def test_cancel_during_d2h_does_not_reuse_source_blocks_early() -> None:
    cache, backend = _cache(num_blocks=2)
    cache.allocate_sequence(1, num_tokens=8)
    _warm_transfer_path(cache, block_count=2)
    _delay(backend._transfer_stream)
    assert cache.request_swap_out(1)
    ticket = cache._kv_records[1].ticket
    assert ticket is not None

    cache.cancel_sequence(1)

    assert cache.block_allocator.num_free_blocks == 0
    assert cache.get_swap_stats()["host_in_use_bytes"] > 0
    assert not ticket.retired
    _poll(cache)
    assert ticket.retired
    assert cache.block_allocator.num_free_blocks == cache.num_blocks
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0
    cache.shutdown()


def test_cancel_during_h2d_does_not_publish_partial_block_table() -> None:
    cache, backend = _cache(num_blocks=2)
    cache.allocate_sequence(1, num_tokens=8)
    assert cache.request_swap_out(1)
    _poll(cache)
    _delay(backend._transfer_stream)
    assert cache.request_swap_in(1)
    ticket = cache._kv_records[1].ticket
    assert ticket is not None
    assert cache.get_block_table(1) == []

    cache.cancel_sequence(1)

    assert cache.block_allocator.num_free_blocks == 0
    _poll(cache)
    assert ticket.retired
    assert cache.block_allocator.num_free_blocks == cache.num_blocks
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0
    cache.shutdown()


def test_ticket_consumer_stream_waits_for_restore_event_before_read() -> None:
    cache, backend = _cache(num_blocks=2)
    cache.allocate_sequence(1, num_tokens=8)
    ids = cache.get_block_table(1)
    expected = torch.arange(128, device="cuda", dtype=torch.float16).reshape(
        1, 2, 2, 4, 1, 8
    )
    cache.get_kv_cache_tensors()[:, ids, ...].copy_(expected)
    assert cache.request_swap_out(1)
    _poll(cache)
    _delay(backend._transfer_stream)
    assert cache.request_swap_in(1)
    record = cache._kv_records[1]
    ticket = record.ticket
    assert ticket is not None
    ticket.wait_on_consumer(torch.cuda.current_stream())
    checksum = cache.get_kv_cache_tensors()[
        :, record.restoring_block_ids, ...
    ].sum()

    assert checksum.item() == expected.sum().item()
    _poll(cache)
    cache.free_sequence(1)
    cache.shutdown()


def test_ticket_retire_keeps_staging_alive_until_event_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, backend = _cache(num_blocks=2)
    cache.allocate_sequence(1, num_tokens=8)
    _warm_transfer_path(cache, block_count=2)
    _delay_recorded_events(monkeypatch)
    assert cache.request_swap_out(1)
    ticket = cache._kv_records[1].ticket
    assert ticket is not None
    staging_ref = weakref.ref(ticket.owned_staging_tensors[0])

    assert ticket.retire() is False
    assert staging_ref() is not None
    assert ticket.owned_staging_tensors
    assert ticket.retire(synchronize=True)
    assert ticket.retired
    assert ticket.owned_staging_tensors == ()
    cache.poll_transfers()
    cache.free_sequence(1)
    cache.shutdown()


def test_shutdown_retires_before_releasing_pool_and_closing_backend() -> None:
    cache, backend = _cache(num_blocks=2)
    cache.allocate_sequence(1, num_tokens=8)
    _delay(backend._transfer_stream)
    assert cache.request_swap_out(1)
    record = cache._kv_records[1]
    ticket = record.ticket
    pool = cache._pinned_pool
    assert ticket is not None and pool is not None
    order: list[str] = []
    original_retire = ticket.retire
    original_release = pool.release
    original_close = backend.close

    def retire(*, synchronize: bool = False) -> bool:
        result = original_retire(synchronize=synchronize)
        if result:
            order.append("ticket.retire")
        return result

    def release(lease) -> None:
        order.append("pool.release")
        original_release(lease)

    def close() -> None:
        order.append("backend.close")
        original_close()

    ticket.retire = retire
    pool.release = release
    backend.close = close
    cache.cancel_sequence(1)

    cache.shutdown()

    assert order.index("ticket.retire") < order.index("pool.release")
    assert order.index("pool.release") < order.index("backend.close")
    assert cache.block_allocator.num_free_blocks == cache.num_blocks
    assert pool.in_use_bytes == 0


def test_repeated_swap_cancel_cycles_have_no_block_or_pinned_leak() -> None:
    cache, _ = _cache(num_blocks=2)
    completed_tickets = []
    for seq_id in range(20):
        cache.allocate_sequence(seq_id, num_tokens=8)
        assert cache.request_swap_out(seq_id)
        ticket = cache._kv_records[seq_id].ticket
        assert ticket is not None
        completed_tickets.append(ticket)
        if seq_id % 2:
            cache.cancel_sequence(seq_id)
            _poll(cache)
            continue
        _poll(cache)
        assert cache.request_swap_in(seq_id)
        ticket = cache._kv_records[seq_id].ticket
        assert ticket is not None
        completed_tickets.append(ticket)
        _poll(cache)
        cache.free_sequence(seq_id)

    stats = cache.get_swap_stats()
    assert cache.block_allocator.num_free_blocks == cache.num_blocks
    assert stats["host_in_use_bytes"] == 0
    assert stats["retiring_records"] == 0
    assert all(ticket.retired for ticket in completed_tickets)
    cache.shutdown()
