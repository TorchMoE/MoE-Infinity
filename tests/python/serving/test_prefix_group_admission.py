from __future__ import annotations

import pytest

from tests.python.serving.prefix_cache_test_utils import (
    SHARED,
    PrefixLease,
    SequenceAllocationPlan,
    SequenceStatus,
    make_group,
    make_seeded_scheduler,
)


def test_group_pins_all_matches_before_eviction() -> None:
    scheduler, _, prefix, _ = make_seeded_scheduler(
        num_blocks=3, max_batch_size=2
    )
    scheduler.add_request(
        make_group("n", [(1, SHARED + [10]), (2, SHARED + [11])])
    )
    scheduler.schedule()
    assert prefix.events.index("pin:10") < prefix.events.index("evict")
    assert prefix.events.index("pin:11") < prefix.events.index("evict")


def test_n_group_allocation_failure_rolls_back_every_sequence_and_lease() -> (
    None
):
    scheduler, cache, prefix, _ = make_seeded_scheduler(
        num_blocks=3, max_batch_size=2
    )
    group = make_group("n", [(1, SHARED + [10]), (2, SHARED + [11])])
    initial_free = cache.block_allocator.num_free_blocks
    initial_refs = tuple(
        cache.block_allocator.ref_count(i) for i in range(cache.num_blocks)
    )
    scheduler.add_request(group)
    output = scheduler.schedule()
    assert output.prefill_seq_ids == []
    assert [seq.status for seq in group.sequences] == [
        SequenceStatus.WAITING,
        SequenceStatus.WAITING,
    ]
    assert cache.block_allocator.num_free_blocks == initial_free
    assert (
        tuple(
            cache.block_allocator.ref_count(i) for i in range(cache.num_blocks)
        )
        == initial_refs
    )
    assert prefix.open_leases == 0


def test_warm_n_group_commits_all_leases_together() -> None:
    scheduler, cache, prefix, _ = make_seeded_scheduler(
        num_blocks=8, max_batch_size=3
    )
    group = make_group(
        "n", [(1, SHARED + [10]), (2, SHARED + [11]), (3, SHARED + [12])]
    )
    scheduler.add_request(group)
    output = scheduler.schedule()
    assert output.prefill_seq_ids == [1, 2, 3]
    assert {
        tuple(cache.get_block_table(seq_id)[:2]) for seq_id in (1, 2, 3)
    } == {(0, 1)}
    assert cache.block_allocator.ref_count(0) == 4
    assert prefix.open_leases == 0


def test_second_lease_prepare_failure_aborts_whole_group(monkeypatch) -> None:
    scheduler, cache, prefix, _ = make_seeded_scheduler(
        num_blocks=8, max_batch_size=2
    )
    group = make_group("n", [(1, SHARED + [10]), (2, SHARED + [11])])
    original = PrefixLease.prepare_adoption
    calls = 0

    def fail_second(self, owner):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second lease prepare failed")
        return original(self, owner)

    monkeypatch.setattr(PrefixLease, "prepare_adoption", fail_second)
    initial_free = cache.block_allocator.num_free_blocks
    initial_refs = tuple(
        cache.block_allocator.ref_count(i) for i in range(cache.num_blocks)
    )
    scheduler.add_request(group)
    with pytest.raises(RuntimeError, match="second lease prepare failed"):
        scheduler.schedule()
    assert not cache.has_sequence(1) and not cache.has_sequence(2)
    assert cache.block_allocator.num_free_blocks == initial_free
    assert (
        tuple(
            cache.block_allocator.ref_count(i) for i in range(cache.num_blocks)
        )
        == initial_refs
    )
    assert prefix.open_leases == 0
    assert [seq.status for seq in group.sequences] == [
        SequenceStatus.WAITING,
        SequenceStatus.WAITING,
    ]


def test_explicit_group_abort_rolls_back_staged_tables_and_prepared_leases() -> (
    None
):
    _, cache, prefix, namespace = make_seeded_scheduler(
        num_blocks=8, max_batch_size=2
    )
    leases = [
        prefix.acquire_prefix_lease(namespace, SHARED + [10], 8),
        prefix.acquire_prefix_lease(namespace, SHARED + [11], 8),
    ]
    plans = [
        SequenceAllocationPlan(1, 9, 8, list(leases[0].match.block_ids)),
        SequenceAllocationPlan(2, 9, 8, list(leases[1].match.block_ids)),
    ]
    initial_free = cache.block_allocator.num_free_blocks
    receipt = cache.prepare_group(plans, leases)
    assert all(lease.state == "prepared" for lease in leases)
    assert not cache.has_sequence(1) and not cache.has_sequence(2)
    cache.abort_group(receipt)
    assert all(lease.state == "aborted" for lease in leases)
    assert cache.block_allocator.num_free_blocks == initial_free
    assert prefix.open_leases == 0


def test_chunked_prefill_uses_same_optional_provider_once() -> None:
    scheduler, _, provider, _ = make_seeded_scheduler(
        num_blocks=8, max_batch_size=1
    )
    sequence = make_group("chunk", [(7, SHARED + [10, 11, 12, 13])]).sequences[
        0
    ]
    first = scheduler._acquire_prefill_lease(sequence, max_prefix_tokens=8)
    assert first.match.num_tokens == 8
    owner = object()
    first.prepare_adoption(owner)
    first.commit_adoption(owner)
    sequence.has_prefix_lease = True
    second = scheduler._acquire_prefill_lease(sequence, max_prefix_tokens=8)
    assert second.match.num_tokens == 0
    second_owner = object()
    second.prepare_adoption(second_owner)
    second.commit_adoption(second_owner)
    assert [event for event in provider.events if event.startswith("pin:")] == [
        "pin:13"
    ]
