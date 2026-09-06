from __future__ import annotations

import pytest

import moe_infinity.serving.prefix_cache as prefix_cache_module
from moe_infinity.serving.prefix_cache import PrefixCache
from moe_infinity.serving.prefix_contract import PrefixLease, PrefixMatch
from tests.python.serving.prefix_cache_test_utils import (
    RefRecorder,
    make_namespace,
)


def test_forced_digest_collision_preserves_multilevel_parent_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prefix_cache_module, "_digest_block", lambda *args: "same"
    )
    cache = PrefixCache(4, 32)
    ns = make_namespace()
    a = [1, 1, 1, 1, 7, 7, 7, 7, 9, 9, 9, 9, 20]
    b = [2, 2, 2, 2, 7, 7, 7, 7, 9, 9, 9, 9, 21]
    cache.insert(ns, a, [10, 11, 12], committed_tokens=12)
    cache.insert(ns, b, [20, 21, 22], committed_tokens=12)
    lease_a = cache.acquire_prefix_lease(ns, a, max_prefix_tokens=12)
    lease_b = cache.acquire_prefix_lease(ns, b, max_prefix_tokens=12)
    assert lease_a.match.block_ids == (10, 11, 12)
    assert lease_b.match.block_ids == (20, 21, 22)
    lease_a.abort()
    lease_b.abort()


def test_lookup_pins_before_lock_is_released_and_abort_balances_refs() -> None:
    refs = RefRecorder()
    cache = PrefixCache(4, 8, on_retain=refs.retain, on_release=refs.release)
    ns = make_namespace()
    cache.insert(ns, [1, 2, 3, 4, 5], [9], committed_tokens=4)
    lease = cache.acquire_prefix_lease(ns, [1, 2, 3, 4, 6], max_prefix_tokens=4)
    assert lease.match.block_ids == (9,)
    assert refs.retained == [[9], [9]]  # cache ownership, then lease ownership
    lease.abort()
    assert refs.released == [[9]]


def test_lease_prepare_requires_same_owner_for_commit_or_abort() -> None:
    released: list[list[int]] = []
    lease = PrefixLease(
        PrefixMatch(4, (9,), (1,)), released.append, lambda: None
    )
    owner = object()
    other = object()
    assert lease.prepare_adoption(owner).block_ids == (9,)
    with pytest.raises(RuntimeError, match="owner/state mismatch"):
        lease.commit_adoption(other)
    with pytest.raises(RuntimeError, match="owner mismatch"):
        lease.abort(other)
    lease.abort(owner)
    assert released == [[9]]


def test_prepared_lease_commit_transfers_without_release() -> None:
    released: list[list[int]] = []
    terminals: list[str] = []
    lease = PrefixLease(
        PrefixMatch(4, (9,), (1,)),
        released.append,
        lambda: terminals.append("closed"),
    )
    owner = object()
    lease.prepare_adoption(owner)
    lease.commit_adoption(owner)
    assert lease.state == "committed"
    assert released == []
    assert terminals == ["closed"]
