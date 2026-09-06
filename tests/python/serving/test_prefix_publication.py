from __future__ import annotations

from unittest.mock import Mock

import pytest

from tests.python.serving.prefix_cache_test_utils import (
    add_prefill,
    make_dflash_engine_and_batch,
    make_prefill_batch,
    make_prefix_capable_engine,
)


def test_publication_uses_successfully_committed_range_only(
    cb_engine_factory,
) -> None:
    engine = make_prefix_capable_engine(cb_engine_factory)
    seq = add_prefill(engine, prompt=list(range(20)), committed=8)
    batch = make_prefill_batch(
        seq,
        context_len=8,
        query_tokens=list(range(8, 12)),
        block_table=engine.kv_cache.get_block_table(seq.seq_id),
    )
    engine._execute_and_commit(batch)
    assert seq.committed_kv_tokens == 12
    lease = engine.prefix_cache.acquire_prefix_lease(
        engine.cache_namespace, list(range(20)), max_prefix_tokens=16
    )
    assert lease.match.num_tokens == 12
    lease.abort()


def test_failed_chunk_does_not_advance_or_publish(
    monkeypatch: pytest.MonkeyPatch, cb_engine_factory
) -> None:
    engine = make_prefix_capable_engine(cb_engine_factory)
    seq = add_prefill(engine, prompt=list(range(20)), committed=8)
    monkeypatch.setattr(
        engine,
        "_execute_batch",
        Mock(side_effect=RuntimeError("forward failed")),
    )
    batch = make_prefill_batch(
        seq, 8, [8, 9, 10, 11], engine.kv_cache.get_block_table(seq.seq_id)
    )
    with pytest.raises(RuntimeError):
        engine._execute_and_commit(batch)
    assert seq.committed_kv_tokens == 8


def test_reused_prefix_disables_dflash_delegation(cb_engine_factory) -> None:
    engine, batch = make_dflash_engine_and_batch(
        cb_engine_factory, context_len=16, has_prefix_lease=True
    )
    assert engine._can_delegate_speculative(batch) is False
