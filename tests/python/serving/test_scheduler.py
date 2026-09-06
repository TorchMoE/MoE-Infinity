# pyright: reportAny=false

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import KVCacheSpec

ROOT = Path(__file__).resolve().parents[3]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")

_SEQUENCE_MODULE = _load_module(
    "moe_infinity.serving.sequence",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
_KV_CACHE_MODULE = _load_module(
    "moe_infinity.serving.kv_cache",
    ROOT / "moe_infinity" / "serving" / "kv_cache.py",
)
_ = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)
_SCHEDULER_MODULE = _load_module(
    "moe_infinity.serving.scheduler",
    ROOT / "moe_infinity" / "serving" / "scheduler.py",
)

SamplingParams = _SEQUENCE_MODULE.SamplingParams
SequenceData = _SEQUENCE_MODULE.SequenceData
SequenceGroup = _SEQUENCE_MODULE.SequenceGroup
SequenceStatus = _SEQUENCE_MODULE.SequenceStatus
PagedKVCache = _KV_CACHE_MODULE.PagedKVCache
Scheduler = _SCHEDULER_MODULE.Scheduler


def _make_cache(*, num_blocks: int = 8) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )


def _make_group(request_id: str, seq_id: int, prompt_len: int) -> SequenceGroup:
    sequence = SequenceData(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(),
    )
    return SequenceGroup(request_id=request_id, sequences=[sequence])


_PrefillRowCommit = _SCHEDULER_MODULE._PrefillRowCommit
PrefillChunk = sys.modules["moe_infinity.serving.batch"].PrefillChunk


def _make_paged_backend(num_blocks: int) -> PagedAttentionBackend:
    backend = PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2, head_dim=8, dtype=torch.float16, block_size=4
        ),
        num_gpu_blocks=num_blocks,
        device=torch.device("cpu"),
    )
    backend.create_layered_store(layer_count=1)
    return backend


def _make_chunk_cache(num_blocks: int = 8) -> PagedKVCache:
    cache = _make_cache(num_blocks=num_blocks)
    backend = _make_paged_backend(num_blocks=num_blocks)
    cache.set_block_store(backend.block_store, owner=backend)
    return cache


def _make_chunk_scheduler(*, max_batch_size: int = 8) -> Scheduler:
    return Scheduler(
        _make_chunk_cache(),
        max_batch_size=max_batch_size,
        max_tokens_per_step=8,
        enable_chunked_prefill=True,
        prefill_chunk_size=4,
    )


def test_fcfs_admission_order() -> None:
    cache = _make_cache()
    scheduler = Scheduler(cache, max_batch_size=8, max_tokens_per_step=128)

    req1 = _make_group("req-1", 1, 3)
    req2 = _make_group("req-2", 2, 2)
    scheduler.add_request(req1)
    scheduler.add_request(req2)

    output = scheduler.schedule()

    assert output.prefill_seq_ids == [1, 2]
    assert output.decode_seq_ids == []
    assert output.preempted_seq_ids == []
    assert output.num_prefill_tokens == 5


def test_memory_pressure_preemption() -> None:
    cache = _make_cache(num_blocks=2)
    scheduler = Scheduler(cache, max_batch_size=8, max_tokens_per_step=128)

    req1 = _make_group("req-1", 1, 8)
    scheduler.add_request(req1)
    first = scheduler.schedule()
    assert first.prefill_seq_ids == [1]
    assert cache.block_allocator.num_free_blocks == 0

    req2 = _make_group("req-2", 2, 4)
    scheduler.add_request(req2)

    second = scheduler.schedule()

    assert second.preempted_seq_ids == [1]
    assert second.prefill_seq_ids == [2]
    assert scheduler.get_running_seq_ids() == [2]
    assert req1.sequences[0].status is SequenceStatus.SWAPPED


def test_preemption_skips_speculative_groups_without_orphaning_them() -> None:
    cache = _make_cache(num_blocks=3)
    scheduler = Scheduler(cache, max_batch_size=8, max_tokens_per_step=128)
    draft = _make_group("draft", 1, 4)
    decode = _make_group("decode", 2, 4)
    verify = _make_group("verify", 3, 4)
    for group in (draft, decode, verify):
        scheduler.add_request(group)
    _ = scheduler.schedule()
    draft.sequences[0].set_status(SequenceStatus.DRAFT)
    decode.sequences[0].set_status(SequenceStatus.DECODE)
    verify.sequences[0].set_status(SequenceStatus.DRAFT)
    verify.sequences[0].set_status(SequenceStatus.VERIFY)
    draft_table = cache.get_block_table(1)
    verify_table = cache.get_block_table(3)

    newcomer = _make_group("new", 4, 4)
    scheduler.add_request(newcomer)
    output = scheduler.schedule()

    assert output.preempted_seq_ids == [2]
    assert output.prefill_seq_ids == [4]
    assert scheduler.get_running_seq_ids() == [1, 3, 4]
    assert [group.request_id for group in scheduler._running] == [
        "draft",
        "verify",
        "new",
    ]
    assert cache.get_block_table(1) == draft_table
    assert cache.get_block_table(3) == verify_table


def test_preemption_preserves_all_non_preemptible_running_groups() -> None:
    cache = _make_cache(num_blocks=2)
    scheduler = Scheduler(cache, max_batch_size=8, max_tokens_per_step=128)
    draft = _make_group("draft", 1, 4)
    verify = _make_group("verify", 2, 4)
    scheduler.add_request(draft)
    scheduler.add_request(verify)
    _ = scheduler.schedule()
    draft.sequences[0].set_status(SequenceStatus.DRAFT)
    verify.sequences[0].set_status(SequenceStatus.DRAFT)
    verify.sequences[0].set_status(SequenceStatus.VERIFY)

    scheduler.add_request(_make_group("blocked", 3, 4))
    output = scheduler.schedule()

    assert output.preempted_seq_ids == []
    assert output.prefill_seq_ids == []
    assert [group.request_id for group in scheduler._running] == [
        "draft",
        "verify",
    ]
    assert scheduler.num_waiting == 1
    assert cache.get_block_table(1)
    assert cache.get_block_table(2)


def test_abort_request() -> None:
    cache = _make_cache()
    scheduler = Scheduler(cache, max_batch_size=8, max_tokens_per_step=128)

    req1 = _make_group("req-1", 1, 4)
    req2 = _make_group("req-2", 2, 4)
    scheduler.add_request(req1)
    scheduler.add_request(req2)
    _ = scheduler.schedule()

    scheduler.abort_request("req-1")
    scheduler.abort_request("req-2")

    assert req1.sequences[0].status is SequenceStatus.CANCELLED
    assert req2.sequences[0].status is SequenceStatus.CANCELLED
    assert scheduler.get_running_seq_ids() == []
    assert scheduler.has_work() is False
    assert cache.block_allocator.num_free_blocks == cache.num_blocks


def test_has_work() -> None:
    scheduler = Scheduler(
        _make_cache(), max_batch_size=8, max_tokens_per_step=128
    )

    assert scheduler.has_work() is False

    req = _make_group("req-1", 1, 2)
    scheduler.add_request(req)
    assert scheduler.has_work() is True

    scheduler.abort_request("req-1")
    assert scheduler.has_work() is False


def test_update_after_step() -> None:
    cache = _make_cache()
    scheduler = Scheduler(cache, max_batch_size=8, max_tokens_per_step=128)

    req = _make_group("req-1", 1, 3)
    sequence = req.sequences[0]
    scheduler.add_request(req)

    output = scheduler.schedule()
    assert output.prefill_seq_ids == [1]
    assert sequence.status is SequenceStatus.PREFILL

    scheduler.update_after_step(completed_seq_ids=[], new_decode_seq_ids=[1])
    assert sequence.status is SequenceStatus.DECODE
    assert cache.get_block_table(1) == [0]

    scheduler.update_after_step(completed_seq_ids=[1], new_decode_seq_ids=[])

    assert sequence.status is SequenceStatus.FINISHED
    assert scheduler.has_work() is False
    assert cache.block_allocator.num_free_blocks == cache.num_blocks


def test_chunking_disabled_preserves_whole_prefill_blocking() -> None:
    scheduler = Scheduler(
        _make_cache(),
        max_batch_size=8,
        max_tokens_per_step=4,
        enable_chunked_prefill=False,
        prefill_chunk_size=2,
    )
    scheduler.add_request(_make_group("long", 1, 5))

    output = scheduler.schedule()

    assert output.prefill_seq_ids == []
    assert output.decode_seq_ids == []
    assert output.prefill_chunks == {}


def test_chunked_prefill_never_exceeds_step_budget() -> None:
    scheduler = Scheduler(
        _make_chunk_cache(),
        max_batch_size=8,
        max_tokens_per_step=4,
        enable_chunked_prefill=True,
        prefill_chunk_size=3,
    )
    scheduler.add_request(_make_group("long", 1, 8))

    first = scheduler.schedule()
    assert first.prefill_chunks == {
        1: PrefillChunk(start_pos=0, num_tokens=3, is_terminal=False)
    }
    assert first.num_prefill_tokens == 3
    scheduler.commit_prefill_step(first.prefill_transaction_id)

    second = scheduler.schedule()
    assert second.prefill_chunks[1].start_pos == 3
    assert second.num_prefill_tokens <= 4


def test_decode_rows_are_scheduled_before_prefill_budget() -> None:
    scheduler = Scheduler(
        _make_chunk_cache(),
        max_batch_size=4,
        max_tokens_per_step=3,
        enable_chunked_prefill=True,
        prefill_chunk_size=3,
    )
    decode_group = _make_group("decode", 1, 1)
    scheduler.add_request(decode_group)
    first = scheduler.schedule()
    scheduler.commit_prefill_step(first.prefill_transaction_id)
    scheduler.update_after_step([], [1])
    scheduler.add_request(_make_group("prefill", 2, 8))

    output = scheduler.schedule()

    assert output.decode_seq_ids == [1]
    assert output.prefill_chunks[2].num_tokens == 2
    assert output.num_decode_tokens + output.num_prefill_tokens == 3


def test_partial_prefills_rotate_and_aged_prefill_stays_ahead_of_new_prefill() -> (
    None
):
    scheduler = Scheduler(
        _make_chunk_cache(num_blocks=16),
        max_batch_size=1,
        max_tokens_per_step=2,
        enable_chunked_prefill=True,
        prefill_chunk_size=2,
        prefill_starvation_threshold_steps=2,
    )
    scheduler.add_request(_make_group("a", 1, 6))
    scheduler.add_request(_make_group("b", 2, 6))
    seen: list[int] = []
    for _ in range(2):
        output = scheduler.schedule()
        seen.extend(output.prefill_seq_ids)
        scheduler.commit_prefill_step(output.prefill_transaction_id)
    scheduler.add_request(_make_group("new", 3, 2))
    third = scheduler.schedule()

    assert seen == [1, 2]
    assert third.prefill_seq_ids == [1]


def test_commit_rejects_already_completed_transaction() -> None:
    scheduler = Scheduler(
        _make_chunk_cache(),
        max_batch_size=2,
        max_tokens_per_step=2,
        enable_chunked_prefill=True,
        prefill_chunk_size=2,
    )
    scheduler.add_request(_make_group("req", 1, 4))
    output = scheduler.schedule()

    scheduler.commit_prefill_step(output.prefill_transaction_id)
    with pytest.raises(RuntimeError, match="unknown prefill transaction"):
        scheduler.commit_prefill_step(output.prefill_transaction_id)


@pytest.mark.parametrize("fail_row", [0, 1])
def test_row_preflight_failure_aborts_entire_group(
    fail_row: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = _make_chunk_scheduler(max_batch_size=2)
    scheduler.add_request(_make_group("a", 1, 4))
    scheduler.add_request(_make_group("b", 2, 4))
    before_free = scheduler.kv_cache.block_allocator.num_free_blocks
    output = scheduler.schedule()
    original = _PrefillRowCommit.prepare_commit
    calls = 0

    def fail_selected_preflight(participant: _PrefillRowCommit) -> None:
        nonlocal calls
        current = calls
        calls += 1
        original(participant)
        if current == fail_row:
            raise RuntimeError("injected row preflight failure")

    monkeypatch.setattr(
        _PrefillRowCommit, "prepare_commit", fail_selected_preflight
    )
    with pytest.raises(RuntimeError, match="injected row preflight failure"):
        scheduler.commit_prefill_step(output.prefill_transaction_id)

    assert scheduler.kv_cache.block_allocator.num_free_blocks == before_free
    assert scheduler.inflight_prefill_seq_ids == []
    assert [scheduler._sequence_map[i].num_computed_tokens for i in (1, 2)] == [
        0,
        0,
    ]
    assert scheduler.schedule().prefill_seq_ids == [1, 2]


def test_scheduled_chunk_stays_inflight_until_commit_or_rollback() -> None:
    scheduler = Scheduler(
        _make_chunk_cache(),
        max_batch_size=2,
        max_tokens_per_step=2,
        enable_chunked_prefill=True,
        prefill_chunk_size=2,
    )
    scheduler.add_request(_make_group("req", 1, 4))
    output = scheduler.schedule()

    assert scheduler.inflight_prefill_seq_ids == [1]
    assert scheduler.schedule().prefill_seq_ids == []
    scheduler.rollback_prefill_step(output.prefill_transaction_id)
    assert scheduler.inflight_prefill_seq_ids == []
    retried = scheduler.schedule()
    assert retried.prefill_chunks[1].start_pos == 0


def test_row_is_recorded_after_reservation_before_cow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _make_chunk_scheduler(max_batch_size=1)
    scheduler.add_request(_make_group("req", 1, 4))
    original = scheduler.kv_cache.ensure_writable_range

    def assert_recorded(seq_id: int, start: int, end: int):
        assert scheduler.kv_cache.has_sequence(seq_id)
        assert scheduler.kv_cache.get_num_reserved_tokens(seq_id) == end
        assert seq_id in scheduler._inflight_prefill
        return original(seq_id, start, end)

    monkeypatch.setattr(
        scheduler.kv_cache, "ensure_writable_range", assert_recorded
    )
    scheduler.schedule()


def test_later_row_prepare_failure_rolls_back_every_prepared_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _make_chunk_scheduler(max_batch_size=2)
    scheduler.add_request(_make_group("a", 1, 4))
    scheduler.add_request(_make_group("b", 2, 4))
    initial_free = scheduler.kv_cache.block_allocator.num_free_blocks
    original = scheduler.kv_cache.block_store.checkpoint
    calls = 0

    def fail_second(block_ids: list[int]):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second-row checkpoint failed")
        return original(block_ids)

    monkeypatch.setattr(
        scheduler.kv_cache.block_store, "checkpoint", fail_second
    )
    with pytest.raises(RuntimeError, match="second-row checkpoint failed"):
        scheduler.schedule()

    assert scheduler.inflight_prefill_seq_ids == []
    assert scheduler.kv_cache.block_allocator.num_free_blocks == initial_free
    assert not scheduler.kv_cache.has_sequence(1)
    assert not scheduler.kv_cache.has_sequence(2)
    assert scheduler._sequence_map[1].status is SequenceStatus.WAITING
    assert scheduler._sequence_map[2].status is SequenceStatus.WAITING
    assert list(scheduler._prefill_queue)[:2] == [1, 2]
