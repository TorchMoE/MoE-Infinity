# pyright: reportAny=false

import importlib.util
import sys
import types
from pathlib import Path

import torch

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
