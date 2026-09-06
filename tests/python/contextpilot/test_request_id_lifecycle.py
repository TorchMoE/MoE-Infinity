# pyright: reportAny=false

import importlib.util
import re
import sys
import types
import uuid
from pathlib import Path

import pytest
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
_ensure_package(
    "moe_infinity.entrypoints", ROOT / "moe_infinity" / "entrypoints"
)
_ensure_package(
    "moe_infinity.entrypoints.openai",
    ROOT / "moe_infinity" / "entrypoints" / "openai",
)

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
_ = _load_module(
    "moe_infinity.serving.memory_manager",
    ROOT / "moe_infinity" / "serving" / "memory_manager.py",
)
_ = _load_module(
    "moe_infinity.serving.scheduler",
    ROOT / "moe_infinity" / "serving" / "scheduler.py",
)
_ = _load_module(
    "moe_infinity.serving.model_runner",
    ROOT / "moe_infinity" / "serving" / "model_runner.py",
)
_ = _load_module(
    "moe_infinity.serving.sampler",
    ROOT / "moe_infinity" / "serving" / "sampler.py",
)
_ENGINE_MODULE = _load_module(
    "moe_infinity.serving.engine",
    ROOT / "moe_infinity" / "serving" / "engine.py",
)
_PROTOCOL_MODULE = _load_module(
    "moe_infinity.entrypoints.openai.protocol",
    ROOT / "moe_infinity" / "entrypoints" / "openai" / "protocol.py",
)

SamplingParams = _SEQUENCE_MODULE.SamplingParams
SequenceData = _SEQUENCE_MODULE.SequenceData
SequenceGroup = _SEQUENCE_MODULE.SequenceGroup
SequenceStatus = _SEQUENCE_MODULE.SequenceStatus
PagedKVCache = _KV_CACHE_MODULE.PagedKVCache
Scheduler = _load_module(
    "moe_infinity.serving.scheduler",
    ROOT / "moe_infinity" / "serving" / "scheduler.py",
).Scheduler
ContinuousBatchingEngine = _ENGINE_MODULE.ContinuousBatchingEngine
random_uuid = _PROTOCOL_MODULE.random_uuid


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


def _make_engine_stub() -> ContinuousBatchingEngine:
    engine = object.__new__(ContinuousBatchingEngine)
    engine._next_seq_id = 0
    engine._sequences = {}
    engine._sequence_to_request_id = {}
    engine._request_to_seq_ids = {}
    engine._request_outputs = {}
    engine._callbacks = {}

    class _SchedulerStub:
        def __init__(self) -> None:
            self.request_ids: list[str] = []

        def add_request(self, seq_group: SequenceGroup) -> None:
            self.request_ids.append(seq_group.request_id)

    engine.scheduler = _SchedulerStub()
    return engine


def test_request_id_generated_as_uuid() -> None:
    request_id = random_uuid()

    assert request_id
    assert str(uuid.UUID(request_id)) == request_id
    assert len(request_id) == 36


def test_request_id_present_in_sequence() -> None:
    request_id = "req-123"
    sequence = SequenceData(
        seq_id=1,
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(),
    )
    group = SequenceGroup(request_id=request_id, sequences=[sequence])

    assert group.request_id == request_id
    assert group.get_sequence(1) is sequence


def test_request_id_stable_through_swap() -> None:
    cache = _make_cache(num_blocks=2)
    scheduler = Scheduler(cache, max_batch_size=8, max_tokens_per_step=128)

    request_id_1 = "req-1"
    request_id_2 = "req-2"
    group_1 = _make_group(request_id_1, seq_id=1, prompt_len=8)
    group_2 = _make_group(request_id_2, seq_id=2, prompt_len=4)

    scheduler.add_request(group_1)
    first = scheduler.schedule()
    assert first.prefill_seq_ids == [1]
    assert scheduler._running[0].request_id == request_id_1

    scheduler.add_request(group_2)
    second = scheduler.schedule()
    assert second.preempted_seq_ids == [1]
    assert second.prefill_seq_ids == [2]
    assert scheduler._swapped[0].request_id == request_id_1
    assert group_1.request_id == request_id_1
    assert group_1.sequences[0].status is SequenceStatus.SWAPPED

    scheduler.update_after_step(completed_seq_ids=[], new_decode_seq_ids=[2])
    scheduler.update_after_step(completed_seq_ids=[2], new_decode_seq_ids=[])

    third = scheduler.schedule()
    assert third.decode_seq_ids == [1]
    assert scheduler._running[0].request_id == request_id_1
    assert group_1.request_id == request_id_1
    assert group_1.sequences[0].status is SequenceStatus.DECODE


def test_request_id_format_compatible_with_cp() -> None:
    request_id = random_uuid()

    assert re.fullmatch(r"[A-Za-z0-9-]+", request_id)


def test_request_id_unique_per_request() -> None:
    engine = _make_engine_stub()

    request_id_1 = random_uuid()
    request_id_2 = random_uuid()

    engine.add_request(
        request_id=request_id_1,
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1),
    )
    engine.add_request(
        request_id=request_id_2,
        prompt_token_ids=[2],
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert request_id_1 != request_id_2
    assert engine.scheduler.request_ids == [request_id_1, request_id_2]
    assert set(engine._request_to_seq_ids) == {request_id_1, request_id_2}
