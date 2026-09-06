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
_BATCH_MODULE = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)

from moe_infinity.runtime.attention_types import PagedBatchLengths  # noqa: E402

SamplingParams = _SEQUENCE_MODULE.SamplingParams
SequenceData = _SEQUENCE_MODULE.SequenceData
SequenceStatus = _SEQUENCE_MODULE.SequenceStatus
PagedKVCache = _KV_CACHE_MODULE.PagedKVCache
BatchBuilder = _BATCH_MODULE.BatchBuilder
BatchMetadata = _BATCH_MODULE.BatchMetadata
SchedulerOutput = _BATCH_MODULE.SchedulerOutput


def _make_sequence(
    seq_id: int,
    prompt_token_ids: list[int],
    *,
    status: object,
    num_computed_tokens: int,
) -> SequenceData:
    sequence = SequenceData(
        seq_id=seq_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(),
        status=status,
        num_computed_tokens=num_computed_tokens,
    )
    return sequence


def _make_cache() -> PagedKVCache:
    return PagedKVCache(
        num_blocks=16,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )


def test_scheduler_output_creation() -> None:
    output = SchedulerOutput(
        prefill_seq_ids=[1, 2],
        decode_seq_ids=[3],
        preempted_seq_ids=[4],
        num_prefill_tokens=8,
        num_decode_tokens=1,
    )

    assert output.prefill_seq_ids == [1, 2]
    assert output.decode_seq_ids == [3]
    assert output.preempted_seq_ids == [4]
    assert output.num_prefill_tokens == 8
    assert output.num_decode_tokens == 1


def test_scheduler_output_verify_fields_default_empty() -> None:
    output = SchedulerOutput(prefill_seq_ids=[1], decode_seq_ids=[2])

    assert output.draft_seq_ids == []
    assert output.verify_seq_ids == []
    assert output.num_verify_tokens == 0
    assert output.num_verify_expert_bytes == 0


def test_scheduler_output_verify_fields_are_copied() -> None:
    draft_ids = [5, 6]
    verify_ids = [7]
    output = SchedulerOutput(
        draft_seq_ids=draft_ids,
        verify_seq_ids=verify_ids,
        num_verify_tokens=16,
        num_verify_expert_bytes=4096,
    )

    assert output.draft_seq_ids == [5, 6]
    assert output.verify_seq_ids == [7]
    assert output.num_verify_tokens == 16
    assert output.num_verify_expert_bytes == 4096
    assert output.draft_seq_ids is not draft_ids
    assert output.verify_seq_ids is not verify_ids


def test_batch_builder_prefill_only() -> None:
    cache = _make_cache()
    sequences = {
        10: _make_sequence(
            10,
            [11, 12, 13],
            status=SequenceStatus.PREFILL,
            num_computed_tokens=0,
        ),
        11: _make_sequence(
            11, [21, 22], status=SequenceStatus.PREFILL, num_computed_tokens=0
        ),
    }
    cache.allocate_sequence(10, num_tokens=3)
    cache.allocate_sequence(11, num_tokens=2)

    metadata = BatchBuilder.from_scheduler_output(
        SchedulerOutput(prefill_seq_ids=[10, 11]),
        sequences,
        cache,
    )

    assert metadata.seq_ids == [10, 11]
    assert metadata.input_token_ids == [11, 12, 13, 21, 22]
    assert metadata.query_lengths == [3, 2]
    assert metadata.context_lengths == [0, 0]
    assert metadata.kv_seq_lengths == [3, 2]
    assert metadata.is_prefill == [True, True]
    assert metadata.block_tables == [[0], [1]]
    assert metadata.query_offsets == [0, 3, 5]


def test_batch_builder_decode_only() -> None:
    cache = _make_cache()
    sequences = {
        20: _make_sequence(
            20, [1, 2, 3], status=SequenceStatus.DECODE, num_computed_tokens=3
        ),
        21: _make_sequence(
            21, [4, 5], status=SequenceStatus.DECODE, num_computed_tokens=2
        ),
        22: _make_sequence(
            22, [6], status=SequenceStatus.DECODE, num_computed_tokens=1
        ),
    }
    cache.allocate_sequence(20, num_tokens=3)
    cache.allocate_sequence(21, num_tokens=2)
    cache.allocate_sequence(22, num_tokens=1)

    metadata = BatchBuilder.from_scheduler_output(
        SchedulerOutput(decode_seq_ids=[20, 21, 22]),
        sequences,
        cache,
    )

    assert metadata.seq_ids == [20, 21, 22]
    assert metadata.input_token_ids == [3, 5, 6]
    assert metadata.query_lengths == [1, 1, 1]
    assert metadata.context_lengths == [3, 2, 1]
    assert metadata.kv_seq_lengths == [4, 3, 2]
    assert metadata.is_prefill == [False, False, False]
    assert metadata.block_tables == [[0], [1], [2]]
    assert metadata.query_offsets == [0, 1, 2, 3]


def test_batch_builder_mixed() -> None:
    cache = _make_cache()
    sequences = {
        30: _make_sequence(
            30, [7, 8], status=SequenceStatus.PREFILL, num_computed_tokens=0
        ),
        31: _make_sequence(
            31, [1, 2, 3], status=SequenceStatus.DECODE, num_computed_tokens=3
        ),
        32: _make_sequence(
            32, [4], status=SequenceStatus.DECODE, num_computed_tokens=1
        ),
    }
    cache.allocate_sequence(30, num_tokens=2)
    cache.allocate_sequence(31, num_tokens=3)
    cache.allocate_sequence(32, num_tokens=1)

    metadata = BatchBuilder.from_scheduler_output(
        SchedulerOutput(prefill_seq_ids=[30], decode_seq_ids=[31, 32]),
        sequences,
        cache,
    )

    assert metadata.seq_ids == [30, 31, 32]
    assert metadata.input_token_ids == [7, 8, 3, 4]
    assert metadata.query_lengths == [2, 1, 1]
    assert metadata.context_lengths == [0, 3, 1]
    assert metadata.kv_seq_lengths == [2, 4, 2]
    assert metadata.is_prefill == [True, False, False]
    assert metadata.query_offsets == [0, 2, 3, 4]


def test_query_offsets_correct() -> None:
    metadata = BatchMetadata(
        seq_ids=[1, 2, 3],
        input_token_ids=[10, 11, 12, 13, 14, 15],
        lengths=PagedBatchLengths(
            query_lengths=[2, 1, 3],
            query_offsets=[0, 2, 3, 6],
            context_lengths=[0, 2, 5],
            kv_seq_lengths=[2, 3, 8],
        ),
        is_prefill=[True, False, False],
        block_tables=[[0], [1], [2]],
        sampling_params=[SamplingParams(), SamplingParams(), SamplingParams()],
    )

    assert metadata.total_tokens == 6
    assert metadata.query_offsets[-1] == metadata.total_tokens
