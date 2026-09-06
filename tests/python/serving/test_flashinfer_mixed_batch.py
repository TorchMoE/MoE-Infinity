# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false, reportMissingParameterType=false, reportUnknownParameterType=false

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from moe_infinity.runtime.flashinfer_utils import HAS_FLASHINFER

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
_BATCH_MODULE = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)

from moe_infinity.runtime.attention_types import PagedBatchLengths  # noqa: E402

SamplingParams = _SEQUENCE_MODULE.SamplingParams
BatchMetadata = _BATCH_MODULE.BatchMetadata
split_prefill_decode_batch = _BATCH_MODULE.split_prefill_decode_batch


def _make_batch(is_prefill: list[bool]) -> BatchMetadata:
    seq_ids = [10 + idx for idx in range(len(is_prefill))]
    query_lengths = [idx + 1 for idx in range(len(is_prefill))]
    context_lengths = [idx * 3 for idx in range(len(is_prefill))]
    kv_seq_lengths = [
        context_len + query_len
        for context_len, query_len in zip(context_lengths, query_lengths)
    ]
    block_tables = [[idx] for idx in range(len(is_prefill))]
    sampling_params = [SamplingParams() for _ in is_prefill]

    input_token_ids: list[int] = []
    query_offsets = [0]
    next_token = 100
    for seq_len in query_lengths:
        input_token_ids.extend(list(range(next_token, next_token + seq_len)))
        next_token += seq_len
        query_offsets.append(len(input_token_ids))

    return BatchMetadata(
        seq_ids=seq_ids,
        input_token_ids=input_token_ids,
        lengths=PagedBatchLengths(
            query_lengths=query_lengths,
            query_offsets=query_offsets,
            context_lengths=context_lengths,
            kv_seq_lengths=kv_seq_lengths,
        ),
        is_prefill=is_prefill,
        block_tables=block_tables,
        sampling_params=sampling_params,
    )


def test_split_prefill_decode_partitions_correctly() -> None:
    batch = _make_batch([True, False, True, False, False])

    split = split_prefill_decode_batch(batch)

    assert split.prefill_indices == [0, 2]
    assert split.decode_indices == [1, 3, 4]
    assert split.prefill_batch is not None
    assert split.decode_batch is not None
    assert split.prefill_batch.seq_ids == [10, 12]
    assert split.prefill_batch.query_lengths == [1, 3]
    assert split.prefill_batch.input_token_ids == [100, 103, 104, 105]
    assert split.prefill_batch.query_offsets == [0, 1, 4]
    assert split.decode_batch.seq_ids == [11, 13, 14]
    assert split.decode_batch.query_lengths == [2, 4, 5]
    assert split.decode_batch.input_token_ids == [
        101,
        102,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
    ]
    assert split.decode_batch.query_offsets == [0, 2, 6, 11]


def test_split_all_prefill_returns_single_batch() -> None:
    batch = _make_batch([True, True, True])

    split = split_prefill_decode_batch(batch)

    assert split.prefill_indices == [0, 1, 2]
    assert split.decode_indices == []
    assert split.prefill_batch is not None
    assert split.decode_batch is None
    assert split.prefill_batch.seq_ids == batch.seq_ids
    assert split.prefill_batch.input_token_ids == batch.input_token_ids
    assert split.prefill_batch.query_offsets == batch.query_offsets


def test_split_all_decode_returns_single_batch() -> None:
    batch = _make_batch([False, False, False])

    split = split_prefill_decode_batch(batch)

    assert split.prefill_indices == []
    assert split.decode_indices == [0, 1, 2]
    assert split.prefill_batch is None
    assert split.decode_batch is not None
    assert split.decode_batch.seq_ids == batch.seq_ids
    assert split.decode_batch.input_token_ids == batch.input_token_ids
    assert split.decode_batch.query_offsets == batch.query_offsets


def test_recombine_outputs_preserves_sequence_order() -> None:
    batch = _make_batch([True, False, True, False])
    split = split_prefill_decode_batch(batch)

    prefill_outputs = torch.arange(0, 1 + 3, dtype=torch.float32).unsqueeze(1)
    decode_outputs = torch.arange(
        100, 100 + (2 + 4), dtype=torch.float32
    ).unsqueeze(1)

    recombined = split.recombine_outputs(prefill_outputs, decode_outputs)

    expected = torch.tensor(
        [
            [0.0],
            [100.0],
            [101.0],
            [1.0],
            [2.0],
            [3.0],
            [102.0],
            [103.0],
            [104.0],
            [105.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(recombined, expected)


@pytest.mark.skipif(
    not HAS_FLASHINFER or not torch.cuda.is_available(),
    reason="requires flashinfer + CUDA",
)
def test_mixed_batch_flashinfer_correctness() -> None:
    batch = _make_batch([True, False, True])
    split = split_prefill_decode_batch(batch)

    full_outputs = torch.randn(batch.total_tokens, 8, device="cuda")
    assert split.prefill_batch is not None
    assert split.decode_batch is not None

    prefill_rows = split.prefill_batch.total_tokens
    prefill_outputs = full_outputs[:prefill_rows]
    decode_outputs = full_outputs[prefill_rows:]
    recombined = split.recombine_outputs(prefill_outputs, decode_outputs)

    torch.testing.assert_close(recombined, full_outputs)
