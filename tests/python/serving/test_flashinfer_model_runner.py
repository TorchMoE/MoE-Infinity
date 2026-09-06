from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from moe_infinity.runtime.attention_types import (
    AttentionMetadata,
    PagedBatchLengths,
)

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
_MODEL_RUNNER_MODULE = _load_module(
    "moe_infinity.serving.model_runner",
    ROOT / "moe_infinity" / "serving" / "model_runner.py",
)

SamplingParams = _SEQUENCE_MODULE.SamplingParams
BatchMetadata = _BATCH_MODULE.BatchMetadata
ModelRunner = _MODEL_RUNNER_MODULE.ModelRunner


class _MockOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _MockExpertTracer:
    def __init__(self) -> None:
        self.created: list[int] = []

    def create_entry(self) -> int:
        idx = len(self.created)
        self.created.append(idx)
        return idx


class _MockEngine:
    def __init__(self, backend: object | None, block_size: int = 4) -> None:
        self.request_id = 0
        self.expert_tracer = _MockExpertTracer()
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]
        self._attention_backend = backend
        self.kv_cache = types.SimpleNamespace(block_size=block_size)

    def _generate_request_id(self) -> int:
        request_id = self.request_id
        self.request_id += 1
        return request_id

    def get_attention_backend(self) -> object | None:
        return self._attention_backend


def _make_prefill_batch() -> BatchMetadata:
    return BatchMetadata(
        seq_ids=[1, 2],
        input_token_ids=[100, 101, 200],
        lengths=PagedBatchLengths(
            query_lengths=[2, 1],
            query_offsets=[0, 2, 3],
            context_lengths=[0, 0],
            kv_seq_lengths=[2, 1],
        ),
        is_prefill=[True, True],
        block_tables=[[10], [20]],
        sampling_params=[SamplingParams(), SamplingParams()],
    )


def _make_decode_batch() -> BatchMetadata:
    return BatchMetadata(
        seq_ids=[9],
        input_token_ids=[55],
        lengths=PagedBatchLengths(
            query_lengths=[1],
            query_offsets=[0, 1],
            context_lengths=[3],
            kv_seq_lengths=[4],
        ),
        is_prefill=[False],
        block_tables=[[7]],
        sampling_params=[SamplingParams()],
    )


def _make_paged_model(events: list[tuple[object, ...]]) -> torch.nn.Module:
    class DeepseekV3PagedAttention(torch.nn.Module):
        @classmethod
        def set_paged_context(cls, backend: object, metadata: object) -> None:
            events.append(("set", backend, metadata))

        @classmethod
        def clear_paged_context(cls) -> None:
            events.append(("clear",))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = DeepseekV3PagedAttention()

        def forward(
            self, input_ids: torch.Tensor, **kwargs: object
        ) -> _MockOutput:
            _ = kwargs
            events.append(("model_forward",))
            batch, seq_len = input_ids.shape
            logits = torch.zeros(batch, seq_len, 32, dtype=torch.float32)
            return _MockOutput(logits)

    return Model()


def test_model_runner_sets_and_clears_paged_context() -> None:
    events: list[tuple[object, ...]] = []
    backend = object()
    model = _make_paged_model(events)
    runner = ModelRunner(
        model, _MockEngine(backend), device=torch.device("cpu")
    )

    logits = runner.execute(_make_prefill_batch())

    assert logits.shape == (3, 32)
    assert events[0][0] == "set"
    assert events[0][1] is backend
    metadata = events[0][2]
    assert isinstance(metadata, AttentionMetadata)
    assert metadata.block_tables.dtype == torch.int32
    assert metadata.block_tables.tolist() == [[10], [20]]
    assert metadata.lengths.kv_seq_lengths.tolist() == [2, 1]
    assert metadata.lengths.query_lengths.tolist() == [2, 1]
    assert metadata.lengths.query_offsets.tolist() == [0, 2, 3]
    assert metadata.max_seq_len == 2
    assert metadata.num_prefill_tokens == 3
    assert metadata.num_decode_tokens == 0
    assert metadata.slot_mapping.tolist() == [40, 41, 80]
    assert metadata.is_prefill is True
    assert events[1] == ("model_forward",)
    assert events[2] == ("clear",)


def test_model_runner_clears_paged_context_on_exception() -> None:
    events: list[tuple[object, ...]] = []
    backend = object()

    class DeepseekV2PagedAttention(torch.nn.Module):
        @classmethod
        def set_paged_context(cls, backend: object, metadata: object) -> None:
            events.append(("set", backend, metadata))

        @classmethod
        def clear_paged_context(cls) -> None:
            events.append(("clear",))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class FailingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = DeepseekV2PagedAttention()

        def forward(
            self, input_ids: torch.Tensor, **kwargs: object
        ) -> _MockOutput:
            _ = (input_ids, kwargs)
            events.append(("model_forward",))
            raise RuntimeError("boom")

    runner = ModelRunner(
        FailingModel(),
        _MockEngine(backend),
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="boom"):
        _ = runner.execute(_make_decode_batch())

    assert events[0][0] == "set"
    assert events[1] == ("model_forward",)
    assert events[2] == ("clear",)


def test_model_runner_skips_paged_context_for_non_paged_models() -> None:
    class Model(torch.nn.Module):
        def forward(
            self, input_ids: torch.Tensor, **kwargs: object
        ) -> _MockOutput:
            _ = kwargs
            batch, seq_len = input_ids.shape
            logits = torch.zeros(batch, seq_len, 16, dtype=torch.float32)
            return _MockOutput(logits)

    runner = ModelRunner(
        Model(),
        _MockEngine(backend=object()),
        device=torch.device("cpu"),
    )

    logits = runner.execute(_make_prefill_batch())
    assert logits.shape == (3, 16)
