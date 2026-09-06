from __future__ import annotations

import types

import torch

from moe_infinity.serving.batch import BatchMetadata
from moe_infinity.serving.model_runner import ModelRunner
from moe_infinity.serving.sequence import SamplingParams
from moe_infinity.spec_decode.protocols import RichForwardResult


class _RichModel:
    def __init__(self) -> None:
        self.calls = 0
        self.config = types.SimpleNamespace(vocab_size=2)

    def eval(self) -> None:
        pass

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> object:
        self.calls += 1
        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        return types.SimpleNamespace(
            logits=torch.cat((hidden, hidden + 1), dim=-1),
            hidden_states=(hidden, hidden + 10),
            past_key_values=kwargs.get("past_key_values", "cache"),
        )

    def modules(self) -> list[object]:
        return []


class _Engine:
    request_id = 0
    expert_layer_modules: list[object] = []
    expert_tracer = None


def _batch() -> BatchMetadata:
    return BatchMetadata(
        seq_ids=[41, 42],
        input_token_ids=[3, 4, 9],
        seq_lengths=[2, 1],
        context_lengths=[0, 7],
        is_prefill=[True, False],
        block_tables=[[2], [5]],
        token_offsets=[0, 2, 3],
        sampling_params=[SamplingParams(), SamplingParams()],
    )


def test_model_runner_rich_execute_returns_packed_hidden_and_row_cache_handles() -> (
    None
):
    model = _RichModel()
    runner = ModelRunner(model, _Engine(), device=torch.device("cpu"))
    handles = (object(), object())

    result = runner.execute_rich(_batch(), cache_handles=handles)

    assert isinstance(result, RichForwardResult)
    assert model.calls == 1
    assert result.logits.shape == (3, 2)
    assert result.hidden_states[0].shape == (3, 1)
    assert result.cache_handles == handles
    assert result.row_offsets == (0, 2, 3)
    assert result.row_lengths == (2, 1)


def test_standard_model_runner_api_remains_logits_only() -> None:
    runner = ModelRunner(_RichModel(), _Engine(), device=torch.device("cpu"))

    logits = runner.execute(_batch())

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (3, 2)


def test_empty_rich_runner_result_keeps_row_alignment_without_forward() -> None:
    model = _RichModel()
    runner = ModelRunner(model, _Engine(), device=torch.device("cpu"))
    batch = BatchMetadata(
        seq_ids=[9],
        input_token_ids=[],
        seq_lengths=[0],
        context_lengths=[4],
        is_prefill=[False],
        block_tables=[[1]],
        token_offsets=[0, 0],
        sampling_params=[SamplingParams()],
    )

    result = runner.execute_rich(batch, cache_handles=("paged-9",))

    assert isinstance(result, RichForwardResult)
    assert result.logits.shape[0] == 0
    assert result.hidden_states == ()
    assert result.cache_handles == ("paged-9",)
    assert result.row_offsets == (0, 0)
    assert model.calls == 0
