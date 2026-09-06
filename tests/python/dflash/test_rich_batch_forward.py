from __future__ import annotations

import types

import pytest
import torch

from moe_infinity.distributed.expert_executor import DistributedExpertExecutor
from moe_infinity.entrypoints.big_modeling import MoE
from moe_infinity.spec_decode._route_ahead_ctx import route_ahead_context
from moe_infinity.spec_decode.backends_rich import BatchedRichBackend
from moe_infinity.spec_decode.dflash import DFlashSpeculator
from moe_infinity.spec_decode.protocols import (
    RichBatchMetadata,
    RichForwardResult,
)
from tests.python.dflash import test_batched_spec as batched


class _BatchModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, dict[str, object]]] = []

    def __call__(self, input_ids: torch.Tensor, **kwargs: object) -> object:
        self.calls.append((input_ids.clone(), dict(kwargs)))
        values = input_ids.to(torch.float32).unsqueeze(-1)
        logits = torch.cat((values, values + 100), dim=-1)
        return types.SimpleNamespace(
            logits=logits,
            hidden_states=(values, values + 10),
            past_key_values=kwargs.get("past_key_values", "new-cache"),
        )

    def modules(self) -> list[object]:
        return []


def _shell(model: _BatchModel) -> MoE:
    shell = MoE.__new__(MoE)
    shell.model = model
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    shell._native_mla_cache = None
    shell._resolve_native_input_device = lambda: torch.device("cpu")
    return shell


def test_rich_batch_metadata_rejects_inconsistent_row_layout() -> None:
    with pytest.raises(ValueError, match="row_offsets"):
        RichBatchMetadata(row_offsets=(0, 2), row_lengths=(2, 1))


def test_native_rich_batch_runs_one_forward_and_preserves_rows_masks_positions() -> (
    None
):
    model = _BatchModel()
    shell = _shell(model)
    input_ids = torch.tensor([[0, 4, 5], [7, 8, 9]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    positions = torch.tensor([[0, 0, 1], [3, 4, 5]])
    cache = object()
    metadata = RichBatchMetadata(
        row_offsets=(0, 2, 5),
        row_lengths=(2, 3),
        attention_mask=mask,
        position_ids=positions,
        cache_handles=(cache, cache),
        request_contexts=("left", "right"),
        route_contexts=("route-left", "route-right"),
    )

    result = shell._native_model_forward_rich(input_ids, metadata)

    assert isinstance(result, RichForwardResult)
    assert len(model.calls) == 1
    called_ids, kwargs = model.calls[0]
    assert torch.equal(called_ids, input_ids)
    assert torch.equal(kwargs["attention_mask"], mask)
    assert torch.equal(kwargs["position_ids"], positions)
    assert result.logits.shape == (2, 3, 2)
    assert result.logits[0, 1].tolist() == [4.0, 104.0]
    assert result.logits[1, 0].tolist() == [7.0, 107.0]
    assert result.hidden_states[1][0, 1, 0].item() == 14
    assert result.cache_handles == (result.cache_handle, result.cache_handle)
    assert result.row_offsets == (0, 2, 5)
    assert result.row_lengths == (2, 3)


def test_native_rich_legacy_list_keeps_singleton_tuple_contract() -> None:
    model = _BatchModel()
    shell = _shell(model)

    result = shell._native_model_forward_rich([4, 5])

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0].shape == (1, 2, 2)


def _tiny_rich_spec(
    *, supports_batch: bool = True
) -> tuple[DFlashSpeculator, object, list[int]]:
    base, target = batched._tiny_spec()
    shell = MoE.__new__(MoE)
    shell.model = target
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    shell._native_mla_cache = None
    shell._resolve_native_input_device = lambda: torch.device("cpu")
    shell._configure_hook = lambda _ids: None
    calls = [0]
    original = shell._native_model_forward_rich

    def counted(*args: object, **kwargs: object) -> object:
        calls[0] += 1
        return original(*args, **kwargs)

    shell._native_model_forward_rich = counted
    shell._supports_native_rich_batch = lambda: supports_batch
    spec = DFlashSpeculator.from_models(
        shell, base.draft, config=base.config, device="cpu"
    )
    return spec, target, calls


def test_rich_backend_is_physical_only_for_declared_row_aware_wrapper() -> None:
    spec, _, _ = _tiny_rich_spec()
    backend = BatchedRichBackend(spec)

    assert backend.capabilities.supports_batch
    assert backend.capabilities.supports_rich_forward
    assert backend.wrapper_supported

    spec.moe._supports_native_rich_batch = lambda: False
    unsupported = BatchedRichBackend(spec)
    assert not unsupported.wrapper_supported


def test_rich_generate_uses_real_multirow_target_forwards_and_matches_independent() -> (
    None
):
    spec, target, calls = _tiny_rich_spec()
    prompts = [batched.PROMPT_A, batched.PROMPT_B]
    input_ids, mask, width = batched._left_pad(prompts)
    model_batch_sizes: list[int] = []
    hook = target.register_forward_pre_hook(
        lambda _module, args: model_batch_sizes.append(int(args[0].shape[0]))
    )

    try:
        output = spec.generate(
            input_ids, max_new_tokens=[5, 7], attention_mask=mask
        )
    finally:
        hook.remove()
    rows = batched._batched_new_tokens(output, spec, width)

    assert spec.rich_forward_batched is True
    assert calls[0] == len(model_batch_sizes) > 0
    assert all(size > 1 for size in model_batch_sizes)
    assert all(size == 2 for size in spec.rich_forward_batch_sizes)

    expected = []
    for prompt, budget in zip(prompts, (5, 7)):
        single, _, _ = _tiny_rich_spec()
        result = single.generate(torch.tensor([prompt]), max_new_tokens=budget)
        expected.append(result[0, len(prompt) :].tolist())
    assert rows == expected


def _run_named_rich_rows(names: list[str]) -> dict[str, list[int]]:
    prompts = {
        "a": [3, 7, 11, 2, 5],
        "b": [41, 6],
        "c": [10, 20, 30, 40],
    }
    spec, target, rich_calls = _tiny_rich_spec()
    input_ids, mask, width = batched._left_pad(
        [prompts[name] for name in names]
    )
    model_batch_sizes: list[int] = []
    hook = target.register_forward_pre_hook(
        lambda _module, args: model_batch_sizes.append(int(args[0].shape[0]))
    )

    try:
        output = spec.generate(input_ids, max_new_tokens=6, attention_mask=mask)
    finally:
        hook.remove()
    rows = batched._batched_new_tokens(output, spec, width)

    assert spec.rich_forward_batched is True
    assert (
        rich_calls[0]
        == len(model_batch_sizes)
        == len(spec.rich_forward_batch_sizes)
    )
    assert rich_calls[0] >= 1
    assert all(size > 1 for size in model_batch_sizes)
    assert all(
        size == len(names) and size > 1
        for size in spec.rich_forward_batch_sizes
    )
    assert len({tuple(row) for row in rows}) == len(rows)
    return {name: rows[row] for row, name in enumerate(names)}


def test_rich_physical_rows_match_independent_and_are_order_composition_invariant() -> (
    None
):
    forward = _run_named_rich_rows(["a", "b", "c"])
    reverse = _run_named_rich_rows(["c", "b", "a"])
    composed = _run_named_rich_rows(["a", "c"])

    independent: dict[str, list[int]] = {}
    prompts = {"a": [3, 7, 11, 2, 5], "b": [41, 6], "c": [10, 20, 30, 40]}
    for name, prompt in prompts.items():
        spec, _, _ = _tiny_rich_spec()
        output = spec.generate(torch.tensor([prompt]), max_new_tokens=6)
        independent[name] = output[0, len(prompt) :].tolist()

    assert forward == reverse == independent
    assert composed == {"a": independent["a"], "c": independent["c"]}


@pytest.mark.parametrize("wrapper_kind", ["mla", "hybrid"])
def test_unsupported_mla_or_hybrid_wrapper_falls_back_per_request(
    wrapper_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _, calls = _tiny_rich_spec()
    if wrapper_kind == "mla":
        spec.moe._get_mla_attention_modules = lambda: [object()]
    else:
        monkeypatch.setattr(
            spec.moe.model.config, "hybrid_attention", True, raising=False
        )
    spec.moe._supports_native_rich_batch = types.MethodType(
        MoE._supports_native_rich_batch, spec.moe
    )
    assert spec.moe._supports_native_rich_batch() is False
    ids, mask, _ = batched._left_pad([batched.PROMPT_A, batched.PROMPT_B])

    spec.generate(ids, max_new_tokens=2, attention_mask=mask)

    assert spec.rich_forward_batched is False
    assert spec.rich_forward_batch_sizes == []
    assert calls[0] >= 2


def test_executor_evidence_retains_every_rich_row_layer_union() -> None:
    executor = DistributedExpertExecutor.__new__(DistributedExpertExecutor)
    executor.prefetcher = None
    mask = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.bool
    )

    with route_ahead_context(row_offsets=(0, 2, 4)):
        executor._maybe_route_ahead_prefetch(7, mask, 3)

    assert executor.last_executor_evidence.actual_expert_union == frozenset(
        {(7, 0), (7, 1), (7, 2)}
    )
    assert (
        executor.last_executor_evidence.actual_expert_union_by_row
        == frozenset({(0, 7, 0), (0, 7, 1), (1, 7, 0), (1, 7, 2)})
    )
