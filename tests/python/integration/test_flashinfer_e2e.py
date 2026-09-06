# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false, reportUnannotatedClassAttribute=false, reportPrivateUsage=false, reportImplicitOverride=false, reportUnusedCallResult=false, reportUnusedFunction=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations

import types
from dataclasses import dataclass

import pytest
import torch

from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import KVCacheSpec
from moe_infinity.runtime.flashinfer_utils import HAS_FLASHINFER
from moe_infinity.serving.batch import BatchMetadata, split_prefill_decode_batch
from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.model_runner import ModelRunner
from moe_infinity.serving.sequence import SamplingParams


class _MockExpertTracer:
    def __init__(self) -> None:
        self._next = 0

    def create_entry(self) -> int:
        entry = self._next
        self._next += 1
        return entry


class _MockOffloadEngine:
    def __init__(self, attention_backend: object | None) -> None:
        self.request_id = 0
        self.expert_tracer = _MockExpertTracer()
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]
        self._attention_backend = attention_backend

    def _generate_request_id(self) -> int:
        rid = self.request_id
        self.request_id += 1
        return rid

    def get_attention_backend(self) -> object | None:
        return self._attention_backend


@dataclass
class _ForwardEvent:
    had_context: bool
    is_prefill: bool


class DeepseekV3PagedAttention(torch.nn.Module):
    _backend: object | None = None
    _metadata: object | None = None
    lifecycle_events: list[str] = []
    forward_events: list[_ForwardEvent] = []

    def __init__(
        self, num_heads: int = 2, num_kv_heads: int = 2, head_dim: int = 8
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    @classmethod
    def set_paged_context(cls, backend: object, metadata: object) -> None:
        cls._backend = backend
        cls._metadata = metadata
        cls.lifecycle_events.append("set")

    @classmethod
    def clear_paged_context(cls) -> None:
        cls.lifecycle_events.append("clear")
        cls._backend = None
        cls._metadata = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        metadata = type(self)._metadata
        backend = type(self)._backend
        is_prefill = bool(getattr(metadata, "is_prefill", False))
        type(self).forward_events.append(
            _ForwardEvent(
                had_context=backend is not None and metadata is not None,
                is_prefill=is_prefill,
            )
        )
        if backend is None or metadata is None:
            return torch.zeros_like(query)

        backend_forward = getattr(backend, "forward")
        return backend_forward(
            query=query,
            key=key,
            value=value,
            attention_metadata=metadata,
            scale=scale,
        )


class _MockPagedModel(torch.nn.Module):
    def __init__(self, vocab_size: int, device: torch.device) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = types.SimpleNamespace(
            vocab_size=vocab_size, eos_token_id=-1
        )
        self.device = device
        self.attn = DeepseekV3PagedAttention()

    def eval(self) -> "_MockPagedModel":
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: object = None,
    ) -> object:
        _ = (position_ids, use_cache, past_key_values)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        token_mask = attention_mask.to(dtype=torch.bool)
        packed_tokens = input_ids[token_mask]
        num_tokens = int(packed_tokens.numel())

        query = torch.zeros(
            (num_tokens, self.attn.num_heads, self.attn.head_dim),
            dtype=torch.float32,
            device=input_ids.device,
        )
        if num_tokens > 0:
            query = query + packed_tokens.to(dtype=torch.float32).view(-1, 1, 1)

        key = query[:, : self.attn.num_kv_heads, :].clone()
        value = key + 0.5
        attn_out = self.attn(query=query, key=key, value=value, scale=1.0)

        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            fill_value=-1e9,
            dtype=torch.float32,
            device=input_ids.device,
        )
        next_ids = ((input_ids + 1) % self.vocab_size).to(dtype=torch.long)
        peak_logit = (
            float(attn_out.mean().item()) if attn_out.numel() > 0 else 0.0
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), peak_logit)
        return types.SimpleNamespace(logits=logits)


class _FakeFlashInferRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        _ = kv_cache
        self.calls += 1
        return torch.zeros_like(query)


def _make_backend(
    device: torch.device, dtype: torch.dtype
) -> PagedAttentionBackend:
    spec = KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=dtype,
        block_size=8,
    )
    return PagedAttentionBackend(spec=spec, num_gpu_blocks=16, device=device)


def _make_engine_config(dtype: str) -> dict[str, object]:
    return {
        "device_memory_ratio": 0.5,
        "kv_cache_ratio": 0.5,
        "max_batch_size": 8,
        "max_tokens_per_step": 16,
        "block_size": 8,
        "num_layers": 1,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype": dtype,
        "eos_token_id": -1,
        "num_kv_blocks": 16,
    }


@pytest.fixture(autouse=True)
def _reset_mock_paged_attention() -> None:
    DeepseekV3PagedAttention._backend = None
    DeepseekV3PagedAttention._metadata = None
    DeepseekV3PagedAttention.lifecycle_events = []
    DeepseekV3PagedAttention.forward_events = []


@pytest.mark.skipif(
    not (HAS_FLASHINFER and torch.cuda.is_available()),
    reason="requires FlashInfer and CUDA",
)
def test_e2e_flashinfer_prefill_decode_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    backend = _make_backend(device=device, dtype=torch.float16)
    if backend._fi_kv_cache is None:
        pytest.skip("FlashInfer backend did not initialize KV cache")

    fake_prefill = _FakeFlashInferRunner()
    fake_decode = _FakeFlashInferRunner()
    backend._fi_prefill = fake_prefill
    backend._fi_decode = fake_decode
    backend._use_flashinfer = True

    plan_calls: list[str] = []
    monkeypatch.setattr(
        backend,
        "_call_prefill_plan",
        lambda *args, **kwargs: plan_calls.append("prefill"),
    )
    monkeypatch.setattr(
        backend,
        "_call_decode_plan",
        lambda *args, **kwargs: plan_calls.append("decode"),
    )

    engine = ContinuousBatchingEngine(
        model=_MockPagedModel(vocab_size=97, device=device),
        engine=_MockOffloadEngine(attention_backend=backend),
        config=_make_engine_config(dtype="float16"),
        tokenizer=None,
    )

    engine.add_request(
        request_id="flashinfer-req",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
    )

    first = engine.step()
    second = engine.step()
    third = engine.step()
    all_outputs = [*first, *second, *third]

    assert len(all_outputs) == 3
    assert [output.token_id for output in all_outputs] == [12, 13, 14]
    assert all(0 <= output.token_id < 97 for output in all_outputs)
    assert all(isinstance(output.token_id, int) for output in all_outputs)
    assert all_outputs[-1].finished is True
    assert not engine.has_pending_requests()

    assert "prefill" in plan_calls
    assert "decode" in plan_calls
    assert fake_prefill.calls >= 1
    assert fake_decode.calls >= 1


@pytest.mark.skipif(
    not (HAS_FLASHINFER and torch.cuda.is_available()),
    reason="requires FlashInfer and CUDA",
)
def test_e2e_flashinfer_chunked_prefill_with_active_decode() -> None:
    device = torch.device("cuda")
    backend = _make_backend(device=device, dtype=torch.float16)
    engine = ContinuousBatchingEngine(
        model=_MockPagedModel(vocab_size=97, device=device),
        engine=_MockOffloadEngine(attention_backend=backend),
        config={
            **_make_engine_config(dtype="float16"),
            "enable_chunked_prefill": True,
            "prefill_chunk_size": 2,
            "max_tokens_per_step": 3,
        },
    )
    engine.add_request(
        "decode", [10], SamplingParams(temperature=0.0, max_tokens=4)
    )
    first = engine.step()
    assert [row.token_id for row in first] == [11]
    engine.add_request(
        "long",
        [20, 21, 22, 23, 24],
        SamplingParams(temperature=0.0, max_tokens=1),
    )

    observed = [*first]
    while engine.has_pending_requests():
        observed.extend(engine.step())

    assert engine.get_request_n_outputs("decode") == [[11, 12, 13, 14]]
    assert engine.get_request_n_outputs("long") == [[25]]
    assert engine.get_stats()["num_prefill_chunks"] >= 4


def _make_chunked_flashinfer_engine() -> ContinuousBatchingEngine:
    device = torch.device("cuda")
    backend = _make_backend(device=device, dtype=torch.float16)
    return ContinuousBatchingEngine(
        model=_MockPagedModel(vocab_size=97, device=device),
        engine=_MockOffloadEngine(attention_backend=backend),
        config={
            **_make_engine_config(dtype="float16"),
            "enable_chunked_prefill": True,
            "prefill_chunk_size": 2,
            "max_tokens_per_step": 2,
        },
    )


def _add_parity_request(engine: ContinuousBatchingEngine) -> None:
    engine.add_request(
        "long",
        [20, 21, 22, 23, 24],
        SamplingParams(temperature=0.0, max_tokens=1),
    )


def _finish_and_capture_terminal(
    engine: ContinuousBatchingEngine,
) -> tuple[torch.Tensor, list[int]]:
    captured: dict[str, torch.Tensor] = {}
    original_execute = engine._execute_batch

    def _capture(batch: BatchMetadata) -> torch.Tensor:
        logits = original_execute(batch)
        captured["logits"] = logits.detach().clone()
        return logits

    engine._execute_batch = _capture  # type: ignore[method-assign]
    outputs: list[int] = []
    while engine.has_pending_requests():
        for row in engine.step():
            outputs.append(row.token_id)
    return captured["logits"], outputs


@pytest.mark.skipif(
    not (HAS_FLASHINFER and torch.cuda.is_available()),
    reason="requires FlashInfer and CUDA",
)
def test_partial_prefill_preemption_preserves_logits_and_output() -> None:
    uninterrupted = _make_chunked_flashinfer_engine()
    resumed = _make_chunked_flashinfer_engine()
    _add_parity_request(uninterrupted)
    _add_parity_request(resumed)

    assert uninterrupted.step() == []
    assert resumed.step() == []
    preempted = resumed.scheduler._preempt_oldest_running_group()
    assert preempted == [0]
    resumed.scheduler._recover_swapped_groups(list(resumed.scheduler._swapped))

    uninterrupted_logits, uninterrupted_output = _finish_and_capture_terminal(
        uninterrupted
    )
    resumed_logits, resumed_output = _finish_and_capture_terminal(resumed)

    torch.testing.assert_close(resumed_logits, uninterrupted_logits)
    assert resumed_output == uninterrupted_output
    assert resumed.kv_cache.block_allocator.num_free_blocks == (
        uninterrupted.kv_cache.block_allocator.num_free_blocks
    )


def test_e2e_serving_engine_without_flashinfer() -> None:
    device = torch.device("cpu")
    backend = _make_backend(device=device, dtype=torch.float32)
    backend._use_flashinfer = False
    backend._fi_prefill = None
    backend._fi_decode = None
    backend._fi_kv_cache = None

    engine = ContinuousBatchingEngine(
        model=_MockPagedModel(vocab_size=64, device=device),
        engine=_MockOffloadEngine(attention_backend=backend),
        config=_make_engine_config(dtype="float32"),
        tokenizer=None,
    )

    engine.add_request(
        request_id="sdpa-req",
        prompt_token_ids=[5, 6],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
    )

    outputs = engine.run_until_done()

    assert outputs["sdpa-req"] == [7, 8, 9, 10]
    assert backend._flashinfer_enabled() is False
    assert DeepseekV3PagedAttention.forward_events
    assert any(
        event.is_prefill for event in DeepseekV3PagedAttention.forward_events
    )
    assert any(
        not event.is_prefill
        for event in DeepseekV3PagedAttention.forward_events
    )


def test_e2e_batch_splitting_with_mixed_sequences() -> None:
    batch = BatchMetadata(
        seq_ids=[101, 102, 103],
        input_token_ids=[11, 12, 21, 31, 32, 33],
        seq_lengths=[2, 1, 3],
        context_lengths=[0, 4, 1],
        is_prefill=[True, False, True],
        block_tables=[[0], [4, 5], [2]],
        token_offsets=[0, 2, 3, 6],
        sampling_params=[SamplingParams(), SamplingParams(), SamplingParams()],
    )

    split = split_prefill_decode_batch(batch)
    assert split.prefill_indices == [0, 2]
    assert split.decode_indices == [1]
    assert split.prefill_batch is not None
    assert split.decode_batch is not None

    assert split.prefill_batch.input_token_ids == [11, 12, 31, 32, 33]
    assert split.prefill_batch.token_offsets == [0, 2, 5]
    assert split.decode_batch.input_token_ids == [21]
    assert split.decode_batch.token_offsets == [0, 1]

    prefill_out = torch.tensor(
        [[100], [101], [102], [103], [104]],
        dtype=torch.float32,
    )
    decode_out = torch.tensor([[200]], dtype=torch.float32)

    recombined = split.recombine_outputs(
        prefill_outputs=prefill_out,
        decode_outputs=decode_out,
    )

    expected = torch.tensor(
        [[100], [101], [200], [102], [103], [104]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(recombined, expected)


def test_e2e_paged_context_lifecycle_with_mock_model() -> None:
    events: list[str] = []

    class DeepseekV2PagedAttention(torch.nn.Module):
        _backend: object | None = None
        _metadata: object | None = None

        @classmethod
        def set_paged_context(cls, backend: object, metadata: object) -> None:
            cls._backend = backend
            cls._metadata = metadata
            events.append("set")

        @classmethod
        def clear_paged_context(cls) -> None:
            events.append("clear")
            cls._backend = None
            cls._metadata = None

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
        ) -> torch.Tensor:
            backend_forward = getattr(type(self)._backend, "forward")
            return backend_forward(
                query=query,
                key=key,
                value=value,
                attention_metadata=type(self)._metadata,
            )

    class _LifecycleBackend:
        def __init__(self) -> None:
            self.spec = types.SimpleNamespace(block_size=4)

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_metadata: object,
            scale: float | None = None,
        ) -> torch.Tensor:
            _ = (key, value, attention_metadata, scale)
            return torch.zeros_like(query)

    class _LifecycleModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=32)
            self.attn = DeepseekV2PagedAttention()

        def eval(self) -> "_LifecycleModel":
            return self

        def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            use_cache: bool = True,
            past_key_values: object = None,
        ) -> object:
            _ = (position_ids, use_cache, past_key_values)
            events.append("forward")

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            packed = input_ids[attention_mask.to(dtype=torch.bool)]
            num_tokens = int(packed.numel())
            q = torch.zeros((num_tokens, 2, 8), dtype=torch.float32)
            k = torch.zeros((num_tokens, 2, 8), dtype=torch.float32)
            v = torch.zeros((num_tokens, 2, 8), dtype=torch.float32)
            _ = self.attn(q, k, v)

            batch_size, seq_len = input_ids.shape
            logits = torch.full((batch_size, seq_len, 32), -1e9)
            next_ids = ((input_ids + 1) % 32).to(dtype=torch.long)
            logits.scatter_(2, next_ids.unsqueeze(-1), 0.0)
            return types.SimpleNamespace(logits=logits)

    model = _LifecycleModel()
    backend = _LifecycleBackend()
    runner = ModelRunner(model=model, engine=_MockOffloadEngine(backend))

    batch = BatchMetadata(
        seq_ids=[1],
        input_token_ids=[9, 10],
        seq_lengths=[2],
        context_lengths=[0],
        is_prefill=[True],
        block_tables=[[0]],
        token_offsets=[0, 2],
        sampling_params=[SamplingParams(temperature=0.0)],
    )

    logits = runner.execute(batch)

    assert logits.shape == (2, 32)
    assert events == ["set", "forward", "clear"]
