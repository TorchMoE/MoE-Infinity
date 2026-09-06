# pyright: reportAny=false, reportImplicitOverride=false

import importlib.util
import sys
import types
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


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
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

from moe_infinity.runtime.attention_types import PagedBatchLengths  # noqa: E402

SamplingParams = _SEQUENCE_MODULE.SamplingParams
BatchMetadata = _BATCH_MODULE.BatchMetadata
ModelRunner = _MODEL_RUNNER_MODULE.ModelRunner


class _MockOutput:
    logits: torch.Tensor

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class MockModel:
    vocab_size: int
    rank3_logits: bool
    config: types.SimpleNamespace
    eval_called: bool

    def __init__(
        self, vocab_size: int = 100, rank3_logits: bool = True
    ) -> None:
        self.vocab_size = vocab_size
        self.rank3_logits = rank3_logits
        self.config = types.SimpleNamespace(vocab_size=vocab_size)
        self.last_kwargs: dict[str, object] = {}
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> _MockOutput:
        self.last_kwargs = {"input_ids": input_ids, **kwargs}
        if self.rank3_logits:
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
        else:
            batch_size = input_ids.shape[0]
            logits = torch.randn(batch_size, self.vocab_size)
        return _MockOutput(logits=logits)


class _MockExpertTracer:
    def __init__(self) -> None:
        self.created: list[int] = []

    def create_entry(self) -> int:
        seq_id = len(self.created)
        self.created.append(seq_id)
        return seq_id


class MockOffloadEngine:
    request_id: int
    expert_tracer: _MockExpertTracer
    expert_layer_modules: list[types.SimpleNamespace]

    def __init__(self) -> None:
        self.request_id = 0
        self.expert_tracer = _MockExpertTracer()
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]

    def _generate_request_id(self) -> int:
        request_id = self.request_id
        self.request_id += 1
        return request_id


class _NoForwardModel(MockModel):
    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> _MockOutput:
        _ = input_ids
        _ = kwargs
        raise AssertionError("forward should not be called")


def _make_batch() -> BatchMetadata:
    return BatchMetadata(
        seq_ids=[10, 11],
        input_token_ids=[11, 12, 13, 21],
        lengths=PagedBatchLengths(
            query_lengths=[3, 1],
            query_offsets=[0, 3, 4],
            context_lengths=[0, 4],
            kv_seq_lengths=[3, 5],
        ),
        is_prefill=[True, False],
        block_tables=[[0], [1]],
        sampling_params=[SamplingParams(), SamplingParams()],
    )


def test_prepare_inputs_builds_padded_batch() -> None:
    runner = ModelRunner(MockModel(), MockOffloadEngine())
    batch = _make_batch()

    model_inputs = runner.prepare_inputs(batch)

    assert model_inputs["input_ids"].tolist() == [[11, 12, 13], [21, 0, 0]]
    assert model_inputs["position_ids"].tolist() == [[0, 1, 2], [4, 0, 0]]
    assert model_inputs["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_execute_runs_forward_and_returns_packed_logits() -> None:
    model = MockModel(vocab_size=32, rank3_logits=True)
    engine = MockOffloadEngine()
    runner = ModelRunner(model, engine)
    batch = _make_batch()

    logits = runner.execute(batch, past_key_values="kv")

    assert logits.shape == (4, 32)
    assert model.eval_called
    assert model.last_kwargs["use_cache"] is True
    assert model.last_kwargs["past_key_values"] == "kv"
    assert engine.request_id == 1
    assert engine.expert_tracer.created == [0, 1]
    assert runner.seq_id_list == [0, 1]
    assert engine.expert_layer_modules[0].seq_id_list == [0, 1]


def test_execute_supports_rank2_logits_for_decode_batches() -> None:
    model = MockModel(vocab_size=16, rank3_logits=False)
    runner = ModelRunner(model, MockOffloadEngine())
    batch = BatchMetadata(
        seq_ids=[1, 2],
        input_token_ids=[30, 31],
        lengths=PagedBatchLengths(
            query_lengths=[1, 1],
            query_offsets=[0, 1, 2],
            context_lengths=[8, 3],
            kv_seq_lengths=[9, 4],
        ),
        is_prefill=[False, False],
        block_tables=[[0], [1]],
        sampling_params=[SamplingParams(), SamplingParams()],
    )

    logits = runner.execute(batch)

    assert logits.shape == (2, 16)


def test_execute_empty_batch_skips_forward() -> None:
    model = _NoForwardModel(vocab_size=7)
    runner = ModelRunner(model, MockOffloadEngine())
    batch = BatchMetadata(
        seq_ids=[99],
        input_token_ids=[],
        lengths=PagedBatchLengths(
            query_lengths=[0],
            query_offsets=[0, 0],
            context_lengths=[10],
            kv_seq_lengths=[10],
        ),
        is_prefill=[False],
        block_tables=[[0]],
        sampling_params=[SamplingParams()],
    )

    logits = runner.execute(batch)

    assert logits.shape == (0, 7)


from transformers.models.qwen3_moe.configuration_qwen3_moe import (  # noqa: E402
    Qwen3MoeConfig,
)

from moe_infinity.models.paged_attention_registry import (  # noqa: E402
    PagedAttentionLayerRegistry,
)
from moe_infinity.models.qwen3_paged_attention import (  # noqa: E402
    Qwen3PagedAttention,
)
from moe_infinity.runtime.attention_backend import (  # noqa: E402
    PagedAttentionBackend,
)
from moe_infinity.runtime.paged_kv_storage import (  # noqa: E402
    PagedKVStorage,
    PagedKVStorageSpec,
)


def _qwen3_config(num_layers: int = 2) -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=num_layers,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        vocab_size=64,
    )


class _PagedEngine:
    def __init__(self, backend: object, block_size: int) -> None:
        self.request_id = 0
        self.expert_tracer = _MockExpertTracer()
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]
        self._attention_backend = backend
        self.kv_cache = types.SimpleNamespace(
            block_size=block_size, storage=getattr(backend, "storage", None)
        )

    def _generate_request_id(self) -> int:
        request_id = self.request_id
        self.request_id += 1
        return request_id

    def get_attention_backend(self) -> object:
        return self._attention_backend


class _TwoLayerQwen3(torch.nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            [
                Qwen3PagedAttention(config, layer_idx=0),
                Qwen3PagedAttention(config, layer_idx=1),
            ]
        )
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size)
        head_dim = config.head_dim

        def _rotary(seq_len: int, device: torch.device):
            cos = torch.ones(1, seq_len, head_dim, device=device)
            sin = torch.zeros(1, seq_len, head_dim, device=device)
            return cos, sin

        self._rotary = _rotary

    def eval(self) -> "_TwoLayerQwen3":
        return super().eval()

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> _MockOutput:
        _ = kwargs
        hidden = self.embed(input_ids)
        cos, sin = self._rotary(input_ids.shape[1], input_ids.device)
        for layer in self.layers:
            attn_out, _ = layer(
                hidden_states=hidden,
                position_embeddings=(cos, sin),
                attention_mask=None,
            )
            hidden = hidden + attn_out
        return _MockOutput(self.lm_head(hidden))


def _make_graph_safe_native_paged_runner(
    *,
    storage_device: torch.device | None = None,
    runner_device: torch.device | None = None,
):
    storage_device = storage_device or torch.device("cpu")
    runner_device = runner_device or storage_device
    spec = PagedKVStorageSpec(
        num_layers=2,
        num_blocks=16,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=storage_device,
    )
    storage = PagedKVStorage(spec)
    backend = PagedAttentionBackend(storage=storage, use_flashinfer=False)
    config = _qwen3_config(num_layers=2)
    model = _TwoLayerQwen3(config).to(storage.spec.device)
    registry = PagedAttentionLayerRegistry.register(model, backend, storage)
    engine = _PagedEngine(backend, block_size=storage.block_size)
    runner = ModelRunner(
        model,
        engine,
        device=runner_device,
        paged_kv_storage=storage,
        paged_attention_registry=registry,
    )
    batch = BatchMetadata(
        seq_ids=[1, 2],
        input_token_ids=[5, 6],
        seq_lengths=[1, 1],
        context_lengths=[8, 3],
        is_prefill=[False, False],
        block_tables=[[0, 1, 2], [3]],
        token_offsets=[0, 1, 2],
        sampling_params=[SamplingParams(), SamplingParams()],
    )
    return runner, batch, storage


def test_prepared_native_paged_decode_preserves_side_effects_and_pointers() -> (
    None
):
    runner, batch, storage = _make_graph_safe_native_paged_runner()
    prepared = runner.allocate_decode_buffers(batch_bucket=2, context_bucket=16)
    pointers = prepared.data_ptrs()
    runner.copy_decode_batch(batch, prepared, scratch_block_ids=[])
    runner.prepare_batch_side_effects(batch)
    logits = runner.forward_prepared_decode(prepared)

    assert logits.shape[0] == 2
    assert prepared.data_ptrs() == pointers
    assert prepared.attention_metadata.kv_storage_owner_id == storage.owner_id
    assert prepared.attention_metadata.seq_lens.tolist() == [9, 4]


def test_copy_decode_batch_rejects_block_id_outside_authoritative_storage() -> (
    None
):
    runner, batch, storage = _make_graph_safe_native_paged_runner()
    batch.block_tables[0] = [storage.num_blocks]
    prepared = runner.allocate_decode_buffers(batch_bucket=2, context_bucket=16)
    with pytest.raises(ValueError, match="block id"):
        runner.copy_decode_batch(batch, prepared, scratch_block_ids=[])


def test_allocate_decode_buffers_requires_exact_runner_storage_device() -> None:
    runner, _, _ = _make_graph_safe_native_paged_runner(
        storage_device=torch.device("cpu"),
        runner_device=torch.device("cuda:0"),
    )
    with pytest.raises(ValueError, match="device"):
        runner.allocate_decode_buffers(batch_bucket=2, context_bucket=16)


def test_every_prepared_buffer_uses_exact_storage_device() -> None:
    runner, _, storage = _make_graph_safe_native_paged_runner()
    prepared = runner.allocate_decode_buffers(batch_bucket=2, context_bucket=16)
    tensors = prepared.tensor_values()
    assert tensors
    assert all(tensor.device == storage.spec.device for tensor in tensors)
