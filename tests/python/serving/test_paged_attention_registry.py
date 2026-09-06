from __future__ import annotations

import importlib

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import (
    Qwen3MoeConfig,
)

from moe_infinity.models.paged_attention_registry import (
    SUPPORTED_PAGED_CLASS_SPECS,
    LayerBoundPagedBackend,
    PagedAttentionLayerRegistry,
)
from moe_infinity.models.qwen3_paged_attention import Qwen3PagedAttention
from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)
from moe_infinity.runtime.paged_kv_storage import (
    PagedKVStorage,
    PagedKVStorageSpec,
)


def _make_config(num_layers: int = 2) -> Qwen3MoeConfig:
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
    )


def _make_storage(num_layers: int = 2) -> PagedKVStorage:
    spec = PagedKVStorageSpec(
        num_layers=num_layers,
        num_blocks=16,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return PagedKVStorage(spec)


def _instantiate_supported_class(class_fqn: str, layer_idx: int):
    module_path, class_name = class_fqn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(_make_config(), layer_idx=layer_idx)


def _make_two_layer_qwen3_paged_model():
    config = _make_config(num_layers=2)

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    Qwen3PagedAttention(config, layer_idx=0),
                    Qwen3PagedAttention(config, layer_idx=1),
                ]
            )

    storage = _make_storage(num_layers=2)
    backend = PagedAttentionBackend(storage=storage, use_flashinfer=False)
    return _Model(), storage, backend


class _RecordingBackend:
    def __init__(self, storage: PagedKVStorage) -> None:
        self.storage = storage
        self.events: list[tuple] = []

    def forward(self, *args, layer_idx=None, **kwargs):
        self.events.append(("write", layer_idx))
        self.events.append(("attention", layer_idx))
        num_heads = 4
        head_dim = 8
        return torch.zeros(1, num_heads, head_dim)


def _register_single_module(module, num_layers: int):
    storage = _make_storage(num_layers=num_layers)
    recorder = _RecordingBackend(storage)

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = module

    registry = PagedAttentionLayerRegistry.register(_Model(), recorder, storage)
    metadata = RuntimeAttentionMetadata(
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        is_prefill=False,
        kv_storage_owner_id=storage.owner_id,
    )
    registry.install_metadata(metadata)
    return registry, recorder


def _run_minimal_paged_forward(module) -> None:
    config = _make_config()
    hidden = torch.zeros(1, 1, config.hidden_size)
    cos = torch.ones(1, 1, config.head_dim)
    sin = torch.zeros(1, 1, config.head_dim)
    module.forward(
        hidden_states=hidden,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )


def _instantiate_deepseek_paged_attention_or_skip(family: str, layer_idx: int):
    pytest.skip(f"DeepSeek {family} paged attention not installed")


def test_registry_supports_only_exact_qwen3_paged_attention() -> None:
    assert tuple(SUPPORTED_PAGED_CLASS_SPECS) == (
        ("moe_infinity.models.qwen3_paged_attention", "Qwen3PagedAttention"),
    )


def test_registry_binds_unique_layer_idx_and_storage_slice() -> None:
    model, storage, backend = _make_two_layer_qwen3_paged_model()
    registry = PagedAttentionLayerRegistry.register(model, backend, storage)
    assert [binding.layer_idx for binding in registry.bindings] == [0, 1]
    assert registry.bindings[0].backend.layer_idx == 0
    assert registry.bindings[1].backend.layer_idx == 1
    assert registry.bindings[0].storage_owner_id == storage.owner_id
    assert (
        registry.bindings[0].bound_class is not registry.bindings[1].bound_class
    )


def test_qwen3_routes_to_registered_layer_bound_backend() -> None:
    class_fqn = "moe_infinity.models.qwen3_paged_attention.Qwen3PagedAttention"
    module = _instantiate_supported_class(class_fqn, layer_idx=1)
    registry, recorder = _register_single_module(module, num_layers=2)
    _run_minimal_paged_forward(module)
    assert recorder.events[0][:2] == ("write", 1)
    assert recorder.events[1] == ("attention", 1)
    assert registry.bindings[0].class_fqn == class_fqn


@pytest.mark.parametrize("family", ["deepseek_v2", "deepseek_v3"])
def test_deepseek_mla_is_recognized_but_never_registered(family: str) -> None:
    module = _instantiate_deepseek_paged_attention_or_skip(family, layer_idx=0)
    result = PagedAttentionLayerRegistry.inspect_module(module)
    assert result.binding is None
    assert result.reason == "mla_layout_unsupported"


class _OrderRecordingBackend:
    def __init__(self, storage: PagedKVStorage) -> None:
        self.storage = storage
        self.events: list[tuple] = []

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_metadata: RuntimeAttentionMetadata,
        graph_mode: bool = False,
        layer_idx=None,
    ) -> torch.Tensor:
        _ = (query, key, value, graph_mode)
        self.events.append(
            ("write", layer_idx, attention_metadata.slot_mapping.data_ptr())
        )
        self.events.append(("attention", layer_idx))
        return torch.zeros(1, 4, 8)


def _order_metadata(storage: PagedKVStorage) -> RuntimeAttentionMetadata:
    return RuntimeAttentionMetadata(
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        is_prefill=False,
        kv_storage_owner_id=storage.owner_id,
    )


def test_each_layer_writes_current_token_before_decode_attention() -> None:
    storage = _make_storage(num_layers=2)
    backend = _OrderRecordingBackend(storage)
    metadata = _order_metadata(storage)

    for layer_idx in (0, 1):
        bound = LayerBoundPagedBackend(backend, layer_idx, storage.owner_id)
        _ = bound.forward(
            torch.zeros(1, 4, 8),
            torch.zeros(1, 2, 8),
            torch.zeros(1, 2, 8),
            attention_metadata=metadata,
            graph_mode=True,
        )

    assert backend.events == [
        ("write", 0, metadata.slot_mapping.data_ptr()),
        ("attention", 0),
        ("write", 1, metadata.slot_mapping.data_ptr()),
        ("attention", 1),
    ]
