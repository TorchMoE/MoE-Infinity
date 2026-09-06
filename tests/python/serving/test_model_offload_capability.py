from __future__ import annotations

import types

import pytest
import torch

from moe_infinity.models.paged_attention_registry import (
    LayerBoundPagedBackend,
    PagedAttentionLayerRegistry,
    PagedLayerBinding,
)
from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import DecodeGraphCapability
from moe_infinity.runtime.paged_kv_storage import (
    PagedKVStorage,
    PagedKVStorageSpec,
)
from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.model_runner import ModelRunner


class _ExplicitCapabilityProvider:
    def __init__(
        self,
        *,
        active_model_hooks: bool = False,
        archer_callbacks: bool = False,
        transfer_scheduler_active: bool = False,
        expert_dispatcher_active: bool = False,
        kv_offload_active: bool = False,
        dynamic_allocations: bool = False,
    ) -> None:
        self._active_model_hooks = active_model_hooks
        self._archer_callbacks = archer_callbacks
        self._transfer_scheduler_active = transfer_scheduler_active
        self._expert_dispatcher_active = expert_dispatcher_active
        self._kv_offload_active = kv_offload_active
        self._dynamic_allocations = dynamic_allocations

    def decode_graph_capability(self) -> DecodeGraphCapability:
        if self._active_model_hooks:
            return DecodeGraphCapability(False, "active_model_hooks")
        if self._archer_callbacks:
            return DecodeGraphCapability(False, "archer_callbacks")
        if self._transfer_scheduler_active:
            return DecodeGraphCapability(False, "transfer_scheduler")
        if self._expert_dispatcher_active:
            return DecodeGraphCapability(False, "expert_dispatcher")
        if self._kv_offload_active:
            return DecodeGraphCapability(False, "kv_offload")
        if self._dynamic_allocations:
            return DecodeGraphCapability(False, "dynamic_allocations")
        return DecodeGraphCapability(True, "eligible")


def _make_storage(
    *,
    num_layers: int = 2,
    num_blocks: int = 16,
    block_size: int = 4,
    device: torch.device | None = None,
) -> PagedKVStorage:
    spec = PagedKVStorageSpec(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=device or torch.device("cpu"),
    )
    return PagedKVStorage(spec)


class _NonPagedModel(torch.nn.Module):
    def modules(self):
        return iter([self])


@pytest.mark.parametrize(
    ("runtime_state", "reason"),
    [
        ({"active_model_hooks": True}, "active_model_hooks"),
        ({"archer_callbacks": True}, "archer_callbacks"),
        ({"transfer_scheduler_active": True}, "transfer_scheduler"),
        ({"expert_dispatcher_active": True}, "expert_dispatcher"),
        ({"kv_offload_active": True}, "kv_offload"),
        ({"dynamic_allocations": True}, "dynamic_allocations"),
    ],
)
def test_runtime_capability_rejects_each_hazard(
    runtime_state, reason: str
) -> None:
    provider = _ExplicitCapabilityProvider(**runtime_state)
    capability = provider.decode_graph_capability()
    assert capability == DecodeGraphCapability(False, reason)


def test_missing_capability_provider_is_rejected() -> None:
    runner = ModelRunner(
        _NonPagedModel(), engine=object(), device=torch.device("cpu")
    )
    assert runner.decode_graph_capability().reason == "missing_capability"


def test_non_paged_model_is_rejected_even_when_resident() -> None:
    provider = _ExplicitCapabilityProvider()
    runner = ModelRunner(
        _NonPagedModel(),
        engine=object(),
        device=torch.device("cpu"),
        decode_graph_capability_provider=provider,
    )
    assert runner.decode_graph_capability().reason == "native_paged_required"


def _native_paged_runner(
    *,
    runtime_storage: PagedKVStorage | None = None,
    backend_storage: PagedKVStorage | None = None,
    runner_device: torch.device | None = None,
) -> ModelRunner:
    if backend_storage is None:
        backend_storage = _make_storage()
    if runtime_storage is None:
        runtime_storage = backend_storage
    backend = PagedAttentionBackend(
        storage=backend_storage, use_flashinfer=False
    )
    engine = types.SimpleNamespace(
        get_attention_backend=lambda: backend,
        kv_cache=None,
    )
    registry = PagedAttentionLayerRegistry(bindings=[], reason="eligible")
    return ModelRunner(
        _NonPagedModel(),
        engine=engine,
        device=runner_device or backend_storage.spec.device,
        paged_kv_storage=runtime_storage,
        paged_attention_registry=registry,
        decode_graph_capability_provider=_ExplicitCapabilityProvider(),
    )


def test_storage_identity_mismatch_is_rejected() -> None:
    runner = _native_paged_runner(
        runtime_storage=_make_storage(), backend_storage=_make_storage()
    )
    assert runner.decode_graph_capability().reason == "kv_storage_mismatch"


def test_flashinfer_plan_path_is_rejected() -> None:
    storage = _make_storage()
    backend = PagedAttentionBackend(storage=storage, use_flashinfer=False)
    backend._use_flashinfer = True
    assert backend.decode_graph_capability().reason == "flashinfer_plan_path"


def _proven_binding(
    storage: PagedKVStorage,
    backend: PagedAttentionBackend,
    layer_idx: int,
    *,
    has_write_proof: bool = True,
) -> PagedLayerBinding:
    class_fqn = "moe_infinity.models.qwen3_paged_attention.Qwen3PagedAttention"
    proxy = LayerBoundPagedBackend(backend, layer_idx, storage.owner_id)
    return PagedLayerBinding(
        module=object(),
        layer_idx=layer_idx,
        base_class=object,
        bound_class=object,
        backend=proxy,
        class_fqn=class_fqn,
        storage_owner_id=storage.owner_id,
        has_write_proof=has_write_proof,
    )


def _make_legacy_paged_kv_cache() -> PagedKVCache:
    return PagedKVCache(
        num_blocks=8,
        block_size=4,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _eligible_runner(
    *,
    storage: PagedKVStorage | None = None,
    registry: PagedAttentionLayerRegistry | None = None,
    kv_cache: object | None = None,
    runner_device: torch.device | None = None,
) -> ModelRunner:
    if storage is None:
        storage = _make_storage()
    backend = PagedAttentionBackend(storage=storage, use_flashinfer=False)
    if registry is None:
        registry = PagedAttentionLayerRegistry(
            bindings=[_proven_binding(storage, backend, 0)],
            reason="eligible",
        )
    engine = types.SimpleNamespace(
        get_attention_backend=lambda: backend,
        kv_cache=kv_cache,
    )
    return ModelRunner(
        _NonPagedModel(),
        engine=engine,
        device=runner_device or storage.spec.device,
        paged_kv_storage=storage,
        paged_attention_registry=registry,
        decode_graph_capability_provider=_ExplicitCapabilityProvider(),
    )


def test_legacy_unbound_paged_kv_cache_is_graph_ineligible() -> None:
    runner = _eligible_runner(kv_cache=_make_legacy_paged_kv_cache())
    assert runner.decode_graph_capability().reason == "kv_storage_mismatch"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs at least two CUDA devices",
)
def test_storage_and_model_runner_device_must_match_exactly() -> None:
    storage = _make_storage(device=torch.device("cuda:0"))
    runner = _eligible_runner(
        storage=storage,
        runner_device=torch.device("cuda:1"),
    )
    assert runner.decode_graph_capability().reason == "kv_storage_mismatch"


def _runner_with_exact_deepseek_paged_attention_or_skip(
    family: str,
) -> ModelRunner:
    pytest.skip(f"DeepSeek {family} paged attention not installed")


@pytest.mark.parametrize("family", ["deepseek_v2", "deepseek_v3"])
def test_deepseek_mla_layout_is_explicitly_rejected(family: str) -> None:
    runner = _runner_with_exact_deepseek_paged_attention_or_skip(family)
    capability = runner.decode_graph_capability()
    assert capability.safe is False
    assert capability.reason == "mla_layout_unsupported"
    assert capability.layer_write_proofs == ()


def test_mla_layout_reason_is_propagated_without_deepseek_installed() -> None:
    registry = PagedAttentionLayerRegistry(
        bindings=[], reason="mla_layout_unsupported"
    )
    runner = _eligible_runner(registry=registry)
    capability = runner.decode_graph_capability()
    assert capability.safe is False
    assert capability.reason == "mla_layout_unsupported"
    assert capability.layer_write_proofs == ()


def _registry_with_unknown_paged_class() -> PagedAttentionLayerRegistry:
    return PagedAttentionLayerRegistry(
        bindings=[], reason="paged_class_unregistered"
    )


def _registry_with_duplicate_or_missing_layer_idx() -> (
    PagedAttentionLayerRegistry
):
    return PagedAttentionLayerRegistry(bindings=[], reason="layer_idx_invalid")


def _registry_without_write_proof() -> PagedAttentionLayerRegistry:
    storage = _make_storage()
    backend = PagedAttentionBackend(storage=storage, use_flashinfer=False)
    return PagedAttentionLayerRegistry(
        bindings=[_proven_binding(storage, backend, 0, has_write_proof=False)],
        reason="eligible",
    )


@pytest.mark.parametrize(
    ("registry", "reason"),
    [
        (_registry_with_unknown_paged_class(), "paged_class_unregistered"),
        (
            _registry_with_duplicate_or_missing_layer_idx(),
            "layer_idx_invalid",
        ),
        (_registry_without_write_proof(), "layer_write_unproven"),
    ],
)
def test_capability_rejects_missing_per_layer_write_proof(
    registry, reason: str
) -> None:
    runner = _eligible_runner(registry=registry)
    assert runner.decode_graph_capability().reason == reason
