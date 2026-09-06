from __future__ import annotations

import pytest
import torch

from tests.python.serving._prefix_bootstrap import (
    ContinuousBatchingEngine,
    ModelRunner,
    PagedKVCache,
    PrefixReuseCapability,
    Scheduler,
)
from tests.python.serving.prefix_cache_test_utils import (
    RecordingLayeredPagedKVStore,
    make_paged_backend,
)


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _base_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": 8,
        "max_tokens_per_step": 128,
        "block_size": 4,
        "num_layers": 2,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype": "float32",
        "eos_token_id": 2,
        "model_memory_bytes": 0,
        "num_kv_blocks": 32,
    }
    config.update(overrides)
    return config


class _MockModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = type("Cfg", (), {"vocab_size": 64, "eos_token_id": 2})()

    def eval(self) -> None:
        return None


def _make_engine(overrides: dict[str, object]) -> ContinuousBatchingEngine:
    engine_obj = type(
        "Eng",
        (),
        {
            "expert_tracer": type("T", (), {"create_entry": lambda self: 0})(),
            "expert_layer_modules": [],
            "request_id": 0,
        },
    )()
    return ContinuousBatchingEngine(
        _MockModel(), engine_obj, _base_config(**overrides)
    )


def test_engine_binds_validated_store_before_scheduler(monkeypatch) -> None:
    events: list[str] = []
    store = RecordingLayeredPagedKVStore(
        num_layers=2, num_blocks=32, num_kv_heads=2, head_dim=8
    )
    monkeypatch.setattr(
        ModelRunner,
        "get_prefix_reuse_capability",
        lambda self: PrefixReuseCapability.active(store.owner, store),
    )
    original_bind = PagedKVCache.set_block_store
    monkeypatch.setattr(
        PagedKVCache,
        "set_block_store",
        lambda self, value, *, owner: (
            events.append("bind"),
            original_bind(self, value, owner=owner),
        )[1],
    )
    original_init = Scheduler.__init__

    def scheduler_init(self, *args, **kwargs):
        events.append("scheduler")
        assert args[0]._block_store is store
        assert kwargs["prefix_lease_provider"] is not None
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Scheduler, "__init__", scheduler_init)
    _make_engine({"enable_prefix_caching": True})
    assert events == ["bind", "scheduler"]


def test_store_geometry_mismatch_fails_closed(monkeypatch) -> None:
    bad_store = RecordingLayeredPagedKVStore(num_layers=99, num_blocks=32)
    monkeypatch.setattr(
        ModelRunner,
        "get_prefix_reuse_capability",
        lambda self: PrefixReuseCapability.active(bad_store.owner, bad_store),
    )
    engine = _make_engine({"enable_prefix_caching": True})
    assert engine.prefix_cache is None
    assert engine.scheduler.prefix_lease_provider is None
    assert str(engine.get_stats()["prefix_cache_disabled_reason"]).startswith(
        "kv-store-binding-mismatch"
    )


@pytest.mark.parametrize(
    ("budget_blocks", "store_blocks", "expected_logical"),
    [(6, 8, 6), (12, 8, 8)],
)
def test_prefix_cache_physical_capacity_may_exceed_logical(
    monkeypatch, budget_blocks, store_blocks, expected_logical
) -> None:
    backend = make_paged_backend(num_blocks=store_blocks)
    store = backend.create_layered_store(layer_count=2)
    monkeypatch.setattr(
        ModelRunner,
        "get_prefix_reuse_capability",
        lambda self: PrefixReuseCapability.active(backend, store),
    )
    engine = _make_engine(
        {"enable_prefix_caching": True, "num_kv_blocks": budget_blocks}
    )
    assert engine.kv_cache.num_blocks == expected_logical
    assert engine.kv_cache.block_store is backend.block_store
    assert engine.scheduler.prefix_lease_provider is engine.prefix_cache


def test_disabled_prefix_cache_has_stable_reason() -> None:
    engine = _make_engine({"enable_prefix_caching": False})
    stats = engine.get_stats()
    assert stats["prefix_cache_active"] is False
    assert stats["prefix_cache_enabled"] is False
    assert stats["prefix_cache_disabled_reason"] == "prefix-caching-disabled"


def test_prefix_cache_max_entries_defaults(monkeypatch) -> None:
    store = RecordingLayeredPagedKVStore(
        num_layers=2, num_blocks=32, num_kv_heads=2, head_dim=8
    )
    monkeypatch.setattr(
        ModelRunner,
        "get_prefix_reuse_capability",
        lambda self: PrefixReuseCapability.active(store.owner, store),
    )
    engine = _make_engine({"enable_prefix_caching": True})
    assert engine.prefix_cache is not None
    assert engine.prefix_cache.max_entries == 1000
