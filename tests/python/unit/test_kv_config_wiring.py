import importlib.machinery
import sys
import types
from types import SimpleNamespace
from typing import Optional, cast

import pytest
import torch

from moe_infinity.engine.kv_cache_offload_coordinator import (
    KVCacheOffloadCoordinator,
)
from moe_infinity.engine.kv_transfer import (
    PinnedBufferPool,
    SyncKVTransferBackend,
)
from moe_infinity.engine.transfer_types import TransferType
from moe_infinity.serving.engine import build_kv_transfer_resources
from moe_infinity.utils.config import ArcherConfig

if (
    "flash_attn" not in sys.modules
    or getattr(sys.modules["flash_attn"], "__spec__", None) is None
):
    flash_attn_stub = sys.modules.get(
        "flash_attn", types.ModuleType("flash_attn")
    )
    flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn", loader=None
    )
    sys.modules["flash_attn"] = flash_attn_stub

from moe_infinity.runtime.model_offload import OffloadEngine


class _DummyScheduler:
    def __init__(self) -> None:
        self.handlers: dict[TransferType, object] = {}

    def register_handler(
        self, transfer_type: TransferType, handler: object
    ) -> None:
        self.handlers[transfer_type] = handler


def test_default_config_has_offload_disabled() -> None:
    config = ArcherConfig()
    assert not config.enable_kv_cache_offload


def test_kv_swap_config_defaults() -> None:
    config = ArcherConfig()
    assert config.kv_swap_mode == "sync"
    assert config.kv_swap_host_memory_bytes == 512 * 1024 * 1024
    assert config.kv_swap_max_inflight_bytes == 256 * 1024 * 1024
    assert config.kv_swap_checksum is False
    assert config.kv_swap_max_retries == 2
    assert config.kv_swap_allow_sync_fallback is True


def test_kv_swap_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="kv_swap_mode"):
        ArcherConfig(kv_swap_mode="hybrid")


def test_kv_swap_rejects_nonpositive_host_cap() -> None:
    with pytest.raises(ValueError, match="kv_swap_host_memory_bytes"):
        ArcherConfig(kv_swap_host_memory_bytes=0)


def test_kv_swap_rejects_nonpositive_inflight_cap() -> None:
    with pytest.raises(ValueError, match="kv_swap_max_inflight_bytes"):
        ArcherConfig(kv_swap_max_inflight_bytes=0)


def test_kv_swap_rejects_inflight_greater_than_host_cap() -> None:
    with pytest.raises(ValueError, match="kv_swap_max_inflight_bytes"):
        ArcherConfig(
            kv_swap_host_memory_bytes=1024,
            kv_swap_max_inflight_bytes=2048,
        )


def test_kv_swap_rejects_negative_retries() -> None:
    with pytest.raises(ValueError, match="kv_swap_max_retries"):
        ArcherConfig(kv_swap_max_retries=-1)


def test_kv_swap_byte_caps_accepted_but_inactive_in_sync_mode() -> None:
    # Byte caps are validated (accepted) but do not construct/budget a pool
    # when the effective mode is sync.
    config = ArcherConfig(
        kv_swap_mode="sync",
        kv_swap_host_memory_bytes=4096,
        kv_swap_max_inflight_bytes=2048,
    )
    backend, pool, fallback_reason = build_kv_transfer_resources(
        config=config,
        device=torch.device("cpu"),
    )
    assert isinstance(backend, SyncKVTransferBackend)
    assert pool is None
    assert fallback_reason is None


def test_default_sync_does_not_construct_or_budget_a_pinned_pool() -> None:
    config = ArcherConfig()
    pool_factory_calls: list[int] = []

    def forbidden_pool_factory(capacity_bytes: int):
        pool_factory_calls.append(capacity_bytes)
        raise AssertionError("default sync must not construct a pinned pool")

    backend, pool, fallback_reason = build_kv_transfer_resources(
        config=config,
        device=torch.device("cpu"),
        pool_factory=forbidden_pool_factory,
    )

    assert config.kv_swap_mode == "sync"
    assert isinstance(backend, SyncKVTransferBackend)
    assert pool is None
    assert fallback_reason is None
    assert pool_factory_calls == []


def test_async_backend_failure_falls_back_to_sync_when_enabled() -> None:
    config = ArcherConfig(
        kv_swap_mode="async",
        kv_swap_allow_sync_fallback=True,
    )
    released_pools: list[object] = []

    class _DisposablePool:
        def __init__(self, capacity_bytes: int) -> None:
            self.capacity_bytes = capacity_bytes

        def close(self) -> None:
            released_pools.append(self)

    def failing_backend_factory(device: torch.device):
        raise RuntimeError("pin unavailable")

    backend, pool, fallback_reason = build_kv_transfer_resources(
        config=config,
        device=torch.device("cpu"),
        pool_factory=_DisposablePool,
        backend_factory=failing_backend_factory,
    )

    assert isinstance(backend, SyncKVTransferBackend)
    assert pool is None
    assert fallback_reason is not None
    assert "pin unavailable" in fallback_reason
    # A partially constructed async pool must be discarded/closed.
    assert len(released_pools) == 1


def test_async_backend_failure_raises_when_fallback_disabled() -> None:
    config = ArcherConfig(
        kv_swap_mode="async",
        kv_swap_allow_sync_fallback=False,
    )

    class _DisposablePool:
        def __init__(self, capacity_bytes: int) -> None:
            self.capacity_bytes = capacity_bytes

        def close(self) -> None:
            return None

    def failing_backend_factory(device: torch.device):
        raise RuntimeError("pin unavailable")

    with pytest.raises(RuntimeError, match="pin unavailable"):
        build_kv_transfer_resources(
            config=config,
            device=torch.device("cpu"),
            pool_factory=_DisposablePool,
            backend_factory=failing_backend_factory,
        )


def test_coordinator_not_registered_when_disabled() -> None:
    scheduler = _DummyScheduler()
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=torch.zeros(2, 4, 8),
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=False),
    )

    coordinator.register_with_scheduler(scheduler)

    assert TransferType.KV_SWAP_OUT not in scheduler.handlers
    assert TransferType.KV_SWAP_IN not in scheduler.handlers


def test_coordinator_registered_when_enabled() -> None:
    scheduler = _DummyScheduler()
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=torch.zeros(2, 4, 8),
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    coordinator.register_with_scheduler(scheduler)

    assert TransferType.KV_SWAP_OUT in scheduler.handlers
    assert TransferType.KV_SWAP_IN in scheduler.handlers


def test_capture_kv_skips_when_disabled() -> None:
    engine = OffloadEngine.__new__(OffloadEngine)
    setattr(engine, "_enable_kv_cache_offload", False)
    setattr(engine, "_captured_kv", {})

    mock_past_kv = ((torch.randn(2, 4, 8), torch.randn(2, 4, 8)),)

    getattr(OffloadEngine, "_capture_kv_cache")(
        engine,
        seq_id=0,
        past_key_values=mock_past_kv,
    )

    assert getattr(engine, "_captured_kv") == {}


def test_capture_reload_active_when_enabled() -> None:
    engine = OffloadEngine.__new__(OffloadEngine)
    setattr(engine, "_enable_kv_cache_offload", True)
    setattr(engine, "_captured_kv", {})
    engine.model = None

    past_key_values = ((torch.randn(2, 4, 8), torch.randn(2, 4, 8)),)

    getattr(OffloadEngine, "_capture_kv_cache")(
        engine,
        seq_id=7,
        past_key_values=past_key_values,
    )
    assert 7 in getattr(engine, "_captured_kv")

    restored = cast(
        Optional[tuple[object, ...]],
        getattr(OffloadEngine, "_reload_kv_cache")(engine, seq_id=7),
    )
    assert restored is not None
    assert len(restored) == len(past_key_values)
    assert 7 not in getattr(engine, "_captured_kv")
