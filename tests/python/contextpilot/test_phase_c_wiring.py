# pyright: reportAny=false, reportPrivateUsage=false

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / relative_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EVICTION_SYNC_MODULE = _load_module(
    "eviction_sync_phasec_test",
    "moe_infinity/serving/eviction_sync.py",
)
_CP_KV_MODULE = _load_module(
    "cp_kv_interface_phasec_test",
    "moe_infinity/serving/cp_kv_interface.py",
)
EvictionSyncAdapter = _EVICTION_SYNC_MODULE.EvictionSyncAdapter
ContextPilotKVManager = _CP_KV_MODULE.ContextPilotKVManager


class _FakeMiddleware:
    def on_request_complete(self, request_id: str) -> None:
        _ = request_id


class _RecordingCPTarget:
    def __init__(self) -> None:
        self.installed_manager: Any = "UNSET"

    def set_cp_kv_manager(self, manager: Any) -> None:
        self.installed_manager = manager


class _FakeEngine:
    def __init__(self) -> None:
        self.scheduler = _RecordingCPTarget()
        self.kv_cache = _RecordingCPTarget()


class _RecordedCP:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def on_request_complete(self, request_id: str) -> None:
        self.removed.append(request_id)


@pytest.fixture
def api_server_env(monkeypatch: pytest.MonkeyPatch):
    import moe_infinity.serving.engine as engine_module

    api_server = importlib.import_module(
        "moe_infinity.entrypoints.openai.api_server_v2"
    )

    monkeypatch.delenv("CONTEXTPILOT_ENABLED", raising=False)
    prev_enabled = api_server._contextpilot_enabled
    prev_middleware = api_server._cp_middleware
    prev_sync = api_server._eviction_sync
    prev_engine_sync = engine_module._eviction_sync
    try:
        yield api_server, engine_module
    finally:
        api_server._contextpilot_enabled = prev_enabled
        api_server._cp_middleware = prev_middleware
        api_server._eviction_sync = prev_sync
        engine_module.set_eviction_sync(prev_engine_sync)


def test_cp_enabled_init_installs_eviction_sync(api_server_env) -> None:
    api_server, engine_module = api_server_env
    canonical_sync = importlib.import_module(
        "moe_infinity.serving.eviction_sync"
    )
    canonical_cp_kv = importlib.import_module(
        "moe_infinity.serving.cp_kv_interface"
    )
    api_server._contextpilot_enabled = True
    api_server._cp_middleware = _FakeMiddleware()
    engine_module.set_eviction_sync(None)

    fake_engine = _FakeEngine()
    api_server._wire_contextpilot_phase_c(fake_engine)

    assert isinstance(
        api_server._eviction_sync, canonical_sync.EvictionSyncAdapter
    )
    assert isinstance(
        engine_module._eviction_sync, canonical_sync.EvictionSyncAdapter
    )
    assert isinstance(
        fake_engine.scheduler.installed_manager,
        canonical_cp_kv.ContextPilotKVManager,
    )
    assert isinstance(
        fake_engine.kv_cache.installed_manager,
        canonical_cp_kv.ContextPilotKVManager,
    )


def test_cp_disabled_init_installs_nothing(api_server_env) -> None:
    api_server, engine_module = api_server_env
    canonical_cp_kv = importlib.import_module(
        "moe_infinity.serving.cp_kv_interface"
    )
    api_server._contextpilot_enabled = False
    api_server._cp_middleware = None
    engine_module.set_eviction_sync(None)

    fake_engine = _FakeEngine()
    api_server._wire_contextpilot_phase_c(fake_engine)

    assert api_server._eviction_sync is None
    assert engine_module._eviction_sync is None
    assert isinstance(
        fake_engine.scheduler.installed_manager,
        canonical_cp_kv.NullCPAwareKVManager,
    )
    assert isinstance(
        fake_engine.kv_cache.installed_manager,
        canonical_cp_kv.NullCPAwareKVManager,
    )


def test_evicted_request_ids_is_bounded() -> None:
    cap = 3
    adapter = EvictionSyncAdapter(
        _FakeMiddleware(), max_tracked_request_ids=cap
    )

    for i in range(50):
        adapter.on_request_finished(f"req-{i}")

    assert len(adapter._evicted_request_ids) <= cap


def test_bounded_eviction_still_dedups_recent_ids() -> None:
    cp = _RecordedCP()
    adapter = EvictionSyncAdapter(cp, max_tracked_request_ids=8)

    adapter.on_request_finished("req-A")
    adapter.on_request_finished("req-A")

    assert cp.removed == ["req-A"]


def test_request_to_blocks_cleaned_on_removal() -> None:
    manager = ContextPilotKVManager(_FakeMiddleware())

    manager.notify_blocks_allocated("req-1", [1, 2, 3])
    assert "req-1" in manager._request_to_blocks

    manager.remove_request("req-1")

    assert "req-1" not in manager._request_to_blocks


def test_remove_request_is_idempotent() -> None:
    manager = ContextPilotKVManager(_FakeMiddleware())

    manager.notify_blocks_allocated("req-1", [1, 2, 3])
    manager.remove_request("req-1")
    manager.remove_request("req-1")

    assert manager._request_to_blocks == {}
