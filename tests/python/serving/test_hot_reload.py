# pyright: reportAny=false, reportCallIssue=false, reportExplicitAny=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import importlib
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

try:
    from fastapi.testclient import TestClient

    import moe_infinity.entrypoints.openai.api_server_v2 as srv
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )


def _snapshot_runtime_state() -> dict[str, Any]:
    return {
        "engine": srv.engine,
        "stream_manager": srv.stream_manager,
        "tokenizer": srv.tokenizer,
        "model_name_global": srv.model_name_global,
        "_engine_task": srv._engine_task,
        "_engine_shutdown_event": srv._engine_shutdown_event,
        "_startup_args": getattr(srv, "_startup_args", None),
        "_model_init_task": getattr(srv, "_model_init_task", None),
    }


def _restore_runtime_state(state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(srv, key, value)


def test_reload_valid_module(monkeypatch: pytest.MonkeyPatch) -> None:
    original_state = _snapshot_runtime_state()

    def _reload(module: Any) -> Any:
        return module

    monkeypatch.setattr(importlib, "reload", _reload)

    try:
        with TestClient(srv.app) as client:
            response = client.post("/v1/reload", json={"modules": ["json"]})

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "ok"
        assert body["reloaded"] == ["json"]
        assert body["errors"] == []
    finally:
        _restore_runtime_state(original_state)


def test_reload_invalid_module() -> None:
    original_state = _snapshot_runtime_state()

    try:
        with TestClient(srv.app) as client:
            response = client.post(
                "/v1/reload", json={"modules": ["nonexistent.module"]}
            )

        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "partial"
        assert body["reloaded"] == []
        assert body["errors"][0]["module"] == "nonexistent.module"
    finally:
        _restore_runtime_state(original_state)


def test_reload_bad_payload() -> None:
    original_state = _snapshot_runtime_state()

    try:
        with TestClient(srv.app) as client:
            response = client.post("/v1/reload", json={"modules": "not_a_list"})

        assert response.status_code == 400
    finally:
        _restore_runtime_state(original_state)


def test_config_get_no_engine() -> None:
    original_state = _snapshot_runtime_state()
    srv.engine = None

    try:
        with TestClient(srv.app) as client:
            response = client.get("/v1/config")

        assert response.status_code == 503
    finally:
        _restore_runtime_state(original_state)


def test_config_post_no_engine() -> None:
    original_state = _snapshot_runtime_state()
    srv.engine = None

    try:
        with TestClient(srv.app) as client:
            response = client.post("/v1/config", json={"foo": "bar"})

        assert response.status_code == 503
    finally:
        _restore_runtime_state(original_state)


def test_reload_invalidates_graphs_before_importlib_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_state = _snapshot_runtime_state()
    events: list[str] = []
    fake_engine = SimpleNamespace(
        has_pending_requests=lambda: False,
        invalidate_cuda_graphs=lambda reason: events.append(
            f"invalidate:{reason}"
        ),
        shutdown=Mock(),
    )
    monkeypatch.setattr(srv, "engine", fake_engine)
    monkeypatch.setattr(
        importlib,
        "reload",
        lambda module: events.append("reload") or module,
    )

    try:
        with TestClient(srv.app) as client:
            response = client.post("/v1/reload", json={"modules": ["json"]})
        assert response.status_code == 200
        assert events == ["invalidate:module_reload", "reload"]
    finally:
        _restore_runtime_state(original_state)


def test_failed_reload_leaves_graphs_invalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_state = _snapshot_runtime_state()
    invalidate = Mock()
    fake_engine = SimpleNamespace(
        has_pending_requests=lambda: False,
        invalidate_cuda_graphs=invalidate,
        shutdown=Mock(),
    )
    monkeypatch.setattr(srv, "engine", fake_engine)
    monkeypatch.setattr(
        importlib,
        "reload",
        Mock(side_effect=RuntimeError("bad")),
    )

    try:
        with TestClient(srv.app) as client:
            response = client.post("/v1/reload", json={"modules": ["json"]})
        assert response.json()["status"] == "partial"
        invalidate.assert_called_once_with("module_reload")
    finally:
        _restore_runtime_state(original_state)


def test_application_shutdown_closes_current_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_state = _snapshot_runtime_state()
    current = SimpleNamespace(
        has_pending_requests=lambda: False,
        shutdown=Mock(),
    )
    monkeypatch.setattr(srv, "engine", current)

    try:
        with TestClient(srv.app):
            pass
        current.shutdown.assert_called_once_with()
    finally:
        _restore_runtime_state(original_state)


def test_hot_replacement_closes_old_engine_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_state = _snapshot_runtime_state()
    events: list[str] = []
    old = SimpleNamespace(shutdown=lambda: events.append("old:close"))
    new = SimpleNamespace()
    monkeypatch.setattr(srv, "engine", old)

    try:
        srv._replace_engine(new)
        assert srv.engine is new
        assert events == ["old:close"]
    finally:
        _restore_runtime_state(original_state)


def test_hot_replacement_waits_for_active_step_and_obeys_lock_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_state = _snapshot_runtime_state()
    step_entered = threading.Event()
    release_step = threading.Event()
    events: list[str] = []

    class BlockingEngine:
        def step(self) -> None:
            step_entered.set()
            assert release_step.wait(timeout=1.0)
            events.append("step:exit")

        def shutdown(self) -> None:
            events.extend(["graph:close", "old:close"])

    old = BlockingEngine()
    new = SimpleNamespace()
    monkeypatch.setattr(srv, "engine", old)
    step_thread = threading.Thread(target=srv._run_engine_step_once)
    replace_thread = threading.Thread(target=lambda: srv._replace_engine(new))

    try:
        step_thread.start()
        assert step_entered.wait(timeout=1.0)
        replace_thread.start()
        assert "old:close" not in events
        release_step.set()
        step_thread.join(timeout=1.0)
        replace_thread.join(timeout=1.0)
        assert not step_thread.is_alive()
        assert not replace_thread.is_alive()
        assert events == ["step:exit", "graph:close", "old:close"]
    finally:
        release_step.set()
        step_thread.join(timeout=1.0)
        replace_thread.join(timeout=1.0)
        _restore_runtime_state(original_state)
