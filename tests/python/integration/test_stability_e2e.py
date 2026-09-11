# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedCallResult=false, reportUnusedVariable=false, reportUntypedFunctionDecorator=false, reportUnannotatedClassAttribute=false, reportUnknownLambdaType=false, reportUnusedParameter=false, reportPrivateUsage=false, reportPrivateLocalImportUsage=false
"""
E2E Integration Tests — Stability Hardening
Tests all stability features working together via TestClient.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

try:
    from fastapi.testclient import TestClient

    import moe_infinity.entrypoints.openai.api_server_v2 as server_module
    import moe_infinity.serving.watchdog as watchdog_module
    from moe_infinity.serving.health import ServerHealthState
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )


class _FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        return [1] * max(1, len(prompt))

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False
    ) -> str:
        _ = token_ids
        _ = skip_special_tokens
        return "ok"

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        _ = tokenize
        _ = add_generation_prompt
        return "\n".join(message.get("content", "") for message in conversation)


class _Output:
    def __init__(self) -> None:
        self.token_id = 1
        self.token_text = "ok"
        self.seq_id = 0
        self.token_logprob = None
        self.top_logprobs = None
        self.finished = True
        self.finish_reason = "stop"
        self.usage = {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }


class _FakeEngine:
    def add_request(self, **kwargs: Any) -> None:
        on_token = kwargs["on_token"]
        on_token(_Output())

    def abort_request(self, request_id: str) -> None:
        _ = request_id

    def has_pending_requests(self) -> bool:
        return False

    def step(self) -> list[Any]:
        return []

    def shutdown(self) -> None:
        return None


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> _FakeTokenizer:
        _ = args
        _ = kwargs
        return _FakeTokenizer()


class _FakeRuntimeEngine:
    def __init__(
        self,
        model: object,
        engine: object,
        config: dict[str, object],
        tokenizer: object,
    ) -> None:
        _ = model
        _ = engine
        _ = tokenizer
        self.config = config


class _FakeMoE:
    def __init__(self, model_name: str, config: dict[str, object]) -> None:
        _ = model_name
        _ = config
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=1,
                num_attention_heads=1,
                hidden_size=16,
                max_position_embeddings=128,
                eos_token_id=2,
                torch_dtype="torch.float16",
            )
        )
        self.engine = object()


def _snapshot_runtime_state(module: Any) -> dict[str, Any]:
    return {
        "engine": module.engine,
        "stream_manager": module.stream_manager,
        "tokenizer": module.tokenizer,
        "model_name_global": module.model_name_global,
        "runtime_max_seq_length": module.runtime_max_seq_length,
        "_engine_task": module._engine_task,
        "_engine_shutdown_event": module._engine_shutdown_event,
        "_model_init_task": module._model_init_task,
        "_startup_args": module._startup_args,
        "_startup_watchdog": module._startup_watchdog,
        "_decode_watchdog": module._decode_watchdog,
        "_watchdog_config": module._watchdog_config,
        "_health_state": module._health_state,
    }


def _restore_runtime_state(module: Any, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(module, key, value)


@pytest.fixture
def runtime_guard() -> Any:
    original_state = _snapshot_runtime_state(server_module)
    try:
        yield
    finally:
        _restore_runtime_state(server_module, original_state)


def _setup_runtime(*, max_seq_length: int = 64, healthy: bool = True) -> None:
    server_module.engine = _FakeEngine()
    server_module.stream_manager = object()
    server_module.tokenizer = _FakeTokenizer()
    server_module.model_name_global = "unit-test-model"
    server_module.runtime_max_seq_length = max_seq_length
    server_module._startup_args = None

    state = ServerHealthState()
    if healthy:
        state.set_healthy()
    else:
        state.set_starting()
    server_module._health_state = state


def _post_completion(
    client: TestClient,
    payload: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    response = client.post("/v1/completions", json=payload)
    return response.status_code, response.json()


def test_health_returns_healthy_when_engine_running(runtime_guard: Any) -> None:
    _setup_runtime(healthy=True)

    with TestClient(server_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "reason": None}


def test_health_returns_starting_when_engine_is_none(
    runtime_guard: Any,
) -> None:
    _setup_runtime(healthy=False)
    server_module.engine = None
    server_module.stream_manager = None

    with TestClient(server_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "starting", "reason": None}


def test_health_returns_unhealthy_when_marked_unhealthy(
    runtime_guard: Any,
) -> None:
    _setup_runtime(healthy=True)
    server_module._health_state.set_unhealthy("decode watchdog timeout")

    with TestClient(server_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "reason": "decode watchdog timeout",
    }


def test_validation_missing_max_tokens_returns_400(runtime_guard: Any) -> None:
    _setup_runtime()

    with TestClient(server_module.app) as client:
        status_code, payload = _post_completion(
            client,
            {"model": "unit-test-model", "prompt": "hello"},
        )

    assert status_code == 400
    assert payload["error"]["code"] == "invalid_request_error"


def test_validation_bad_temperature_returns_param(runtime_guard: Any) -> None:
    _setup_runtime()

    with TestClient(server_module.app) as client:
        status_code, payload = _post_completion(
            client,
            {
                "model": "unit-test-model",
                "prompt": "hello",
                "max_tokens": 8,
                "temperature": 5.0,
            },
        )

    assert status_code == 400
    assert payload["error"]["param"] == "temperature"


def test_validation_over_context_returns_context_length_exceeded(
    runtime_guard: Any,
) -> None:
    _setup_runtime(max_seq_length=6)

    with TestClient(server_module.app) as client:
        status_code, payload = _post_completion(
            client,
            {
                "model": "unit-test-model",
                "prompt": "hello",
                "max_tokens": 2,
            },
        )

    assert status_code == 400
    assert payload["error"]["code"] == "context_length_exceeded"


def test_mixed_valid_invalid_valid_requests_keep_service_healthy(
    runtime_guard: Any,
) -> None:
    _setup_runtime(max_seq_length=10, healthy=True)

    statuses: list[int] = []
    invalid_codes: list[str] = []

    with TestClient(server_module.app) as client:
        for idx in range(3):
            status_code, _ = _post_completion(
                client,
                {
                    "model": "unit-test-model",
                    "prompt": f"ok{idx}",
                    "max_tokens": 2,
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
            )
            statuses.append(status_code)

        invalid_requests = [
            {"model": "unit-test-model", "prompt": "bad-missing-max"},
            {
                "model": "unit-test-model",
                "prompt": "bad-temp",
                "max_tokens": 2,
                "temperature": 5.0,
            },
            {
                "model": "unit-test-model",
                "prompt": "123456789",
                "max_tokens": 2,
            },
        ]
        for payload in invalid_requests:
            status_code, body = _post_completion(client, payload)
            statuses.append(status_code)
            invalid_codes.append(body["error"]["code"])

        for idx in range(3):
            status_code, _ = _post_completion(
                client,
                {
                    "model": "unit-test-model",
                    "prompt": f"go{idx}",
                    "max_tokens": 1,
                },
            )
            statuses.append(status_code)

        health_response = client.get("/health")

    assert statuses == [200, 200, 200, 400, 400, 400, 200, 200, 200]
    assert invalid_codes == [
        "invalid_request_error",
        "invalid_request_error",
        "context_length_exceeded",
    ]
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "healthy", "reason": None}


def test_server_without_watchdog_flags_keeps_watchdogs_none(
    monkeypatch: Any, runtime_guard: Any
) -> None:
    server_module.engine = None
    server_module.stream_manager = None
    server_module.tokenizer = None
    server_module.model_name_global = None
    server_module._startup_args = argparse.Namespace(
        model="unit-test-model",
        offload_dir="/tmp/offload",
        device_memory_ratio=0.75,
        kv_cache_ratio=0.25,
        max_batch_size=4,
        enable_prefix_caching=False,
        startup_timeout=None,
        decode_step_timeout=None,
        enable_pyspy_dump=False,
    )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer),
    )

    original_import_module = importlib.import_module

    def _fake_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "moe_infinity":
            return SimpleNamespace(MoE=_FakeMoE)
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(
        server_module.importlib, "import_module", _fake_import_module
    )
    monkeypatch.setattr(
        server_module,
        "ContinuousBatchingEngine",
        _FakeRuntimeEngine,
    )
    monkeypatch.setattr(
        server_module,
        "_ensure_engine_loop_running",
        lambda: None,
    )

    with (
        patch.object(
            watchdog_module,
            "start_startup_watchdog",
        ) as startup_mock,
        patch.object(
            watchdog_module,
            "start_decode_watchdog",
        ) as decode_mock,
    ):
        asyncio.run(server_module._initialize_model())

    startup_mock.assert_not_called()
    decode_mock.assert_not_called()
    assert server_module._startup_watchdog is None
    assert server_module._decode_watchdog is None


def test_phase_policy_cli_defaults_off_and_can_enable(monkeypatch) -> None:
    base = [
        "api_server_v2",
        "--model",
        "fixture/model",
        "--offload-dir",
        "/tmp/moe-policy-test",
    ]
    monkeypatch.setattr(sys, "argv", base)
    assert server_module.parse_args().phase_specific_expert_policy is False

    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--phase-specific-expert-policy"],
    )
    assert server_module.parse_args().phase_specific_expert_policy is True

    monkeypatch.setattr(
        sys,
        "argv",
        [*base, "--no-phase-specific-expert-policy"],
    )
    assert server_module.parse_args().phase_specific_expert_policy is False


def test_prometheus_formats_phase_policy_metrics() -> None:
    body = server_module._format_prometheus_metrics(
        {
            "expert_policy": {
                "enabled": 1,
                "resident_bytes": 4096,
                "prefill_hits": 3,
                "decode_hits": 7,
                "transition_hits": 2,
            }
        }
    )
    assert 'moe_expert_cache_hits_total{phase="prefill"} 3' in body
    assert 'moe_expert_cache_hits_total{phase="decode"} 7' in body
    assert "moe_expert_cache_resident_bytes 4096" in body
    assert "moe_expert_cache_transition_hits_total 2" in body
