# pyright: reportAny=false, reportCallIssue=false, reportExplicitAny=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

try:
    from fastapi.testclient import TestClient

    import moe_infinity.entrypoints.openai.api_server_v2 as srv
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )


def _make_mock_stats() -> dict[str, Any]:
    return {
        "pending_requests": 2,
        "completed_requests": 10,
        "cancelled_requests": 1,
        "num_steps": 100,
        "total_generated_tokens": 500,
        "kv_cache_num_blocks": 32,
        "kv_cache_free_blocks": 20,
        "sequence_status_counts": {
            "waiting": 1,
            "prefill": 0,
            "decode": 1,
            "finished": 10,
            "swapped": 0,
            "cancelled": 1,
        },
        "memory": {
            "device": "cpu",
            "total_gpu_memory_bytes": 0,
            "expert_cache_bytes": 0,
            "kv_cache_bytes": 0,
        },
    }


def _completion_payload() -> dict[str, Any]:
    return {
        "model": "test-model",
        "prompt": "hello",
        "max_tokens": 4,
    }


def _mock_model() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=32,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=2,
            dtype="float32",
        ),
        dtype="float32",
    )


def _parse_args(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra: list[str] | None = None,
) -> Any:
    argv = [
        "api_server_v2.py",
        "--model",
        "test-model",
        "--offload-dir",
        "/tmp/offload",
    ]
    if extra:
        argv.extend(extra)
    monkeypatch.setattr(sys, "argv", argv)
    return srv.parse_args()


def _configure_auth_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_keys: str | None,
    rate_limit: int = 0,
) -> None:
    if api_keys is None:
        monkeypatch.delenv("MOE_API_KEYS", raising=False)
        configured_keys = ""
    else:
        monkeypatch.setenv("MOE_API_KEYS", api_keys)
        configured_keys = api_keys

    srv._configure_auth(configured_keys, rate_limit)


@dataclass
class _MockOutput:
    seq_id: int = 0
    token_id: int = 1
    token_text: str = "ok"
    finished: bool = True
    finish_reason: str = "stop"
    token_logprob: float | None = None
    top_logprobs: dict[int, float] | None = None
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        }
    )


@pytest.fixture
def client() -> Any:
    original_engine = srv.engine
    original_model_name = srv.model_name_global
    original_tokenizer = srv.tokenizer
    original_health_state = srv._health_state
    original_stream_manager = srv.stream_manager
    original_api_keys = set(getattr(srv, "_api_keys", set()))
    original_rate_limit_rpm = getattr(srv, "_rate_limit_rpm", 0)
    original_rate_limit_buckets = {
        key: list(value)
        for key, value in getattr(srv, "_rate_limit_buckets", {}).items()
    }
    original_max_waiting_requests = getattr(srv, "_max_waiting_requests", 0)
    original_max_n = getattr(srv, "_max_n", 16)

    mock_engine = MagicMock()
    mock_engine.get_stats.return_value = _make_mock_stats()
    mock_engine.scheduler = SimpleNamespace(num_waiting=0)
    mock_engine.has_pending_requests.return_value = False
    mock_engine.step.return_value = []

    def _add_request(**kwargs: Any) -> None:
        on_token = kwargs.get("on_token")
        if callable(on_token):
            _ = on_token(_MockOutput())

    mock_engine.add_request.side_effect = _add_request
    mock_engine.abort_request.return_value = None

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "ok"
    mock_tokenizer.apply_chat_template.return_value = "user: hello\nassistant:"

    srv.engine = mock_engine
    srv.stream_manager = object()
    srv.model_name_global = "test-model"
    srv.tokenizer = mock_tokenizer
    srv._health_state.set_healthy()

    try:
        with TestClient(srv.app) as test_client:
            yield test_client
    finally:
        srv.engine = original_engine
        srv.stream_manager = original_stream_manager
        srv.model_name_global = original_model_name
        srv.tokenizer = original_tokenizer
        srv._health_state = original_health_state
        srv._api_keys = original_api_keys
        srv._rate_limit_rpm = original_rate_limit_rpm
        srv._rate_limit_buckets = original_rate_limit_buckets
        srv._max_waiting_requests = original_max_waiting_requests
        srv._max_n = original_max_n


def test_list_models(client: TestClient) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "test-model"
    assert payload["data"][0]["object"] == "model"


def test_initialize_with_model_forwards_speculative_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _capture_engine(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(srv, "ContinuousBatchingEngine", _capture_engine)
    model = _mock_model()
    offload_engine = object()
    moe_model = SimpleNamespace(model=model, engine=offload_engine)
    speculator = object()

    srv.initialize_with_model(
        moe_model=moe_model,
        model_name="test-model",
        tok=None,
        max_seq_length=128,
        speculative_draft=speculator,
        enable_deepseek_mla_paging=True,
        max_resident_paged_speculative_sessions=3,
        min_free_mla_blocks_after_admission=2,
    )

    assert captured["model"] is model
    assert captured["engine"] is offload_engine
    assert captured["speculative_draft"] is speculator
    assert captured["decode_graph_capability_provider"] is moe_model


def test_decode_cuda_graphs_are_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse_args(monkeypatch)

    assert args.enable_decode_cuda_graphs is False
    config = srv._build_engine_config(args, _mock_model())
    assert config["enable_decode_cuda_graphs"] is False


def test_decode_cuda_graph_cli_values_reach_engine_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parse_args(
        monkeypatch,
        extra=[
            "--enable-decode-cuda-graphs",
            "--decode-cuda-graph-batch-sizes",
            "1",
            "2",
            "4",
            "--decode-cuda-graph-context-sizes",
            "128",
            "256",
            "--decode-cuda-graph-warmup-iters",
            "3",
            "--decode-cuda-graph-max-memory-bytes",
            "1073741824",
        ],
    )

    config = srv._build_engine_config(args, _mock_model())

    assert config["enable_decode_cuda_graphs"] is True
    assert config["decode_cuda_graph_batch_sizes"] == (1, 2, 4)
    assert config["decode_cuda_graph_context_sizes"] == (128, 256)
    assert config["decode_cuda_graph_warmup_iters"] == 3
    assert config["decode_cuda_graph_max_memory_bytes"] == 1073741824


def test_dflash_paged_cli_defaults_remain_off_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "api_server_v2.py",
            "--model",
            "demo/model",
            "--offload-dir",
            "/tmp/offload",
        ],
    )

    args = srv.parse_args()

    assert args.enable_deepseek_mla_paging is False
    assert args.max_resident_paged_speculative_sessions == 1
    assert args.min_free_mla_blocks_after_admission == 1


def test_dflash_paged_cli_values_forward_to_engine_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "api_server_v2.py",
            "--model",
            "demo/model",
            "--offload-dir",
            "/tmp/offload",
            "--enable-deepseek-mla-paging",
            "--max-resident-paged-speculative-sessions",
            "4",
            "--min-free-mla-blocks-after-admission",
            "3",
        ],
    )
    args = srv.parse_args()
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=32,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=2,
            dtype="float32",
        ),
        dtype="float32",
    )

    config = srv._build_engine_config(args=args, model=model)

    assert config["enable_deepseek_mla_paging"] is True
    assert config["max_resident_paged_speculative_sessions"] == 4
    assert config["min_free_mla_blocks_after_admission"] == 3


def test_list_models_engine_not_ready(client: TestClient) -> None:
    srv.engine = None

    response = client.get("/v1/models")

    assert response.status_code == 503


def test_admin_stats(client: TestClient) -> None:
    response = client.get("/admin/stats")

    assert response.status_code == 200
    payload = response.json()
    for key in _make_mock_stats():
        assert key in payload


def test_admin_stats_engine_not_ready(client: TestClient) -> None:
    srv.engine = None

    response = client.get("/admin/stats")

    assert response.status_code == 503


def test_metrics_endpoint(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "moe_requests_completed" in body
    assert "moe_queue_depth" in body
    assert "moe_kv_cache_free_blocks" in body
    assert "moe_tokens_generated_total" in body


def test_metrics_endpoint_exports_graph_counters_and_bounded_reasons(
    client: TestClient,
) -> None:
    srv.engine.get_stats.return_value = {
        **_make_mock_stats(),
        "cuda_graph": {
            "captures": 2,
            "replays": 9,
            "capture_failures": 1,
            "graphs": 2,
            "graph_pool_bytes": 4096,
            "scratch_kv_bytes": 2048,
            "fallback_reasons": {
                "expert_dispatcher": 7,
                "not_decode": 3,
                "unbounded-exception-text": 99,
            },
            "capability_reason": "expert_dispatcher",
        },
    }

    response = client.get("/metrics")

    assert "moe_cuda_graph_captures_total 2" in response.text
    assert "moe_cuda_graph_replays_total 9" in response.text
    assert "moe_cuda_graph_capture_failures_total 1" in response.text
    assert "moe_cuda_graph_instances 2" in response.text
    assert "moe_cuda_graph_pool_bytes 4096" in response.text
    assert "moe_cuda_graph_scratch_kv_bytes 2048" in response.text
    assert (
        'moe_cuda_graph_fallback_total{reason="expert_dispatcher"} 7'
        in response.text
    )
    assert 'reason="unbounded-exception-text"' not in response.text


def test_metrics_engine_not_ready(client: TestClient) -> None:
    srv.engine = None

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "moe_requests_completed 0" in response.text
    assert "moe_queue_depth 0" in response.text


def test_auth_rejects_missing_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_auth_state(monkeypatch, api_keys="test-key-1")

    response = client.post("/v1/completions", json=_completion_payload())

    assert response.status_code == 401


def test_auth_accepts_valid_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_auth_state(monkeypatch, api_keys="test-key-1")

    response = client.post(
        "/v1/completions",
        json=_completion_payload(),
        headers={"Authorization": "Bearer test-key-1"},
    )

    assert response.status_code != 401


def test_auth_rejects_invalid_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_auth_state(monkeypatch, api_keys="test-key-1")

    response = client.post(
        "/v1/completions",
        json=_completion_payload(),
        headers={"Authorization": "Bearer wrong-key"},
    )

    assert response.status_code == 401


def test_auth_exempts_health(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_auth_state(monkeypatch, api_keys="test-key-1")

    response = client.get("/health")

    assert response.status_code == 200


def test_auth_exempts_metrics(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_auth_state(monkeypatch, api_keys="test-key-1")

    response = client.get("/metrics")

    assert response.status_code == 200


def test_auth_disabled_when_no_keys(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_auth_state(monkeypatch, api_keys=None)

    response = client.post("/v1/completions", json=_completion_payload())

    assert response.status_code != 401


def test_rate_limit_rejects_excess(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_auth_state(monkeypatch, api_keys="test-key-1", rate_limit=2)
    headers = {"Authorization": "Bearer test-key-1"}

    first = client.post(
        "/v1/completions", json=_completion_payload(), headers=headers
    )
    second = client.post(
        "/v1/completions", json=_completion_payload(), headers=headers
    )
    third = client.post(
        "/v1/completions", json=_completion_payload(), headers=headers
    )

    assert first.status_code != 429
    assert second.status_code != 429
    assert third.status_code == 429


def test_backpressure_rejects_when_full(client: TestClient) -> None:
    srv._max_waiting_requests = 2
    assert srv.engine is not None
    setattr(srv.engine, "scheduler", SimpleNamespace(num_waiting=2))

    response = client.post("/v1/completions", json=_completion_payload())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "queue_full"


def test_backpressure_allows_when_below_limit(client: TestClient) -> None:
    srv._max_waiting_requests = 10
    assert srv.engine is not None
    setattr(srv.engine, "scheduler", SimpleNamespace(num_waiting=2))

    response = client.post("/v1/completions", json=_completion_payload())

    assert response.status_code != 503


def test_completion_logprobs_response(client: TestClient) -> None:
    assert srv.engine is not None
    mock_engine = cast(Any, srv.engine)

    def _add_request(**kwargs: Any) -> None:
        on_token = kwargs.get("on_token")
        if callable(on_token):
            _ = on_token(
                _MockOutput(
                    token_id=7,
                    token_text="ok",
                    token_logprob=-0.1,
                    top_logprobs={7: -0.1, 3: -1.2},
                )
            )

    mock_engine.add_request.side_effect = _add_request

    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 4,
            "logprobs": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["logprobs"]["token_logprobs"] == [-0.1]
    assert payload["choices"][0]["logprobs"]["top_logprobs"][0] == {
        "7": -0.1,
        "3": -1.2,
    }


def test_chat_completion_logprobs_response(client: TestClient) -> None:
    assert srv.engine is not None
    mock_engine = cast(Any, srv.engine)

    def _add_request(**kwargs: Any) -> None:
        on_token = kwargs.get("on_token")
        if callable(on_token):
            _ = on_token(
                _MockOutput(
                    token_id=9,
                    token_text="ok",
                    token_logprob=-0.25,
                    top_logprobs={9: -0.25, 4: -0.75},
                )
            )

    mock_engine.add_request.side_effect = _add_request

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "logprobs": True,
            "top_logprobs": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["logprobs"]["token_logprobs"] == [-0.25]
    assert payload["choices"][0]["logprobs"]["top_logprobs"][0] == {
        "9": -0.25,
        "4": -0.75,
    }


def test_completion_n_multiple_choices(client: TestClient) -> None:
    assert srv.engine is not None
    mock_engine = cast(Any, srv.engine)

    def _add_request(**kwargs: Any) -> None:
        on_token = kwargs.get("on_token")
        num_sequences = int(kwargs.get("n", 1))
        if callable(on_token):
            for seq_id in range(num_sequences):
                _ = on_token(
                    _MockOutput(
                        seq_id=seq_id,
                        token_id=seq_id + 1,
                        token_text=f"choice-{seq_id}",
                        finished=True,
                    )
                )

    mock_engine.add_request.side_effect = _add_request

    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 4,
            "n": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["choices"]) == 2
    assert [choice["index"] for choice in payload["choices"]] == [0, 1]


def test_completion_n_exceeds_max(client: TestClient) -> None:
    srv._max_n = 2

    response = client.post(
        "/v1/completions",
        json={
            "model": "test-model",
            "prompt": "hello",
            "max_tokens": 4,
            "n": 5,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request_error"


def test_json_mode_valid_json(client: TestClient) -> None:
    assert srv.engine is not None
    mock_engine = cast(Any, srv.engine)

    def _add_request(**kwargs: Any) -> None:
        on_token = kwargs.get("on_token")
        if callable(on_token):
            _ = on_token(
                _MockOutput(
                    seq_id=0,
                    token_id=1,
                    token_text='{"ok": true}',
                    finished=True,
                    finish_reason="stop",
                )
            )

    mock_engine.add_request.side_effect = _add_request

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "stop"


def test_json_mode_invalid_json(client: TestClient) -> None:
    assert srv.engine is not None
    mock_engine = cast(Any, srv.engine)

    def _add_request(**kwargs: Any) -> None:
        on_token = kwargs.get("on_token")
        if callable(on_token):
            _ = on_token(
                _MockOutput(
                    seq_id=0,
                    token_id=1,
                    token_text="hello world",
                    finished=True,
                    finish_reason="stop",
                )
            )

    mock_engine.add_request.side_effect = _add_request

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "error"


def test_text_mode_ignores_json_validation(client: TestClient) -> None:
    assert srv.engine is not None
    mock_engine = cast(Any, srv.engine)

    def _add_request(**kwargs: Any) -> None:
        on_token = kwargs.get("on_token")
        if callable(on_token):
            _ = on_token(
                _MockOutput(
                    seq_id=0,
                    token_id=1,
                    token_text="hello world",
                    finished=True,
                    finish_reason="stop",
                )
            )

    mock_engine.add_request.side_effect = _add_request

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "response_format": {"type": "text"},
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "stop"
