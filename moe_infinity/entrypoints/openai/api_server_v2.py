import argparse
import asyncio
import collections
import importlib
import json
import logging
import math
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from threading import Event, Lock, RLock
from types import SimpleNamespace
from typing import Any, Literal, Optional, cast

try:
    fastapi = importlib.import_module("fastapi")
    uvicorn = importlib.import_module("uvicorn")
    _fastapi_responses = importlib.import_module("fastapi.responses")
    HTTPException = getattr(fastapi, "HTTPException")
    Request = getattr(fastapi, "Request")
    JSONResponse = getattr(_fastapi_responses, "JSONResponse")
    Response = getattr(_fastapi_responses, "Response")
    StreamingResponse = getattr(_fastapi_responses, "StreamingResponse")
except Exception:

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Request:  # type: ignore[no-redef]
        async def is_disconnected(self) -> bool:
            return False

    class Response:  # type: ignore[no-redef]
        def __init__(
            self,
            status_code: int = 200,
            content: Any = None,
            media_type: Optional[str] = None,
        ) -> None:
            self.status_code = status_code
            self.content = content
            self.media_type = media_type

    class JSONResponse(Response):  # type: ignore[no-redef]
        pass

    class StreamingResponse(Response):  # type: ignore[no-redef]
        def __init__(
            self,
            content: Any,
            media_type: Optional[str] = None,
            status_code: int = 200,
        ) -> None:
            super().__init__(status_code=status_code, content=content)
            self.media_type = media_type

    class _FallbackFastAPI:
        def on_event(self, *_args: Any, **_kwargs: Any) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator

        def middleware(self, *_args: Any, **_kwargs: Any) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator

        def get(self, *_args: Any, **_kwargs: Any) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator

        def post(self, *_args: Any, **_kwargs: Any) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator

    fastapi = SimpleNamespace(FastAPI=_FallbackFastAPI)
    uvicorn = SimpleNamespace(run=lambda *args, **kwargs: None)

import moe_infinity.serving.watchdog as watchdog_module
from moe_infinity.serving.cuda_graph import FALLBACK_REASONS
from moe_infinity.serving.engine import ContinuousBatchingEngine, RequestOutput
from moe_infinity.serving.health import ServerHealthState
from moe_infinity.serving.sequence import SamplingParams
from moe_infinity.serving.stream import StreamManager
from moe_infinity.serving.validation import (
    ContextLengthExceededError,
    InvalidRequestError,
    validate_context_length,
    validate_required_params,
    validate_sampling_params,
)

from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    LogProbs,
    ModelCard,
    ModelList,
    UsageInfo,
    create_error_response,
    random_uuid,
)

TIMEOUT_KEEP_ALIVE = 5

app = fastapi.FastAPI()
engine: Optional[ContinuousBatchingEngine] = None
stream_manager: Optional[StreamManager] = None
tokenizer: Optional[object] = None
model_name_global: Optional[str] = None
runtime_max_seq_length: int = 4096

_engine_task: Optional[asyncio.Task[None]] = None
_engine_shutdown_event: Optional[asyncio.Event] = None
_engine_lifecycle_lock = RLock()
_model_init_task: Optional[asyncio.Task[None]] = None
_startup_args: Optional[argparse.Namespace] = None
_health_state = ServerHealthState()
_startup_watchdog: Optional[Any] = None
_decode_watchdog: Optional[Any] = None
_watchdog_config: Optional[Any] = None

_contextpilot_enabled: bool = False
_contextpilot_debug: bool = False
_contextpilot_fault: str = "none"
_contextpilot_state_lock = Lock()
_contextpilot_fallback_count: int = 0
_contextpilot_last_fallback_count: int = 0

_cp_logger = logging.getLogger("moe_infinity.contextpilot")
_logger = logging.getLogger("moe_infinity.api_server")
_cp_middleware: Optional[Any] = None
_eviction_sync: Optional[Any] = None
_cp_reorder_latencies: collections.deque[float] = collections.deque(maxlen=100)

_api_keys: set[str] = set()
_rate_limit_rpm: int = 0
_rate_limit_buckets: dict[str, list[float]] = {}
_rate_limit_lock = Lock()
_max_waiting_requests: int = 0
_max_n: int = 16


@dataclass
class _SequenceResultState:
    seq_id: int
    token_texts: list[str] = field(default_factory=list)
    text_offsets: list[int] = field(default_factory=list)
    token_logprobs: list[Optional[float]] = field(default_factory=list)
    top_logprobs: list[Optional[dict[int, float]]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    cumulative_logprob: float = 0.0
    token_count: int = 0
    current_text_offset: int = 0


@dataclass
class _FinalSequenceResult:
    seq_id: int
    text: str
    usage: dict[str, int]
    finish_reason: Optional[str]
    logprobs: Optional[LogProbs]
    cumulative_logprob: float
    token_count: int


def _resolve_contextpilot_enabled(cli_enabled: bool) -> bool:
    if os.environ.get("CONTEXTPILOT_ENABLED", "1") == "0":
        return False
    return bool(cli_enabled)


def _configure_auth(api_key_csv: Optional[str], rate_limit: int) -> None:
    global _api_keys
    global _rate_limit_rpm
    global _rate_limit_buckets

    configured_keys = {
        key.strip() for key in (api_key_csv or "").split(",") if key.strip()
    }
    _api_keys = configured_keys
    _rate_limit_rpm = max(0, int(rate_limit))
    with _rate_limit_lock:
        _rate_limit_buckets = {}


def _check_auth(request: Any) -> Any:
    if not _api_keys:
        return None

    if request.url.path in {"/health", "/metrics", "/v1/models"}:
        return None

    auth_header = str(request.headers.get("Authorization", ""))
    scheme, _, token = auth_header.partition(" ")
    if scheme != "Bearer" or not token or token not in _api_keys:
        return create_error_response(
            status_code=401,
            message="Invalid or missing API key.",
            error_type="authentication_error",
            code="invalid_api_key",
        )

    if _rate_limit_rpm <= 0:
        return None

    now = time.time()
    window_start = now - 60.0
    with _rate_limit_lock:
        bucket = _rate_limit_buckets.setdefault(token, [])
        bucket[:] = [
            timestamp for timestamp in bucket if timestamp >= window_start
        ]
        if len(bucket) >= _rate_limit_rpm:
            return create_error_response(
                status_code=429,
                message="Rate limit exceeded. Please retry later.",
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
            )
        bucket.append(now)

    return None


def _check_backpressure(runtime_engine: object) -> Any:
    if _max_waiting_requests <= 0:
        return None

    scheduler = getattr(runtime_engine, "scheduler", None)
    num_waiting = getattr(scheduler, "num_waiting", 0)
    if int(num_waiting) < _max_waiting_requests:
        return None

    return create_error_response(
        status_code=503,
        message="Server is at capacity. Please retry later.",
        error_type="server_error",
        code="queue_full",
    )


_configure_auth(os.environ.get("MOE_API_KEYS"), 0)


@app.middleware("http")
async def auth_middleware(request: Request, call_next: Any) -> Response:
    auth_result = _check_auth(request)
    if auth_result is not None:
        return auth_result
    return await call_next(request)


def _is_contextpilot_active() -> bool:
    with _contextpilot_state_lock:
        enabled = _contextpilot_enabled
    return enabled and os.environ.get("CONTEXTPILOT_ENABLED", "1") != "0"


def _current_contextpilot_fault() -> str:
    with _contextpilot_state_lock:
        return _contextpilot_fault


def _record_contextpilot_fallback(request_id: str, exc: Exception) -> None:
    with _contextpilot_state_lock:
        global _contextpilot_fallback_count
        global _contextpilot_last_fallback_count
        _contextpilot_fallback_count += 1
        _contextpilot_last_fallback_count = _contextpilot_fallback_count
    _cp_logger.warning(
        "request_id=%s CP middleware error, falling back: %s",
        request_id,
        exc,
    )


def _record_contextpilot_reorder_latency(latency_ms: float) -> None:
    with _contextpilot_state_lock:
        _cp_reorder_latencies.append(float(latency_ms))


def _compute_latency_stats(latencies: list[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0

    avg_latency_ms = sum(latencies) / len(latencies)
    ordered = sorted(latencies)
    p99_index = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return float(avg_latency_ms), float(ordered[p99_index])


def _log_contextpilot_request_metrics(
    *, request_id: str, middleware: Any
) -> None:
    metrics_fn = getattr(middleware, "get_last_request_metrics", None)
    metrics: dict[str, Any]
    if callable(metrics_fn):
        try:
            metrics = cast(dict[str, Any], metrics_fn())
        except Exception:
            metrics = {}
    else:
        metrics = {}

    reorder_latency_ms = float(metrics.get("reorder_latency_ms", 0.0) or 0.0)
    dedup_latency_ms = float(metrics.get("dedup_latency_ms", 0.0) or 0.0)
    tokens_saved = int(metrics.get("tokens_saved", 0) or 0)
    savings_pct = float(metrics.get("savings_pct", 0.0) or 0.0)

    if callable(metrics_fn):
        _record_contextpilot_reorder_latency(reorder_latency_ms)

    _cp_logger.info(
        "[contextpilot] request_id=%s reorder=%.1fms dedup=%.1fms tokens_saved=%d (%.1f%%)",
        request_id,
        reorder_latency_ms,
        dedup_latency_ms,
        tokens_saved,
        savings_pct,
    )


def _contextpilot_circuit_breaker_state() -> str:
    with _contextpilot_state_lock:
        enabled = _contextpilot_enabled
        fault = _contextpilot_fault
        middleware_present = _cp_middleware is not None

    if not enabled:
        return "closed"
    if fault != "none":
        return "open"
    if not middleware_present:
        return "half_open"
    return "closed"


def _contextpilot_eviction_sync_counters() -> dict[str, int]:
    sync_obj = _eviction_sync
    if sync_obj is None:
        return {"incoming": 0, "removed": 0, "not_found": 0}

    get_counters_fn = getattr(sync_obj, "get_counters", None)
    if not callable(get_counters_fn):
        return {"incoming": 0, "removed": 0, "not_found": 0}

    try:
        counters = cast(dict[str, Any], get_counters_fn())
    except Exception:
        return {"incoming": 0, "removed": 0, "not_found": 0}

    return {
        "incoming": int(counters.get("evict_incoming", 0) or 0),
        "removed": int(counters.get("evict_removed", 0) or 0),
        "not_found": int(counters.get("evict_not_found", 0) or 0),
    }


def _contextpilot_index_size(middleware: Optional[Any]) -> int:
    if middleware is None:
        return 0

    cp_obj = getattr(middleware, "_cp", None)
    live_index_obj = getattr(cp_obj, "live_index", None)
    if isinstance(live_index_obj, dict):
        return len(live_index_obj)
    return 0


def _ensure_cp_middleware_initialized() -> Optional[Any]:
    if not _is_contextpilot_active():
        return None

    with _contextpilot_state_lock:
        global _cp_middleware
        if _cp_middleware is not None:
            return _cp_middleware

    try:
        from moe_infinity.serving.contextpilot_middleware import (
            ContextPilotMiddleware,
        )

        middleware = ContextPilotMiddleware(use_gpu=False)
    except Exception as exc:
        _cp_logger.warning(
            "Failed to initialize ContextPilot middleware: %s",
            exc,
        )
        return None

    with _contextpilot_state_lock:
        if _cp_middleware is None:
            _cp_middleware = middleware
            _cp_logger.info("ContextPilot middleware initialized")
        return _cp_middleware


def _process_chat_messages_with_contextpilot(
    messages: Any,
    *,
    request_id: str,
) -> Any:
    if not isinstance(messages, list):
        return messages
    if not _is_contextpilot_active():
        return messages

    middleware = _ensure_cp_middleware_initialized()
    if middleware is None:
        return messages

    try:
        fault = _current_contextpilot_fault()
        if fault != "none":
            raise RuntimeError(f"CP fault injected: {fault}")

        processed_messages = middleware.process_chat_request(messages)
        _log_contextpilot_request_metrics(
            request_id=request_id,
            middleware=middleware,
        )
        return processed_messages
    except Exception as exc:
        _record_contextpilot_fallback(request_id, exc)
        return messages


def _process_completion_prompt_with_contextpilot(
    prompt: str,
    *,
    request_id: str,
) -> str:
    if not _is_contextpilot_active():
        return prompt

    middleware = _ensure_cp_middleware_initialized()
    if middleware is None:
        return prompt

    try:
        fault = _current_contextpilot_fault()
        if fault != "none":
            raise RuntimeError(f"CP fault injected: {fault}")

        optimized_prompt = middleware.process_completion_request(prompt)
        _log_contextpilot_request_metrics(
            request_id=request_id,
            middleware=middleware,
        )
        return optimized_prompt
    except Exception as exc:
        _record_contextpilot_fallback(request_id, exc)
        return prompt


def initialize_with_model(
    *,
    moe_model: object,
    model_name: Optional[str],
    tok: Optional[object],
    max_seq_length: int,
    device_memory_ratio: float = 0.75,
    kv_cache_ratio: float = 0.25,
    max_batch_size: int = 32,
    enable_prefix_caching: bool = False,
    speculative_draft: Optional[Any] = None,
    enable_decode_cuda_graphs: bool = False,
    decode_cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    decode_cuda_graph_context_sizes: tuple[int, ...] = (
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ),
    decode_cuda_graph_warmup_iters: int = 2,
    decode_cuda_graph_max_memory_bytes: int = 0,
    enable_deepseek_mla_paging: bool = False,
    max_resident_paged_speculative_sessions: int = 1,
    min_free_mla_blocks_after_admission: int = 1,
) -> None:
    """Initialize the v2 server with a pre-loaded MoE model.

    This is the programmatic entry point used by ``MoE.serve()``.  Instead of
    loading a model from scratch (which ``_initialize_model`` does when the
    server is started via CLI), this function wires a *pre-existing* MoE
    instance into the continuous-batching engine.
    """
    global engine, stream_manager, tokenizer, model_name_global
    global runtime_max_seq_length, _health_state

    hf_model = getattr(moe_model, "model", moe_model)
    offload_engine = getattr(moe_model, "engine", None)

    args = argparse.Namespace(
        device_memory_ratio=device_memory_ratio,
        kv_cache_ratio=kv_cache_ratio,
        max_batch_size=max_batch_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_decode_cuda_graphs=enable_decode_cuda_graphs,
        decode_cuda_graph_batch_sizes=decode_cuda_graph_batch_sizes,
        decode_cuda_graph_context_sizes=decode_cuda_graph_context_sizes,
        decode_cuda_graph_warmup_iters=decode_cuda_graph_warmup_iters,
        decode_cuda_graph_max_memory_bytes=decode_cuda_graph_max_memory_bytes,
        enable_deepseek_mla_paging=enable_deepseek_mla_paging,
        max_resident_paged_speculative_sessions=(
            max_resident_paged_speculative_sessions
        ),
        min_free_mla_blocks_after_admission=(
            min_free_mla_blocks_after_admission
        ),
    )
    engine_config = _build_engine_config(args=args, model=hf_model)

    tokenizer = tok
    model_name_global = model_name
    configured_max_seq_length = engine_config.get("max_seq_length")
    if isinstance(configured_max_seq_length, int):
        runtime_max_seq_length = configured_max_seq_length
    else:
        runtime_max_seq_length = int(max_seq_length)

    initialized_engine = ContinuousBatchingEngine(
        model=hf_model,
        engine=offload_engine,
        config=engine_config,
        tokenizer=tokenizer,
        speculative_draft=speculative_draft,
        decode_graph_capability_provider=moe_model,
    )
    if stream_manager is None:
        stream_manager = StreamManager()

    engine = initialized_engine
    _health_state.set_healthy()


def parse_prompt_format(prompt: Any) -> tuple[bool, list[Any]]:
    prompt_is_tokens = False
    prompts: list[Any] = [prompt]
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")
        if isinstance(prompt[0], str):
            prompts = prompt
        elif isinstance(prompt[0], int):
            prompt_is_tokens = True
            prompts = [prompt]
        elif (
            isinstance(prompt[0], list)
            and prompt[0]
            and isinstance(prompt[0][0], int)
        ):
            prompt_is_tokens = True
            prompts = prompt
        else:
            raise ValueError(
                "prompt must be a string, array of strings, array of tokens, or array of token arrays"
            )
    return prompt_is_tokens, prompts


def _extract_stop_sequences(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str)]
    return []


def _decode_token_text(token_id: int, token_text: Optional[str]) -> str:
    if token_text is not None:
        return token_text
    if tokenizer is None:
        return ""
    decode_fn = getattr(tokenizer, "decode", None)
    if not callable(decode_fn):
        return ""
    try:
        return str(decode_fn([token_id], skip_special_tokens=False))
    except Exception:
        return ""


def _build_sampling_params(
    *,
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
    stop: Any,
    logprobs: Optional[int | bool] = None,
) -> SamplingParams:
    return SamplingParams(
        temperature=float(temperature) if temperature is not None else 1.0,
        top_p=float(top_p) if top_p is not None else 1.0,
        logprobs=int(logprobs) if logprobs else 0,
        max_tokens=cast(int, max_tokens),
        stop=_extract_stop_sequences(stop),
    )


def _validate_parallel_sampling(
    *,
    n: Optional[int],
    best_of: Optional[int] = None,
) -> Optional[Any]:
    requested_n = int(n or 1)
    requested_best_of = int(best_of or requested_n)

    if requested_n <= 0:
        return create_error_response(
            status_code=400,
            message="n must be greater than 0",
            error_type="invalid_request_error",
            code="invalid_request_error",
            param="n",
        )
    if requested_n > _max_n:
        return create_error_response(
            status_code=400,
            message=f"n must be less than or equal to {_max_n}",
            error_type="invalid_request_error",
            code="invalid_request_error",
            param="n",
        )
    if requested_best_of <= 0:
        return create_error_response(
            status_code=400,
            message="best_of must be greater than 0",
            error_type="invalid_request_error",
            code="invalid_request_error",
            param="best_of",
        )
    if requested_best_of > _max_n:
        return create_error_response(
            status_code=400,
            message=f"best_of must be less than or equal to {_max_n}",
            error_type="invalid_request_error",
            code="invalid_request_error",
            param="best_of",
        )
    if requested_best_of < requested_n:
        return create_error_response(
            status_code=400,
            message="best_of must be greater than or equal to n",
            error_type="invalid_request_error",
            code="invalid_request_error",
            param="best_of",
        )
    return None


def _finalize_sequence_result(
    state: _SequenceResultState,
    *,
    include_logprobs: bool,
) -> _FinalSequenceResult:
    logprobs: Optional[LogProbs] = None
    if include_logprobs:
        logprobs = LogProbs(
            text_offset=state.text_offsets,
            token_logprobs=state.token_logprobs,
            tokens=state.token_texts,
            top_logprobs=state.top_logprobs,
        )

    return _FinalSequenceResult(
        seq_id=state.seq_id,
        text="".join(state.token_texts),
        usage=dict(state.usage),
        finish_reason=state.finish_reason,
        logprobs=logprobs,
        cumulative_logprob=state.cumulative_logprob,
        token_count=state.token_count,
    )


def _select_best_results(
    results: list[_FinalSequenceResult],
    *,
    n: int,
    best_of: int,
    use_logprobs: bool,
) -> list[_FinalSequenceResult]:
    if len(results) <= n and best_of <= n:
        return sorted(results, key=lambda result: result.seq_id)

    if use_logprobs:
        ranked = sorted(
            results,
            key=lambda result: (-result.cumulative_logprob, result.seq_id),
        )
    else:
        ranked = sorted(
            results,
            key=lambda result: (-result.token_count, result.seq_id),
        )

    selected = ranked[:n]
    return sorted(selected, key=lambda result: result.seq_id)


def _apply_response_format_validation(
    result: _FinalSequenceResult,
    response_format: Any,
) -> _FinalSequenceResult:
    if response_format is None:
        return result

    format_type = str(getattr(response_format, "type", "text"))
    if format_type != "json_object":
        return result

    try:
        json.loads(result.text)
    except Exception:
        result.finish_reason = "error"

    return result


def _format_prometheus_metrics(stats: dict[str, object]) -> str:
    lines = []
    lines.append("# HELP moe_requests_completed Total completed requests")
    lines.append("# TYPE moe_requests_completed counter")
    lines.append(f"moe_requests_completed {stats.get('completed_requests', 0)}")
    lines.append("# HELP moe_requests_cancelled Total cancelled requests")
    lines.append("# TYPE moe_requests_cancelled counter")
    lines.append(f"moe_requests_cancelled {stats.get('cancelled_requests', 0)}")
    lines.append("# HELP moe_queue_depth Current pending requests")
    lines.append("# TYPE moe_queue_depth gauge")
    lines.append(f"moe_queue_depth {stats.get('pending_requests', 0)}")
    lines.append("# HELP moe_tokens_generated_total Total tokens generated")
    lines.append("# TYPE moe_tokens_generated_total counter")
    lines.append(
        f"moe_tokens_generated_total {stats.get('total_generated_tokens', 0)}"
    )
    lines.append("# HELP moe_kv_cache_free_blocks Free KV cache blocks")
    lines.append("# TYPE moe_kv_cache_free_blocks gauge")
    lines.append(
        f"moe_kv_cache_free_blocks {stats.get('kv_cache_free_blocks', 0)}"
    )
    lines.append("# HELP moe_kv_cache_total_blocks Total KV cache blocks")
    lines.append("# TYPE moe_kv_cache_total_blocks gauge")
    lines.append(
        f"moe_kv_cache_total_blocks {stats.get('kv_cache_num_blocks', 0)}"
    )
    lines.append("# HELP moe_engine_steps_total Total engine steps")
    lines.append("# TYPE moe_engine_steps_total counter")
    lines.append(f"moe_engine_steps_total {stats.get('num_steps', 0)}")
    cuda_graph = stats.get("cuda_graph", {})
    graph_stats = (
        cast(dict[str, object], cuda_graph)
        if isinstance(cuda_graph, dict)
        else {}
    )
    graph_metrics = (
        ("moe_cuda_graph_captures_total", "captures", "counter"),
        ("moe_cuda_graph_replays_total", "replays", "counter"),
        (
            "moe_cuda_graph_capture_failures_total",
            "capture_failures",
            "counter",
        ),
        ("moe_cuda_graph_instances", "graphs", "gauge"),
        ("moe_cuda_graph_pool_bytes", "graph_pool_bytes", "gauge"),
        (
            "moe_cuda_graph_scratch_kv_bytes",
            "scratch_kv_bytes",
            "gauge",
        ),
    )
    for metric_name, stats_key, metric_type in graph_metrics:
        lines.append(f"# TYPE {metric_name} {metric_type}")
        lines.append(f"{metric_name} {graph_stats.get(stats_key, 0)}")

    fallback_value = graph_stats.get("fallback_reasons", {})
    fallback_reasons = (
        cast(dict[str, object], fallback_value)
        if isinstance(fallback_value, dict)
        else {}
    )
    lines.append("# TYPE moe_cuda_graph_fallback_total counter")
    for reason in FALLBACK_REASONS:
        count = fallback_reasons.get(reason, 0)
        lines.append(
            f'moe_cuda_graph_fallback_total{{reason="{reason}"}} {count}'
        )
    return "\n".join(lines) + "\n"


def _tokenize_text(prompt: str) -> list[int]:
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="tokenizer not initialized")
    encode_fn = getattr(tokenizer, "encode", None)
    if not callable(encode_fn):
        raise HTTPException(
            status_code=500, detail="tokenizer.encode unavailable"
        )

    token_ids = encode_fn(prompt)
    if isinstance(token_ids, list):
        return [int(token_id) for token_id in token_ids]
    raise HTTPException(status_code=500, detail="failed to encode prompt")


def _chat_prompt_to_token_ids(request: ChatCompletionRequest) -> list[int]:
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="tokenizer not initialized")

    if isinstance(request.messages, str):
        return _tokenize_text(request.messages)

    apply_chat_template_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template_fn):
        prompt = apply_chat_template_fn(
            conversation=request.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return _tokenize_text(str(prompt))

    rendered = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in request.messages
    )
    rendered += "\nassistant:"
    return _tokenize_text(rendered)


def _ensure_runtime_ready() -> tuple[ContinuousBatchingEngine, StreamManager]:
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    if stream_manager is None:
        raise HTTPException(
            status_code=503, detail="stream manager not initialized"
        )
    return engine, stream_manager


async def _wait_non_stream_result(
    *,
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    raw_request: Request,
    expected_sequences: int = 1,
) -> list[_FinalSequenceResult]:
    runtime_engine, _ = _ensure_runtime_ready()
    done_event = Event()
    finished_seq_ids: set[int] = set()
    results_by_seq_id: dict[int, _SequenceResultState] = {}

    def on_token(output: RequestOutput) -> None:
        state = results_by_seq_id.setdefault(
            output.seq_id,
            _SequenceResultState(
                seq_id=output.seq_id,
                usage={
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": 0,
                    "total_tokens": len(prompt_token_ids),
                },
            ),
        )
        token_text = _decode_token_text(output.token_id, output.token_text)
        state.token_texts.append(token_text)
        state.token_count += 1
        if sampling_params.logprobs > 0:
            state.text_offsets.append(state.current_text_offset)
            state.token_logprobs.append(output.token_logprob)
            state.top_logprobs.append(output.top_logprobs)
            state.current_text_offset += len(token_text)
        if output.token_logprob is not None:
            state.cumulative_logprob += output.token_logprob
        if output.usage is not None:
            state.usage = output.usage
        if output.finished:
            state.finish_reason = output.finish_reason or "stop"
            finished_seq_ids.add(output.seq_id)
            if len(finished_seq_ids) >= expected_sequences:
                done_event.set()

    runtime_engine.add_request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        on_token=on_token,
        n=expected_sequences,
    )

    while not done_event.is_set():
        if await raw_request.is_disconnected():
            runtime_engine.abort_request(request_id)
            raise HTTPException(status_code=499, detail="client disconnected")
        await asyncio.sleep(0.01)

    return [
        _finalize_sequence_result(
            state,
            include_logprobs=sampling_params.logprobs > 0,
        )
        for seq_id, state in sorted(results_by_seq_id.items())
    ]


def _next_stream_event(generator: Any) -> Optional[str]:
    try:
        return next(generator)
    except StopIteration:
        return None


def _model_to_json(model: object, *, exclude_none: bool = False) -> str:
    model_dump_json_fn = getattr(model, "model_dump_json", None)
    if callable(model_dump_json_fn):
        return str(model_dump_json_fn(exclude_none=exclude_none))

    json_fn = getattr(model, "json", None)
    if callable(json_fn):
        return str(json_fn(exclude_none=exclude_none))

    raise TypeError("response model does not support JSON serialization")


def _make_completion_stream_chunk(
    *,
    request_id: str,
    created: int,
    model_name: str,
    token_text: str,
    finish_reason: Optional[str],
) -> str:
    resolved_finish_reason: Optional[Literal["stop", "length"]] = None
    if finish_reason == "stop":
        resolved_finish_reason = "stop"
    elif finish_reason == "length":
        resolved_finish_reason = "length"

    chunk = CompletionStreamResponse(
        id=f"cmpl-{request_id}",
        created=created,
        model=model_name,
        choices=[
            CompletionResponseStreamChoice(
                index=0,
                text=token_text,
                finish_reason=resolved_finish_reason,
            )
        ],
    )
    return f"data: {_model_to_json(chunk, exclude_none=True)}\n\n"


async def _completion_event_generator(
    *,
    request_id: str,
    created: int,
    model_name: str,
    stream: Any,
    raw_request: Request,
) -> Any:
    runtime_engine, _ = _ensure_runtime_ready()
    while True:
        if await raw_request.is_disconnected():
            runtime_engine.abort_request(request_id)
            return
        try:
            event = await asyncio.wait_for(
                asyncio.to_thread(_next_stream_event, stream),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            continue

        if event is None:
            return
        if event == "data: [DONE]\n\n":
            yield event
            return

        if not event.startswith("data: "):
            continue
        payload_text = event[len("data: ") :].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue

        choices = payload.get("choices", [])
        first_choice = choices[0] if choices else {}
        delta = first_choice.get("delta", {})
        token_text = delta.get("content", "") if isinstance(delta, dict) else ""
        finish_reason = first_choice.get("finish_reason")
        yield _make_completion_stream_chunk(
            request_id=request_id,
            created=created,
            model_name=model_name,
            token_text=token_text,
            finish_reason=finish_reason,
        )


async def _chat_event_generator(
    *, request_id: str, stream: Any, raw_request: Request
) -> Any:
    runtime_engine, _ = _ensure_runtime_ready()
    while True:
        if await raw_request.is_disconnected():
            runtime_engine.abort_request(request_id)
            return
        try:
            event = await asyncio.wait_for(
                asyncio.to_thread(_next_stream_event, stream),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            continue

        if event is None:
            return
        yield event
        if event == "data: [DONE]\n\n":
            return


def _run_engine_step_once() -> None:
    with _engine_lifecycle_lock:
        current_engine = engine
        if current_engine is not None:
            _ = current_engine.step()


def _replace_engine(new_engine: ContinuousBatchingEngine) -> None:
    global engine
    with _engine_lifecycle_lock:
        old_engine = engine
        engine = new_engine
        if old_engine is not None and old_engine is not new_engine:
            old_engine.shutdown()


async def _engine_loop() -> None:
    while (
        _engine_shutdown_event is not None
        and not _engine_shutdown_event.is_set()
    ):
        with _engine_lifecycle_lock:
            current_engine = engine
        if current_engine is None:
            await asyncio.sleep(0.01)
            continue

        with _engine_lifecycle_lock:
            current_engine = engine
            has_pending_requests = (
                current_engine is not None
                and current_engine.has_pending_requests()
            )
            if has_pending_requests:
                if _decode_watchdog is not None:
                    _decode_watchdog.activate()
                _ = current_engine.step()
                if _decode_watchdog is not None:
                    _decode_watchdog.feed()
        if has_pending_requests:
            await asyncio.sleep(0)
            continue

        if _decode_watchdog is not None:
            _decode_watchdog.deactivate()
        await asyncio.sleep(0.005)


def _ensure_engine_loop_running() -> None:
    global _engine_task
    global _engine_shutdown_event

    if engine is None:
        return

    if _engine_shutdown_event is None or _engine_shutdown_event.is_set():
        _engine_shutdown_event = asyncio.Event()

    if _engine_task is None or _engine_task.done():
        _engine_task = asyncio.create_task(_engine_loop())


async def _initialize_model() -> None:
    global engine
    global _health_state
    global runtime_max_seq_length
    global stream_manager
    global tokenizer
    global model_name_global
    global _startup_watchdog
    global _decode_watchdog
    global _watchdog_config

    args = _startup_args
    if args is None or engine is not None:
        return

    _health_state.set_starting()
    _startup_watchdog = None
    _decode_watchdog = None
    _watchdog_config = None

    startup_timeout = getattr(args, "startup_timeout", None)
    decode_step_timeout = getattr(args, "decode_step_timeout", None)
    if startup_timeout is not None or decode_step_timeout is not None:
        _watchdog_config = watchdog_module.WatchdogConfig(
            startup_timeout=startup_timeout,
            decode_step_timeout=decode_step_timeout,
            enable_pyspy_dump=getattr(args, "enable_pyspy_dump", False),
        )
        _startup_watchdog = watchdog_module.start_startup_watchdog(
            _health_state,
            _watchdog_config,
            is_ready=lambda: engine is not None,
        )

    try:
        from transformers import AutoTokenizer

        model_name_global = args.model
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
        )

        enable_deepseek_mla_paging = bool(
            getattr(args, "enable_deepseek_mla_paging", False)
        )
        max_resident_paged_speculative_sessions = int(
            getattr(args, "max_resident_paged_speculative_sessions", 1)
        )
        min_free_mla_blocks_after_admission = int(
            getattr(args, "min_free_mla_blocks_after_admission", 1)
        )
        moe_config = {
            "offload_path": os.path.join(args.offload_dir, args.model),
            "device_memory_ratio": args.device_memory_ratio,
            "enable_deepseek_mla_paging": enable_deepseek_mla_paging,
            "max_resident_paged_speculative_sessions": (
                max_resident_paged_speculative_sessions
            ),
            "min_free_mla_blocks_after_admission": (
                min_free_mla_blocks_after_admission
            ),
        }
        if args.enable_prefix_caching:
            moe_config["enable_prefix_caching"] = True

        moe_module = importlib.import_module("moe_infinity")
        moe_ctor = getattr(moe_module, "MoE", None)
        if moe_ctor is None:
            raise RuntimeError("MoE is unavailable in this environment")

        moe_model = moe_ctor(args.model, moe_config)
        engine_config = _build_engine_config(args=args, model=moe_model.model)
        speculative_draft = None
        draft_path = getattr(args, "speculative_draft", None)
        if draft_path:
            resolve_spec = getattr(moe_model, "_resolve_spec_strategy", None)
            if not callable(resolve_spec):
                raise RuntimeError("loaded MoE model cannot configure DFlash")
            resolve_spec(str(draft_path))
            speculative_draft = getattr(moe_model, "_dflash_speculator", None)
            if speculative_draft is None:
                raise RuntimeError("failed to configure DFlash speculator")

        initialized_engine = ContinuousBatchingEngine(
            model=moe_model.model,
            engine=moe_model.engine,
            config=engine_config,
            tokenizer=tokenizer,
            speculative_draft=speculative_draft,
            decode_graph_capability_provider=moe_model,
        )
        if stream_manager is None:
            stream_manager = StreamManager()

        _replace_engine(initialized_engine)
        configured_max_seq_length = engine_config.get("max_seq_length")
        if isinstance(configured_max_seq_length, int):
            runtime_max_seq_length = configured_max_seq_length

        if _watchdog_config is not None:
            _decode_watchdog = watchdog_module.start_decode_watchdog(
                _health_state,
                _watchdog_config,
            )

        _ensure_engine_loop_running()
    except Exception as exc:
        # A failed async init must surface as UNHEALTHY, not hang forever at
        # STARTING: the exception is stored on the un-awaited init task and is
        # otherwise never seen (issue #146 Bug 2).
        _logger.exception("model initialization failed")
        _health_state.set_unhealthy(f"model initialization failed: {exc}")
        return
    finally:
        if _startup_watchdog is not None:
            _startup_watchdog.cancel()

    _health_state.set_healthy()


@app.on_event("startup")
async def startup_event() -> None:
    global _model_init_task
    global _engine_task
    global stream_manager

    if stream_manager is None:
        stream_manager = StreamManager()

    _ = _ensure_cp_middleware_initialized()

    if _startup_args is not None and engine is None:
        if _model_init_task is None or _model_init_task.done():
            _model_init_task = asyncio.create_task(_initialize_model())
        return

    _ensure_engine_loop_running()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _engine_task
    global _model_init_task
    global _decode_watchdog

    if _model_init_task is not None:
        if not _model_init_task.done():
            _model_init_task.cancel()
        with suppress(asyncio.CancelledError):
            await _model_init_task
        _model_init_task = None

    if _engine_shutdown_event is not None:
        _engine_shutdown_event.set()

    if _engine_task is not None:
        await _engine_task
        _engine_task = None

    if _decode_watchdog is not None:
        _decode_watchdog.deactivate()
        _decode_watchdog.stop()
        _decode_watchdog.join(timeout=1.0)
        _decode_watchdog = None

    with _engine_lifecycle_lock:
        current_engine = engine
        if current_engine is not None:
            current_engine.shutdown()


@app.get("/health")
async def health():
    status_dict = _health_state.get_status_dict()
    is_healthy = _health_state.is_healthy()
    status_code = 200 if is_healthy else 503
    return JSONResponse(content=status_dict, status_code=status_code)


@app.get("/admin/stats")
async def admin_stats():
    if engine is None:
        return create_error_response(
            status_code=503,
            message="Service is starting. Please retry shortly.",
            error_type="server_error",
            code="service_starting",
        )

    return JSONResponse(content=engine.get_stats())


@app.get("/metrics")
async def metrics():
    stats = engine.get_stats() if engine is not None else {}
    body = _format_prometheus_metrics(stats)
    return Response(content=body, media_type="text/plain; charset=utf-8")


@app.post("/contextpilot/toggle")
async def contextpilot_toggle(payload: dict[str, Any]) -> JSONResponse:
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(
            status_code=400,
            detail="'enabled' must be a boolean",
        )

    with _contextpilot_state_lock:
        global _contextpilot_enabled
        _contextpilot_enabled = enabled

    if enabled:
        _ = _ensure_cp_middleware_initialized()

    return JSONResponse(
        content={"enabled": enabled},
        status_code=200,
    )


@app.post("/contextpilot/inject-fault")
async def contextpilot_inject_fault(payload: dict[str, Any]) -> JSONResponse:
    with _contextpilot_state_lock:
        if not _contextpilot_debug:
            raise HTTPException(
                status_code=403,
                detail="ContextPilot debug endpoints are disabled",
            )

    fault = payload.get("fault")
    if fault not in {"none", "reorder_exception"}:
        raise HTTPException(
            status_code=400,
            detail="'fault' must be one of: none, reorder_exception",
        )

    duration_s = payload.get("duration_s", 0)
    if not isinstance(duration_s, int) or duration_s < 0:
        raise HTTPException(
            status_code=400,
            detail="'duration_s' must be a non-negative integer",
        )

    with _contextpilot_state_lock:
        global _contextpilot_fault
        _contextpilot_fault = fault

    return JSONResponse(
        content={
            "fault": fault,
            "duration_s": duration_s,
        },
        status_code=200,
    )


@app.get("/contextpilot/status")
async def contextpilot_status() -> JSONResponse:
    with _contextpilot_state_lock:
        enabled = _contextpilot_enabled
        debug = _contextpilot_debug
        fault = _contextpilot_fault
        fallback_count = _contextpilot_fallback_count
        last_fallback_count = _contextpilot_last_fallback_count
        middleware = _cp_middleware
        latencies = list(_cp_reorder_latencies)

    token_stats: dict[str, Any] = {}
    request_metrics: dict[str, Any] = {}
    if middleware is not None:
        get_token_savings_fn = getattr(middleware, "get_token_savings", None)
        if callable(get_token_savings_fn):
            with suppress(Exception):
                token_stats = cast(dict[str, Any], get_token_savings_fn())

        get_request_metrics_fn = getattr(
            middleware, "get_last_request_metrics", None
        )
        if callable(get_request_metrics_fn):
            with suppress(Exception):
                request_metrics = cast(dict[str, Any], get_request_metrics_fn())

    avg_reorder_latency_ms, p99_reorder_latency_ms = _compute_latency_stats(
        latencies
    )
    eviction_sync_counters = _contextpilot_eviction_sync_counters()
    cp_index_size = _contextpilot_index_size(middleware)
    circuit_breaker_state = _contextpilot_circuit_breaker_state()

    requests_processed = int(
        request_metrics.get(
            "requests_processed",
            token_stats.get("requests_processed", 0),
        )
        or 0
    )
    reorder_count = int(request_metrics.get("reorder_count", 0) or 0)
    dedup_count = int(request_metrics.get("dedup_count", 0) or 0)
    token_savings_total = int(token_stats.get("total_tokens_saved", 0) or 0)
    token_savings_avg_pct = float(
        token_stats.get("avg_savings_pct", 0.0) or 0.0
    )

    return JSONResponse(
        content={
            "enabled": enabled,
            "circuit_breaker_state": circuit_breaker_state,
            "requests_processed": requests_processed,
            "reorder_count": reorder_count,
            "dedup_count": dedup_count,
            "avg_reorder_latency_ms": avg_reorder_latency_ms,
            "p99_reorder_latency_ms": p99_reorder_latency_ms,
            "token_savings_total": token_savings_total,
            "token_savings_avg_pct": token_savings_avg_pct,
            "eviction_sync": eviction_sync_counters,
            "cp_index_size": cp_index_size,
            "last_fallback_count": last_fallback_count,
            "debug": debug,
            "fault": fault,
            "fallback_count": fallback_count,
            "env_enabled": os.environ.get("CONTEXTPILOT_ENABLED", "1") != "0",
        },
        status_code=200,
    )


@app.post("/v1/completions")
async def completion(request: CompletionRequest, raw_request: Request):
    if engine is None:
        return create_error_response(
            status_code=503,
            message="Service is starting. Please retry shortly.",
            error_type="server_error",
            code="service_starting",
        )

    runtime_engine, runtime_stream_manager = _ensure_runtime_ready()
    backpressure_response = _check_backpressure(runtime_engine)
    if backpressure_response is not None:
        return backpressure_response
    created_time = int(time.monotonic())
    model_name = request.model or model_name_global or "unknown"

    try:
        validate_required_params(request.max_tokens)
        validate_sampling_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
    except InvalidRequestError as exc:
        return create_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
            param=exc.param,
        )

    parallel_validation = _validate_parallel_sampling(
        n=request.n,
        best_of=request.best_of,
    )
    if parallel_validation is not None:
        return parallel_validation

    requested_n = int(request.n or 1)
    requested_best_of = int(request.best_of or requested_n)

    try:
        prompt_is_tokens, prompts = parse_prompt_format(request.prompt)
    except ValueError as exc:
        return create_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    sampling_params = _build_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
        logprobs=request.logprobs,
    )

    if request.stream:
        if requested_n != 1 or requested_best_of != 1:
            return create_error_response(
                status_code=400,
                message="streaming completions only support n=1 and best_of=1",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        if len(prompts) != 1:
            return create_error_response(
                status_code=400,
                message="streaming completion only supports a single prompt",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )

        request_id = random_uuid()
        prompt = prompts[0]
        if prompt_is_tokens:
            prompt_token_ids = [int(token_id) for token_id in prompt]
        else:
            processed_prompt = _process_completion_prompt_with_contextpilot(
                str(prompt),
                request_id=request_id,
            )
            prompt_token_ids = _tokenize_text(processed_prompt)

        try:
            validate_context_length(
                len(prompt_token_ids),
                cast(int, request.max_tokens),
                runtime_max_seq_length,
            )
        except ContextLengthExceededError as exc:
            return create_error_response(
                status_code=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="context_length_exceeded",
            )

        stream = runtime_stream_manager.create_stream(
            request_id=request_id,
            model=model_name,
        )

        def on_stream_token(output: RequestOutput) -> None:
            runtime_stream_manager.push_token(
                request_id=request_id,
                token_text=_decode_token_text(
                    output.token_id, output.token_text
                ),
                finished=output.finished,
                finish_reason=output.finish_reason,
            )

        runtime_engine.add_request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            on_token=on_stream_token,
            n=1,
        )

        return StreamingResponse(
            _completion_event_generator(
                request_id=request_id,
                created=created_time,
                model_name=model_name,
                stream=stream,
                raw_request=raw_request,
            ),
            media_type="text/event-stream",
        )

    choices: list[CompletionResponseChoice] = []
    usage_prompt_tokens = 0
    usage_completion_tokens = 0
    for index, prompt in enumerate(prompts):
        request_id = random_uuid()
        if prompt_is_tokens:
            prompt_token_ids = [int(token_id) for token_id in prompt]
        else:
            processed_prompt = _process_completion_prompt_with_contextpilot(
                str(prompt),
                request_id=request_id,
            )
            prompt_token_ids = _tokenize_text(processed_prompt)

        try:
            validate_context_length(
                len(prompt_token_ids),
                cast(int, request.max_tokens),
                runtime_max_seq_length,
            )
        except ContextLengthExceededError as exc:
            return create_error_response(
                status_code=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="context_length_exceeded",
            )

        sequence_results = await _wait_non_stream_result(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            raw_request=raw_request,
            expected_sequences=requested_best_of,
        )

        selected_results = _select_best_results(
            sequence_results,
            n=requested_n,
            best_of=requested_best_of,
            use_logprobs=sampling_params.logprobs > 0,
        )
        selected_results = [
            _apply_response_format_validation(result, request.response_format)
            for result in selected_results
        ]

        for result in selected_results:
            choices.append(
                CompletionResponseChoice(
                    index=len(choices),
                    text=result.text,
                    logprobs=result.logprobs,
                    finish_reason=cast(
                        Literal["stop", "length", "error"],
                        result.finish_reason or "stop",
                    ),
                )
            )
            usage_completion_tokens += result.usage.get("completion_tokens", 0)
        usage_prompt_tokens += len(prompt_token_ids)

    return CompletionResponse(
        id=f"cmpl-{random_uuid()}",
        created=created_time,
        model=model_name,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=usage_prompt_tokens,
            completion_tokens=usage_completion_tokens,
            total_tokens=usage_prompt_tokens + usage_completion_tokens,
        ),
    )


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest, raw_request: Request):
    if engine is None:
        return create_error_response(
            status_code=503,
            message="Service is starting. Please retry shortly.",
            error_type="server_error",
            code="service_starting",
        )

    runtime_engine, runtime_stream_manager = _ensure_runtime_ready()
    backpressure_response = _check_backpressure(runtime_engine)
    if backpressure_response is not None:
        return backpressure_response
    created_time = int(time.monotonic())
    model_name = request.model or model_name_global or "unknown"
    request_id = random_uuid()

    try:
        validate_required_params(request.max_tokens)
        validate_sampling_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
    except InvalidRequestError as exc:
        return create_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
            param=exc.param,
        )

    parallel_validation = _validate_parallel_sampling(n=request.n)
    if parallel_validation is not None:
        return parallel_validation

    requested_n = int(request.n or 1)

    sampling_params = _build_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
        logprobs=request.top_logprobs if request.logprobs else request.logprobs,
    )
    processed_messages = _process_chat_messages_with_contextpilot(
        request.messages,
        request_id=request_id,
    )
    original_messages = request.messages
    request.messages = processed_messages
    try:
        prompt_token_ids = _chat_prompt_to_token_ids(request)
    finally:
        request.messages = original_messages

    try:
        validate_context_length(
            len(prompt_token_ids),
            cast(int, request.max_tokens),
            runtime_max_seq_length,
        )
    except ContextLengthExceededError as exc:
        return create_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="context_length_exceeded",
        )

    if request.stream:
        if requested_n != 1:
            return create_error_response(
                status_code=400,
                message="streaming chat completions only support n=1",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        stream = runtime_stream_manager.create_stream(
            request_id=request_id,
            model=model_name,
        )

        def on_stream_token(output: RequestOutput) -> None:
            runtime_stream_manager.push_token(
                request_id=request_id,
                token_text=_decode_token_text(
                    output.token_id, output.token_text
                ),
                finished=output.finished,
                finish_reason=output.finish_reason,
            )

        runtime_engine.add_request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            on_token=on_stream_token,
            n=1,
        )
        return StreamingResponse(
            _chat_event_generator(
                request_id=request_id,
                stream=stream,
                raw_request=raw_request,
            ),
            media_type="text/event-stream",
        )

    sequence_results = await _wait_non_stream_result(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        raw_request=raw_request,
        expected_sequences=requested_n,
    )
    sequence_results = [
        _apply_response_format_validation(result, request.response_format)
        for result in sequence_results
    ]

    usage_completion_tokens = sum(
        result.usage.get("completion_tokens", 0) for result in sequence_results
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=created_time,
        model=model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=index,
                message=ChatMessage(role="assistant", content=result.text),
                logprobs=result.logprobs,
                finish_reason=cast(
                    Literal["stop", "length", "error"],
                    result.finish_reason or "stop",
                ),
            )
            for index, result in enumerate(sequence_results)
        ],
        usage=UsageInfo(
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=usage_completion_tokens,
            total_tokens=len(prompt_token_ids) + usage_completion_tokens,
        ),
    )


@app.get("/v1/models")
async def list_models():
    if engine is None:
        return create_error_response(
            status_code=503,
            message="Service is starting. Please retry shortly.",
            error_type="server_error",
            code="service_starting",
        )

    model_name = model_name_global or "unknown"
    card = ModelCard(id=model_name)
    return ModelList(data=[card])


@app.post("/v1/reload")
async def reload_modules(payload: dict[str, Any]) -> JSONResponse:
    modules = payload.get("modules", [])
    if not isinstance(modules, list) or not all(
        isinstance(m, str) for m in modules
    ):
        raise HTTPException(
            status_code=400, detail="'modules' must be a list of strings"
        )
    reloaded = []
    errors = []
    with _engine_lifecycle_lock:
        current_engine = engine
        if current_engine is not None:
            current_engine.invalidate_cuda_graphs("module_reload")
        for module_name in modules:
            try:
                mod = importlib.import_module(module_name)
                importlib.reload(mod)
                reloaded.append(module_name)
            except Exception as e:
                errors.append({"module": module_name, "error": str(e)})
    status = "ok" if not errors else "partial"
    return JSONResponse(
        content={"status": status, "reloaded": reloaded, "errors": errors}
    )


@app.get("/v1/config")
async def get_config() -> JSONResponse:
    if engine is None:
        return create_error_response(
            status_code=503,
            message="Service starting",
            error_type="server_error",
            code="service_starting",
        )
    return JSONResponse(content=engine.get_config())


@app.post("/v1/config")
async def update_config(payload: dict[str, Any]) -> JSONResponse:
    if engine is None:
        return create_error_response(
            status_code=503,
            message="Service starting",
            error_type="server_error",
            code="service_starting",
        )
    updated = engine.update_config(payload)
    return JSONResponse(content={"status": "ok", "updated": updated})


def _resolve_int_attr(config: object, *names: str) -> Optional[int]:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    return None


def _resolve_dtype(model: object) -> str:
    model_config = getattr(model, "config", None)
    if model_config is not None:
        torch_dtype = getattr(model_config, "dtype", None)
        if torch_dtype is None:
            torch_dtype = getattr(model_config, "torch_dtype", None)
        if isinstance(torch_dtype, str):
            return torch_dtype.replace("torch.", "")
        if torch_dtype is not None:
            return str(torch_dtype).replace("torch.", "")
    return "float16"


def _build_engine_config(
    args: argparse.Namespace, model: object
) -> dict[str, object]:
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise RuntimeError(
            "model config is required to initialize serving engine"
        )
    # Multimodal MoE checkpoints (e.g. Qwen3.5-MoE VL) nest the text backbone
    # dimensions under text_config; get_text_config() returns self otherwise.
    text_config = (
        model_config.get_text_config()
        if hasattr(model_config, "get_text_config")
        else model_config
    )

    num_layers = _resolve_int_attr(
        text_config,
        "num_hidden_layers",
        "num_layers",
        "n_layer",
    )
    num_attention_heads = _resolve_int_attr(
        text_config,
        "num_attention_heads",
        "n_head",
    )
    num_kv_heads = _resolve_int_attr(
        text_config,
        "num_key_value_heads",
        "num_kv_heads",
        "n_head_kv",
    )
    hidden_size = _resolve_int_attr(text_config, "hidden_size", "n_embd")
    max_seq_length = _resolve_int_attr(
        text_config,
        "max_position_embeddings",
        "max_seq_len",
        "max_sequence_length",
        "n_positions",
        "model_max_length",
    )
    head_dim = _resolve_int_attr(text_config, "head_dim")

    if num_layers is None:
        raise RuntimeError("unable to resolve model num_layers")
    if num_attention_heads is None:
        raise RuntimeError("unable to resolve model num_attention_heads")
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads
    if head_dim is None:
        if hidden_size is None:
            raise RuntimeError("unable to resolve model hidden_size/head_dim")
        head_dim = hidden_size // max(1, num_attention_heads)

    eos_token_id: Optional[int] = _resolve_int_attr(
        model_config, "eos_token_id"
    )
    if eos_token_id is None:
        eos_token_id = _resolve_int_attr(text_config, "eos_token_id")
    if eos_token_id is None:
        config_eos = getattr(model_config, "eos_token_id", None) or getattr(
            text_config, "eos_token_id", None
        )
        if (
            isinstance(config_eos, list)
            and config_eos
            and isinstance(config_eos[0], int)
        ):
            eos_token_id = config_eos[0]

    config: dict[str, object] = {
        "device_memory_ratio": args.device_memory_ratio,
        "kv_cache_ratio": args.kv_cache_ratio,
        "max_batch_size": args.max_batch_size,
        "max_tokens_per_step": 2048,
        "max_seq_length": max_seq_length or 4096,
        "block_size": 16,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": _resolve_dtype(model),
        "enable_decode_cuda_graphs": bool(
            getattr(args, "enable_decode_cuda_graphs", False)
        ),
        "decode_cuda_graph_batch_sizes": tuple(
            getattr(
                args,
                "decode_cuda_graph_batch_sizes",
                (1, 2, 4, 8, 16, 32),
            )
        ),
        "decode_cuda_graph_context_sizes": tuple(
            getattr(
                args,
                "decode_cuda_graph_context_sizes",
                (128, 256, 512, 1024, 2048, 4096),
            )
        ),
        "decode_cuda_graph_warmup_iters": getattr(
            args, "decode_cuda_graph_warmup_iters", 2
        ),
        "decode_cuda_graph_max_memory_bytes": getattr(
            args, "decode_cuda_graph_max_memory_bytes", 0
        ),
        "enable_deepseek_mla_paging": bool(
            getattr(args, "enable_deepseek_mla_paging", False)
        ),
        "max_resident_paged_speculative_sessions": int(
            getattr(args, "max_resident_paged_speculative_sessions", 1)
        ),
        "min_free_mla_blocks_after_admission": int(
            getattr(args, "min_free_mla_blocks_after_admission", 1)
        ),
    }
    if eos_token_id is not None:
        config["eos_token_id"] = eos_token_id
    if args.enable_prefix_caching:
        config["enable_prefix_caching"] = True
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE-Infinity OpenAI-Compatible RESTful API server v2."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--offload-dir", type=str, required=True)
    parser.add_argument(
        "--speculative-draft",
        type=str,
        default=None,
        help="DFlash drafter checkpoint for greedy batch-1 serving",
    )
    parser.add_argument("--device-memory-ratio", type=float, default=0.75)
    parser.add_argument("--kv-cache-ratio", type=float, default=0.25)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument(
        "--enable-deepseek-mla-paging",
        action="store_true",
        default=False,
        help="Enable default-off DeepSeek V2/V3 paged MLA serving",
    )
    parser.add_argument(
        "--max-resident-paged-speculative-sessions",
        type=int,
        default=1,
        help="Maximum concurrent resident paged-MLA speculative sessions",
    )
    parser.add_argument(
        "--min-free-mla-blocks-after-admission",
        type=int,
        default=1,
        help=(
            "Minimum MLA blocks that remain free after reserving all active "
            "and newly admitted requests' full declared budgets plus maximum "
            "transient DFlash verify peaks"
        ),
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--rate-limit", type=int, default=0)
    parser.add_argument("--max-waiting-requests", type=int, default=0)
    parser.add_argument("--max-n", type=int, default=16)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--enable-decode-cuda-graphs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--decode-cuda-graph-batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument(
        "--decode-cuda-graph-context-sizes",
        nargs="+",
        type=int,
        default=[128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument(
        "--decode-cuda-graph-warmup-iters",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--decode-cuda-graph-max-memory-bytes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=None,
        help="Startup watchdog timeout in seconds (disabled by default)",
    )
    parser.add_argument(
        "--decode-step-timeout",
        type=float,
        default=None,
        help="Decode step watchdog timeout in seconds (disabled by default)",
    )
    parser.add_argument(
        "--enable-pyspy-dump",
        action="store_true",
        help="Enable py-spy stack dump on watchdog timeout (requires py-spy installed)",
    )
    parser.add_argument(
        "--enable-contextpilot",
        action="store_true",
        default=False,
        help="Enable ContextPilot middleware",
    )
    parser.add_argument(
        "--contextpilot-debug",
        action="store_true",
        default=False,
        help="Enable CP debug endpoints (inject-fault)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _max_waiting_requests = max(0, int(args.max_waiting_requests))
    _max_n = max(1, int(args.max_n))
    _configure_auth(
        args.api_key or os.environ.get("MOE_API_KEYS"), args.rate_limit
    )
    with _contextpilot_state_lock:
        _contextpilot_enabled = _resolve_contextpilot_enabled(
            args.enable_contextpilot
        )
        _contextpilot_debug = bool(args.contextpilot_debug)
        _contextpilot_fault = "none"
    _startup_args = args

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
