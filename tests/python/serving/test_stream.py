import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Optional, TypedDict, cast

import pytest

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
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")

_STREAM_MODULE = _load_module(
    "moe_infinity.serving.stream",
    ROOT / "moe_infinity" / "serving" / "stream.py",
)

StreamManager = _STREAM_MODULE.StreamManager
format_done_event = _STREAM_MODULE.format_done_event
format_sse_event = _STREAM_MODULE.format_sse_event


class _ChoicePayload(TypedDict):
    index: int
    delta: dict[str, str]
    finish_reason: Optional[str]


class _StreamPayload(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[_ChoicePayload]


def _parse_event_payload(event: str) -> _StreamPayload:
    prefix = "data: "
    assert event.startswith(prefix)
    assert event.endswith("\n\n")
    payload = event[len(prefix) : -2]
    return cast(_StreamPayload, json.loads(payload))


def test_format_sse_event() -> None:
    assert format_sse_event('{"x":1}') == 'data: {"x":1}\n\n'


def test_format_done_event() -> None:
    assert format_done_event() == "data: [DONE]\n\n"


def test_stream_manager_push_and_yield() -> None:
    manager = StreamManager()
    stream = manager.create_stream(request_id="req-1", model="demo-model")

    manager.push_token(request_id="req-1", token_text="Hello", finished=False)
    payload = _parse_event_payload(next(stream))

    assert payload["id"].startswith("chatcmpl-")
    assert payload["object"] == "chat.completion.chunk"
    assert payload["model"] == "demo-model"
    choices = payload["choices"]
    assert isinstance(choices, list)
    assert len(choices) == 1
    choice = choices[0]
    assert choice["index"] == 0
    assert choice["delta"] == {"content": "Hello"}
    assert choice["finish_reason"] is None


def test_stream_manager_finish() -> None:
    manager = StreamManager()
    stream = manager.create_stream(request_id="req-finish", model="demo-model")

    manager.push_token(request_id="req-finish", token_text="", finished=True)

    payload = _parse_event_payload(next(stream))
    choice = payload["choices"][0]
    assert choice["delta"] == {}
    assert choice["finish_reason"] == "stop"
    assert next(stream) == format_done_event()

    with pytest.raises(StopIteration):
        next(stream)


def test_multiple_streams_independent() -> None:
    manager = StreamManager()
    stream_a = manager.create_stream(request_id="req-a", model="model-a")
    stream_b = manager.create_stream(request_id="req-b", model="model-b")

    manager.push_token(request_id="req-b", token_text="B", finished=False)
    manager.push_token(request_id="req-a", token_text="A", finished=False)

    payload_b = _parse_event_payload(next(stream_b))
    payload_a = _parse_event_payload(next(stream_a))
    assert payload_b["model"] == "model-b"
    assert payload_b["choices"][0]["delta"] == {"content": "B"}
    assert payload_a["model"] == "model-a"
    assert payload_a["choices"][0]["delta"] == {"content": "A"}

    manager.push_token(request_id="req-a", token_text="", finished=True)
    manager.push_token(request_id="req-b", token_text="", finished=True)

    _ = next(stream_a)
    assert next(stream_a) == format_done_event()
    _ = next(stream_b)
    assert next(stream_b) == format_done_event()
