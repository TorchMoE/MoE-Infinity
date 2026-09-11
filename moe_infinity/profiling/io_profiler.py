from __future__ import annotations

import atexit
import functools
import json
import os
import random
import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Optional, Protocol, cast


class _TraceableCallable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _ContextManagerLike(Protocol):
    def __enter__(self) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None: ...


class _NoOpContext:
    __slots__: tuple[()] = ()

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


class _TimingContext:
    __slots__: tuple[str, ...] = (
        "_emit_event",
        "_stage",
        "_layer",
        "_expert",
        "_bytes",
        "_ts_ns",
        "_start_ns",
    )

    _emit_event: Callable[[dict[str, object]], None]
    _stage: str
    _layer: int | None
    _expert: int | None
    _bytes: int
    _ts_ns: int
    _start_ns: int

    def __init__(
        self,
        emit_event: Callable[[dict[str, object]], None],
        stage: str,
        layer: int | None,
        expert: int | None,
        bytes: int,
    ):
        self._emit_event = emit_event
        self._stage = stage
        self._layer = layer
        self._expert = expert
        self._bytes = bytes
        self._ts_ns = 0
        self._start_ns = 0

    def __enter__(self) -> None:
        self._ts_ns = time.time_ns()
        self._start_ns = time.perf_counter_ns()
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        dur_ns = time.perf_counter_ns() - self._start_ns
        event: dict[str, object] = {
            "ts_ns": self._ts_ns,
            "stage": self._stage,
            "layer": self._layer,
            "expert": self._expert,
            "dur_ns": dur_ns,
            "bytes": self._bytes,
        }
        self._emit_event(event)
        return False


_NOOP_CONTEXT = _NoOpContext()


class IOProfiler:
    _instance: Optional["IOProfiler"] = None
    _instance_lock: threading.Lock = threading.Lock()

    _pid: int
    enabled: bool
    _out_path: str | None
    _sample: float
    _events: list[dict[str, object]]
    _lock: threading.Lock

    @classmethod
    def instance(cls) -> "IOProfiler":
        pid = os.getpid()
        with cls._instance_lock:
            if cls._instance is None or cls._instance._pid != pid:
                cls._instance = cls(pid=pid)
        return cls._instance

    def __init__(self, pid: int | None = None):
        self._pid = os.getpid() if pid is None else pid
        self.enabled = os.getenv("MOE_INFINITY_PROFILE_IO", "0") == "1"
        self._out_path = os.getenv("MOE_INFINITY_PROFILE_IO_OUT")
        self._sample = self._parse_sample(
            os.getenv("MOE_INFINITY_PROFILE_IO_SAMPLE", "1.0")
        )
        self._events = []
        self._lock = threading.Lock()

        if self.enabled:
            _ = atexit.register(self._flush_atexit)

    @staticmethod
    def _parse_sample(sample_raw: str) -> float:
        try:
            sample = float(sample_raw)
        except (TypeError, ValueError):
            return 1.0
        if sample < 0.0:
            return 0.0
        if sample > 1.0:
            return 1.0
        return sample

    def _emit_event(self, event: dict[str, object]) -> None:
        with self._lock:
            self._events.append(event)

    _RESERVED_EVENT_KEYS: tuple[str, ...] = (
        "ts_ns",
        "stage",
        "layer",
        "expert",
        "dur_ns",
        "bytes",
    )

    def record(
        self,
        stage: str,
        *,
        layer: int | None = None,
        expert: int | None = None,
        bytes: int = 0,
        fields: dict[str, object] | None = None,
    ) -> None:
        if not self.enabled:
            return

        event: dict[str, object] = {
            "ts_ns": time.time_ns(),
            "stage": stage,
            "layer": layer,
            "expert": expert,
            "dur_ns": 0,
            "bytes": bytes,
        }
        if fields:
            for key in fields:
                if key in self._RESERVED_EVENT_KEYS:
                    raise ValueError(
                        f"overlap-prefetch field {key!r} may not overwrite a reserved event key"
                    )
            event.update(fields)
        self._emit_event(event)

    def time(
        self,
        stage: str,
        *,
        layer: int | None = None,
        expert: int | None = None,
        bytes: int = 0,
    ) -> _ContextManagerLike:
        if not self.enabled:
            return _NOOP_CONTEXT

        if self._sample <= 0.0:
            return _NOOP_CONTEXT

        if self._sample < 1.0 and random.random() >= self._sample:
            return _NOOP_CONTEXT

        return _TimingContext(
            emit_event=self._emit_event,
            stage=stage,
            layer=layer,
            expert=expert,
            bytes=bytes,
        )

    def trace(
        self,
        stage: str,
        *,
        layer: int | None = None,
        expert: int | None = None,
        bytes: int = 0,
    ) -> Callable[[_TraceableCallable], _TraceableCallable]:
        def decorator(fn: _TraceableCallable) -> _TraceableCallable:
            @functools.wraps(fn)
            def wrapper(*args: object, **kwargs: object) -> object:
                with self.time(
                    stage,
                    layer=layer,
                    expert=expert,
                    bytes=bytes,
                ):
                    return fn(*args, **kwargs)

            return cast(_TraceableCallable, wrapper)

        return decorator

    def flush(self) -> None:
        if not self._out_path:
            return

        with self._lock:
            events = self._events
            if not events:
                return
            self._events = []

        with open(self._out_path, "a", encoding="utf-8") as f:
            for event in events:
                _ = f.write(json.dumps(event, ensure_ascii=False))
                _ = f.write("\n")

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def _flush_atexit(self) -> None:
        if self.enabled:
            self.flush()
