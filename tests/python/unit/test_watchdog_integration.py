# pyright: reportAny=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false, reportUnusedCallResult=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingParameterType=false
from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

try:
    import moe_infinity.serving.watchdog as watchdog_module

    MODULE_NAME = "moe_infinity.entrypoints.openai.api_server_v2"
    importlib.import_module(MODULE_NAME)
except TypeError:
    pytest.skip(
        "Pydantic v1 incompatible with Python 3.12+", allow_module_level=True
    )


class _FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        _ = prompt
        return [1, 2, 3]


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
        speculative_draft: object = None,
        decode_graph_capability_provider: object = None,
    ) -> None:
        self.model = model
        self.engine = engine
        self.config = config
        self.tokenizer = tokenizer
        self.speculative_draft = speculative_draft
        self.decode_graph_capability_provider = decode_graph_capability_provider


class _FakeMoE:
    last_config: dict[str, object] | None = None

    def __init__(self, model_name: str, config: dict[str, object]) -> None:
        _ = model_name
        type(self).last_config = config
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


class _LoopEngine:
    def __init__(
        self, pending_schedule: list[bool], shutdown_event: asyncio.Event
    ) -> None:
        self._pending_schedule = pending_schedule
        self._shutdown_event = shutdown_event
        self.step_calls = 0

    def has_pending_requests(self) -> bool:
        if self._pending_schedule:
            is_pending = self._pending_schedule.pop(0)
        else:
            is_pending = False
        if not self._pending_schedule and not is_pending:
            self._shutdown_event.set()
        return is_pending

    def step(self) -> list[Any]:
        self.step_calls += 1
        return []


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
        "_startup_watchdog": getattr(module, "_startup_watchdog", None),
        "_decode_watchdog": getattr(module, "_decode_watchdog", None),
        "_watchdog_config": getattr(module, "_watchdog_config", None),
        "_health_state": module._health_state,
    }


def _restore_runtime_state(module: Any, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(module, key, value)


def _startup_args(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "model": "unit-test-model",
        "offload_dir": "/tmp/offload",
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": 4,
        "enable_prefix_caching": False,
        "startup_timeout": None,
        "decode_step_timeout": None,
        "enable_pyspy_dump": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _patch_initialize_model_dependencies(monkeypatch: Any, module: Any) -> None:
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
        module.importlib, "import_module", _fake_import_module, raising=True
    )
    monkeypatch.setattr(
        module,
        "ContinuousBatchingEngine",
        _FakeRuntimeEngine,
        raising=True,
    )
    monkeypatch.setattr(
        module,
        "_ensure_engine_loop_running",
        lambda: None,
        raising=True,
    )


def test_no_watchdog_without_flags(monkeypatch: Any) -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)
    _patch_initialize_model_dependencies(monkeypatch, module)

    module.engine = None
    module.stream_manager = None
    module.tokenizer = None
    module.model_name_global = None
    module._startup_args = _startup_args(
        startup_timeout=None,
        decode_step_timeout=None,
    )

    try:
        with (
            patch.object(
                watchdog_module, "start_startup_watchdog"
            ) as startup_mock,
            patch.object(
                watchdog_module, "start_decode_watchdog"
            ) as decode_mock,
        ):
            asyncio.run(module._initialize_model())

        startup_mock.assert_not_called()
        decode_mock.assert_not_called()
        assert module._startup_watchdog is None
        assert module._decode_watchdog is None
    finally:
        _restore_runtime_state(module, original_state)


def test_watchdog_enabled_with_flags(monkeypatch: Any) -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)
    _patch_initialize_model_dependencies(monkeypatch, module)

    module.engine = None
    module.stream_manager = None
    module.tokenizer = None
    module.model_name_global = None
    module._startup_args = _startup_args(
        startup_timeout=12.0,
        decode_step_timeout=0.8,
        enable_pyspy_dump=True,
    )

    startup_watchdog = MagicMock()
    decode_watchdog = MagicMock()

    try:
        with (
            patch.object(
                watchdog_module,
                "start_startup_watchdog",
                return_value=startup_watchdog,
            ) as startup_mock,
            patch.object(
                watchdog_module,
                "start_decode_watchdog",
                return_value=decode_watchdog,
            ) as decode_mock,
        ):
            asyncio.run(module._initialize_model())

        startup_mock.assert_called_once()
        decode_mock.assert_called_once()
        startup_watchdog.cancel.assert_called_once()

        config = decode_mock.call_args.args[1]
        assert isinstance(config, watchdog_module.WatchdogConfig)
        assert config.startup_timeout == 12.0
        assert config.decode_step_timeout == 0.8
        assert config.enable_pyspy_dump is True

        assert module._decode_watchdog is decode_watchdog
    finally:
        _restore_runtime_state(module, original_state)


@pytest.mark.parametrize(
    ("mla_overrides", "expected_mla_config"),
    [
        (
            {},
            {
                "enable_deepseek_mla_paging": False,
                "max_resident_paged_speculative_sessions": 1,
                "min_free_mla_blocks_after_admission": 1,
            },
        ),
        (
            {
                "enable_deepseek_mla_paging": True,
                "max_resident_paged_speculative_sessions": 3,
                "min_free_mla_blocks_after_admission": 5,
            },
            {
                "enable_deepseek_mla_paging": True,
                "max_resident_paged_speculative_sessions": 3,
                "min_free_mla_blocks_after_admission": 5,
            },
        ),
    ],
)
def test_partial_namespace_starts_watchdog_and_preserves_mla_config(
    monkeypatch: Any,
    mla_overrides: dict[str, object],
    expected_mla_config: dict[str, object],
) -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)
    _patch_initialize_model_dependencies(monkeypatch, module)

    module.engine = None
    module.stream_manager = None
    module.tokenizer = None
    module.model_name_global = None
    module._startup_args = _startup_args(
        startup_timeout=12.0,
        decode_step_timeout=0.8,
        **mla_overrides,
    )
    _FakeMoE.last_config = None

    startup_watchdog = MagicMock()
    decode_watchdog = MagicMock()
    try:
        with (
            patch.object(
                watchdog_module,
                "start_startup_watchdog",
                return_value=startup_watchdog,
            ) as startup_mock,
            patch.object(
                watchdog_module,
                "start_decode_watchdog",
                return_value=decode_watchdog,
            ) as decode_mock,
        ):
            asyncio.run(module._initialize_model())

        startup_mock.assert_called_once()
        decode_mock.assert_called_once()
        startup_watchdog.cancel.assert_called_once()
        assert isinstance(module.engine, _FakeRuntimeEngine)
        assert _FakeMoE.last_config is not None
        assert {
            key: _FakeMoE.last_config[key] for key in expected_mla_config
        } == expected_mla_config
    finally:
        _restore_runtime_state(module, original_state)


def test_feed_called_in_engine_loop() -> None:
    module: Any = importlib.import_module(MODULE_NAME)
    original_state = _snapshot_runtime_state(module)

    async def _run() -> None:
        shutdown_event = asyncio.Event()
        fake_watchdog = MagicMock()
        fake_engine = _LoopEngine([True, False], shutdown_event)

        module._engine_shutdown_event = shutdown_event
        module.engine = fake_engine
        module._decode_watchdog = fake_watchdog

        await module._engine_loop()

        assert fake_engine.step_calls == 1
        fake_watchdog.activate.assert_called_once()
        fake_watchdog.feed.assert_called_once()
        fake_watchdog.deactivate.assert_called_once()

    try:
        asyncio.run(_run())
    finally:
        _restore_runtime_state(module, original_state)
