# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from types import SimpleNamespace

from moe_infinity.runtime.model_offload import OffloadEngine
from moe_infinity.utils import ArcherConfig


class _RecordingHandle:
    def __init__(self) -> None:
        self.calls = []

    def configure_phase_policy(
        self,
        enabled,
        prefill_admission,
        decode_admission,
        prefill_weight,
        decode_weight,
        starvation_limit,
    ) -> None:
        self.calls.append(
            (
                enabled,
                prefill_admission,
                decode_admission,
                prefill_weight,
                decode_weight,
                starvation_limit,
            )
        )


class _RecordingDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def configure_phase_policy(self, *args) -> None:
        self.calls.append(args)


def _bare_engine(archer_config) -> OffloadEngine:
    engine = object.__new__(OffloadEngine)
    engine.archer_config = archer_config
    engine.archer_engine = _RecordingHandle()
    engine.expert_dispatcher = _RecordingDispatcher()
    engine.expert_prefetcher = SimpleNamespace(phase_policy=None)
    return engine


def _config(enabled, monkeypatch) -> ArcherConfig:
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    return ArcherConfig(
        offload_path="/tmp",
        use_native_engine=False,
        phase_specific_expert_policy=enabled,
    )


def test_enabled_configures_handle_not_dispatcher(monkeypatch) -> None:
    engine = _bare_engine(_config(True, monkeypatch))
    engine._configure_native_phase_policy()

    assert engine.archer_engine.calls == [
        (True, "transient_on_pressure", "cache", 1.0, 4.0, 8)
    ]
    assert engine.expert_dispatcher.calls == []
    assert engine.expert_prefetcher.phase_policy.enabled is True


def test_disabled_assigns_settings_but_does_not_configure(monkeypatch) -> None:
    engine = _bare_engine(_config(False, monkeypatch))
    engine._configure_native_phase_policy()

    assert engine.archer_engine.calls == []
    assert engine.expert_dispatcher.calls == []
    assert engine.expert_prefetcher.phase_policy.enabled is False
