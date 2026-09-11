"""Engine Task 6 Step 5: per-round DRAFT->VERIFY->DRAFT admission wiring.

Drives ``ContinuousBatchingEngine`` through the opt-in single-round
``SpecSession`` seam with a fake session speculator and the REAL 2-D verify
scheduler, asserting:

* opt-in: with verify budgets configured, the engine registers each pending
  verify's EXACT projected ``expert_bytes`` via ``set_verify_demand`` and runs
  the verify only after the scheduler admits it;
* default unchanged: without budgets (or with a non-session speculator) the
  delegated request keeps the whole-request ``generate()`` path;
* liveness: a demand larger than a single-quantum budget still completes as the
  carried deficit accrues (unadmitted rounds stay DRAFT, deficit carries).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_package(name: str, path: Path) -> None:
    if sys.modules.get(name) is None:
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
for _name in (
    "sequence",
    "kv_cache",
    "batch",
    "memory_manager",
    "scheduler",
    "model_runner",
    "sampler",
    "engine",
):
    _ = _load_module(
        f"moe_infinity.serving.{_name}",
        ROOT / "moe_infinity" / "serving" / f"{_name}.py",
    )

from moe_infinity.serving.engine import (  # type: ignore[reportMissingImports]  # noqa: E402
    ContinuousBatchingEngine,
    RequestOutput,
)
from moe_infinity.serving.sequence import (  # type: ignore[reportMissingImports]  # noqa: E402
    SamplingParams,
)

BLOCK_TOKENS = 4
DEMAND_BYTES = 320


class _MockConfig:
    def __init__(self) -> None:
        self.vocab_size = 100
        self.eos_token_id = 2


class MockModel:
    def __init__(self) -> None:
        self.config = _MockConfig()

    def eval(self) -> None:
        pass

    def forward(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> types.SimpleNamespace:
        _ = kwargs
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.config.vocab_size)
        return types.SimpleNamespace(logits=logits)


class _Tracer:
    def __init__(self) -> None:
        self._next = 0

    def create_entry(self) -> int:
        entry = self._next
        self._next += 1
        return entry


class MockOffloadEngine:
    def __init__(self) -> None:
        self.request_id = 0
        self.expert_tracer = _Tracer()
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]

    def _generate_request_id(self) -> int:
        rid = self.request_id
        self.request_id += 1
        return rid


class _FakeSession:
    def __init__(self, anchor: int, max_new_tokens: int) -> None:
        self.emitted = [anchor]
        self.max_new_tokens = max_new_tokens
        self.finished = max_new_tokens <= 1
        self._anchor = anchor


class _FakeDraft:
    def __init__(self, tokens: int, expert_bytes: int) -> None:
        self.tokens = tokens
        self.expert_bytes = expert_bytes
        self.expert_union = frozenset({(0, 1), (0, 3)})


class FakeSessionSpeculator:
    """Session-seam double: deterministic token stream + fixed byte demand."""

    def __init__(self, expert_bytes: int = DEMAND_BYTES) -> None:
        self.moe = types.SimpleNamespace(_cached_past_key_values=None)
        self.expert_bytes = expert_bytes
        self.draft_rounds = 0
        self.verify_rounds = 0

    def begin_session(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        collect_route_union: bool = False,
    ) -> _FakeSession:
        _ = (temperature, stop_token_ids, top_k, top_p, collect_route_union)
        anchor = int(input_ids[0, -1].item()) + 1
        return _FakeSession(anchor, int(max_new_tokens))

    def draft_round(self, session: _FakeSession) -> _FakeDraft:
        self.draft_rounds += 1
        return _FakeDraft(tokens=BLOCK_TOKENS, expert_bytes=self.expert_bytes)

    def verify_round(self, session: _FakeSession) -> object:
        self.verify_rounds += 1
        session.emitted.append(session.emitted[-1] + 1)
        if len(session.emitted) >= session.max_new_tokens:
            session.finished = True
        return types.SimpleNamespace(finished=session.finished)


class WholeRequestSpeculator:
    def __init__(self) -> None:
        self.moe = types.SimpleNamespace(_cached_past_key_values=None)
        self.calls = 0

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        _ = (temperature, stop_token_ids, top_k, top_p)
        self.calls += 1
        last = int(input_ids[0, -1].item())
        cont = torch.arange(
            last + 1, last + max_new_tokens + 1, dtype=input_ids.dtype
        ).unsqueeze(0)
        return torch.cat([input_ids, cont], dim=1)


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": 8,
        "max_tokens_per_step": 64,
        "block_size": 4,
        "num_layers": 1,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype": "float32",
        "eos_token_id": 2,
    }
    config.update(overrides)
    return config


def _verify_budgets(
    *, token_budget: int = 16, byte_budget: int = 4096
) -> dict[str, object]:
    return {
        "verify_token_budget": token_budget,
        "verify_expert_byte_budget": byte_budget,
        "verify_token_deficit_cap": 64,
        "verify_expert_byte_deficit_cap": 8192,
    }


def _engine(speculator: object, config: dict[str, object]):
    return ContinuousBatchingEngine(
        model=MockModel(),
        engine=MockOffloadEngine(),
        config=config,
        speculative_draft=speculator,
    )


def _record_demands(
    engine: ContinuousBatchingEngine,
) -> list[dict[str, object]]:
    recorded: list[dict[str, object]] = []
    original = engine.scheduler.set_verify_demand

    def spy(seq_id, tokens, expert_bytes, in_flight):  # type: ignore[no-untyped-def]
        recorded.append(
            {
                "seq_id": seq_id,
                "tokens": tokens,
                "expert_bytes": expert_bytes,
                "in_flight": in_flight,
            }
        )
        return original(seq_id, tokens, expert_bytes, in_flight)

    engine.scheduler.set_verify_demand = spy  # type: ignore[assignment]
    return recorded


def test_engine_drives_session_rounds_with_exact_byte_demand() -> None:
    speculator = FakeSessionSpeculator(expert_bytes=DEMAND_BYTES)
    engine = _engine(speculator, {**_config(), **_verify_budgets()})
    assert engine._verify_scheduling_enabled is True
    recorded = _record_demands(engine)

    engine.add_request(
        request_id="req-spec",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
    )
    outputs = engine.step()

    assert [o.token_id for o in outputs] == [11, 12, 13, 14]
    assert outputs[-1].finished is True
    assert speculator.draft_rounds == speculator.verify_rounds
    assert speculator.verify_rounds == 3
    # every registered demand carried the EXACT projected byte sum, never a
    # fabricated expert-count estimate.
    assert recorded and all(
        d["tokens"] == BLOCK_TOKENS
        and d["expert_bytes"] == DEMAND_BYTES
        and d["in_flight"] is False
        for d in recorded
    )
    assert engine.has_pending_requests() is False


def test_engine_without_budgets_uses_whole_request_path() -> None:
    speculator = WholeRequestSpeculator()
    engine = _engine(speculator, _config())
    assert engine._verify_scheduling_enabled is False

    engine.add_request(
        request_id="req-plain",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
    )
    outputs = engine.step()

    assert speculator.calls == 1
    assert [o.token_id for o in outputs] == [11, 12, 13]


def test_engine_non_session_speculator_falls_back_even_with_budgets() -> None:
    speculator = WholeRequestSpeculator()
    engine = _engine(speculator, {**_config(), **_verify_budgets()})
    assert engine._verify_scheduling_enabled is True
    assert engine._can_drive_verify_rounds() is False

    engine.add_request(
        request_id="req-plain",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=3),
    )
    outputs = engine.step()

    assert speculator.calls == 1
    assert [o.token_id for o in outputs] == [11, 12, 13]


def test_engine_completes_under_tight_token_budget_via_carried_deficit() -> (
    None
):
    speculator = FakeSessionSpeculator(expert_bytes=DEMAND_BYTES)
    engine = _engine(
        speculator,
        {
            **_config(),
            **_verify_budgets(token_budget=2, byte_budget=DEMAND_BYTES),
        },
    )

    engine.add_request(
        request_id="req-tight",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
    )
    outputs = engine.step()

    # BLOCK_TOKENS (4) exceeds the per-quantum token budget (2); each verify is
    # admitted only after the carried deficit accrues, yet the request still
    # completes with the full token stream.
    assert [o.token_id for o in outputs] == [11, 12, 13, 14]
    assert outputs[-1].finished is True


def test_admission_raises_when_budget_cannot_cover_a_single_verify() -> None:
    speculator = FakeSessionSpeculator(expert_bytes=DEMAND_BYTES)
    engine = _engine(
        speculator,
        {
            **_config(),
            **_verify_budgets(token_budget=0, byte_budget=DEMAND_BYTES),
        },
    )

    engine.add_request(
        request_id="req-bad",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
    )
    with pytest.raises(RuntimeError, match="never admitted"):
        engine.step()
