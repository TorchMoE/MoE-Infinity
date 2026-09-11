# pyright: reportAny=false

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

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


def _load_module(module_name: str, file_path: Path):
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

_SEQUENCE_MODULE = _load_module(
    "moe_infinity.serving.sequence",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
_KV_CACHE_MODULE = _load_module(
    "moe_infinity.serving.kv_cache",
    ROOT / "moe_infinity" / "serving" / "kv_cache.py",
)
_ = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)
_SCHEDULER_MODULE = _load_module(
    "moe_infinity.serving.scheduler",
    ROOT / "moe_infinity" / "serving" / "scheduler.py",
)

Deficit2D = _SCHEDULER_MODULE.Deficit2D
VerifyDemand = _SCHEDULER_MODULE.VerifyDemand
admit_verify_demands = _SCHEDULER_MODULE.admit_verify_demands
Scheduler = _SCHEDULER_MODULE.Scheduler
PagedKVCache = _KV_CACHE_MODULE.PagedKVCache
SamplingParams = _SEQUENCE_MODULE.SamplingParams
SequenceData = _SEQUENCE_MODULE.SequenceData
SequenceGroup = _SEQUENCE_MODULE.SequenceGroup


def _make_cache(*, num_blocks: int = 8) -> object:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )


def _make_group(request_id: str, seq_id: int, prompt_len: int) -> object:
    sequence = SequenceData(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(),
    )
    return SequenceGroup(request_id=request_id, sequences=[sequence])


def _verify_scheduler() -> object:
    return Scheduler(
        _make_cache(),
        max_batch_size=8,
        max_tokens_per_step=128,
        verify_token_budget=64,
        verify_expert_byte_budget=200,
        verify_token_deficit_cap=64,
        verify_expert_byte_deficit_cap=360,
    )


def test_inflight_is_seated_before_new_rounds() -> None:
    demands = [
        VerifyDemand(1, tokens=16, expert_bytes=80, in_flight=False),
        VerifyDemand(2, tokens=16, expert_bytes=80, in_flight=True),
    ]
    out = admit_verify_demands(
        demands,
        budget=Deficit2D(tokens=16, expert_bytes=80),
        carried=Deficit2D(tokens=0, expert_bytes=0),
        deficit_cap=Deficit2D(tokens=32, expert_bytes=160),
    )
    assert out.seated_ids == (2,)
    assert out.admitted_ids == ()


def test_both_dimensions_must_fit_and_unused_budget_carries() -> None:
    demands = [VerifyDemand(1, tokens=8, expert_bytes=90, in_flight=False)]
    out = admit_verify_demands(
        demands,
        budget=Deficit2D(tokens=16, expert_bytes=80),
        carried=Deficit2D(tokens=0, expert_bytes=0),
        deficit_cap=Deficit2D(tokens=32, expert_bytes=160),
    )
    assert out.admitted_ids == ()
    assert out.carried == Deficit2D(tokens=16, expert_bytes=80)


def test_carried_deficit_is_capped() -> None:
    out = admit_verify_demands(
        [], Deficit2D(16, 80), Deficit2D(30, 150), Deficit2D(32, 160)
    )
    assert out.carried == Deficit2D(32, 160)


def test_inflight_seated_then_remaining_budget_admits_new() -> None:
    demands = [
        VerifyDemand(1, tokens=8, expert_bytes=40, in_flight=True),
        VerifyDemand(2, tokens=8, expert_bytes=40, in_flight=False),
    ]
    out = admit_verify_demands(
        demands, Deficit2D(16, 80), Deficit2D(0, 0), Deficit2D(32, 160)
    )
    assert out.seated_ids == (1,)
    assert out.admitted_ids == (2,)
    assert out.carried == Deficit2D(0, 0)


def test_new_rounds_are_admitted_fcfs_until_pool_exhausts() -> None:
    demands = [
        VerifyDemand(1, tokens=8, expert_bytes=40, in_flight=False),
        VerifyDemand(2, tokens=8, expert_bytes=40, in_flight=False),
        VerifyDemand(3, tokens=8, expert_bytes=40, in_flight=False),
    ]
    out = admit_verify_demands(
        demands, Deficit2D(16, 80), Deficit2D(0, 0), Deficit2D(32, 160)
    )
    assert out.seated_ids == ()
    assert out.admitted_ids == (1, 2)
    assert out.carried == Deficit2D(0, 0)


def test_large_demand_admits_once_carry_accumulates() -> None:
    demand = VerifyDemand(1, tokens=24, expert_bytes=120, in_flight=False)
    budget = Deficit2D(16, 80)
    cap = Deficit2D(32, 160)

    first = admit_verify_demands([demand], budget, Deficit2D(0, 0), cap)
    assert first.admitted_ids == ()
    assert first.carried == Deficit2D(16, 80)

    second = admit_verify_demands([demand], budget, first.carried, cap)
    assert second.admitted_ids == (1,)
    assert second.carried == Deficit2D(8, 40)


def test_seating_beyond_quantum_floors_carry_at_zero() -> None:
    demands = [VerifyDemand(1, tokens=40, expert_bytes=200, in_flight=True)]
    out = admit_verify_demands(
        demands, Deficit2D(16, 80), Deficit2D(0, 0), Deficit2D(64, 320)
    )
    assert out.seated_ids == (1,)
    assert out.admitted_ids == ()
    assert out.carried == Deficit2D(0, 0)


def test_negative_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError):
        admit_verify_demands(
            [], Deficit2D(-1, 0), Deficit2D(0, 0), Deficit2D(32, 160)
        )
    with pytest.raises(ValueError):
        admit_verify_demands(
            [VerifyDemand(1, tokens=-1, expert_bytes=0)],
            Deficit2D(16, 80),
            Deficit2D(0, 0),
            Deficit2D(32, 160),
        )


def test_cap_must_cover_largest_single_demand_in_each_dimension() -> None:
    with pytest.raises(ValueError, match="largest single demand"):
        admit_verify_demands(
            [VerifyDemand(1, tokens=40, expert_bytes=10)],
            Deficit2D(16, 80),
            Deficit2D(0, 0),
            Deficit2D(32, 160),
        )
    with pytest.raises(ValueError, match="largest single demand"):
        admit_verify_demands(
            [VerifyDemand(1, tokens=10, expert_bytes=200)],
            Deficit2D(16, 80),
            Deficit2D(0, 0),
            Deficit2D(32, 160),
        )


def test_admission_is_side_effect_free() -> None:
    demands = [
        VerifyDemand(1, tokens=8, expert_bytes=40, in_flight=False),
        VerifyDemand(2, tokens=8, expert_bytes=40, in_flight=True),
    ]
    budget = Deficit2D(16, 80)
    carried = Deficit2D(0, 0)
    cap = Deficit2D(32, 160)

    first = admit_verify_demands(demands, budget, carried, cap)
    second = admit_verify_demands(demands, budget, carried, cap)

    assert first == second
    assert demands[0] == VerifyDemand(1, tokens=8, expert_bytes=40)
    assert budget == Deficit2D(16, 80)
    assert carried == Deficit2D(0, 0)


def test_scheduler_admits_verify_demands_by_tokens_and_bytes() -> None:
    scheduler = _verify_scheduler()
    scheduler.set_verify_demand(1, tokens=16, expert_bytes=80, in_flight=True)
    scheduler.set_verify_demand(2, tokens=16, expert_bytes=80, in_flight=False)
    scheduler.set_verify_demand(3, tokens=16, expert_bytes=180, in_flight=False)

    output = scheduler.schedule()

    assert output.verify_seq_ids == [1, 2]
    assert output.draft_seq_ids == [3]
    assert output.num_verify_tokens == 32
    assert output.num_verify_expert_bytes == 160
    assert scheduler.carried_verify_deficit == Deficit2D(32, 40)


def test_scheduler_verify_is_disabled_without_budgets() -> None:
    scheduler = Scheduler(
        _make_cache(), max_batch_size=8, max_tokens_per_step=128
    )
    scheduler.set_verify_demand(1, tokens=16, expert_bytes=80, in_flight=True)

    output = scheduler.schedule()

    assert output.verify_seq_ids == []
    assert output.draft_seq_ids == []
    assert output.num_verify_tokens == 0
    assert output.num_verify_expert_bytes == 0


def test_scheduler_clear_verify_demand_removes_it() -> None:
    scheduler = _verify_scheduler()
    scheduler.set_verify_demand(1, tokens=16, expert_bytes=80, in_flight=True)
    scheduler.set_verify_demand(2, tokens=16, expert_bytes=80, in_flight=False)
    scheduler.clear_verify_demand(2)

    output = scheduler.schedule()

    assert output.verify_seq_ids == [1]
    assert output.draft_seq_ids == []


def test_ordinary_prefill_output_has_empty_verify_lists() -> None:
    scheduler = _verify_scheduler()
    scheduler.add_request(_make_group("req-1", 1, 3))

    output = scheduler.schedule()

    assert output.prefill_seq_ids == [1]
    assert output.draft_seq_ids == []
    assert output.verify_seq_ids == []
    assert output.num_verify_tokens == 0
    assert output.num_verify_expert_bytes == 0
