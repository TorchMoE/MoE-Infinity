from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from moe_infinity.memory.adaptive_memory import (
    MemoryTargets,
    ResizeDirection,
    ResizeOutcome,
)
from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.memory_resize import (
    ResizeReceipt,
    ServingMemoryResizer,
    TransactionalServingMemoryResizer,
)
from moe_infinity.serving.scheduler import Scheduler
from moe_infinity.serving.sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
)


class FakeEvent:
    def __init__(self, *, complete: bool = True) -> None:
        self._complete = complete

    def query(self) -> bool:
        return self._complete

    def complete(self) -> None:
        self._complete = True


def completed_receipt(
    *, device_id: int, post_publish_event: FakeEvent | None = None
) -> ResizeReceipt:
    event = FakeEvent()
    return ResizeReceipt(
        device_id=device_id,
        completion_events=(event,),
        post_publish_event=post_publish_event,
        admissions_paused=True,
    )


def make_cache(*, num_blocks: int) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )


def make_running_scheduler(
    *, num_blocks: int, prompt_tokens: int
) -> tuple[PagedKVCache, Scheduler]:
    cache = make_cache(num_blocks=num_blocks)
    scheduler = Scheduler(cache, max_batch_size=8, max_tokens_per_step=128)
    sequence = SequenceData(
        seq_id=1,
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(),
    )
    scheduler.add_request(SequenceGroup(request_id="r1", sequences=[sequence]))
    scheduler.schedule()
    return cache, scheduler


@dataclass
class FakeExpertPool:
    device_id: int
    resident_bytes: int
    limit_bytes: int = field(init=False)
    evicted_experts_are_resident: bool = True

    def __post_init__(self) -> None:
        self.limit_bytes = self.resident_bytes

    def reserve_victims(self, device_id: int, target_bytes: int) -> object:
        assert device_id == self.device_id
        return SimpleNamespace(ready=True, target_bytes=target_bytes)

    def commit_reserved_victims(self, reservation: object) -> int:
        self.limit_bytes = int(reservation.target_bytes)
        self.resident_bytes = self.limit_bytes
        self.evicted_experts_are_resident = False
        return self.resident_bytes

    def cancel_reservation(self, reservation: object) -> None:
        _ = reservation


class FakeFlashinferWrapper:
    def __init__(self, workspace: torch.Tensor, layout: str) -> None:
        self.workspace = workspace
        self.layout = layout
        self.plan_calls: list[SimpleNamespace] = []
        self.released = False

    def plan(self, *args: object, **kwargs: object) -> None:
        page_indices = kwargs.get(
            "page_indices", args[2] if len(args) == 8 else args[1]
        )
        values = (
            page_indices.tolist()
            if isinstance(page_indices, torch.Tensor)
            else page_indices
        )
        maximum = max(values, default=-1)
        self.plan_calls.append(SimpleNamespace(max_page_index=maximum))

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        _ = kv_cache
        return query


def make_flashinfer_cache(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_blocks: int,
    next_prefill_plan_error: Exception | None = None,
) -> PagedKVCache:
    class Prefill(FakeFlashinferWrapper):
        def plan(self, *args: object, **kwargs: object) -> None:
            nonlocal next_prefill_plan_error
            if next_prefill_plan_error is not None:
                error, next_prefill_plan_error = next_prefill_plan_error, None
                raise error
            super().plan(*args, **kwargs)

    module = SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=Prefill,
        BatchDecodeWithPagedKVCacheWrapper=FakeFlashinferWrapper,
    )
    from moe_infinity.runtime import flashinfer_utils

    monkeypatch.setattr(flashinfer_utils, "HAS_FLASHINFER", True)
    monkeypatch.setattr(
        flashinfer_utils, "get_flashinfer_module", lambda: module
    )
    monkeypatch.setattr(
        flashinfer_utils,
        "get_workspace",
        lambda device: torch.empty(1, device=device),
    )
    return make_cache(num_blocks=num_blocks)


def prefill_inputs(
    *, num_pages: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.zeros((1, 2, 2, 8), dtype=torch.float16)
    key = torch.zeros((1, 2, num_pages * 4, 8), dtype=torch.float16)
    return query, key, key.clone()


def decode_inputs(
    *, num_pages: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.zeros((1, 2, 1, 8), dtype=torch.float16)
    key = torch.zeros((1, 2, num_pages * 4, 8), dtype=torch.float16)
    return query, key, key.clone()


def test_resize_rejects_live_block_tables() -> None:
    cache = make_cache(num_blocks=8)
    cache.allocate_sequence(1, num_tokens=4)
    with pytest.raises(RuntimeError, match="referenced KV blocks"):
        cache.resize_physical_num_blocks(4, completed_receipt(device_id=0))


def test_scheduler_drain_resize_restore_preserves_tokens() -> None:
    cache, scheduler = make_running_scheduler(num_blocks=8, prompt_tokens=4)
    receipt = scheduler.quiesce_for_kv_resize()
    assert scheduler.admissions_paused is True
    assert cache.block_allocator.num_free_blocks == 8
    assert all(event.query() for event in receipt.completion_events)
    cache.resize_physical_num_blocks(6, receipt)
    scheduler.restore_after_kv_resize(receipt)
    assert scheduler.admissions_paused is False
    assert scheduler.get_running_seq_ids() == [1]
    assert cache._require_sequence(1).num_computed_tokens() == 4


def test_receiver_growth_is_not_called_when_donor_shrink_fails() -> None:
    expert = Mock(shrink_to=Mock(return_value=False))
    kv = Mock()
    result = ServingMemoryResizer(
        expert, kv, reserve_probe=lambda _: 2**40
    ).apply(
        0,
        MemoryTargets(0, 512, 8, ResizeDirection.EXPERT_TO_KV, "kv_pressure"),
        current_expert_bytes=1024,
        current_kv_blocks=4,
        kv_block_bytes=64,
    )
    assert result.device_id == 0
    assert result.outcome is ResizeOutcome.REJECTED
    kv.resize_physical_num_blocks.assert_not_called()


def test_old_storage_is_retained_until_cuda_completion_event(
    monkeypatch,
) -> None:
    cache, scheduler = make_running_scheduler(num_blocks=8, prompt_tokens=4)
    event = FakeEvent(complete=False)
    monkeypatch.setattr(
        scheduler, "_record_resize_completion_event", lambda: event
    )
    with pytest.raises(TimeoutError, match="CUDA completion"):
        scheduler.quiesce_for_kv_resize(timeout_s=0.01)
    assert cache.num_blocks == 8
    assert scheduler.admissions_paused is False


def test_quiesce_failure_reopens_admissions_and_restores_queues() -> None:
    cache, scheduler = make_running_scheduler(num_blocks=8, prompt_tokens=4)
    scheduler.inject_swap_failure_after(1)
    before = scheduler.snapshot_queue_ids()
    with pytest.raises(RuntimeError, match="swap drain failed"):
        scheduler.quiesce_for_kv_resize()
    assert scheduler.snapshot_queue_ids() == before
    assert scheduler.admissions_paused is False


def test_expert_eviction_then_kv_growth_failure_reports_partial_commit() -> (
    None
):
    expert = FakeExpertPool(device_id=0, resident_bytes=1024)
    kv = Mock(
        resize_physical_num_blocks=Mock(side_effect=torch.OutOfMemoryError())
    )
    resizer = ServingMemoryResizer(expert, kv, reserve_probe=lambda _: 2**40)
    result = resizer.apply(
        0,
        MemoryTargets(0, 512, 8, ResizeDirection.EXPERT_TO_KV, "kv_pressure"),
        current_expert_bytes=1024,
        current_kv_blocks=4,
        kv_block_bytes=64,
    )
    assert result.outcome is ResizeOutcome.PARTIAL_DONOR_COMMITTED
    assert (result.expert_bytes, result.kv_blocks) == (512, 4)
    assert expert.limit_bytes == 512
    assert expert.evicted_experts_are_resident is False


def test_serving_flashinfer_wrappers_rebuild_independently_and_old_bundle_lives(
    monkeypatch,
) -> None:
    cache = make_flashinfer_cache(monkeypatch, num_blocks=8)
    old_store = cache._kv_cache
    old_prefill = cache._fi_prefill
    old_decode = cache._fi_decode
    assert old_prefill is not old_decode
    post_publish = FakeEvent(complete=False)
    receipt = completed_receipt(device_id=0, post_publish_event=post_publish)
    cache.resize_physical_num_blocks(4, receipt)
    assert cache._kv_cache.shape[1] == 4
    assert cache._fi_prefill is not old_prefill
    assert cache._fi_decode is not old_decode
    assert cache._fi_prefill is not cache._fi_decode
    assert receipt.retained_objects[0] is old_store
    assert receipt.retained_objects[1] is old_prefill
    assert receipt.retained_objects[2] is old_decode
    assert old_prefill.released is False and old_decode.released is False
    post_publish.complete()
    receipt.release_retained_objects()
    cache._compute_attention(*prefill_inputs(num_pages=4))
    cache._compute_attention(*decode_inputs(num_pages=4))
    assert cache._fi_prefill.plan_calls[-1].max_page_index < 4
    assert cache._fi_decode.plan_calls[-1].max_page_index < 4


def test_serving_first_replan_failure_restores_complete_old_bundle(
    monkeypatch,
) -> None:
    cache = make_flashinfer_cache(
        monkeypatch,
        num_blocks=8,
        next_prefill_plan_error=RuntimeError("stale page plan"),
    )
    old = (
        cache._kv_cache,
        cache.block_allocator,
        cache._fi_prefill,
        cache._fi_decode,
        cache.num_blocks,
    )
    receipt = completed_receipt(device_id=0)
    with pytest.raises(RuntimeError, match="stale page plan"):
        cache.resize_physical_num_blocks(4, receipt)
        cache._compute_attention(*prefill_inputs(num_pages=4))
    assert cache._kv_cache is old[0]
    assert cache.block_allocator is old[1]
    assert cache._fi_prefill is old[2]
    assert cache._fi_decode is old[3]
    assert cache.num_blocks == old[4]
    assert cache._fi_prefill.plan_calls[-1].max_page_index < 8
    assert cache._fi_decode.plan_calls[-1].max_page_index < 8
    assert receipt.admissions_paused is False


def test_transactional_resizer_quiesces_and_publishes_effective_targets() -> (
    None
):
    receipt = completed_receipt(device_id=0)
    scheduler = Mock(
        quiesce_for_kv_resize=Mock(return_value=receipt),
        restore_after_kv_resize=Mock(),
    )
    expert = Mock(resize_cache=Mock(return_value={"resident_bytes": 512}))
    kv = Mock(num_blocks=4, resize_physical_num_blocks=Mock())
    resizer = TransactionalServingMemoryResizer(
        device_id=0,
        scheduler=scheduler,
        expert_cache=expert,
        kv_cache=kv,
        reserve_probe=lambda _: 2**40,
        free_reserve_bytes=128,
    )
    result = resizer.apply(
        0,
        MemoryTargets(0, 512, 8, ResizeDirection.EXPERT_TO_KV, "kv_pressure"),
        current_expert_bytes=1024,
        current_kv_blocks=4,
        kv_block_bytes=64,
    )
    assert result.outcome is ResizeOutcome.COMMITTED
    expert.resize_cache.assert_called_once_with(0, 512)
    kv.resize_physical_num_blocks.assert_called_once_with(8, receipt)
    scheduler.restore_after_kv_resize.assert_called_once_with(receipt)
