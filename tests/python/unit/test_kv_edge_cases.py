import importlib.machinery
import sys
import types
from collections import deque
from typing import cast

import pytest
import torch

from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.scheduler import Scheduler
from moe_infinity.serving.sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
    SequenceStatus,
)

if (
    "flash_attn" not in sys.modules
    or getattr(sys.modules["flash_attn"], "__spec__", None) is None
):
    flash_attn_stub = sys.modules.get(
        "flash_attn", types.ModuleType("flash_attn")
    )
    flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec(
        name="flash_attn", loader=None
    )
    sys.modules["flash_attn"] = flash_attn_stub

from moe_infinity.runtime.model_offload import OffloadEngine


def _make_kv_cache(
    num_blocks: int = 4,
    dtype: torch.dtype = torch.float16,
) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=dtype,
        device=torch.device("cpu"),
    )


def _make_scheduler(num_blocks: int = 4) -> Scheduler:
    return Scheduler(
        kv_cache=_make_kv_cache(num_blocks=num_blocks),
        max_batch_size=4,
        max_tokens_per_step=64,
    )


def test_partial_swap_failure_rollback() -> None:
    kv_cache = _make_kv_cache()
    seq_id = 7
    kv_cache.allocate_sequence(seq_id=seq_id, num_tokens=4)
    kv_cache.swap_out(seq_id)

    swapped_cpu_buffers = cast(
        dict[int, object],
        getattr(kv_cache, "_swapped_cpu_buffers"),
    )
    swapped_out_sequences = cast(
        set[int],
        getattr(kv_cache, "_swapped_out_sequences"),
    )

    class _FaultyCpuBuffer:
        def to(self, *args: object, **kwargs: object) -> torch.Tensor:
            _ = (args, kwargs)
            raise RuntimeError("simulated swap_in failure")

    swapped_cpu_buffers[seq_id] = _FaultyCpuBuffer()
    before_entries = dict(swapped_cpu_buffers)

    with pytest.raises(RuntimeError, match="simulated swap_in failure"):
        kv_cache.swap_in(seq_id)

    assert seq_id in before_entries
    assert seq_id not in swapped_cpu_buffers
    assert seq_id in swapped_out_sequences


def test_gpu_oom_on_swap_in_handled_gracefully(monkeypatch) -> None:
    scheduler = _make_scheduler(num_blocks=2)
    sequence = SequenceData(
        seq_id=42,
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(),
    )
    sequence.set_status(SequenceStatus.SWAPPED)
    group = SequenceGroup(request_id="oom-req", sequences=[sequence])

    swapped_queue = cast(deque[SequenceGroup], getattr(scheduler, "_swapped"))
    request_map = cast(
        dict[str, SequenceGroup], getattr(scheduler, "_request_map")
    )
    sequence_map = cast(
        dict[int, SequenceData], getattr(scheduler, "_sequence_map")
    )
    running_queue = cast(deque[SequenceGroup], getattr(scheduler, "_running"))

    swapped_queue.append(group)
    request_map[group.request_id] = group
    sequence_map[sequence.seq_id] = sequence

    def _raise_oom(_seq_id: int) -> None:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(scheduler.kv_cache, "swap_in", _raise_oom)

    scheduler._recover_swapped_groups([group])

    assert sequence.status is SequenceStatus.SWAPPED
    assert group in swapped_queue
    assert group not in running_queue


def test_dtype_preserved_through_swap_cycle() -> None:
    kv_cache = _make_kv_cache(dtype=torch.float16)
    seq_id = 3
    kv_cache.allocate_sequence(seq_id=seq_id, num_tokens=8)
    block_ids = kv_cache.get_block_table(seq_id)
    assert block_ids

    original = torch.randn(
        (
            kv_cache.num_layers,
            len(block_ids),
            2,
            kv_cache.block_size,
            kv_cache.num_heads,
            kv_cache.head_dim,
        ),
        dtype=torch.float16,
    )
    kv_tensor = kv_cache.get_kv_cache_tensors()
    kv_tensor[:, block_ids, ...] = original

    kv_cache.swap_out(seq_id)
    kv_tensor[:, block_ids, ...] = torch.zeros_like(original)
    kv_cache.swap_in(seq_id)

    restored = kv_cache.get_kv_cache_tensors()[:, block_ids, ...]
    assert restored.dtype == original.dtype
    assert torch.allclose(restored, original)


def test_swap_in_with_empty_cpu_cache_noop() -> None:
    kv_cache = _make_kv_cache()
    seq_id = 99
    kv_cache.allocate_sequence(seq_id=seq_id, num_tokens=4)

    swapped_cpu_buffers_before = dict(
        cast(dict[int, torch.Tensor], getattr(kv_cache, "_swapped_cpu_buffers"))
    )
    swapped_out_before = set(
        cast(set[int], getattr(kv_cache, "_swapped_out_sequences"))
    )

    kv_cache.swap_in(seq_id)

    swapped_cpu_buffers_after = cast(
        dict[int, torch.Tensor],
        getattr(kv_cache, "_swapped_cpu_buffers"),
    )
    swapped_out_after = cast(
        set[int],
        getattr(kv_cache, "_swapped_out_sequences"),
    )

    assert swapped_cpu_buffers_after == swapped_cpu_buffers_before
    assert swapped_out_after == swapped_out_before


def test_capture_kv_with_none_past_kv_values_noop() -> None:
    engine = OffloadEngine.__new__(OffloadEngine)
    setattr(engine, "_enable_kv_cache_offload", True)
    setattr(engine, "_captured_kv", {})

    OffloadEngine._capture_kv_cache(
        engine,
        seq_id=0,
        past_key_values=None,
    )

    assert getattr(engine, "_captured_kv") == {}


def test_ensure_sequence_capacity_grows_only_at_page_boundaries() -> None:
    cache = _make_kv_cache(num_blocks=4)

    cache.ensure_sequence_capacity(seq_id=9, total_tokens=3)
    assert cache.get_num_reserved_tokens(9) == 3
    assert cache.get_block_table(9) == [0]

    cache.ensure_sequence_capacity(seq_id=9, total_tokens=4)
    assert cache.get_block_table(9) == [0]
    cache.ensure_sequence_capacity(seq_id=9, total_tokens=5)
    assert cache.get_num_reserved_tokens(9) == 5
    assert cache.get_block_table(9) == [0, 1]


def test_ensure_sequence_capacity_is_idempotent_and_never_shrinks() -> None:
    cache = _make_kv_cache(num_blocks=4)
    cache.ensure_sequence_capacity(seq_id=10, total_tokens=5)
    free_after_first_reservation = cache.block_allocator.num_free_blocks

    cache.ensure_sequence_capacity(seq_id=10, total_tokens=5)
    assert cache.block_allocator.num_free_blocks == free_after_first_reservation
    with pytest.raises(ValueError, match="cannot shrink"):
        cache.ensure_sequence_capacity(seq_id=10, total_tokens=4)


def test_reservation_failure_leaves_no_partial_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _make_kv_cache(num_blocks=4)
    monkeypatch.setattr(
        cache.block_allocator,
        "allocate",
        lambda count: (_ for _ in ()).throw(RuntimeError("allocation failed")),
    )
    with pytest.raises(RuntimeError, match="allocation failed"):
        cache.ensure_sequence_capacity(seq_id=11, total_tokens=5)
    assert cache.has_sequence(11) is False
    assert cache.block_allocator.num_free_blocks == 4
