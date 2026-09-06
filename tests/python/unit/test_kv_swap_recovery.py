from typing import cast

import torch

from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.scheduler import Scheduler
from moe_infinity.serving.sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
    SequenceStatus,
)


def _make_kv_cache(num_blocks: int) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )


def _make_scheduler(num_blocks: int) -> Scheduler:
    return Scheduler(
        kv_cache=_make_kv_cache(num_blocks),
        max_batch_size=4,
        max_tokens_per_step=64,
    )


def _make_group(
    request_id: str, seq_id: int, prompt_len: int = 8
) -> SequenceGroup:
    return SequenceGroup(
        request_id=request_id,
        sequences=[
            SequenceData(
                seq_id=seq_id,
                prompt_token_ids=list(range(prompt_len)),
                sampling_params=SamplingParams(),
            )
        ],
    )


def test_swap_recovery_lifecycle() -> None:
    scheduler = _make_scheduler(num_blocks=4)
    group1 = _make_group("req-1", seq_id=1, prompt_len=4)
    scheduler.add_request(group1)

    first = scheduler.schedule()
    assert first.prefill_seq_ids == [1]
    scheduler.update_after_step(completed_seq_ids=[], new_decode_seq_ids=[1])
    assert group1.sequences[0].status is SequenceStatus.DECODE

    group2 = _make_group("req-2", seq_id=2, prompt_len=12)
    scheduler.add_request(group2)

    preempt_cycle = scheduler.schedule()
    assert 1 in preempt_cycle.preempted_seq_ids
    assert group1.sequences[0].status is SequenceStatus.SWAPPED

    scheduler.update_after_step(completed_seq_ids=[2], new_decode_seq_ids=[])

    recovery_cycle = scheduler.schedule()
    assert group1.sequences[0].status is SequenceStatus.DECODE
    assert 1 in recovery_cycle.decode_seq_ids


def test_no_recovery_when_gpu_full() -> None:
    scheduler = _make_scheduler(num_blocks=2)
    group1 = _make_group("req-1", seq_id=10)
    scheduler.add_request(group1)

    first = scheduler.schedule()
    assert first.prefill_seq_ids == [10]
    assert group1.sequences[0].status is SequenceStatus.PREFILL

    group2 = _make_group("req-2", seq_id=20)
    scheduler.add_request(group2)

    preempt_cycle = scheduler.schedule()
    assert 10 in preempt_cycle.preempted_seq_ids
    assert group1.sequences[0].status is SequenceStatus.SWAPPED

    no_recovery_cycle = scheduler.schedule()
    assert no_recovery_cycle.decode_seq_ids == []
    assert group1.sequences[0].status is SequenceStatus.SWAPPED


def test_free_gpu_blocks_preserves_cpu_buffer() -> None:
    kv_cache = _make_kv_cache(num_blocks=2)
    kv_cache.allocate_sequence(seq_id=33, num_tokens=8)
    kv_cache.swap_out(seq_id=33)
    swapped_cpu_buffers = cast(
        dict[int, torch.Tensor],
        getattr(kv_cache, "_swapped_cpu_buffers"),
    )
    sequence_tables = cast(
        dict[int, object],
        getattr(kv_cache, "_sequence_tables"),
    )

    assert 33 in swapped_cpu_buffers
    blocks_before = kv_cache.get_block_table(33)
    assert len(blocks_before) == 2

    kv_cache.free_gpu_blocks(seq_id=33)

    assert 33 in swapped_cpu_buffers
    assert 33 in sequence_tables
    assert kv_cache.get_block_table(33) == []
    assert kv_cache.block_allocator.num_free_blocks == 2


def _make_paged_backend(num_blocks: int):
    from moe_infinity.runtime.attention_backend import PagedAttentionBackend
    from moe_infinity.runtime.attention_types import KVCacheSpec

    backend = PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=1, head_dim=8, dtype=torch.float16, block_size=4
        ),
        num_gpu_blocks=num_blocks,
        device=torch.device("cpu"),
    )
    backend.create_layered_store(layer_count=1)
    return backend


def test_partial_prefill_recovers_to_prefill_at_same_offset() -> None:
    cache = _make_kv_cache(4)
    backend = _make_paged_backend(num_blocks=4)
    cache.set_block_store(backend.block_store, owner=backend)
    scheduler = Scheduler(
        kv_cache=cache,
        max_batch_size=1,
        max_tokens_per_step=4,
        enable_chunked_prefill=True,
        prefill_chunk_size=4,
    )
    group = _make_group("partial", seq_id=41, prompt_len=8)
    scheduler.add_request(group)
    first = scheduler.schedule()
    scheduler.commit_prefill_step(first.prefill_transaction_id)
    assert group.sequences[0].num_computed_tokens == 4

    preempted = scheduler._preempt_oldest_running_group()
    assert preempted == [41]
    assert group.sequences[0].status is SequenceStatus.SWAPPED
    scheduler._recover_swapped_groups([group])

    assert group.sequences[0].status is SequenceStatus.PREFILL
    resumed = scheduler.schedule()
    assert resumed.prefill_chunks[41].start_pos == 4
