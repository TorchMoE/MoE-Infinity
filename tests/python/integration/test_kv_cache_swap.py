from moe_infinity.memory.kv_cache_manager import KVCacheManager


def test_prepare_commit_cycle() -> None:
    mgr = KVCacheManager(num_gpu_blocks=50, num_cpu_blocks=50, block_size=16)

    assert mgr.allocate_blocks_for_sequence("seq1", num_tokens=64)
    orig_gpu_ids = mgr.get_block_table("seq1")
    assert orig_gpu_ids == [0, 1, 2, 3]

    swap_out_pairs = mgr.prepare_swap_out(0, "seq1")
    assert len(swap_out_pairs) == 4
    assert [gpu_id for gpu_id, _ in swap_out_pairs] == orig_gpu_ids

    mgr.commit_swap_out(0, "seq1", swap_out_pairs)
    assert mgr.get_block_table("seq1") == []
    assert mgr.num_free_gpu_blocks == 50
    assert mgr.num_free_cpu_blocks == 46

    swap_in_pairs = mgr.prepare_swap_in(0, "seq1", orig_gpu_ids)
    assert len(swap_in_pairs) == 4
    assert [cpu_id for cpu_id, _ in swap_in_pairs] == [
        cpu_id for _, cpu_id in swap_out_pairs
    ]

    mgr.commit_swap_in(0, "seq1", orig_gpu_ids, swap_in_pairs)
    assert mgr.num_free_cpu_blocks == 50
    assert len(mgr.get_block_table("seq1")) == 4


def test_multi_sequence() -> None:
    mgr = KVCacheManager(num_gpu_blocks=40, num_cpu_blocks=20, block_size=16)

    assert mgr.allocate_blocks_for_sequence("seq_a", num_tokens=16)
    assert mgr.allocate_blocks_for_sequence("seq_b", num_tokens=33)
    assert mgr.allocate_blocks_for_sequence("seq_c", num_tokens=64)
    assert mgr.allocate_blocks_for_sequence("seq_d", num_tokens=17)

    assert mgr.get_block_table("seq_a") == [0]
    assert mgr.get_block_table("seq_b") == [1, 2, 3]
    assert mgr.get_block_table("seq_c") == [4, 5, 6, 7]
    assert mgr.get_block_table("seq_d") == [8, 9]

    mgr.free_sequence("seq_b")
    assert mgr.get_block_table("seq_b") == []
    assert mgr.get_block_table("seq_a") == [0]
    assert mgr.get_block_table("seq_c") == [4, 5, 6, 7]
    assert mgr.get_block_table("seq_d") == [8, 9]


def test_allocation_failure() -> None:
    mgr = KVCacheManager(num_gpu_blocks=4, num_cpu_blocks=4, block_size=16)

    assert mgr.allocate_blocks_for_sequence("seq1", num_tokens=64)
    assert not mgr.append_token_block("seq1")
    assert mgr.num_free_gpu_blocks == 0

    mgr.free_sequence("seq1")
    assert mgr.num_free_gpu_blocks == 4
