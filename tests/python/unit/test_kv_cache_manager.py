from moe_infinity.memory.kv_cache_manager import KVCacheManager, MemoryBudget


def test_kv_cache_manager_creation() -> None:
    mgr = KVCacheManager(50, 20)
    assert mgr.block_size == 16
    assert mgr.num_gpu_blocks == 50
    assert mgr.num_cpu_blocks == 20
    assert mgr.num_free_gpu_blocks == 50
    assert mgr.num_free_cpu_blocks == 20


def test_allocate_and_free() -> None:
    mgr = KVCacheManager(10, 8, block_size=4)
    ok = mgr.allocate_blocks_for_sequence("seq-a", num_tokens=9)
    assert ok is True
    assert mgr.num_free_gpu_blocks == 7

    mgr.free_sequence("seq-a")
    assert mgr.num_free_gpu_blocks == 10
    assert mgr.get_block_table("seq-a") == []


def test_get_block_table() -> None:
    mgr = KVCacheManager(10, 8, block_size=4)
    ok = mgr.allocate_blocks_for_sequence("seq-b", num_tokens=12)
    assert ok is True
    assert mgr.get_block_table("seq-b") == [0, 1, 2]


def test_prepare_commit_swap_out() -> None:
    mgr = KVCacheManager(10, 10, block_size=4)
    ok = mgr.allocate_blocks_for_sequence("seq-c", num_tokens=8)
    assert ok is True
    assert mgr.num_free_gpu_blocks == 8

    pairs = mgr.prepare_swap_out(0, "seq-c")
    assert len(pairs) == 2

    mgr.commit_swap_out(0, "seq-c", pairs)
    assert mgr.num_free_gpu_blocks == 10
    assert mgr.num_free_cpu_blocks == 8
    assert mgr.get_block_table("seq-c") == []


def test_prepare_commit_swap_in() -> None:
    mgr = KVCacheManager(10, 10, block_size=4)
    ok = mgr.allocate_blocks_for_sequence("seq-d", num_tokens=8)
    assert ok is True

    pairs_out = mgr.prepare_swap_out(0, "seq-d")
    assert len(pairs_out) == 2
    orig_gpu_ids = [gpu_id for gpu_id, _ in pairs_out]
    mgr.commit_swap_out(0, "seq-d", pairs_out)

    pairs_in = mgr.prepare_swap_in(0, "seq-d", orig_gpu_ids)
    assert len(pairs_in) == 2
    mgr.commit_swap_in(0, "seq-d", orig_gpu_ids, pairs_in)

    assert len(mgr.get_block_table("seq-d")) == 2
    assert mgr.num_free_cpu_blocks == 10


def test_memory_budget() -> None:
    budget = MemoryBudget()
    assert isinstance(budget.device_memory_ratio, float)
    assert isinstance(budget.kv_cache_memory_ratio, float)
    assert isinstance(budget.host_memory_ratio, float)
    assert budget.total_gpu_memory_bytes > 0
    assert budget.kv_cache_gpu_bytes >= 0


def test_over_allocation_fails() -> None:
    mgr = KVCacheManager(2, 2, block_size=4)
    ok = mgr.allocate_blocks_for_sequence("seq-e", num_tokens=9)
    assert ok is False
    assert mgr.num_free_gpu_blocks == 2
