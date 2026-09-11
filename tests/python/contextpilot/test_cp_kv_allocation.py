from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, cast

import torch

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
KV_CACHE_PATH = Path(ROOT) / "moe_infinity" / "serving" / "kv_cache.py"


class _CPAwareKVManagerProtocol(Protocol):
    allocated_calls: list[tuple[int, list[int]]]
    freed_calls: list[tuple[int, list[int]]]

    def notify_blocks_allocated(
        self, seq_id: int, block_hashes: list[int]
    ) -> None: ...

    def notify_blocks_freed(
        self, seq_id: int, block_hashes: list[int]
    ) -> None: ...


class _PagedKVCacheProtocol(Protocol):
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None: ...

    def set_cp_kv_manager(self, manager: _CPAwareKVManagerProtocol) -> None: ...

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> None: ...

    def free_sequence(self, seq_id: int) -> None: ...

    def get_block_table(self, seq_id: int) -> list[int]: ...


def _load_paged_kv_cache_class() -> type[_PagedKVCacheProtocol]:
    module_name = "task26_kv_cache"
    spec = importlib.util.spec_from_file_location(module_name, KV_CACHE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {KV_CACHE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return cast(type[_PagedKVCacheProtocol], getattr(module, "PagedKVCache"))


class _MockCPManager:
    def __init__(self) -> None:
        self.allocated_calls: list[tuple[int, list[int]]] = []
        self.freed_calls: list[tuple[int, list[int]]] = []

    def notify_blocks_allocated(
        self, seq_id: int, block_hashes: list[int]
    ) -> None:
        self.allocated_calls.append((seq_id, list(block_hashes)))

    def notify_blocks_freed(self, seq_id: int, block_hashes: list[int]) -> None:
        self.freed_calls.append((seq_id, list(block_hashes)))


def _new_cache() -> _PagedKVCacheProtocol:
    PagedKVCache = _load_paged_kv_cache_class()
    return PagedKVCache(
        num_blocks=8,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )


def test_allocation_notifies_cp() -> None:
    cache = _new_cache()
    manager = _MockCPManager()
    cache.set_cp_kv_manager(manager)

    cache.allocate_sequence(seq_id=11, num_tokens=5)

    assert manager.allocated_calls == [(11, [0, 1])]


def test_free_notifies_cp() -> None:
    cache = _new_cache()
    manager = _MockCPManager()
    cache.set_cp_kv_manager(manager)

    cache.allocate_sequence(seq_id=12, num_tokens=7)
    cache.free_sequence(seq_id=12)

    assert manager.freed_calls == [(12, [0, 1])]


def test_no_cp_manager_is_noop() -> None:
    cache = _new_cache()

    cache.allocate_sequence(seq_id=13, num_tokens=3)
    assert cache.get_block_table(13) == [0]

    cache.free_sequence(seq_id=13)
