import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol, cast

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PREFIX_CACHE_PATH = Path(ROOT) / "moe_infinity" / "serving" / "prefix_cache.py"
_MISSING_MODULE = object()


class PrefixCacheProtocol(Protocol):
    block_size: int
    max_entries: int

    def __init__(
        self,
        block_size: int = 16,
        max_entries: int = 1000,
    ) -> None: ...

    def lookup(self, token_ids: list[int]) -> tuple[int, list[int]]: ...

    def insert(self, token_ids: list[int], block_ids: list[int]) -> None: ...

    def evict_lru(self, n: int = 1) -> list[str]: ...

    @property
    def num_entries(self) -> int: ...

    @property
    def hit_rate(self) -> float: ...


def _load_prefix_cache_objects() -> (
    tuple[
        type[PrefixCacheProtocol],
        Callable[[list[int]], str],
    ]
):
    module_name = "task14_prefix_cache"
    spec = importlib.util.spec_from_file_location(
        module_name, PREFIX_CACHE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {PREFIX_CACHE_PATH}")

    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name, _MISSING_MODULE)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_module is _MISSING_MODULE:
            _ = sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = cast(ModuleType, previous_module)
    return (
        cast(type[PrefixCacheProtocol], getattr(module, "PrefixCache")),
        cast(Callable[[list[int]], str], getattr(module, "hash_token_block")),
    )


def test_lookup_empty_cache() -> None:
    PrefixCache, _ = _load_prefix_cache_objects()
    cache = PrefixCache(block_size=4, max_entries=8)

    matched_tokens, matched_blocks = cache.lookup([1, 2, 3, 4])

    assert matched_tokens == 0
    assert matched_blocks == []
    assert cache.num_entries == 0
    assert cache.hit_rate == 0.0


def test_insert_and_lookup_exact() -> None:
    PrefixCache, _ = _load_prefix_cache_objects()
    cache = PrefixCache(block_size=4, max_entries=8)
    tokens = [10, 11, 12, 13, 20, 21, 22, 23]

    cache.insert(tokens, [100, 101])
    matched_tokens, matched_blocks = cache.lookup(tokens)

    assert matched_tokens == 8
    assert matched_blocks == [100, 101]
    assert cache.num_entries == 2
    assert cache.hit_rate == 1.0


def test_partial_prefix_match() -> None:
    PrefixCache, _ = _load_prefix_cache_objects()
    cache = PrefixCache(block_size=16, max_entries=16)
    tokens = list(range(64))
    block_ids = [200, 201, 202, 203]

    cache.insert(tokens, block_ids)
    matched_tokens, matched_blocks = cache.lookup(tokens[:48])

    assert matched_tokens == 48
    assert matched_blocks == block_ids[:3]


def test_lru_eviction() -> None:
    PrefixCache, _ = _load_prefix_cache_objects()
    cache = PrefixCache(block_size=4, max_entries=2)

    first_block = [1, 2, 3, 4]
    second_block = [5, 6, 7, 8]
    third_block = [9, 10, 11, 12]

    cache.insert(first_block, [1])
    cache.insert(second_block, [2])
    cache.insert(third_block, [3])

    assert cache.num_entries == 2
    assert cache.lookup(first_block) == (0, [])
    assert cache.lookup(second_block) == (4, [2])
    assert cache.lookup(third_block) == (4, [3])


def test_hit_rate_tracking() -> None:
    PrefixCache, _ = _load_prefix_cache_objects()
    cache = PrefixCache(block_size=4, max_entries=8)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    cache.insert(tokens, [10, 11])
    assert cache.lookup([1, 2, 3, 4]) == (4, [10])
    assert cache.lookup([9, 9, 9, 9]) == (0, [])
    assert cache.lookup(tokens) == (8, [10, 11])
    assert abs(cache.hit_rate - (2.0 / 3.0)) < 1e-12
