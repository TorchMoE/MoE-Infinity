import os

import pytest

from moe_infinity import _store


def _prefix(name: str) -> str:
    path = f"/tmp/moe_multimodel_{name}/"
    os.makedirs(path, exist_ok=True)
    return path


def test_second_handle_raises_instead_of_corrupting():
    first = _store.prefetch_handle(_prefix("a"), 0.5)
    with pytest.raises(RuntimeError):
        _store.prefetch_handle(_prefix("b"), 0.5)
    del first


def test_handle_construction_after_destroy_is_allowed():
    first = _store.prefetch_handle(_prefix("c"), 0.5)
    del first
    second = _store.prefetch_handle(_prefix("d"), 0.5)
    del second
