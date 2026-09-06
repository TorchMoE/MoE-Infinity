from __future__ import annotations

from tests.python.serving.prefix_cache_test_utils import (
    make_cache,
    make_qwen_runner,
)


def test_prefix_capability_rejects_missing_or_duplicate_qwen_layers() -> None:
    runner = make_qwen_runner(layer_indices=[0, 1, 1], expected_layers=3)
    capability = runner.get_prefix_reuse_capability(make_cache())
    assert capability.supported is False
    assert capability.reason == "incomplete-paged-layer-registry"


def test_prefix_capability_active_for_complete_qwen_registry() -> None:
    runner = make_qwen_runner(layer_indices=[0, 1, 2], expected_layers=3)
    capability = runner.get_prefix_reuse_capability(make_cache())
    assert capability.supported is True
    assert capability.reason == "active"
    assert capability.backend is not None
    assert capability.block_store is not None
