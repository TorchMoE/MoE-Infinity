import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_importable(name):
    if name not in sys.modules:
        try:
            __import__(name)
        except (ImportError, OSError):
            sys.modules[name] = MagicMock()


_ensure_importable("nvtx")
_ensure_importable("moe_infinity._store")
_ensure_importable("moe_infinity._engine")


def test_import_attention_backend():
    from moe_infinity.runtime.attention_backend import AttentionBackend

    assert AttentionBackend is not None


def test_import_attention_metadata():
    from moe_infinity.runtime.attention_backend import AttentionMetadata

    assert AttentionMetadata is not None


def test_import_placeholder_backend():
    from moe_infinity.runtime.attention_backend import (
        PlaceholderAttentionBackend,
    )

    assert PlaceholderAttentionBackend is not None


def test_attention_metadata_has_required_fields():
    from moe_infinity.runtime.attention_backend import AttentionMetadata

    meta = AttentionMetadata(
        is_prefill=True,
        block_table=None,
        slot_mapping=None,
    )
    assert meta.is_prefill is True
    assert meta.block_table is None
    assert meta.slot_mapping is None


def test_placeholder_backend_implements_protocol():
    from moe_infinity.runtime.attention_backend import (
        AttentionBackend,
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    assert isinstance(backend, AttentionBackend)


def test_placeholder_backend_supports_dtype():
    from moe_infinity.runtime.attention_backend import (
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    assert backend.supports_dtype(torch.float16) is True
    assert backend.supports_dtype(torch.bfloat16) is True
    assert backend.supports_dtype(torch.float32) is True


def test_placeholder_backend_get_kv_cache_shape():
    from moe_infinity.runtime.attention_backend import (
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    shape = backend.get_kv_cache_shape(
        num_blocks=8, block_size=16, num_kv_heads=4, head_size=64
    )
    assert isinstance(shape, tuple)
    assert len(shape) >= 3


def test_placeholder_backend_forward_returns_none_or_tensor():
    from moe_infinity.runtime.attention_backend import (
        AttentionMetadata,
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    query = torch.zeros(1, 4, 64)
    key = torch.zeros(1, 4, 64)
    value = torch.zeros(1, 4, 64)
    meta = AttentionMetadata(
        is_prefill=True, block_table=None, slot_mapping=None
    )
    result = backend.forward(
        query, key, value, kv_cache=None, attn_metadata=meta
    )
    assert result is None or isinstance(result, torch.Tensor)


def test_attention_backend_is_runtime_checkable():
    from moe_infinity.runtime.attention_backend import (
        AttentionBackend,
        PlaceholderAttentionBackend,
    )

    backend = PlaceholderAttentionBackend()
    assert isinstance(backend, AttentionBackend)


class _FakeEvent:
    def __init__(self, complete: bool) -> None:
        self.complete = complete

    def query(self) -> bool:
        return self.complete


def test_paged_backend_resize_requires_synchronized_receipt_and_recreates_stores():
    from moe_infinity.engine.memory_resize import ResizeReceipt
    from moe_infinity.runtime.attention_backend import PagedAttentionBackend
    from moe_infinity.runtime.attention_types import KVCacheSpec

    backend = PagedAttentionBackend(
        KVCacheSpec(2, 8, torch.float32, 4), 8, torch.device("cpu")
    )
    old_k, old_v = backend.k_cache, backend.v_cache
    pending = ResizeReceipt(
        device_id=0,
        request_queues_drained=True,
        dispatch_queues_drained=True,
        cuda_events=(_FakeEvent(False),),
        admissions_paused=True,
    )
    with pytest.raises(RuntimeError, match="synchronized"):
        backend.resize_num_blocks(0, 4, pending)
    assert backend.k_cache is old_k and backend.v_cache is old_v

    receipt = ResizeReceipt(
        device_id=0,
        request_queues_drained=True,
        dispatch_queues_drained=True,
        cuda_events=(_FakeEvent(True),),
        admissions_paused=True,
    )
    backend.resize_num_blocks(0, 4, receipt)
    assert backend.k_cache is not old_k and backend.v_cache is not old_v
    assert backend.k_cache.shape[0] == backend.v_cache.shape[0] == 4
    assert backend.k_cache.dtype == old_k.dtype
    assert backend.v_cache.dtype == old_v.dtype
    assert backend.k_cache.device == old_k.device
    assert backend.v_cache.device == old_v.device
    assert backend.k_cache.stride() == old_k.stride()
    assert backend.v_cache.stride() == old_v.stride()
