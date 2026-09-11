import itertools
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from moe_infinity.engine.memory_resize import ResizeReceipt
from moe_infinity.runtime import attention_backend as attention_backend_module
from moe_infinity.runtime import flashinfer_utils
from moe_infinity.runtime.attention_types import AttentionMetadata, KVCacheSpec
from moe_infinity.runtime.kv_cache_format import allocate_layered_paged_kv_store


def _make_layered_store(
    *,
    format_name: str = "int8_sym",
    num_layers: int = 2,
    num_blocks: int = 4,
    owner_id: str = "serving-engine-1",
):
    return allocate_layered_paged_kv_store(
        owner_id=owner_id,
        format_name=format_name,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        execution_dtype=torch.float32,
        device=torch.device("cpu"),
    )


from moe_infinity.runtime.attention_types import (
    AttentionMetadata,
    KVCacheSpec,
    PagedBatchLengths,
)
from moe_infinity.serving.kv_cache import PagedKVCache

PagedAttentionBackend = attention_backend_module.PagedAttentionBackend


class _FakePrefillWrapper:
    def __init__(self, workspace: torch.Tensor, layout: str) -> None:
        self.workspace = workspace
        self.layout = layout
        self.plan_args = None
        self.run_args = None

    def plan(self, *args, **kwargs) -> None:
        self.plan_args = (args, kwargs)

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        self.run_args = (query, kv_cache)
        return torch.zeros_like(query)


class _FakeDecodeWrapper:
    def __init__(self, workspace: torch.Tensor, layout: str) -> None:
        self.workspace = workspace
        self.layout = layout
        self.plan_args = None
        self.run_args = None

    def plan(self, *args, **kwargs) -> None:
        self.plan_args = (args, kwargs)

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        self.run_args = (query, kv_cache)
        return torch.zeros_like(query)


def _spec() -> KVCacheSpec:
    return KVCacheSpec(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        block_size=4,
    )


def _prefill_metadata(num_tokens: int) -> AttentionMetadata:
    return AttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int64),
        lengths=PagedBatchLengths(
            query_lengths=[num_tokens],
            query_offsets=[0, num_tokens],
            context_lengths=[0],
            kv_seq_lengths=[num_tokens],
        ),
        max_seq_len=num_tokens,
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        slot_mapping=torch.arange(num_tokens, dtype=torch.long),
        is_prefill=True,
    )


def _decode_metadata(seq_len: int) -> AttentionMetadata:
    return AttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int64),
        lengths=PagedBatchLengths(
            query_lengths=[1],
            query_offsets=[0, 1],
            context_lengths=[seq_len - 1],
            kv_seq_lengths=[seq_len],
        ),
        max_seq_len=seq_len,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([seq_len - 1], dtype=torch.long),
        is_prefill=False,
    )


def _enable_fake_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=_FakePrefillWrapper,
        BatchDecodeWithPagedKVCacheWrapper=_FakeDecodeWrapper,
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils, "HAS_FLASHINFER", True
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "get_flashinfer_module",
        lambda: fake_module,
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "get_workspace",
        lambda device: torch.empty(1024, dtype=torch.uint8, device=device),
    )


class _CompleteEvent:
    def query(self) -> bool:
        return True


def _resize_receipt() -> ResizeReceipt:
    return ResizeReceipt(
        device_id=0,
        request_queues_drained=True,
        dispatch_queues_drained=True,
        cuda_events=(_CompleteEvent(),),
        admissions_paused=True,
    )


def test_flashinfer_resize_recreates_store_and_both_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=8, device=torch.device("cpu")
    )
    old_store = backend._fi_kv_cache
    old_prefill = backend._fi_prefill
    old_decode = backend._fi_decode
    backend.resize_num_blocks(0, 4, _resize_receipt())
    assert backend._fi_kv_cache is not old_store
    assert backend._fi_kv_cache is not None
    assert backend._fi_kv_cache.shape[0] == 4
    assert backend._fi_prefill is not old_prefill
    assert backend._fi_decode is not old_decode
    assert backend._fi_prefill is not backend._fi_decode

    backend.forward(
        query=torch.randn(4, 4, 8),
        key=torch.randn(4, 2, 8),
        value=torch.randn(4, 2, 8),
        attention_metadata=_prefill_metadata(4),
    )
    backend.forward(
        query=torch.randn(1, 4, 8),
        key=torch.randn(1, 2, 8),
        value=torch.randn(1, 2, 8),
        attention_metadata=_decode_metadata(4),
    )
    assert backend._fi_prefill.plan_args is not None
    assert max(backend._fi_prefill.plan_args[0][2].tolist()) < 4
    assert backend._fi_decode.plan_args is not None
    assert max(backend._fi_decode.plan_args[0][1].tolist()) < 4


def test_flashinfer_kv_cache_layout_nhd_with_mocked_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=10,
        device=torch.device("cpu"),
    )
    assert backend._fi_kv_cache is not None
    assert backend._fi_kv_cache.shape == (10, 2, 4, 2, 8)


def test_flashinfer_prefill_metadata_is_int32_with_mocked_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=4,
        device=torch.device("cpu"),
    )

    query = torch.randn(4, 4, 8)
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    out = backend.forward(
        query=query,
        key=key,
        value=value,
        attention_metadata=_prefill_metadata(num_tokens=4),
    )

    assert out.shape == (4, 4, 8)
    assert backend._fi_prefill is not None
    assert backend._fi_prefill.plan_args is not None
    plan_args = backend._fi_prefill.plan_args[0]
    assert plan_args[0].dtype == torch.int32
    assert plan_args[1].dtype == torch.int32
    assert plan_args[2].dtype == torch.int32
    assert plan_args[3].dtype == torch.int32


def test_flashinfer_decode_metadata_is_int32_with_mocked_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=4,
        device=torch.device("cpu"),
    )

    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    backend.write_kv_flashinfer(
        key=key,
        value=value,
        slot_mapping=torch.arange(4, dtype=torch.long),
    )

    out = backend.forward(
        query=torch.randn(1, 4, 8),
        key=key[:1],
        value=value[:1],
        attention_metadata=_decode_metadata(seq_len=4),
    )
    assert out.shape == (1, 4, 8)
    assert backend._fi_decode is not None
    assert backend._fi_decode.plan_args is not None
    plan_args = backend._fi_decode.plan_args[0]
    assert plan_args[0].dtype == torch.int32
    assert plan_args[1].dtype == torch.int32
    assert plan_args[2].dtype == torch.int32


def test_write_kv_flashinfer_writes_expected_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=2,
        device=torch.device("cpu"),
    )

    key = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
    value = key + 1000.0
    slot_mapping = torch.tensor([0, 3, 4], dtype=torch.long)

    backend.write_kv_flashinfer(key=key, value=value, slot_mapping=slot_mapping)
    assert backend._fi_kv_cache is not None
    for i in range(slot_mapping.shape[0]):
        slot = int(slot_mapping[i].item())
        block_id = slot // backend.spec.block_size
        token_offset = slot % backend.spec.block_size
        torch.testing.assert_close(
            backend._fi_kv_cache[block_id, 0, token_offset], key[i]
        )
        torch.testing.assert_close(
            backend._fi_kv_cache[block_id, 1, token_offset], value[i]
        )


def test_fallback_prefill_without_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "HAS_FLASHINFER",
        False,
    )
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=10,
        device=torch.device("cpu"),
    )

    out = backend.forward(
        query=torch.randn(4, 4, 8),
        key=torch.randn(4, 2, 8),
        value=torch.randn(4, 2, 8),
        attention_metadata=_prefill_metadata(num_tokens=4),
    )
    assert out.shape == (4, 4, 8)
    assert backend._fi_prefill is None


def test_fallback_decode_without_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "HAS_FLASHINFER",
        False,
    )
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=10,
        device=torch.device("cpu"),
    )
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    backend.write_kv(key=key, value=value, slot_mapping=torch.arange(4))

    out = backend.forward(
        query=torch.randn(1, 4, 8),
        key=key[:1],
        value=value[:1],
        attention_metadata=_decode_metadata(seq_len=4),
    )
    assert out.shape == (1, 4, 8)
    assert backend._fi_decode is None


def test_flashinfer_qo_indptr_uses_chunk_queries_not_total_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=8, device=torch.device("cpu")
    )
    metadata = AttentionMetadata(
        block_tables=torch.tensor([[0, 1, 0], [2, 3, 4]], dtype=torch.int32),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([2, 3], dtype=torch.int32),
            query_offsets=torch.tensor([0, 2, 5], dtype=torch.int32),
            context_lengths=torch.tensor([4, 6], dtype=torch.int32),
            kv_seq_lengths=torch.tensor([6, 9], dtype=torch.int32),
        ),
        max_seq_len=9,
        num_prefill_tokens=5,
        num_decode_tokens=0,
        slot_mapping=torch.tensor([4, 5, 14, 15, 16]),
        is_prefill=True,
    )
    backend.forward(
        query=torch.randn(5, 4, 8),
        key=torch.randn(5, 2, 8),
        value=torch.randn(5, 2, 8),
        attention_metadata=metadata,
    )

    assert backend._fi_prefill is not None
    plan_args = backend._fi_prefill.plan_args[0]
    assert plan_args[0].tolist() == [0, 2, 5]
    assert plan_args[1].tolist() == [0, 2, 5]
    assert plan_args[3].tolist() == [2, 1]


def _make_serving_cache(num_blocks: int, block_size: int) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_layered_store_checkpoint_restores_both_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=4, device=torch.device("cpu")
    )
    backend.create_layered_store(layer_count=1)
    checkpoint = backend.block_store.checkpoint([1])
    key = torch.full((2, 2, 8), 7.0)
    value = torch.full((2, 2, 8), 9.0)
    slots = torch.tensor([4, 5])
    backend.write_kv(key, value, slots)
    backend.write_kv_flashinfer(key, value, slots)

    backend.block_store.restore([1], checkpoint)

    payload = backend.block_store.export_blocks([1])
    assert torch.count_nonzero(payload.k_cache) == 0
    assert torch.count_nonzero(payload.v_cache) == 0
    assert torch.count_nonzero(payload.fi_kv_cache) == 0


def test_swap_exports_and_restores_runtime_backend_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=_spec(), num_gpu_blocks=4, device=torch.device("cpu")
    )
    backend.create_layered_store(layer_count=1)
    cache = _make_serving_cache(num_blocks=4, block_size=4)
    cache.set_block_store(backend.block_store, owner=backend)
    cache.allocate_sequence(3, num_tokens=4)
    key = torch.arange(64, dtype=torch.float32).reshape(4, 2, 8)
    value = key + 100.0
    backend.write_kv(key, value, torch.arange(4))
    backend.write_kv_flashinfer(key, value, torch.arange(4))

    cache.swap_out(3)
    cache.free_gpu_blocks(3)
    cache.swap_in(3)

    restored = backend.block_store.export_blocks(cache.get_block_table(3))
    torch.testing.assert_close(
        restored.fi_kv_cache[0, :, 0], key.reshape(1, 4, 2, 8)
    )
    torch.testing.assert_close(
        restored.fi_kv_cache[0, :, 1], value.reshape(1, 4, 2, 8)
    )


class RecordingPrefill:
    def plan(
        self,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        *args,
        **kwargs,
    ) -> None:
        self.plan_args = SimpleNamespace(
            qo_indptr=qo_indptr.clone(),
            kv_indptr=kv_indptr.clone(),
            kv_indices=kv_indices.clone(),
            kv_last_page_len=kv_last_page_len.clone(),
        )

    def run(self, query, kv_cache):
        return query


def make_recording_backend(monkeypatch: pytest.MonkeyPatch):
    prefill, decode = RecordingPrefill(), Mock(plan=Mock(), run=Mock())
    module = SimpleNamespace(
        BatchPrefillWithPagedKVCacheWrapper=lambda workspace, layout: prefill,
        BatchDecodeWithPagedKVCacheWrapper=lambda workspace, layout: decode,
    )
    monkeypatch.setattr(flashinfer_utils, "HAS_FLASHINFER", True)
    monkeypatch.setattr(
        flashinfer_utils, "get_flashinfer_module", lambda: module
    )
    monkeypatch.setattr(
        flashinfer_utils, "get_workspace", lambda device: torch.empty(1)
    )
    spec = KVCacheSpec(
        num_kv_heads=2, head_dim=8, dtype=torch.float16, block_size=16
    )
    backend = PagedAttentionBackend(
        spec, num_gpu_blocks=16, device=torch.device("cpu")
    )
    backend.create_layered_store(layer_count=1)
    return backend, prefill


def make_tables(kv_seq_lengths: list[int], block_size: int) -> torch.Tensor:
    rows = [
        (length + block_size - 1) // block_size for length in kv_seq_lengths
    ]
    table = torch.zeros(len(rows), max(rows), dtype=torch.int32)
    cursor = 0
    for row, count in enumerate(rows):
        table[row, :count] = torch.arange(
            cursor, cursor + count, dtype=torch.int32
        )
        cursor += count
    return table


def make_slots(
    query_lengths: list[int], kv_seq_lengths: list[int]
) -> torch.Tensor:
    slots = []
    for query_len, kv_len in zip(query_lengths, kv_seq_lengths):
        slots.extend(range(kv_len - query_len, kv_len))
    return torch.tensor(slots, dtype=torch.int64)


def make_q(tokens: int) -> torch.Tensor:
    return torch.zeros(tokens, 2, 8, dtype=torch.float16)


make_k = make_q
make_v = make_q


@pytest.mark.parametrize(
    ("query_lengths", "kv_seq_lengths", "expected_qo", "expected_last_page"),
    [
        ([80], [80], [0, 80], [16]),
        ([16], [80], [0, 16], [16]),
        ([3, 5], [67, 21], [0, 3, 8], [3, 5]),
    ],
)
def test_flashinfer_qo_uses_query_and_kv_pages_use_total(
    query_lengths,
    kv_seq_lengths,
    expected_qo,
    expected_last_page,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, recording_prefill = make_recording_backend(monkeypatch)
    query_offsets = [0, *itertools.accumulate(query_lengths)]
    context_lengths = [
        kv - query for query, kv in zip(query_lengths, kv_seq_lengths)
    ]
    metadata = AttentionMetadata(
        block_tables=make_tables(kv_seq_lengths, block_size=16),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor(query_lengths, dtype=torch.int32),
            query_offsets=torch.tensor(query_offsets, dtype=torch.int32),
            context_lengths=torch.tensor(context_lengths, dtype=torch.int32),
            kv_seq_lengths=torch.tensor(kv_seq_lengths, dtype=torch.int32),
        ),
        max_seq_len=max(kv_seq_lengths),
        num_prefill_tokens=sum(query_lengths),
        num_decode_tokens=0,
        slot_mapping=make_slots(query_lengths, kv_seq_lengths),
        is_prefill=True,
    )
    backend.forward(
        query=make_q(sum(query_lengths)),
        key=make_k(sum(query_lengths)),
        value=make_v(sum(query_lengths)),
        metadata=metadata,
        layer_idx=0,
    )
    assert recording_prefill.plan_args.qo_indptr.tolist() == expected_qo
    assert (
        recording_prefill.plan_args.kv_last_page_len.tolist()
        == expected_last_page
    )


def test_export_import_checkpoint_restore_cover_every_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    backend = attention_backend_module.PagedAttentionBackend(
        spec=KVCacheSpec(2, 8, torch.float16, 4),
        num_gpu_blocks=4,
        device=torch.device("cpu"),
    )
    store = backend.create_layered_store(layer_count=3)
    assert backend.block_store is store
    assert store.owner is backend
    assert backend.k_cache is store.k_cache
    assert backend.v_cache is store.v_cache
    assert backend._fi_kv_cache is store.fi_kv_cache
    for layer in range(3):
        store.k_cache[layer, 1].fill_(10 + layer)
        store.v_cache[layer, 1].fill_(20 + layer)
        store.fi_kv_cache[layer, 1].fill_(30 + layer)
    payload = store.export_blocks([1])
    store.import_blocks([2], payload)
    checkpoint = store.checkpoint([2])
    store.zero_blocks([2])
    store.restore([2], checkpoint)
    for layer in range(3):
        assert torch.all(store.k_cache[layer, 2] == 10 + layer)
        assert torch.all(store.v_cache[layer, 2] == 20 + layer)
        assert torch.all(store.fi_kv_cache[layer, 2] == 30 + layer)


@pytest.mark.skipif(
    not flashinfer_utils.HAS_FLASHINFER or not torch.cuda.is_available(),
    reason="requires flashinfer + CUDA",
)
def test_flashinfer_workspace_reuse_across_batches() -> None:
    backend = attention_backend_module.PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2,
            head_dim=16,
            dtype=torch.float16,
            block_size=4,
        ),
        num_gpu_blocks=16,
        device=torch.device("cuda"),
    )

    metadata = AttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([4], dtype=torch.int32, device="cuda"),
            query_offsets=torch.tensor(
                [0, 4], dtype=torch.int32, device="cuda"
            ),
            context_lengths=torch.tensor([0], dtype=torch.int32, device="cuda"),
            kv_seq_lengths=torch.tensor([4], dtype=torch.int32, device="cuda"),
        ),
        max_seq_len=4,
        num_prefill_tokens=4,
        num_decode_tokens=0,
        slot_mapping=torch.arange(4, dtype=torch.long, device="cuda"),
        is_prefill=True,
    )

    query = torch.randn(4, 4, 16, dtype=torch.float16, device="cuda")
    key = torch.randn(4, 2, 16, dtype=torch.float16, device="cuda")
    value = torch.randn(4, 2, 16, dtype=torch.float16, device="cuda")

    workspace0 = backend._fi_workspace
    backend.forward(
        query=query,
        key=key,
        value=value,
        attention_metadata=metadata,
    )
    backend.forward(
        query=query,
        key=key,
        value=value,
        attention_metadata=metadata,
    )
    assert backend._fi_workspace is workspace0


def test_int8_request_does_not_allocate_duplicate_flashinfer_cache(
    monkeypatch,
) -> None:
    _enable_fake_flashinfer(monkeypatch)
    spec = _spec()
    spec.format_name = "int8_sym"
    backend = attention_backend_module.PagedAttentionBackend(
        spec, 4, torch.device("cpu")
    )
    store = _make_layered_store(
        format_name="int8_sym", num_layers=2, num_blocks=4
    )
    backend.bind_store(store, owner_id="serving-engine-1")
    assert backend._fi_kv_cache is None
    assert backend.store is store
    assert backend.effective_format == "int8_sym"
    assert backend.format_decision_reason == "flashinfer_no_int8_sym_contract"
