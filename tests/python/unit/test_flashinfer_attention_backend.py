import itertools
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from moe_infinity.runtime import attention_backend as attention_backend_module
from moe_infinity.runtime import flashinfer_utils
from moe_infinity.runtime.attention_types import (
    AttentionMetadata,
    KVCacheSpec,
    PagedBatchLengths,
)

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
        attention_backend_module.flashinfer_utils,
        "HAS_FLASHINFER",
        True,
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "get_flashinfer_module",
        lambda: fake_module,
    )
    monkeypatch.setattr(
        attention_backend_module.flashinfer_utils,
        "get_workspace",
        lambda device: torch.empty(
            1024,
            dtype=torch.uint8,
            device=device,
        ),
    )


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
