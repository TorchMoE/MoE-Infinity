# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
# Regression guards distilled from the PR #195 review thread
# (danielesalpietro / mfethe1): each of these failed silently or loudly on
# earlier mainline states and is cheap to keep because none needs a GPU.

import torch

from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import (
    AttentionMetadata as RuntimeAttentionMetadata,
)
from moe_infinity.runtime.attention_types import (
    KVCacheSpec,
    PagedBatchLengths,
)


def _spec() -> KVCacheSpec:
    return KVCacheSpec(
        num_kv_heads=2, head_dim=16, dtype=torch.float16, block_size=16
    )


def _backend(num_layers: int = 2) -> PagedAttentionBackend:
    return PagedAttentionBackend(
        spec=_spec(),
        num_gpu_blocks=8,
        device=torch.device("cpu"),
        num_layers=num_layers,
    )


def _tokens(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    key = torch.randn(4, 2, 16, dtype=torch.float16, generator=generator)
    value = torch.randn(4, 2, 16, dtype=torch.float16, generator=generator)
    return key, value


def test_fallback_allocation_carries_layer_dimension() -> None:
    backend = _backend(num_layers=3)
    assert backend._k_cache.ndim == 6
    assert backend._v_cache.ndim == 5
    assert backend._k_cache.shape[0] == 3
    assert backend._v_cache.shape[0] == 3


def test_write_kv_layer_one_preserves_layer_zero() -> None:
    backend = _backend(num_layers=2)
    slots = torch.arange(4)
    key0, value0 = _tokens(seed=0)
    key1, value1 = _tokens(seed=1)

    backend.write_kv(key0, value0, slots, layer_idx=0)
    layer0_k = backend._k_cache[0].clone()

    backend.write_kv(key1, value1, slots, layer_idx=1)

    torch.testing.assert_close(backend._k_cache[0], layer0_k)
    assert not torch.equal(backend._k_cache[1], layer0_k)


def test_decode_forward_writes_kv() -> None:
    backend = _backend(num_layers=1)
    prefill_key, prefill_value = _tokens(seed=2)
    prefill_meta = RuntimeAttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([4], dtype=torch.int32),
            query_offsets=torch.tensor([0, 4], dtype=torch.int32),
            context_lengths=torch.tensor([0], dtype=torch.int32),
            kv_seq_lengths=torch.tensor([4], dtype=torch.int32),
        ),
        max_seq_len=4,
        num_prefill_tokens=4,
        num_decode_tokens=0,
        slot_mapping=torch.tensor([0, 1, 2, 3]),
        is_prefill=True,
    )
    backend.forward(
        query=torch.randn(4, 2, 16, dtype=torch.float16),
        key=prefill_key,
        value=prefill_value,
        metadata=prefill_meta,
        layer_idx=0,
    )

    decode_slot = 4
    before = backend._k_cache[0, 0, :, :, decode_slot % 16, :].clone()
    decode_key, decode_value = _tokens(seed=3)
    decode_meta = RuntimeAttentionMetadata(
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([1], dtype=torch.int32),
            query_offsets=torch.tensor([0, 1], dtype=torch.int32),
            context_lengths=torch.tensor([4], dtype=torch.int32),
            kv_seq_lengths=torch.tensor([5], dtype=torch.int32),
        ),
        max_seq_len=5,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([decode_slot]),
        is_prefill=False,
    )
    backend.forward(
        query=torch.randn(1, 2, 16, dtype=torch.float16),
        key=decode_key[:1],
        value=decode_value[:1],
        metadata=decode_meta,
        layer_idx=0,
    )
    after = backend._k_cache[0, 0, :, :, decode_slot % 16, :]
    assert not torch.equal(after, before)


def test_int8_spec_leaves_fallback_caches_unallocated() -> None:
    spec = _spec()
    spec.format_name = "int8_sym"
    backend = PagedAttentionBackend(
        spec=spec, num_gpu_blocks=8, device=torch.device("cpu")
    )
    assert backend._is_int8
    assert backend._k_cache is None
    assert backend._v_cache is None
