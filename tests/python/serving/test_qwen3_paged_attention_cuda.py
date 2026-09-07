import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from moe_infinity.models.qwen3_paged_attention import Qwen3PagedAttention
from moe_infinity.runtime.attention_backend import (
    LayerRegistration,
    PagedAttentionBackend,
)
from moe_infinity.runtime.attention_types import (
    AttentionMetadata,
    KVCacheSpec,
    PagedBatchLengths,
)
from moe_infinity.runtime.flashinfer_utils import HAS_FLASHINFER


@pytest.mark.skipif(
    not (HAS_FLASHINFER and torch.cuda.is_available()),
    reason="requires real FlashInfer and CUDA",
)
def test_real_qwen3_paged_attention_runs_chunk_through_flashinfer() -> None:
    device = torch.device("cuda")
    config = Qwen3MoeConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=1,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
    )
    attention = (
        Qwen3PagedAttention(config, layer_idx=0)
        .to(device=device, dtype=torch.float16)
        .eval()
    )
    backend = PagedAttentionBackend(
        spec=KVCacheSpec(2, 64, torch.float16, 4),
        num_gpu_blocks=4,
        num_layers=1,
        device=device,
    )
    assert backend._flashinfer_enabled()
    backend.register_layers([LayerRegistration(0, id(attention))])
    prefix_k = torch.randn(4, 2, 64, device=device, dtype=torch.float16)
    prefix_v = torch.randn_like(prefix_k)
    prefix_slots = torch.arange(4, device=device)
    backend.write_kv(prefix_k, prefix_v, prefix_slots, layer_idx=0)
    backend.write_kv_flashinfer(prefix_k, prefix_v, prefix_slots, layer_idx=0)
    metadata = AttentionMetadata(
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        lengths=PagedBatchLengths(
            query_lengths=torch.tensor([2], dtype=torch.int32, device=device),
            query_offsets=torch.tensor(
                [0, 2], dtype=torch.int32, device=device
            ),
            context_lengths=torch.tensor([4], dtype=torch.int32, device=device),
            kv_seq_lengths=torch.tensor([6], dtype=torch.int32, device=device),
        ),
        max_seq_len=6,
        num_prefill_tokens=2,
        num_decode_tokens=0,
        slot_mapping=torch.tensor([4, 5], device=device),
        is_prefill=True,
    )
    hidden = torch.randn(1, 2, 32, device=device, dtype=torch.float16)
    cos = torch.ones(1, 2, 64, device=device, dtype=torch.float16)
    sin = torch.zeros(1, 2, 64, device=device, dtype=torch.float16)
    Qwen3PagedAttention.set_paged_context(backend, metadata)
    try:
        output, weights = attention(
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )
    finally:
        Qwen3PagedAttention.clear_paged_context()

    assert output.shape == hidden.shape
    assert weights is None
    assert torch.isfinite(output).all()
    assert backend.last_flashinfer_plan.lengths.query_offsets.tolist() == [
        0,
        2,
    ]
    assert backend.last_flashinfer_plan.lengths.kv_seq_lengths.tolist() == [6]
