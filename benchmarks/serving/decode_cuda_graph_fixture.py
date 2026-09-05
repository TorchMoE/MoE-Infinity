from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from moe_infinity.runtime.paged_kv_storage import PagedKVStorage
    from moe_infinity.serving.batch import BatchMetadata
    from moe_infinity.serving.cuda_graph import CudaGraphRunner
    from moe_infinity.serving.model_runner import ModelRunner


@dataclass
class BenchmarkFixture:
    model_runner: ModelRunner
    graph_runner: CudaGraphRunner
    storage: PagedKVStorage
    make_batch: Callable[[int, int], BatchMetadata]
    _request_blocks: list[int] = field(repr=False)

    def close(self) -> None:
        self.graph_runner.close()
        if self._request_blocks:
            self.storage.block_allocator.free(self._request_blocks)
            self._request_blocks = []


def build_fixture(
    *,
    batch_sizes: tuple[int, ...],
    context_sizes: tuple[int, ...],
    warmup_iters: int,
    max_graph_memory_bytes: int = 0,
    device: torch.device | None = None,
) -> BenchmarkFixture:
    # Initialize the serving package through its normal engine-first path before
    # importing runtime storage, whose allocator compatibility shim lives in
    # ``serving.kv_cache``.
    importlib.import_module("moe_infinity.serving")
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )

    from moe_infinity.models.paged_attention_registry import (
        PagedAttentionLayerRegistry,
    )
    from moe_infinity.models.qwen3_paged_attention import Qwen3PagedAttention
    from moe_infinity.runtime.attention_backend import PagedAttentionBackend
    from moe_infinity.runtime.attention_types import DecodeGraphCapability
    from moe_infinity.runtime.paged_kv_storage import (
        PagedKVStorage,
        PagedKVStorageSpec,
    )
    from moe_infinity.serving.batch import BatchMetadata
    from moe_infinity.serving.cuda_graph import CudaGraphRunner
    from moe_infinity.serving.model_runner import ModelRunner
    from moe_infinity.serving.sequence import SamplingParams

    class TwoLayerQwen3(torch.nn.Module):
        def __init__(self, config: Qwen3MoeConfig) -> None:
            super().__init__()
            self.config = config
            self.embed = torch.nn.Embedding(
                config.vocab_size, config.hidden_size
            )
            self.layers = torch.nn.ModuleList(
                [
                    Qwen3PagedAttention(config, layer_idx=index)
                    for index in range(2)
                ]
            )
            self.lm_head = torch.nn.Linear(
                config.hidden_size, config.vocab_size
            )

        def forward(self, input_ids: torch.Tensor, **_: object) -> object:
            hidden = self.embed(input_ids)
            seq_len = input_ids.shape[1]
            head_dim = self.config.head_dim
            cos = torch.ones(1, seq_len, head_dim, device=input_ids.device)
            sin = torch.zeros(1, seq_len, head_dim, device=input_ids.device)
            for layer in self.layers:
                attention, _ = layer(
                    hidden_states=hidden,
                    position_embeddings=(cos, sin),
                    attention_mask=None,
                )
                hidden = hidden + attention
            return SimpleNamespace(logits=self.lm_head(hidden))

    class CapabilityProvider:
        def decode_graph_capability(self) -> DecodeGraphCapability:
            return DecodeGraphCapability(True, "eligible")

    if not torch.cuda.is_available():
        raise RuntimeError("fixture mode requires CUDA")
    target = device or torch.device("cuda", 0)
    block_size = 16
    max_batch = max(batch_sizes)
    max_blocks_per_row = math.ceil(max(context_sizes) / block_size)
    request_block_count = max_batch * max_blocks_per_row
    storage = PagedKVStorage(
        PagedKVStorageSpec(
            num_layers=2,
            num_blocks=request_block_count + max_batch + 16,
            block_size=block_size,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device=target,
        )
    )
    request_blocks = storage.block_allocator.allocate(request_block_count)
    backend = PagedAttentionBackend(storage=storage, use_flashinfer=False)
    config = Qwen3MoeConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        vocab_size=64,
    )
    torch.manual_seed(0)
    model = TwoLayerQwen3(config).to(target).eval()
    registry = PagedAttentionLayerRegistry.register(model, backend, storage)
    engine = SimpleNamespace(
        request_id=0,
        kv_cache=SimpleNamespace(storage=storage, block_size=block_size),
        get_attention_backend=lambda: backend,
    )
    model_runner = ModelRunner(
        model,
        engine,
        device=target,
        paged_kv_storage=storage,
        paged_attention_registry=registry,
        decode_graph_capability_provider=CapabilityProvider(),
    )
    graph_runner = CudaGraphRunner(
        model_runner,
        storage,
        enabled=True,
        batch_buckets=batch_sizes,
        context_buckets=context_sizes,
        warmup_iters=warmup_iters,
        max_graph_memory_bytes=max_graph_memory_bytes,
    )

    def make_batch(batch_size: int, context_size: int) -> BatchMetadata:
        blocks_needed = math.ceil(context_size / block_size)
        block_tables = []
        for row in range(batch_size):
            start = row * max_blocks_per_row
            block_tables.append(request_blocks[start : start + blocks_needed])
        return BatchMetadata(
            seq_ids=list(range(batch_size)),
            input_token_ids=[20 + row for row in range(batch_size)],
            seq_lengths=[1] * batch_size,
            context_lengths=[context_size - 1] * batch_size,
            is_prefill=[False] * batch_size,
            block_tables=block_tables,
            token_offsets=list(range(batch_size + 1)),
            sampling_params=[SamplingParams() for _ in range(batch_size)],
        )

    return BenchmarkFixture(
        model_runner,
        graph_runner,
        storage,
        make_batch,
        request_blocks,
    )


__all__ = ["BenchmarkFixture", "build_fixture"]
