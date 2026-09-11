import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import KVCacheSpec

ROOT = Path(__file__).resolve().parents[3]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def _load_module(module_name: str, file_path: Path) -> types.ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")

_ = _load_module(
    "moe_infinity.serving.sequence",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
_ = _load_module(
    "moe_infinity.serving.kv_cache",
    ROOT / "moe_infinity" / "serving" / "kv_cache.py",
)
_ = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)
_ = _load_module(
    "moe_infinity.serving.memory_manager",
    ROOT / "moe_infinity" / "serving" / "memory_manager.py",
)
_ = _load_module(
    "moe_infinity.serving.scheduler",
    ROOT / "moe_infinity" / "serving" / "scheduler.py",
)
_ = _load_module(
    "moe_infinity.serving.model_runner",
    ROOT / "moe_infinity" / "serving" / "model_runner.py",
)
_ = _load_module(
    "moe_infinity.serving.sampler",
    ROOT / "moe_infinity" / "serving" / "sampler.py",
)
_ = _load_module(
    "moe_infinity.serving.engine",
    ROOT / "moe_infinity" / "serving" / "engine.py",
)

from moe_infinity.engine.kv_transfer import (
    CopyTicket,
    KVTransferState,
    PinnedBufferPool,
)
from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.kv_cache import PagedKVCache
from moe_infinity.serving.sequence import SamplingParams


@dataclass
class MockConfig:
    vocab_size: int = 100
    eos_token_id: int = 2


class MockModel:
    config: MockConfig

    def __init__(self, vocab_size: int = 100, eos_token_id: int = 2) -> None:
        self.config = MockConfig(
            vocab_size=vocab_size, eos_token_id=eos_token_id
        )

    def eval(self) -> None:
        return None

    def forward(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> types.SimpleNamespace:
        _ = kwargs
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.config.vocab_size),
            fill_value=-1e9,
            dtype=torch.float32,
        )
        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                token_id = int(input_ids[batch_idx, token_idx].item())
                logits[
                    batch_idx,
                    token_idx,
                    (token_id + 1) % self.config.vocab_size,
                ] = 0.0
        return types.SimpleNamespace(logits=logits)


class MockOffloadEngine:
    request_id: int
    expert_tracer: types.SimpleNamespace
    expert_layer_modules: list[types.SimpleNamespace]

    def __init__(self) -> None:
        self.request_id = 0
        self.expert_tracer = types.SimpleNamespace(create_entry=lambda: 0)
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]

    def _generate_request_id(self) -> int:
        request_id = self.request_id
        self.request_id += 1
        return request_id


def _make_config(
    *, num_kv_blocks: int = 2, max_batch_size: int = 2
) -> dict[str, object]:
    return {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": 0.25,
        "max_batch_size": max_batch_size,
        "max_tokens_per_step": 16,
        "block_size": 4,
        "num_layers": 1,
        "num_kv_heads": 2,
        "head_dim": 8,
        "dtype": "float16",
        "eos_token_id": 2,
        "num_kv_blocks": num_kv_blocks,
    }


def _make_engine(
    *, num_kv_blocks: int = 2, max_batch_size: int = 2
) -> ContinuousBatchingEngine:
    return ContinuousBatchingEngine(
        model=MockModel(),
        engine=MockOffloadEngine(),
        config=_make_config(
            num_kv_blocks=num_kv_blocks, max_batch_size=max_batch_size
        ),
    )


def test_cancel_waiting_request_frees_nothing() -> None:
    engine = _make_engine(num_kv_blocks=4, max_batch_size=1)
    original_free_blocks = engine.kv_cache.block_allocator.num_free_blocks

    engine.add_request(
        request_id="req-1",
        prompt_token_ids=[10, 11, 12, 13, 14],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )
    engine.add_request(
        request_id="req-2",
        prompt_token_ids=[20, 21, 22, 23, 24],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )

    _ = engine.step()
    free_blocks_before_cancel = engine.kv_cache.block_allocator.num_free_blocks

    engine.abort_request("req-2")

    assert (
        engine.kv_cache.block_allocator.num_free_blocks
        == free_blocks_before_cancel
    )
    assert (
        engine.kv_cache.block_allocator.num_free_blocks < original_free_blocks
    )


def test_cancel_running_request_frees_blocks() -> None:
    engine = _make_engine(num_kv_blocks=2, max_batch_size=1)
    original_free_blocks = engine.kv_cache.block_allocator.num_free_blocks

    engine.add_request(
        request_id="req-1",
        prompt_token_ids=[10, 11, 12, 13, 14],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )

    _ = engine.step()
    assert (
        engine.kv_cache.block_allocator.num_free_blocks < original_free_blocks
    )

    engine.abort_request("req-1")

    assert (
        engine.kv_cache.block_allocator.num_free_blocks == original_free_blocks
    )
    assert engine.has_pending_requests() is False


def test_cancel_nonexistent_is_noop() -> None:
    engine = _make_engine()
    original_free_blocks = engine.kv_cache.block_allocator.num_free_blocks

    engine.abort_request("missing-request")

    assert (
        engine.kv_cache.block_allocator.num_free_blocks == original_free_blocks
    )
    assert engine.get_stats()["cancelled_requests"] == 0
    assert engine.has_pending_requests() is False


def test_cancelled_request_not_in_step_output() -> None:
    engine = _make_engine(num_kv_blocks=4, max_batch_size=2)

    engine.add_request(
        request_id="req-1",
        prompt_token_ids=[10],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )
    engine.add_request(
        request_id="req-2",
        prompt_token_ids=[20],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )

    first_step = engine.step()
    assert {output.request_id for output in first_step} == {"req-1", "req-2"}

    engine.abort_request("req-1")

    future_step = engine.step()
    assert [output.request_id for output in future_step] == ["req-2"]


def test_no_block_leak_after_cancel() -> None:
    engine = _make_engine(num_kv_blocks=2, max_batch_size=1)
    original_free_blocks = engine.kv_cache.block_allocator.num_free_blocks

    engine.add_request(
        request_id="req-1",
        prompt_token_ids=[10, 11, 12, 13, 14],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )
    _ = engine.step()

    engine.abort_request("req-1")
    assert (
        engine.kv_cache.block_allocator.num_free_blocks == original_free_blocks
    )

    engine.add_request(
        request_id="req-2",
        prompt_token_ids=[30, 31, 32, 33, 34],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )
    _ = engine.step()

    assert (
        engine.kv_cache.block_allocator.num_free_blocks < original_free_blocks
    )


class _FakeEvent:
    def __init__(self) -> None:
        self.done = False

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.done = True


class _FakeAsyncBackend:
    asynchronous = True

    def __init__(self) -> None:
        self.events: list[_FakeEvent] = []

    def submit_d2h(
        self,
        source_cache: torch.Tensor,
        destination: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        source = source_cache.index_select(
            block_dim, torch.tensor(block_ids, dtype=torch.long)
        )
        destination.copy_(source)
        event = _FakeEvent()
        self.events.append(event)
        return CopyTicket(
            device=source.device,
            stream=None,
            event=event,
            owned_staging_tensors=(),
            submitted_ns=1,
            nbytes=source.numel() * source.element_size(),
        )

    def submit_h2d(
        self,
        source: torch.Tensor,
        destination_cache: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        destination_cache.index_copy_(
            block_dim,
            torch.tensor(block_ids, dtype=torch.long),
            source,
        )
        event = _FakeEvent()
        self.events.append(event)
        return CopyTicket(
            device=destination_cache.device,
            stream=None,
            event=event,
            owned_staging_tensors=(),
            submitted_ns=1,
            nbytes=source.numel() * source.element_size(),
        )

    def close(self) -> None:
        return None


def _make_async_cache(backend: _FakeAsyncBackend) -> PagedKVCache:
    pool = PinnedBufferPool(
        capacity_bytes=1 << 20,
        allocator=lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    return PagedKVCache(
        num_blocks=2,
        block_size=4,
        num_layers=1,
        num_heads=1,
        head_dim=8,
        dtype=torch.float16,
        device=torch.device("cpu"),
        transfer_backend=backend,
        pinned_pool=pool,
    )


def test_cancel_during_d2h_holds_resources_until_event_completion() -> None:
    backend = _FakeAsyncBackend()
    cache = _make_async_cache(backend)
    cache.allocate_sequence(5, num_tokens=8)
    assert cache.request_swap_out(5)
    assert cache.transfer_state(5) is KVTransferState.SWAP_OUT_IN_FLIGHT
    held_lease_bytes = cache.get_swap_stats()["host_in_use_bytes"]

    cache.cancel_sequence(5)

    assert cache.block_allocator.num_free_blocks == 0
    assert cache.get_swap_stats()["host_in_use_bytes"] == held_lease_bytes

    assert backend.events[-1].done is False
    assert cache.poll_transfers() == []
    assert cache.block_allocator.num_free_blocks == 0
    assert cache.get_swap_stats()["host_in_use_bytes"] == held_lease_bytes

    backend.events[-1].done = True
    cache.poll_transfers()

    assert cache.block_allocator.num_free_blocks == 2
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0


def test_cancel_during_h2d_holds_resources_until_event_completion() -> None:
    backend = _FakeAsyncBackend()
    cache = _make_async_cache(backend)
    cache.allocate_sequence(6, num_tokens=8)
    assert cache.request_swap_out(6)
    backend.events[-1].done = True
    cache.poll_transfers()
    assert cache.request_swap_in(6)
    assert cache.transfer_state(6) is KVTransferState.SWAP_IN_IN_FLIGHT

    cache.cancel_sequence(6)

    assert cache.block_allocator.num_free_blocks == 0

    backend.events[-1].done = True
    cache.poll_transfers()

    assert cache.block_allocator.num_free_blocks == 2
    assert cache.get_swap_stats()["host_in_use_bytes"] == 0


def test_async_cache_shutdown_twice_is_safe() -> None:
    backend = _FakeAsyncBackend()
    cache = _make_async_cache(backend)
    cache.allocate_sequence(8, num_tokens=8)
    assert cache.request_swap_out(8)

    cache.shutdown()
    cache.shutdown()

    assert cache.get_swap_stats()["host_in_use_bytes"] == 0


def _make_paged_backend(
    num_blocks: int, device: torch.device
) -> PagedAttentionBackend:
    backend = PagedAttentionBackend(
        spec=KVCacheSpec(
            num_kv_heads=2, head_dim=8, dtype=torch.float16, block_size=4
        ),
        num_gpu_blocks=num_blocks,
        device=device,
    )
    backend.create_layered_store(layer_count=1)
    return backend


def test_cancel_partial_prefill_frees_reserved_chunks() -> None:
    engine = _make_engine(num_kv_blocks=4, max_batch_size=1)
    backend = _make_paged_backend(num_blocks=4, device=engine.kv_cache.device)
    engine.kv_cache.set_block_store(backend.block_store, owner=backend)
    engine.scheduler.chunked_prefill_requested = True
    engine.scheduler.prefill_chunk_size = 4
    engine.scheduler.max_tokens_per_step = 4
    engine.scheduler.set_chunked_prefill_runtime_enabled(True)
    original_free = engine.kv_cache.block_allocator.num_free_blocks
    engine.add_request(
        "partial",
        list(range(10)),
        SamplingParams(max_tokens=2, temperature=0.0),
    )

    assert engine.step() == []
    assert engine.kv_cache.block_allocator.num_free_blocks < original_free
    engine.abort_request("partial")

    assert engine.kv_cache.block_allocator.num_free_blocks == original_free
    assert engine.has_pending_requests() is False
