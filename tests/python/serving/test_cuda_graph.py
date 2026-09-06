# pyright: reportAny=false, reportImplicitOverride=false, reportPrivateUsage=false

import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from tests.python.ops.conftest import requires_cuda

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
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark_module() -> types.ModuleType:
    return _load_module(
        "decode_cuda_graph_benchmark_test",
        ROOT / "benchmarks" / "serving" / "decode_cuda_graph.py",
    )


_ensure_package("moe_infinity", ROOT / "moe_infinity")
_ensure_package("moe_infinity.serving", ROOT / "moe_infinity" / "serving")
_ensure_package("moe_infinity.runtime", ROOT / "moe_infinity" / "runtime")

_ATTENTION_TYPES_MODULE = _load_module(
    "moe_infinity.runtime.attention_types",
    ROOT / "moe_infinity" / "runtime" / "attention_types.py",
)
_SEQUENCE_MODULE = _load_module(
    "moe_infinity.serving.sequence",
    ROOT / "moe_infinity" / "serving" / "sequence.py",
)
_BATCH_MODULE = _load_module(
    "moe_infinity.serving.batch",
    ROOT / "moe_infinity" / "serving" / "batch.py",
)
_MODEL_RUNNER_MODULE = _load_module(
    "moe_infinity.serving.model_runner",
    ROOT / "moe_infinity" / "serving" / "model_runner.py",
)
_CUDA_GRAPH_MODULE = _load_module(
    "moe_infinity.serving.cuda_graph",
    ROOT / "moe_infinity" / "serving" / "cuda_graph.py",
)

SamplingParams = _SEQUENCE_MODULE.SamplingParams
BatchMetadata = _BATCH_MODULE.BatchMetadata
DecodeGraphCapability = _ATTENTION_TYPES_MODULE.DecodeGraphCapability
PagedLayerWriteProof = _ATTENTION_TYPES_MODULE.PagedLayerWriteProof
CudaGraphRunner = _CUDA_GRAPH_MODULE.CudaGraphRunner
GraphKey = _CUDA_GRAPH_MODULE.GraphKey
ModelRunner = _MODEL_RUNNER_MODULE.ModelRunner


@dataclass(frozen=True)
class _FakeCudaOps:
    available: bool = True

    def memory_allocated(self, device: torch.device) -> int:
        return 100


class _FakeStorage:
    def __init__(self, device: torch.device, *, block_size: int = 16) -> None:
        self.owner_id = "owner-1"
        self.spec = types.SimpleNamespace(
            device=device, block_size=block_size, num_blocks=1024, num_layers=2
        )
        self.block_size = block_size
        self.num_blocks = 1024
        self._scratch: set[int] = set()

    def reserve_graph_scratch_blocks(self, count: int) -> list[int]:
        block_ids = list(range(count))
        self._scratch.update(block_ids)
        return block_ids

    def release_graph_scratch_blocks(self, block_ids: list[int]) -> None:
        self._scratch.difference_update(block_ids)

    @property
    def num_graph_scratch_blocks(self) -> int:
        return len(self._scratch)

    @property
    def graph_scratch_blocks(self) -> frozenset[int]:
        return frozenset(self._scratch)


def _make_registry(
    bindings: list[tuple[str, int]], owner_id: str
) -> types.SimpleNamespace:
    binding_objs = [
        types.SimpleNamespace(
            class_fqn=class_fqn,
            layer_idx=layer_idx,
            storage_owner_id=owner_id,
            has_write_proof=True,
        )
        for class_fqn, layer_idx in bindings
    ]
    return types.SimpleNamespace(bindings=binding_objs, reason="eligible")


def _eligible_capability_for_registry(
    registry: types.SimpleNamespace, owner_id: str
) -> DecodeGraphCapability:
    proofs = tuple(
        PagedLayerWriteProof(
            class_fqn=binding.class_fqn,
            layer_idx=binding.layer_idx,
            storage_owner_id=owner_id,
            writer="writer",
            writes_before_attention=True,
            allocation_free=True,
        )
        for binding in registry.bindings
    )
    return DecodeGraphCapability(
        True,
        "eligible",
        storage_owner_id=owner_id,
        layer_write_proofs=proofs,
    )


class _FakeModelRunner:
    def __init__(
        self,
        device: torch.device,
        registry: types.SimpleNamespace,
        capability: DecodeGraphCapability,
    ) -> None:
        self.device = device
        self.paged_attention_registry = registry
        self._capability = capability

    def decode_graph_capability(self) -> DecodeGraphCapability:
        return self._capability

    def _resolve_vocab_size(self) -> int:
        return 8

    def allocate_decode_buffers(self, **_: object) -> object:
        raise AssertionError("allocate_decode_buffers should not run on CPU")

    def copy_decode_batch(self, *_: object) -> None:
        raise AssertionError("copy_decode_batch should not run on CPU")

    def prepare_batch_side_effects(self, _: object) -> None:
        return None

    def forward_prepared_decode(self, _: object) -> torch.Tensor:
        raise AssertionError("forward_prepared_decode should not run on CPU")


def _make_cpu_gate_runner(
    *,
    batch_buckets: tuple[int, ...] = (1, 2, 4),
    context_buckets: tuple[int, ...] = (16, 32),
    enabled: bool = True,
    storage_device: torch.device | None = None,
    runner_device: torch.device | None = None,
    device: torch.device | None = None,
    capability_reason: str = "eligible",
    max_graph_memory_bytes: int = 0,
) -> CudaGraphRunner:
    base = device or torch.device("cpu")
    storage_device = storage_device or base
    runner_device = runner_device or base
    owner_id = "owner-1"
    storage = _FakeStorage(storage_device)
    storage.owner_id = owner_id
    registry = _make_registry([("pkg.Qwen3PagedAttention", 0)], owner_id)
    capability = (
        _eligible_capability_for_registry(registry, owner_id)
        if capability_reason == "eligible"
        else DecodeGraphCapability(False, capability_reason)
    )
    model_runner = _FakeModelRunner(runner_device, registry, capability)
    runner = CudaGraphRunner(
        model_runner,
        storage,
        enabled=enabled,
        batch_buckets=batch_buckets,
        context_buckets=context_buckets,
        max_graph_memory_bytes=max_graph_memory_bytes,
    )
    runner._cuda_ops = _FakeCudaOps(available=True)
    return runner


def _make_decode_batch(
    *,
    batch_size: int = 2,
    context_lengths: list[int] | None = None,
    input_token_ids: list[int] | None = None,
) -> BatchMetadata:
    if context_lengths is None:
        context_lengths = list(range(batch_size))
    if input_token_ids is None:
        input_token_ids = [idx + 10 for idx in range(batch_size)]
    return BatchMetadata(
        seq_ids=list(range(batch_size)),
        input_token_ids=input_token_ids,
        seq_lengths=[1] * batch_size,
        context_lengths=context_lengths,
        is_prefill=[False] * batch_size,
        block_tables=[[0] for _ in range(batch_size)],
        token_offsets=list(range(batch_size + 1)),
        sampling_params=[SamplingParams() for _ in range(batch_size)],
    )


def _make_prefill_batch() -> BatchMetadata:
    return BatchMetadata(
        seq_ids=[1, 2],
        input_token_ids=[11, 12],
        seq_lengths=[1, 1],
        context_lengths=[0, 0],
        is_prefill=[True, False],
        block_tables=[[0], [0]],
        token_offsets=[0, 1, 2],
        sampling_params=[SamplingParams(), SamplingParams()],
    )


def _fake_captured_state_with_one_buffer_on(device: torch.device) -> object:
    good = torch.device("cuda", 0)

    class _T:
        def __init__(self, dev: torch.device) -> None:
            self.device = dev

    class _Buffers:
        def tensor_values(self) -> tuple[object, ...]:
            return (_T(good), _T(good), _T(device))

    return types.SimpleNamespace(
        graph=object(),
        buffers=_Buffers(),
        output_logits=_T(good),
        generation=0,
    )


def test_select_key_uses_next_batch_and_context_bucket() -> None:
    runner = _make_cpu_gate_runner(
        batch_buckets=(1, 2, 4), context_buckets=(16, 32)
    )
    decision = runner.check_eligibility(
        _make_decode_batch(batch_size=3, context_lengths=[3, 17, 7])
    )
    assert decision.eligible
    assert decision.key == GraphKey(batch_size=4, context_size=32)


@pytest.mark.parametrize(
    ("batch", "reason"),
    [
        (_make_prefill_batch(), "not_decode"),
        (_make_decode_batch(batch_size=5), "no_batch_bucket"),
        (
            _make_decode_batch(batch_size=1, context_lengths=[64]),
            "no_context_bucket",
        ),
    ],
)
def test_gate_returns_bounded_fallback_reason(
    batch: BatchMetadata, reason: str
) -> None:
    runner = _make_cpu_gate_runner(
        batch_buckets=(1, 2, 4), context_buckets=(16, 32)
    )
    assert runner.check_eligibility(batch).reason == reason


def test_disabled_by_default_and_environment_kill_switch(monkeypatch) -> None:
    runner = _make_cpu_gate_runner(enabled=False)
    assert runner.check_eligibility(_make_decode_batch()).reason == "disabled"
    monkeypatch.setenv("MOE_DISABLE_CUDA_GRAPHS", "1")
    runner = _make_cpu_gate_runner(enabled=True)
    assert (
        runner.check_eligibility(_make_decode_batch()).reason == "kill_switch"
    )


def test_negative_graph_memory_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_graph_memory_bytes"):
        _make_cpu_gate_runner(max_graph_memory_bytes=-1)


@pytest.mark.parametrize(
    "capability_reason",
    ["native_paged_required", "expert_dispatcher"],
    ids=["resident_non_paged", "offloaded_moe"],
)
def test_unsafe_runtime_executes_eager_without_capture(
    capability_reason: str,
) -> None:
    runner = _make_cpu_gate_runner(capability_reason=capability_reason)

    assert runner.try_execute(_make_decode_batch()) is None
    assert runner.stats()["captures"] == 0
    expected_reason = (
        "kill_switch"
        if os.environ.get("MOE_DISABLE_CUDA_GRAPHS") == "1"
        else capability_reason
    )
    assert runner.stats()["fallback_reasons"][expected_reason] == 1


def test_capture_failure_quarantines_key_and_returns_eager_signal(
    monkeypatch,
) -> None:
    runner = _make_cpu_gate_runner(enabled=True)
    monkeypatch.setattr(
        runner, "_capture_key", Mock(side_effect=RuntimeError("capture"))
    )
    result = runner.try_execute(_make_decode_batch())
    assert result is None
    assert runner.stats()["fallback_reasons"]["capture_failed"] == 1
    assert (
        runner.check_eligibility(_make_decode_batch()).reason == "quarantined"
    )


def test_scratch_reservation_failure_falls_back_as_insufficient_memory(
    monkeypatch,
) -> None:
    runner = _make_cpu_gate_runner(enabled=True)
    monkeypatch.setattr(
        runner.storage,
        "reserve_graph_scratch_blocks",
        Mock(side_effect=RuntimeError("BlockAllocator exhausted")),
    )
    assert runner.try_execute(_make_decode_batch()) is None
    assert runner.stats()["fallback_reasons"]["insufficient_memory"] == 1


def test_graph_gate_rejects_storage_runner_device_mismatch() -> None:
    runner = _make_cpu_gate_runner(
        storage_device=torch.device("cuda", 0),
        runner_device=torch.device("cuda", 1),
    )
    decision = runner.check_eligibility(_make_decode_batch())
    assert decision.reason == "kv_storage_mismatch"
    assert runner.storage.num_graph_scratch_blocks == 0


def test_graph_gate_rejects_any_buffer_on_another_device() -> None:
    runner = _make_cpu_gate_runner(storage_device=torch.device("cuda", 0))
    state = _fake_captured_state_with_one_buffer_on(torch.device("cuda", 1))
    assert runner._validate_state_devices(state) is False
    assert runner.check_state_eligibility(state).reason == "kv_storage_mismatch"


def test_invalidate_waits_for_replay_lock_and_advances_generation() -> None:
    runner = _make_cpu_gate_runner(enabled=True)
    generation = runner.generation
    runner.invalidate("module_reload")
    assert runner.generation == generation + 1
    assert runner.stats()["graphs"] == 0


def test_benchmark_result_schema_does_not_claim_speedup() -> None:
    module = _load_benchmark_module()
    result = module.build_result(
        config={"batch_size": 4},
        eager_us=[100.0, 101.0],
        replay_us=[90.0, 91.0],
        graph_stats={"replays": 2},
        environment={"gpu": "test"},
    )

    assert result["schema_version"] == 1
    assert result["measurements"]["eager_us"] == [100.0, 101.0]
    assert result["measurements"]["replay_us"] == [90.0, 91.0]
    assert "claimed_speedup" not in result


def test_benchmark_fixture_mode_needs_no_model_or_offload_args() -> None:
    module = _load_benchmark_module()
    args = module.parse_args(
        ["--mode", "fixture", "--output-json", "/tmp/out.json"]
    )
    module.validate_args(args)
    assert args.model is None
    assert args.offload_dir is None


def test_benchmark_model_mode_requires_model_and_offload_dir() -> None:
    module = _load_benchmark_module()
    args = module.parse_args(
        ["--mode", "model", "--output-json", "/tmp/out.json"]
    )
    with pytest.raises(ValueError, match="--model and --offload-dir"):
        module.validate_args(args)


@requires_cuda
@pytest.mark.parametrize("real_batch_size", [1, 2, 3, 4])
def test_capture_and_replay_matches_eager_with_padding(
    real_batch_size: int,
) -> None:
    from moe_infinity.kernel.paged_kv_write import paged_kv_write_
    from moe_infinity.runtime.paged_kv_storage import (
        PagedKVStorage,
        PagedKVStorageSpec,
    )

    device = torch.device("cuda", 0)
    storage = PagedKVStorage(
        PagedKVStorageSpec(
            num_layers=2,
            num_blocks=32,
            block_size=4,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device=device,
        )
    )
    class_fqn = "moe_infinity.models.qwen3_paged_attention.Qwen3PagedAttention"
    bindings = [
        types.SimpleNamespace(
            class_fqn=class_fqn,
            layer_idx=layer_idx,
            storage_owner_id=storage.owner_id,
            has_write_proof=True,
        )
        for layer_idx in range(2)
    ]

    class FixtureModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=4)
            self.metadata = None

        def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            **_: object,
        ) -> object:
            assert self.metadata is not None
            token_values = input_ids[:, 0].to(torch.float32)
            key = token_values[:, None, None].expand(-1, 2, 8).contiguous()
            value = key + 100
            for layer_idx in range(2):
                paged_kv_write_(
                    storage,
                    layer_idx=layer_idx,
                    key=key + layer_idx,
                    value=value + layer_idx,
                    slot_mapping=self.metadata.slot_mapping,
                )
            logits = torch.stack(
                (
                    token_values,
                    position_ids[:, 0].to(torch.float32),
                    self.metadata.seq_lens.to(torch.float32),
                    self.metadata.block_tables[:, 0].to(torch.float32),
                ),
                dim=1,
            )
            return types.SimpleNamespace(logits=logits[:, None, :])

    model = FixtureModel().to(device)

    class Registry:
        reason = "eligible"

        def __init__(self) -> None:
            self.bindings = bindings

        def install_metadata(self, metadata: object) -> None:
            model.metadata = metadata

        def clear_metadata(self) -> None:
            model.metadata = None

    capability = DecodeGraphCapability(True, "eligible")
    backend = types.SimpleNamespace(
        storage=storage,
        device=device,
        decode_graph_capability=lambda: capability,
    )
    engine = types.SimpleNamespace(
        kv_cache=types.SimpleNamespace(storage=storage),
        get_attention_backend=lambda: backend,
    )
    provider = types.SimpleNamespace(
        decode_graph_capability=lambda: capability,
    )
    model_runner = ModelRunner(
        model,
        engine,
        device=device,
        paged_kv_storage=storage,
        paged_attention_registry=Registry(),
        decode_graph_capability_provider=provider,
    )
    graph_runner = CudaGraphRunner(
        model_runner,
        storage,
        enabled=True,
        batch_buckets=(1, 2, 4),
        context_buckets=(16, 32),
        warmup_iters=1,
    )
    context_lengths = [2 + index * 3 for index in range(real_batch_size)]
    input_token_ids = [20 + index for index in range(real_batch_size)]
    block_tables = [
        list(range(4 + index * 4, 8 + index * 4))
        for index in range(real_batch_size)
    ]
    batch = BatchMetadata(
        seq_ids=list(range(real_batch_size)),
        input_token_ids=input_token_ids,
        seq_lengths=[1] * real_batch_size,
        context_lengths=context_lengths,
        is_prefill=[False] * real_batch_size,
        block_tables=block_tables,
        token_offsets=list(range(real_batch_size + 1)),
        sampling_params=[SamplingParams() for _ in range(real_batch_size)],
    )

    actual = graph_runner.try_execute(batch)

    assert actual is not None, (
        graph_runner.stats(),
        graph_runner._quarantined,
    )
    expected = torch.tensor(
        [
            [token, context, context + 1, blocks[0]]
            for token, context, blocks in zip(
                input_token_ids, context_lengths, block_tables
            )
        ],
        dtype=torch.float32,
        device=device,
    )
    torch.testing.assert_close(actual, expected)
    assert actual.shape[0] == real_batch_size
    for state in graph_runner._graphs.values():
        assert state.output_logits.device == storage.spec.device
        assert all(
            tensor.device == storage.spec.device
            for tensor in state.buffers.tensor_values()
        )
    for row, context_length in enumerate(context_lengths):
        block_idx = context_length // storage.block_size
        slot = (
            block_tables[row][block_idx] * storage.block_size
            + context_length % storage.block_size
        )
        block_id, offset = divmod(slot, storage.block_size)
        for layer_idx in range(2):
            assert torch.all(
                storage.value_cache[layer_idx, block_id, :, :, offset]
                == input_token_ids[row] + 100 + layer_idx
            )

    first_output = actual.clone()
    next_context_lengths = [value + 1 for value in context_lengths]
    next_input_token_ids = [value + 10 for value in input_token_ids]
    next_block_tables = [list(reversed(blocks)) for blocks in block_tables]
    next_batch = BatchMetadata(
        seq_ids=list(range(real_batch_size)),
        input_token_ids=next_input_token_ids,
        seq_lengths=[1] * real_batch_size,
        context_lengths=next_context_lengths,
        is_prefill=[False] * real_batch_size,
        block_tables=next_block_tables,
        token_offsets=list(range(real_batch_size + 1)),
        sampling_params=[SamplingParams() for _ in range(real_batch_size)],
    )

    next_actual = graph_runner.try_execute(next_batch)

    assert next_actual is not None
    next_expected = torch.tensor(
        [
            [token, context, context + 1, blocks[0]]
            for token, context, blocks in zip(
                next_input_token_ids,
                next_context_lengths,
                next_block_tables,
            )
        ],
        dtype=torch.float32,
        device=device,
    )
    torch.testing.assert_close(next_actual, next_expected)
    torch.testing.assert_close(actual, first_output)
    assert graph_runner.stats()["replays"] == 2
    graph_runner.close()
