from __future__ import annotations

import gc
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for E2E tests",
    ),
]


def _resolve_int_attr(config: object, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    return None


def _resolve_dtype(model: object) -> torch.dtype:
    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype):
        return model_dtype

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            parameter_source = parameters()
            if isinstance(parameter_source, Iterator):
                first_param = next(parameter_source, None)
            elif isinstance(parameter_source, Iterable):
                first_param = next(iter(parameter_source), None)
            else:
                first_param = None
        except Exception:
            first_param = None
        if isinstance(first_param, torch.Tensor):
            return first_param.dtype

    cfg = getattr(model, "config", None)
    cfg_dtype = getattr(cfg, "torch_dtype", None)
    if isinstance(cfg_dtype, torch.dtype):
        return cfg_dtype
    if isinstance(cfg_dtype, str):
        mapping = {
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        normalized = cfg_dtype.replace("torch.", "")
        if normalized in mapping:
            return mapping[normalized]

    return torch.float16


def _build_engine_config(
    model: object, kv_cache_ratio: float, adaptive: bool = False
) -> dict[str, object]:
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise RuntimeError("model.config is required")

    num_layers = _resolve_int_attr(
        model_config,
        "num_hidden_layers",
        "num_layers",
        "n_layer",
    )
    num_attention_heads = _resolve_int_attr(
        model_config,
        "num_attention_heads",
        "n_head",
    )
    num_kv_heads = _resolve_int_attr(
        model_config,
        "num_key_value_heads",
        "num_kv_heads",
        "n_head_kv",
    )
    hidden_size = _resolve_int_attr(model_config, "hidden_size", "n_embd")
    head_dim = _resolve_int_attr(model_config, "head_dim")
    eos_token_id = _resolve_int_attr(model_config, "eos_token_id")

    if num_layers is None or num_attention_heads is None:
        raise RuntimeError("unable to resolve model layer/head config")
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads
    if head_dim is None:
        if hidden_size is None:
            raise RuntimeError("unable to resolve model head_dim")
        head_dim = hidden_size // max(1, num_attention_heads)

    config: dict[str, object] = {
        "device_memory_ratio": 0.75,
        "kv_cache_ratio": kv_cache_ratio,
        "max_batch_size": 64,
        "max_tokens_per_step": 4096,
        "block_size": 16,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": _resolve_dtype(model),
        "adaptive_memory_enabled": adaptive,
        "adaptive_memory_interval_steps": 1,
    }
    if isinstance(eos_token_id, int):
        config["eos_token_id"] = eos_token_id
    return config


def _require_local_model_dir(model_name: str) -> str:
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            ignore_patterns=["flax*", "tf*"],
        )
    except Exception as exc:
        pytest.skip(f"Model not cached locally ({model_name}): {exc}")


def _make_prompt_batches(tokenizer: Any, *, seed: int) -> list[list[int]]:
    seed_ids = tokenizer(
        "the quick brown fox jumps over the lazy dog",
        add_special_tokens=False,
    ).input_ids
    token_pool = [int(token_id) for token_id in seed_ids] or [1]
    rng = random.Random(seed)

    prompt_batches: list[list[int]] = []
    for _ in range(24):
        prompt_batches.append([rng.choice(token_pool) for _ in range(384)])
    return prompt_batches


def _run_batch(
    engine: Any, prompt_batches: list[list[int]]
) -> dict[str, list[int]]:
    from moe_infinity.serving.sequence import SamplingParams

    for idx, prompt_ids in enumerate(prompt_batches):
        engine.add_request(
            request_id=f"mem-{idx}",
            prompt_token_ids=list(prompt_ids),
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=8,
            ),
        )
    return engine.run_until_done()


@pytest.fixture(scope="module")
def memory_bundle(model_name: str, tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("transformers")
    pytest.importorskip("moe_infinity")

    from transformers import AutoTokenizer

    from moe_infinity import MoE
    from moe_infinity.serving.engine import ContinuousBatchingEngine

    local_model_dir = _require_local_model_dir(model_name)
    offload_path = Path(tmp_path_factory.mktemp("e2e_memory")) / "offload"
    offload_path.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        moe_model = MoE(
            local_model_dir,
            {
                "offload_path": str(offload_path),
                "device_memory_ratio": 0.75,
            },
        )
    except Exception as exc:
        pytest.skip(f"Unable to initialize model for memory test: {exc}")

    def build_engine(kv_cache_ratio: float, adaptive: bool = False) -> Any:
        return ContinuousBatchingEngine(
            model=moe_model.model,
            engine=moe_model.engine,
            config=_build_engine_config(
                moe_model.model,
                kv_cache_ratio=kv_cache_ratio,
                adaptive=adaptive,
            ),
            tokenizer=tokenizer,
        )

    return tokenizer, build_engine


def _measure_peak(
    build_engine: Any,
    prompt_batches: list[list[int]],
    kv_cache_ratio: float,
) -> tuple[int, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    engine = build_engine(kv_cache_ratio=kv_cache_ratio)
    swap_count = [0]
    original_swap_out = engine.kv_cache.swap_out

    def counting_swap_out(seq_id: int) -> None:
        swap_count[0] += 1
        original_swap_out(seq_id)

    with patch.object(
        engine.kv_cache,
        "swap_out",
        side_effect=counting_swap_out,
    ):
        _run_batch(engine, prompt_batches)

    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_delta = max(0, peak_allocated - baseline_allocated)

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return peak_delta, swap_count[0]


def test_kv_memory_peak_reduced_under_offload(memory_bundle) -> None:
    tokenizer, build_engine = memory_bundle
    prompts = _make_prompt_batches(tokenizer, seed=20260331)

    torch.manual_seed(20260331)
    torch.cuda.manual_seed_all(20260331)
    baseline_peak, baseline_swaps = _measure_peak(
        build_engine,
        prompts,
        kv_cache_ratio=0.35,
    )

    torch.manual_seed(20260331)
    torch.cuda.manual_seed_all(20260331)
    offload_peak, offload_swaps = _measure_peak(
        build_engine,
        prompts,
        kv_cache_ratio=0.05,
    )

    assert offload_swaps > 0
    assert offload_peak <= baseline_peak, (
        f"Expected offload peak <= baseline peak, got {offload_peak} > "
        f"{baseline_peak} (baseline_swaps={baseline_swaps}, "
        f"offload_swaps={offload_swaps})"
    )


@dataclass(frozen=True)
class PressureResult:
    output_token_ids: list[int]
    hard_budget_violations: int
    minimum_capacity_violations: int
    min_free_gpu_bytes: int
    configured_reserve_bytes: int
    resize_count: int
    max_resize_count: int
    completed: bool
    fallback_static: bool
    resize_failures: int
    failure_limit: int


def run_pressure(
    memory_bundle: tuple[Any, Any], *, adaptive: bool, seed: int
) -> PressureResult:
    tokenizer, build_engine = memory_bundle
    engine = build_engine(kv_cache_ratio=0.25, adaptive=adaptive)
    outputs = _run_batch(engine, _make_prompt_batches(tokenizer, seed=seed))
    adaptive_stats = engine.get_stats()["memory"]["adaptive"]
    devices = adaptive_stats["devices"]
    return PressureResult(
        output_token_ids=[
            token for request in sorted(outputs) for token in outputs[request]
        ],
        hard_budget_violations=sum(
            int(item.get("hard_budget_violations", 0))
            for item in devices.values()
        ),
        minimum_capacity_violations=sum(
            int(item.get("minimum_capacity_violations", 0))
            for item in devices.values()
        ),
        min_free_gpu_bytes=min(
            int(item.get("min_free_gpu_bytes", 0)) for item in devices.values()
        ),
        configured_reserve_bytes=min(
            int(item.get("configured_reserve_bytes", 0))
            for item in devices.values()
        ),
        resize_count=sum(
            int(item.get("resize_count", 0)) for item in devices.values()
        ),
        max_resize_count=sum(
            int(item.get("max_resize_count", 0)) for item in devices.values()
        ),
        completed=bool(adaptive_stats["completed"]),
        fallback_static=any(
            bool(item.get("fallback_static", False))
            for item in devices.values()
        ),
        resize_failures=sum(
            int(item.get("resize_failures", 0)) for item in devices.values()
        ),
        failure_limit=int(adaptive_stats["failure_limit"]),
    )


def test_adaptive_pressure_preserves_reserve_and_outputs(memory_bundle) -> None:
    fixed = run_pressure(memory_bundle, adaptive=False, seed=11)
    adaptive = run_pressure(memory_bundle, adaptive=True, seed=11)
    assert adaptive.output_token_ids == fixed.output_token_ids
    assert adaptive.hard_budget_violations == 0
    assert adaptive.minimum_capacity_violations == 0
    assert adaptive.min_free_gpu_bytes >= adaptive.configured_reserve_bytes
    assert adaptive.resize_count <= adaptive.max_resize_count


def test_injected_resize_oom_falls_back_to_static(memory_bundle) -> None:
    _, build_engine = memory_bundle
    engine = build_engine(kv_cache_ratio=0.25, adaptive=True)
    if not engine._memory_resizers:
        pytest.skip(
            "serving expert mutation adapter unavailable in model fixture"
        )


def test_int8_storage_bytes_and_ratio_match_descriptor() -> None:
    from moe_infinity.runtime.kv_cache_format import (
        KVCacheFormat,
        allocate_layered_paged_kv_store,
    )

    block_size, kv_heads, head_dim = 16, 8, 128
    int8_page = KVCacheFormat.parse("int8_sym").page_size_bytes(
        block_size=block_size, num_kv_heads=kv_heads, head_dim=head_dim
    )
    native_page = KVCacheFormat.parse("native").page_size_bytes(
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        execution_dtype=torch.float16,
    )
    assert int8_page == 33_280
    assert native_page == 65_536
    assert int8_page / native_page == pytest.approx(0.5078125)
    assert int8_page / native_page <= 0.52

    num_layers, num_blocks = 2, 4
    store = allocate_layered_paged_kv_store(
        owner_id="mem-int8",
        format_name="int8_sym",
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        execution_dtype=torch.float16,
        device=torch.device("cpu"),
    )
    assert store.nbytes == num_layers * num_blocks * int8_page
