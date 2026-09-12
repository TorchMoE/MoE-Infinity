from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
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
    model: object, kv_cache_ratio: float
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


def _make_prompt_batches(
    tokenizer: Any,
    *,
    seed: int,
    num_prompts: int,
    min_len: int,
    max_len: int,
) -> list[list[int]]:
    seed_ids = tokenizer(
        "the quick brown fox jumps over the lazy dog",
        add_special_tokens=False,
    ).input_ids
    token_pool = [int(token_id) for token_id in seed_ids] or [1]
    rng = random.Random(seed)

    prompts: list[list[int]] = []
    for _ in range(num_prompts):
        length = rng.randint(min_len, max_len)
        prompts.append([rng.choice(token_pool) for _ in range(length)])
    return prompts


def _run_batch(
    engine: Any,
    prompt_batches: list[list[int]],
    max_new_tokens: int,
) -> list[list[int]]:
    from moe_infinity.serving.sequence import SamplingParams

    for idx, prompt_ids in enumerate(prompt_batches):
        engine.add_request(
            request_id=f"parity-{idx}",
            prompt_token_ids=list(prompt_ids),
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=max_new_tokens,
            ),
        )

    outputs = engine.run_until_done()
    return [outputs[f"parity-{idx}"] for idx in range(len(prompt_batches))]


@pytest.fixture(scope="module")
def parity_bundle(model_name: str, tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("transformers")
    pytest.importorskip("moe_infinity")

    from transformers import AutoTokenizer

    from moe_infinity import MoE
    from moe_infinity.serving.engine import ContinuousBatchingEngine

    local_model_dir = _require_local_model_dir(model_name)
    offload_path = Path(tmp_path_factory.mktemp("e2e_parity")) / "offload"
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
        pytest.skip(f"Unable to initialize model for parity test: {exc}")

    def build_engine(kv_cache_ratio: float) -> Any:
        return ContinuousBatchingEngine(
            model=moe_model.model,
            engine=moe_model.engine,
            config=_build_engine_config(
                moe_model.model,
                kv_cache_ratio=kv_cache_ratio,
            ),
            tokenizer=tokenizer,
        )

    return tokenizer, build_engine


def test_kv_parity_tokens_identical_under_pressure(parity_bundle) -> None:
    tokenizer, build_engine = parity_bundle
    prompts = _make_prompt_batches(
        tokenizer,
        seed=20260331,
        num_prompts=32,
        min_len=24,
        max_len=96,
    )

    torch.manual_seed(20260331)
    torch.cuda.manual_seed_all(20260331)
    baseline_engine = build_engine(kv_cache_ratio=0.35)
    baseline_tokens = _run_batch(
        baseline_engine,
        prompts,
        max_new_tokens=16,
    )

    torch.manual_seed(20260331)
    torch.cuda.manual_seed_all(20260331)
    pressure_engine = build_engine(kv_cache_ratio=0.05)
    swap_count = [0]
    original_swap_out = pressure_engine.kv_cache.swap_out

    def counting_swap_out(seq_id: int) -> None:
        swap_count[0] += 1
        original_swap_out(seq_id)

    with patch.object(
        pressure_engine.kv_cache,
        "swap_out",
        side_effect=counting_swap_out,
    ):
        pressure_tokens = _run_batch(
            pressure_engine,
            prompts,
            max_new_tokens=16,
        )

    assert len(baseline_tokens) == 32
    assert len(pressure_tokens) == 32
    assert baseline_tokens == pressure_tokens
    assert swap_count[0] > 0


def test_int8_vs_native_decode_logit_and_token_agreement() -> None:
    """Model-free logit/token-agreement gate over >= 2048 decode steps.

    Runs an identical synthetic GQA decode loop through the native and INT8
    attention backends; the INT8 storage path must keep per-step logit cosine
    >= 0.999, mean absolute logit error <= 0.02, and greedy token agreement
    >= 99% across the whole run.
    """
    import math

    from moe_infinity.runtime.attention_backend import PagedAttentionBackend
    from moe_infinity.runtime.attention_types import (
        AttentionMetadata,
        KVCacheSpec,
    )
    from moe_infinity.runtime.kv_cache_format import (
        allocate_layered_paged_kv_store,
    )

    torch.manual_seed(2026)
    device = torch.device("cpu")
    kv_heads, q_heads, head_dim = 4, 8, 32
    block_size, vocab = 16, 64
    total_steps = 2048
    num_blocks = (total_steps + block_size - 1) // block_size + 2
    scale = 1.0 / math.sqrt(head_dim)

    native = PagedAttentionBackend(
        KVCacheSpec(kv_heads, head_dim, torch.float16, block_size),
        num_blocks,
        device,
    )
    int8 = PagedAttentionBackend(
        KVCacheSpec(kv_heads, head_dim, torch.float16, block_size, "int8_sym"),
        num_blocks,
        device,
    )
    int8.bind_store(
        allocate_layered_paged_kv_store(
            owner_id="parity-int8",
            format_name="int8_sym",
            num_layers=1,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            execution_dtype=torch.float16,
            device=device,
        ),
        owner_id="parity-int8",
    )

    proj = torch.randn(q_heads * head_dim, vocab, dtype=torch.float32)
    block_table = torch.arange(num_blocks, dtype=torch.int64).reshape(1, -1)

    cosines: list[float] = []
    abs_errs: list[float] = []
    matches = 0

    for step in range(total_steps):
        key = torch.randn(1, kv_heads, head_dim, dtype=torch.float16)
        value = torch.randn(1, kv_heads, head_dim, dtype=torch.float16)
        query = torch.randn(1, q_heads, head_dim, dtype=torch.float16)
        slot = torch.tensor([step])
        native.write_kv(key, value, slot)
        int8.write_chunk(
            layer_idx=0, key_chunk=key, value_chunk=value, slot_mapping=slot
        )
        seq_len = step + 1
        meta = AttentionMetadata(
            block_tables=block_table,
            seq_lens=torch.tensor([seq_len], dtype=torch.int64),
            max_seq_len=seq_len,
            num_prefill_tokens=0,
            num_decode_tokens=1,
            slot_mapping=slot,
            is_prefill=False,
        )
        native_out = native.forward(
            query, query, query, attention_metadata=meta, scale=scale
        )
        int8_out = int8.forward(
            query,
            query,
            query,
            layer_idx=0,
            attention_metadata=meta,
            scale=scale,
        )
        nl = (native_out[0].float().flatten() @ proj).float()
        il = (int8_out[0].float().flatten() @ proj).float()
        cosines.append(
            float(torch.nn.functional.cosine_similarity(nl, il, dim=0))
        )
        abs_errs.append(float((nl - il).abs().mean()))
        if int(nl.argmax()) == int(il.argmax()):
            matches += 1

    mean_logit_cosine = sum(cosines) / len(cosines)
    mean_absolute_logit_error = sum(abs_errs) / len(abs_errs)
    total_tokens = total_steps
    matching_tokens = matches

    assert total_tokens >= 2048
    assert mean_logit_cosine >= 0.999
    assert mean_absolute_logit_error <= 0.02
    assert matching_tokens / total_tokens >= 0.99
    assert int8.effective_format == "int8_sym"
