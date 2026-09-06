from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MOE_PREFIX_CACHE_CUDA") != "1",
    reason="set MOE_PREFIX_CACHE_CUDA=1 on a Qwen3/FlashInfer CUDA runner",
)

MODEL = "Qwen/Qwen3-30B-A3B"


@pytest.fixture
def qwen3_engine_factory(tmp_path: Path):
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from moe_infinity.entrypoints.big_modeling import MoE
    from moe_infinity.entrypoints.openai.api_server_v2 import (
        _build_engine_config,
    )
    from moe_infinity.serving.engine import ContinuousBatchingEngine

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    def factory(
        *, enable_prefix_caching: bool, prefix_cache_max_entries: int = 1000
    ):
        owner = MoE(
            MODEL,
            {
                "offload_path": str(
                    tmp_path
                    / ("enabled" if enable_prefix_caching else "disabled")
                ),
                "device_memory_ratio": 0.5,
            },
        )
        args = SimpleNamespace(
            device_memory_ratio=0.5,
            kv_cache_ratio=0.25,
            max_batch_size=4,
            enable_prefix_caching=enable_prefix_caching,
            prefix_cache_max_entries=prefix_cache_max_entries,
        )
        config = _build_engine_config(args, owner.model)
        engine = ContinuousBatchingEngine(
            owner.model, owner.engine, config, tokenizer=tokenizer
        )
        if enable_prefix_caching:
            assert engine.get_stats()["prefix_cache_active"] is True
        return engine

    return factory, tokenizer


def tokenize_to_at_least_64_tokens(tokenizer) -> list[int]:
    ids = tokenizer.encode(
        "Layer-complete shared prefix for exact KV reuse. ",
        add_special_tokens=False,
    )
    return (ids * ((64 + len(ids) - 1) // len(ids)))[:64]


def unrelated_64_tokens(tokenizer) -> list[int]:
    ids = tokenizer.encode(
        "Unrelated eviction pressure sequence. ", add_special_tokens=False
    )
    return (ids * ((64 + len(ids) - 1) // len(ids)))[:64]


def run_with_logits(engine, request_id: str, prompt: list[int]):
    from moe_infinity.serving.sequence import SamplingParams

    captured: list[torch.Tensor] = []
    plans: list[object] = []
    original = engine._execute_batch

    def capture(batch):
        logits = original(batch)
        captured.append(
            engine._extract_last_token_logits(logits, batch)
            .detach()
            .float()
            .cpu()
        )
        plans.append(
            engine.model_runner._get_attention_backend().last_flashinfer_plan
        )
        return logits

    engine._execute_batch = capture
    try:
        engine.add_request(
            request_id, prompt, SamplingParams(temperature=0.0, max_tokens=4)
        )
        return (
            cast(list[int], engine.run_until_done()[request_id]),
            torch.cat(captured),
            plans,
        )
    finally:
        engine._execute_batch = original


def test_real_flashinfer_warm_suffix_matches_cold_logits(
    tmp_path: Path,
) -> None:
    """Warm-vs-cold parity via the subprocess benchmark harness.

    Loading two engines in one process trips the single-owner
    kTopologyHandle guard, so each mode runs in its own subprocess. Cold and
    warm execute different FlashInfer kernel schedules (full prefill vs
    append), which are individually correct but not bitwise-identical
    (tests/python/integration/test_flashinfer_kernel_parity.py), and expert
    accumulation order is nondeterministic, so near-tie argmax swaps are
    tolerated; only disagreement beyond the shared top-2 pair fails.
    """
    import json
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "suite.json"
    cmd = [
        sys.executable,
        str(root / "benchmarks" / "serving" / "prefix_cache_benchmark.py"),
        "--model",
        os.environ.get("MOE_PREFIX_PARITY_MODEL", MODEL),
        "--offload-dir",
        os.environ.get("MOE_PREFIX_PARITY_OFFLOAD", str(tmp_path / "offload")),
        "--shared-prefix-tokens",
        "64",
        "--suffix-tokens",
        "8",
        "--output-json",
        str(output),
        "--parity-report",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=root, env=env)
    assert proc.returncode == 0
    payload = json.loads(output.read_text())

    modes = payload["modes"]
    assert (
        modes["disabled"]["token_digest"]
        == modes["enabled_cold"]["token_digest"]
    ), "disabled and cold share one kernel path and must match bitwise"

    warm = modes["enabled_warm"]
    assert warm["prefix_cache_active"] is True
    assert warm["hits_total"] >= 1
    assert warm["geometry"]["query_offsets"] == [0, 8]
    assert warm["geometry"]["context_lengths"] == [64]
    assert warm["geometry"]["kv_seq_lengths"] == [72]

    parity = payload["parity"]
    assert parity["logits_max_abs"]["disabled_vs_cold"] == 0.0
    if parity["first_flip_step"] is not None:
        cold_top2 = parity["cold_flip_top2"]["token_ids"]
        warm_top2 = parity["warm_flip_top2"]["token_ids"]
        assert cold_top2[0] in warm_top2 and warm_top2[0] in cold_top2, (
            "warm and cold disagree beyond a near-tie candidate swap, "
            "which indicates a real warm-path bug, not numerical noise"
        )


def test_active_reference_survives_lru_eviction(qwen3_engine_factory) -> None:
    from moe_infinity.serving.sequence import SamplingParams

    factory, tokenizer = qwen3_engine_factory
    baseline = factory(enable_prefix_caching=False)
    engine = factory(enable_prefix_caching=True, prefix_cache_max_entries=4)
    shared = tokenize_to_at_least_64_tokens(tokenizer)
    expected, _, _ = run_with_logits(baseline, "expected", shared + [200])
    run_with_logits(engine, "prime", shared + [199])
    engine.add_request(
        "active", shared + [200], SamplingParams(temperature=0.0, max_tokens=4)
    )
    engine.step()
    active_seq = engine._request_to_seq_ids["active"][0]
    shared_block = engine.kv_cache.get_block_table(active_seq)[0]
    before = engine.kv_cache.block_allocator.ref_count(shared_block)
    engine.add_request(
        "pressure",
        unrelated_64_tokens(tokenizer),
        SamplingParams(temperature=0.0, max_tokens=1),
    )
    engine.step()
    during_active = engine.kv_cache.block_allocator.ref_count(shared_block)
    outputs = engine.run_until_done()
    after_eviction = engine.kv_cache.block_allocator.ref_count(shared_block)
    assert outputs["active"] == expected
    assert before >= 2
    assert during_active == 1
    assert after_eviction == 0
