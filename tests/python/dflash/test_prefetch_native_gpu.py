# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Opt-in native-extension smoke test for the batched ``prefetch_tensors`` API.

Task 8 of ``docs/superpowers/plans/2026-08-14-pd-dflash-serving-scheduler.md``
(candidate hop 1). Verifies that the rebuilt ``moe_infinity._store`` exposes the
batched ``prefetch_handle.prefetch_tensors(tensor_ids, priority=1)`` binding and
that it enqueues a saturated ``E_l x L`` block in one call without raising -- the
pre-Task-8 no-op ``prefetch_tensors(request_id, buffer)`` binding would reject a
single positional tensor-id list, so a passing call proves the new native API is
built and wired.

The offloaded target is loaded exactly once via a module-scoped fixture: the
native archer engine keeps process-global topology/task-pool state that does not
survive a second in-process offload load, so each test must share one engine.

Opt-in via ``MOE_DFLASH_SERVING_GPU=1`` with the offloaded target present in the
HF cache. Without the gate this collects and skips cleanly: no CUDA, no model
load, no filesystem, no network at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest
import torch

TARGET_REPO = os.environ.get("MOE_PREFETCH_NATIVE_MODEL", "openai/gpt-oss-20b")


def _hf_home() -> Path:
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        val = os.environ.get(var)
        if val:
            base = Path(val)
            return base / "hub" if var == "XDG_CACHE_HOME" else base
    return Path.home() / ".cache" / "huggingface"


def _checkpoint_present(repo: str) -> bool:
    hub = _hf_home()
    hub = hub if hub.name == "hub" else hub / "hub"
    return (hub / f"models--{repo.replace('/', '--')}").is_dir()


def _skip_reason() -> Optional[str]:
    if not os.environ.get("MOE_DFLASH_SERVING_GPU"):
        return "MOE_DFLASH_SERVING_GPU unset (opt-in native prefetch smoke)"
    if not torch.cuda.is_available():
        return "CUDA unavailable (native prefetch smoke)"
    if not _checkpoint_present(TARGET_REPO):
        return f"checkpoint not present in $HF_HOME: {TARGET_REPO}"
    return None


SKIP_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(
    SKIP_REASON is not None, reason=SKIP_REASON or "gpu-gated"
)


_LOADED_MODEL: Optional[object] = None


@pytest.fixture(scope="module")
def offloaded_prefetcher():
    global _LOADED_MODEL
    from moe_infinity import MoE

    offload = os.environ.get(
        "MOE_PREFETCH_NATIVE_OFFLOAD", "/tmp/opencode/moe-offload/gpt-oss-20b"
    )
    os.makedirs(offload, exist_ok=True)
    ratio = float(os.environ.get("MOE_DFLASH_MEM_RATIO", "0.2"))
    model = MoE(
        TARGET_REPO,
        {"offload_path": offload, "device_memory_ratio": ratio},
    )
    _LOADED_MODEL = model
    prefetcher = model.engine.expert_prefetcher
    assert prefetcher is not None and prefetcher.archer_engine is not None
    yield prefetcher


def _run_one_token(prefetcher) -> None:
    model = _LOADED_MODEL
    assert model is not None
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TARGET_REPO)
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    model.generate(
        inputs.input_ids,
        max_new_tokens=1,
        do_sample=False,
    )


def _saturated_ids(prefetcher) -> list[int]:
    return [tid for _key, tid in sorted(prefetcher.expert_tensor_map.items())]


def test_native_batched_prefetch_tensors_issues_saturated_block(
    offloaded_prefetcher,
) -> None:
    engine = offloaded_prefetcher.archer_engine
    tensor_ids = _saturated_ids(offloaded_prefetcher)
    assert tensor_ids, "no offloaded expert tensors to issue"

    assert engine.prefetch_tensors(tensor_ids) is None
    assert engine.prefetch_tensors(tensor_ids, 1) is None


def test_native_batched_prefetch_tensors_empty_is_noop(
    offloaded_prefetcher,
) -> None:
    assert offloaded_prefetcher.archer_engine.prefetch_tensors([]) is None


def test_native_batched_prefetch_experts_list_uses_batched_path(
    offloaded_prefetcher,
) -> None:
    layers = sorted(
        {layer for layer, _e in offloaded_prefetcher.expert_tensor_map}
    )
    some_layer = layers[0]
    experts = sorted(
        expert
        for layer, expert in offloaded_prefetcher.expert_tensor_map
        if layer == some_layer
    )
    offloaded_prefetcher.prefetch_experts_list(some_layer, experts)


def test_native_priority_bands_accept_each_service_class_reverse_order(
    offloaded_prefetcher,
) -> None:
    from moe_infinity.memory.expert_prefetcher import (
        BACKGROUND_PREFETCH_PRIORITY,
        ON_DEMAND_PRIORITY,
        ROUTE_AHEAD_PRIORITY,
    )

    engine = offloaded_prefetcher.archer_engine
    tensor_ids = _saturated_ids(offloaded_prefetcher)
    assert tensor_ids, "no offloaded expert tensors to issue"

    for priority in (
        BACKGROUND_PREFETCH_PRIORITY,
        ROUTE_AHEAD_PRIORITY,
        ON_DEMAND_PRIORITY,
    ):
        assert engine.prefetch_tensors(tensor_ids, priority) is None
        torch.cuda.synchronize()


def test_native_reset_cache_is_non_terminal(offloaded_prefetcher) -> None:
    engine = offloaded_prefetcher.archer_engine
    tensor_ids = _saturated_ids(offloaded_prefetcher)
    assert tensor_ids, "no offloaded expert tensors to issue"

    assert engine.reset_cache() is None
    torch.cuda.synchronize()

    assert engine.prefetch_tensors(tensor_ids, 1) is None
    torch.cuda.synchronize()
    assert engine.reset_cache() is None
    torch.cuda.synchronize()


def test_native_route_ahead_priority_knob_issues_each_band(
    offloaded_prefetcher,
) -> None:
    from moe_infinity.memory.expert_prefetcher import (
        BACKGROUND_PREFETCH_PRIORITY,
        ON_DEMAND_PRIORITY,
        ROUTE_AHEAD_PRIORITY,
    )

    layers = sorted(
        {layer for layer, _e in offloaded_prefetcher.expert_tensor_map}
    )
    some_layer = layers[0]
    experts = sorted(
        expert
        for layer, expert in offloaded_prefetcher.expert_tensor_map
        if layer == some_layer
    )
    original = offloaded_prefetcher.route_ahead_priority
    try:
        for band in (
            BACKGROUND_PREFETCH_PRIORITY,
            ROUTE_AHEAD_PRIORITY,
            ON_DEMAND_PRIORITY,
        ):
            offloaded_prefetcher.route_ahead_priority = band
            offloaded_prefetcher.prefetch_experts_list(some_layer, experts)
            torch.cuda.synchronize()
    finally:
        offloaded_prefetcher.route_ahead_priority = original


def test_native_compute_samples_are_drainable(offloaded_prefetcher) -> None:
    dispatcher = offloaded_prefetcher.expert_dispatcher
    dispatcher.set_overlap_compute_timing_enabled(True)
    assert dispatcher.drain_compute_samples() == []
    _run_one_token(offloaded_prefetcher)
    samples = dispatcher.drain_compute_samples()
    assert samples
    assert all(
        s.invocation_id > 0
        and s.kernel_end_offset_ns > s.kernel_start_offset_ns
        and s.kernel_duration_ns
        == s.kernel_end_offset_ns - s.kernel_start_offset_ns
        and s.forward_return_host_ns >= 0
        and s.output_complete_host_ns >= s.forward_return_host_ns
        and s.output_delay_ns
        == s.output_complete_host_ns - s.forward_return_host_ns
        and s.layer_id >= 0
        for s in samples
    )
    assert len({s.invocation_id for s in samples}) == 1
    assert dispatcher.drain_compute_samples() == []
