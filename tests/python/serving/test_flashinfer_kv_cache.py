# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false, reportMissingParameterType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

KV_CACHE_PATH = Path(ROOT) / "moe_infinity" / "serving" / "kv_cache.py"


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
        return torch.full_like(query, 7.0)


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
        return torch.full_like(query, 11.0)


def _load_module(module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, KV_CACHE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_compute_attention_uses_flashinfer_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kv_cache_module = _load_module("test_flashinfer_serving_kv_cache_uses_fi")
    fake_module = type(
        "_FakeFlashInferModule",
        (),
        {
            "BatchPrefillWithPagedKVCacheWrapper": _FakePrefillWrapper,
            "BatchDecodeWithPagedKVCacheWrapper": _FakeDecodeWrapper,
        },
    )

    monkeypatch.setattr(
        kv_cache_module.flashinfer_utils, "HAS_FLASHINFER", True
    )
    monkeypatch.setattr(
        kv_cache_module.flashinfer_utils,
        "get_flashinfer_module",
        lambda: fake_module,
    )

    def _fake_workspace(device: torch.device) -> torch.Tensor:
        return torch.empty(1024, dtype=torch.uint8, device=device)

    monkeypatch.setattr(
        kv_cache_module.flashinfer_utils,
        "get_workspace",
        _fake_workspace,
    )

    cache = kv_cache_module.PagedKVCache(
        num_blocks=4,
        block_size=4,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    prefill_out = cache._compute_attention(
        query=torch.randn(1, 2, 3, 8),
        key=torch.randn(1, 2, 3, 8),
        value=torch.randn(1, 2, 3, 8),
        is_causal=True,
        layer_idx=1,
    )
    decode_out = cache._compute_attention(
        query=torch.randn(1, 2, 1, 8),
        key=torch.randn(1, 2, 4, 8),
        value=torch.randn(1, 2, 4, 8),
        is_causal=True,
        layer_idx=1,
    )

    assert torch.all(prefill_out == 7.0)
    assert torch.all(decode_out == 11.0)
    assert cache._fi_prefill is not None
    assert cache._fi_decode is not None

    prefill_plan_args = cache._fi_prefill.plan_args[0]
    assert prefill_plan_args[0].dtype == torch.int32
    assert prefill_plan_args[1].dtype == torch.int32
    assert prefill_plan_args[2].dtype == torch.int32
    assert prefill_plan_args[3].dtype == torch.int32

    decode_plan_args = cache._fi_decode.plan_args[0]
    assert decode_plan_args[0].dtype == torch.int32
    assert decode_plan_args[1].dtype == torch.int32
    assert decode_plan_args[2].dtype == torch.int32

    assert cache._fi_prefill.run_args is not None
    assert cache._fi_decode.run_args is not None
    assert cache._fi_prefill.run_args[1].shape == cache._kv_cache[1].shape
    assert cache._fi_decode.run_args[1].shape == cache._kv_cache[1].shape
    torch.testing.assert_close(
        cache._fi_prefill.run_args[1], cache._kv_cache[1]
    )
    torch.testing.assert_close(cache._fi_decode.run_args[1], cache._kv_cache[1])


def test_compute_attention_falls_back_without_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kv_cache_module = _load_module("test_flashinfer_serving_kv_cache_fallback")
    monkeypatch.setattr(
        kv_cache_module.flashinfer_utils,
        "HAS_FLASHINFER",
        False,
    )

    cache = kv_cache_module.PagedKVCache(
        num_blocks=4,
        block_size=4,
        num_layers=1,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    query = torch.randn(1, 2, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn(1, 2, 3, 8)

    output = cache._compute_attention(query=query, key=key, value=value)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
    )
    torch.testing.assert_close(output, expected)


def test_kv_cache_tensor_layout_compatible_with_flashinfer() -> None:
    kv_cache_module = _load_module("test_flashinfer_serving_kv_cache_layout")
    cache = kv_cache_module.PagedKVCache(
        num_blocks=8,
        block_size=16,
        num_layers=3,
        num_heads=4,
        head_dim=32,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )

    assert cache._kv_cache.shape == (3, 8, 2, 16, 4, 32)
    assert cache._kv_cache[0].shape == (8, 2, 16, 4, 32)
