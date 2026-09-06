import builtins
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

import moe_infinity.runtime.flashinfer_utils as flashinfer_utils


def _import_flashinfer_available() -> bool:
    try:
        import flashinfer  # noqa: F401

        return True
    except Exception:
        return False


def _load_flashinfer_utils_with_blocked_import() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[3]
        / "moe_infinity"
        / "runtime"
        / "flashinfer_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_flashinfer_utils_missing", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    original_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "flashinfer":
            raise ImportError("blocked flashinfer import for test")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = _guarded_import
    try:
        spec.loader.exec_module(module)
    finally:
        builtins.__import__ = original_import

    return module


def test_has_flashinfer_flag_matches_import() -> None:
    importable = _import_flashinfer_available()
    module = importlib.import_module("moe_infinity.runtime.flashinfer_utils")
    importlib.reload(module)
    assert module.HAS_FLASHINFER is importable


def test_get_workspace_returns_correct_shape_and_dtype() -> None:
    ws = flashinfer_utils.get_workspace(torch.device("cpu"))
    assert ws.dtype == torch.uint8
    assert ws.numel() == 128 * 1024 * 1024
    assert ws.device.type == "cpu"


def test_get_workspace_caches_per_device() -> None:
    d = torch.device("cpu")
    ws1 = flashinfer_utils.get_workspace(d)
    ws2 = flashinfer_utils.get_workspace(d)
    assert ws1 is ws2


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA to validate different-device workspace caching",
)
def test_get_workspace_different_devices() -> None:
    ws_cpu = flashinfer_utils.get_workspace(torch.device("cpu"))
    ws_cuda = flashinfer_utils.get_workspace(torch.device("cuda"))
    assert ws_cpu is not ws_cuda
    assert ws_cpu.device.type == "cpu"
    assert ws_cuda.device.type == "cuda"


def test_graceful_when_flashinfer_missing() -> None:
    module = _load_flashinfer_utils_with_blocked_import()
    assert getattr(module, "HAS_FLASHINFER") is False
    ws = module.get_workspace(torch.device("cpu"))
    assert ws.dtype == torch.uint8
    assert ws.numel() == 128 * 1024 * 1024
