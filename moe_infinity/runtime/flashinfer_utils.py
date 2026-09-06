from __future__ import annotations

from typing import Final, Optional

import torch

try:
    import flashinfer as _flashinfer

    _has_flashinfer = True
except Exception:
    _flashinfer = None
    _has_flashinfer = False


HAS_FLASHINFER: Final[bool] = _has_flashinfer


# Shared prefill+decode scratch: chunked append-attention split-KV scratch
# (tmp_s/LSE/partial-O) grows with query rows and KV pages, overflowing the
# 128 MiB FlashInfer example default on chunk 2; 256 MiB gives ~2x headroom.
_WORKSPACE_SIZE_BYTES = 256 * 1024 * 1024
_WORKSPACE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _device_key(device: torch.device) -> tuple[str, int | None]:
    return (device.type, device.index)


def get_workspace(device: torch.device) -> torch.Tensor:
    key = _device_key(device)
    workspace = _WORKSPACE_CACHE.get(key)
    if workspace is None:
        workspace = torch.empty(
            _WORKSPACE_SIZE_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        _WORKSPACE_CACHE[key] = workspace
    return workspace


def get_flashinfer_module() -> Optional[object]:
    return _flashinfer
