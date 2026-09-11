"""Marlin W4A16 (INT4 weight, FP16 activation) GEMM wrapper.

Bundled from IST-DASLab/marlin (Apache 2.0). Provides weight packing utilities
and a Python interface to the native CUDA kernel.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import moe_infinity._marlin as _C

    _MARLIN_AVAILABLE = True
except ImportError:
    _MARLIN_AVAILABLE = False


def is_marlin_available() -> bool:
    return _MARLIN_AVAILABLE


def marlin_supports_shape(
    k: int,
    n: int,
    groupsize: int,
    extension_available: bool | None = None,
) -> bool:
    available = (
        _MARLIN_AVAILABLE
        if extension_available is None
        else extension_available
    )
    return bool(
        available and groupsize in (-1, 128) and k % 128 == 0 and n % 256 == 0
    )


def _get_perms() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    perm: list[int] = []
    for i in range(32):
        perm1: list[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm_tensor = torch.tensor(perm, dtype=torch.int32)

    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_tensor = torch.tensor(scale_perm, dtype=torch.int32)

    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in range(2)] * 4)
    scale_perm_single_tensor = torch.tensor(
        scale_perm_single, dtype=torch.int32
    )

    return perm_tensor, scale_perm_tensor, scale_perm_single_tensor


_PERM, _SCALE_PERM, _SCALE_PERM_SINGLE = _get_perms()


def pack_marlin_weight(
    weight: torch.Tensor,
    scales: torch.Tensor,
    groupsize: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack FP16 quantized weight + scales into Marlin format.

    Args:
        weight: INT32 quantized weight [K, N] with values in [0, 15]
        scales: FP16 scales [K//groupsize, N] or [1, N]
        groupsize: -1 (per-column) or 128

    Returns:
        (packed_weight, packed_scales) in Marlin layout
    """
    assert groupsize in (-1, 128)

    K, N = weight.shape
    assert K % 128 == 0, f"K={K} must be divisible by 128"
    assert N % 256 == 0, f"N={N} must be divisible by 256"

    tile = 16
    perm = _PERM.to(weight.device)

    w = weight.reshape(K // tile, tile, N // tile, tile)
    w = w.permute(0, 2, 1, 3).contiguous()
    w = w.reshape(K // tile, N * tile)

    w = w.reshape(-1, perm.numel())[:, perm].reshape(w.shape)

    packed = torch.zeros(
        (w.shape[0], w.shape[1] // 8), dtype=torch.int32, device=w.device
    )
    for shift in range(8):
        packed |= (w[:, shift::8].to(torch.int32) & 0xF) << (4 * shift)

    if groupsize == -1:
        s_perm = _SCALE_PERM_SINGLE.to(scales.device)
    else:
        s_perm = _SCALE_PERM.to(scales.device)

    s = scales.reshape(-1, s_perm.numel())[:, s_perm]
    s = s.reshape(-1, N).contiguous()

    return packed, s


def marlin_quantize(
    weight: torch.Tensor,
    groupsize: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16 weight to Marlin INT4 format.

    Args:
        weight: FP16 weight [K, N]
        groupsize: -1 (per-column) or 128

    Returns:
        (packed_weight, scales) ready for marlin_gemm
    """
    assert groupsize in (-1, 128)
    K, N = weight.shape
    assert K % 128 == 0, f"K={K} must be divisible by 128"
    assert N % 256 == 0, f"N={N} must be divisible by 256"

    maxq = 15

    if groupsize == -1:
        scales = weight.abs().amax(dim=0, keepdim=True) / (maxq / 2)
        scales = scales.clamp(min=1e-10)
        w_int = torch.round(weight / scales).int() + (maxq + 1) // 2
    else:
        n_groups = K // groupsize
        w_reshaped = weight.reshape(n_groups, groupsize, N)
        scales = w_reshaped.abs().amax(dim=1) / (maxq / 2)
        scales = scales.clamp(min=1e-10)
        w_int = (
            torch.round(w_reshaped / scales.unsqueeze(1)).int()
            + (maxq + 1) // 2
        )
        w_int = w_int.reshape(K, N)

    w_int = w_int.clamp(0, maxq)
    scales = scales.to(torch.float16)

    return pack_marlin_weight(w_int, scales, groupsize)


def prepare_workspace(prob_n: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(prob_n // 128 * 16, dtype=torch.int32, device=device)


def marlin_gemm(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Marlin FP16 x INT4 GEMM.

    Args:
        input: [M, K] float16
        packed_weight: Marlin-packed INT4 weights
        scales: Packed FP16 scales
        workspace: From prepare_workspace()

    Returns:
        [M, N] float16 output
    """
    if not _MARLIN_AVAILABLE:
        raise RuntimeError(
            "moe_infinity._marlin not available. "
            "Rebuild with CUDA support: pip install -e ."
        )

    M, K = input.shape
    N = scales.shape[1]

    output = torch.empty((M, N), dtype=torch.float16, device=input.device)

    _C.mul(
        input.to(torch.float16).contiguous(),
        packed_weight.contiguous(),
        output,
        scales.contiguous(),
        workspace.contiguous(),
        -1,
        -1,
        -1,
        16,
    )

    return output


def reference_dequant_gemm(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    K: int,
    N: int,
    groupsize: int = -1,
) -> torch.Tensor:
    """Reference implementation: unpack INT4, dequantize, matmul."""
    w_int = torch.zeros((K, N), dtype=torch.int32, device=packed_weight.device)
    for shift in range(8):
        w_int[shift::8, :] = (packed_weight >> (4 * shift)) & 0xF

    perm = _PERM.to(w_int.device)
    inv_perm = torch.argsort(perm)
    w_int = w_int.reshape(-1, perm.numel())[:, inv_perm].reshape(K, N)

    w_fp = w_int.float() - 8.0

    if groupsize == -1:
        w_fp = w_fp * scales.float()
    else:
        n_groups = K // groupsize
        w_fp = w_fp.reshape(n_groups, groupsize, N) * scales.float().unsqueeze(
            1
        )
        w_fp = w_fp.reshape(K, N)

    return input.float() @ w_fp.float()
