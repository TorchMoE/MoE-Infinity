# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""GPU transfer benchmark; formats are capability-probed at runtime.

This module measures the *real* host-to-device (H2D) cost of staging expert
representations. The canonical, static-low, and adaptive arms differ only in
which stored representation (BF16 vs a low-bit format) is fetched for an
expert, so the physically meaningful quantity that separates the arms is the
H2D payload and the event-timed transfer duration for each representation.

Two responsibilities live here:

* :func:`measure_h2d_bandwidth` performs actual pinned-host -> device copies
  timed with CUDA events and reports measured bytes/second per storage dtype.
  It probes the device at runtime and skips formats the device cannot host.
* :func:`aligned_bytes` and :func:`format_storage_dtype` describe how a stored
  representation is accounted, matching the KAIO alignment used by the native
  Archer store and the manifest schema.

No number returned by :func:`measure_h2d_bandwidth` is fabricated: every value
is produced by a real GPU copy on the device the benchmark runs on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from moe_infinity.runtime.expert_precision import ExpertFormat

ARMS = ("canonical", "static_low", "adaptive")

# KAIO alignment used by the native store and the derivative manifest schema.
_KAIO_ALIGNMENT = 4096

# Bytes-per-element must match the storage dtype the manifest schema records
# for each format; changing these silently corrupts H2D accounting.
_FORMAT_ELEMENT_BYTES: Dict[ExpertFormat, float] = {
    ExpertFormat.BF16: 2.0,
    ExpertFormat.FP8_E4M3_BLOCK128: 1.0,
    ExpertFormat.MARLIN_INT4_GROUP128: 0.5,
}

# The canonical (reference) representation and the fixed static-low
# representation used by the static_low arm.
CANONICAL_FORMAT = ExpertFormat.BF16
STATIC_LOW_FORMAT = ExpertFormat.FP8_E4M3_BLOCK128


def align_up(value: int, alignment: int = _KAIO_ALIGNMENT) -> int:
    if value < 0:
        raise ValueError("cannot align a negative size")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def format_element_bytes(fmt: ExpertFormat) -> float:
    try:
        return _FORMAT_ELEMENT_BYTES[fmt]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unsupported storage format: {fmt}") from exc


def payload_bytes(numel: int, fmt: ExpertFormat) -> int:
    """Raw (unaligned) payload size of ``numel`` weights stored as ``fmt``."""
    if numel < 0:
        raise ValueError("numel must be nonnegative")
    return int(round(numel * format_element_bytes(fmt)))


def aligned_bytes(numel: int, fmt: ExpertFormat) -> int:
    """KAIO-aligned resident size of ``numel`` weights stored as ``fmt``."""
    return align_up(payload_bytes(numel, fmt))


@dataclass(frozen=True)
class TransferMeasurement:
    """Measured H2D characteristics for one storage dtype."""

    format: str
    bytes_transferred: int
    seconds: float
    bytes_per_second: float
    available: bool
    skip_reason: Optional[str] = None


def _torch_storage_dtype(fmt: ExpertFormat):
    import torch

    mapping = {
        ExpertFormat.BF16: torch.bfloat16,
        ExpertFormat.FP8_E4M3_BLOCK128: getattr(torch, "float8_e4m3fn", None),
        ExpertFormat.MARLIN_INT4_GROUP128: torch.uint8,
    }
    return mapping.get(fmt)


def measure_h2d_bandwidth(
    numel: int,
    formats=(CANONICAL_FORMAT, STATIC_LOW_FORMAT),
    *,
    iters: int = 20,
    warmup: int = 3,
    device: str = "cuda",
) -> Dict[str, TransferMeasurement]:
    """Event-time real pinned-host -> device copies for each format.

    Returns a mapping from format value to :class:`TransferMeasurement`. A
    format the current device cannot host (unknown storage dtype, allocation
    failure) is reported with ``available=False`` and a ``skip_reason`` rather
    than raising, mirroring the runtime capability probe requirement.
    """
    import torch

    if not torch.cuda.is_available():
        return {
            fmt.value: TransferMeasurement(
                format=fmt.value,
                bytes_transferred=0,
                seconds=0.0,
                bytes_per_second=0.0,
                available=False,
                skip_reason="cuda_unavailable",
            )
            for fmt in formats
        }

    results: Dict[str, TransferMeasurement] = {}
    for fmt in formats:
        storage_dtype = _torch_storage_dtype(fmt)
        if storage_dtype is None:
            results[fmt.value] = TransferMeasurement(
                fmt.value, 0, 0.0, 0.0, False, "storage_dtype_unavailable"
            )
            continue
        try:
            host = torch.empty(numel, dtype=storage_dtype, pin_memory=True)
            dev = torch.empty(numel, dtype=storage_dtype, device=device)
        except Exception as exc:  # pragma: no cover - device dependent
            results[fmt.value] = TransferMeasurement(
                fmt.value,
                0,
                0.0,
                0.0,
                False,
                f"alloc_failed:{type(exc).__name__}",
            )
            continue

        bytes_each = host.element_size() * host.numel()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(max(0, warmup)):
            dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(max(1, iters)):
            dev.copy_(host, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        total_seconds = start.elapsed_time(end) / 1000.0
        per_copy_seconds = total_seconds / max(1, iters)
        bytes_per_second = (
            bytes_each / per_copy_seconds if per_copy_seconds > 0 else 0.0
        )
        results[fmt.value] = TransferMeasurement(
            format=fmt.value,
            bytes_transferred=int(bytes_each),
            seconds=float(per_copy_seconds),
            bytes_per_second=float(bytes_per_second),
            available=True,
            skip_reason=None,
        )
        del host, dev
        torch.cuda.empty_cache()
    return results
