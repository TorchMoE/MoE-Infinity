# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from dataclasses import dataclass

from moe_infinity.runtime.expert_precision import ExpertFormat


@dataclass(frozen=True)
class ReleasedAdaptiveEntry:
    checkpoint_fingerprint: str
    format: ExpertFormat
    converter_version: str
    quality_attestation_sha256: str


# Serving approval is exact over all four fields. The initial allowlist is
# empty on purpose: adding an entry requires a separate reviewed source change
# carrying the fingerprint, format, converter version, and attestation digest.
RELEASED_ADAPTIVE_ENTRIES: frozenset[ReleasedAdaptiveEntry] = frozenset()


def is_released(entry: ReleasedAdaptiveEntry) -> bool:
    return entry in RELEASED_ADAPTIVE_ENTRIES
