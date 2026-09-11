from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from moe_infinity.utils.fp8 import dequant_fp8_blockwise, quant_fp8_blockwise


@dataclass(frozen=True)
class FP8ExpertVariant:
    weights: tuple[torch.Tensor, ...]
    scales: tuple[torch.Tensor, ...]

    def dequantized_weights(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            dequant_fp8_blockwise(weight, scale, dtype=torch.bfloat16)
            for weight, scale in zip(self.weights, self.scales)
        )


def build_fp8_expert_variant(source) -> FP8ExpertVariant:
    pairs = [quant_fp8_blockwise(weight, block_size=128) for weight in source]
    return FP8ExpertVariant(
        tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)
    )


def run_bf16_expert(x: torch.Tensor, weights) -> torch.Tensor:
    gate, up, down = weights
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def run_fp8_expert(x: torch.Tensor, variant: FP8ExpertVariant) -> torch.Tensor:
    return run_bf16_expert(x, variant.dequantized_weights())


def validate_fp8_expert_variant(
    source,
    variant: FP8ExpertVariant,
    *,
    relative_l2_max: float = 0.05,
    cosine_min: float = 0.995,
) -> None:
    restored = variant.dequantized_weights()
    if len(source) != len(restored):
        raise ValueError("quality_tensor")
    for original, actual in zip(source, restored):
        if original.shape != actual.shape or not torch.isfinite(actual).all():
            raise ValueError("quality_nonfinite")
        relative_l2 = (
            actual.float() - original.float()
        ).norm() / original.float().norm().clamp_min(1e-12)
        cosine = F.cosine_similarity(
            original.float().flatten(), actual.float().flatten(), dim=0
        )
        if not math.isfinite(relative_l2.item()) or not math.isfinite(
            cosine.item()
        ):
            raise ValueError("quality_nonfinite")
        if relative_l2.item() > relative_l2_max or cosine.item() < cosine_min:
            raise ValueError("quality_tensor")
