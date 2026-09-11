# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import torch


class ExpertFormat(str, Enum):
    BF16 = "bf16"
    FP8_E4M3_BLOCK128 = "fp8_e4m3_block128"
    MARLIN_INT4_GROUP128 = "marlin_int4_group128"
    GPT_OSS_MXFP4 = "gpt_oss_mxfp4"
    GLM_FP8_BLOCK128 = "glm_fp8_block128"
    DEEPSEEK_V4_FP4 = "deepseek_v4_fp4"
    GPTQ = "gptq"
    AWQ = "awq"


class ExecutionKind(str, Enum):
    BF16_GEMM = "bf16_gemm"
    FP8_DEQUANT_BF16_GEMM = "fp8_dequant_bf16_gemm"
    MARLIN_W4A16 = "marlin_w4a16"
    GPT_OSS_MXFP4 = "gpt_oss_mxfp4"
    DEEPSEEK_V4_FP4 = "deepseek_v4_fp4"
    LEGACY_QUANTIZED = "legacy_quantized"


@dataclass(frozen=True)
class FormatCapability:
    format: ExpertFormat
    execution: ExecutionKind
    quality_rank: int
    tensor_roles: tuple[str, ...]
    scale_owner: Literal["inline", "manifest", "kernel"]
    supported_storage_dtypes: tuple[torch.dtype, ...]
    output_dtype: torch.dtype
    block_size: Optional[int]
    group_size: Optional[int]
    requires_extension: Optional[str]
    protected: bool


@dataclass(frozen=True)
class CandidateVariantSpec:
    layer_id: int
    expert_id: int
    format: ExpertFormat
    execution: ExecutionKind
    tensor_ids: tuple[int, ...]
    tensor_roles: tuple[str, ...]
    payload_bytes: int
    aligned_bytes: int
    workspace_bytes: int
    source_format: ExpertFormat
    converter_version: str


@dataclass(frozen=True)
class ExpertVariantSpec(CandidateVariantSpec):
    quality_attestation_sha256: str


@dataclass(frozen=True)
class ResidentGeneration:
    format: ExpertFormat
    aligned_bytes: int
    generation: int
    state: Literal["active", "retiring"] = "active"


@dataclass(frozen=True)
class HardwareCapabilities:
    cuda_available: bool
    compute_capability: Optional[tuple[int, int]]
    extension_names: frozenset[str]


@dataclass(frozen=True)
class ModelPrecisionCapabilities:
    model_type: str
    formats: dict  # dict[ExpertFormat, FormatCapability]
    protected_reason: Optional[str]


CONVERTER_CANDIDATE_MODEL_TYPES = frozenset(
    {"mixtral", "qwen3_moe", "deepseek_v2"}
)

PROTECTED_MODEL_PATHS = {
    "gpt_oss": "protected:gpt_oss_mxfp4",
    "glm_moe_dsa": "protected:glm_fp8",
    "deepseek_v3": "protected:existing_fp8",
    "deepseek_v4": "protected:deepseek_v4_fp4",
}

# Quality rank is lower-is-better: BF16 is the highest-quality reference.
_QUALITY_RANK_BF16 = 0
_QUALITY_RANK_FP8 = 1
_QUALITY_RANK_MARLIN = 2

_ROLES_UNSCALED = ("gate.weight", "up.weight", "down.weight")
_ROLES_SCALED = (
    "gate.weight",
    "gate.scale",
    "up.weight",
    "up.scale",
    "down.weight",
    "down.scale",
)


def bf16_capability() -> FormatCapability:
    return FormatCapability(
        format=ExpertFormat.BF16,
        execution=ExecutionKind.BF16_GEMM,
        quality_rank=_QUALITY_RANK_BF16,
        tensor_roles=_ROLES_UNSCALED,
        scale_owner="kernel",
        supported_storage_dtypes=(torch.bfloat16,),
        output_dtype=torch.bfloat16,
        block_size=None,
        group_size=None,
        requires_extension=None,
        protected=False,
    )


def fp8_capability() -> FormatCapability:
    return FormatCapability(
        format=ExpertFormat.FP8_E4M3_BLOCK128,
        execution=ExecutionKind.FP8_DEQUANT_BF16_GEMM,
        quality_rank=_QUALITY_RANK_FP8,
        tensor_roles=_ROLES_SCALED,
        scale_owner="manifest",
        supported_storage_dtypes=(torch.float8_e4m3fn,),
        output_dtype=torch.bfloat16,
        block_size=128,
        group_size=None,
        requires_extension=None,
        protected=False,
    )


def marlin_capability() -> FormatCapability:
    return FormatCapability(
        format=ExpertFormat.MARLIN_INT4_GROUP128,
        execution=ExecutionKind.MARLIN_W4A16,
        quality_rank=_QUALITY_RANK_MARLIN,
        tensor_roles=_ROLES_SCALED,
        scale_owner="manifest",
        supported_storage_dtypes=(torch.uint8,),
        output_dtype=torch.bfloat16,
        block_size=None,
        group_size=128,
        requires_extension="_marlin",
        protected=False,
    )


# Protected model-specific format identities. Each carries protected=True so a
# protected model can never leak a mutable candidate capability.
_PROTECTED_FORMAT_BY_REASON = {
    "protected:gpt_oss_mxfp4": (
        ExpertFormat.GPT_OSS_MXFP4,
        ExecutionKind.GPT_OSS_MXFP4,
    ),
    "protected:glm_fp8": (
        ExpertFormat.GLM_FP8_BLOCK128,
        ExecutionKind.FP8_DEQUANT_BF16_GEMM,
    ),
    "protected:existing_fp8": (
        ExpertFormat.GLM_FP8_BLOCK128,
        ExecutionKind.FP8_DEQUANT_BF16_GEMM,
    ),
    "protected:deepseek_v4_fp4": (
        ExpertFormat.DEEPSEEK_V4_FP4,
        ExecutionKind.DEEPSEEK_V4_FP4,
    ),
    "protected:gptq": (ExpertFormat.GPTQ, ExecutionKind.LEGACY_QUANTIZED),
    "protected:awq": (ExpertFormat.AWQ, ExecutionKind.LEGACY_QUANTIZED),
    "protected:mxfp4": (
        ExpertFormat.GPT_OSS_MXFP4,
        ExecutionKind.GPT_OSS_MXFP4,
    ),
    "protected:fp8": (
        ExpertFormat.GLM_FP8_BLOCK128,
        ExecutionKind.FP8_DEQUANT_BF16_GEMM,
    ),
}


def _protected_capability(
    fmt: ExpertFormat, execution: ExecutionKind
) -> FormatCapability:
    return FormatCapability(
        format=fmt,
        execution=execution,
        quality_rank=_QUALITY_RANK_BF16,
        tensor_roles=_ROLES_UNSCALED,
        scale_owner="kernel",
        supported_storage_dtypes=(),
        output_dtype=torch.bfloat16,
        block_size=None,
        group_size=None,
        requires_extension=None,
        protected=True,
    )


def protected_capabilities(
    model_type: str, reason: str
) -> ModelPrecisionCapabilities:
    # Every returned capability is protected=True so adaptive conversion never
    # applies to validated model-specific low-bit kernels.
    fmt, execution = _PROTECTED_FORMAT_BY_REASON.get(
        reason, (ExpertFormat.BF16, ExecutionKind.LEGACY_QUANTIZED)
    )
    formats = {fmt: _protected_capability(fmt, execution)}
    return ModelPrecisionCapabilities(model_type, formats, reason)


def probe_hardware_capabilities(
    extension_names: set[str],
) -> HardwareCapabilities:
    # Neutral probe: CUDA availability + compute capability only; never a GPU
    # marketing name or assumed bandwidth (explicit registry non-goal).
    cuda_available = bool(torch.cuda.is_available())
    compute_capability: Optional[tuple[int, int]] = None
    if cuda_available:
        try:
            major, minor = torch.cuda.get_device_capability()
            compute_capability = (int(major), int(minor))
        except Exception:
            compute_capability = None
    return HardwareCapabilities(
        cuda_available=cuda_available,
        compute_capability=compute_capability,
        extension_names=frozenset(extension_names),
    )


def resolve_model_precision_capabilities(
    config: object,
    extension_names: set[str],
) -> ModelPrecisionCapabilities:
    # Order is load-bearing: protected model_type is matched before quant_method
    # before candidate enumeration; model type/name alone never approves serving.
    model_type = str(getattr(config, "model_type", ""))
    quant_method = str(
        (getattr(config, "quantization_config", None) or {}).get(
            "quant_method", ""
        )
    ).lower()
    if model_type in PROTECTED_MODEL_PATHS:
        return protected_capabilities(
            model_type, PROTECTED_MODEL_PATHS[model_type]
        )
    if quant_method in {"gptq", "awq", "mxfp4", "fp8"}:
        return protected_capabilities(model_type, f"protected:{quant_method}")
    if model_type not in CONVERTER_CANDIDATE_MODEL_TYPES:
        return ModelPrecisionCapabilities(
            model_type, {}, "unsupported:converter_candidate"
        )
    formats = {ExpertFormat.BF16: bf16_capability()}
    formats[ExpertFormat.FP8_E4M3_BLOCK128] = fp8_capability()
    if "_marlin" in extension_names:
        formats[ExpertFormat.MARLIN_INT4_GROUP128] = marlin_capability()
    return ModelPrecisionCapabilities(model_type, formats, None)
