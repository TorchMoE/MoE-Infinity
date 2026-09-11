# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from moe_infinity.runtime.adaptive_precision_allowlist import (
    ReleasedAdaptiveEntry,
)
from moe_infinity.runtime.expert_precision import (
    ExecutionKind,
    ExpertFormat,
    ExpertVariantSpec,
)

_KAIO_ALIGNMENT = 4096

_DTYPE_ELEMENT_SIZE = {
    "float8_e4m3fn": 1,
    "uint8": 1,
    "int32": 4,
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
}

_DERIVATIVE_INDEX_KEYS = frozenset(
    {
        "schema_version",
        "generation",
        "canonical_max_tensor_id",
        "canonical_max_file_id",
        "records",
    }
)

_DERIVATIVE_RECORD_KEYS = frozenset(
    {"dtype", "file_id", "offset", "sha256", "shape", "size", "tensor_id"}
)

_CURRENT_KEYS = frozenset(
    {
        "derivative_index_sha256",
        "generation",
        "manifest_sha256",
        "quality_attestation_sha256",
        "schema_version",
    }
)

_SNAPSHOT_ROW_KEYS = frozenset(
    {"tensor_id", "dtype", "shape", "size", "file_id", "offset"}
)


def canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_derivative_document(document: dict) -> dict:
    if set(document) != _DERIVATIVE_INDEX_KEYS:
        raise ValueError("derivative index schema mismatch")
    if document["schema_version"] != 1:
        raise ValueError("derivative index schema_version mismatch")
    canonical_max_tensor_id = document["canonical_max_tensor_id"]
    records = document["records"]
    if not isinstance(records, list):
        raise ValueError("derivative index records must be a list")

    seen_ids: set[int] = set()
    intervals: dict[int, list[tuple[int, int]]] = {}
    ordered = sorted(records, key=lambda row: row["tensor_id"])
    for row in ordered:
        if set(row) != _DERIVATIVE_RECORD_KEYS:
            raise ValueError("derivative record schema mismatch")
        dtype = row["dtype"]
        if dtype not in _DTYPE_ELEMENT_SIZE:
            raise ValueError(f"unknown derivative dtype: {dtype}")
        tensor_id = row["tensor_id"]
        if tensor_id <= canonical_max_tensor_id:
            raise ValueError("derivative tensor_id not above canonical maximum")
        if tensor_id in seen_ids:
            raise ValueError("duplicate derivative tensor_id")
        seen_ids.add(tensor_id)
        shape = row["shape"]
        if not shape or any(int(dim) <= 0 for dim in shape):
            raise ValueError("derivative shape must have positive dimensions")
        expected_size = (
            math.prod(int(dim) for dim in shape) * _DTYPE_ELEMENT_SIZE[dtype]
        )
        if row["size"] != expected_size:
            raise ValueError("derivative size does not match shape and dtype")
        offset = row["offset"]
        if offset < 0 or offset % _KAIO_ALIGNMENT != 0:
            raise ValueError("derivative offset is not aligned")
        file_id = row["file_id"]
        interval = (offset, offset + row["size"])
        for existing in intervals.get(file_id, []):
            if interval[0] < existing[1] and existing[0] < interval[1]:
                raise ValueError("derivative file intervals overlap")
        intervals.setdefault(file_id, []).append(interval)
    return {**document, "records": ordered}


def write_derivative_index(
    path,
    generation: str,
    canonical_max_tensor_id: int,
    canonical_max_file_id: int,
    records: Iterable[dict],
) -> None:
    document = {
        "schema_version": 1,
        "generation": generation,
        "canonical_max_tensor_id": canonical_max_tensor_id,
        "canonical_max_file_id": canonical_max_file_id,
        "records": sorted(records, key=lambda row: row["tensor_id"]),
    }
    _validate_derivative_document(document)
    Path(path).write_bytes(canonical_json_bytes(document))


def load_derivative_overlay(path, native_handle) -> None:
    document = json.loads(Path(path).read_text())
    document = _validate_derivative_document(document)
    generation = document["generation"]
    native_handle.begin_derivative_overlay(
        generation,
        document["canonical_max_tensor_id"],
        document["canonical_max_file_id"],
    )
    try:
        for row in document["records"]:
            native_handle.register_derivative_tensor(
                generation,
                row["tensor_id"],
                row["file_id"],
                row["offset"],
                row["size"],
                row["shape"],
                row["dtype"],
            )
    except Exception:
        native_handle.abort_derivative_overlay(generation)
        raise
    native_handle.commit_derivative_overlay(generation)


def compute_checkpoint_fingerprint(
    native_handle, model_signature, offload_path
) -> dict:
    offload_path = Path(offload_path)
    snapshot = native_handle.get_canonical_tensor_index_snapshot()
    if not snapshot:
        raise ValueError("canonical tensor index snapshot is empty")
    for row in snapshot:
        if set(row) != _SNAPSHOT_ROW_KEYS:
            raise ValueError("canonical snapshot row schema mismatch")
    ordered = sorted(snapshot, key=lambda row: row["tensor_id"])
    canonical_max_tensor_id = max(int(row["tensor_id"]) for row in ordered)
    canonical_max_file_id = max(int(row["file_id"]) for row in ordered)

    partitions = []
    for file_id in sorted({int(row["file_id"]) for row in ordered}):
        partition_path = offload_path / f"archer_param_{file_id}"
        payload = partition_path.read_bytes()
        partitions.append(
            {
                "file_id": file_id,
                "sha256": _sha256_hex(payload),
                "size": len(payload),
            }
        )

    envelope = {
        "schema_version": 1,
        "tensor_index": ordered,
        "model_signature": model_signature,
        "partitions": partitions,
    }
    checkpoint_fingerprint = _sha256_hex(canonical_json_bytes(envelope))
    tensor_index_sha256 = _sha256_hex(canonical_json_bytes(ordered))
    model_signature_sha256 = _sha256_hex(canonical_json_bytes(model_signature))
    return {
        "schema_version": 1,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "tensor_index_sha256": tensor_index_sha256,
        "model_signature_sha256": model_signature_sha256,
        "canonical_max_tensor_id": canonical_max_tensor_id,
        "canonical_max_file_id": canonical_max_file_id,
        "partitions": partitions,
    }


def _variant_to_dict(variant: ExpertVariantSpec) -> dict:
    return {
        "layer_id": variant.layer_id,
        "expert_id": variant.expert_id,
        "format": variant.format.value,
        "execution": variant.execution.value,
        "tensor_ids": list(variant.tensor_ids),
        "tensor_roles": list(variant.tensor_roles),
        "payload_bytes": variant.payload_bytes,
        "aligned_bytes": variant.aligned_bytes,
        "workspace_bytes": variant.workspace_bytes,
        "source_format": variant.source_format.value,
        "converter_version": variant.converter_version,
        "quality_attestation_sha256": variant.quality_attestation_sha256,
    }


def _variant_from_dict(row: dict) -> ExpertVariantSpec:
    return ExpertVariantSpec(
        layer_id=row["layer_id"],
        expert_id=row["expert_id"],
        format=ExpertFormat(row["format"]),
        execution=ExecutionKind(row["execution"]),
        tensor_ids=tuple(row["tensor_ids"]),
        tensor_roles=tuple(row["tensor_roles"]),
        payload_bytes=row["payload_bytes"],
        aligned_bytes=row["aligned_bytes"],
        workspace_bytes=row["workspace_bytes"],
        source_format=ExpertFormat(row["source_format"]),
        converter_version=row["converter_version"],
        quality_attestation_sha256=row["quality_attestation_sha256"],
    )


@dataclass(frozen=True)
class ExpertVariantManifest:
    model_name: str
    checkpoint_fingerprint: str
    generation: str
    canonical_max_tensor_id: int
    canonical_max_file_id: int
    derivative_index_sha256: str
    variants: tuple[ExpertVariantSpec, ...]
    converter_version: str = "adaptive-expert-v1"
    store_signature_version: int = 1
    quality_attestation_sha256: str = ""
    variant_payload_sha256: str = ""
    complete: bool = True

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        checkpoint_fingerprint: str,
        generation: str,
        canonical_max_tensor_id: int,
        canonical_max_file_id: int,
        derivative_index_sha256: str,
        variants: Iterable[ExpertVariantSpec],
    ) -> "ExpertVariantManifest":
        return cls(
            model_name=model_name,
            checkpoint_fingerprint=checkpoint_fingerprint,
            generation=generation,
            canonical_max_tensor_id=canonical_max_tensor_id,
            canonical_max_file_id=canonical_max_file_id,
            derivative_index_sha256=derivative_index_sha256,
            variants=tuple(variants),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "model_name": self.model_name,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "store_signature_version": self.store_signature_version,
            "converter_version": self.converter_version,
            "generation": self.generation,
            "canonical_max_tensor_id": self.canonical_max_tensor_id,
            "canonical_max_file_id": self.canonical_max_file_id,
            "derivative_index_sha256": self.derivative_index_sha256,
            "quality_attestation_sha256": self.quality_attestation_sha256,
            "variant_payload_sha256": self.variant_payload_sha256,
            "complete": self.complete,
            "variants": [
                _variant_to_dict(variant) for variant in self.variants
            ],
        }

    @classmethod
    def from_dict(cls, document: dict) -> "ExpertVariantManifest":
        return cls(
            model_name=document["model_name"],
            checkpoint_fingerprint=document["checkpoint_fingerprint"],
            generation=document["generation"],
            canonical_max_tensor_id=document["canonical_max_tensor_id"],
            canonical_max_file_id=document["canonical_max_file_id"],
            derivative_index_sha256=document["derivative_index_sha256"],
            variants=tuple(
                _variant_from_dict(row) for row in document["variants"]
            ),
        )

    def write_atomic(self, path) -> None:
        Path(path).write_bytes(canonical_json_bytes(self.to_dict()))

    @classmethod
    def load_for_serving(
        cls,
        generation_dir,
        checkpoint_fingerprint: str,
        released_entries: frozenset[ReleasedAdaptiveEntry],
    ) -> "ExpertVariantManifest":
        generation_dir = Path(generation_dir)
        document = json.loads((generation_dir / "manifest.v1.json").read_text())
        manifest = cls.from_dict(document)
        if manifest.checkpoint_fingerprint != checkpoint_fingerprint:
            raise ValueError("manifest checkpoint fingerprint mismatch")
        for variant in manifest.variants:
            entry = ReleasedAdaptiveEntry(
                checkpoint_fingerprint=manifest.checkpoint_fingerprint,
                format=variant.format,
                converter_version=variant.converter_version,
                quality_attestation_sha256=variant.quality_attestation_sha256,
            )
            if entry not in released_entries:
                raise ValueError("manifest is not released")
        return manifest

    @classmethod
    def load_current(
        cls, root, *, native_handle, register_overlay: bool = True
    ) -> "ExpertVariantManifest":
        root = Path(root)
        current = json.loads((root / "CURRENT").read_text())
        if set(current) != _CURRENT_KEYS:
            raise ValueError("CURRENT schema mismatch")
        if current["schema_version"] != 1:
            raise ValueError("CURRENT schema_version mismatch")
        generation = current["generation"]
        generation_dir = root / "generations" / generation

        index_bytes = (generation_dir / "derivative-index.v1.json").read_bytes()
        attestation_bytes = (
            generation_dir / "quality-attestation.v1.json"
        ).read_bytes()
        manifest_bytes = (generation_dir / "manifest.v1.json").read_bytes()

        if _sha256_hex(index_bytes) != current["derivative_index_sha256"]:
            raise ValueError("manifest_digest_mismatch")
        if (
            _sha256_hex(attestation_bytes)
            != current["quality_attestation_sha256"]
        ):
            raise ValueError("manifest_digest_mismatch")
        if _sha256_hex(manifest_bytes) != current["manifest_sha256"]:
            raise ValueError("manifest_digest_mismatch")

        index_document = _validate_derivative_document(json.loads(index_bytes))
        if index_document["generation"] != generation:
            raise ValueError("derivative generation mismatch")
        for record in index_document["records"]:
            partition = root.parent / f"archer_param_{record['file_id']}"
            with partition.open("rb") as handle:
                handle.seek(record["offset"])
                payload = handle.read(record["size"])
            if (
                len(payload) != record["size"]
                or _sha256_hex(payload) != record["sha256"]
            ):
                raise ValueError("derivative payload checksum mismatch")

        from moe_infinity.runtime.expert_variant_build import (
            validate_quality_attestation,
        )

        attestation = json.loads(attestation_bytes)
        validate_quality_attestation(attestation)
        manifest_document = json.loads(manifest_bytes)
        if (
            manifest_document.get("derivative_index_sha256")
            != current["derivative_index_sha256"]
            or manifest_document.get("quality_attestation_sha256")
            != current["quality_attestation_sha256"]
            or attestation["derivative_index_sha256"]
            != current["derivative_index_sha256"]
            or attestation["generation"] != generation
        ):
            raise ValueError("manifest_digest_mismatch")
        manifest = cls.from_dict(manifest_document)
        if register_overlay:
            load_derivative_overlay(
                generation_dir / "derivative-index.v1.json", native_handle
            )
        return manifest
