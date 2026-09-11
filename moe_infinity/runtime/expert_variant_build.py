# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DERIVATIVES_DIRNAME = "adaptive_derivatives"
JOURNAL_FILENAME = "build-journal.v1.json"
GENERATIONS_DIRNAME = "generations"
CURRENT_FILENAME = "CURRENT"
QUALITY_ATTESTATION_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_fingerprint",
        "generation",
        "converter_version",
        "converter_source_commit",
        "variant_payload_sha256",
        "derivative_index_sha256",
        "thresholds",
        "formats",
        "raw_result_sha256",
        "workload_sha256",
        "hardware",
        "software",
        "passed",
    }
)

_JOURNAL_KEYS = frozenset(
    {
        "canonical_max_file_id",
        "canonical_max_tensor_id",
        "completed_tensors",
        "converter_version",
        "generation",
        "next_file_id",
        "next_tensor_id",
        "published_ranges",
        "reserved_file_id_range",
        "reserved_tensor_id_range",
        "schema_version",
        "state",
    }
)


def _canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
    )


def _durable_replace_bytes(path: Path, data: bytes) -> None:
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# Intentionally separate from the module-level _durable_replace_bytes /
# _fsync_directory publish helpers: those are monkeypatched by publish-path
# fault-injection tests, so journal persistence must stay on this real,
# always-durable path to remain crash-safe and out of captured event lists.
def _persist_journal_bytes(path: Path, data: bytes) -> None:
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _generation_name(index: int) -> str:
    return "g" + f"{index:08d}"


def _generation_index(name: str) -> int:
    return int(name[1:])


def validate_quality_attestation(document: dict) -> None:
    if (
        set(document) != QUALITY_ATTESTATION_KEYS
        or document["schema_version"] != 1
    ):
        raise ValueError("quality attestation schema mismatch")
    if document["passed"] is not True:
        raise ValueError("quality_model")
    if [row["format"] for row in document["formats"]] != sorted(
        row["format"] for row in document["formats"]
    ):
        raise ValueError("quality attestation formats are not sorted")
    numeric = []
    for row in document["formats"]:
        numeric.extend(
            value for value in row.values() if isinstance(value, (int, float))
        )
    numeric.extend(document["thresholds"].values())
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("quality_nonfinite")
    thresholds = document["thresholds"]
    for row in document["formats"]:
        baseline = float(row["perplexity_baseline"])
        adaptive = float(row["perplexity_adaptive"])
        if baseline <= 0 or adaptive / baseline - 1 > float(
            thresholds["perplexity_relative_increase_max"]
        ):
            raise ValueError("quality_model")
        if float(row["greedy_agreement"]) < float(
            thresholds["greedy_agreement_min"]
        ):
            raise ValueError("quality_model")


@dataclass
class DerivativeBuildJournal:
    offload_path: Path
    canonical_max_tensor_id: int
    canonical_max_file_id: int
    converter_version: str
    generation: str
    reserved_tensor_id_range: tuple[int, int]
    reserved_file_id_range: tuple[int, int]
    next_tensor_id: int
    next_file_id: int
    state: str = "reserved"
    completed_tensors: dict[int, dict] = field(default_factory=dict)
    published_ranges: list[dict] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.offload_path / DERIVATIVES_DIRNAME

    @property
    def journal_path(self) -> Path:
        return self.root / JOURNAL_FILENAME

    @property
    def generation_dir(self) -> Path:
        return self.root / GENERATIONS_DIRNAME / self.generation

    @property
    def tensor_id_range(self) -> tuple[int, int]:
        return self.reserved_tensor_id_range

    @property
    def file_id_range(self) -> tuple[int, int]:
        return self.reserved_file_id_range

    @classmethod
    def _load_document(cls, offload_path: Path) -> Optional[dict]:
        journal_path = offload_path / DERIVATIVES_DIRNAME / JOURNAL_FILENAME
        if not journal_path.is_file():
            return None
        document = json.loads(journal_path.read_text())
        if set(document) != _JOURNAL_KEYS:
            raise ValueError("build journal schema mismatch")
        return document

    @classmethod
    def _from_document(
        cls, offload_path: Path, document: dict
    ) -> "DerivativeBuildJournal":
        completed = {
            int(row["tensor_id"]): dict(row)
            for row in document["completed_tensors"]
        }
        return cls(
            offload_path=Path(offload_path),
            canonical_max_tensor_id=int(document["canonical_max_tensor_id"]),
            canonical_max_file_id=int(document["canonical_max_file_id"]),
            converter_version=str(document["converter_version"]),
            generation=str(document["generation"]),
            reserved_tensor_id_range=tuple(
                document["reserved_tensor_id_range"]
            ),
            reserved_file_id_range=tuple(document["reserved_file_id_range"]),
            next_tensor_id=int(document["next_tensor_id"]),
            next_file_id=int(document["next_file_id"]),
            state=str(document["state"]),
            completed_tensors=completed,
            published_ranges=list(document["published_ranges"]),
        )

    @classmethod
    def reserve(
        cls,
        offload_path,
        canonical_max_tensor_id: int,
        canonical_max_file_id: int,
        tensor_count: int,
        file_count: int,
        converter_version: str,
    ) -> "DerivativeBuildJournal":
        offload_path = Path(offload_path)
        document = cls._load_document(offload_path)
        if document is not None:
            existing = cls._from_document(offload_path, document)
            base_tensor = existing.next_tensor_id - 1
            base_file = existing.next_file_id - 1
            generation_index = _generation_index(existing.generation) + 1
            carried_ranges = list(existing.published_ranges)
        else:
            base_tensor = canonical_max_tensor_id
            base_file = canonical_max_file_id
            generation_index = 1
            carried_ranges = []

        tensor_start = base_tensor + 1
        tensor_end = base_tensor + tensor_count
        file_start = base_file + 1
        file_end = base_file + file_count
        generation = _generation_name(generation_index)

        journal = cls(
            offload_path=offload_path,
            canonical_max_tensor_id=canonical_max_tensor_id,
            canonical_max_file_id=canonical_max_file_id,
            converter_version=converter_version,
            generation=generation,
            reserved_tensor_id_range=(tensor_start, tensor_end),
            reserved_file_id_range=(file_start, file_end),
            next_tensor_id=tensor_end + 1,
            next_file_id=file_end + 1,
            state="reserved",
            published_ranges=carried_ranges,
        )
        journal.generation_dir.mkdir(parents=True, exist_ok=True)
        journal._persist()
        return journal

    def _document(self) -> dict:
        completed = [
            self.completed_tensors[tensor_id]
            for tensor_id in sorted(self.completed_tensors)
        ]
        return {
            "canonical_max_file_id": self.canonical_max_file_id,
            "canonical_max_tensor_id": self.canonical_max_tensor_id,
            "completed_tensors": completed,
            "converter_version": self.converter_version,
            "generation": self.generation,
            "next_file_id": self.next_file_id,
            "next_tensor_id": self.next_tensor_id,
            "published_ranges": self.published_ranges,
            "reserved_file_id_range": list(self.reserved_file_id_range),
            "reserved_tensor_id_range": list(self.reserved_tensor_id_range),
            "schema_version": 1,
            "state": self.state,
        }

    def _persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _persist_journal_bytes(
            self.journal_path, _canonical_json_bytes(self._document())
        )
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def mark_writing(self, completed_tensors: dict) -> None:
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.completed_tensors = dict(completed_tensors)
        self.state = "writing"
        self._persist()

    def mark_indexed(self) -> None:
        self.state = "indexed"
        self._persist()

    def mark_attested(self) -> None:
        self.state = "attested"
        self._persist()

    def abort_corrupt_generation(self) -> None:
        self.state = "aborted"
        self._persist()

    def publish_current(self) -> None:
        if self.state != "attested":
            raise RuntimeError("generation is not attested")
        self.state = "published"
        self._persist()

    def publish_attested_generation(
        self,
        *,
        payloads: dict,
        derivative_index: dict,
        quality_attestation: dict,
        manifest: dict,
        current: dict,
    ) -> None:
        for file_id in sorted(payloads):
            payload_path = self.offload_path / f"archer_param_{file_id}"
            _durable_replace_bytes(payload_path, payloads[file_id])
        _fsync_directory(self.offload_path)

        gen_dir = self.generation_dir
        _durable_replace_bytes(
            gen_dir / "derivative-index.v1.json",
            _canonical_json_bytes(derivative_index),
        )
        _durable_replace_bytes(
            gen_dir / "quality-attestation.v1.json",
            _canonical_json_bytes(quality_attestation),
        )
        _durable_replace_bytes(
            gen_dir / "manifest.v1.json",
            _canonical_json_bytes(manifest),
        )
        _fsync_directory(gen_dir)

        _durable_replace_bytes(
            self.root / CURRENT_FILENAME, _canonical_json_bytes(current)
        )
        _fsync_directory(self.root)

        self.published_ranges = self.published_ranges + [
            {
                "generation": self.generation,
                "file_id_range": list(self.reserved_file_id_range),
                "tensor_id_range": list(self.reserved_tensor_id_range),
            }
        ]
        self.state = "published"
        self._persist()


def recover_or_reserve_generation(
    offload_path,
    *,
    canonical_max_tensor_id: int,
    canonical_max_file_id: int,
    tensor_count: int,
    file_count: int,
) -> DerivativeBuildJournal:
    offload_path = Path(offload_path)
    document = DerivativeBuildJournal._load_document(offload_path)
    if document is not None and document["state"] in {
        "reserved",
        "writing",
        "indexed",
    }:
        return DerivativeBuildJournal._from_document(offload_path, document)
    return DerivativeBuildJournal.reserve(
        offload_path,
        canonical_max_tensor_id,
        canonical_max_file_id,
        tensor_count,
        file_count,
        "adaptive-expert-v1",
    )
