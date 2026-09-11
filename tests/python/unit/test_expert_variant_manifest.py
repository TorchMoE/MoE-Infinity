# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import hashlib
import json

import pytest

from moe_infinity.runtime.adaptive_precision_allowlist import (
    ReleasedAdaptiveEntry,
)
from moe_infinity.runtime.expert_precision import (
    ExecutionKind,
    ExpertFormat,
    ExpertVariantSpec,
)
from moe_infinity.runtime.expert_variant_manifest import (
    ExpertVariantManifest,
    canonical_json_bytes,
    compute_checkpoint_fingerprint,
    load_derivative_overlay,
    write_derivative_index,
)


class RecordingOverlayHandle:
    def __init__(self, fail_tensor_id=None):
        self.calls = []
        self.fail_tensor_id = fail_tensor_id

    def begin_derivative_overlay(
        self, generation, canonical_max_tensor_id, canonical_max_file_id
    ):
        self.calls.append(
            (
                "begin",
                generation,
                canonical_max_tensor_id,
                canonical_max_file_id,
            )
        )

    def register_derivative_tensor(
        self, generation, tensor_id, file_id, offset, size, shape, dtype
    ):
        if tensor_id == self.fail_tensor_id:
            raise RuntimeError("injected registration failure")
        self.calls.append(
            (
                "register",
                generation,
                tensor_id,
                file_id,
                offset,
                size,
                shape,
                dtype,
            )
        )

    def commit_derivative_overlay(self, generation):
        self.calls.append(("commit", generation))

    def abort_derivative_overlay(self, generation):
        self.calls.append(("abort", generation))


@pytest.fixture
def valid_derivative_index(tmp_path):
    path = tmp_path / "derivative-index.v1.json"
    write_derivative_index(
        path,
        "g00000001",
        100,
        3,
        [
            {
                "tensor_id": 101,
                "file_id": 4,
                "offset": 0,
                "size": 2,
                "shape": [1],
                "dtype": "bfloat16",
                "sha256": "d" * 64,
            },
            {
                "tensor_id": 102,
                "file_id": 4,
                "offset": 4096,
                "size": 4,
                "shape": [1],
                "dtype": "float32",
                "sha256": "c" * 64,
            },
        ],
    )
    return path


def _variant():
    return ExpertVariantSpec(
        layer_id=1,
        expert_id=2,
        format=ExpertFormat.FP8_E4M3_BLOCK128,
        execution=ExecutionKind.FP8_DEQUANT_BF16_GEMM,
        tensor_ids=(101, 102, 103, 104, 105, 106),
        tensor_roles=(
            "gate.weight",
            "gate.scale",
            "up.weight",
            "up.scale",
            "down.weight",
            "down.scale",
        ),
        payload_bytes=600,
        aligned_bytes=24576,
        workspace_bytes=1200,
        source_format=ExpertFormat.BF16,
        converter_version="adaptive-expert-v1",
        quality_attestation_sha256="b" * 64,
    )


def test_serving_validation_requires_exact_released_entry(tmp_path):
    generation = tmp_path / "adaptive_derivatives" / "generations" / "g00000001"
    generation.mkdir(parents=True)
    index = generation / "derivative-index.v1.json"
    index.write_bytes(b"index")
    manifest = ExpertVariantManifest.create(
        model_name="model",
        checkpoint_fingerprint="a" * 64,
        generation="g00000001",
        canonical_max_tensor_id=100,
        canonical_max_file_id=3,
        derivative_index_sha256=hashlib.sha256(b"index").hexdigest(),
        variants=[_variant()],
    )
    manifest.write_atomic(generation / "manifest.v1.json")
    released = ReleasedAdaptiveEntry(
        "a" * 64, ExpertFormat.FP8_E4M3_BLOCK128, "adaptive-expert-v1", "b" * 64
    )
    loaded = ExpertVariantManifest.load_for_serving(
        generation, "a" * 64, frozenset({released})
    )
    assert loaded.variants == (_variant(),)
    with pytest.raises(ValueError, match="manifest is not released"):
        ExpertVariantManifest.load_for_serving(
            generation, "a" * 64, frozenset()
        )


def test_derivative_index_is_canonical_json_and_registers_explicit_records(
    tmp_path,
):
    records = [
        {
            "tensor_id": 102,
            "file_id": 4,
            "offset": 4096,
            "size": 4,
            "shape": [1],
            "dtype": "float32",
            "sha256": "c" * 64,
        },
        {
            "tensor_id": 101,
            "file_id": 4,
            "offset": 0,
            "size": 2,
            "shape": [1],
            "dtype": "bfloat16",
            "sha256": "d" * 64,
        },
    ]
    path = tmp_path / "derivative-index.v1.json"
    write_derivative_index(path, "g00000001", 100, 3, records)
    expected = {
        "schema_version": 1,
        "generation": "g00000001",
        "canonical_max_tensor_id": 100,
        "canonical_max_file_id": 3,
        "records": sorted(records, key=lambda row: row["tensor_id"]),
    }
    assert path.read_bytes() == (
        json.dumps(
            expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
    )
    native = RecordingOverlayHandle()
    load_derivative_overlay(path, native)
    assert native.calls == [
        ("begin", "g00000001", 100, 3),
        ("register", "g00000001", 101, 4, 0, 2, [1], "bfloat16"),
        ("register", "g00000001", 102, 4, 4096, 4, [1], "float32"),
        ("commit", "g00000001"),
    ]


def test_manifest_digest_mismatch_registers_nothing(tmp_path):
    root = tmp_path / "adaptive_derivatives"
    generation = root / "generations" / "g00000001"
    generation.mkdir(parents=True)
    index_bytes = b"{}\n"
    attestation_bytes = b"{}\n"
    manifest_bytes = b"{}\n"
    (generation / "derivative-index.v1.json").write_bytes(index_bytes)
    (generation / "quality-attestation.v1.json").write_bytes(attestation_bytes)
    (generation / "manifest.v1.json").write_bytes(manifest_bytes)
    (root / "CURRENT").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "generation": "g00000001",
                "derivative_index_sha256": hashlib.sha256(
                    index_bytes
                ).hexdigest(),
                "quality_attestation_sha256": hashlib.sha256(
                    attestation_bytes
                ).hexdigest(),
                "manifest_sha256": "0" * 64,
            }
        )
    )
    native = RecordingOverlayHandle()
    with pytest.raises(ValueError, match="manifest_digest_mismatch"):
        ExpertVariantManifest.load_current(root, native_handle=native)
    assert native.calls == []


def test_overlay_registration_failure_aborts_without_commit(
    valid_derivative_index,
):
    native = RecordingOverlayHandle(fail_tensor_id=102)
    with pytest.raises(RuntimeError, match="injected registration failure"):
        load_derivative_overlay(valid_derivative_index, native)
    assert native.calls[-1] == ("abort", "g00000001")
    assert all(call[0] != "commit" for call in native.calls)


def test_checkpoint_fingerprint_uses_canonical_snapshot_binding(tmp_path):
    class SnapshotHandle:
        def __init__(self):
            self.calls = 0

        def get_canonical_tensor_index_snapshot(self):
            self.calls += 1
            return [
                {
                    "tensor_id": 2,
                    "dtype": "float32",
                    "shape": [1],
                    "size": 4,
                    "file_id": 0,
                    "offset": 4096,
                },
                {
                    "tensor_id": 1,
                    "dtype": "bfloat16",
                    "shape": [1],
                    "size": 2,
                    "file_id": 0,
                    "offset": 0,
                },
            ]

    (tmp_path / "archer_param_0").write_bytes(b"partition")
    native = SnapshotHandle()
    one = compute_checkpoint_fingerprint(
        native, {"model_type": "qwen3_moe"}, tmp_path
    )
    two = compute_checkpoint_fingerprint(
        native, {"model_type": "qwen3_moe"}, tmp_path
    )
    assert native.calls == 2
    assert one == two
    assert one["canonical_max_tensor_id"] == 2
    assert one["canonical_max_file_id"] == 0
    assert set(one) == {
        "schema_version",
        "checkpoint_fingerprint",
        "tensor_index_sha256",
        "model_signature_sha256",
        "canonical_max_tensor_id",
        "canonical_max_file_id",
        "partitions",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update({"unknown": 1}),
        lambda doc: doc["records"][0].update({"dtype": "6"}),
        lambda doc: doc["records"][0].update({"size": 3}),
        lambda doc: doc["records"].append(
            {**doc["records"][0], "tensor_id": 103}
        ),
    ],
)
def test_derivative_index_schema_rejects_unknown_dtype_size_and_overlap(
    valid_derivative_index, mutate
):
    document = json.loads(valid_derivative_index.read_text())
    mutate(document)
    valid_derivative_index.write_bytes(canonical_json_bytes(document))
    native = RecordingOverlayHandle()
    with pytest.raises(ValueError):
        load_derivative_overlay(valid_derivative_index, native)
    assert native.calls == []
