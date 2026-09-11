# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import pytest

from moe_infinity.runtime.expert_variant_build import (
    DerivativeBuildJournal,
    recover_or_reserve_generation,
)


def test_crash_retry_never_collides_with_canonical_ids(tmp_path):
    first = recover_or_reserve_generation(
        tmp_path,
        canonical_max_tensor_id=100,
        canonical_max_file_id=3,
        tensor_count=6,
        file_count=2,
    )
    assert first.tensor_id_range == (101, 106)
    assert first.file_id_range == (4, 5)
    first.mark_writing({})
    first.generation_dir.joinpath("derivative-index.v1.json.tmp").write_bytes(
        b"partial"
    )
    second = recover_or_reserve_generation(
        tmp_path,
        canonical_max_tensor_id=100,
        canonical_max_file_id=3,
        tensor_count=6,
        file_count=2,
    )
    assert second.generation == first.generation
    assert second.tensor_id_range == first.tensor_id_range
    second.abort_corrupt_generation()
    third = recover_or_reserve_generation(
        tmp_path,
        canonical_max_tensor_id=100,
        canonical_max_file_id=3,
        tensor_count=6,
        file_count=2,
    )
    assert third.tensor_id_range[0] > second.tensor_id_range[1]
    assert third.file_id_range[0] > second.file_id_range[1]


def test_current_is_published_only_after_index_and_manifest_are_durable(
    tmp_path,
):
    journal = DerivativeBuildJournal.reserve(
        tmp_path, 100, 3, 6, 2, "adaptive-expert-v1"
    )
    with pytest.raises(RuntimeError, match="generation is not attested"):
        journal.publish_current()
    assert not (tmp_path / "adaptive_derivatives" / "CURRENT").exists()


def test_publication_order_puts_current_last(tmp_path, monkeypatch):
    # Patch the globals the class methods actually close over, so the
    # patches land even if another test dropped or replaced the module's
    # sys.modules entry.
    build_ns = DerivativeBuildJournal.publish_attested_generation.__globals__

    events = []
    monkeypatch.setitem(
        build_ns,
        "_durable_replace_bytes",
        lambda path, data: events.append(("replace", path.name)),
    )
    monkeypatch.setitem(
        build_ns,
        "_fsync_directory",
        lambda path: events.append(("fsync_dir", path.name)),
    )
    journal = DerivativeBuildJournal.reserve(
        tmp_path, 100, 3, 1, 1, "adaptive-expert-v1"
    )
    journal.publish_attested_generation(
        payloads={4: b"payload"},
        derivative_index={"schema_version": 1},
        quality_attestation={"schema_version": 1},
        manifest={"schema_version": 1},
        current={"schema_version": 1},
    )
    assert events == [
        ("replace", "archer_param_4"),
        ("fsync_dir", tmp_path.name),
        ("replace", "derivative-index.v1.json"),
        ("replace", "quality-attestation.v1.json"),
        ("replace", "manifest.v1.json"),
        ("fsync_dir", "g00000001"),
        ("replace", "CURRENT"),
        ("fsync_dir", "adaptive_derivatives"),
    ]


def test_publish_attested_generation_persists_published_state(tmp_path):
    journal = DerivativeBuildJournal.reserve(
        tmp_path, 100, 3, 1, 1, "adaptive-expert-v1"
    )
    journal.publish_attested_generation(
        payloads={4: b"payload"},
        derivative_index={"schema_version": 1},
        quality_attestation={"schema_version": 1},
        manifest={"schema_version": 1},
        current={"schema_version": 1},
    )
    reopened = recover_or_reserve_generation(
        tmp_path,
        canonical_max_tensor_id=100,
        canonical_max_file_id=3,
        tensor_count=1,
        file_count=1,
    )
    assert reopened.generation != journal.generation
    assert reopened.tensor_id_range[0] > journal.tensor_id_range[1]


def test_durable_replace_fsyncs_temp_before_replace(tmp_path, monkeypatch):
    import moe_infinity.runtime.expert_variant_build as build

    events = []
    monkeypatch.setattr(
        build.os, "fsync", lambda fd: events.append(("fsync", fd))
    )
    monkeypatch.setattr(
        build.os,
        "replace",
        lambda src, dst: events.append(("replace", src.name, dst.name)),
    )
    build._durable_replace_bytes(tmp_path / "manifest.v1.json", b"{}\n")
    assert events[0][0] == "fsync"
    assert events[1] == ("replace", "manifest.v1.json.tmp", "manifest.v1.json")
