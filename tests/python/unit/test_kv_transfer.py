import torch

from moe_infinity.engine.kv_transfer import (
    KV_FORMAT_VERSION,
    CopyTicket,
    KVObjectKey,
    KVObjectMetadata,
    PageableCPUBufferRecord,
    PinnedBufferPool,
    validate_metadata,
)


def test_pool_enforces_cap_and_reuses_exact_shape() -> None:
    allocations: list[torch.Tensor] = []

    def allocate(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype)
        allocations.append(tensor)
        return tensor

    pool = PinnedBufferPool(capacity_bytes=64, allocator=allocate)
    first = pool.acquire((8,), torch.float32)
    second = pool.acquire((8,), torch.float32)
    third = pool.acquire((1,), torch.float32)
    assert first is not None and second is not None
    assert third is None
    assert pool.in_use_bytes == 64
    pool.release(first)
    reused = pool.acquire((8,), torch.float32)
    assert reused is not None
    assert reused.lease_id != first.lease_id
    assert len(allocations) == 2


def test_pageable_record_owns_and_releases_blocking_clone() -> None:
    source = torch.arange(8, dtype=torch.float32)
    record = PageableCPUBufferRecord.from_blocking_clone(source)
    owned = record.require_tensor()
    assert owned.device.type == "cpu"
    assert not owned.is_pinned()
    assert owned.data_ptr() != source.data_ptr()
    torch.testing.assert_close(owned, source)
    assert record.nbytes == source.numel() * source.element_size()
    record.release()
    assert record.tensor is None
    try:
        record.require_tensor()
    except RuntimeError as exc:
        assert "released" in str(exc)
    else:
        raise AssertionError(
            "released pageable record must not expose a tensor"
        )


def make_metadata(**updates: object) -> tuple[KVObjectMetadata, torch.Tensor]:
    payload = torch.zeros((1, 2, 2, 4, 1, 8), dtype=torch.float16)
    values: dict[str, object] = {
        "format_version": KV_FORMAT_VERSION,
        "key": KVObjectKey(seq_id=3, generation=11),
        "shape": tuple(payload.shape),
        "dtype": payload.dtype,
        "nbytes": payload.numel() * payload.element_size(),
        "num_tokens": 7,
        "block_count": 2,
        "block_size": 4,
        "valid_tokens_last_block": 3,
        "checksum_crc32": None,
    }
    values.update(updates)
    return KVObjectMetadata(**values), payload


def assert_invalid(field: str, **updates: object) -> None:
    metadata, payload = make_metadata(**updates)
    try:
        validate_metadata(
            metadata,
            payload,
            expected_key=KVObjectKey(3, 11),
            block_ids=[1, 4],
            total_blocks=8,
        )
    except ValueError as exc:
        assert field in str(exc)
    else:
        raise AssertionError(f"{field} mismatch must be rejected")


def test_metadata_rejects_shape_independently() -> None:
    assert_invalid("shape", shape=(1, 3, 2, 4, 1, 8))


def test_metadata_rejects_dtype_independently() -> None:
    assert_invalid("dtype", dtype=torch.bfloat16)


def test_metadata_rejects_byte_length_independently() -> None:
    assert_invalid("nbytes", nbytes=255)


def test_metadata_rejects_block_count_independently() -> None:
    assert_invalid("block_count", block_count=3)


def test_metadata_rejects_generation_independently() -> None:
    assert_invalid("generation", key=KVObjectKey(3, 12))


def test_metadata_rejects_block_bounds_and_duplicates() -> None:
    metadata, payload = make_metadata()
    for block_ids in ([-1, 4], [1, 8], [1, 1]):
        try:
            validate_metadata(
                metadata,
                payload,
                expected_key=KVObjectKey(3, 11),
                block_ids=block_ids,
                total_blocks=8,
            )
        except ValueError as exc:
            assert "block_ids" in str(exc)
        else:
            raise AssertionError("invalid block IDs must be rejected")


def test_metadata_validates_padding_and_token_bounds() -> None:
    assert_invalid("num_tokens", num_tokens=9)
    assert_invalid("block_count", num_tokens=4, block_count=2)
    assert_invalid("valid_tokens_last_block", valid_tokens_last_block=4)
    metadata, payload = make_metadata()
    validate_metadata(
        metadata,
        payload,
        expected_key=KVObjectKey(3, 11),
        block_ids=[1, 4],
        total_blocks=8,
    )


class ManualEvent:
    def __init__(self, done: bool) -> None:
        self.done = done

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.done = True


def test_copy_ticket_cannot_retire_before_completion() -> None:
    event = ManualEvent(done=False)
    staging = torch.ones(4)
    ticket = CopyTicket(
        device=torch.device("cpu"),
        stream=None,
        event=event,
        owned_staging_tensors=(staging,),
        submitted_ns=1,
        nbytes=staging.numel() * staging.element_size(),
    )
    assert ticket.retire() is False
    assert ticket.owned_staging_tensors[0] is staging
    event.done = True
    assert ticket.retire() is True
    assert ticket.retired
    assert ticket.owned_staging_tensors == ()
