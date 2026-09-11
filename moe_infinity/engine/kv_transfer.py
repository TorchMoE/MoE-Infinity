"""Shared KV transfer primitives for asynchronous hierarchical KV swap.

This module defines the storage/DMA vocabulary shared by the serving
``PagedKVCache`` and the native-engine coordinator: transfer/block state
enums, metadata with independent validation, a sync-only pageable host
record, an async-only bounded pinned-buffer pool with leases, copy tickets
that own their event/staging tensors, and sync/CUDA transfer backends.

It intentionally contains no scheduler, no request-lifecycle state, and no
external-store implementation. ``ExternalKVStore`` is a future extension
boundary only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

import torch

KV_FORMAT_VERSION = 1


class KVTransferState(str, Enum):
    """Storage/DMA state of one sequence's KV. Never a request lifecycle."""

    GPU_RESIDENT = "gpu_resident"
    SWAP_OUT_IN_FLIGHT = "swap_out_in_flight"
    HOST_RESIDENT = "host_resident"
    SWAP_IN_IN_FLIGHT = "swap_in_in_flight"
    CANCEL_PENDING = "cancel_pending"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BlockOwnership(str, Enum):
    FREE = "free"
    ALLOCATED = "allocated"
    EVICTING = "evicting"
    RESTORING = "restoring"


@dataclass(frozen=True)
class KVObjectKey:
    seq_id: int
    generation: int


@dataclass(frozen=True)
class KVObjectMetadata:
    format_version: int
    key: KVObjectKey
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    num_tokens: int
    block_count: int
    block_size: int
    valid_tokens_last_block: int
    checksum_crc32: int | None


@dataclass
class PinnedBufferLease:
    tensor: torch.Tensor
    nbytes: int
    lease_id: int


@dataclass
class PageableCPUBufferRecord:
    """Sync-only owner of one pageable host copy; never belongs to a pool."""

    tensor: torch.Tensor | None
    nbytes: int

    @classmethod
    def from_blocking_clone(
        cls, source: torch.Tensor
    ) -> "PageableCPUBufferRecord":
        tensor = source.detach().to("cpu").clone()
        return cls(tensor=tensor, nbytes=tensor.numel() * tensor.element_size())

    def require_tensor(self) -> torch.Tensor:
        if self.tensor is None:
            raise RuntimeError("pageable CPU buffer has been released")
        return self.tensor

    def release(self) -> None:
        self.tensor = None


class CompletionEvent(Protocol):
    def query(self) -> bool: ...
    def synchronize(self) -> None: ...


@dataclass
class CopyTicket:
    device: torch.device
    stream: "torch.cuda.Stream | None"
    event: CompletionEvent
    owned_staging_tensors: tuple[torch.Tensor, ...]
    submitted_ns: int
    nbytes: int
    retired: bool = field(default=False, init=False)

    def query(self) -> bool:
        return self.event.query()

    def wait_on_consumer(self, consumer_stream: "torch.cuda.Stream") -> None:
        if self.retired:
            raise RuntimeError("cannot wait on a retired copy ticket")
        consumer_stream.wait_event(self.event)

    def retire(self, *, synchronize: bool = False) -> bool:
        if self.retired:
            return True
        if synchronize:
            self.event.synchronize()
        elif not self.event.query():
            return False
        self.owned_staging_tensors = ()
        self.retired = True
        return True


class KVTransferBackend(Protocol):
    @property
    def asynchronous(self) -> bool: ...

    def submit_d2h(
        self,
        source_cache: torch.Tensor,
        destination: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket: ...

    def submit_h2d(
        self,
        source: torch.Tensor,
        destination_cache: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket: ...

    def close(self) -> None: ...


class ExternalKVStore(Protocol):
    """Future host↔external boundary; no implementation in this change."""

    def submit_put(
        self, key: KVObjectKey, source: memoryview, metadata: KVObjectMetadata
    ) -> str: ...
    def submit_get(
        self,
        key: KVObjectKey,
        destination: memoryview,
        metadata: KVObjectMetadata,
    ) -> str: ...
    def poll(self, operation_id: str) -> str: ...
    def cancel(self, operation_id: str) -> bool: ...
    def delete(self, key: KVObjectKey) -> None: ...
    def close(self) -> None: ...


@dataclass
class KVSwapReservation:
    direction: str  # exactly "out" or "in"
    keys: tuple[KVObjectKey, ...]
    host_leases: dict[KVObjectKey, PinnedBufferLease]  # async mode only
    block_ids: dict[KVObjectKey, list[int]]
    total_nbytes: int
    submitted: bool = False


@dataclass(frozen=True)
class KVTransferCompletion:
    seq_id: int
    generation: int
    direction: str
    success: bool
    cancelled: bool
    error: str | None
    bytes_transferred: int


class PinnedBufferPool:
    """Bounded pinned-host buffer pool for async KV swap.

    A hard byte cap governs new allocations. Exhaustion returns ``None`` as
    backpressure; the pool never silently allocates pageable memory or
    exceeds ``capacity_bytes``. Released buffers of an exact ``(shape, dtype)``
    are cached and reused without re-allocation.
    """

    def __init__(
        self,
        capacity_bytes: int,
        allocator: "Callable[[tuple[int, ...], torch.dtype], torch.Tensor] | None" = None,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be > 0")
        self.capacity_bytes = capacity_bytes
        self._allocator = allocator or (
            lambda shape, dtype: torch.empty(
                shape, dtype=dtype, pin_memory=True
            )
        )
        self._cached: dict[
            tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]
        ] = {}
        self._leased: dict[int, PinnedBufferLease] = {}
        self._allocated_bytes = 0
        self._next_lease_id = 1
        self.peak_in_use_bytes = 0
        self.backpressure_total = 0

    @property
    def in_use_bytes(self) -> int:
        return sum(lease.nbytes for lease in self._leased.values())

    def acquire(self, shape, dtype) -> PinnedBufferLease | None:
        key = (tuple(shape), dtype)
        nbytes = int(torch.empty((), dtype=dtype).element_size())
        for dimension in shape:
            nbytes *= int(dimension)
        cached = self._cached.get(key, [])
        if cached:
            tensor = cached.pop()
        elif self._allocated_bytes + nbytes <= self.capacity_bytes:
            tensor = self._allocator(tuple(shape), dtype)
            self._allocated_bytes += nbytes
        else:
            self.backpressure_total += 1
            return None
        lease = PinnedBufferLease(tensor, nbytes, self._next_lease_id)
        self._next_lease_id += 1
        self._leased[lease.lease_id] = lease
        self.peak_in_use_bytes = max(self.peak_in_use_bytes, self.in_use_bytes)
        return lease

    def release(self, lease: PinnedBufferLease) -> None:
        owned = self._leased.pop(lease.lease_id, None)
        if owned is None:
            raise ValueError(f"unknown or released lease {lease.lease_id}")
        key = (tuple(owned.tensor.shape), owned.tensor.dtype)
        self._cached.setdefault(key, []).append(owned.tensor)

    def close(self) -> None:
        """Drop all cached buffers. Leased buffers must already be released."""
        self._cached.clear()


def validate_metadata(
    metadata: KVObjectMetadata,
    payload: torch.Tensor,
    *,
    expected_key: KVObjectKey,
    block_ids: list[int],
    total_blocks: int,
) -> None:
    """Independently validate every KV metadata invariant.

    Each invariant raises a ``ValueError`` whose message contains the failing
    field name so callers can diagnose exactly one violation at a time. The
    padded payload (full blocks) is validated; attention later uses only
    ``num_tokens`` and never interprets padding as valid KV.
    """
    if metadata.format_version != KV_FORMAT_VERSION:
        raise ValueError(
            f"format_version mismatch: {metadata.format_version} "
            f"!= {KV_FORMAT_VERSION}"
        )
    if metadata.key != expected_key:
        raise ValueError(
            "generation/key mismatch: metadata key "
            f"{metadata.key} != expected {expected_key}"
        )
    payload_shape = tuple(payload.shape)
    if tuple(metadata.shape) != payload_shape or len(payload_shape) != 6:
        raise ValueError(
            f"shape mismatch: metadata {tuple(metadata.shape)} vs payload "
            f"{payload_shape} (must be 6-dimensional serving layout)"
        )
    if metadata.dtype != payload.dtype:
        raise ValueError(
            f"dtype mismatch: metadata {metadata.dtype} != payload "
            f"{payload.dtype}"
        )
    actual_nbytes = payload.numel() * payload.element_size()
    if metadata.nbytes != actual_nbytes:
        raise ValueError(
            f"nbytes mismatch: metadata {metadata.nbytes} != payload "
            f"{actual_nbytes}"
        )
    if metadata.block_size <= 0:
        raise ValueError(f"block_size must be positive: {metadata.block_size}")
    if len(block_ids) != metadata.block_count:
        raise ValueError(
            f"block_count mismatch: metadata {metadata.block_count} != "
            f"len(block_ids) {len(block_ids)}"
        )
    for block_id in block_ids:
        if block_id < 0 or block_id >= total_blocks:
            raise ValueError(
                f"block_ids out of range: {block_id} not in "
                f"[0, {total_blocks})"
            )
    if len(set(block_ids)) != len(block_ids):
        raise ValueError(f"block_ids contain duplicates: {block_ids}")
    capacity = metadata.block_count * metadata.block_size
    if metadata.num_tokens < 0 or metadata.num_tokens > capacity:
        raise ValueError(
            f"num_tokens out of bounds: {metadata.num_tokens} not in "
            f"[0, {capacity}]"
        )
    expected_block_count = (
        0
        if metadata.num_tokens == 0
        else math.ceil(metadata.num_tokens / metadata.block_size)
    )
    if metadata.block_count != expected_block_count:
        raise ValueError(
            f"block_count mismatch: {metadata.block_count} != "
            f"ceil({metadata.num_tokens}/{metadata.block_size}) "
            f"= {expected_block_count}"
        )
    if metadata.block_count == 0:
        expected_padding = 0
    else:
        expected_padding = ((metadata.num_tokens - 1) % metadata.block_size) + 1
    if metadata.valid_tokens_last_block != expected_padding:
        raise ValueError(
            "valid_tokens_last_block mismatch: "
            f"{metadata.valid_tokens_last_block} != {expected_padding}"
        )


class _ImmediateEvent:
    """A completion event that is always done, for the synchronous backend."""

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class SyncKVTransferBackend:
    """Blocking pageable transfer backend.

    Performs only blocking tensor copies and immediate-ticket signaling. It
    must never construct, retain, acquire, or release a ``PinnedBufferPool`` or
    ``PinnedBufferLease``; sync host allocation is owned by the cache through
    ``PageableCPUBufferRecord``.
    """

    @property
    def asynchronous(self) -> bool:
        return False

    def submit_d2h(
        self,
        source_cache: torch.Tensor,
        destination: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        idx = torch.tensor(
            block_ids, dtype=torch.long, device=source_cache.device
        )
        gathered = source_cache.index_select(block_dim, idx)
        destination.copy_(gathered)
        return CopyTicket(
            device=source_cache.device,
            stream=None,
            event=_ImmediateEvent(),
            owned_staging_tensors=(),
            submitted_ns=time.monotonic_ns(),
            nbytes=destination.numel() * destination.element_size(),
        )

    def submit_h2d(
        self,
        source: torch.Tensor,
        destination_cache: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        idx = torch.tensor(
            block_ids, dtype=torch.long, device=destination_cache.device
        )
        staging = source.to(
            device=destination_cache.device, dtype=destination_cache.dtype
        )
        destination_cache.index_copy_(block_dim, idx, staging)
        return CopyTicket(
            device=destination_cache.device,
            stream=None,
            event=_ImmediateEvent(),
            owned_staging_tensors=(),
            submitted_ns=time.monotonic_ns(),
            nbytes=staging.numel() * staging.element_size(),
        )

    def close(self) -> None:
        return None


class CudaKVTransferBackend:
    """Asynchronous pinned-host CUDA transfer backend.

    Uses one CUDA transfer stream per device, nonblocking copies, and CUDA
    events. D2H allocates no host tensor (the destination is a caller-owned
    pinned buffer). Producer→transfer→consumer ordering is enforced via
    ``wait_stream``/``record_stream`` and completion events.
    """

    def __init__(self, device) -> None:
        self._device = torch.device(device)
        self._transfer_stream = torch.cuda.Stream(self._device)

    @property
    def asynchronous(self) -> bool:
        return True

    def submit_d2h(
        self,
        source_cache: torch.Tensor,
        destination: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        # Ensure prior KV producer writes happen-before the gather.
        self._transfer_stream.wait_stream(
            torch.cuda.current_stream(self._device)
        )
        with torch.cuda.stream(self._transfer_stream):
            idx = torch.tensor(
                block_ids, dtype=torch.long, device=source_cache.device
            )
            staging = source_cache.index_select(block_dim, idx).contiguous()
            destination.copy_(staging, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._transfer_stream)
            staging.record_stream(self._transfer_stream)
        return CopyTicket(
            device=self._device,
            stream=self._transfer_stream,
            event=event,
            owned_staging_tensors=(staging,),
            submitted_ns=time.monotonic_ns(),
            nbytes=staging.numel() * staging.element_size(),
        )

    def submit_h2d(
        self,
        source: torch.Tensor,
        destination_cache: torch.Tensor,
        *,
        block_ids: list[int],
        block_dim: int,
    ) -> CopyTicket:
        with torch.cuda.stream(self._transfer_stream):
            staging = source.to(self._device, non_blocking=True).contiguous()
            idx = torch.tensor(
                block_ids, dtype=torch.long, device=destination_cache.device
            )
            destination_cache.index_copy_(block_dim, idx, staging)
            event = torch.cuda.Event()
            event.record(self._transfer_stream)
            staging.record_stream(self._transfer_stream)
        return CopyTicket(
            device=self._device,
            stream=self._transfer_stream,
            event=event,
            owned_staging_tensors=(staging,),
            submitted_ns=time.monotonic_ns(),
            nbytes=staging.numel() * staging.element_size(),
        )

    def close(self) -> None:
        if torch.cuda.is_available():
            self._transfer_stream.synchronize()
