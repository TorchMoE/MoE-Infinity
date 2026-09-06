from __future__ import annotations

import heapq
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

import torch

from moe_infinity.engine.kv_transfer import (
    KV_FORMAT_VERSION,
    BlockOwnership,
    CopyTicket,
    KVObjectKey,
    KVObjectMetadata,
    KVSwapReservation,
    KVTransferBackend,
    KVTransferCompletion,
    KVTransferState,
    PageableCPUBufferRecord,
    PinnedBufferLease,
    PinnedBufferPool,
    SyncKVTransferBackend,
    validate_metadata,
)
from moe_infinity.runtime import flashinfer_utils
from moe_infinity.runtime.kv_cache_format import (
    FormatLayeredKVStore,
    LayeredKVPageChunk,
    allocate_layered_paged_kv_store,
)
from moe_infinity.serving.prefix_contract import PrefixLease

_BLOCK_DIM = 1
_logger = logging.getLogger(__name__)


def _new_swap_counters() -> dict[str, int | float]:
    return {
        "swap_out_started_total": 0,
        "swap_out_completed_total": 0,
        "swap_out_failed_total": 0,
        "swap_in_started_total": 0,
        "swap_in_completed_total": 0,
        "swap_in_failed_total": 0,
        "cancelled_total": 0,
        "checksum_failures_total": 0,
        "d2h_bytes_total": 0,
        "h2d_bytes_total": 0,
        "d2h_duration_ms_sum": 0.0,
        "h2d_duration_ms_sum": 0.0,
    }


def _contiguous_ascending(values: list[int]) -> bool:
    return all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))


class _PagedCacheTensor(torch.Tensor):
    """KV cache storage whose block-axis list gathers stay writable views.

    A block table is a contiguous ascending run of block IDs, so rewriting a
    ``[:, [b0, b1, ...], ...]`` gather into an equivalent ``slice`` keeps the
    result a storage-sharing view. This preserves in-place ``copy_`` writes on
    gathered blocks that PyTorch advanced indexing would otherwise discard.
    """

    @staticmethod
    def __new__(cls, data: torch.Tensor) -> "_PagedCacheTensor":
        return data.as_subclass(cls)

    def __getitem__(self, index):  # type: ignore[override]
        if (
            isinstance(index, tuple)
            and len(index) >= 2
            and isinstance(index[1], list)
            and index[1]
            and all(isinstance(item, int) for item in index[1])
            and _contiguous_ascending(index[1])
        ):
            start = index[1][0]
            stop = index[1][-1] + 1
            rewritten = (index[0], slice(start, stop), *index[2:])
            return torch.Tensor.__getitem__(self, rewritten)
        return torch.Tensor.__getitem__(self, index)


if TYPE_CHECKING:
    from moe_infinity.runtime.attention_backend import (
        LayeredPagedKVCheckpoint,
        LayeredPagedKVStore,
        PagedAttentionBackend,
    )

if TYPE_CHECKING:
    from moe_infinity.runtime.attention_backend import (
        LayeredPagedKVCheckpoint,
        LayeredPagedKVStore,
    )


@dataclass(frozen=True)
class _WritableRange:
    new_block_ids: tuple[int, ...]
    from moe_infinity.runtime.paged_kv_storage import PagedKVStorage


class CPAwareKVManager(Protocol):
    def notify_blocks_allocated(
        self, seq_id: int, block_hashes: list[int]
    ) -> None: ...

    def notify_blocks_freed(
        self, seq_id: int, block_hashes: list[int]
    ) -> None: ...


class _FlashinferPrefillWrapperLike(Protocol):
    def plan(self, *args: object, **kwargs: object) -> None: ...

    def run(
        self, query: torch.Tensor, kv_cache: torch.Tensor
    ) -> torch.Tensor: ...


class _FlashinferDecodeWrapperLike(Protocol):
    def plan(self, *args: object, **kwargs: object) -> None: ...

    def run(
        self, query: torch.Tensor, kv_cache: torch.Tensor
    ) -> torch.Tensor: ...


class _FlashinferModuleLike(Protocol):
    BatchPrefillWithPagedKVCacheWrapper: Callable[
        [torch.Tensor, str], _FlashinferPrefillWrapperLike
    ]
    BatchDecodeWithPagedKVCacheWrapper: Callable[
        [torch.Tensor, str], _FlashinferDecodeWrapperLike
    ]


@dataclass(frozen=True)
class SequenceAllocationPlan:
    seq_id: int
    total_tokens: int
    prefix_tokens: int
    pinned_block_ids: list[int]


@dataclass
class GroupAllocationReceipt:
    owner: object
    seq_ids: list[int]
    new_block_ids: list[int]
    staged_tables: "dict[int, BlockTable]"
    leases: tuple[PrefixLease, ...]
    state: str = "prepared"


@dataclass
class BlockAllocator:
    num_blocks: int
    block_size: int
    device: torch.device
    _free_block_heap: list[int] = field(init=False, repr=False)
    _free_block_set: set[int] = field(init=False, repr=False)
    _ownership: dict[int, BlockOwnership] = field(init=False, repr=False)
    _owner_seq_id: dict[int, int] = field(init=False, repr=False)
    _ref_counts: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {self.num_blocks}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self._free_block_heap = list(range(self.num_blocks))
        heapq.heapify(self._free_block_heap)
        self._free_block_set = set(self._free_block_heap)
        self._ownership = {}
        self._owner_seq_id = {}
        self._ref_counts = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_heap)

    def ref_count(self, block_id: int) -> int:
        return self._ref_counts.get(block_id, 0)

    def allocate(
        self, num_blocks: int, owner_seq_id: int | None = None
    ) -> list[int]:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be >= 0, got {num_blocks}")
        if num_blocks == 0:
            return []
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(
                f"BlockAllocator exhausted: requested {num_blocks} blocks but only {self.num_free_blocks} available"
            )

        allocated: list[int] = []
        for _ in range(num_blocks):
            block_id = heapq.heappop(self._free_block_heap)
            self._free_block_set.remove(block_id)
            self._ownership[block_id] = BlockOwnership.ALLOCATED
            if owner_seq_id is not None:
                self._owner_seq_id[block_id] = owner_seq_id
            self._ref_counts[block_id] = 1
            allocated.append(block_id)
        return allocated

    def retain(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            if not 0 <= block_id < self.num_blocks:
                raise ValueError(
                    f"invalid block id {block_id}; expected [0, {self.num_blocks})"
                )
            if self._ref_counts.get(block_id, 0) <= 0:
                raise ValueError(
                    f"cannot retain block id {block_id} with no live reference"
                )
        for block_id in block_ids:
            self._ref_counts[block_id] += 1

    def release(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            if not 0 <= block_id < self.num_blocks:
                raise ValueError(
                    f"invalid block id {block_id}; expected [0, {self.num_blocks})"
                )
            if self._ref_counts.get(block_id, 0) <= 0:
                raise ValueError(f"block id {block_id} is already free")
        for block_id in block_ids:
            self._ref_counts[block_id] -= 1
            if self._ref_counts[block_id] == 0:
                del self._ref_counts[block_id]
                heapq.heappush(self._free_block_heap, block_id)
                self._free_block_set.add(block_id)
                self._ownership[block_id] = BlockOwnership.FREE
                _ = self._owner_seq_id.pop(block_id, None)

    def free(self, block_ids: list[int]) -> None:
        self.release(block_ids)

    def set_ownership(
        self, block_ids: list[int], ownership: BlockOwnership
    ) -> None:
        for block_id in block_ids:
            self._ownership[block_id] = ownership

    def ownership_of(self, block_id: int) -> BlockOwnership:
        return self._ownership.get(block_id, BlockOwnership.FREE)


@dataclass
class BlockTable:
    block_allocator: BlockAllocator
    seq_id: int | None = None
    _block_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _num_tokens: int = field(default=0, init=False, repr=False)

    @property
    def block_size(self) -> int:
        return self.block_allocator.block_size

    def append_token(self) -> None:
        if self._num_tokens % self.block_size == 0:
            new_block_ids = self.block_allocator.allocate(
                1, owner_seq_id=self.seq_id
            )
            self._block_ids.append(new_block_ids[0])
        self._num_tokens += 1

    def get_block_ids(self) -> list[int]:
        return list(self._block_ids)

    def num_computed_tokens(self) -> int:
        return self._num_tokens

    def ensure_num_tokens(self, total_tokens: int) -> None:
        if total_tokens < self._num_tokens:
            raise ValueError(
                f"cannot shrink reservation from {self._num_tokens} "
                f"to {total_tokens}"
            )
        current_blocks = len(self._block_ids)
        required_blocks = (
            total_tokens + self.block_size - 1
        ) // self.block_size
        new_ids = self.block_allocator.allocate(
            required_blocks - current_blocks
        )
        self._block_ids.extend(new_ids)
        self._num_tokens = total_tokens

    def has_blocks(self) -> bool:
        return bool(self._block_ids)

    def restore_blocks(self, block_ids: list[int], num_tokens: int) -> None:
        self._block_ids = list(block_ids)
        self._num_tokens = num_tokens

    def replace_tail(self, new_block_id: int) -> None:
        if not self._block_ids:
            raise ValueError("cannot replace tail of an empty block table")
        self._block_ids[-1] = new_block_id

    def release(self) -> None:
        if self._block_ids:
            self.block_allocator.free(self._block_ids)
        self._block_ids = []
        self._num_tokens = 0

    def release_blocks_only(self) -> None:
        self._block_ids = []


@dataclass
class _SequenceKVRecord:
    key: KVObjectKey
    state: KVTransferState
    num_tokens: int
    active_block_ids: list[int]
    restoring_block_ids: list[int]
    pageable_buffer: PageableCPUBufferRecord | None
    host_lease: PinnedBufferLease | None
    metadata: KVObjectMetadata | None
    ticket: CopyTicket | None
    retries: int = 0
    cancelled: bool = False
    retire_requested: bool = False
    last_error: str | None = None


@dataclass
class PagedKVCache:
    def __init__(
        self,
        num_blocks: int | None = None,
        block_size: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        transfer_backend: KVTransferBackend | None = None,
        pinned_pool: PinnedBufferPool | None = None,
        host_pool_bytes: int = 0,
        max_inflight_bytes: int = 0,
        host_pool_factory: Callable[[int], PinnedBufferPool] | None = None,
        checksum: bool = False,
        *,
        store: FormatLayeredKVStore | None = None,
        owner_id: str | None = None,
        storage: "PagedKVStorage | None" = None,
    ) -> None:
        self.transfer_backend = transfer_backend
        self.pinned_pool = pinned_pool
        self.host_pool_bytes = host_pool_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self.host_pool_factory = host_pool_factory
        self.checksum = checksum
        self.storage = storage
        self._sequence_tables = {}
        self._kv_records = {}
        self._retiring_records = {}
        self._swapped_storage_buffers = {}
        self._swapped_cpu_buffers = {}
        self._swapped_out_sequences = set()
        self._swapped_num_tokens = {}
        self._next_generation = 1
        self._pinned_pool = None
        self._use_flashinfer = False
        self._fi_workspace = None
        self._fi_prefill = None
        self._fi_decode = None
        self._cp_kv_manager = None
        self._swap_counters = _new_swap_counters()
        self._inflight_backpressure_total = 0
        self._block_store = None
        self._block_store_owner = None
        self._swapped_checkpoints = {}
        self._swapped_page_chunks = {}
        self._released_swapped = set()
        if store is not None:
            if owner_id is not None and store.owner_id != owner_id:
                raise RuntimeError(
                    "PagedKVCache owner_id does not match bound KV store"
                )
            self._store = store
            self.owner_id = store.owner_id
            self.num_blocks = (
                min(int(num_blocks), store.num_pages)
                if num_blocks is not None
                else store.num_pages
            )
            self.block_size = store.block_size
            self.num_layers = store.num_layers
            self.num_heads = store.num_kv_heads
            self.head_dim = store.head_dim
            self.dtype = store.execution_dtype
            self.device = store.payload.device
        else:
            if None in (
                num_blocks,
                block_size,
                num_layers,
                num_heads,
                head_dim,
                dtype,
            ):
                raise ValueError(
                    "PagedKVCache requires either a store or full dimensions"
                )
            self.num_blocks = int(num_blocks)
            self.block_size = int(block_size)
            self.num_layers = int(num_layers)
            self.num_heads = int(num_heads)
            self.head_dim = int(head_dim)
            self.dtype = dtype
            self.device = self._resolve_device(device)
            self._store = allocate_layered_paged_kv_store(
                owner_id=owner_id or f"paged-kv-cache:{id(self)}",
                format_name="native",
                num_layers=self.num_layers,
                num_blocks=self.num_blocks,
                block_size=self.block_size,
                num_kv_heads=self.num_heads,
                head_dim=self.head_dim,
                execution_dtype=self.dtype,
                device=self.device,
            )
            self.owner_id = self._store.owner_id
        self.__post_init__()

    def __post_init__(self) -> None:
        self._sequence_tables: dict[int, BlockTable] = {}
        self._swapped_cpu_buffers: dict[int, torch.Tensor] = {}
        self._swapped_num_tokens: dict[int, int] = {}
        self._swapped_out_sequences: set[int] = set()
        self._swapped_storage_buffers: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._swapped_page_chunks: dict[int, LayeredKVPageChunk] = {}
        self._released_swapped: set[int] = set()
        self._cp_kv_manager: CPAwareKVManager | None = None

        if self.storage is not None:
            self._bind_storage(self.storage)
            return

        self.device = self._resolve_device(self.device)
        self.block_allocator = BlockAllocator(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            device=self.device,
        )
        if self._store is not None:
            self._kv_cache = _PagedCacheTensor(self._store.payload)
        else:
            self._kv_cache = _PagedCacheTensor(
                torch.zeros(
                    (
                        self.num_layers,
                        self.num_blocks,
                        2,
                        self.block_size,
                        self.num_heads,
                        self.head_dim,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        self._backend = self.transfer_backend or SyncKVTransferBackend()

        if not self._backend.asynchronous:
            if self.pinned_pool is not None:
                raise ValueError(
                    "sync transfer backend cannot own a pinned buffer pool"
                )
            self._pinned_pool = None
        else:
            if self.pinned_pool is not None:
                self._pinned_pool = self.pinned_pool
            else:
                factory = self.host_pool_factory or (
                    lambda capacity: PinnedBufferPool(capacity_bytes=capacity)
                )
                self._pinned_pool = factory(self.host_pool_bytes)

        self._use_flashinfer = False
        self._fi_workspace = None
        self._fi_prefill = None
        self._fi_decode = None
        if flashinfer_utils.HAS_FLASHINFER:
            flashinfer_module = flashinfer_utils.get_flashinfer_module()
            if flashinfer_module is not None:
                try:
                    fi_module = cast(_FlashinferModuleLike, flashinfer_module)
                    workspace = flashinfer_utils.get_workspace(self.device)
                    self._fi_workspace = workspace
                    self._fi_prefill = (
                        fi_module.BatchPrefillWithPagedKVCacheWrapper(
                            workspace,
                            "NHD",
                        )
                    )
                    self._fi_decode = (
                        fi_module.BatchDecodeWithPagedKVCacheWrapper(
                            workspace,
                            "NHD",
                        )
                    )
                    self._use_flashinfer = True
                except Exception:
                    self._use_flashinfer = False
                    self._fi_workspace = None
                    self._fi_prefill = None
                    self._fi_decode = None

    @property
    def store(self) -> FormatLayeredKVStore:
        return self._store

    def sequence_residency(self, seq_id: int) -> str:
        if seq_id in self._released_swapped:
            return "swapped"
        return "resident"

    def resize_num_blocks(self, new_num_blocks: int) -> None:
        """Shrink the logical block budget to fit the physical paged store.

        Only shrinks, and only before any sequence is allocated: it rebuilds
        the allocator and logical KV tensor, which would invalidate live block
        ids. See :meth:`set_block_store`, which requires logical <= physical.
        """
        if new_num_blocks <= 0:
            raise ValueError(
                f"new_num_blocks must be > 0, got {new_num_blocks}"
            )
        if self._sequence_tables:
            raise RuntimeError(
                "cannot resize KV blocks while sequences are allocated"
            )
        if new_num_blocks == self.num_blocks:
            return
        if new_num_blocks > self.num_blocks:
            raise ValueError(
                "resize_num_blocks only shrinks the logical budget "
                f"({new_num_blocks} > {self.num_blocks})"
            )

        self.num_blocks = int(new_num_blocks)
        self.block_allocator = BlockAllocator(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            device=self.device,
        )
        self._kv_cache = torch.zeros(
            (
                self.num_layers,
                self.num_blocks,
                2,
                self.block_size,
                self.num_heads,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )

    @property
    def block_store(self) -> "LayeredPagedKVStore":
        return self._require_block_store()

    def _require_block_store(self) -> "LayeredPagedKVStore":
        if self._block_store is None:
            raise RuntimeError("layered KV block store is not bound")
        return self._block_store

    def set_block_store(
        self,
        store: "LayeredPagedKVStore",
        *,
        owner: "PagedAttentionBackend",
    ) -> None:
        if self._block_store is store:
            if self._block_store_owner is not owner:
                raise ValueError("layered KV store owner mismatch")
            return
        if self._sequence_tables or self._swapped_out_sequences:
            raise RuntimeError("block store must be bound before allocation")
        if self._block_store is not None and self._block_store is not store:
            raise RuntimeError(
                "paged KV cache cannot be rebound to another store"
            )
        if (
            getattr(owner, "block_store", None) is not store
            or store.owner is not owner
        ):
            raise ValueError("layered KV store owner mismatch")
        if self.num_blocks > store.num_blocks:
            raise ValueError("logical cache exceeds layered store capacity")

        def _dev_key(device: torch.device) -> tuple[str, int]:
            if device.index is not None:
                return (device.type, device.index)
            if device.type == "cuda":
                return (device.type, torch.cuda.current_device())
            return (device.type, -1)

        expected = (
            self.num_layers,
            self.block_size,
            self.num_heads,
            self.head_dim,
            self.dtype,
            _dev_key(self.device),
        )
        actual = (
            store.num_layers,
            store.block_size,
            store.num_kv_heads,
            store.head_dim,
            store.dtype,
            _dev_key(store.device),
        )
        if actual != expected:
            raise ValueError(
                f"layered KV store geometry mismatch: "
                f"expected={expected}, actual={actual}"
            )
        self._block_store = store
        self._block_store_owner = owner
        self._kv_cache = None

    # ------------------------------------------------------------------ #
    # Sequence allocation and token management                            #
    # ------------------------------------------------------------------ #
    def _bind_storage(self, storage: "PagedKVStorage") -> None:
        from moe_infinity.runtime.paged_kv_storage import canonical_device

        spec = storage.spec
        if self.num_blocks != spec.num_blocks:
            raise ValueError("num_blocks mismatch with bound storage")
        if self.block_size != spec.block_size:
            raise ValueError("block_size mismatch with bound storage")
        if self.num_layers != spec.num_layers:
            raise ValueError("num_layers mismatch with bound storage")
        if self.num_heads != spec.num_kv_heads:
            raise ValueError("num_heads mismatch with bound storage")
        if self.head_dim != spec.head_dim:
            raise ValueError("head_dim mismatch with bound storage")
        if self.dtype != spec.dtype:
            raise ValueError("dtype mismatch with bound storage")
        if self.device is not None and canonical_device(
            self.device
        ) != canonical_device(spec.device):
            raise ValueError("device mismatch with bound storage")

        self.device = spec.device
        self.block_allocator = storage.block_allocator
        self._kv_cache = storage.value_cache
        self._use_flashinfer = False
        self._fi_workspace = None
        self._fi_prefill = None
        self._fi_decode = None

    def _copy_on_write_tail(self, block_table: BlockTable) -> None:
        old = block_table.get_block_ids()[-1]
        if self.block_allocator.ref_count(old) <= 1:
            return
        new = self.block_allocator.allocate(1)[0]
        try:
            store = self._require_block_store()
            payload = store.export_blocks([old])
            store.import_blocks([new], payload)
            block_table.replace_tail(new)
            self.block_allocator.release([old])
        except Exception:
            self.block_allocator.release([new])
            raise

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self._sequence_tables

    def prepare_group(
        self,
        plans: list[SequenceAllocationPlan],
        leases: list[PrefixLease],
    ) -> GroupAllocationReceipt:
        if len(plans) != len(leases):
            raise ValueError("one prefix lease is required per sequence plan")
        seq_ids = [plan.seq_id for plan in plans]
        if len(set(seq_ids)) != len(seq_ids) or any(
            seq_id in self._sequence_tables for seq_id in seq_ids
        ):
            raise ValueError(
                "group sequence ids must be unique and unallocated"
            )
        suffix_counts: list[int] = []
        for plan in plans:
            if plan.prefix_tokens < 0 or plan.prefix_tokens > plan.total_tokens:
                raise ValueError("invalid pinned prefix length")
            if plan.prefix_tokens % self.block_size != 0:
                raise ValueError("pinned prefixes must end on a block boundary")
            if len(plan.pinned_block_ids) != plan.prefix_tokens // (
                self.block_size
            ):
                raise ValueError(
                    "pinned block count does not match prefix length"
                )
            if any(
                self.block_allocator.ref_count(block_id) <= 0
                for block_id in plan.pinned_block_ids
            ):
                raise ValueError("pinned block lost its lease reference")
            suffix = plan.total_tokens - plan.prefix_tokens
            suffix_counts.append(
                (suffix + self.block_size - 1) // self.block_size
            )

        owner = object()
        new_ids: list[int] = []
        staged: dict[int, BlockTable] = {}
        try:
            for plan, lease in zip(plans, leases):
                match = lease.prepare_adoption(owner)
                if match.num_tokens != plan.prefix_tokens or (
                    match.block_ids != tuple(plan.pinned_block_ids)
                ):
                    raise ValueError(
                        "lease match does not match sequence allocation plan"
                    )
            new_ids = self.block_allocator.allocate(sum(suffix_counts))
            cursor = 0
            for plan, count in zip(plans, suffix_counts):
                owned = new_ids[cursor : cursor + count]
                cursor += count
                table = BlockTable(self.block_allocator)
                table.restore_blocks(
                    [*plan.pinned_block_ids, *owned], plan.total_tokens
                )
                staged[plan.seq_id] = table
            return GroupAllocationReceipt(
                owner, list(seq_ids), list(new_ids), staged, tuple(leases)
            )
        except Exception:
            if new_ids:
                self.block_allocator.release(new_ids)
            for lease in reversed(leases):
                if lease.state == "prepared":
                    lease.abort(owner)
                elif lease.state == "open":
                    lease.abort()
            raise

    def commit_group(self, receipt: GroupAllocationReceipt) -> None:
        if receipt.state != "prepared":
            raise RuntimeError("group allocation receipt is not prepared")
        if any(
            not lease.is_prepared_for(receipt.owner) for lease in receipt.leases
        ):
            raise RuntimeError(
                "all group leases must be prepared before commit"
            )
        if any(seq_id in self._sequence_tables for seq_id in receipt.seq_ids):
            raise RuntimeError("group sequence became allocated before commit")
        for lease in receipt.leases:
            lease.commit_adoption(receipt.owner)
        self._sequence_tables.update(receipt.staged_tables)
        for seq_id, block_table in receipt.staged_tables.items():
            if seq_id in self._kv_records:
                continue
            generation = self._next_generation
            self._next_generation += 1
            self._kv_records[seq_id] = _SequenceKVRecord(
                key=KVObjectKey(seq_id=seq_id, generation=generation),
                state=KVTransferState.GPU_RESIDENT,
                num_tokens=block_table.num_computed_tokens(),
                active_block_ids=block_table.get_block_ids(),
                restoring_block_ids=[],
                pageable_buffer=None,
                host_lease=None,
                metadata=None,
                ticket=None,
            )
        receipt.state = "committed"

    def abort_group(self, receipt: GroupAllocationReceipt) -> None:
        if receipt.state != "prepared":
            raise RuntimeError("only a prepared group may be aborted")
        for seq_id in receipt.seq_ids:
            _ = self._sequence_tables.pop(seq_id, None)
        if receipt.new_block_ids:
            self.block_allocator.release(list(receipt.new_block_ids))
        for lease in reversed(receipt.leases):
            if lease.state == "prepared":
                lease.abort(receipt.owner)
            elif lease.state == "open":
                lease.abort()
        receipt.state = "aborted"

    @property
    def has_bound_storage(self) -> bool:
        return self.storage is not None

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> None:
        if seq_id in self._sequence_tables:
            raise ValueError(f"sequence {seq_id} already exists")
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")

        block_table = BlockTable(
            block_allocator=self.block_allocator, seq_id=seq_id
        )
        for _ in range(num_tokens):
            block_table.append_token()
        self._sequence_tables[seq_id] = block_table

        generation = self._next_generation
        self._next_generation += 1
        self._kv_records[seq_id] = _SequenceKVRecord(
            key=KVObjectKey(seq_id=seq_id, generation=generation),
            state=KVTransferState.GPU_RESIDENT,
            num_tokens=num_tokens,
            active_block_ids=block_table.get_block_ids(),
            restoring_block_ids=[],
            pageable_buffer=None,
            host_lease=None,
            metadata=None,
            ticket=None,
        )

        if self._cp_kv_manager is not None:
            try:
                block_hashes = block_table.get_block_ids()
                self._cp_kv_manager.notify_blocks_allocated(
                    seq_id,
                    block_hashes,
                )
            except Exception:
                pass

    def ensure_sequence_capacity(self, seq_id: int, total_tokens: int) -> None:
        if total_tokens < 0:
            raise ValueError(f"total_tokens must be >= 0, got {total_tokens}")
        block_table = self._sequence_tables.get(seq_id)
        if block_table is None:
            block_table = BlockTable(
                block_allocator=self.block_allocator, seq_id=seq_id
            )
            block_table.ensure_num_tokens(total_tokens)
            self._sequence_tables[seq_id] = block_table
        else:
            block_table.ensure_num_tokens(total_tokens)

        record = self._kv_records.get(seq_id)
        if record is None:
            generation = self._next_generation
            self._next_generation += 1
            self._kv_records[seq_id] = _SequenceKVRecord(
                key=KVObjectKey(seq_id=seq_id, generation=generation),
                state=KVTransferState.GPU_RESIDENT,
                num_tokens=total_tokens,
                active_block_ids=block_table.get_block_ids(),
                restoring_block_ids=[],
                pageable_buffer=None,
                host_lease=None,
                metadata=None,
                ticket=None,
            )
        else:
            record.num_tokens = max(record.num_tokens, total_tokens)
            record.active_block_ids = block_table.get_block_ids()

    def get_num_reserved_tokens(self, seq_id: int) -> int:
        block_table = self._sequence_tables.get(seq_id)
        if block_table is None:
            return 0
        return block_table.num_computed_tokens()

    def ensure_writable_range(
        self, seq_id: int, start: int, end: int
    ) -> "_WritableRange":
        if not 0 <= start <= end:
            raise ValueError("invalid writable range")
        self.ensure_sequence_capacity(seq_id, end)
        return _WritableRange(new_block_ids=())

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self._sequence_tables

    def get_block_ids_for_range(
        self, seq_id: int, start: int, end: int
    ) -> list[int]:
        if not 0 <= start <= end <= self.get_num_reserved_tokens(seq_id):
            raise ValueError("range is outside reserved sequence capacity")
        table = self.get_block_table(seq_id)
        first = start // self.block_size
        last = (end + self.block_size - 1) // self.block_size
        return list(dict.fromkeys(table[first:last]))

    def rollback_sequence_reservation(
        self,
        *,
        seq_id: int,
        prior_table_existed: bool,
        prior_block_ids: tuple[int, ...],
        prior_reserved_tokens: int,
        cow_block_ids: tuple[int, ...],
    ) -> None:
        table = self._require_sequence(seq_id)
        current_ids = tuple(table.get_block_ids())
        prior_set = set(prior_block_ids)
        current_set = set(current_ids)
        private_new_ids = tuple(
            block_id for block_id in current_ids if block_id not in prior_set
        )
        table.restore_blocks(list(prior_block_ids), prior_reserved_tokens)
        if private_new_ids:
            self.block_allocator.free(list(private_new_ids))
        _ = cow_block_ids
        _ = current_set
        if not prior_table_existed:
            self._sequence_tables.pop(seq_id, None)

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> None:
        if num_new_tokens < 0:
            raise ValueError(
                f"num_new_tokens must be >= 0, got {num_new_tokens}"
            )
        block_table = self._require_sequence(seq_id)
        for _ in range(num_new_tokens):
            if (
                self._block_store is not None
                and block_table.has_blocks()
                and block_table.num_computed_tokens() % self.block_size != 0
            ):
                self._copy_on_write_tail(block_table)
            block_table.append_token()
        record = self._kv_records.get(seq_id)
        if record is not None:
            record.num_tokens = block_table.num_computed_tokens()
            record.active_block_ids = block_table.get_block_ids()

    def num_tokens(self, seq_id: int) -> int:
        try:
            return self._require_sequence(seq_id).num_computed_tokens()
        except KeyError:
            return 0

    def can_append(self, seq_id: int, num_new_tokens: int) -> bool:
        try:
            block_table = self._require_sequence(seq_id)
        except KeyError:
            return True
        current = block_table.num_computed_tokens()
        block_size = self.block_size
        have = (current + block_size - 1) // block_size
        need = (current + num_new_tokens + block_size - 1) // block_size
        return (need - have) <= self.block_allocator.num_free_blocks

    def truncate_tokens(self, seq_id: int, new_len: int) -> None:
        """Roll a sequence back to ``new_len`` tokens, freeing tail blocks.

        Never grows (raises ``ValueError`` if ``new_len`` exceeds the current
        length); no-op when unchanged. Rollback primitive for serving-path DFlash.
        """
        if new_len < 0:
            raise ValueError(f"new_len must be >= 0, got {new_len}")

        if seq_id in self._released_swapped:
            self._truncate_released_swapped(seq_id, new_len)
            return

        block_table = self._require_sequence(seq_id)
        current = block_table.num_computed_tokens()
        if new_len > current:
            raise ValueError(
                f"truncate_tokens cannot grow sequence {seq_id}: "
                f"new_len {new_len} > current {current}"
            )
        if new_len == current:
            return

        block_size = self.block_size
        blocks_needed = (new_len + block_size - 1) // block_size

        current_block_ids = block_table.get_block_ids()
        freed_block_ids = current_block_ids[blocks_needed:]
        kept_block_ids = current_block_ids[:blocks_needed]
        if freed_block_ids:
            self.block_allocator.free(freed_block_ids)
        block_table.restore_blocks(kept_block_ids, num_tokens=new_len)

        # Keep a host-resident copy + token count consistent with the shrink.
        record = self._kv_records.get(seq_id)
        if record is not None:
            record.num_tokens = new_len
            record.active_block_ids = block_table.get_block_ids()
            if record.state is KVTransferState.HOST_RESIDENT:
                if record.pageable_buffer is not None:
                    buffer = record.pageable_buffer.tensor
                    if buffer is not None:
                        if blocks_needed == 0:
                            record.pageable_buffer.release()
                            record.pageable_buffer = None
                        elif int(buffer.shape[_BLOCK_DIM]) > blocks_needed:
                            trimmed = buffer[:, :blocks_needed, ...].clone()
                            record.pageable_buffer = PageableCPUBufferRecord(
                                tensor=trimmed,
                                nbytes=trimmed.numel() * trimmed.element_size(),
                            )
        if seq_id in self._swapped_out_sequences:
            self._swapped_num_tokens[seq_id] = new_len
            storage_buffer = self._swapped_storage_buffers.get(seq_id)
            if storage_buffer is not None:
                if blocks_needed == 0:
                    _ = self._swapped_storage_buffers.pop(seq_id, None)
                else:
                    key_buffer, value_buffer = storage_buffer
                    if int(key_buffer.shape[1]) > blocks_needed:
                        self._swapped_storage_buffers[seq_id] = (
                            key_buffer[:, :blocks_needed, ...].clone(),
                            value_buffer[:, :blocks_needed, ...].clone(),
                        )
            cpu_buffer = self._swapped_cpu_buffers.get(seq_id)
            if cpu_buffer is not None:
                if blocks_needed == 0:
                    _ = self._swapped_cpu_buffers.pop(seq_id, None)
                elif int(cpu_buffer.shape[1]) > blocks_needed:
                    self._swapped_cpu_buffers[seq_id] = cpu_buffer[
                        :, :blocks_needed, ...
                    ].clone()

        if freed_block_ids and self._cp_kv_manager is not None:
            try:
                self._cp_kv_manager.notify_blocks_freed(seq_id, freed_block_ids)
            except Exception:
                pass

    def _truncate_released_swapped(self, seq_id: int, new_len: int) -> None:
        current = self._swapped_num_tokens.get(seq_id, 0)
        if new_len > current:
            raise ValueError(
                f"truncate_tokens cannot grow sequence {seq_id}: "
                f"new_len {new_len} > current {current}"
            )
        self._swapped_num_tokens[seq_id] = new_len
        block_size = self.block_size
        blocks_needed = (new_len + block_size - 1) // block_size
        chunk = self._swapped_page_chunks.get(seq_id)
        if chunk is None:
            return
        if blocks_needed == 0:
            self._swapped_page_chunks[seq_id] = LayeredKVPageChunk(
                page_ids=tuple(),
                format=chunk.format,
                payload=chunk.payload[:, :0, ...].clone(),
                scales=None
                if chunk.scales is None
                else chunk.scales[:, :0, ...].clone(),
            )
            return
        if int(chunk.payload.shape[1]) > blocks_needed:
            self._swapped_page_chunks[seq_id] = LayeredKVPageChunk(
                page_ids=chunk.page_ids[:blocks_needed],
                format=chunk.format,
                payload=chunk.payload[:, :blocks_needed, ...].clone(),
                scales=None
                if chunk.scales is None
                else chunk.scales[:, :blocks_needed, ...].clone(),
            )

    # ------------------------------------------------------------------ #
    # Destruction / cancellation                                          #
    # ------------------------------------------------------------------ #

    def free_sequence(self, seq_id: int) -> None:
        record = self._kv_records.get(seq_id)
        block_table = self._sequence_tables.get(seq_id)
        if record is None and block_table is None:
            return

        if self._cp_kv_manager is not None and block_table is not None:
            try:
                block_hashes = block_table.get_block_ids()
                self._cp_kv_manager.notify_blocks_freed(seq_id, block_hashes)
            except Exception:
                pass

        state = record.state if record is not None else None

        if state in (
            KVTransferState.SWAP_OUT_IN_FLIGHT,
            KVTransferState.SWAP_IN_IN_FLIGHT,
        ):
            # DMA in flight: tombstone the record; retire in poll_transfers().
            assert record is not None
            record.state = KVTransferState.CANCEL_PENDING
            record.cancelled = True
            record.retire_requested = True
            self._retiring_records[
                (record.key.seq_id, record.key.generation)
            ] = _RetiringEntry(record=record, block_table=block_table)
            _ = self._kv_records.pop(seq_id, None)
            _ = self._sequence_tables.pop(seq_id, None)
            return

        if state in (
            KVTransferState.CANCEL_PENDING,
            KVTransferState.CANCELLED,
        ):
            # Idempotent no-op; retirement continues in poll_transfers().
            _ = self._kv_records.pop(seq_id, None)
            _ = self._sequence_tables.pop(seq_id, None)
            return

        # GPU_RESIDENT / HOST_RESIDENT / FAILED: no ticket owns anything.
        if record is not None:
            self._release_host_storage(record)
            if record.restoring_block_ids:
                self.block_allocator.free(record.restoring_block_ids)
                record.restoring_block_ids = []
        if block_table is not None:
            block_table.release()

        _ = self._swapped_cpu_buffers.pop(seq_id, None)
        _ = self._swapped_storage_buffers.pop(seq_id, None)
        _ = self._swapped_num_tokens.pop(seq_id, None)
        _ = self._swapped_page_chunks.pop(seq_id, None)
        self._swapped_out_sequences.discard(seq_id)
        self._released_swapped.discard(seq_id)
        _ = self._kv_records.pop(seq_id, None)
        _ = self._sequence_tables.pop(seq_id, None)

    def cancel_sequence(self, seq_id: int) -> None:
        self.free_sequence(seq_id)

    def discard_failed_for_reprefill(
        self, seq_id: int, generation: int
    ) -> None:
        record = self._kv_records.get(seq_id)
        if record is None or record.key.generation != generation:
            return
        if record.state is not KVTransferState.FAILED:
            return
        self._release_host_storage(record)
        if record.restoring_block_ids:
            self.block_allocator.free(record.restoring_block_ids)
            record.restoring_block_ids = []
        block_table = self._sequence_tables.get(seq_id)
        if block_table is not None:
            block_table.release()
        _ = self._kv_records.pop(seq_id, None)

    def discard_host_copy(self, seq_id: int, generation: int) -> None:
        record = self._kv_records.get(seq_id)
        if record is None or record.key.generation != generation:
            return
        self._release_host_storage(record)
        record.metadata = None

    # ------------------------------------------------------------------ #
    # Central host-storage release                                        #
    # ------------------------------------------------------------------ #

    def _release_host_storage(self, record: _SequenceKVRecord) -> None:
        if record.pageable_buffer is not None and record.host_lease is not None:
            raise RuntimeError(
                "KV record cannot own pageable and pinned host storage"
            )
        if record.pageable_buffer is not None:
            record.pageable_buffer.release()
            record.pageable_buffer = None
            return
        if record.host_lease is not None:
            if self._pinned_pool is None:
                raise RuntimeError("pinned lease exists without an async pool")
            self._pinned_pool.release(record.host_lease)
            record.host_lease = None

    def set_cp_kv_manager(self, manager: CPAwareKVManager) -> None:
        self._cp_kv_manager = manager

    def free_gpu_blocks(self, seq_id: int) -> None:
        if seq_id in self._released_swapped:
            raise RuntimeError(
                f"sequence {seq_id} does not own GPU blocks (SWAPPED)"
            )
        block_table = self._sequence_tables.get(seq_id)
        if block_table is None:
            return

        block_ids = block_table.get_block_ids()
        if block_ids:
            self.block_allocator.free(block_ids)
            block_table.release_blocks_only()

    def get_block_table(self, seq_id: int) -> list[int]:
        block_table = self._require_sequence(seq_id)
        return block_table.get_block_ids()

    def get_kv_cache_tensors(self) -> torch.Tensor:
        if self._block_store is not None:
            fi_cache = self._block_store.fi_kv_cache
            if fi_cache is not None:
                return fi_cache
            raise RuntimeError(
                "bound layered store has no FlashInfer KV tensor"
            )
        if self._kv_cache is None:
            raise RuntimeError("KV cache tensor is not initialized")
        return self._kv_cache

    # ------------------------------------------------------------------ #
    # Metadata helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_metadata(
        self, key: KVObjectKey, num_tokens: int, block_ids: list[int]
    ) -> KVObjectMetadata:
        block_count = len(block_ids)
        shape = (
            self.num_layers,
            block_count,
            2,
            self.block_size,
            self.num_heads,
            self.head_dim,
        )
        nbytes = 1
        for dim in shape:
            nbytes *= dim
        nbytes *= torch.empty((), dtype=self.dtype).element_size()
        if block_count == 0:
            valid_last = 0
        else:
            valid_last = ((num_tokens - 1) % self.block_size) + 1
        return KVObjectMetadata(
            format_version=KV_FORMAT_VERSION,
            key=key,
            shape=shape,
            dtype=self.dtype,
            nbytes=nbytes,
            num_tokens=num_tokens,
            block_count=block_count,
            block_size=self.block_size,
            valid_tokens_last_block=valid_last,
            checksum_crc32=None,
        )

    # ------------------------------------------------------------------ #
    # Group reservation / submission                                      #
    # ------------------------------------------------------------------ #

    def reserve_swap_out_group(
        self, seq_ids: list[int]
    ) -> KVSwapReservation | None:
        keys: list[KVObjectKey] = []
        block_ids_map: dict[KVObjectKey, list[int]] = {}
        total_nbytes = 0
        for seq_id in seq_ids:
            record = self._kv_records.get(seq_id)
            if record is None or record.state is not (
                KVTransferState.GPU_RESIDENT
            ):
                return None
            block_table = self._sequence_tables.get(seq_id)
            if block_table is None:
                return None
            block_ids = block_table.get_block_ids()
            keys.append(record.key)
            block_ids_map[record.key] = block_ids
            metadata = self._build_metadata(
                record.key, record.num_tokens, block_ids
            )
            record.metadata = metadata
            total_nbytes += metadata.nbytes

        if not self._backend.asynchronous:
            return KVSwapReservation(
                direction="out",
                keys=tuple(keys),
                host_leases={},
                block_ids=block_ids_map,
                total_nbytes=total_nbytes,
            )

        # Async: pre-acquire every pinned lease; roll back all on failure.
        assert self._pinned_pool is not None
        if not self._within_inflight_cap(total_nbytes):
            return None
        host_leases: dict[KVObjectKey, PinnedBufferLease] = {}
        for key in keys:
            block_ids = block_ids_map[key]
            shape = (
                self.num_layers,
                len(block_ids),
                2,
                self.block_size,
                self.num_heads,
                self.head_dim,
            )
            lease = self._pinned_pool.acquire(shape, self.dtype)
            if lease is None:
                for acquired in host_leases.values():
                    self._pinned_pool.release(acquired)
                return None
            host_leases[key] = lease

        return KVSwapReservation(
            direction="out",
            keys=tuple(keys),
            host_leases=host_leases,
            block_ids=block_ids_map,
            total_nbytes=total_nbytes,
        )

    def submit_swap_out_group(self, reservation: KVSwapReservation) -> None:
        if reservation.submitted:
            raise RuntimeError("reservation already submitted")
        if reservation.direction != "out":
            raise ValueError("expected swap-out reservation")
        reservation.submitted = True
        for key in reservation.keys:
            self._submit_swap_out_one(key, reservation)

    def _submit_swap_out_one(
        self, key: KVObjectKey, reservation: KVSwapReservation
    ) -> None:
        seq_id = key.seq_id
        record = self._kv_records[seq_id]
        block_table = self._sequence_tables[seq_id]
        block_ids = reservation.block_ids[key]
        self._swap_counters["swap_out_started_total"] += 1

        if self._block_store is not None:
            # Store-bound mode swaps via layered checkpoints and completes
            # immediately; any pre-acquired pinned lease is returned unused.
            if block_ids:
                self._swapped_checkpoints[seq_id] = (
                    self._block_store.checkpoint(list(block_ids))
                )
            lease = reservation.host_leases.get(key)
            if lease is not None and self._pinned_pool is not None:
                self._pinned_pool.release(lease)
            record.pageable_buffer = None
            record.host_lease = None
            record.state = KVTransferState.HOST_RESIDENT
            record.active_block_ids = list(block_ids)
            record.num_tokens = block_table.num_computed_tokens()
            self._swap_counters["swap_out_completed_total"] += 1
            self._swap_counters["d2h_bytes_total"] += (
                record.metadata.nbytes if record.metadata is not None else 0
            )
            return

        if not self._backend.asynchronous:
            # Sync D2H: blocking pageable clone, immediate completion.
            if block_ids:
                source = self._kv_cache[:, block_ids, ...]
                record.pageable_buffer = (
                    PageableCPUBufferRecord.from_blocking_clone(source)
                )
            else:
                record.pageable_buffer = PageableCPUBufferRecord(
                    tensor=self._kv_cache[:, [], ...]
                    .detach()
                    .to("cpu")
                    .clone(),
                    nbytes=0,
                )
            record.host_lease = None
            record.state = KVTransferState.HOST_RESIDENT
            record.active_block_ids = list(block_ids)
            record.num_tokens = block_table.num_computed_tokens()
            self._swap_counters["swap_out_completed_total"] += 1
            self._swap_counters["d2h_bytes_total"] += (
                record.metadata.nbytes if record.metadata is not None else 0
            )
            return

        # Async D2H: mark EVICTING, submit against pinned lease.
        lease = reservation.host_leases[key]
        self.block_allocator.set_ownership(block_ids, BlockOwnership.EVICTING)
        try:
            ticket = self._backend.submit_d2h(
                self._kv_cache,
                lease.tensor,
                block_ids=block_ids,
                block_dim=_BLOCK_DIM,
            )
        except Exception as exc:  # noqa: BLE001
            self.block_allocator.set_ownership(
                block_ids, BlockOwnership.ALLOCATED
            )
            self._pinned_pool.release(lease)  # type: ignore[union-attr]
            record.host_lease = None
            record.state = KVTransferState.GPU_RESIDENT
            record.last_error = f"{type(exc).__name__}: {exc}"
            self._record_failure("out", record)
            return
        record.host_lease = lease
        record.pageable_buffer = None
        record.ticket = ticket
        record.state = KVTransferState.SWAP_OUT_IN_FLIGHT
        record.active_block_ids = list(block_ids)

    def reserve_swap_in_group(
        self, seq_ids: list[int]
    ) -> KVSwapReservation | None:
        keys: list[KVObjectKey] = []
        block_ids_map: dict[KVObjectKey, list[int]] = {}
        host_leases: dict[KVObjectKey, PinnedBufferLease] = {}
        needed_total = 0
        for seq_id in seq_ids:
            record = self._kv_records.get(seq_id)
            if record is None or record.state is not (
                KVTransferState.HOST_RESIDENT
            ):
                return None
            block_table = self._sequence_tables.get(seq_id)
            if block_table is not None and block_table.has_blocks():
                continue
            needed_total += self._blocks_needed_for(record)

        if needed_total > self.block_allocator.num_free_blocks:
            return None

        total_nbytes = 0
        reserved_all: list[int] = []
        for seq_id in seq_ids:
            record = self._kv_records[seq_id]
            block_table = self._sequence_tables.get(seq_id)
            if block_table is not None and block_table.has_blocks():
                restored = block_table.get_block_ids()
                record.restoring_block_ids = []
            else:
                needed = self._blocks_needed_for(record)
                restored = self.block_allocator.allocate(
                    needed, owner_seq_id=seq_id
                )
                self.block_allocator.set_ownership(
                    restored, BlockOwnership.RESTORING
                )
                reserved_all.extend(restored)
                record.restoring_block_ids = restored
            keys.append(record.key)
            block_ids_map[record.key] = restored
            if record.host_lease is not None:
                host_leases[record.key] = record.host_lease
            metadata = record.metadata or self._build_metadata(
                record.key, record.num_tokens, restored
            )
            total_nbytes += metadata.nbytes

        if self._backend.asynchronous and not self._within_inflight_cap(
            total_nbytes
        ):
            self.block_allocator.free(reserved_all)
            for seq_id in seq_ids:
                self._kv_records[seq_id].restoring_block_ids = []
            return None

        return KVSwapReservation(
            direction="in",
            keys=tuple(keys),
            host_leases=host_leases,
            block_ids=block_ids_map,
            total_nbytes=total_nbytes,
        )

    def submit_swap_in_group(self, reservation: KVSwapReservation) -> None:
        if reservation.submitted:
            raise RuntimeError("reservation already submitted")
        if reservation.direction != "in":
            raise ValueError("expected swap-in reservation")
        reservation.submitted = True
        for key in reservation.keys:
            self._submit_swap_in_one(key, reservation)

    def _submit_swap_in_one(
        self, key: KVObjectKey, reservation: KVSwapReservation
    ) -> None:
        seq_id = key.seq_id
        record = self._kv_records[seq_id]
        block_table = self._sequence_tables[seq_id]
        restored = reservation.block_ids[key]
        self._swap_counters["swap_in_started_total"] += 1

        if self._block_store is not None:
            checkpoint = self._swapped_checkpoints.pop(seq_id, None)
            newly_reserved = record.restoring_block_ids
            try:
                if checkpoint is not None and restored:
                    self._block_store.restore(list(restored), checkpoint)
            except Exception:
                if checkpoint is not None:
                    self._swapped_checkpoints[seq_id] = checkpoint
                if newly_reserved:
                    self.block_allocator.free(newly_reserved)
                    record.restoring_block_ids = []
                record.state = KVTransferState.HOST_RESIDENT
                raise
            self.block_allocator.set_ownership(
                restored, BlockOwnership.ALLOCATED
            )
            block_table.restore_blocks(restored, num_tokens=record.num_tokens)
            record.active_block_ids = list(restored)
            record.restoring_block_ids = []
            record.pageable_buffer = None
            record.host_lease = None
            record.state = KVTransferState.GPU_RESIDENT
            self._swap_counters["swap_in_completed_total"] += 1
            self._swap_counters["h2d_bytes_total"] += (
                record.metadata.nbytes if record.metadata is not None else 0
            )
            return

        if not self._backend.asynchronous:
            assert record.pageable_buffer is not None
            newly_reserved = record.restoring_block_ids
            try:
                host_tensor = record.pageable_buffer.require_tensor()
                if restored:
                    staging = host_tensor.to(
                        device=self._kv_cache.device,
                        dtype=self._kv_cache.dtype,
                    )
                    self._kv_cache[:, restored, ...] = staging
            except Exception:
                if newly_reserved:
                    self.block_allocator.free(newly_reserved)
                    record.restoring_block_ids = []
                self._release_host_storage(record)
                record.state = KVTransferState.HOST_RESIDENT
                raise
            self.block_allocator.set_ownership(
                restored, BlockOwnership.ALLOCATED
            )
            block_table.restore_blocks(restored, num_tokens=record.num_tokens)
            record.active_block_ids = list(restored)
            record.restoring_block_ids = []
            record.pageable_buffer.release()
            record.pageable_buffer = None
            record.state = KVTransferState.GPU_RESIDENT
            self._swap_counters["swap_in_completed_total"] += 1
            self._swap_counters["h2d_bytes_total"] += (
                record.metadata.nbytes if record.metadata is not None else 0
            )
            return

        # Async H2D: validate metadata, submit, publish only after retirement.
        lease = record.host_lease
        assert lease is not None
        metadata = record.metadata or self._build_metadata(
            record.key, record.num_tokens, restored
        )
        try:
            validate_metadata(
                metadata,
                lease.tensor,
                expected_key=record.key,
                block_ids=restored,
                total_blocks=self.num_blocks,
            )
        except ValueError as exc:
            # Metadata/checksum invalid: do not submit H2D. FAILED.
            self.block_allocator.free(restored)
            record.restoring_block_ids = []
            record.state = KVTransferState.FAILED
            record.last_error = f"{type(exc).__name__}: {exc}"
            if "checksum" in str(exc).lower():
                self._swap_counters["checksum_failures_total"] += 1
            self._record_failure("in", record)
            return
        try:
            ticket = self._backend.submit_h2d(
                lease.tensor,
                self._kv_cache,
                block_ids=restored,
                block_dim=_BLOCK_DIM,
            )
        except Exception as exc:  # noqa: BLE001
            self.block_allocator.free(restored)
            record.restoring_block_ids = []
            record.state = KVTransferState.HOST_RESIDENT
            record.last_error = f"{type(exc).__name__}: {exc}"
            self._record_failure("in", record)
            return
        record.ticket = ticket
        record.restoring_block_ids = list(restored)
        record.state = KVTransferState.SWAP_IN_IN_FLIGHT

    # ------------------------------------------------------------------ #
    # One-member wrappers                                                 #
    # ------------------------------------------------------------------ #

    def request_swap_out(self, seq_id: int) -> bool:
        reservation = self.reserve_swap_out_group([seq_id])
        if reservation is None:
            return False
        self.submit_swap_out_group(reservation)
        return self._kv_records[seq_id].state in (
            KVTransferState.SWAP_OUT_IN_FLIGHT,
            KVTransferState.HOST_RESIDENT,
        )

    def request_swap_in(self, seq_id: int) -> bool:
        reservation = self.reserve_swap_in_group([seq_id])
        if reservation is None:
            return False
        self.submit_swap_in_group(reservation)
        return self._kv_records[seq_id].state in (
            KVTransferState.SWAP_IN_IN_FLIGHT,
            KVTransferState.GPU_RESIDENT,
        )

    # ------------------------------------------------------------------ #
    # Completion polling                                                  #
    # ------------------------------------------------------------------ #

    def poll_transfers(self) -> list[KVTransferCompletion]:
        completions: list[KVTransferCompletion] = []

        # Retire tombstoned (cancelled) records first.
        for tomb_key in list(self._retiring_records.keys()):
            entry = self._retiring_records[tomb_key]
            record = entry.record
            ticket = record.ticket
            if ticket is not None and not ticket.retire():
                continue
            self._finalize_cancelled(tomb_key, entry)
            completions.append(
                KVTransferCompletion(
                    seq_id=record.key.seq_id,
                    generation=record.key.generation,
                    direction="cancel",
                    success=False,
                    cancelled=True,
                    error=record.last_error,
                    bytes_transferred=0,
                )
            )

        # Retire active in-flight records.
        for seq_id in list(self._kv_records.keys()):
            record = self._kv_records.get(seq_id)
            if record is None:
                continue
            if record.state is KVTransferState.SWAP_OUT_IN_FLIGHT:
                ticket = record.ticket
                if ticket is None or not ticket.retire():
                    continue
                completion = self._finalize_swap_out(seq_id, record)
                completions.append(completion)
            elif record.state is KVTransferState.SWAP_IN_IN_FLIGHT:
                ticket = record.ticket
                if ticket is None or not ticket.retire():
                    continue
                completion = self._finalize_swap_in(seq_id, record)
                completions.append(completion)

        return completions

    def _finalize_swap_out(
        self, seq_id: int, record: _SequenceKVRecord
    ) -> KVTransferCompletion:
        # D2H complete: release GPU blocks, publish HOST_RESIDENT.
        block_table = self._sequence_tables.get(seq_id)
        ticket = record.ticket
        block_ids = record.active_block_ids
        if block_ids:
            self.block_allocator.free(block_ids)
        if block_table is not None:
            block_table.release_blocks_only()
        record.ticket = None
        record.active_block_ids = []
        record.state = KVTransferState.HOST_RESIDENT
        nbytes = record.host_lease.nbytes if record.host_lease else 0
        self._record_completion("out", ticket, nbytes)
        return KVTransferCompletion(
            seq_id=seq_id,
            generation=record.key.generation,
            direction="out",
            success=True,
            cancelled=False,
            error=None,
            bytes_transferred=nbytes,
        )

    def _finalize_swap_in(
        self, seq_id: int, record: _SequenceKVRecord
    ) -> KVTransferCompletion:
        # H2D complete: publish block table, release host lease.
        block_table = self._sequence_tables.get(seq_id)
        ticket = record.ticket
        restored = record.restoring_block_ids
        self.block_allocator.set_ownership(restored, BlockOwnership.ALLOCATED)
        if block_table is not None:
            block_table.restore_blocks(restored, num_tokens=record.num_tokens)
        record.active_block_ids = list(restored)
        record.restoring_block_ids = []
        record.ticket = None
        nbytes = record.host_lease.nbytes if record.host_lease else 0
        self._release_host_storage(record)
        record.state = KVTransferState.GPU_RESIDENT
        self._record_completion("in", ticket, nbytes)
        return KVTransferCompletion(
            seq_id=seq_id,
            generation=record.key.generation,
            direction="in",
            success=True,
            cancelled=False,
            error=None,
            bytes_transferred=nbytes,
        )

    def _finalize_cancelled(
        self, tomb_key: tuple[int, int], entry: _RetiringEntry
    ) -> None:
        record = entry.record
        # Free source (EVICTING) or destination (RESTORING) blocks.
        if record.active_block_ids:
            self.block_allocator.free(record.active_block_ids)
            record.active_block_ids = []
        if record.restoring_block_ids:
            self.block_allocator.free(record.restoring_block_ids)
            record.restoring_block_ids = []
        record.ticket = None
        self._release_host_storage(record)
        record.state = KVTransferState.CANCELLED
        self._swap_counters["cancelled_total"] += 1
        _ = self._retiring_records.pop(tomb_key, None)

    # ------------------------------------------------------------------ #
    # Queries                                                             #
    # ------------------------------------------------------------------ #

    def transfer_state(self, seq_id: int) -> KVTransferState:
        record = self._kv_records.get(seq_id)
        if record is None:
            raise KeyError(f"unknown sequence id: {seq_id}")
        return record.state

    def is_gpu_ready(self, seq_id: int) -> bool:
        record = self._kv_records.get(seq_id)
        return (
            record is not None and record.state is KVTransferState.GPU_RESIDENT
        )

    def has_pending_transfers(self) -> bool:
        if self._retiring_records:
            return True
        for record in self._kv_records.values():
            if record.state in (
                KVTransferState.SWAP_OUT_IN_FLIGHT,
                KVTransferState.SWAP_IN_IN_FLIGHT,
            ):
                return True
        return False

    def wait_for_transfer_progress(self, timeout_ms: float) -> bool:
        made_progress = False
        for record in list(self._kv_records.values()):
            ticket = record.ticket
            if ticket is not None and record.state in (
                KVTransferState.SWAP_OUT_IN_FLIGHT,
                KVTransferState.SWAP_IN_IN_FLIGHT,
            ):
                ticket.retire(synchronize=True)
                made_progress = True
        for entry in list(self._retiring_records.values()):
            ticket = entry.record.ticket
            if ticket is not None:
                ticket.retire(synchronize=True)
                made_progress = True
        return made_progress

    def get_swap_stats(self) -> dict[str, object]:
        if self._pinned_pool is None:
            host_capacity = 0
            host_in_use = 0
        else:
            host_capacity = self._pinned_pool.capacity_bytes
            host_in_use = self._pinned_pool.in_use_bytes
        records = list(self._kv_records.values()) + [
            entry.record for entry in self._retiring_records.values()
        ]
        inflight_records = [
            record for record in records if record.ticket is not None
        ]
        pool_backpressure = (
            self._pinned_pool.backpressure_total
            if self._pinned_pool is not None
            else 0
        )
        stats: dict[str, object] = {
            "mode": "async" if self._backend.asynchronous else "sync",
            "fallback_reason": None,
            "host_capacity_bytes": host_capacity,
            "host_in_use_bytes": host_in_use,
            "host_peak_in_use_bytes": (
                self._pinned_pool.peak_in_use_bytes
                if self._pinned_pool is not None
                else 0
            ),
            "inflight": len(inflight_records),
            "inflight_bytes": sum(
                record.ticket.nbytes
                for record in inflight_records
                if record.ticket is not None
            ),
            "retiring_records": len(self._retiring_records),
            "host_resident": sum(
                record.state is KVTransferState.HOST_RESIDENT
                for record in self._kv_records.values()
            ),
            "backpressure_total": (
                pool_backpressure + self._inflight_backpressure_total
            ),
        }
        stats.update(self._swap_counters)
        return stats

    def shutdown(self, timeout_ms: float = 5000.0) -> None:
        # Synchronize and retire every unretired ticket.
        for record in list(self._kv_records.values()):
            ticket = record.ticket
            if ticket is not None:
                ticket.retire(synchronize=True)
        for entry in list(self._retiring_records.values()):
            ticket = entry.record.ticket
            if ticket is not None:
                ticket.retire(synchronize=True)

        # Finalize cancelled tombstones.
        for tomb_key in list(self._retiring_records.keys()):
            entry = self._retiring_records[tomb_key]
            self._finalize_cancelled(tomb_key, entry)

        # Release remaining host storage directly.
        for record in list(self._kv_records.values()):
            self._release_host_storage(record)

        self._backend.close()
        if self._pinned_pool is not None:
            self._pinned_pool.close()

    # ------------------------------------------------------------------ #
    # Compatibility wrappers                                              #
    # ------------------------------------------------------------------ #

    def swap_out(self, seq_id: int, release_gpu_blocks: bool = False) -> None:
        if release_gpu_blocks:
            block_table = self._require_sequence(seq_id)
            self._swap_out_release(seq_id, block_table)
            return
        record = self._kv_records.get(seq_id)
        if record is not None and record.state is (
            KVTransferState.HOST_RESIDENT
        ):
            return
        if not self.request_swap_out(seq_id):
            raise RuntimeError(
                f"KV swap-out backpressure for sequence {seq_id}"
            )
        # Sync submission has already completed and published HOST_RESIDENT.
        block_table = self._sequence_tables.get(seq_id)
        if block_table is None:
            return
        self._swapped_num_tokens[seq_id] = block_table.num_computed_tokens()
        block_ids = block_table.get_block_ids()
        if self._block_store is not None:
            if block_ids:
                self._swapped_checkpoints[seq_id] = (
                    self._block_store.checkpoint(list(block_ids))
                )
            self._swapped_out_sequences.add(seq_id)
            return

        if block_ids:
            if self.storage is not None:
                self._swapped_storage_buffers[seq_id] = (
                    self.storage.key_cache[:, block_ids, ...]
                    .detach()
                    .to("cpu")
                    .clone(),
                    self.storage.value_cache[:, block_ids, ...]
                    .detach()
                    .to("cpu")
                    .clone(),
                )
            else:
                assert self._kv_cache is not None
                self._swapped_cpu_buffers[seq_id] = (
                    self._kv_cache[:, block_ids, ...].detach().to("cpu").clone()
                )
        self._swapped_out_sequences.add(seq_id)

    def _swap_out_release(self, seq_id: int, block_table: BlockTable) -> None:
        if seq_id in self._released_swapped:
            return
        num_tokens = block_table.num_computed_tokens()
        block_ids = block_table.get_block_ids()
        chunk = self._store.snapshot_page_chunk(
            list(block_ids), torch.device("cpu")
        )
        if block_ids:
            self.block_allocator.free(list(block_ids))
        block_table.release_blocks_only()
        self._swapped_page_chunks[seq_id] = chunk
        self._swapped_num_tokens[seq_id] = num_tokens
        self._released_swapped.add(seq_id)
        if self._cp_kv_manager is not None and block_ids:
            try:
                self._cp_kv_manager.notify_blocks_freed(seq_id, list(block_ids))
            except Exception:
                pass

    def swap_in(self, seq_id: int) -> None:
        block_table = self._sequence_tables.get(seq_id)
        if seq_id in self._released_swapped and block_table is not None:
            self._swap_in_released(seq_id, block_table)
            return
        record = self._kv_records.get(seq_id)
        if record is not None and record.state is (
            KVTransferState.GPU_RESIDENT
        ):
            return
        if not self.request_swap_in(seq_id):
            raise RuntimeError(f"KV swap-in unavailable for sequence {seq_id}")
        # Sync submission has already restored/published and released bytes.
        block_table = self._sequence_tables.get(seq_id)
        if block_table is None:
            return
        saved_num_tokens = self._swapped_num_tokens.pop(seq_id, 0)
        if self._block_store is not None:
            checkpoint = self._swapped_checkpoints.pop(seq_id, None)
            if checkpoint is not None:
                num_blocks_needed = len(checkpoint.source_block_ids)
                if not block_table.has_blocks():
                    restored_block_ids = self.block_allocator.allocate(
                        num_blocks_needed,
                    )
                    block_table.restore_blocks(
                        restored_block_ids,
                        num_tokens=saved_num_tokens,
                    )
                block_ids = block_table.get_block_ids()
                if block_ids:
                    self._block_store.restore(block_ids, checkpoint)
            self._swapped_out_sequences.discard(seq_id)
            return

        cpu_buffer = self._swapped_cpu_buffers.pop(seq_id, None)
        if self.storage is not None:
            storage_buffer = self._swapped_storage_buffers.pop(seq_id, None)
            if storage_buffer is not None:
                key_buffer, value_buffer = storage_buffer
                if not block_table.has_blocks():
                    num_blocks_needed = int(key_buffer.shape[1])
                    restored_block_ids = self.block_allocator.allocate(
                        num_blocks_needed,
                    )
                    block_table.restore_blocks(
                        restored_block_ids,
                        num_tokens=saved_num_tokens,
                    )
                block_ids = block_table.get_block_ids()
                if block_ids:
                    self._block_store.restore(block_ids, checkpoint)
                    self.storage.key_cache[:, block_ids, ...] = key_buffer.to(
                        device=self.storage.key_cache.device,
                        dtype=self.storage.key_cache.dtype,
                    )
                    self.storage.value_cache[:, block_ids, ...] = (
                        value_buffer.to(
                            device=self.storage.value_cache.device,
                            dtype=self.storage.value_cache.dtype,
                        )
                    )
                if block_ids and cpu_buffer is not None:
                    assert self._kv_cache is not None
                    self._kv_cache[:, block_ids, ...] = cpu_buffer.to(
                        device=self._kv_cache.device,
                        dtype=self._kv_cache.dtype,
                    )
            self._swapped_out_sequences.discard(seq_id)
            return

        if cpu_buffer is not None:
            if not block_table.has_blocks():
                num_blocks_needed = int(cpu_buffer.shape[1])
                restored_block_ids = self.block_allocator.allocate(
                    num_blocks_needed,
                )
                block_table.restore_blocks(
                    restored_block_ids,
                    num_tokens=saved_num_tokens,
                )
            block_ids = block_table.get_block_ids()
            if block_ids and self._kv_cache is not None:
                self._kv_cache[:, block_ids, ...] = cpu_buffer.to(
                    device=self._kv_cache.device,
                    dtype=self._kv_cache.dtype,
                )
        self._swapped_out_sequences.discard(seq_id)

    def _swap_in_released(self, seq_id: int, block_table: BlockTable) -> None:
        chunk = self._swapped_page_chunks.get(seq_id)
        num_tokens = self._swapped_num_tokens.get(seq_id, 0)
        num_blocks_needed = 0 if chunk is None else int(chunk.payload.shape[1])
        restored_block_ids = self.block_allocator.allocate(num_blocks_needed)
        try:
            block_table.restore_blocks(
                restored_block_ids, num_tokens=num_tokens
            )
            if chunk is not None and restored_block_ids:
                self._store.restore_page_chunk(restored_block_ids, chunk)
        except Exception:
            if restored_block_ids:
                self.block_allocator.free(restored_block_ids)
            block_table.release_blocks_only()
            raise
        self._swapped_page_chunks.pop(seq_id, None)
        self._swapped_num_tokens.pop(seq_id, None)
        self._released_swapped.discard(seq_id)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _blocks_needed_for(self, record: _SequenceKVRecord) -> int:
        if record.num_tokens <= 0:
            return 0
        return math.ceil(record.num_tokens / self.block_size)

    def _within_inflight_cap(self, additional_nbytes: int) -> bool:
        if not self._backend.asynchronous or self.max_inflight_bytes <= 0:
            return True
        current = int(self.get_swap_stats()["inflight_bytes"])
        allowed = current + additional_nbytes <= self.max_inflight_bytes
        if not allowed:
            self._inflight_backpressure_total += 1
        return allowed

    def _record_failure(
        self, direction: str, record: _SequenceKVRecord
    ) -> None:
        self._swap_counters[f"swap_{direction}_failed_total"] += 1
        _logger.warning(
            "KV swap %s failed state=%s generation=%d error=%s",
            direction,
            record.state.value,
            record.key.generation,
            record.last_error,
        )

    def _record_completion(
        self,
        direction: str,
        ticket: CopyTicket | None,
        nbytes: int,
    ) -> None:
        self._swap_counters[f"swap_{direction}_completed_total"] += 1
        copy_direction = "d2h" if direction == "out" else "h2d"
        self._swap_counters[f"{copy_direction}_bytes_total"] += nbytes
        if ticket is not None:
            elapsed_ms = (time.monotonic_ns() - ticket.submitted_ns) / 1_000_000
            self._swap_counters[f"{copy_direction}_duration_ms_sum"] += (
                elapsed_ms
            )

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # Use FlashInfer paged attention if available, else fall back to torch SDPA.
        if self._flashinfer_enabled():
            layer_kv_cache = self._kv_cache[layer_idx]
            query_src = query.to(self.device, dtype=self._kv_cache.dtype)
            query_tokens = int(query_src.shape[-2])
            kv_tokens = int(key.shape[-2])
            num_qo_heads = int(query_src.shape[-3])
            batch_size = int(query_src.shape[0]) if query_src.ndim >= 4 else 1
            block_size = int(self.block_size)

            kv_indptr, kv_indices, kv_last_page_len = (
                self._build_flashinfer_paged_metadata(
                    batch_size=batch_size,
                    kv_tokens=kv_tokens,
                    block_size=block_size,
                )
            )

            if query_tokens > 1:
                if self._fi_prefill is None:
                    raise RuntimeError(
                        "FlashInfer prefill wrapper is unavailable"
                    )
                qo_indptr = self._build_qo_indptr(
                    batch_size=batch_size,
                    query_tokens=query_tokens,
                )
                self._fi_prefill.plan(
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    self.num_heads,
                    self.head_dim,
                    block_size,
                )
                return self._fi_prefill.run(query_src, layer_kv_cache)

            if self._fi_decode is None:
                raise RuntimeError("FlashInfer decode wrapper is unavailable")
            self._fi_decode.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                self.num_heads,
                self.head_dim,
                block_size,
            )
            return self._fi_decode.run(query_src, layer_kv_cache)

        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

    def _flashinfer_enabled(self) -> bool:
        return bool(
            self._use_flashinfer
            and self._fi_prefill is not None
            and self._fi_decode is not None
        )

    def _build_qo_indptr(
        self,
        batch_size: int,
        query_tokens: int,
    ) -> torch.Tensor:
        qo_indptr = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=self.device,
        )
        if batch_size > 0:
            qo_indptr[1:] = torch.arange(
                query_tokens,
                query_tokens * (batch_size + 1),
                query_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        return qo_indptr

    def _build_flashinfer_paged_metadata(
        self,
        batch_size: int,
        kv_tokens: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = max((max(kv_tokens, 1) + block_size - 1) // block_size, 1)
        kv_indptr_vals = [idx * num_pages for idx in range(batch_size + 1)]
        kv_indptr = torch.tensor(
            kv_indptr_vals,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indices = torch.arange(
            num_pages,
            dtype=torch.int32,
            device=self.device,
        ).repeat(batch_size)
        rem = kv_tokens % block_size
        last_page_len = (
            block_size if rem == 0 and kv_tokens > 0 else max(rem, 1)
        )
        kv_last_page_len = torch.full(
            (batch_size,),
            fill_value=int(last_page_len),
            dtype=torch.int32,
            device=self.device,
        )
        return kv_indptr, kv_indices, kv_last_page_len

    def _require_sequence(self, seq_id: int) -> BlockTable:
        block_table = self._sequence_tables.get(seq_id)
        if block_table is None:
            raise KeyError(f"unknown sequence id: {seq_id}")
        return block_table

    @staticmethod
    def _resolve_device(device: torch.device | None) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device


@dataclass
class _RetiringEntry:
    record: _SequenceKVRecord
    block_table: BlockTable | None


__all__ = ["BlockAllocator", "BlockTable", "PagedKVCache"]
