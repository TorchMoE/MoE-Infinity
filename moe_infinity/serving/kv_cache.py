from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

import torch

from moe_infinity.runtime import flashinfer_utils
from moe_infinity.serving.prefix_contract import PrefixLease

if TYPE_CHECKING:
    from moe_infinity.runtime.attention_backend import (
        LayeredPagedKVCheckpoint,
        LayeredPagedKVStore,
        PagedAttentionBackend,
    )

if TYPE_CHECKING:
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
        self._ref_counts = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_heap)

    def ref_count(self, block_id: int) -> int:
        return self._ref_counts.get(block_id, 0)

    def allocate(self, num_blocks: int) -> list[int]:
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

    def free(self, block_ids: list[int]) -> None:
        self.release(block_ids)


@dataclass
class BlockTable:
    block_allocator: BlockAllocator
    _block_ids: list[int] = field(default_factory=list, init=False, repr=False)
    _num_tokens: int = field(default=0, init=False, repr=False)

    @property
    def block_size(self) -> int:
        return self.block_allocator.block_size

    def append_token(self) -> None:
        if self._num_tokens % self.block_size == 0:
            new_block_ids = self.block_allocator.allocate(1)
            self._block_ids.append(new_block_ids[0])
        self._num_tokens += 1

    def get_block_ids(self) -> list[int]:
        return list(self._block_ids)

    def num_computed_tokens(self) -> int:
        return self._num_tokens

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
class PagedKVCache:
    num_blocks: int
    block_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device | None = None
    storage: "PagedKVStorage | None" = None
    block_allocator: BlockAllocator = field(init=False)
    _sequence_tables: dict[int, BlockTable] = field(
        init=False, default_factory=dict
    )
    _swapped_cpu_buffers: dict[int, torch.Tensor] = field(
        init=False, default_factory=dict
    )
    _swapped_storage_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = (
        field(init=False, default_factory=dict)
    )
    _swapped_num_tokens: dict[int, int] = field(
        init=False, default_factory=dict
    )
    _swapped_out_sequences: set[int] = field(init=False, default_factory=set)
    _kv_cache: torch.Tensor | None = field(init=False)
    _use_flashinfer: bool = field(init=False, default=False)
    _fi_workspace: torch.Tensor | None = field(init=False, default=None)
    _fi_prefill: _FlashinferPrefillWrapperLike | None = field(
        init=False, default=None
    )
    _fi_decode: _FlashinferDecodeWrapperLike | None = field(
        init=False, default=None
    )
    _cp_kv_manager: CPAwareKVManager | None = field(init=False, default=None)
    _block_store: "LayeredPagedKVStore | None" = field(init=False, default=None)
    _block_store_owner: "PagedAttentionBackend | None" = field(
        init=False, default=None
    )
    _swapped_checkpoints: dict[int, "LayeredPagedKVCheckpoint"] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {self.head_dim}")

        if self.storage is not None:
            self._bind_storage(self.storage)
            return

        self.device = self._resolve_device(self.device)
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

        block_table = BlockTable(block_allocator=self.block_allocator)
        for _ in range(num_tokens):
            block_table.append_token()
        self._sequence_tables[seq_id] = block_table

        if self._cp_kv_manager is not None:
            try:
                block_hashes = block_table.get_block_ids()
                self._cp_kv_manager.notify_blocks_allocated(
                    seq_id,
                    block_hashes,
                )
            except Exception:
                pass

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

    def truncate_tokens(self, seq_id: int, new_len: int) -> None:
        """Roll a sequence back to ``new_len`` tokens, freeing tail blocks.

        Never grows (raises ``ValueError`` if ``new_len`` exceeds the current
        length); no-op when unchanged. Rollback primitive for serving-path DFlash.
        """
        if new_len < 0:
            raise ValueError(f"new_len must be >= 0, got {new_len}")
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

    def free_sequence(self, seq_id: int) -> None:
        block_table = self._sequence_tables.pop(seq_id, None)
        if block_table is None:
            return

        if self._cp_kv_manager is not None:
            try:
                block_hashes = block_table.get_block_ids()
                self._cp_kv_manager.notify_blocks_freed(seq_id, block_hashes)
            except Exception:
                pass

        block_table.release()
        _ = self._swapped_cpu_buffers.pop(seq_id, None)
        _ = self._swapped_storage_buffers.pop(seq_id, None)
        _ = self._swapped_num_tokens.pop(seq_id, None)
        self._swapped_out_sequences.discard(seq_id)

    def set_cp_kv_manager(self, manager: CPAwareKVManager) -> None:
        self._cp_kv_manager = manager

    def free_gpu_blocks(self, seq_id: int) -> None:
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

    def swap_out(self, seq_id: int) -> None:
        block_table = self._require_sequence(seq_id)
        if seq_id in self._swapped_out_sequences:
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
            assert self._kv_cache is not None
            self._swapped_cpu_buffers[seq_id] = (
                self._kv_cache[:, block_ids, ...].detach().to("cpu").clone()
            )
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
                self._swapped_cpu_buffers[seq_id] = (
                    self._kv_cache[:, block_ids, ...].detach().to("cpu").clone()
                )
        self._swapped_out_sequences.add(seq_id)

    def swap_in(self, seq_id: int) -> None:
        block_table = self._require_sequence(seq_id)
        if seq_id not in self._swapped_out_sequences:
            return

        if self._block_store is not None:
            checkpoint = self._swapped_checkpoints.pop(seq_id, None)
            saved_num_tokens = self._swapped_num_tokens.pop(seq_id, 0)
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
                    self._block_store.restore(list(block_ids), checkpoint)
            self._swapped_out_sequences.discard(seq_id)
            return

        cpu_buffer = self._swapped_cpu_buffers.pop(seq_id, None)
        saved_num_tokens = self._swapped_num_tokens.pop(seq_id, 0)
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
            if block_ids:
                assert self._kv_cache is not None
                self._kv_cache[:, block_ids, ...] = cpu_buffer.to(
                    device=self._kv_cache.device,
                    dtype=self._kv_cache.dtype,
                )

        self._swapped_out_sequences.discard(seq_id)

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


__all__ = ["BlockAllocator", "BlockTable", "PagedKVCache"]
