from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from moe_infinity.serving.prefix_contract import (
    PrefixLease,
    PrefixMatch,
)

EntryId = int

_ROOT_ENTRY_ID: EntryId = 0


@dataclass(frozen=True)
class CacheNamespace:
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_config_digest: str
    adapter_id: str | None
    adapter_revision: str | None
    dtype: str
    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    attention_backend: str
    attention_layout: str
    position_config_digest: str
    runtime_epoch: str


def hash_token_block(token_ids: list[int]) -> str:
    token_bytes = bytes(token_id % 256 for token_id in token_ids)
    return hashlib.sha256(token_bytes).hexdigest()[:16]


def _digest_block(
    namespace: CacheNamespace,
    parent_entry_id: EntryId,
    token_block: tuple[int, ...],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(repr(namespace).encode("utf-8"))
    hasher.update(parent_entry_id.to_bytes(8, "little", signed=True))
    for token_id in token_block:
        hasher.update(int(token_id).to_bytes(8, "little", signed=True))
    return hasher.hexdigest()[:16]


@dataclass
class _CacheEntry:
    entry_id: EntryId
    namespace: CacheNamespace
    parent_entry_id: EntryId
    digest: str
    token_block: tuple[int, ...]
    block_id: int
    child_entry_ids: set[EntryId] = field(default_factory=set)


def _noop(ids: list[int]) -> None:
    _ = ids


class PrefixCache:
    def __init__(
        self,
        block_size: int = 16,
        max_entries: int = 1000,
        *,
        on_retain: Callable[[list[int]], None] | None = None,
        on_release: Callable[[list[int]], None] | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")

        self.block_size = block_size
        self.max_entries = max_entries
        self._on_retain = on_retain if on_retain is not None else _noop
        self._on_release = on_release if on_release is not None else _noop

        self._lock = threading.RLock()
        self._entries: dict[EntryId, _CacheEntry] = {}
        self._roots: dict[CacheNamespace, EntryId] = {}
        self._digest_buckets: dict[str, list[EntryId]] = {}
        self._lru: OrderedDict[EntryId, None] = OrderedDict()
        self._next_entry_id: EntryId = 1
        self._leased_entry_counts: dict[EntryId, int] = {}

        self._hits = 0
        self._misses = 0
        self._matched_tokens_total = 0
        self._open_leases = 0

    @property
    def open_leases(self) -> int:
        return self._open_leases

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def hits_total(self) -> int:
        return self._hits

    @property
    def matched_tokens_total(self) -> int:
        return self._matched_tokens_total

    def _root_entry_id(self, namespace: CacheNamespace) -> EntryId:
        root = self._roots.get(namespace)
        if root is None:
            root = _ROOT_ENTRY_ID
            self._roots[namespace] = root
        return root

    def _find_child(
        self,
        namespace: CacheNamespace,
        parent_entry_id: EntryId,
        token_block: tuple[int, ...],
        digest: str,
    ) -> _CacheEntry | None:
        for candidate_id in self._digest_buckets.get(digest, ()):
            candidate = self._entries.get(candidate_id)
            if candidate is None:
                continue
            if (
                candidate.namespace == namespace
                and candidate.parent_entry_id == parent_entry_id
                and candidate.token_block == token_block
            ):
                return candidate
        return None

    def _iter_full_blocks(
        self, token_ids: list[int], max_blocks: int
    ) -> list[tuple[int, ...]]:
        blocks: list[tuple[int, ...]] = []
        limit = min(max_blocks, len(token_ids) // self.block_size)
        for index in range(limit):
            start = index * self.block_size
            blocks.append(tuple(token_ids[start : start + self.block_size]))
        return blocks

    def insert(
        self,
        namespace: CacheNamespace,
        token_ids: list[int],
        block_ids: list[int],
        committed_tokens: int,
    ) -> None:
        if committed_tokens < 0:
            raise ValueError(
                f"committed_tokens must be >= 0, got {committed_tokens}"
            )
        with self._lock:
            num_full_blocks = min(
                committed_tokens // self.block_size, len(block_ids)
            )
            token_blocks = self._iter_full_blocks(token_ids, num_full_blocks)
            parent_entry_id = self._root_entry_id(namespace)
            for index, token_block in enumerate(token_blocks):
                digest = _digest_block(namespace, parent_entry_id, token_block)
                existing = self._find_child(
                    namespace, parent_entry_id, token_block, digest
                )
                if existing is not None:
                    self._lru.move_to_end(existing.entry_id)
                    parent_entry_id = existing.entry_id
                    continue

                entry = _CacheEntry(
                    entry_id=self._next_entry_id,
                    namespace=namespace,
                    parent_entry_id=parent_entry_id,
                    digest=digest,
                    token_block=token_block,
                    block_id=block_ids[index],
                )
                self._next_entry_id += 1
                self._entries[entry.entry_id] = entry
                self._digest_buckets.setdefault(digest, []).append(
                    entry.entry_id
                )
                self._lru[entry.entry_id] = None
                if parent_entry_id in self._entries:
                    self._entries[parent_entry_id].child_entry_ids.add(
                        entry.entry_id
                    )
                self._on_retain([entry.block_id])
                parent_entry_id = entry.entry_id

            self._evict_to_capacity()

    def acquire_prefix_lease(
        self,
        namespace: CacheNamespace,
        token_ids: list[int],
        max_prefix_tokens: int,
    ) -> PrefixLease:
        with self._lock:
            max_blocks = max(max_prefix_tokens, 0) // self.block_size
            token_blocks = self._iter_full_blocks(token_ids, max_blocks)
            matched_block_ids: list[int] = []
            matched_entry_ids: list[EntryId] = []
            parent_entry_id = self._root_entry_id(namespace)
            for token_block in token_blocks:
                digest = _digest_block(namespace, parent_entry_id, token_block)
                entry = self._find_child(
                    namespace, parent_entry_id, token_block, digest
                )
                if entry is None:
                    break
                matched_block_ids.append(entry.block_id)
                matched_entry_ids.append(entry.entry_id)
                self._lru.move_to_end(entry.entry_id)
                parent_entry_id = entry.entry_id

            if not matched_block_ids:
                self._misses += 1
                return PrefixLease.empty()

            self._hits += 1
            self._matched_tokens_total += (
                len(matched_block_ids) * self.block_size
            )
            self._on_retain(list(matched_block_ids))
            self._open_leases += 1
            for entry_id in matched_entry_ids:
                self._leased_entry_counts[entry_id] = (
                    self._leased_entry_counts.get(entry_id, 0) + 1
                )
            match = PrefixMatch(
                num_tokens=len(matched_block_ids) * self.block_size,
                block_ids=tuple(matched_block_ids),
                entry_ids=tuple(matched_entry_ids),
            )
            leased_entry_ids = tuple(matched_entry_ids)
            return PrefixLease(
                match,
                self._lease_release,
                lambda: self._lease_terminal(leased_entry_ids),
            )

    def _lease_release(self, block_ids: list[int]) -> None:
        with self._lock:
            self._on_release(list(block_ids))

    def _lease_terminal(self, entry_ids: tuple[int, ...]) -> None:
        with self._lock:
            self._open_leases -= 1
            for entry_id in entry_ids:
                count = self._leased_entry_counts.get(entry_id, 0)
                if count <= 1:
                    self._leased_entry_counts.pop(entry_id, None)
                else:
                    self._leased_entry_counts[entry_id] = count - 1

    def evict_until(self, predicate: Callable[[], bool]) -> None:
        with self._lock:
            while not predicate():
                victim = self._next_evictable_entry()
                if victim is None:
                    break
                self._remove_subtree(victim)

    def _evict_to_capacity(self) -> None:
        while len(self._entries) > self.max_entries:
            victim = self._next_evictable_entry()
            if victim is None:
                break
            self._remove_subtree(victim)

    def _next_evictable_entry(self) -> EntryId | None:
        for entry_id in self._lru:
            if self._subtree_is_leased(entry_id):
                continue
            return entry_id
        return None

    def _subtree_is_leased(self, entry_id: EntryId) -> bool:
        if self._leased_entry_counts.get(entry_id, 0) > 0:
            return True
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        return any(
            self._subtree_is_leased(child_id)
            for child_id in entry.child_entry_ids
        )

    def _remove_subtree(self, entry_id: EntryId) -> None:
        entry = self._entries.get(entry_id)
        if entry is None:
            self._lru.pop(entry_id, None)
            return
        for child_id in list(entry.child_entry_ids):
            self._remove_subtree(child_id)
        parent = self._entries.get(entry.parent_entry_id)
        if parent is not None:
            parent.child_entry_ids.discard(entry_id)
        bucket = self._digest_buckets.get(entry.digest)
        if bucket is not None:
            if entry_id in bucket:
                bucket.remove(entry_id)
            if not bucket:
                del self._digest_buckets[entry.digest]
        self._lru.pop(entry_id, None)
        del self._entries[entry_id]
        self._on_release([entry.block_id])


__all__ = [
    "CacheNamespace",
    "PrefixCache",
    "hash_token_block",
]
