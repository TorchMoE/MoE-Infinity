from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def hash_block_tokens(parent_hash: int, token_ids: tuple[int, ...]) -> int:
    return hash((parent_hash, token_ids))


@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: Optional[int] = None
    prev: Optional["KVCacheBlock"] = field(
        default=None, repr=False, compare=False
    )
    next: Optional["KVCacheBlock"] = field(
        default=None, repr=False, compare=False
    )


class FreeBlockQueue:
    _head: Optional[KVCacheBlock]
    _tail: Optional[KVCacheBlock]
    _size: int

    def __init__(self) -> None:
        self._head = None
        self._tail = None
        self._size = 0

    def append(self, block: KVCacheBlock) -> None:
        block.prev = self._tail
        block.next = None
        if self._tail is not None:
            self._tail.next = block
        else:
            self._head = block
        self._tail = block
        self._size += 1

    def popleft(self) -> Optional[KVCacheBlock]:
        if self._head is None:
            return None

        block = self._head
        self._head = block.next
        if self._head is not None:
            self._head.prev = None
        else:
            self._tail = None

        block.prev = None
        block.next = None
        self._size -= 1
        return block

    def remove(self, block: KVCacheBlock) -> None:
        if block.prev is not None:
            block.prev.next = block.next
        else:
            self._head = block.next

        if block.next is not None:
            block.next.prev = block.prev
        else:
            self._tail = block.prev

        block.prev = None
        block.next = None
        self._size -= 1

    def __len__(self) -> int:
        return self._size


class BlockPool:
    _blocks: list[KVCacheBlock]
    _free_queue: FreeBlockQueue
    _hash_map: dict[int, KVCacheBlock]

    def __init__(self, num_blocks: int):
        self._blocks = [KVCacheBlock(block_id=i) for i in range(num_blocks)]
        self._free_queue = FreeBlockQueue()
        self._hash_map = {}

        for block in self._blocks:
            self._free_queue.append(block)

    def allocate_block(self) -> Optional[KVCacheBlock]:
        block = self._free_queue.popleft()
        if block is None:
            return None

        if block.block_hash is not None:
            cached = self._hash_map.get(block.block_hash)
            if cached is block:
                del self._hash_map[block.block_hash]
            block.block_hash = None

        block.ref_cnt = 1
        return block

    def free_block(self, block: KVCacheBlock) -> None:
        if block.ref_cnt <= 0:
            raise ValueError(
                f"block {block.block_id} has non-positive ref_cnt={block.ref_cnt}"
            )

        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self._free_queue.append(block)

    def get_cached_block(self, block_hash: int) -> Optional[KVCacheBlock]:
        block = self._hash_map.get(block_hash)
        if block is None:
            return None

        if block.ref_cnt == 0:
            self._free_queue.remove(block)

        block.ref_cnt += 1
        return block

    def cache_full_block(self, block: KVCacheBlock, block_hash: int) -> None:
        if block.block_hash is not None and block.block_hash != block_hash:
            existing_mapping = self._hash_map.get(block.block_hash)
            if existing_mapping is block:
                del self._hash_map[block.block_hash]

        colliding = self._hash_map.get(block_hash)
        if colliding is not None and colliding is not block:
            colliding.block_hash = None

        block.block_hash = block_hash
        self._hash_map[block_hash] = block

    def num_free_blocks(self) -> int:
        return len(self._free_queue)

    def num_cached_blocks(self) -> int:
        return len(self._hash_map)

    def total_blocks(self) -> int:
        return len(self._blocks)

    def referenced_block_ids(self) -> list[int]:
        return [block.block_id for block in self._blocks if block.ref_cnt > 0]

    def removable_tail_ids(self, target_blocks: int) -> list[int]:
        if target_blocks < 0 or target_blocks > len(self._blocks):
            raise ValueError("target_blocks must be within the pool")
        tail = self._blocks[target_blocks:]
        if self.referenced_block_ids():
            return []
        return [block.block_id for block in tail]
