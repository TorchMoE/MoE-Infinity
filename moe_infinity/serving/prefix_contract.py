from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from moe_infinity.serving.prefix_cache import CacheNamespace

OwnerToken = object


@dataclass(frozen=True)
class PrefixMatch:
    num_tokens: int
    block_ids: tuple[int, ...]
    entry_ids: tuple[int, ...]


@dataclass
class PrefixLease:
    match: PrefixMatch
    _release: Callable[[list[int]], None]
    _terminal: Callable[[], None]
    _state: str = "open"
    _prepared_owner: OwnerToken | None = None

    @classmethod
    def empty(cls) -> "PrefixLease":
        return cls(PrefixMatch(0, (), ()), lambda ids: None, lambda: None)

    @property
    def state(self) -> str:
        return self._state

    def is_prepared_for(self, owner: OwnerToken) -> bool:
        return self._state == "prepared" and self._prepared_owner is owner

    def prepare_adoption(self, owner: OwnerToken) -> PrefixMatch:
        if self._state != "open":
            raise RuntimeError(f"lease is already {self._state}")
        if owner is None:
            raise ValueError("lease adoption owner must not be None")
        self._prepared_owner = owner
        self._state = "prepared"
        return self.match

    def commit_adoption(self, owner: OwnerToken) -> PrefixMatch:
        if self._state != "prepared" or self._prepared_owner is not owner:
            raise RuntimeError("lease adoption owner/state mismatch")
        self._state = "committed"
        self._prepared_owner = None
        self._terminal()
        return self.match

    def abort(self, owner: OwnerToken | None = None) -> None:
        if self._state not in {"open", "prepared"}:
            raise RuntimeError(f"lease is already {self._state}")
        if self._state == "prepared" and self._prepared_owner is not owner:
            raise RuntimeError("lease adoption owner mismatch")
        self._release(list(self.match.block_ids))
        self._state = "aborted"
        self._prepared_owner = None
        self._terminal()


class PrefixLeaseProvider(Protocol):
    def acquire_prefix_lease(
        self,
        namespace: "CacheNamespace",
        token_ids: list[int],
        max_prefix_tokens: int,
    ) -> PrefixLease: ...


__all__ = ["OwnerToken", "PrefixLease", "PrefixLeaseProvider", "PrefixMatch"]
