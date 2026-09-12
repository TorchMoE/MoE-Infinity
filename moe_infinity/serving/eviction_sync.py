from __future__ import annotations

import threading
from collections import OrderedDict
from enum import Enum
from typing import Optional, Protocol

DEFAULT_MAX_TRACKED_REQUEST_IDS = 100_000


class _CPMiddlewareLike(Protocol):
    def on_request_complete(self, request_id: str) -> None: ...


class EvictionEvent(Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"
    FREED = "freed"
    SWAPPED = "swapped"


class EvictionSyncAdapter:
    def __init__(
        self,
        cp_middleware: Optional[_CPMiddlewareLike] = None,
        *,
        max_tracked_request_ids: int = DEFAULT_MAX_TRACKED_REQUEST_IDS,
    ):
        """Takes optional CP middleware instance for remove_requests calls."""
        if max_tracked_request_ids <= 0:
            raise ValueError("max_tracked_request_ids must be > 0")
        self._cp_middleware: Optional[_CPMiddlewareLike] = cp_middleware
        self._lock: threading.RLock = threading.RLock()
        self._max_tracked_request_ids: int = int(max_tracked_request_ids)
        self._evicted_request_ids: OrderedDict[str, None] = OrderedDict()
        self._evict_incoming: int = 0
        self._evict_removed: int = 0
        self._evict_not_found: int = 0
        self._event_counters: dict[EvictionEvent, int] = {
            event: 0 for event in EvictionEvent
        }

    def on_request_finished(self, request_id: str) -> None:
        """Terminal completion — CP index should be evicted."""
        self._handle_event(request_id, EvictionEvent.COMPLETED)

    def on_request_aborted(self, request_id: str) -> None:
        """Terminal abort — CP index should be evicted."""
        self._handle_event(request_id, EvictionEvent.ABORTED)

    def on_kv_blocks_freed(self, request_id: str) -> None:
        """True KV deallocation — CP index should be evicted."""
        self._handle_event(request_id, EvictionEvent.FREED)

    def on_kv_blocks_swapped(self, request_id: str) -> None:
        """Swap-out only — NO CP action (blocks recoverable)."""
        self._handle_event(request_id, EvictionEvent.SWAPPED)

    def get_counters(self) -> dict[str, int]:
        """Returns: evict_incoming, evict_removed, evict_not_found."""
        with self._lock:
            return {
                "evict_incoming": self._evict_incoming,
                "evict_removed": self._evict_removed,
                "evict_not_found": self._evict_not_found,
            }

    def get_event_counters(self) -> dict[str, int]:
        with self._lock:
            return {
                event.value: self._event_counters[event]
                for event in EvictionEvent
            }

    def _handle_event(
        self,
        request_id: str,
        event: EvictionEvent,
    ) -> None:
        should_evict = False
        with self._lock:
            self._event_counters[event] += 1
            should_evict = event is not EvictionEvent.SWAPPED

        if should_evict:
            self._evict_request(request_id)

    def _evict_request(self, request_id: str) -> None:
        middleware_complete = (
            self._cp_middleware.on_request_complete
            if self._cp_middleware is not None
            else None
        )
        should_notify = False

        with self._lock:
            self._evict_incoming += 1

            if request_id in self._evicted_request_ids:
                self._evict_not_found += 1
                return

            self._evicted_request_ids[request_id] = None
            while (
                len(self._evicted_request_ids) > self._max_tracked_request_ids
            ):
                self._evicted_request_ids.popitem(last=False)

            if callable(middleware_complete):
                self._evict_removed += 1
                should_notify = True

        if should_notify and middleware_complete is not None:
            middleware_complete(request_id)
