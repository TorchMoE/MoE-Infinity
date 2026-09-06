"""DFlash route-ahead prefetch context (Track A3).

``contextvars.ContextVar``-backed, async/thread-safe marker that the current
forward is a DFlash VERIFY forward, plus an optional prefetcher handle bound
at activation. ``DFlashSpeculator.generate`` activates it around the verify
call; ``DistributedExpertExecutor.dispatch_local`` consumes it to pin +
enqueue the ACTUAL routed expert union for the layer being dispatched, before
any expert read (``.sisyphus/plans/dflash-deferred-tracks-plan.md``, Track A
A0 section 2). Default INACTIVE: non-spec decode and spec-off paths never set
the flag and observe zero behavior change; the context only triggers cache
warming, never routing or output changes.

Intentionally a LEAF module (no ``moe_infinity`` imports) so
``distributed/expert_executor.py`` can lazy-import it without the
``spec_decode.__init__`` -> ``dflash`` -> ``big_modeling`` ->
``model_offload`` -> ``expert_executor`` cycle. A4/A5 reuse this seam.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator, Optional

route_ahead_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "dflash_route_ahead_active", default=False
)
route_ahead_prefetcher: contextvars.ContextVar[Optional[Any]] = (
    contextvars.ContextVar("dflash_route_ahead_prefetcher", default=None)
)
# A5 metrics handle: a ``RouteAheadStats`` (``_route_ahead_stats``) when the
# speculator opted in, else None. Typed ``Any`` to keep this module a leaf.
route_ahead_stats: contextvars.ContextVar[Optional[Any]] = (
    contextvars.ContextVar("dflash_route_ahead_stats", default=None)
)
route_ahead_row_offsets: contextvars.ContextVar[tuple[int, ...]] = (
    contextvars.ContextVar("dflash_route_ahead_row_offsets", default=())
)


def is_active() -> bool:
    return route_ahead_active.get()


def current_prefetcher() -> Optional[Any]:
    return route_ahead_prefetcher.get()


def current_stats() -> Optional[Any]:
    return route_ahead_stats.get()


def current_row_offsets() -> tuple[int, ...]:
    return route_ahead_row_offsets.get()


@contextmanager
def route_ahead_context(
    prefetcher: Optional[Any] = None,
    stats: Optional[Any] = None,
    row_offsets: tuple[int, ...] = (),
) -> Iterator[None]:
    """Activate the route-ahead context; token-reset in ``finally``.

    A raise inside the wrapped forward can never leak the active state into
    later non-spec decode; nested scopes restore the outer prefetcher handle.
    ``stats`` (A5) is an optional read-only metrics recorder; ``None``
    (default) keeps the seam zero-overhead.
    """
    active_token = route_ahead_active.set(True)
    prefetcher_token = route_ahead_prefetcher.set(prefetcher)
    stats_token = route_ahead_stats.set(stats)
    offsets_token = route_ahead_row_offsets.set(tuple(row_offsets))
    try:
        yield
    finally:
        route_ahead_row_offsets.reset(offsets_token)
        route_ahead_stats.reset(stats_token)
        route_ahead_prefetcher.reset(prefetcher_token)
        route_ahead_active.reset(active_token)


__all__ = [
    "route_ahead_active",
    "route_ahead_prefetcher",
    "route_ahead_stats",
    "is_active",
    "current_prefetcher",
    "current_stats",
    "current_row_offsets",
    "route_ahead_context",
]
