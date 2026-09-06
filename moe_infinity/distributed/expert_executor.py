# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

from moe_infinity.utils import ArcherConfig

try:
    import nvtx  # pyright: ignore[reportMissingTypeStubs]
except Exception:
    nvtx = None

try:
    from moe_infinity.profiling.io_profiler import (  # pyright: ignore[reportMissingImports]
        IOProfiler,
    )
except Exception:
    IOProfiler = None


def _nvtx_ctx(name: str):
    if nvtx is None:
        return nullcontext()

    annotate = getattr(nvtx, "annotate", None)
    if annotate is None:
        return nullcontext()

    return annotate(name, color="green")


def _profiler_instance():
    if IOProfiler is None:
        return None

    instance = getattr(IOProfiler, "instance", None)
    if instance is None:
        return None

    return instance()


_route_ahead_impl = None


def _load_route_ahead_impl():
    # Lazy import: a top-level import would be circular (spec_decode.__init__
    # -> dflash -> big_modeling -> model_offload -> this module). Both targets
    # are leaf modules, so importing them at first dispatch is safe.
    global _route_ahead_impl
    if _route_ahead_impl is None:
        import moe_infinity.spec_decode._route_ahead_ctx as route_ahead_ctx
        from moe_infinity.spec_decode._prefetch_route import (
            union_experts_from_mask,
        )

        _route_ahead_impl = (route_ahead_ctx, union_experts_from_mask)
    return _route_ahead_impl


def _call_expert_dispatcher(method, *args, **kwargs):
    global _expert_dispatcher
    func = getattr(_expert_dispatcher, method)
    return func(*args, **kwargs)


def _layer_expert_nbytes(prefetcher, layer_id, expert_ids):
    """``{expert_id: stored_bytes}`` for this layer's prefetched set, or None.

    Reads the registration-time ``ExpertPrefetcher.expert_nbytes_map`` -- a real
    ``dict`` only on the offloaded native path. Mocks, resident runs, and any
    engine without the map yield ``None`` so the A5 recorder keeps byte-accurate
    absence instead of a fabricated average expert size, and never calls
    ``int()`` on a mock attribute.
    """
    if not expert_ids:
        return None
    nbytes_map = getattr(prefetcher, "expert_nbytes_map", None)
    if not isinstance(nbytes_map, dict) or not nbytes_map:
        return None
    entry = {}
    for expert_id in expert_ids:
        nbytes = nbytes_map.get((layer_id, expert_id))
        if nbytes is not None:
            entry[expert_id] = int(nbytes)
    return entry or None


def _executor_evidence(**kwargs):
    # Lazy for the same package-cycle reason as ``_load_route_ahead_impl``.
    from moe_infinity.spec_decode.protocols import ExecutorEvidence

    return ExecutorEvidence(**kwargs)


def _prefetcher_hit_rate(prefetcher):
    getter = getattr(prefetcher, "get_hit_rate", None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rate = float(value)
    return rate if 0.0 <= rate <= 1.0 else None


class DistributedExpertExecutor:
    def __init__(self, archer_config: ArcherConfig):
        self.archer_config = archer_config
        self.expert_dispatcher = cast(Any, None)
        self.device_map_manager = cast(Any, None)
        self.prefetcher = None
        self._speculative_prefetch_overlap = bool(
            getattr(archer_config, "speculative_prefetch_overlap", False)
        )
        self._gpu_only_expert_routing = bool(
            getattr(archer_config, "gpu_only_expert_routing", False)
        )
        self._last_dispatch_used_native_routing = False
        self._gpu_route_fallback_count = 0
        self._pending_prefetch = None
        self._pending_prefetch_failure_safe = False
        self.last_executor_evidence = _executor_evidence(
            wiring_reachable=True,
            fallback_reason="context_inactive",
        )

    def set_expert_dispatcher(self, expert_dispatcher):
        global _expert_dispatcher
        _expert_dispatcher = expert_dispatcher
        self.expert_dispatcher = expert_dispatcher

    def set_device_map_manager(self, device_map_manager):
        self.device_map_manager = device_map_manager

    def set_prefetcher(self, prefetcher):
        self.prefetcher = prefetcher

    def trigger_speculative_prefetch(self, layer_id, router_logits):
        if self.prefetcher is not None:
            self.prefetcher.speculative_prefetch(layer_id, router_logits)

    def _maybe_route_ahead_prefetch(
        self, layer_id, router_mask, num_expert, prefetcher=None
    ) -> bool:
        """Track A3 route-ahead seam; True iff the union prefetch fired.

        Active only inside a DFlash verify forward (``_route_ahead_ctx``).
        Pins the ACTUAL routed union of this layer via
        ``fetch_experts_lock_cache`` and enqueues it through the A2
        explicit-set ``speculative_prefetch`` -- cache warming for the reads
        this dispatch is about to issue, never a routing change. Inactive
        context, no prefetcher (resident mode / ``speculative_prefetch``
        config off), or an empty union returns False so the caller falls
        through to the legacy mean/topk path byte-identically.

        A4: when the context carries a ``RouteAheadStats`` handle, the
        predicted set and this layer's mask are reported to it (read-only
        observation; ``None`` handle = zero overhead). Offloaded
        executor-backed models (DeepSeek/Qwen/Mixtral) reach this seam with
        no resident-only gate; gpt-oss never reaches it at all
        (``model_offload.py`` wires no ``expert_executor`` into
        ``SyncGptOssMLP``).
        """
        ctx, union_experts_from_mask = _load_route_ahead_impl()
        if not ctx.is_active():
            available_prefetcher = prefetcher or self.prefetcher
            self.last_executor_evidence = _executor_evidence(
                wiring_reachable=True,
                prefetcher_present=available_prefetcher is not None,
                fallback_reason="context_inactive",
            )
            return False
        stats = ctx.current_stats()
        route_prefetcher = prefetcher
        if route_prefetcher is None:
            route_prefetcher = ctx.current_prefetcher()
        if route_prefetcher is None:
            route_prefetcher = self.prefetcher
        mask_2d = router_mask.reshape(-1, num_expert)
        union_expert_ids = union_experts_from_mask(mask_2d)
        row_union: set[tuple[int, int, int]] = set()
        row_offsets = ctx.current_row_offsets()
        if row_offsets and row_offsets[-1] == int(mask_2d.shape[0]):
            for row in range(len(row_offsets) - 1):
                row_ids = union_experts_from_mask(
                    mask_2d[row_offsets[row] : row_offsets[row + 1]]
                )
                row_union.update(
                    (row, int(layer_id), int(expert_id))
                    for expert_id in row_ids
                )
        fired = False
        fallback_reason = None
        expert_nbytes = _layer_expert_nbytes(
            route_prefetcher, layer_id, union_expert_ids
        )
        if route_prefetcher is not None and union_expert_ids:
            # A0 section 2/5 (A4 guard): pin exactly ONE layer's union per
            # dispatch -- ``ReplaceCacheCandidates`` is global and clears the
            # background queues (task_scheduler.h), so folding several layers
            # into one pin would evict candidates the next layer's dispatch
            # still needs. Never batch pins across layers; never pin the
            # empty set (short-circuited above).
            try:
                route_prefetcher.fetch_experts_lock_cache(
                    layer_id, union_expert_ids
                )
                route_prefetcher.speculative_prefetch(
                    layer_id,
                    expert_ids=union_expert_ids,
                    prefetch_layer_id=layer_id,
                )
                fired = True
            except Exception as exc:
                # Route-ahead is cache-warming observation only. A prefetch
                # failure must preserve the legacy expert dispatch/output path.
                fallback_reason = f"prefetch_exception:{type(exc).__name__}"
        elif not union_expert_ids:
            fallback_reason = "empty_actual_union"
        else:
            fallback_reason = "prefetcher_absent"

        prefetched_bytes = (
            sum(expert_nbytes.values())
            if fired and expert_nbytes is not None
            else 0
        )
        cache_hit_rate = _prefetcher_hit_rate(route_prefetcher)
        self.last_executor_evidence = _executor_evidence(
            wiring_reachable=True,
            prefetcher_present=route_prefetcher is not None,
            attempted_layers=(int(layer_id),),
            fired_layers=((int(layer_id),) if fired else ()),
            actual_expert_union=frozenset(
                (int(layer_id), int(expert_id))
                for expert_id in union_expert_ids
            ),
            actual_expert_union_by_row=frozenset(row_union),
            prefetched_bytes=prefetched_bytes,
            coverage=(1.0 if fired or not union_expert_ids else 0.0),
            cache_hit_rate=cache_hit_rate,
            fallback_reason=fallback_reason,
        )
        if stats is not None:
            # A5 read-only observation: predicted == the pinned union when
            # the prefetch fired, else [] (coverage 0 for this layer).
            predicted_ids = union_expert_ids if fired else []
            observe_attempt = getattr(stats, "observe_executor_attempt", None)
            if callable(observe_attempt):
                try:
                    observe_attempt(
                        layer_id,
                        union_expert_ids,
                        actual_ids_by_row=row_union,
                        prefetcher_present=route_prefetcher is not None,
                        fired=fired,
                        fallback_reason=fallback_reason,
                        prefetched_bytes=prefetched_bytes,
                        cache_hit_rate=cache_hit_rate,
                    )
                except Exception:
                    # Observer failures are isolated from expert dispatch.
                    pass
            try:
                stats.observe_layer(
                    layer_id,
                    predicted_ids,
                    mask_2d,
                    expert_nbytes=(expert_nbytes if fired else None),
                )
            except Exception:
                # Observer failures are isolated from expert dispatch.
                pass
        return fired

    def _can_use_gpu_only_routing(self, router_mask) -> bool:
        if not self._gpu_only_expert_routing:
            return False
        if not torch.is_tensor(router_mask) or not router_mask.is_cuda:
            return False
        if not hasattr(self.expert_dispatcher, "dispatch_experts"):
            return False
        if not hasattr(self.expert_dispatcher, "take_last_active_experts"):
            return False
        route_ahead_ctx, _ = _load_route_ahead_impl()
        if route_ahead_ctx.is_active():
            return False
        return True

    def _dispatch_eager_local(self, layer_id, router_mask, num_expert):
        expert_count = (
            torch.sum(router_mask.view((-1, num_expert)), dim=0)
            .cpu()
            .numpy()
            .flatten()
        )
        expert_list = (
            np.arange(num_expert).astype(int)[expert_count > 0].tolist()
        )
        self.expert_dispatcher.set_expected_queue(len(expert_list))
        total_gpus = torch.cuda.device_count()
        for expert_id in expert_list:
            self.expert_dispatcher.enqueue_expert(
                layer_id, expert_id, expert_id % total_gpus, False
            )
        self.expert_dispatcher.notify_fetch_start()
        return expert_list

    def get_gpu_routing_stats(self):
        stats = {
            "route_batches": 0,
            "route_failures": 0,
            "last_active_experts": 0,
            "last_route_handoff_us": 0,
            "completion_events_retired": 0,
        }
        getter = getattr(self.expert_dispatcher, "get_routing_stats", None)
        if getter is not None:
            stats.update({key: int(value) for key, value in getter().items()})
        stats["fallback_count"] = int(self._gpu_route_fallback_count)
        return stats

    def dispatch_local(
        self,
        layer_id,
        hidden_states,
        router_mask,
        router_weights,
        router_logits=None,
        prefetcher=None,
    ):
        profiler = _profiler_instance()
        routing_nvtx_ctx = _nvtx_ctx("moe_routing")
        routing_profiler_ctx = (
            profiler.time("routing", layer=layer_id, expert=-1)
            if profiler is not None
            else nullcontext()
        )
        with routing_nvtx_ctx:
            with routing_profiler_ctx:
                num_expert = router_mask.shape[-1]
                native_requested = bool(
                    self._gpu_only_expert_routing
                    and torch.is_tensor(router_mask)
                    and router_mask.is_cuda
                )
                use_native_routing = self._can_use_gpu_only_routing(router_mask)
                expert_list = None

        self.expert_dispatcher.set_inputs(
            hidden_states, router_mask.bool(), router_weights
        )

        # Route-ahead pin + enqueue must precede every enqueue_expert below
        # (A0 section 2). Inactive context: no-op, legacy flow unchanged.
        route_ahead_handled = self._maybe_route_ahead_prefetch(
            layer_id, router_mask, num_expert, prefetcher
        )
        route_ahead_attempted = bool(
            self.last_executor_evidence.attempted_layers
        )

        dispatch_nvtx_ctx = _nvtx_ctx("expert_dispatch")
        dispatch_profiler_ctx = (
            profiler.time("expert_dispatch", layer=layer_id, expert=-1)
            if profiler is not None
            else nullcontext()
        )
        with dispatch_nvtx_ctx:
            with dispatch_profiler_ctx:
                if use_native_routing:
                    with _nvtx_ctx("gpu_route_submit"):
                        with (
                            profiler.time(
                                "gpu_route_submit", layer=layer_id, expert=-1
                            )
                            if profiler is not None
                            else nullcontext()
                        ):
                            self.expert_dispatcher.dispatch_experts(layer_id)
                else:
                    if native_requested:
                        self._gpu_route_fallback_count += 1
                    with _nvtx_ctx("gpu_route_fallback"):
                        with (
                            profiler.time(
                                "gpu_route_fallback",
                                layer=layer_id,
                                expert=-1,
                            )
                            if profiler is not None
                            else nullcontext()
                        ):
                            expert_list = self._dispatch_eager_local(
                                layer_id, router_mask, num_expert
                            )

        self._last_dispatch_used_native_routing = use_native_routing

        if prefetcher is None:
            prefetcher = self.prefetcher

        if route_ahead_handled:
            # A0 section 3: the exact-union prefetch REPLACES the legacy
            # mean(0)/topk prediction for this dispatch, so neither the
            # overlap-triggered nor the deferred pooled call may fire.
            pending_router_logits = None
        elif (
            self._speculative_prefetch_overlap
            and prefetcher is not None
            and router_logits is not None
        ):
            if route_ahead_attempted:
                try:
                    self.trigger_speculative_prefetch(layer_id, router_logits)
                except Exception:
                    pass
            else:
                self.trigger_speculative_prefetch(layer_id, router_logits)
            pending_router_logits = None
        else:
            pending_router_logits = router_logits

        self._pending_prefetch = (
            prefetcher,
            layer_id,
            expert_list,
            pending_router_logits,
        )
        self._pending_prefetch_failure_safe = route_ahead_attempted

    def wait_dispatch_local(self):
        profiler = _profiler_instance()
        wait_nvtx_ctx = _nvtx_ctx("expert_wait_barrier")
        wait_profiler_ctx = (
            profiler.time("sync_wait", expert=-1)
            if profiler is not None
            else nullcontext()
        )
        with wait_nvtx_ctx:
            with wait_profiler_ctx:
                completion_profiler_ctx = (
                    profiler.time("expert_completion_handoff", expert=-1)
                    if profiler is not None
                    else nullcontext()
                )
                with completion_profiler_ctx:
                    result = self.expert_dispatcher.wait_expert()

        pending = getattr(self, "_pending_prefetch", None)
        if pending is not None:
            prefetcher, layer_id, expert_list, router_logits = pending
            self._pending_prefetch = None
            if expert_list is None and self._last_dispatch_used_native_routing:
                expert_list = list(
                    self.expert_dispatcher.take_last_active_experts()
                )
            failure_safe = self._pending_prefetch_failure_safe
            self._pending_prefetch_failure_safe = False
            if prefetcher is not None:
                if failure_safe:
                    try:
                        prefetcher.correct_prefetch(layer_id + 1, expert_list)
                    except Exception:
                        pass
                else:
                    prefetcher.correct_prefetch(layer_id + 1, expert_list)
            if router_logits is not None:
                if failure_safe:
                    try:
                        self.trigger_speculative_prefetch(
                            layer_id, router_logits
                        )
                    except Exception:
                        pass
                else:
                    self.trigger_speculative_prefetch(layer_id, router_logits)

        return result

    def dispatch(self, hidden_states, router_mask, layer_id):
        num_expert = router_mask.shape[-1]
        expert_count = (
            torch.sum(router_mask.view((-1, num_expert)), dim=0)
            .cpu()
            .numpy()
            .flatten()
        )

        expert_list = (
            np.arange(num_expert).astype(int)[expert_count > 0].tolist()
        )

        device_list = self.device_map_manager.get_target_device(expert_list)
        visited_ranks = set()
        rank_wait_cnt = {r: 0 for r in range(dist.get_world_size())}
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            visited_ranks.add(rank)
            rank_wait_cnt[rank] += 1

        futures = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("set_inputs", hidden_states.cpu(), router_mask.cpu()),
                )
                futures.append(future)
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("set_expected_queue", rank_wait_cnt[rank]),
                )
                futures.append(future)
            else:
                self.expert_dispatcher.set_inputs(hidden_states, router_mask)
                self.expert_dispatcher.set_expected_queue(rank_wait_cnt[rank])

        # wait for all futures
        for future in futures:
            future.wait()

        futures = []
        for k, device_meta in enumerate(device_list):
            rank, gpu_id, expert_id = device_meta
            if rank == dist.get_rank():
                self.expert_dispatcher.enqueue_expert(
                    layer_id, expert_id, gpu_id, False
                )
            else:
                future = rpc.rpc_async(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("enqueue_expert", layer_id, expert_id, gpu_id, True),
                )
                futures.append(future)

        # wait for all futures
        for future in futures:
            future.wait()

        result_list = []
        for rank in visited_ranks:
            if rank != dist.get_rank():
                result = rpc.rpc_sync(
                    f"worker_{rank}",
                    _call_expert_dispatcher,
                    args=("wait_expert",),
                )
                result_list += result
            else:
                result = self.expert_dispatcher.wait_expert()
                result_list += result

        return result_list


# Alias for backward compatibility
ExpertExecutor = DistributedExpertExecutor
