# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team


from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from transformers import PretrainedConfig

from moe_infinity.memory.overlap_budget import (
    Candidate,
    OverlapBudgetController,
)
from moe_infinity.utils import parse_moe_param

_SCHEDULER_CAPABILITY_METHODS = (
    "schedule_prefetch_tensors",
    "cancel_prefetch_generation",
    "drain_prefetch_samples",
    "get_inflight_prefetch_bytes",
)
_DISPATCHER_TIMING_METHODS = (
    "set_inputs_with_invocation",
    "set_overlap_compute_timing_enabled",
    "drain_compute_samples",
)


class NativeOverlapCapabilities:
    def __init__(self, engine: Any, dispatcher: Any):
        self.scheduler_ready = engine is not None and all(
            callable(getattr(engine, name, None))
            for name in _SCHEDULER_CAPABILITY_METHODS
        )
        self.dispatcher_timing_ready = dispatcher is not None and all(
            callable(getattr(dispatcher, name, None))
            for name in _DISPATCHER_TIMING_METHODS
        )

    @property
    def enforce_ready(self) -> bool:
        return self.scheduler_ready and self.dispatcher_timing_ready


# Native prefetch priority bands; must mirror core/prefetch/task_scheduler.h.
ON_DEMAND_PRIORITY = 0
ROUTE_AHEAD_PRIORITY = 1
BACKGROUND_PREFETCH_PRIORITY = 2

try:
    import nvtx  # type: ignore[reportMissingTypeStubs]
except ImportError:
    nvtx = None

HAS_NVTX = nvtx is not None

try:
    from moe_infinity.profiling.io_profiler import (  # pyright: ignore[reportMissingImports]
        IOProfiler,
    )
except Exception:
    IOProfiler = None


def _hit_rate_from_visit_counts(counts: Any) -> Optional[float]:
    if counts is None or not hasattr(counts, "numel"):
        return None
    try:
        if counts.numel() == 0 or counts.dim() != 2 or counts.shape[1] < 4:
            return None
        visit = float(counts[:, 0].sum().item())
        hit = float(counts[:, 3].sum().item())
    except Exception:
        return None
    if visit <= 0.0:
        return None
    return hit / visit


class ExpertPrefetcher(object):
    cache_file_rd: Optional[Any] = None
    first_k_dense_replace: int = 0
    route_ahead_priority: int = ROUTE_AHEAD_PRIORITY
    archer_engine: Any
    expert_dispatcher: Optional[Any] = None
    expert_tensor_map: dict[tuple[int, int], int]
    expert_nbytes_map: dict[tuple[int, int], int]

    def __init__(self, config: PretrainedConfig):
        print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        self.archer_engine: Optional[Any] = None
        self.expert_dispatcher: Optional[Any] = None
        self.expert_tensor_map: Dict[Tuple[int, int], int] = {}
        self.expert_nbytes_map: Dict[Tuple[int, int], int] = {}
        self._last_speculative_prediction: Set[int] = set()
        self.overlap_controller: Optional[OverlapBudgetController] = None
        self._overlap_policy: str = "off"
        self._overlap_caps: Optional[NativeOverlapCapabilities] = None
        self._overlap_generation: int = 0
        self._overlap_stats: Dict[str, int] = {}

    def set_archer_engine(self, archer_engine: Any):
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    @property
    def num_offloaded_experts(self) -> int:
        engine = self.archer_engine
        checker = (
            getattr(engine, "is_tensor_offloaded", None)
            if engine is not None
            else None
        )
        if not callable(checker):
            return len(self.expert_tensor_map)
        count = 0
        for tensor_id in self.expert_tensor_map.values():
            try:
                if checker(int(tensor_id)):
                    count += 1
            except Exception:
                continue
        return count

    def get_hit_rate(self) -> float:
        dispatcher = self.expert_dispatcher
        if dispatcher is not None:
            getter = getattr(dispatcher, "get_cache_hit_rate", None)
            if callable(getter):
                try:
                    rate = float(getter())
                except Exception:
                    rate = 0.0
                if rate:
                    return rate
        engine = self.archer_engine
        getter = (
            getattr(engine, "get_hit_rate", None)
            if engine is not None
            else None
        )
        if callable(getter):
            try:
                counts = getter()
            except Exception:
                counts = None
            rate = _hit_rate_from_visit_counts(counts)
            if rate is not None:
                return rate
        return 0.0

    def expert_occupancy_bytes(self) -> float:
        total = 0.0
        dispatcher = self.expert_dispatcher
        if dispatcher is not None:
            getter = getattr(dispatcher, "get_cache_occupancy_bytes", None)
            if callable(getter):
                try:
                    total += float(getter())
                except Exception:
                    pass
        engine = self.archer_engine
        getter = (
            getattr(engine, "get_expert_occupancy_bytes", None)
            if engine is not None
            else None
        )
        if callable(getter):
            try:
                total += float(getter())
            except Exception:
                pass
        return total

    def wasted_prefetch_bytes(self) -> float:
        engine = self.archer_engine
        getter = (
            getattr(engine, "get_wasted_prefetch_bytes", None)
            if engine is not None
            else None
        )
        if callable(getter):
            try:
                return float(getter())
            except Exception:
                return 0.0
        return 0.0

    def prefetch_experts_list(
        self,
        layer_id: int,
        expert_list: List[int],
        priority: Optional[int] = None,
    ):
        if self.archer_engine is None:
            return
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        if not tensor_ids:
            return
        band = self.route_ahead_priority if priority is None else priority
        batched_issue = getattr(self.archer_engine, "prefetch_tensors", None)
        if callable(batched_issue):
            batched_issue(tensor_ids, priority=band)
            return
        for tensor_id in tensor_ids:
            gpu_id = self.archer_engine.get_node_default_device([tensor_id])
            self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)

    def fetch_experts_lock_cache(self, layer_id: int, expert_list: List[int]):
        if self.archer_engine is None:
            return
        tensor_ids = []
        for j in expert_list:
            tensor_ids.append(self.expert_tensor_map[(layer_id, j)])
        self.archer_engine.replace_cache_candidates(tensor_ids)

    def prefetch_experts(self, layer_id: int, expert_matrix):
        if self.archer_engine is None:
            return
        profiler = IOProfiler.instance() if IOProfiler is not None else None
        nvtx_cm = nullcontext()
        if HAS_NVTX and nvtx is not None:
            nvtx_cm = nvtx.annotate("prefetch_trigger", color="green")
        profiler_cm = (
            profiler.time("prefetch_trigger")
            if profiler is not None
            else nullcontext()
        )

        with profiler_cm:
            with nvtx_cm:
                expert_list = []
                for i in range(layer_id, self.num_layers):
                    for j in range(self.num_experts):
                        if expert_matrix[i, j] > 0:
                            expert_list.append(
                                (
                                    self.expert_tensor_map[(i, j)],
                                    expert_matrix[i, j],
                                )
                            )
                ordered_expert_list = sorted(
                    expert_list, key=lambda x: x[1], reverse=True
                )
                tensor_ids = [x[0] for x in ordered_expert_list]
                assert len(np.unique(tensor_ids)) == len(tensor_ids)
                self.archer_engine.replace_cache_candidates(tensor_ids)
                for tensor_id in tensor_ids:
                    gpu_id = self.archer_engine.get_node_default_device(
                        [tensor_id]
                    )
                    self.archer_engine.enqueue_prefetch(tensor_id, gpu_id)

    def speculative_prefetch(
        self,
        layer_idx: int,
        router_logits: Optional[Any] = None,
        *,
        expert_ids: Optional[List[int]] = None,
        prefetch_layer_id: Optional[int] = None,
    ):
        """Speculatively prefetch experts for an upcoming layer. Two modes:

        * Legacy (default): pool ``router_logits`` via ``mean(0)`` and
          prefetch the ``min(2, num_experts)`` top experts for
          ``layer_idx + 1``. Unchanged pre-A2 behavior (non-spec decode).
        * Explicit (DFlash route-ahead seam, Track A2): when ``expert_ids``
          is given, prefetch exactly that set for ``prefetch_layer_id``
          (default ``layer_idx + 1``) via ``prefetch_experts_list`` -- no
          mean/topk pooling. Empty ``expert_ids`` is a safe no-op.

        Returns ``None``. Raises ``ValueError`` if both are ``None``.
        """
        if expert_ids is not None:
            if not expert_ids:
                return
            target_layer = (
                prefetch_layer_id
                if prefetch_layer_id is not None
                else layer_idx + 1
            )
            if target_layer >= self.num_layers:
                return
            self.prefetch_experts_list(target_layer, list(expert_ids))
            self._last_speculative_prediction = set(expert_ids)
            return

        if router_logits is None:
            raise ValueError(
                "speculative_prefetch requires router_logits (legacy mode) "
                + "or expert_ids (explicit route-ahead mode); got neither."
            )

        next_layer = layer_idx + 1
        if next_layer >= self.num_layers:
            return

        num_experts_to_prefetch = min(2, self.num_experts)
        if hasattr(router_logits, "topk"):
            import torch

            topk_indices: List[int] = (
                torch.topk(
                    router_logits.float().view(-1, self.num_experts).mean(0),
                    num_experts_to_prefetch,
                )
                .indices.cpu()
                .tolist()
            )
        else:
            logits_np = (
                np.array(router_logits).reshape(-1, self.num_experts).mean(0)
            )
            topk_indices = np.argsort(logits_np)[-num_experts_to_prefetch:][
                ::-1
            ].tolist()

        self._last_speculative_prediction = set(topk_indices)
        if self._overlap_active():
            generation, _accepted = self.plan_candidates(
                next_layer, topk_indices
            )
            return generation

        self.prefetch_experts_list(
            next_layer, topk_indices, priority=BACKGROUND_PREFETCH_PRIORITY
        )
        return None

    def correct_prefetch(
        self,
        layer_idx: int,
        actual_expert_ids: List[int],
        predicted_expert_ids: Optional[Set[int]] = None,
    ):
        if layer_idx >= self.num_layers:
            self._last_speculative_prediction = set()
            return

        predicted = predicted_expert_ids
        if predicted is None:
            predicted = getattr(self, "_last_speculative_prediction", set())

        missed = [e for e in actual_expert_ids if e not in predicted]
        if missed:
            self.prefetch_experts_list(layer_idx, missed)

        self._last_speculative_prediction = set()

    def configure_overlap_policy(self, config: Any) -> None:
        policy = getattr(config, "overlap_prefetch_policy", "off")
        self._overlap_policy = policy
        if policy == "off":
            self.overlap_controller = None
            self._overlap_caps = None
            return

        self.overlap_controller = OverlapBudgetController(
            policy=policy,
            alpha=config.overlap_prefetch_ewma_alpha,
            safety_factor=config.overlap_prefetch_safety_factor,
            max_window_bytes=config.overlap_prefetch_max_window_bytes,
            max_inflight_bytes=config.overlap_prefetch_max_inflight_bytes,
            cold_start_experts=config.overlap_prefetch_cold_start_experts,
        )
        self._overlap_generation = 0
        self._overlap_stats = {
            "decisions": 0,
            "native_capability_misses": 0,
            "stale_compute_samples": 0,
        }
        caps = NativeOverlapCapabilities(
            self.archer_engine, self.expert_dispatcher
        )
        self._overlap_caps = caps
        if caps.dispatcher_timing_ready:
            self.expert_dispatcher.set_overlap_compute_timing_enabled(True)

    def _overlap_active(self) -> bool:
        return (
            getattr(self, "_overlap_policy", "off") in ("observe", "enforce")
            and getattr(self, "overlap_controller", None) is not None
        )

    def _next_generation(self) -> int:
        self._overlap_generation += 1
        return self._overlap_generation

    def plan_candidates(
        self,
        layer_id: int,
        ranked_expert_ids: List[int],
        scores: Optional[List[float]] = None,
    ) -> Tuple[Optional[int], List[int]]:
        if not self._overlap_active():
            return None, list(ranked_expert_ids)

        caps = self._overlap_caps
        if self._overlap_policy == "observe":
            if caps is None or not caps.scheduler_ready:
                self._overlap_stats["native_capability_misses"] += 1
            self._overlap_stats["decisions"] += 1
            self.prefetch_experts_list(
                layer_id,
                list(ranked_expert_ids),
                priority=BACKGROUND_PREFETCH_PRIORITY,
            )
            return None, list(ranked_expert_ids)

        if caps is None or not caps.enforce_ready:
            self._overlap_stats["native_capability_misses"] += 1
            self._overlap_stats["decisions"] += 1
            return None, []

        candidates = []
        for position, expert_id in enumerate(ranked_expert_ids):
            score = (
                scores[position]
                if scores is not None and position < len(scores)
                else float(len(ranked_expert_ids) - position)
            )
            nbytes = self.expert_nbytes_map.get((layer_id, expert_id))
            candidates.append(Candidate(expert_id, score, nbytes))

        inflight = int(self.archer_engine.get_inflight_prefetch_bytes())
        decision = self.overlap_controller.admit(
            layer_id, candidates, inflight_bytes=inflight
        )
        self._overlap_stats["decisions"] += 1
        if not decision.expert_ids:
            return None, []

        generation = self._next_generation()
        issued_nbytes = {
            expert_id: int(self.expert_nbytes_map[(layer_id, expert_id)])
            for expert_id in decision.expert_ids
        }
        self.overlap_controller.record_issue(
            layer_id, generation, issued_nbytes
        )
        tensor_ids = [
            self.expert_tensor_map[(layer_id, expert_id)]
            for expert_id in decision.expert_ids
        ]
        admission = self.archer_engine.schedule_prefetch_tensors(
            tensor_ids,
            priority=self.route_ahead_priority,
            generation=generation,
            layer_id=layer_id,
            max_inflight_bytes=self.overlap_controller.max_inflight_bytes,
        )
        accepted = list(getattr(admission, "accepted_tensor_ids", []) or [])
        return generation, accepted

    def correct_to_native_route(
        self, layer_id: int, actual_expert_ids: List[int]
    ) -> None:
        if not self._overlap_active():
            return
        caps = self._overlap_caps
        if caps is None or not caps.scheduler_ready:
            return
        self.drain_native_prefetch_samples()

    def abort_prefetch_generations(
        self, generations: List[int], reason: str
    ) -> None:
        if not self._overlap_active():
            return
        caps = self._overlap_caps
        if caps is None or not caps.scheduler_ready:
            return
        try:
            for generation in generations:
                self.archer_engine.cancel_prefetch_generation(
                    generation, -1, []
                )
        finally:
            self.drain_native_prefetch_samples()

    def drain_native_prefetch_samples(self) -> None:
        if not self._overlap_active():
            return
        caps = self._overlap_caps
        if caps is None or not caps.scheduler_ready:
            return
        self.archer_engine.drain_prefetch_samples()

    def observe_compute_samples(self, samples: List[Any]) -> None:
        if not self._overlap_active():
            return
        for sample in samples:
            self.overlap_controller.observe_compute(
                sample.layer_id,
                sample.kernel_start_offset_ns,
                sample.kernel_end_offset_ns,
            )

    def record_stale_compute_samples(self, count: int) -> None:
        if not self._overlap_active():
            return
        self._overlap_stats["stale_compute_samples"] = self._overlap_stats.get(
            "stale_compute_samples", 0
        ) + int(count)

    def overlap_prefetch_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = dict(self._overlap_stats)
        if self.overlap_controller is not None:
            stats.update(self.overlap_controller.snapshot())
        return stats
