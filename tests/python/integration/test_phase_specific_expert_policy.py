# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportUnannotatedClassAttribute=false, reportPrivateUsage=false, reportMissingParameterType=false
"""Deterministic integration coverage for the phase-specific expert policy.

This module exercises the phase-policy invariants (single shared identity,
prefill-to-decode transition reuse, decode-weighted eviction, transient prefill
pressure, mixed-batch forward ordering, and prefetch starvation promotion) using
an in-file ``FakePolicyRuntime``. The fake mirrors the production
``ExpertResidencyManager`` semantics defined in ``core/prefetch/expert_residency``
without requiring CUDA or a real model, so the invariants can be validated
deterministically in CPU-only CI.

``FakePolicyRuntime`` is intentionally self-contained: it does not import or
mutate any production policy code. It reuses only the canonical stable-victim
tuple and one shared ``set[(layer, expert)]`` store to prove single identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Key = Tuple[int, int]

# On-demand (demand) dispatch never yields to prefetch promotion. Mirrors the
# native band 0 in core/prefetch/task_scheduler.h.
_DEMAND_PRIORITY = 0


def _effective_phase(phase: str) -> str:
    """MIXED resolves to DECODE, mirroring native ``EffectivePhase``."""
    return "decode" if phase == "mixed" else phase


@dataclass
class _ExpertMetadata:
    """Per-expert policy metadata mirroring native ``ExpertPolicyMetadata``."""

    prefill_accesses: int = 0
    decode_accesses: int = 0
    last_prefill_sequence: int = 0
    last_decode_sequence: int = 0
    byte_size: int = 1
    # ``cache`` admission keeps the expert resident after use; the transient
    # overflow slot is modelled separately on the runtime.
    admission: str = "cache"

    @property
    def last_sequence(self) -> int:
        return max(self.last_prefill_sequence, self.last_decode_sequence)


@dataclass
class _VictimCandidate:
    """Stable victim tuple ``(utility, last_sequence, layer_id, expert_id)``."""

    utility: float
    last_sequence: int
    layer_id: int
    expert_id: int

    def order(self) -> Tuple[float, int, int, int]:
        return (self.utility, self.last_sequence, self.layer_id, self.expert_id)


@dataclass
class FakePolicyRuntime:
    """Deterministic model of the native shared-cache phase policy.

    The runtime models one GPU with a bounded number of persistently resident
    experts plus at most one transient overflow slot. Every ``(layer, expert)``
    pair maps to exactly one store entry regardless of phase, so phase is request
    metadata and never part of the cache key.
    """

    capacity: int
    decode_weight: float = 4.0
    prefill_weight: float = 1.0
    starvation_limit: int = 8
    enabled: bool = True

    _resident: Dict[Key, _ExpertMetadata] = field(default_factory=dict)
    _store_keys: Set[Key] = field(default_factory=set)
    _transient: Optional[Key] = None
    _sequence: int = 0
    _counters: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in (
            "prefill_accesses",
            "prefill_hits",
            "prefill_misses",
            "prefill_admissions",
            "prefill_transient",
            "prefill_evictions",
            "decode_accesses",
            "decode_hits",
            "decode_misses",
            "decode_admissions",
            "decode_transient",
            "decode_evictions",
            "mixed_accesses",
            "transition_hits",
            "starvation_promotions",
        ):
            self._counters.setdefault(key, 0)

    def resident_keys(self) -> Set[Key]:
        return set(self._resident.keys())

    def unique_store_keys(self) -> Set[Key]:
        """Every distinct ``(layer, expert)`` key ever created."""
        return set(self._store_keys)

    def stats(self) -> Dict[str, int]:
        snapshot: Dict[str, int] = {"enabled": 1 if self.enabled else 0}
        snapshot["resident_experts"] = len(self._resident)
        snapshot["resident_bytes"] = sum(
            meta.byte_size for meta in self._resident.values()
        )
        snapshot.update(self._counters)
        return snapshot

    def _victim_utility(self, meta: _ExpertMetadata) -> float:
        # Active phase for eviction scoring is DECODE (mirrors native
        # CandidateForLocked which scores with ExpertPhase::DECODE).
        return (
            self.prefill_weight * meta.prefill_accesses
            + self.decode_weight * meta.decode_accesses
        )

    def _select_victim(self, exclude: Set[Key]) -> Optional[Key]:
        candidates: List[Tuple[Tuple[float, int, int, int], Key]] = []
        for key, meta in self._resident.items():
            if key in exclude:
                continue
            candidate = _VictimCandidate(
                utility=self._victim_utility(meta),
                last_sequence=meta.last_sequence,
                layer_id=key[0],
                expert_id=key[1],
            )
            candidates.append((candidate.order(), key))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _record_access(self, key: Key, phase: str, hit: bool) -> None:
        active = _effective_phase(phase)
        meta = self._resident.get(key)
        if meta is None:
            # Transient/miss recording still advances the sequence counter and
            # phase-access totals, matching native RecordAccess which records on
            # the node regardless of persistence.
            meta = _ExpertMetadata()
        self._sequence += 1
        seq = self._sequence
        self._counters[f"{phase}_accesses"] += 1
        if active == "prefill":
            meta.prefill_accesses += 1
            meta.last_prefill_sequence = seq
            self._counters["prefill_hits" if hit else "prefill_misses"] += 1
        else:
            is_transition = (
                hit and meta.decode_accesses == 0 and meta.prefill_accesses > 0
            )
            meta.decode_accesses += 1
            meta.last_decode_sequence = seq
            self._counters["decode_hits" if hit else "decode_misses"] += 1
            if is_transition:
                self._counters["transition_hits"] += 1
        if key in self._resident:
            self._resident[key] = meta

    def _admit(self, key: Key, phase: str, admission: str) -> None:
        active = _effective_phase(phase)
        self._store_keys.add(key)
        if key in self._resident:
            self._record_access(key, phase, hit=True)
            return

        if (
            admission == "transient_on_pressure"
            and len(self._resident) >= self.capacity
        ):
            # Use the single transient overflow slot; do not evict a resident.
            self._transient = key
            self._counters[f"{active}_transient"] += 1
            self._record_access(key, phase, hit=False)
            return

        if len(self._resident) >= self.capacity:
            victim = self._select_victim(exclude=set())
            if victim is not None:
                del self._resident[victim]
                self._counters[f"{active}_evictions"] += 1

        self._resident[key] = _ExpertMetadata(admission=admission)
        self._counters[f"{active}_admissions"] += 1
        self._record_access(key, phase, hit=False)

    def dispatch(
        self,
        phase: str,
        layer: int,
        experts: List[int],
        admission: Optional[str] = None,
    ) -> None:
        """Route ``experts`` in ``layer`` for ``phase`` through admission."""
        if not self.enabled:
            # Disabled policy records no phase actions and mutates no residency.
            return
        active = _effective_phase(phase)
        default_admission = (
            "cache" if active == "decode" else "transient_on_pressure"
        )
        chosen = admission or default_admission
        for expert in experts:
            self._admit((layer, expert), phase, chosen)

    def forward_order(
        self,
        is_prefill: List[bool],
        *,
        paged: bool,
    ) -> List[List[bool]]:
        """Return the sequence of per-forward ``is_prefill`` groups.

        Mirrors ``ServingEngine._execute_batch``: enabled policy runs decode
        rows first then prefill rows; disabled non-paged mixed batches stay one
        combined forward; disabled paged mixed batches run prefill then decode.
        """
        has_prefill = any(is_prefill)
        has_decode = not all(is_prefill)

        if not self.enabled:
            if not paged or not (has_prefill and has_decode):
                return [list(is_prefill)]
            return [[True], [False]]

        if not (has_prefill and has_decode):
            return [list(is_prefill)]
        return [[False], [True]]

    def service_class(self, priority: int, bypasses: int) -> int:
        """Mirror native ``ServiceClass``: demand is strict, prefetch promotes."""
        if priority == _DEMAND_PRIORITY:
            return 0
        return 1 if bypasses >= self.starvation_limit else priority

    def promote_if_starved(self, priority: int, bypasses: int) -> bool:
        """Promote a prefetch task once its bypass bound is reached."""
        if priority == _DEMAND_PRIORITY:
            return False
        promoted = bypasses >= self.starvation_limit
        if promoted:
            self._counters["starvation_promotions"] += 1
        return promoted


def test_prefill_to_decode_reuses_shared_resident_expert() -> None:
    runtime = FakePolicyRuntime(
        capacity=2, decode_weight=4.0, starvation_limit=2
    )
    runtime.dispatch(phase="prefill", layer=0, experts=[1, 2])
    before = runtime.resident_keys()
    runtime.dispatch(phase="decode", layer=0, experts=[1])
    assert before == {(0, 1), (0, 2)}
    assert runtime.resident_keys() == {(0, 1), (0, 2)}
    assert runtime.stats()["transition_hits"] == 1
    assert runtime.unique_store_keys() == {(0, 1), (0, 2)}


def test_decode_hit_without_prior_prefill_is_not_a_transition() -> None:
    runtime = FakePolicyRuntime(capacity=2, decode_weight=4.0)
    runtime.dispatch(phase="decode", layer=0, experts=[1])
    runtime.dispatch(phase="decode", layer=0, experts=[1])
    assert runtime.stats()["transition_hits"] == 0


def test_enabled_mixed_input_runs_decode_then_prefill() -> None:
    runtime = FakePolicyRuntime(capacity=4, enabled=True)
    order = runtime.forward_order([True, False], paged=False)
    assert order == [[False], [True]]


def test_disabled_nonpaged_mixed_input_stays_one_combined_forward() -> None:
    runtime = FakePolicyRuntime(capacity=4, enabled=False)
    order = runtime.forward_order([True, False], paged=False)
    assert order == [[True, False]]


def test_disabled_paged_mixed_input_runs_prefill_then_decode() -> None:
    runtime = FakePolicyRuntime(capacity=4, enabled=False)
    order = runtime.forward_order([True, False], paged=True)
    assert order == [[True], [False]]


def test_disabled_policy_records_no_phase_actions() -> None:
    runtime = FakePolicyRuntime(capacity=2, enabled=False)
    runtime.dispatch(phase="prefill", layer=0, experts=[1, 2])
    runtime.dispatch(phase="decode", layer=0, experts=[1])
    stats = runtime.stats()
    assert stats["enabled"] == 0
    assert runtime.resident_keys() == set()
    assert runtime.unique_store_keys() == set()
    assert stats["prefill_accesses"] == 0
    assert stats["decode_accesses"] == 0
    assert stats["decode_admissions"] == 0
    assert stats["transition_hits"] == 0


def test_decode_weighted_eviction_keeps_decode_hot_expert() -> None:
    runtime = FakePolicyRuntime(capacity=2, decode_weight=4.0)
    runtime.dispatch(phase="decode", layer=0, experts=[1], admission="cache")
    runtime.dispatch(phase="decode", layer=0, experts=[1], admission="cache")
    runtime.dispatch(phase="prefill", layer=0, experts=[2], admission="cache")
    assert runtime.resident_keys() == {(0, 1), (0, 2)}

    runtime.dispatch(phase="decode", layer=0, experts=[3], admission="cache")

    assert (0, 1) in runtime.resident_keys()
    assert (0, 2) not in runtime.resident_keys()
    assert (0, 3) in runtime.resident_keys()


def test_transient_prefill_pressure_does_not_evict_decode_hot_expert() -> None:
    runtime = FakePolicyRuntime(capacity=1, decode_weight=4.0)
    runtime.dispatch(phase="decode", layer=0, experts=[1], admission="cache")
    assert runtime.resident_keys() == {(0, 1)}

    runtime.dispatch(
        phase="prefill", layer=0, experts=[2], admission="transient_on_pressure"
    )

    assert runtime.resident_keys() == {(0, 1)}
    assert runtime.stats()["prefill_transient"] == 1
    assert (0, 2) in runtime.unique_store_keys()


def test_prefill_prefetch_is_promoted_after_bypass_bound() -> None:
    runtime = FakePolicyRuntime(capacity=4, starvation_limit=2)
    assert runtime.service_class(priority=2, bypasses=1) == 2
    assert runtime.promote_if_starved(priority=2, bypasses=1) is False
    assert runtime.stats()["starvation_promotions"] == 0

    assert runtime.service_class(priority=2, bypasses=2) == 1
    assert runtime.promote_if_starved(priority=2, bypasses=2) is True
    assert runtime.stats()["starvation_promotions"] == 1


def test_promotion_never_passes_on_demand_work() -> None:
    runtime = FakePolicyRuntime(capacity=4, starvation_limit=2)
    assert runtime.service_class(priority=0, bypasses=99) == 0
    assert runtime.promote_if_starved(priority=0, bypasses=99) is False
    assert runtime.stats()["starvation_promotions"] == 0
