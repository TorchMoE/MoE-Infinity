from __future__ import annotations

import types
from dataclasses import dataclass, field

import pytest
import torch

from moe_infinity.serving.engine import ContinuousBatchingEngine
from moe_infinity.serving.mla_cache import MLAPagedKVCache
from moe_infinity.serving.sequence import SamplingParams
from moe_infinity.serving.spec_cache_adapter import (
    EXECUTION_CONTEXT_PAGED_MLA,
    PagedCacheAdapter,
    PagedCacheSnapshot,
)
from moe_infinity.serving.spec_session_driver import (
    EXECUTION_CONTEXT_TEMPORARY_DYNAMIC,
    SpecSessionDriver,
    TemporaryDynamicCacheContext,
)
from moe_infinity.spec_decode.dflash import DFlashSpeculator
from moe_infinity.spec_decode.protocols import RichForwardResult


def _cache() -> MLAPagedKVCache:
    return MLAPagedKVCache(
        num_blocks=8,
        block_size=2,
        num_layers=2,
        latent_dim=3,
        rope_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_adapter_append_snapshot_truncate_and_metadata_use_owned_pages() -> (
    None
):
    cache = _cache()
    adapter = PagedCacheAdapter(cache, seq_id=7, initial_length=3)
    original_pages = tuple(cache.get_block_table(7))

    snapshot = adapter.snapshot()
    adapter.append(4)
    metadata = adapter.build_attention_metadata(
        query_length=4, is_prefill=False
    )

    assert snapshot == PagedCacheSnapshot(3, original_pages)
    assert adapter.cache_kind == "paged"
    assert adapter.mode == EXECUTION_CONTEXT_PAGED_MLA
    assert metadata.seq_id == 7
    assert metadata.seq_lens.tolist() == [7]
    assert metadata.block_tables[0, :4].tolist() == cache.get_block_table(7)
    assert metadata.slot_mapping.tolist() == [
        cache.get_block_table(7)[1] * 2 + 1,
        cache.get_block_table(7)[2] * 2,
        cache.get_block_table(7)[2] * 2 + 1,
        cache.get_block_table(7)[3] * 2,
    ]

    adapter.truncate(5)
    assert adapter.logical_length() == 5
    adapter.restore(snapshot)
    assert adapter.logical_length() == 3
    assert tuple(cache.get_block_table(7)) == original_pages


def test_adapter_isolates_sequences_and_release_frees_only_its_owner() -> None:
    cache = _cache()
    first = PagedCacheAdapter(cache, seq_id=1, initial_length=2)
    second = PagedCacheAdapter(cache, seq_id=2, initial_length=2)
    second_pages = tuple(cache.get_block_table(2))

    first.append(3)
    first.truncate(1)

    assert tuple(cache.get_block_table(2)) == second_pages
    assert second.logical_length() == 2
    first.release()
    with pytest.raises(KeyError, match="unknown sequence id: 1"):
        cache.get_block_table(1)
    assert tuple(cache.get_block_table(2)) == second_pages


def test_adapter_release_is_idempotent_and_rejects_later_mutation() -> None:
    adapter = PagedCacheAdapter(_cache(), seq_id=3, initial_length=1)

    adapter.release()
    adapter.release()

    with pytest.raises(RuntimeError, match="released"):
        adapter.append(1)
    with pytest.raises(RuntimeError, match="released"):
        adapter.snapshot()


def test_adapter_reports_resident_only_preemption_without_duplicate_storage() -> (
    None
):
    cache = _cache()
    adapter = PagedCacheAdapter(cache, seq_id=4, initial_length=3)
    storage = cache.get_mla_cache_tensors()
    pages = tuple(cache.get_block_table(4))

    assert adapter.swap_out() is False
    assert adapter.swap_in() is False
    assert adapter.cache is cache
    assert adapter.cache.get_mla_cache_tensors() is storage
    assert tuple(cache.get_block_table(4)) == pages


def test_dflash_target_forward_passes_mla_metadata_and_returns_engine_handle() -> (
    None
):
    cache = _cache()
    adapter = PagedCacheAdapter(cache, seq_id=5, initial_length=2)
    metadata = adapter.build_attention_metadata(query_length=2, is_prefill=True)
    calls: list[object] = []

    def rich(
        token_ids: list[int],
        attention_metadata: object,
        logits_to_keep: int = 0,
    ) -> RichForwardResult:
        calls.append(attention_metadata)
        return RichForwardResult(
            logits=torch.zeros(1, len(token_ids), 8),
            hidden_states=(torch.zeros(1, len(token_ids), 4),),
            cache_handle=cache,
        )

    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.moe = types.SimpleNamespace(_native_model_forward_rich=rich)

    _, _, handle = speculator._forward_target(
        torch.tensor([[1, 2]]),
        past_key_values=adapter,
        logits_to_keep=1,
        attention_metadata=metadata,
    )

    assert calls == [metadata]
    assert handle is cache


class _DraftCache:
    def __init__(self) -> None:
        self.released = False

    def crop(self, length: int) -> None:
        if length == 0:
            self.released = True


@dataclass
class _DriverSession:
    emitted: list[int]
    target_kv: object
    draft_kv: _DraftCache = field(default_factory=_DraftCache)
    finished: bool = False

    def clear_pending(self) -> None:
        pass


class _PagedDriverSpeculator:
    def __init__(
        self, cache: MLAPagedKVCache, *, dflash_block_size: int = 2
    ) -> None:
        self.moe = types.SimpleNamespace(
            _native_mla_cache=cache,
            _cached_past_key_values=object(),
            _get_mla_attention_modules=lambda: [object()],
        )
        self.received_adapter: object | None = None
        self.begin_calls: list[dict[str, object]] = []
        self.begin_error: BaseException | None = None
        self.append_on_draft = 0
        self.config = types.SimpleNamespace(block_size=dflash_block_size)

    def begin_session(
        self,
        input_ids: torch.Tensor,
        *,
        target_cache_adapter: object | None = None,
        **kwargs: object,
    ) -> _DriverSession:
        self.begin_calls.append(
            {**kwargs, "target_cache_adapter": target_cache_adapter}
        )
        self.received_adapter = target_cache_adapter
        if self.begin_error is not None:
            raise self.begin_error
        return _DriverSession(
            emitted=[int(input_ids[0, -1]) + 1],
            target_kv=(
                target_cache_adapter
                if target_cache_adapter is not None
                else object()
            ),
        )

    def draft_round(self, session: _DriverSession) -> object:
        if self.append_on_draft:
            session.target_kv.append(self.append_on_draft)
        return types.SimpleNamespace(tokens=2, expert_bytes=0)

    def verify_round(self, session: _DriverSession) -> object:
        del session
        return types.SimpleNamespace(committed_count=1)


def test_standalone_driver_keeps_paged_mla_default_off() -> None:
    driver = SpecSessionDriver(_PagedDriverSpeculator(_cache()))

    record = _begin_driver_record(driver, "standalone-off", 5, [1, 2])

    assert record.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    assert driver.admission_stats["enabled"] is False


@pytest.mark.parametrize(
    "block_size",
    [pytest.param(None, id="missing"), pytest.param("5", id="non-int"), 0, 1],
)
def test_invalid_dflash_block_size_falls_back_without_mla_allocation(
    block_size: object,
) -> None:
    cache = _cache()
    speculator = _PagedDriverSpeculator(cache)
    if block_size is None:
        del speculator.config.block_size
    else:
        speculator.config.block_size = block_size
    driver = SpecSessionDriver(speculator, enable_paged_mla=True)

    record = _begin_driver_record(driver, "invalid-block", 48, [1, 2])

    assert record.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    assert record.diagnostics()["paged_mla_admission"] == {
        "eligible": False,
        "admitted": False,
        "reason": "ineligible",
    }
    assert speculator.received_adapter is None
    assert cache.free_block_count == cache.num_blocks


def test_driver_selects_paged_mla_for_eligible_greedy_session_without_dynamic_context() -> (
    None
):
    speculator = _PagedDriverSpeculator(_cache())
    previous_dense_owner = speculator.moe._cached_past_key_values
    driver = SpecSessionDriver(speculator, enable_paged_mla=True)

    record = driver.begin(
        request_id="paged",
        seq_id=6,
        prompt_token_ids=[1, 2],
        max_new_tokens=3,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        stop_token_ids=(),
        callbacks=(),
    )

    assert record.execution_context.mode == EXECUTION_CONTEXT_PAGED_MLA
    assert driver.execution_context_mode == EXECUTION_CONTEXT_PAGED_MLA
    assert not isinstance(
        record.execution_context, TemporaryDynamicCacheContext
    )
    assert record.spec_session.target_kv is speculator.received_adapter
    assert isinstance(record.spec_session.target_kv, PagedCacheAdapter)
    assert (
        record.spec_session.target_kv.cache is speculator.moe._native_mla_cache
    )
    assert speculator.moe._cached_past_key_values is previous_dense_owner

    draft_cache = record.spec_session.draft_kv
    driver.release(record)
    assert draft_cache.released
    with pytest.raises(KeyError, match="unknown sequence id: 6"):
        speculator.moe._native_mla_cache.get_block_table(6)


def test_driver_cancel_releases_paged_target_and_drafter_owners() -> None:
    speculator = _PagedDriverSpeculator(_cache())
    driver = SpecSessionDriver(speculator, enable_paged_mla=True)
    record = driver.begin(
        request_id="cancel",
        seq_id=12,
        prompt_token_ids=[1, 2],
        max_new_tokens=3,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        stop_token_ids=(),
        callbacks=(),
    )
    draft_cache = record.spec_session.draft_kv

    driver.cancel(12)

    assert record.cancelled and record.released
    assert draft_cache.released
    assert 12 not in driver.sessions
    with pytest.raises(KeyError, match="unknown sequence id: 12"):
        speculator.moe._native_mla_cache.get_block_table(12)


def test_driver_keeps_sampled_request_on_explicit_stage4a_fallback() -> None:
    speculator = _PagedDriverSpeculator(_cache())
    driver = SpecSessionDriver(speculator, enable_paged_mla=True)

    record = driver.begin(
        request_id="sampled",
        seq_id=7,
        prompt_token_ids=[1],
        max_new_tokens=2,
        temperature=0.7,
        top_k=0,
        top_p=1.0,
        stop_token_ids=(),
        callbacks=(),
    )

    assert record.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    assert isinstance(record.execution_context, TemporaryDynamicCacheContext)
    assert speculator.received_adapter is None
    assert speculator.begin_calls[0]["temperature"] == 0.7


def test_driver_caps_concurrent_resident_paged_mla_sessions() -> None:
    speculator = _PagedDriverSpeculator(_cache())
    driver = SpecSessionDriver(
        speculator,
        enable_paged_mla=True,
        max_resident_paged_speculative_sessions=1,
        min_free_mla_blocks_after_admission=1,
    )

    first = _begin_driver_record(driver, "first", 31, [1, 2])
    second = _begin_driver_record(driver, "second", 32, [3, 4])

    assert first.execution_context.mode == EXECUTION_CONTEXT_PAGED_MLA
    assert second.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    assert second.diagnostics()["paged_mla_admission"] == {
        "eligible": True,
        "admitted": False,
        "reason": "session_cap",
    }
    assert driver.admission_stats["counters"]["session_cap"] == 1


def test_driver_preserves_free_block_reserve_with_immediate_stage4a_fallback() -> (
    None
):
    cache = _cache()
    speculator = _PagedDriverSpeculator(cache)
    driver = SpecSessionDriver(
        speculator,
        enable_paged_mla=True,
        max_resident_paged_speculative_sessions=2,
        min_free_mla_blocks_after_admission=8,
    )

    record = _begin_driver_record(driver, "reserve", 33, [1, 2])

    assert record.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    assert record.diagnostics()["paged_mla_admission"]["reason"] == (
        "free_block_reserve"
    )
    assert cache.free_block_count == 8
    assert driver.admission_stats["counters"]["free_block_reserve"] == 1


def test_driver_admits_when_transient_peak_plus_reserve_exactly_fits() -> None:
    cache = MLAPagedKVCache(6, 2, 1, 3, 2, torch.float32, torch.device("cpu"))
    driver = SpecSessionDriver(
        _PagedDriverSpeculator(cache, dflash_block_size=5),
        enable_paged_mla=True,
        max_resident_paged_speculative_sessions=1,
        min_free_mla_blocks_after_admission=1,
    )

    record = _begin_driver_record(driver, "exact", 40, [1, 2], max_new_tokens=4)

    assert record.execution_context.mode == EXECUTION_CONTEXT_PAGED_MLA
    assert record.diagnostics()["paged_mla_admission"]["reason"] == "admitted"


def test_driver_falls_back_when_transient_peak_is_one_block_short() -> None:
    cache = MLAPagedKVCache(5, 2, 1, 3, 2, torch.float32, torch.device("cpu"))
    driver = SpecSessionDriver(
        _PagedDriverSpeculator(cache, dflash_block_size=5),
        enable_paged_mla=True,
        max_resident_paged_speculative_sessions=1,
        min_free_mla_blocks_after_admission=1,
    )

    record = _begin_driver_record(driver, "short", 41, [1, 2], max_new_tokens=4)

    assert record.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    assert record.diagnostics()["paged_mla_admission"]["reason"] == (
        "free_block_reserve"
    )
    assert cache.free_block_count == 5


def test_reserved_transient_peak_prevents_mid_verify_exhaustion() -> None:
    cache = MLAPagedKVCache(6, 2, 1, 3, 2, torch.float32, torch.device("cpu"))
    driver = SpecSessionDriver(
        _PagedDriverSpeculator(cache, dflash_block_size=5),
        enable_paged_mla=True,
        max_resident_paged_speculative_sessions=2,
        min_free_mla_blocks_after_admission=1,
    )
    first = _begin_driver_record(
        driver, "peak-owner", 46, [1, 2], max_new_tokens=4
    )

    competing = _begin_driver_record(
        driver, "peak-competitor", 47, [3, 4, 5], max_new_tokens=1
    )
    assert (
        competing.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    )

    first.spec_session.target_kv.append(8)
    assert len(cache.get_block_table(46)) == 5
    assert cache.free_block_count == 1


def test_driver_accounts_active_declared_headroom_and_release() -> None:
    cache = MLAPagedKVCache(8, 2, 1, 3, 2, torch.float32, torch.device("cpu"))
    driver = SpecSessionDriver(
        _PagedDriverSpeculator(cache, dflash_block_size=5),
        enable_paged_mla=True,
        max_resident_paged_speculative_sessions=2,
        min_free_mla_blocks_after_admission=1,
    )
    first = _begin_driver_record(
        driver, "first-budget", 42, [1, 2], max_new_tokens=4
    )

    blocked = _begin_driver_record(
        driver, "blocked-budget", 43, [1, 2], max_new_tokens=2
    )
    assert blocked.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    assert blocked.diagnostics()["paged_mla_admission"]["reason"] == (
        "free_block_reserve"
    )

    driver.release(first)
    admitted = _begin_driver_record(
        driver, "released-budget", 44, [1, 2], max_new_tokens=2
    )
    assert admitted.execution_context.mode == EXECUTION_CONTEXT_PAGED_MLA


def test_driver_counts_backend_begin_failure_instead_of_admission() -> None:
    cache = _cache()
    speculator = _PagedDriverSpeculator(cache)
    speculator.begin_error = RuntimeError("begin failed")
    driver = SpecSessionDriver(speculator, enable_paged_mla=True)

    with pytest.raises(RuntimeError, match="begin failed"):
        _begin_driver_record(driver, "begin-failed", 45, [1, 2])

    assert cache.free_block_count == 8
    assert driver.admission_stats["counters"]["admitted"] == 0
    assert driver.admission_stats["counters"]["begin_failed"] == 1


def test_driver_admits_after_prior_paged_session_releases() -> None:
    driver = SpecSessionDriver(
        _PagedDriverSpeculator(_cache()),
        enable_paged_mla=True,
        max_resident_paged_speculative_sessions=1,
        min_free_mla_blocks_after_admission=1,
    )
    first = _begin_driver_record(driver, "first", 34, [1, 2])
    blocked = _begin_driver_record(driver, "blocked", 35, [1, 2])
    assert blocked.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC

    driver.release(first)
    admitted = _begin_driver_record(driver, "admitted", 36, [1, 2])

    assert admitted.execution_context.mode == EXECUTION_CONTEXT_PAGED_MLA
    assert admitted.diagnostics()["paged_mla_admission"]["reason"] == "admitted"
    assert driver.admission_stats["active_sessions"] == 1


def test_engine_stats_expose_paged_mla_admission_decisions() -> None:
    driver = SpecSessionDriver(
        _PagedDriverSpeculator(_cache()),
        enable_paged_mla=True,
        max_resident_paged_speculative_sessions=1,
        min_free_mla_blocks_after_admission=1,
    )
    _begin_driver_record(driver, "first", 37, [1, 2])
    _begin_driver_record(driver, "blocked", 38, [1, 2])
    engine = ContinuousBatchingEngine.__new__(ContinuousBatchingEngine)
    engine._spec_session_driver = driver
    engine._sequences = {}
    engine._request_to_seq_ids = {}
    engine._completed_request_ids = set()
    engine._cancelled_request_ids = set()
    engine._request_failures = {}
    engine._num_steps = 0
    engine._total_generated_tokens = 0
    engine.kv_cache = types.SimpleNamespace(
        num_blocks=1,
        block_allocator=types.SimpleNamespace(num_free_blocks=1),
    )
    engine.memory_manager = types.SimpleNamespace(report=lambda: {})

    stats = engine.get_stats()

    assert stats["paged_mla_admission"] == driver.admission_stats
    assert stats["paged_mla_admission"]["counters"]["session_cap"] == 1


class _ServingModel:
    config = types.SimpleNamespace(vocab_size=128, eos_token_id=99)

    def eval(self) -> None:
        pass

    def forward(
        self, input_ids: torch.Tensor, **kwargs: object
    ) -> types.SimpleNamespace:
        del kwargs
        logits = torch.full((*input_ids.shape, 128), -1e9)
        for row in range(input_ids.shape[0]):
            for col in range(input_ids.shape[1]):
                logits[row, col, int(input_ids[row, col]) + 1] = 0
        return types.SimpleNamespace(logits=logits)


class _ServingOffload:
    def __init__(self) -> None:
        self.request_id = 0
        self.expert_tracer = types.SimpleNamespace(create_entry=lambda: 0)
        self.expert_layer_modules = [types.SimpleNamespace(seq_id_list=[])]

    def _generate_request_id(self) -> int:
        value = self.request_id
        self.request_id += 1
        return value


def _paged_engine(
    speculator: _PagedDriverSpeculator, *, enabled: bool
) -> ContinuousBatchingEngine:
    return ContinuousBatchingEngine(
        model=_ServingModel(),
        engine=_ServingOffload(),
        config={
            "device_memory_ratio": 0.75,
            "kv_cache_ratio": 0.25,
            "max_batch_size": 8,
            "max_tokens_per_step": 16,
            "block_size": 4,
            "num_layers": 1,
            "num_kv_heads": 2,
            "head_dim": 8,
            "dtype": "float32",
            "eos_token_id": 99,
            "num_kv_blocks": 32,
            "verify_token_budget": 8,
            "verify_expert_byte_budget": 128,
            "verify_token_deficit_cap": 32,
            "verify_expert_byte_deficit_cap": 512,
            "enable_deepseek_mla_paging": enabled,
            "max_resident_paged_speculative_sessions": 1,
            "min_free_mla_blocks_after_admission": 1,
        },
        speculative_draft=speculator,
    )


def test_engine_add_request_step_selects_paged_mla_only_when_enabled() -> None:
    enabled_engine = _paged_engine(
        _PagedDriverSpeculator(_cache()), enabled=True
    )
    enabled_engine.add_request(
        "enabled", [1, 2], SamplingParams(temperature=0.0, max_tokens=3)
    )
    enabled_engine.step()
    enabled_record = next(iter(enabled_engine.speculative_sessions.values()))
    assert enabled_record.execution_context.mode == EXECUTION_CONTEXT_PAGED_MLA

    default_off_engine = _paged_engine(
        _PagedDriverSpeculator(_cache()), enabled=False
    )
    default_off_engine.add_request(
        "default-off", [1, 2], SamplingParams(temperature=0.0, max_tokens=3)
    )
    default_off_engine.step()
    default_record = next(
        iter(default_off_engine.speculative_sessions.values())
    )
    assert (
        default_record.execution_context.mode
        == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    )


def test_engine_append_exhaustion_records_failure_and_releases_paged_owner() -> (
    None
):
    cache = MLAPagedKVCache(5, 2, 1, 3, 2, torch.float32, torch.device("cpu"))
    speculator = _PagedDriverSpeculator(cache)
    speculator.append_on_draft = 2
    engine = _paged_engine(speculator, enabled=True)
    engine.add_request(
        "exhaust", [1, 2], SamplingParams(temperature=0.0, max_tokens=4)
    )
    engine.step()
    seq_id = engine._request_to_seq_ids["exhaust"][0]
    cache.allocate_sequence(999, 8)

    with pytest.raises(RuntimeError, match="BlockAllocator exhausted"):
        engine.step()

    assert engine.get_request_failure("exhaust") == {
        "phase": "draft",
        "failure_type": "RuntimeError",
        "code": "speculative_draft_failed",
    }
    assert engine.speculative_sessions == {}
    assert engine.has_pending_requests() is False
    with pytest.raises(KeyError, match=f"unknown sequence id: {seq_id}"):
        cache.get_block_table(seq_id)


def _begin_driver_record(
    driver: SpecSessionDriver,
    request_id: str,
    seq_id: int,
    prompt_token_ids: list[int],
    max_new_tokens: int = 3,
):
    return driver.begin(
        request_id=request_id,
        seq_id=seq_id,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        stop_token_ids=(),
        callbacks=(),
    )


def test_driver_keeps_non_deepseek_owner_on_explicit_stage4a_fallback() -> None:
    speculator = _PagedDriverSpeculator(_cache())
    speculator.moe._get_mla_attention_modules = lambda: []
    driver = SpecSessionDriver(speculator, enable_paged_mla=True)

    record = driver.begin(
        request_id="qwen-or-hybrid",
        seq_id=8,
        prompt_token_ids=[1],
        max_new_tokens=2,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        stop_token_ids=(),
        callbacks=(),
    )

    assert record.execution_context.mode == EXECUTION_CONTEXT_TEMPORARY_DYNAMIC
    assert speculator.received_adapter is None


def test_canonical_session_prefill_and_verify_use_transient_mla_slots_then_truncate() -> (
    None
):
    cache = _cache()
    adapter = PagedCacheAdapter(cache, seq_id=10, initial_length=2)
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.device = "cpu"
    speculator.config = types.SimpleNamespace(
        block_size=2,
        target_layer_ids=[0],
        mask_token_id=0,
    )
    speculator._drafter_has_kv_cache = False
    speculator.moe = types.SimpleNamespace(engine=None)
    speculator.target = types.SimpleNamespace(
        config=types.SimpleNamespace(eos_token_id=None),
        generation_config=types.SimpleNamespace(eos_token_id=None),
    )
    speculator._configure_target_hooks = lambda input_ids: None
    speculator.route_ahead_stats = None
    metadata_seen: list[object] = []

    def forward_target(
        input_ids: torch.Tensor,
        past_key_values: object = None,
        logits_to_keep: int = 0,
        *,
        attention_metadata: object = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], object]:
        del logits_to_keep, kwargs
        metadata_seen.append(attention_metadata)
        query_len = int(input_ids.shape[1])
        logits = torch.zeros(1, query_len, 16)
        if query_len == 2 and past_key_values is adapter:
            logits[0, 0, 9] = 1
            logits[0, 1, 8] = 1
        else:
            logits[0, -1, 3] = 1
        hidden = (
            torch.zeros(1, query_len, 4),
            torch.zeros(1, query_len, 4),
        )
        return logits, hidden, cache

    speculator._forward_target = forward_target
    session = speculator.begin_session(
        torch.tensor([[1, 2]]),
        max_new_tokens=2,
        temperature=0.0,
        stop_token_ids=[],
        target_cache_adapter=adapter,
    )

    assert session.target_kv is adapter
    assert metadata_seen[0].is_prefill is True
    assert metadata_seen[0].seq_id == 10
    assert adapter.logical_length() == 2

    block = torch.tensor([[session.anchor, 4]])
    session._pending = True
    session._pending_block = block
    session._pending_prev_start = 2
    session._pending_cache_snapshot = adapter.snapshot()
    session.pending_draft_probs = None

    result = speculator.verify_round(session)

    assert result.accept == 0
    assert metadata_seen[1].is_prefill is False
    assert metadata_seen[1].seq_lens.tolist() == [4]
    assert metadata_seen[1].slot_mapping.numel() == 2
    assert adapter.logical_length() == 3
    assert session.start == 3
    assert session.step_trace[-1].target_cache_len == 3


def test_real_deepseek_rich_forward_writes_the_adapter_owned_cache_for_verify() -> (
    None
):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.deepseek_v3.modeling_deepseek_v3")
    from moe_infinity.entrypoints.big_modeling import MoE
    from moe_infinity.models.deepseek_mla_attention import adapt_deepseek_model

    config = transformers.DeepseekV3Config(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        n_group=1,
        topk_group=1,
        rope_interleave=True,
        vocab_size=32,
        first_k_dense_replace=1,
    )
    model = transformers.DeepseekV3ForCausalLM(config).eval()
    cache = MLAPagedKVCache(4, 2, 1, 4, 4, torch.float32, torch.device("cpu"))
    assert len(adapt_deepseek_model(model, cache, enabled=True)) == 1
    adapter = PagedCacheAdapter(cache, seq_id=11, initial_length=2)

    shell = MoE.__new__(MoE)
    shell.model = model
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    shell._native_mla_cache = cache
    shell._resolve_native_input_device = lambda: torch.device("cpu")

    prefill = shell._native_model_forward_rich(
        [1, 2],
        adapter.build_attention_metadata(query_length=2, is_prefill=True),
    )
    before_verify = cache.get_mla_cache_tensors().clone()
    adapter.append(2)
    verify = shell._native_model_forward_rich(
        [3, 4],
        adapter.build_attention_metadata(query_length=2, is_prefill=False),
    )

    assert isinstance(prefill, RichForwardResult)
    assert isinstance(verify, RichForwardResult)
    assert prefill.cache_handle is cache and verify.cache_handle is cache
    assert shell._cached_past_key_values is None
    assert not torch.equal(cache.get_mla_cache_tensors(), before_verify)


def test_rich_verify_uses_one_position_per_tentative_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moe_infinity.models.deepseek_mla_attention as mla_attention
    from moe_infinity.entrypoints.big_modeling import MoE

    calls: list[dict[str, object]] = []

    class _Model:
        def __call__(self, input_ids: torch.Tensor, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return types.SimpleNamespace(
                logits=torch.zeros(1, input_ids.shape[1], 8),
                hidden_states=(torch.zeros(1, input_ids.shape[1], 4),),
            )

    monkeypatch.setattr(
        mla_attention, "set_deepseek_mla_context", lambda *args: None
    )
    monkeypatch.setattr(
        mla_attention, "clear_deepseek_mla_context", lambda *args: None
    )
    shell = MoE.__new__(MoE)
    shell.model = _Model()
    shell._native_mla_cache = object()
    shell._native_attention_backend = None
    shell._resolve_native_input_device = lambda: torch.device("cpu")
    shell._get_mla_attention_modules = lambda: [object()]
    metadata = types.SimpleNamespace(
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        slot_mapping=torch.tensor([2, 3], dtype=torch.int64),
        is_prefill=False,
    )

    shell._native_model_forward_rich([3, 4], metadata)

    assert calls[0]["position_ids"].tolist() == [[2, 3]]
