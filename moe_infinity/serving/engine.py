from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import ceil
from typing import Callable, Optional, Protocol, cast

import torch

from moe_infinity.runtime.attention_backend import PagedAttentionBackend
from moe_infinity.runtime.attention_types import DECODE_GRAPH_REASONS

from .batch import (
    BatchBuilder,
    BatchMetadata,
    _slice_batch,
    split_prefill_decode_batch,
)
from .cuda_graph import CudaGraphRunner
from .kv_cache import PagedKVCache
from .memory_manager import MemoryManager
from .model_runner import ModelRunner
from .prefix_cache import CacheNamespace, PrefixCache
from .sampler import Sampler
from .scheduler import Scheduler
from .sequence import (
    SamplingParams,
    SequenceData,
    SequenceGroup,
    SequenceStatus,
)
from .spec_session_driver import (
    ServingSpecSession,
    SpecSessionDriver,
)

logger = logging.getLogger(__name__)


class EvictionSyncAdapter(Protocol):
    def on_request_finished(self, request_id: str) -> None: ...

    def on_request_aborted(self, request_id: str) -> None: ...


class SpeculativeGenerator(Protocol):
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor: ...


_eviction_sync: Optional[EvictionSyncAdapter] = None

_VERIFY_ADMISSION_MAX_RETRIES = 1024


def _debug_cleanup_reporting_failure(
    *,
    action: str,
    request_id: str,
    phase: str,
    primary: BaseException,
    reporting_error: BaseException,
) -> None:
    try:
        logger.debug(
            "cleanup %s attachment failed for request %s during %s; "
            "primary=%s reporting=%s",
            action,
            request_id,
            phase,
            type(primary).__name__,
            type(reporting_error).__name__,
            exc_info=reporting_error,
        )
    except BaseException:
        return


def set_eviction_sync(adapter: Optional[EvictionSyncAdapter]) -> None:
    global _eviction_sync
    _eviction_sync = adapter


@dataclass
class RequestOutput:
    request_id: str
    seq_id: int
    token_id: int
    token_text: Optional[str]
    finished: bool
    finish_reason: Optional[str] = None
    token_logprob: Optional[float] = None
    top_logprobs: Optional[dict[int, float]] = None
    usage: Optional[dict[str, int]] = None


class ContinuousBatchingEngine:
    model: object
    engine: object
    config: dict[str, object]
    tokenizer: Optional[object]
    device: torch.device
    dtype: torch.dtype
    eos_token_id: Optional[int]
    memory_manager: MemoryManager
    kv_cache: PagedKVCache
    scheduler: Scheduler
    model_runner: ModelRunner
    sampler: Sampler
    batch_builder: BatchBuilder
    _next_seq_id: int
    _num_steps: int
    _total_generated_tokens: int
    speculative_draft: SpeculativeGenerator | None

    def __init__(
        self,
        model: object,
        engine: object,
        config: dict[str, object],
        tokenizer: Optional[object] = None,
        speculative_draft: SpeculativeGenerator | None = None,
        decode_graph_capability_provider: object = None,
    ) -> None:
        self.model = model
        self.engine = engine
        self.config = dict(config)
        self.tokenizer = tokenizer
        self.device = self._resolve_device(model)
        self.dtype = self._resolve_dtype(self.config["dtype"])
        self.eos_token_id = self._resolve_eos_token_id(model, self.config)

        self.memory_manager = MemoryManager(
            device=self.device,
            device_memory_ratio=self._get_float_config(
                "device_memory_ratio", 0.75
            ),
            kv_cache_ratio=self._get_float_config("kv_cache_ratio", 0.25),
        )
        _ = self.memory_manager.compute_budget(
            model_memory_bytes=self._resolve_model_memory_bytes(model)
        )

        block_size = self._get_int_config("block_size")
        num_layers = self._get_int_config("num_layers")
        num_kv_heads = self._get_int_config("num_kv_heads")
        head_dim = self._get_int_config("head_dim")
        num_blocks = self._resolve_num_blocks(
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

        self.model_runner = ModelRunner(model, engine, device=self.device)

        self.prefix_cache: PrefixCache | None = None
        self.cache_namespace: CacheNamespace | None = None
        self._prefix_cache_enabled = self._get_bool_config(
            "enable_prefix_caching", False
        )
        self._prefix_cache_disabled_reason = "prefix-caching-disabled"
        self._runtime_epoch = 0
        self._prefix_cache_invalidations = 0

        capability = self._resolve_prefix_capability()
        physical_blocks = (
            capability.block_store.num_blocks
            if capability is not None
            and capability.supported
            and capability.block_store is not None
            else num_blocks
        )
        logical_num_blocks = min(num_blocks, physical_blocks)

        backend_storage = self._resolve_backend_storage(engine)
        self.kv_cache = PagedKVCache(
            num_blocks=logical_num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
            storage=backend_storage,
        )

        provider = self._maybe_bind_prefix_cache(capability)

        self.scheduler = Scheduler(
            self.kv_cache,
            max_batch_size=self._get_int_config("max_batch_size", 32),
            max_tokens_per_step=self._get_int_config(
                "max_tokens_per_step", 2048
            ),
            prefix_lease_provider=provider,
            cache_namespace=(
                self.cache_namespace if provider is not None else None
            ),
            verify_token_budget=self._get_optional_int_config(
                "verify_token_budget"
            ),
            verify_expert_byte_budget=self._get_optional_int_config(
                "verify_expert_byte_budget"
            ),
            verify_token_deficit_cap=self._get_optional_int_config(
                "verify_token_deficit_cap"
            ),
            verify_expert_byte_deficit_cap=self._get_optional_int_config(
                "verify_expert_byte_deficit_cap"
            ),
        )
        from moe_infinity.models.paged_attention_registry import (
            PagedAttentionLayerRegistry,
        )

        storage = self.kv_cache.storage
        backend = backend_storage and self._resolve_attention_backend(engine)
        if storage is None:
            self.paged_attention_registry = PagedAttentionLayerRegistry.empty(
                reason="native_paged_required"
            )
        else:
            self.paged_attention_registry = (
                PagedAttentionLayerRegistry.register(
                    model=model, backend=backend, storage=storage
                )
            )
        self.model_runner.paged_kv_storage = storage
        self.model_runner.paged_attention_registry = (
            self.paged_attention_registry
        )
        self.model_runner.decode_graph_capability_provider = (
            decode_graph_capability_provider
        )
        self.cuda_graph_runner = CudaGraphRunner(
            self.model_runner,
            storage,
            enabled=self._get_bool_config("enable_decode_cuda_graphs", False),
            batch_buckets=self._get_int_tuple_config(
                "decode_cuda_graph_batch_sizes", (1, 2, 4, 8, 16, 32)
            ),
            context_buckets=self._get_int_tuple_config(
                "decode_cuda_graph_context_sizes",
                (128, 256, 512, 1024, 2048, 4096),
            ),
            warmup_iters=self._get_int_config(
                "decode_cuda_graph_warmup_iters", 2
            ),
            max_graph_memory_bytes=self._get_int_config(
                "decode_cuda_graph_max_memory_bytes", 0
            ),
        )
        self.sampler = Sampler()
        self.batch_builder = BatchBuilder()
        self.speculative_draft = speculative_draft
        enable_paged_mla = self.config.get("enable_deepseek_mla_paging", False)
        if not isinstance(enable_paged_mla, bool):
            raise ValueError(
                "enable_deepseek_mla_paging must be a boolean value"
            )
        max_resident_paged_sessions = self._get_int_config(
            "max_resident_paged_speculative_sessions", 1
        )
        min_free_mla_blocks = self._get_int_config(
            "min_free_mla_blocks_after_admission", 1
        )
        if max_resident_paged_sessions < 0:
            raise ValueError(
                "max_resident_paged_speculative_sessions must be >= 0"
            )
        if min_free_mla_blocks < 1:
            raise ValueError("min_free_mla_blocks_after_admission must be >= 1")
        self._verify_scheduling_enabled = (
            self.scheduler.verify_scheduling_enabled
        )
        self._spec_session_driver = (
            SpecSessionDriver(
                speculative_draft,
                enable_paged_mla=enable_paged_mla,
                max_resident_paged_speculative_sessions=(
                    max_resident_paged_sessions
                ),
                min_free_mla_blocks_after_admission=min_free_mla_blocks,
            )
            if self._can_drive_verify_rounds()
            else None
        )

        self._next_seq_id = 0
        self._shutdown = False
        self._sequences: dict[int, SequenceData] = {}
        self._sequence_to_request_id: dict[int, str] = {}
        self._request_to_seq_ids: dict[str, list[int]] = {}
        self._request_outputs: dict[str, dict[int, list[int]]] = {}
        self._callbacks: dict[str, list[Callable[[RequestOutput], None]]] = {}
        self._completed_request_ids: set[str] = set()
        self._cancelled_request_ids: set[str] = set()
        self._request_failures: dict[str, dict[str, str]] = {}
        self._num_steps = 0
        self._total_generated_tokens = 0

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        on_token: Optional[Callable[[RequestOutput], None]] = None,
        n: int = 1,
    ) -> None:
        if request_id in self._request_to_seq_ids:
            raise ValueError(f"request_id '{request_id}' already exists")
        if n <= 0:
            raise ValueError("n must be greater than 0")

        sequences: list[SequenceData] = []
        seq_ids: list[int] = []
        output_map: dict[int, list[int]] = {}
        for _ in range(n):
            seq_id = self._next_seq_id
            self._next_seq_id += 1
            seq_ids.append(seq_id)
            sequence = SequenceData(
                seq_id=seq_id,
                prompt_token_ids=list(prompt_token_ids),
                sampling_params=sampling_params,
            )
            sequences.append(sequence)
            self._sequences[seq_id] = sequence
            self._sequence_to_request_id[seq_id] = request_id
            output_map[seq_id] = []

        group = SequenceGroup(request_id=request_id, sequences=sequences)

        self._request_to_seq_ids[request_id] = seq_ids
        self._request_outputs[request_id] = output_map
        if on_token is not None:
            self._callbacks.setdefault(request_id, []).append(on_token)

        self.scheduler.add_request(group)

    def step(self) -> list[RequestOutput]:
        self._prepare_speculative_rounds()
        scheduler_output = self.scheduler.schedule()
        outputs = self._verify_speculative_rounds(
            scheduler_output.verify_seq_ids
        )
        if (
            not scheduler_output.prefill_seq_ids
            and not scheduler_output.decode_seq_ids
        ):
            if outputs:
                self._num_steps += 1
            return outputs

        batch = self.batch_builder.from_scheduler_output(
            scheduler_output,
            self._sequences,
            self.kv_cache,
        )
        if batch.total_tokens == 0:
            raise RuntimeError(
                "scheduler produced an empty batch; empty prompts are not supported"
            )

        if self._spec_session_driver is None and self._can_delegate_speculative(
            batch
        ):
            if self._can_drive_verify_rounds():
                return self._step_speculative_session(batch)
            return self._step_speculative(batch)

        logits = self._execute_and_commit(batch)
        # Compatibility limit: pre-Stage4a session doubles do not accept a
        # request-scoped generator and retain the original singleton,
        # whole-request Step-5 behavior. Canonical SpecSession implementations
        # always take the persistent path below.
        if (
            self._spec_session_driver is not None
            and not self._spec_session_driver.supports_request_generator
            and self._can_delegate_speculative(batch)
        ):
            outputs.extend(self._step_speculative_session(batch))
            return outputs

        speculative_indices = [
            index
            for index, seq_id in enumerate(batch.seq_ids)
            if self._can_start_persistent_speculative(
                seq_id, batch.is_prefill[index]
            )
        ]
        for index in speculative_indices:
            outputs.extend(
                self._begin_persistent_speculative(batch.seq_ids[index])
            )

        speculative_index_set = set(speculative_indices)
        fallback_indices = [
            index
            for index in range(len(batch.seq_ids))
            if index not in speculative_index_set
        ]
        if fallback_indices:
            fallback_batch = (
                batch
                if len(fallback_indices) == len(batch.seq_ids)
                else _slice_batch(batch, fallback_indices)
            )
            outputs.extend(self._step_standard(fallback_batch))

        if speculative_indices and not fallback_indices:
            self._num_steps += 1
        return outputs

    def _step_standard(self, batch: BatchMetadata) -> list[RequestOutput]:
        """Execute the ordinary serving path for a scheduler-selected subset."""

        logits = self._execute_batch(batch)
        last_token_logits = self._extract_last_token_logits(logits, batch)
        sampler_output = self.sampler.sample(
            last_token_logits,
            batch.sampling_params,
        )
        next_token_ids = sampler_output.token_ids

        outputs: list[RequestOutput] = []
        completed_seq_ids: list[int] = []
        new_decode_seq_ids: list[int] = []
        touched_request_ids: set[str] = set()

        for index, seq_id in enumerate(batch.seq_ids):
            sequence = self._sequences[seq_id]
            request_id = self._sequence_to_request_id[seq_id]
            touched_request_ids.add(request_id)
            token_id = int(next_token_ids[index].item())

            sequence.append_output_token(token_id)
            self._request_outputs[request_id][seq_id].append(token_id)
            self._total_generated_tokens += 1

            finish_reason = self._get_finish_reason(sequence, token_id)
            finished = finish_reason is not None
            if finished:
                completed_seq_ids.append(seq_id)
            else:
                new_decode_seq_ids.append(seq_id)

            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    seq_id=seq_id,
                    token_id=token_id,
                    token_text=self._decode_token(token_id),
                    finished=finished,
                    finish_reason=finish_reason,
                    token_logprob=(
                        sampler_output.token_logprobs[index]
                        if sampler_output.token_logprobs is not None
                        else None
                    ),
                    top_logprobs=(
                        sampler_output.top_logprobs[index]
                        if sampler_output.top_logprobs is not None
                        else None
                    ),
                    usage=(self._build_usage(sequence) if finished else None),
                )
            )

        self.scheduler.update_after_step(
            completed_seq_ids=completed_seq_ids,
            new_decode_seq_ids=new_decode_seq_ids,
        )
        self._num_steps += 1

        finished_request_ids = {
            request_id
            for request_id in touched_request_ids
            if self._is_request_finished(request_id)
        }
        self._completed_request_ids.update(finished_request_ids)

        for output in outputs:
            for callback in self._callbacks.get(output.request_id, []):
                callback(output)

        for request_id in finished_request_ids:
            if _eviction_sync is not None:
                _eviction_sync.on_request_finished(request_id)
            _ = self._callbacks.pop(request_id, None)

        return outputs

    @property
    def speculative_sessions(self) -> dict[int, ServingSpecSession]:
        """Live Stage 4a records, keyed by serving sequence id."""
        driver = self._spec_session_driver
        return {} if driver is None else driver.sessions

    def _can_start_persistent_speculative(
        self, seq_id: int, is_prefill: bool
    ) -> bool:
        """Check semantics that the temporary DynamicCache path can preserve."""
        if self._spec_session_driver is None or not is_prefill:
            return False
        sequence = self._sequences[seq_id]
        params = sequence.sampling_params
        return (
            sequence.status is SequenceStatus.PREFILL
            and not sequence.output_token_ids
            and params.max_tokens > 0
            and not params.stop
            and params.temperature >= 0
            and params.top_p > 0
            and params.top_p <= 1
            and params.repetition_penalty == 1.0
            and params.logprobs <= 0
            and not self._has_unsupported_speculative_metadata(params)
        )

    @staticmethod
    def _has_unsupported_speculative_metadata(params: SamplingParams) -> bool:
        for name in (
            "grammar",
            "guided_decoding",
            "response_format",
            "logit_bias",
            "logits_processors",
        ):
            if getattr(params, name, None):
                return True
        for name in ("presence_penalty", "frequency_penalty", "min_p"):
            if float(getattr(params, name, 0.0) or 0.0) != 0.0:
                return True
        return False

    def _begin_persistent_speculative(self, seq_id: int) -> list[RequestOutput]:
        driver = self._spec_session_driver
        if driver is None:
            raise RuntimeError(
                "persistent speculative driver is not configured"
            )
        sequence = self._sequences[seq_id]
        request_id = self._sequence_to_request_id[seq_id]
        params = sequence.sampling_params
        stop_token_ids = (
            [self.eos_token_id] if self.eos_token_id is not None else []
        )
        generator: torch.Generator | None = None
        if params.temperature > 0:
            generator_device = (
                self.device
                if self.device.type == "cuda"
                else torch.device("cpu")
            )
            generator = torch.Generator(device=generator_device)
            base_seed = self._get_int_config("speculative_seed", 0)
            generator.manual_seed(base_seed + seq_id)
        record = driver.begin(
            request_id=request_id,
            seq_id=seq_id,
            prompt_token_ids=sequence.prompt_token_ids,
            max_new_tokens=params.max_tokens,
            temperature=params.temperature,
            top_k=max(0, params.top_k),
            top_p=params.top_p,
            stop_token_ids=stop_token_ids,
            callbacks=tuple(self._callbacks.get(request_id, ())),
            generator=generator,
        )
        sequence.set_status(SequenceStatus.DRAFT)
        committed = driver.commit(record)
        return self._publish_speculative_commit(record, committed)

    def _prepare_speculative_rounds(self) -> None:
        driver = self._spec_session_driver
        if driver is None:
            return
        for record in tuple(driver.sessions.values()):
            sequence = self._sequences.get(record.seq_id)
            if (
                record.cancelled
                or record.released
                or sequence is None
                or sequence.status is not SequenceStatus.DRAFT
                or record.finished
            ):
                continue
            try:
                draft = driver.draft(record)
            except BaseException as exc:
                self._fail_speculative_request(record, "draft", exc)
                raise
            if record.cancelled or record.released:
                continue
            self.scheduler.set_verify_demand(
                record.seq_id,
                tokens=int(getattr(draft, "tokens")),
                expert_bytes=int(getattr(draft, "expert_bytes")),
                in_flight=False,
            )

    def _verify_speculative_rounds(
        self, admitted_seq_ids: list[int]
    ) -> list[RequestOutput]:
        driver = self._spec_session_driver
        if driver is None:
            return []
        outputs: list[RequestOutput] = []
        for seq_id in admitted_seq_ids:
            record = driver.sessions.get(seq_id)
            sequence = self._sequences.get(seq_id)
            if (
                record is None
                or sequence is None
                or record.pending_draft is None
            ):
                continue
            sequence.set_status(SequenceStatus.VERIFY)
            try:
                _ = driver.verify(record)
            except BaseException as exc:
                self._fail_speculative_request(record, "verify", exc)
                raise
            self.scheduler.clear_verify_demand(seq_id)
            if (
                record.cancelled
                or record.released
                or seq_id not in self._sequences
            ):
                continue
            committed = driver.commit(record)
            outputs.extend(self._publish_speculative_commit(record, committed))
        return outputs

    def _fail_speculative_request(
        self,
        failed_record: ServingSpecSession,
        phase: str,
        primary: BaseException,
    ) -> None:
        """Fail one request atomically while preserving its backend exception."""
        request_id = failed_record.request_id
        failure = {
            "phase": phase,
            "failure_type": type(primary).__name__,
            "code": f"speculative_{phase}_failed",
        }
        self._request_failures[request_id] = failure
        seq_ids = list(self._request_to_seq_ids.get(request_id, ()))
        cleanup_errors: list[BaseException] = []
        driver = self._spec_session_driver
        if driver is not None:
            records = [
                record
                for record in tuple(driver.sessions.values())
                if record.request_id == request_id
            ]
            for record in records:
                self.scheduler.clear_verify_demand(record.seq_id)
                try:
                    driver.fail(record, failure)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        try:
            self.scheduler.abort_request(request_id)
        except BaseException as exc:
            cleanup_errors.append(exc)

        self._callbacks.pop(request_id, None)
        self._request_outputs.pop(request_id, None)
        self._request_to_seq_ids.pop(request_id, None)
        for seq_id in seq_ids:
            self.scheduler.clear_verify_demand(seq_id)
            self._sequence_to_request_id.pop(seq_id, None)
            self._sequences.pop(seq_id, None)
        if _eviction_sync is not None:
            try:
                _eviction_sync.on_request_aborted(request_id)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            cleanup_metadata = tuple(cleanup_errors)
            try:
                setattr(primary, "session_cleanup_errors", cleanup_metadata)
            except BaseException as reporting_error:
                _debug_cleanup_reporting_failure(
                    action="metadata",
                    request_id=request_id,
                    phase=phase,
                    primary=primary,
                    reporting_error=reporting_error,
                )
            try:
                add_note = getattr(primary, "add_note", None)
            except BaseException as reporting_error:
                _debug_cleanup_reporting_failure(
                    action="note-lookup",
                    request_id=request_id,
                    phase=phase,
                    primary=primary,
                    reporting_error=reporting_error,
                )
                add_note = None
            if callable(add_note):
                for cleanup_error in cleanup_metadata:
                    try:
                        add_note(
                            "speculative request cleanup failed: "
                            f"{cleanup_error}"
                        )
                    except BaseException as reporting_error:
                        _debug_cleanup_reporting_failure(
                            action="note",
                            request_id=request_id,
                            phase=phase,
                            primary=primary,
                            reporting_error=reporting_error,
                        )

    def _publish_speculative_commit(
        self,
        record: ServingSpecSession,
        committed: tuple[int, ...],
    ) -> list[RequestOutput]:
        if record.cancelled or record.released:
            return []
        sequence = self._sequences.get(record.seq_id)
        if sequence is None:
            return []

        outputs: list[RequestOutput] = []
        for token_id in committed:
            sequence.append_output_token(token_id)
            self._request_outputs[record.request_id][record.seq_id].append(
                token_id
            )
            self._total_generated_tokens += 1
            finish_reason = self._get_finish_reason(sequence, token_id)
            finished = finish_reason is not None
            output = RequestOutput(
                request_id=record.request_id,
                seq_id=record.seq_id,
                token_id=token_id,
                token_text=self._decode_token(token_id),
                finished=finished,
                finish_reason=finish_reason,
                usage=(self._build_usage(sequence) if finished else None),
            )
            outputs.append(output)
            for callback in record.callbacks:
                callback(output)
            if finished:
                break

        finished = record.finished or self._output_finished(outputs)
        committed_count = len(outputs)
        if finished:
            if sequence.status in (SequenceStatus.DRAFT, SequenceStatus.VERIFY):
                sequence.set_status(SequenceStatus.FINISHED)
            self.scheduler.update_after_step(
                completed_seq_ids=[record.seq_id],
                new_decode_seq_ids=[],
                committed_counts={record.seq_id: committed_count},
            )
            driver = self._spec_session_driver
            if driver is not None:
                driver.release(record)
            if self._is_request_finished(record.request_id):
                self._completed_request_ids.add(record.request_id)
                if _eviction_sync is not None:
                    _eviction_sync.on_request_finished(record.request_id)
                self._callbacks.pop(record.request_id, None)
        else:
            if sequence.status is SequenceStatus.VERIFY:
                sequence.set_status(SequenceStatus.DRAFT)
            self.scheduler.update_after_step(
                completed_seq_ids=[],
                new_decode_seq_ids=[],
                committed_counts={record.seq_id: committed_count},
            )
        return outputs

    def _can_delegate_speculative(self, batch: BatchMetadata) -> bool:
        """Whether this fresh singleton request can use the proven sync loop.

        DFlash owns a separate ``DynamicCache`` here. The paged serving cache is
        used only for admission accounting and freed when the delegated request
        completes. Mixed batches, resumed decode rows, sampling, penalties, and
        logprob requests stay on the existing serving path unchanged.
        """
        if self.speculative_draft is None or len(batch.seq_ids) != 1:
            return False
        if batch.is_prefill != [True]:
            return False
        if batch.kv_seq_lengths != batch.query_lengths:
            return False

        sequence = self._sequences[batch.seq_ids[0]]
        if getattr(sequence, "has_prefix_lease", False):
            return False
        params = sequence.sampling_params
        return (
            not sequence.output_token_ids
            and not params.stop
            and params.max_tokens <= self.scheduler.max_tokens_per_step
            and params.temperature == 0
            and params.top_k <= 0
            and params.top_p >= 1.0
            and params.repetition_penalty == 1.0
            and params.logprobs <= 0
        )

    def _step_speculative(self, batch: BatchMetadata) -> list[RequestOutput]:
        """Complete one eligible request through DFlash's own DynamicCache.

        ``DFlashSpeculator.generate`` is the already GPU-proven greedy loop. A
        single serving ``step`` may therefore emit several accepted tokens;
        each is still recorded and streamed as an individual ``RequestOutput``.
        """
        speculator = self.speculative_draft
        if speculator is None:
            raise RuntimeError("speculative generator is not configured")

        seq_id = batch.seq_ids[0]
        sequence = self._sequences[seq_id]
        request_id = self._sequence_to_request_id[seq_id]
        prompt = torch.tensor([sequence.prompt_token_ids], dtype=torch.long)
        stop_token_ids = (
            [self.eos_token_id] if self.eos_token_id is not None else None
        )

        owner = cast(object | None, getattr(speculator, "moe", None))
        if owner is not None:
            setattr(owner, "_cached_past_key_values", None)
        try:
            generated = speculator.generate(
                prompt,
                max_new_tokens=sequence.sampling_params.max_tokens,
                temperature=0.0,
                stop_token_ids=stop_token_ids,
                top_k=sequence.sampling_params.top_k,
                top_p=sequence.sampling_params.top_p,
            )
        finally:
            if owner is not None:
                setattr(owner, "_cached_past_key_values", None)

        if generated.ndim != 2 or generated.shape[0] != 1:
            raise RuntimeError(
                "speculative generator must return token ids with shape [1, seq]"
            )
        prompt_len = sequence.prompt_length
        generated_ids = cast(
            list[int], generated[0, prompt_len:].to(device="cpu").tolist()
        )

        outputs: list[RequestOutput] = []
        for token_id in generated_ids:
            sequence.append_output_token(token_id)
            self._request_outputs[request_id][seq_id].append(token_id)
            self._total_generated_tokens += 1

            finish_reason = self._get_finish_reason(sequence, token_id)
            finished = finish_reason is not None
            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    seq_id=seq_id,
                    token_id=token_id,
                    token_text=self._decode_token(token_id),
                    finished=finished,
                    finish_reason=finish_reason,
                    usage=(self._build_usage(sequence) if finished else None),
                )
            )
            if finished:
                break

        self.scheduler.update_after_step(
            completed_seq_ids=[seq_id],
            new_decode_seq_ids=[],
            committed_counts={seq_id: len(outputs)},
        )
        self._num_steps += 1
        self._completed_request_ids.add(request_id)

        for output in outputs:
            for callback in self._callbacks.get(request_id, []):
                callback(output)
        if _eviction_sync is not None:
            _eviction_sync.on_request_finished(request_id)
        _ = self._callbacks.pop(request_id, None)
        return outputs

    def _can_drive_verify_rounds(self) -> bool:
        """Opt-in gate for the Step-5 per-round DRAFT->VERIFY->DRAFT driver.

        Off by default: only when the operator configured verify budgets AND
        the speculator exposes the single-round ``SpecSession`` seam. Otherwise
        the delegated request keeps the proven whole-request ``generate()`` path
        unchanged, so default serving stays byte-for-byte identical.
        """
        speculator = self.speculative_draft
        return (
            self._verify_scheduling_enabled
            and speculator is not None
            and callable(getattr(speculator, "begin_session", None))
            and callable(getattr(speculator, "draft_round", None))
            and callable(getattr(speculator, "verify_round", None))
        )

    def _step_speculative_session(
        self, batch: BatchMetadata
    ) -> list[RequestOutput]:
        """Drive one eligible request through the scheduled single-round seam.

        Per round the engine drafts a block, registers that pending verify's
        EXACT token/byte demand with the 2-D verify scheduler
        (``set_verify_demand`` with the drafter-projected ``expert_nbytes`` sum,
        never a fabricated estimate), and runs the verify ONLY once the
        scheduler admits it (``verify_seq_ids``). The serving sequence advances
        PREFILL -> DRAFT -> (VERIFY -> DRAFT)* -> FINISHED; emitted tokens stream
        exactly as on the whole-request path. This runs only under the opt-in
        gate; ordinary serving and non-session speculators are untouched.
        """
        speculator = self.speculative_draft
        if speculator is None:
            raise RuntimeError("speculative generator is not configured")

        seq_id = batch.seq_ids[0]
        sequence = self._sequences[seq_id]
        request_id = self._sequence_to_request_id[seq_id]
        prompt = torch.tensor([sequence.prompt_token_ids], dtype=torch.long)
        stop_token_ids = (
            [self.eos_token_id] if self.eos_token_id is not None else None
        )
        params = sequence.sampling_params

        owner = cast(object | None, getattr(speculator, "moe", None))
        if owner is not None:
            setattr(owner, "_cached_past_key_values", None)

        outputs: list[RequestOutput] = []
        try:
            session = speculator.begin_session(
                prompt,
                max_new_tokens=params.max_tokens,
                temperature=0.0,
                stop_token_ids=stop_token_ids,
                top_k=params.top_k,
                top_p=params.top_p,
                collect_route_union=True,
            )
            sequence.set_status(SequenceStatus.DRAFT)

            streamed = 0
            outputs.extend(
                self._emit_session_tokens(
                    sequence, request_id, seq_id, session, streamed
                )
            )
            streamed = len(session.emitted)

            while not session.finished and not self._output_finished(outputs):
                draft = speculator.draft_round(session)
                self._admit_verify_round(seq_id, draft)
                sequence.set_status(SequenceStatus.VERIFY)
                speculator.verify_round(session)
                self.scheduler.clear_verify_demand(seq_id)
                if not session.finished:
                    sequence.set_status(SequenceStatus.DRAFT)

                outputs.extend(
                    self._emit_session_tokens(
                        sequence, request_id, seq_id, session, streamed
                    )
                )
                streamed = len(session.emitted)
                if self._output_finished(outputs):
                    break
        finally:
            if owner is not None:
                setattr(owner, "_cached_past_key_values", None)
            self.scheduler.clear_verify_demand(seq_id)

        if sequence.status in (
            SequenceStatus.DRAFT,
            SequenceStatus.VERIFY,
        ):
            sequence.set_status(SequenceStatus.FINISHED)

        self.scheduler.update_after_step(
            completed_seq_ids=[seq_id],
            new_decode_seq_ids=[],
            committed_counts={seq_id: len(outputs)},
        )
        self._num_steps += 1
        self._completed_request_ids.add(request_id)

        for output in outputs:
            for callback in self._callbacks.get(request_id, []):
                callback(output)
        if _eviction_sync is not None:
            _eviction_sync.on_request_finished(request_id)
        _ = self._callbacks.pop(request_id, None)
        return outputs

    def _admit_verify_round(self, seq_id: int, draft: object) -> None:
        """Register a pending verify's exact demand and gate on admission.

        Seats the demand as a new (non-in-flight) DRAFT round and lets
        ``Scheduler`` decide when the verify runs; unadmitted rounds carry their
        2-D deficit and are retried as the budget accrues. Raises when a
        correctly-shaped demand is never admitted (a budget misconfiguration:
        e.g. a zero budget in a dimension the demand needs).
        """
        tokens = int(getattr(draft, "tokens"))
        expert_bytes = int(getattr(draft, "expert_bytes"))
        self.scheduler.set_verify_demand(
            seq_id,
            tokens=tokens,
            expert_bytes=expert_bytes,
            in_flight=False,
        )
        for _ in range(_VERIFY_ADMISSION_MAX_RETRIES):
            output = self.scheduler.schedule()
            if seq_id in output.verify_seq_ids:
                return
        raise RuntimeError(
            "verify round for seq_id "
            f"{seq_id} was never admitted after "
            f"{_VERIFY_ADMISSION_MAX_RETRIES} scheduling passes; raise "
            "verify_token_budget / verify_expert_byte_budget to cover a single "
            f"verify (tokens={tokens}, expert_bytes={expert_bytes})"
        )

    def _emit_session_tokens(
        self,
        sequence: SequenceData,
        request_id: str,
        seq_id: int,
        session: object,
        streamed: int,
    ) -> list[RequestOutput]:
        """Stream the session's newly emitted tokens as ``RequestOutput`` rows."""
        emitted = cast(list[int], getattr(session, "emitted"))
        outputs: list[RequestOutput] = []
        for token_id in emitted[streamed:]:
            if self._output_finished(outputs):
                break
            sequence.append_output_token(token_id)
            self._request_outputs[request_id][seq_id].append(token_id)
            self._total_generated_tokens += 1

            finish_reason = self._get_finish_reason(sequence, token_id)
            finished = finish_reason is not None
            outputs.append(
                RequestOutput(
                    request_id=request_id,
                    seq_id=seq_id,
                    token_id=token_id,
                    token_text=self._decode_token(token_id),
                    finished=finished,
                    finish_reason=finish_reason,
                    usage=(self._build_usage(sequence) if finished else None),
                )
            )
        return outputs

    @staticmethod
    def _output_finished(outputs: list[RequestOutput]) -> bool:
        return bool(outputs) and outputs[-1].finished

    def run_until_done(self) -> dict[str, list[int] | list[list[int]]]:
        while self.has_pending_requests():
            outputs = self.step()
            if outputs:
                continue

            pending_request_ids = self._pending_request_ids()
            raise RuntimeError(
                f"engine made no progress with pending requests: {pending_request_ids}"
            )

        return {
            request_id: self._format_request_outputs(request_id)
            for request_id in self._request_outputs
        }

    def get_request_n_outputs(self, request_id: str) -> list[list[int]]:
        if request_id not in self._request_outputs:
            raise KeyError(f"unknown request_id '{request_id}'")

        return [
            list(self._request_outputs[request_id][seq_id])
            for seq_id in self._request_to_seq_ids.get(request_id, [])
        ]

    def abort_request(self, request_id: str) -> None:
        seq_ids = list(self._request_to_seq_ids.get(request_id, []))
        was_pending = any(
            self._sequences[seq_id].status
            in {
                SequenceStatus.WAITING,
                SequenceStatus.PREFILL,
                SequenceStatus.DECODE,
                SequenceStatus.DRAFT,
                SequenceStatus.VERIFY,
                SequenceStatus.SWAPPED,
            }
            for seq_id in seq_ids
            if seq_id in self._sequences
        )

        driver = self._spec_session_driver
        if driver is not None:
            for seq_id in seq_ids:
                driver.cancel(seq_id)
                self.scheduler.clear_verify_demand(seq_id)
        self.scheduler.abort_request(request_id)
        _ = self._callbacks.pop(request_id, None)

        if not was_pending:
            return

        if _eviction_sync is not None:
            _eviction_sync.on_request_aborted(request_id)

        self._cancelled_request_ids.add(request_id)
        _ = self._request_outputs.pop(request_id, None)
        _ = self._request_to_seq_ids.pop(request_id, None)

        for seq_id in seq_ids:
            _ = self._sequence_to_request_id.pop(seq_id, None)
            _ = self._sequences.pop(seq_id, None)

    def has_pending_requests(self) -> bool:
        return bool(self._pending_request_ids())

    def _resolve_prefix_capability(self) -> object | None:
        getter = getattr(self.model_runner, "get_prefix_reuse_capability", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:
            return None

    def _build_cache_namespace(self) -> CacheNamespace:
        return CacheNamespace(
            model_id=str(self.config.get("model_id", "")),
            model_revision=str(self.config.get("model_revision", "")),
            tokenizer_id=str(self.config.get("tokenizer_id", "")),
            tokenizer_revision=str(self.config.get("tokenizer_revision", "")),
            tokenizer_config_digest=str(
                self.config.get("tokenizer_config_digest", "")
            ),
            adapter_id=None,
            adapter_revision=None,
            dtype=str(self.dtype).removeprefix("torch."),
            block_size=self.kv_cache.block_size,
            num_layers=self.kv_cache.num_layers,
            num_kv_heads=self.kv_cache.num_heads,
            head_dim=self.kv_cache.head_dim,
            attention_backend=str(
                self.config.get("attention_backend", "flashinfer-paged")
            ),
            attention_layout=str(self.config.get("attention_layout", "NHD")),
            position_config_digest=str(
                self.config.get("position_config_digest", "")
            ),
            runtime_epoch=f"epoch-{self._runtime_epoch}",
        )

    def _maybe_bind_prefix_cache(
        self, capability: object | None
    ) -> object | None:
        if not self._prefix_cache_enabled:
            self._prefix_cache_disabled_reason = "prefix-caching-disabled"
            return None
        if capability is None or not getattr(capability, "supported", False):
            self._prefix_cache_disabled_reason = (
                getattr(capability, "reason", None)
                or "prefix-aware-prefill-unavailable"
            )
            return None
        backend = getattr(capability, "backend", None)
        store = getattr(capability, "block_store", None)
        if backend is None or store is None:
            self._prefix_cache_disabled_reason = (
                "prefix-aware-prefill-unavailable"
            )
            return None
        try:
            self.kv_cache.set_block_store(store, owner=backend)
        except (RuntimeError, ValueError) as exc:
            self._prefix_cache_disabled_reason = (
                f"kv-store-binding-mismatch: {exc}"
            )
            return None
        self.cache_namespace = self._build_cache_namespace()
        self.prefix_cache = PrefixCache(
            block_size=self.kv_cache.block_size,
            max_entries=self._get_int_config("prefix_cache_max_entries", 1000),
            on_retain=self.kv_cache.block_allocator.retain,
            on_release=self.kv_cache.block_allocator.release,
        )
        self._prefix_cache_disabled_reason = ""
        return self.prefix_cache

    def invalidate_prefix_cache(self, reason: str) -> None:
        _ = reason
        self._runtime_epoch += 1
        self._prefix_cache_invalidations += 1
        if self.prefix_cache is not None:
            self.cache_namespace = self._build_cache_namespace()
            self.scheduler.cache_namespace = self.cache_namespace

    def _prefix_cache_stats(self) -> dict[str, object]:
        prefix_cache = getattr(self, "prefix_cache", None)
        active = prefix_cache is not None
        entries = prefix_cache.num_entries if active else 0
        open_leases = prefix_cache.open_leases if active else 0
        hits_total = prefix_cache.hits_total if active else 0
        matched_tokens_total = (
            prefix_cache.matched_tokens_total if active else 0
        )
        return {
            "prefix_cache_enabled": getattr(
                self, "_prefix_cache_enabled", False
            ),
            "prefix_cache_active": active,
            "prefix_cache_disabled_reason": (
                ""
                if active
                else getattr(self, "_prefix_cache_disabled_reason", "")
            ),
            "prefix_cache_entries": entries,
            "prefix_cache_open_leases": open_leases,
            "prefix_cache_hits_total": hits_total,
            "prefix_cache_matched_tokens_total": matched_tokens_total,
            "prefix_cache_invalidations_total": (
                getattr(self, "_prefix_cache_invalidations", 0)
            ),
        }

    def invalidate_cuda_graphs(self, reason: str) -> None:
        self.cuda_graph_runner.invalidate(reason)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self.cuda_graph_runner.close()
        self._shutdown = True

    def get_stats(self) -> dict[str, object]:
        status_counts = {status.value: 0 for status in SequenceStatus}
        for sequence in self._sequences.values():
            status_counts[sequence.status.value] += 1

        graph_runner = getattr(self, "cuda_graph_runner", None)
        cuda_graph_stats = graph_runner.stats() if graph_runner else {}
        storage = getattr(self.kv_cache, "storage", None)
        scratch_kv_bytes = 0
        if storage is not None:
            scratch_kv_bytes = (
                storage.num_graph_scratch_blocks
                * storage.spec.block_size
                * storage.spec.num_layers
                * 2
                * storage.spec.num_kv_heads
                * storage.spec.head_dim
                * torch.empty((), dtype=storage.spec.dtype).element_size()
            )
        graph_pool_bytes = int(cuda_graph_stats.get("graph_pool_bytes", 0))
        set_graph_usage = getattr(
            self.memory_manager, "set_cuda_graph_usage", None
        )
        if callable(set_graph_usage):
            set_graph_usage(
                graph_pool_bytes=graph_pool_bytes,
                scratch_kv_bytes=scratch_kv_bytes,
            )

        capability_fn = getattr(
            getattr(self, "model_runner", None),
            "decode_graph_capability",
            None,
        )
        capability = capability_fn() if callable(capability_fn) else None
        registry = getattr(self, "paged_attention_registry", None)
        if capability is not None and registry is not None:
            capability_reason = (
                capability.reason
                if capability.reason in DECODE_GRAPH_REASONS
                else "missing_capability"
            )
            bindings = tuple(registry.bindings)
            proved_write_layers = sum(
                1 for binding in bindings if binding.has_write_proof
            )
            cuda_graph_stats.update(
                {
                    "scratch_kv_bytes": scratch_kv_bytes,
                    "kv_storage_owner_id": (
                        storage.owner_id if storage is not None else None
                    ),
                    "capability_safe": (
                        capability.safe and capability_reason == "eligible"
                    ),
                    "capability_reason": capability_reason,
                    "registered_paged_layers": len(bindings),
                    "proved_write_layers": proved_write_layers,
                }
            )

        stats: dict[str, object] = {
            "pending_requests": len(self._pending_request_ids()),
            "completed_requests": len(self._completed_request_ids),
            "cancelled_requests": len(self._cancelled_request_ids),
            "failed_requests": len(self._request_failures),
            "num_steps": self._num_steps,
            "total_generated_tokens": self._total_generated_tokens,
            "kv_cache_num_blocks": self.kv_cache.num_blocks,
            "kv_cache_free_blocks": self.kv_cache.block_allocator.num_free_blocks,
            "sequence_status_counts": status_counts,
            "speculative_execution_context": (
                self._spec_session_driver.execution_context_mode
                if self._spec_session_driver is not None
                else None
            ),
            "speculative_sessions": [
                record.diagnostics()
                for record in self.speculative_sessions.values()
            ],
            "paged_mla_admission": (
                self._spec_session_driver.admission_stats
                if self._spec_session_driver is not None
                else None
            ),
            "memory": self.memory_manager.report(),
            "cuda_graph": cuda_graph_stats,
        }
        stats.update(self._prefix_cache_stats())
        return stats

    def get_request_failure(self, request_id: str) -> dict[str, str]:
        if request_id not in self._request_failures:
            raise KeyError(f"request_id '{request_id}' has no recorded failure")
        return dict(self._request_failures[request_id])

    def get_config(self) -> dict[str, object]:
        config: dict[str, object] = {}
        for key, value in self.config.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                config[key] = value
            else:
                config[key] = str(value)
        return config

    def update_config(self, updates: dict[str, object]) -> dict[str, object]:
        applied: dict[str, object] = {}
        for key in ("max_batch_size", "max_tokens_per_step"):
            if key not in updates:
                continue
            value = updates[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer value")
            self.config[key] = value
            setattr(self.scheduler, key, value)
            applied[key] = value
        return applied

    def _resolve_num_blocks(
        self,
        *,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> int:
        explicit_num_blocks = self.config.get("num_kv_blocks")
        if explicit_num_blocks is not None:
            if isinstance(explicit_num_blocks, bool) or not isinstance(
                explicit_num_blocks, int
            ):
                raise ValueError(
                    "num_kv_blocks must be an integer when provided"
                )
            num_blocks = explicit_num_blocks
        else:
            num_blocks = self.memory_manager.get_max_kv_blocks(
                block_size=block_size,
                num_layers=num_layers,
                num_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=self.dtype,
            )

        if num_blocks > 0:
            return num_blocks
        return self._fallback_num_blocks(block_size)

    def _fallback_num_blocks(self, block_size: int) -> int:
        max_tokens_per_step = self._get_int_config("max_tokens_per_step", 1)
        return max(1, ceil(max_tokens_per_step / max(1, block_size)))

    def _execute_batch(self, batch: BatchMetadata) -> torch.Tensor:
        has_prefill = any(batch.is_prefill)
        has_decode = any(not p for p in batch.is_prefill)
        uses_paged = bool(self.paged_attention_registry.bindings)

        if not (has_prefill and has_decode):
            if has_decode and not has_prefill:
                return self._execute_decode_batch(batch)
            return self.model_runner.execute(batch)

        if not uses_paged:
            return self.model_runner.execute(batch)

        split = split_prefill_decode_batch(batch)
        prefill_logits = None
        decode_logits = None
        if split.prefill_batch is not None:
            prefill_logits = self.model_runner.execute(split.prefill_batch)
        if split.decode_batch is not None:
            decode_logits = self._execute_decode_batch(split.decode_batch)
        return split.recombine_outputs(prefill_logits, decode_logits)

    def _execute_and_commit(self, batch: BatchMetadata) -> torch.Tensor:
        logits = self._execute_batch(batch)
        query_lengths = batch.query_lengths
        is_prefill = batch.is_prefill
        for index, seq_id in enumerate(batch.seq_ids):
            if not is_prefill[index]:
                continue
            sequence = self._sequences.get(seq_id)
            if sequence is None:
                continue
            sequence.committed_kv_tokens += query_lengths[index]
            self._publish_committed_prefix(seq_id, sequence)
        return logits

    def _publish_committed_prefix(
        self, seq_id: int, sequence: SequenceData
    ) -> None:
        if self.prefix_cache is None or self.cache_namespace is None:
            return
        committed = sequence.committed_kv_tokens
        if committed <= 0 or committed > sequence.prompt_length:
            committed = min(committed, sequence.prompt_length)
        if committed <= 0:
            return
        block_size = self.kv_cache.block_size
        full_blocks = committed // block_size
        if full_blocks <= 0:
            return
        block_table = self.kv_cache.get_block_table(seq_id)
        if len(block_table) < full_blocks:
            return
        self.prefix_cache.insert(
            self.cache_namespace,
            sequence.prompt_token_ids,
            block_table[:full_blocks],
            committed_tokens=full_blocks * block_size,
        )

    def _execute_decode_batch(self, batch: BatchMetadata) -> torch.Tensor:
        graph_logits = self.cuda_graph_runner.try_execute(batch)
        if graph_logits is not None:
            return graph_logits
        return self.model_runner.execute(batch)

    @staticmethod
    def _extract_last_token_logits(
        logits: torch.Tensor,
        batch: BatchMetadata,
    ) -> torch.Tensor:
        last_token_indices: list[int] = []
        query_offsets = batch.query_offsets
        for index, seq_length in enumerate(batch.query_lengths):
            if seq_length <= 0:
                raise RuntimeError(
                    "scheduled sequence has no tokens; empty prompts are not supported"
                )
            last_token_indices.append(query_offsets[index + 1] - 1)

        return logits[last_token_indices]

    def _get_finish_reason(
        self,
        sequence: SequenceData,
        token_id: int,
    ) -> Optional[str]:
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            return "stop"
        if (
            len(sequence.output_token_ids)
            >= sequence.sampling_params.max_tokens
        ):
            return "length"

        if not sequence.sampling_params.stop:
            return None

        decoded_text = self._decode_tokens(sequence.output_token_ids)
        if decoded_text is None:
            return None

        if any(
            decoded_text.endswith(stop_text)
            for stop_text in sequence.sampling_params.stop
        ):
            return "stop"
        return None

    @staticmethod
    def _build_usage(sequence: SequenceData) -> dict[str, int]:
        prompt_tokens = sequence.prompt_length
        completion_tokens = len(sequence.output_token_ids)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _pending_request_ids(self) -> list[str]:
        pending_request_ids: list[str] = []
        active_statuses = {
            SequenceStatus.WAITING,
            SequenceStatus.PREFILL,
            SequenceStatus.DECODE,
            SequenceStatus.DRAFT,
            SequenceStatus.VERIFY,
            SequenceStatus.SWAPPED,
        }

        for request_id, seq_ids in self._request_to_seq_ids.items():
            if any(
                self._sequences[seq_id].status in active_statuses
                for seq_id in seq_ids
                if seq_id in self._sequences
            ):
                pending_request_ids.append(request_id)

        return pending_request_ids

    def _is_request_finished(self, request_id: str) -> bool:
        seq_ids = self._request_to_seq_ids.get(request_id, [])
        if not seq_ids:
            return False

        return all(
            self._sequences[seq_id].status is SequenceStatus.FINISHED
            for seq_id in seq_ids
            if seq_id in self._sequences
        )

    def _format_request_outputs(
        self, request_id: str
    ) -> list[int] | list[list[int]]:
        outputs = self.get_request_n_outputs(request_id)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def _decode_token(self, token_id: int) -> Optional[str]:
        return self._decode_tokens([token_id])

    def _decode_tokens(self, token_ids: list[int]) -> Optional[str]:
        if self.tokenizer is None:
            return None

        decode = getattr(self.tokenizer, "decode", None)
        if not callable(decode):
            return None

        try:
            decoded = decode(token_ids, skip_special_tokens=False)
        except TypeError:
            try:
                decoded = decode(token_ids)
            except Exception:
                return None
        except Exception:
            return None

        if isinstance(decoded, str):
            return decoded
        return None

    @staticmethod
    def _resolve_dtype(value: object) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value

        dtype_map = {
            "float16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(str(value).lower())
        if dtype is None:
            raise ValueError(f"unsupported dtype: {value}")
        return dtype

    @staticmethod
    def _resolve_eos_token_id(
        model: object,
        config: dict[str, object],
    ) -> Optional[int]:
        eos_token_id = config.get("eos_token_id")
        if isinstance(eos_token_id, int):
            return eos_token_id

        model_config = getattr(model, "config", None)
        model_eos_token_id = getattr(model_config, "eos_token_id", None)
        if isinstance(model_eos_token_id, int):
            return model_eos_token_id
        return None

    @staticmethod
    def _resolve_device(model: object) -> torch.device:
        model_device = getattr(model, "device", None)
        if (
            isinstance(model_device, torch.device)
            and model_device.type == "cuda"
        ):
            if not torch.cuda.is_available():
                return torch.device("cpu")
            return model_device

        parameters_fn = getattr(model, "parameters", None)
        if callable(parameters_fn):
            try:
                parameter_source = parameters_fn()
                if isinstance(parameter_source, Iterator):
                    first_param = next(parameter_source, None)
                elif isinstance(parameter_source, Iterable):
                    first_param = next(iter(parameter_source), None)
                else:
                    first_param = None
            except StopIteration:
                first_param = None
            except Exception:
                first_param = None

            if (
                isinstance(first_param, torch.Tensor)
                and first_param.device.type == "cuda"
            ):
                if not torch.cuda.is_available():
                    return torch.device("cpu")
                return first_param.device

        # model.device / first_param.device returned "cpu" — when CUDA is
        # available this means the model is managed by OffloadEngine which
        # moves tensors to GPU during forward (model_offload.py:1117-1130).
        if torch.cuda.is_available():
            last_gpu = torch.cuda.device_count() - 1
            return torch.device(f"cuda:{last_gpu}")
        return torch.device("cpu")

    @staticmethod
    def _resolve_attention_backend(engine: object) -> object | None:
        getter = getattr(engine, "get_attention_backend", None)
        if callable(getter):
            backend = getter()
            if backend is not None:
                return backend
        for attr_name in (
            "attention_backend",
            "_attention_backend",
            "_native_attention_backend",
        ):
            backend = getattr(engine, attr_name, None)
            if backend is not None:
                return backend
        return None

    @classmethod
    def _resolve_backend_storage(cls, engine: object) -> object | None:
        backend = cls._resolve_attention_backend(engine)
        if not isinstance(backend, PagedAttentionBackend):
            return None
        return backend.storage

    @staticmethod
    def _resolve_model_memory_bytes(model: object) -> int:
        get_memory_footprint = getattr(model, "get_memory_footprint", None)
        if callable(get_memory_footprint):
            try:
                memory_bytes = get_memory_footprint()
            except Exception:
                memory_bytes = None
            if isinstance(memory_bytes, (int, float)):
                return max(0, int(memory_bytes))

        total_bytes = 0
        for attribute_name in ("parameters", "buffers"):
            iterator_factory = getattr(model, attribute_name, None)
            if not callable(iterator_factory):
                continue

            try:
                tensor_source = iterator_factory()
            except Exception:
                continue

            if isinstance(tensor_source, Iterator):
                iterator: Iterable[object] = tensor_source
            elif isinstance(tensor_source, Iterable):
                iterator = tensor_source
            else:
                continue

            try:
                for tensor in iterator:
                    if isinstance(tensor, torch.Tensor):
                        total_bytes += tensor.numel() * tensor.element_size()
            except Exception:
                continue

        return max(0, int(total_bytes))

    def _get_float_config(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a float-compatible value")
        return float(value)

    def _get_optional_int_config(self, key: str) -> Optional[int]:
        if key not in self.config:
            return None
        value = self.config[key]
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer value")
        return value

    def _get_int_config(self, key: str, default: Optional[int] = None) -> int:
        if default is None:
            if key not in self.config:
                raise KeyError(key)
            value = self.config[key]
        else:
            value = self.config.get(key, default)

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer value")
        return value

    def _get_bool_config(self, key: str, default: bool = False) -> bool:
        value = self.config.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean value")
        return bool(value)

    def _get_bool_config(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean value")
        return value

    def _get_int_tuple_config(
        self, key: str, default: tuple[int, ...]
    ) -> tuple[int, ...]:
        if key not in self.config:
            return default
        value = self.config[key]
        if isinstance(value, (str, bytes)) or not isinstance(
            value, (list, tuple)
        ):
            raise ValueError(f"{key} must be a sequence of integers")
        result: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{key} must contain only integers")
            result.append(item)
        return tuple(result)


__all__ = [
    "ContinuousBatchingEngine",
    "RequestOutput",
    "set_eviction_sync",
]
