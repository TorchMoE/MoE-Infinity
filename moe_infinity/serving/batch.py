# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from moe_infinity.runtime.attention_types import PagedBatchLengths

from .kv_cache import PagedKVCache
from .sequence import SamplingParams, SequenceData


@dataclass
class SchedulerOutput:
    prefill_seq_ids: list[int] = field(default_factory=list)
    decode_seq_ids: list[int] = field(default_factory=list)
    preempted_seq_ids: list[int] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    draft_seq_ids: list[int] = field(default_factory=list)
    verify_seq_ids: list[int] = field(default_factory=list)
    num_verify_tokens: int = 0
    num_verify_expert_bytes: int = 0

    def __post_init__(self) -> None:
        self.prefill_seq_ids = list(self.prefill_seq_ids)
        self.decode_seq_ids = list(self.decode_seq_ids)
        self.preempted_seq_ids = list(self.preempted_seq_ids)
        self.draft_seq_ids = list(self.draft_seq_ids)
        self.verify_seq_ids = list(self.verify_seq_ids)


@dataclass
class BatchMetadata:
    seq_ids: list[int]
    input_token_ids: list[int]
    lengths: PagedBatchLengths
    is_prefill: list[bool]
    block_tables: list[list[int]]
    sampling_params: list[SamplingParams]

    def __init__(
        self,
        seq_ids: list[int],
        input_token_ids: list[int],
        lengths: PagedBatchLengths | None = None,
        is_prefill: list[bool] | None = None,
        block_tables: list[list[int]] | None = None,
        sampling_params: list[SamplingParams] | None = None,
        seq_lengths: list[int] | None = None,
        context_lengths: list[int] | None = None,
        token_offsets: list[int] | None = None,
    ) -> None:
        self.seq_ids = seq_ids
        self.input_token_ids = input_token_ids
        self.is_prefill = list(is_prefill or [])
        self.block_tables = list(block_tables or [])
        self.sampling_params = list(sampling_params or [])
        if lengths is None:
            if seq_lengths is None:
                raise ValueError(
                    "BatchMetadata requires lengths or seq_lengths"
                )
            contexts = list(
                context_lengths
                if context_lengths is not None
                else [0] * len(seq_lengths)
            )
            offsets = list(
                token_offsets
                if token_offsets is not None
                else [
                    sum(seq_lengths[:index])
                    for index in range(len(seq_lengths) + 1)
                ]
            )
            lengths = PagedBatchLengths(
                query_lengths=list(seq_lengths),
                query_offsets=offsets,
                context_lengths=contexts,
                kv_seq_lengths=[
                    int(context) + int(query)
                    for context, query in zip(contexts, seq_lengths)
                ],
            )
        self.lengths = lengths
        self.__post_init__()

    def __post_init__(self) -> None:
        self.lengths.validate()
        expected = len(self.seq_ids)
        query_lengths = [int(value) for value in self.lengths.query_lengths]
        query_offsets = [int(value) for value in self.lengths.query_offsets]
        for field_name, value in (
            ("is_prefill", self.is_prefill),
            ("block_tables", self.block_tables),
            ("sampling_params", self.sampling_params),
        ):
            if len(value) != expected:
                raise ValueError(
                    f"{field_name} must have length {expected}, got {len(value)}"
                )
        if len(query_lengths) != expected:
            raise ValueError(
                f"query_lengths must have length {expected}, "
                f"got {len(query_lengths)}"
            )
        if query_offsets[-1] != len(self.input_token_ids):
            raise ValueError(
                "query_lengths must sum to the number of packed input tokens"
            )

    @property
    def query_lengths(self) -> list[int]:
        return [int(value) for value in self.lengths.query_lengths]

    @property
    def query_offsets(self) -> list[int]:
        return [int(value) for value in self.lengths.query_offsets]

    @property
    def context_lengths(self) -> list[int]:
        return [int(value) for value in self.lengths.context_lengths]

    @property
    def kv_seq_lengths(self) -> list[int]:
        return [int(value) for value in self.lengths.kv_seq_lengths]

    @property
    def seq_lengths(self) -> list[int]:
        return self.query_lengths

    @property
    def token_offsets(self) -> list[int]:
        return self.query_offsets

    @property
    def total_tokens(self) -> int:
        return len(self.input_token_ids)


@dataclass
class SplitBatchMetadata:
    original_batch: BatchMetadata
    prefill_batch: BatchMetadata | None
    decode_batch: BatchMetadata | None
    prefill_indices: list[int]
    decode_indices: list[int]

    def recombine_outputs(
        self,
        prefill_outputs: torch.Tensor | None,
        decode_outputs: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.prefill_batch is not None:
            if prefill_outputs is None:
                raise ValueError("prefill_outputs is required")
            if prefill_outputs.shape[0] != self.prefill_batch.total_tokens:
                raise ValueError(
                    "prefill_outputs row count must match "
                    "prefill_batch.total_tokens"
                )
        elif prefill_outputs is not None and prefill_outputs.shape[0] != 0:
            raise ValueError("prefill_outputs must be None or empty")

        if self.decode_batch is not None:
            if decode_outputs is None:
                raise ValueError("decode_outputs is required")
            if decode_outputs.shape[0] != self.decode_batch.total_tokens:
                raise ValueError(
                    "decode_outputs row count must match "
                    "decode_batch.total_tokens"
                )
        elif decode_outputs is not None and decode_outputs.shape[0] != 0:
            raise ValueError("decode_outputs must be None or empty")

        total_tokens = self.original_batch.total_tokens
        sample_tensor = prefill_outputs
        if sample_tensor is None:
            sample_tensor = decode_outputs
        if sample_tensor is None:
            return torch.empty(0)

        output_shape = (total_tokens, *sample_tensor.shape[1:])
        result = torch.empty(
            output_shape,
            dtype=sample_tensor.dtype,
            device=sample_tensor.device,
        )

        query_offsets = self.original_batch.query_offsets
        prefill_cursor = 0
        decode_cursor = 0
        for seq_idx in range(len(self.original_batch.seq_ids)):
            seq_start = query_offsets[seq_idx]
            seq_end = query_offsets[seq_idx + 1]
            seq_len = seq_end - seq_start
            if seq_len == 0:
                continue

            if self.original_batch.is_prefill[seq_idx]:
                if prefill_outputs is None:
                    raise ValueError("prefill_outputs is required")
                source_start = prefill_cursor
                source_end = prefill_cursor + seq_len
                prefill_cursor = source_end
                result[seq_start:seq_end] = prefill_outputs[
                    source_start:source_end
                ]
            else:
                if decode_outputs is None:
                    raise ValueError("decode_outputs is required")
                source_start = decode_cursor
                source_end = decode_cursor + seq_len
                decode_cursor = source_end
                result[seq_start:seq_end] = decode_outputs[
                    source_start:source_end
                ]

        return result


def _slice_batch(batch: BatchMetadata, seq_indices: list[int]) -> BatchMetadata:
    if not seq_indices:
        raise ValueError("seq_indices must not be empty")

    seq_ids = [batch.seq_ids[i] for i in seq_indices]
    query_lengths = [batch.query_lengths[i] for i in seq_indices]
    context_lengths = [batch.context_lengths[i] for i in seq_indices]
    kv_seq_lengths = [batch.kv_seq_lengths[i] for i in seq_indices]
    is_prefill = [batch.is_prefill[i] for i in seq_indices]
    block_tables = [batch.block_tables[i] for i in seq_indices]
    sampling_params = [batch.sampling_params[i] for i in seq_indices]

    src_offsets = batch.query_offsets
    input_token_ids: list[int] = []
    query_offsets = [0]
    for i in seq_indices:
        start = src_offsets[i]
        end = src_offsets[i + 1]
        tokens = batch.input_token_ids[start:end]
        input_token_ids.extend(tokens)
        query_offsets.append(query_offsets[-1] + len(tokens))

    return BatchMetadata(
        seq_ids=seq_ids,
        input_token_ids=input_token_ids,
        lengths=PagedBatchLengths(
            query_lengths=query_lengths,
            query_offsets=query_offsets,
            context_lengths=context_lengths,
            kv_seq_lengths=kv_seq_lengths,
        ),
        is_prefill=is_prefill,
        block_tables=block_tables,
        sampling_params=sampling_params,
    )


def split_prefill_decode_batch(batch: BatchMetadata) -> SplitBatchMetadata:
    prefill_indices = [
        idx for idx, is_prefill in enumerate(batch.is_prefill) if is_prefill
    ]
    decode_indices = [
        idx for idx, is_prefill in enumerate(batch.is_prefill) if not is_prefill
    ]

    prefill_batch = (
        _slice_batch(batch, prefill_indices) if prefill_indices else None
    )
    decode_batch = (
        _slice_batch(batch, decode_indices) if decode_indices else None
    )

    return SplitBatchMetadata(
        original_batch=batch,
        prefill_batch=prefill_batch,
        decode_batch=decode_batch,
        prefill_indices=prefill_indices,
        decode_indices=decode_indices,
    )


class BatchBuilder:
    @staticmethod
    def from_scheduler_output(
        scheduler_output: SchedulerOutput,
        sequences: dict[int, SequenceData],
        kv_cache: PagedKVCache,
    ) -> BatchMetadata:
        seq_ids = [
            *scheduler_output.prefill_seq_ids,
            *scheduler_output.decode_seq_ids,
        ]

        input_token_ids: list[int] = []
        query_lengths: list[int] = []
        context_lengths: list[int] = []
        is_prefill: list[bool] = []
        block_tables: list[list[int]] = []
        sampling_params: list[SamplingParams] = []

        for seq_id in scheduler_output.prefill_seq_ids:
            sequence = sequences[seq_id]
            tokens = sequence.prompt_token_ids[sequence.num_computed_tokens :]
            input_token_ids.extend(tokens)
            query_lengths.append(len(tokens))
            context_lengths.append(sequence.num_computed_tokens)
            is_prefill.append(True)
            block_tables.append(kv_cache.get_block_table(seq_id))
            sampling_params.append(sequence.sampling_params)

        for seq_id in scheduler_output.decode_seq_ids:
            sequence = sequences[seq_id]
            token = (
                sequence.total_token_ids[-1:]
                if sequence.total_token_ids
                else []
            )
            input_token_ids.extend(token)
            query_lengths.append(len(token))
            context_lengths.append(sequence.num_computed_tokens)
            is_prefill.append(False)
            block_tables.append(kv_cache.get_block_table(seq_id))
            sampling_params.append(sequence.sampling_params)

        query_offsets = [0]
        for length in query_lengths:
            query_offsets.append(query_offsets[-1] + length)
        kv_seq_lengths = [
            context_len + query_len
            for context_len, query_len in zip(context_lengths, query_lengths)
        ]

        return BatchMetadata(
            seq_ids=seq_ids,
            input_token_ids=input_token_ids,
            lengths=PagedBatchLengths(
                query_lengths=query_lengths,
                query_offsets=query_offsets,
                context_lengths=context_lengths,
                kv_seq_lengths=kv_seq_lengths,
            ),
            is_prefill=is_prefill,
            block_tables=block_tables,
            sampling_params=sampling_params,
        )


__all__ = [
    "BatchBuilder",
    "BatchMetadata",
    "SchedulerOutput",
    "SplitBatchMetadata",
    "split_prefill_decode_batch",
]
