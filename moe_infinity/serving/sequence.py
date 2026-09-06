from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SequenceStatus(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    DRAFT = "draft"
    VERIFY = "verify"
    FINISHED = "finished"
    SWAPPED = "swapped"
    CANCELLED = "cancelled"


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    logprobs: int = 0
    max_tokens: int = 256
    stop: list[str] = field(default_factory=list)
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        self.stop = list(self.stop)


@dataclass
class SequenceData:
    seq_id: int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    num_computed_tokens: int = 0
    committed_kv_tokens: int = 0
    has_prefix_lease: bool = False
    created_at: float = field(default_factory=time.monotonic)

    def set_status(self, new_status: SequenceStatus) -> None:
        self._validate_transition(new_status)
        self.status = new_status

    def append_output_token(self, token_id: int) -> None:
        if self.status in (SequenceStatus.FINISHED, SequenceStatus.CANCELLED):
            raise RuntimeError(
                f"cannot append output token to sequence in state {self.status.value}"
            )
        self.output_token_ids.append(token_id)
        self.num_computed_tokens = len(self.prompt_token_ids) + len(
            self.output_token_ids
        )

    def _validate_transition(self, new_status: SequenceStatus) -> None:
        if self.status == new_status:
            return

        allowed_transitions: dict[SequenceStatus, set[SequenceStatus]] = {
            SequenceStatus.WAITING: {
                SequenceStatus.PREFILL,
                SequenceStatus.SWAPPED,
                SequenceStatus.CANCELLED,
            },
            SequenceStatus.PREFILL: {
                SequenceStatus.DECODE,
                SequenceStatus.DRAFT,
                SequenceStatus.FINISHED,
                SequenceStatus.SWAPPED,
                SequenceStatus.CANCELLED,
            },
            SequenceStatus.DECODE: {
                SequenceStatus.FINISHED,
                SequenceStatus.SWAPPED,
                SequenceStatus.CANCELLED,
            },
            SequenceStatus.DRAFT: {
                SequenceStatus.VERIFY,
                SequenceStatus.FINISHED,
                SequenceStatus.SWAPPED,
                SequenceStatus.CANCELLED,
            },
            SequenceStatus.VERIFY: {
                SequenceStatus.DRAFT,
                SequenceStatus.FINISHED,
                SequenceStatus.SWAPPED,
                SequenceStatus.CANCELLED,
            },
            SequenceStatus.SWAPPED: {
                SequenceStatus.WAITING,
                SequenceStatus.PREFILL,
                SequenceStatus.DECODE,
                SequenceStatus.DRAFT,
                SequenceStatus.FINISHED,
                SequenceStatus.CANCELLED,
            },
            SequenceStatus.FINISHED: set(),
            SequenceStatus.CANCELLED: set(),
        }

        if new_status not in allowed_transitions[self.status]:
            raise ValueError(
                f"invalid transition from {self.status.value} to {new_status.value}"
            )

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.output_token_ids]


@dataclass
class SequenceGroup:
    request_id: str
    sequences: list[SequenceData]
    arrival_time: float = field(default_factory=time.monotonic)

    @property
    def sequence_ids(self) -> list[int]:
        return [sequence.seq_id for sequence in self.sequences]

    def get_sequence(self, seq_id: int) -> Optional[SequenceData]:
        for sequence in self.sequences:
            if sequence.seq_id == seq_id:
                return sequence
        return None


Sequence = SequenceData


__all__ = [
    "SamplingParams",
    "Sequence",
    "SequenceData",
    "SequenceGroup",
    "SequenceStatus",
]
