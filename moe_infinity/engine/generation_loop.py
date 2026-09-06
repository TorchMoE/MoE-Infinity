from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch
import torch.nn.functional as F

from moe_infinity.engine.types import SamplingParams, SequenceStatus
from moe_infinity.memory.kv_cache_manager import KVCacheManager
from moe_infinity.runtime.attention_types import AttentionMetadata, KVCacheSpec


class PromptTooLongError(ValueError):
    pass


class KVCacheAllocationError(RuntimeError):
    pass


@dataclass
class GenerationResult:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    finish_reason: SequenceStatus


class SpecDecodeStrategy(Protocol):
    def run(
        self,
        *,
        engine: GenerationEngine,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> list[int]: ...


class GenerationEngine:
    kv_mgr: KVCacheManager
    kv_spec: KVCacheSpec
    num_layers: int
    vocab_size: int
    eos_token_id: int
    _model_forward: Optional[
        Callable[[list[int], AttentionMetadata], torch.Tensor]
    ]
    max_seq_length: Optional[int]
    spec_strategy: Optional[SpecDecodeStrategy]

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        kv_spec: KVCacheSpec,
        num_layers: int,
        vocab_size: int,
        model_forward_fn: Optional[
            Callable[[list[int], AttentionMetadata], torch.Tensor]
        ] = None,
        eos_token_id: int = 2,
        max_seq_length: Optional[int] = None,
        spec_strategy: Optional[SpecDecodeStrategy] = None,
    ) -> None:
        self.kv_mgr = kv_cache_manager
        self.kv_spec = kv_spec
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self._model_forward = model_forward_fn
        self.max_seq_length = max_seq_length
        self.spec_strategy = spec_strategy

    def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> GenerationResult:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")

        sp = sampling_params or SamplingParams()

        if self._spec_strategy_applies(sp, batch_size=1):
            assert self.spec_strategy is not None
            rid = request_id or str(uuid.uuid4())
            generated_ids = list(
                self.spec_strategy.run(
                    engine=self,
                    prompt_token_ids=prompt_token_ids,
                    sampling_params=sp,
                    request_id=rid,
                )
            )
            finish_reason = (
                SequenceStatus.FINISHED_STOPPED
                if generated_ids and generated_ids[-1] == self.eos_token_id
                else SequenceStatus.FINISHED_LENGTH
            )
            return GenerationResult(
                request_id=rid,
                prompt_token_ids=prompt_token_ids,
                output_token_ids=generated_ids,
                finish_reason=finish_reason,
            )

        return self._generate_standard(prompt_token_ids, sp, request_id)

    def _spec_strategy_applies(
        self, sp: SamplingParams, batch_size: int
    ) -> bool:
        if self.spec_strategy is None:
            return False
        supports = getattr(self.spec_strategy, "supports_engine_request", None)
        if callable(supports):
            return bool(supports(sp, batch_size=batch_size))
        if batch_size != 1:
            return False
        if (
            sp.temperature > 0
            or sp.top_p < 1.0
            or sp.top_k > 0
            or getattr(sp, "do_sample", False)
        ):
            return False
        return True

    def _generate_standard(
        self,
        prompt_token_ids: list[int],
        sp: SamplingParams,
        request_id: Optional[str] = None,
    ) -> GenerationResult:
        rid = request_id or str(uuid.uuid4())

        num_prompt_tokens = len(prompt_token_ids)
        if (
            self.max_seq_length is not None
            and num_prompt_tokens > self.max_seq_length
        ):
            raise PromptTooLongError(
                f"prompt length {num_prompt_tokens} exceeds max_seq_length {self.max_seq_length}"
            )

        if not self.kv_mgr.allocate_blocks_for_sequence(rid, num_prompt_tokens):
            raise KVCacheAllocationError(
                f"Failed to allocate KV blocks for {num_prompt_tokens} tokens"
            )

        output_ids: list[int] = []
        finish_reason = SequenceStatus.FINISHED_LENGTH

        try:
            block_table = self.kv_mgr.get_block_table(rid)
            slot_mapping = self._compute_slot_mapping(
                block_table, num_prompt_tokens
            )

            prefill_meta = AttentionMetadata(
                block_tables=torch.tensor([block_table], dtype=torch.int32),
                seq_lens=torch.tensor([num_prompt_tokens], dtype=torch.int32),
                max_seq_len=num_prompt_tokens,
                num_prefill_tokens=num_prompt_tokens,
                num_decode_tokens=0,
                slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64),
                is_prefill=True,
            )

            prefill_logits = self._run_forward(
                token_ids=prompt_token_ids,
                attention_metadata=prefill_meta,
            )

            if sp.max_tokens <= 0:
                return GenerationResult(
                    request_id=rid,
                    prompt_token_ids=prompt_token_ids,
                    output_token_ids=output_ids,
                    finish_reason=finish_reason,
                )

            next_token = self._sample(prefill_logits[-1:], sp)
            output_ids.append(int(next_token))

            for _ in range(sp.max_tokens - 1):
                if output_ids[-1] == self.eos_token_id:
                    finish_reason = SequenceStatus.FINISHED_STOPPED
                    break

                current_seq_len = num_prompt_tokens + len(output_ids)

                if current_seq_len > len(block_table) * self.kv_spec.block_size:
                    if not self.kv_mgr.append_token_block(rid):
                        finish_reason = SequenceStatus.FINISHED_LENGTH
                        break
                    block_table = self.kv_mgr.get_block_table(rid)

                decode_slot = self._compute_slot_mapping(
                    block_table, current_seq_len
                )[-1:]

                decode_meta = AttentionMetadata(
                    block_tables=torch.tensor([block_table], dtype=torch.int32),
                    seq_lens=torch.tensor([current_seq_len], dtype=torch.int32),
                    max_seq_len=current_seq_len,
                    num_prefill_tokens=0,
                    num_decode_tokens=1,
                    slot_mapping=torch.tensor(decode_slot, dtype=torch.int64),
                    is_prefill=False,
                )

                decode_logits = self._run_forward(
                    token_ids=[output_ids[-1]],
                    attention_metadata=decode_meta,
                )
                output_ids.append(int(self._sample(decode_logits, sp)))

            if output_ids and output_ids[-1] == self.eos_token_id:
                finish_reason = SequenceStatus.FINISHED_STOPPED

        finally:
            self.kv_mgr.free_sequence(rid)

        return GenerationResult(
            request_id=rid,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_ids,
            finish_reason=finish_reason,
        )

    def _compute_slot_mapping(
        self,
        block_table: list[int],
        num_tokens: int,
    ) -> list[int]:
        block_size = self.kv_spec.block_size
        slots: list[int] = []
        for token_idx in range(num_tokens):
            block_idx = token_idx // block_size
            token_offset = token_idx % block_size
            if block_idx < len(block_table):
                slots.append(block_table[block_idx] * block_size + token_offset)
        return slots

    def _run_forward(
        self,
        token_ids: list[int],
        attention_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self._model_forward is not None:
            return self._model_forward(token_ids, attention_metadata)
        return torch.randn(len(token_ids), self.vocab_size)

    def _sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        final_logits = logits[-1]

        if params.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if params.temperature == 0:
            return int(final_logits.argmax().item())

        if params.temperature != 1.0:
            final_logits = final_logits / params.temperature

        if params.top_k > 0:
            top_k = min(params.top_k, final_logits.size(-1))
            topk_indices = torch.topk(final_logits, top_k).indices
            filtered = torch.full_like(final_logits, float("-inf"))
            filtered[topk_indices] = final_logits[topk_indices]
            final_logits = filtered

        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                final_logits, descending=True
            )
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            remove_mask = cumulative_probs > params.top_p
            if remove_mask.numel() > 0:
                remove_mask[0] = False
            filtered_sorted = sorted_logits.masked_fill(
                remove_mask, float("-inf")
            )

            filtered = torch.full_like(final_logits, float("-inf"))
            filtered[sorted_indices] = filtered_sorted
            final_logits = filtered

        probs = F.softmax(final_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return int(sampled.item())


__all__ = [
    "GenerationEngine",
    "GenerationResult",
    "KVCacheAllocationError",
    "PromptTooLongError",
    "SpecDecodeStrategy",
]
