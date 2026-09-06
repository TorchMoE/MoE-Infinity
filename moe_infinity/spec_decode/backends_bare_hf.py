"""Physical-cohort greedy and sampled execution for bare HF targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import torch

from moe_infinity.spec_decode._dflash_ops import (
    acceptance_lengths,
    build_block_with_prefixes,
    committed_tokens_ragged,
)
from moe_infinity.spec_decode._dflash_sample_ops import (
    _validate_generator_device,
    acceptance_sampled,
    committed_tokens_sampled,
    warped_probs,
)
from moe_infinity.spec_decode.protocols import (
    BackendCapabilities,
    ExecutorEvidence,
    NativeStepTrace,
    RequestSpec,
    SamplingContext,
    SessionTrace,
)


@dataclass(frozen=True)
class BareHFCohortResult:
    """Generated rows and compatibility diagnostics from one physical cohort."""

    generated_token_ids: tuple[tuple[int, ...], ...]
    step_trace: tuple[NativeStepTrace, ...]
    target_cache: Any
    draft_cache: Any
    session_traces: tuple[SessionTrace, ...] = ()

    @property
    def generated_lengths(self) -> tuple[int, ...]:
        return tuple(len(row) for row in self.generated_token_ids)


class BatchedBareHFBackend:
    """Execute a mixed greedy/sampled cohort with one shared dense cache.

    Rows use lockstep minimum-commit rollback.  A row that commits farther than
    the shared physical cache carries its already-emitted tail into the next
    verify block as a known prefix; those re-confirmed tokens are not emitted a
    second time.
    """

    name = "dflash-batched-bare-hf"
    allows_rich_forward = False

    def __init__(self, speculator: Any) -> None:
        self.speculator = speculator
        pairing_evidence = getattr(speculator, "pairing_evidence", None)
        executor_evidence = ExecutorEvidence(
            wiring_reachable=False,
            prefetcher_present=False,
            fallback_reason="executor_unreachable",
        )
        self.capabilities = BackendCapabilities(
            supports_batch=True,
            supports_sampling=True,
            supports_ragged_rows=True,
            cache_kind="dense_dynamic",
            supports_route_ahead=False,
            supports_rich_forward=False,
            **(
                {"pairing_evidence": pairing_evidence}
                if pairing_evidence is not None
                else {}
            ),
            executor_evidence=executor_evidence,
        )

    def supports(self, request: RequestSpec) -> bool:
        del request
        rich = callable(
            getattr(
                self.speculator.moe,
                "_native_model_forward_rich",
                None,
            )
        )
        return (not rich) or self.allows_rich_forward

    def cohort_key(self, request: RequestSpec) -> Hashable:
        del request
        return ("mixed-greedy-sampled", self.capabilities.cache_kind)

    @torch.no_grad()
    def execute_cohort(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: tuple[int, ...],
        stop_token_ids: tuple[int, ...],
        attention_mask: torch.Tensor,
        sampling_contexts: tuple[SamplingContext, ...] | None = None,
        stop_token_ids_by_row: tuple[tuple[int, ...], ...] | None = None,
    ) -> BareHFCohortResult:
        """Run dense-cache batching while preserving each row's sampler."""
        from transformers import DynamicCache

        spec = self.speculator
        if (
            callable(getattr(spec.moe, "_native_model_forward_rich", None))
            and not self.allows_rich_forward
        ):
            raise NotImplementedError(
                "BatchedBareHFBackend requires a bare HF target; the MoE "
                "rich-forward seam is batch==1 only"
            )
        if input_ids.ndim != 2:
            raise ValueError(
                "BatchedBareHFBackend expects input_ids of shape [batch, seq], "
                f"got {tuple(input_ids.shape)}"
            )

        batch, padded_prompt = int(input_ids.shape[0]), int(input_ids.shape[1])
        if len(max_new_tokens) != batch:
            raise ValueError(
                f"per-sequence max_new_tokens has {len(max_new_tokens)} entries "
                f"for batch size {batch}"
            )
        budgets = [int(value) for value in max_new_tokens]
        if any(value < 0 for value in budgets):
            raise ValueError(f"max_new_tokens must be >= 0, got {budgets}")

        input_ids = input_ids.to(spec.device)
        attention_mask = attention_mask.to(device=spec.device)
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} != "
                f"input_ids shape {tuple(input_ids.shape)}"
            )
        binary = (attention_mask == 0) | (attention_mask == 1)
        if not bool(torch.all(binary).item()):
            raise ValueError("attention_mask must be 0/1 valued")
        attention_mask = attention_mask.to(dtype=torch.long)
        if int(attention_mask[:, -1].min()) != 1:
            raise ValueError(
                "batched DFlash requires LEFT-padded prompts: every row's last "
                "token must be real (attention_mask[:, -1] == 1)"
            )
        steps = attention_mask[:, 1:] - attention_mask[:, :-1]
        if steps.numel() and int(steps.min()) < 0:
            raise ValueError(
                "batched DFlash requires LEFT-padded prompts: each "
                "attention_mask row must be 0*1* (pads first, then real tokens)"
            )

        if sampling_contexts is None:
            samplings = tuple(SamplingContext() for _ in range(batch))
        else:
            samplings = tuple(sampling_contexts)
            if len(samplings) != batch:
                raise ValueError(
                    f"sampling_contexts has {len(samplings)} entries for "
                    f"batch size {batch}"
                )
        for row, sampling in enumerate(samplings):
            if not isinstance(sampling, SamplingContext):
                raise TypeError(
                    f"sampling_contexts[{row}] must be SamplingContext"
                )
            if sampling.is_sampled and budgets[row] > 0:
                _validate_generator_device(sampling.generator, input_ids.device)
        if stop_token_ids_by_row is None:
            stop_sets = tuple(set(stop_token_ids) for _ in range(batch))
        else:
            if len(stop_token_ids_by_row) != batch:
                raise ValueError(
                    f"stop_token_ids_by_row has {len(stop_token_ids_by_row)} "
                    f"entries for batch size {batch}"
                )
            stop_sets = tuple(set(row) for row in stop_token_ids_by_row)

        block_size = int(spec.config.block_size)
        layer_ids = list(spec.config.target_layer_ids)
        mask_token_id = int(spec.config.mask_token_id)
        pads = padded_prompt - attention_mask.sum(dim=1)

        spec._configure_target_hooks(input_ids)
        prefill_position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
        logits, hidden_states, target_kv = spec._forward_target(
            input_ids,
            past_key_values=None,
            logits_to_keep=1,
            attention_mask=attention_mask,
            position_ids=prefill_position_ids,
        )
        greedy_anchors = logits[:, -1, :].argmax(dim=-1)
        anchors: list[int] = []
        for row, sampling in enumerate(samplings):
            if budgets[row] <= 0 or sampling.is_greedy:
                anchors.append(int(greedy_anchors[row]))
                continue
            anchor_probs = warped_probs(
                logits[row, -1],
                sampling.temperature,
                sampling.top_k,
                sampling.top_p,
            )
            anchors.append(
                int(
                    torch.multinomial(
                        anchor_probs,
                        num_samples=1,
                        generator=sampling.generator,
                    )
                )
            )
        context_feature = spec._extract_context_feature(
            hidden_states, layer_ids
        ).to(spec.device)
        emitted: list[list[int]] = [
            ([anchors[row]] if budgets[row] > 0 else []) for row in range(batch)
        ]
        session_traces = [
            SessionTrace(
                request_id=f"direct-{row}",
                backend=self.name,
                cache_kind=self.capabilities.cache_kind,
                sampled=samplings[row].is_sampled,
                route_ahead_status="disabled",
                pairing_evidence=self.capabilities.pairing_evidence,
                executor_evidence=self.capabilities.executor_evidence,
            )
            for row in range(batch)
        ]
        finished = [
            budgets[row] <= 0
            or (bool(stop_sets[row]) and anchors[row] in stop_sets[row])
            for row in range(batch)
        ]
        start = padded_prompt
        draft_kv = DynamicCache() if spec._drafter_has_kv_cache else None
        step_trace: list[NativeStepTrace] = []

        def active(row: int) -> bool:
            return not finished[row] and len(emitted[row]) < budgets[row]

        while any(active(row) for row in range(batch)):
            prev_start = start
            pendings = [
                len(emitted[row]) - (start - padded_prompt)
                for row in range(batch)
            ]
            prefixes = [
                emitted[row][start - padded_prompt :] if active(row) else []
                for row in range(batch)
            ]
            block = build_block_with_prefixes(
                prefixes, mask_token_id, block_size
            ).to(spec.device)

            drafter_out = spec._run_drafter(
                block, context_feature, start, draft_kv
            )
            draft_logits = spec.lm_head(drafter_out)[:, -(block_size - 1) :, :]
            draft_prob_rows: list[torch.Tensor | None] = [None] * batch
            sampled_reconstruction = [False] * batch
            for row, sampling in enumerate(samplings):
                if not active(row):
                    continue
                pending = pendings[row]
                if sampling.is_greedy:
                    if pending < block_size:
                        block[row, pending:] = draft_logits[
                            row, pending - 1 :
                        ].argmax(dim=-1)
                    continue
                if pending > 1:
                    # Tokens beyond the physical shared-cache position were
                    # already chosen by this row's prior logical verify. Feed
                    # them only to reconstruct cache state; do not draft,
                    # accept, emit, or consume request RNG again.
                    sampled_reconstruction[row] = True
                    continue
                probabilities = warped_probs(
                    draft_logits[row],
                    sampling.temperature,
                    sampling.top_k,
                    sampling.top_p,
                )
                draft_prob_rows[row] = probabilities
                for slot in range(block_size - 1):
                    block[row, slot + 1] = torch.multinomial(
                        probabilities[slot],
                        num_samples=1,
                        generator=sampling.generator,
                    )

            cache_snapshot = spec._snapshot_target_cache(target_kv)
            block_attention = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        batch,
                        start - padded_prompt + block_size,
                        dtype=attention_mask.dtype,
                        device=spec.device,
                    ),
                ],
                dim=1,
            )
            block_position_ids = torch.arange(
                start,
                start + block_size,
                device=spec.device,
                dtype=torch.long,
            ).unsqueeze(0) - pads.unsqueeze(1)
            logits, hidden_states, target_kv = spec._verify_target_block(
                block,
                target_kv,
                attention_mask=block_attention,
                position_ids=block_position_ids,
            )
            posterior = logits.argmax(dim=-1).to(spec.device)
            greedy_accepts = acceptance_lengths(block, posterior)
            greedy_committed = committed_tokens_ragged(
                block, posterior, greedy_accepts
            )

            step_cc: dict[int, int] = {}
            for row in range(batch):
                if not active(row):
                    continue
                pending = pendings[row]
                sampling = samplings[row]
                if sampled_reconstruction[row]:
                    step_cc[row] = pending - 1
                    continue
                if sampling.is_sampled:
                    draft_probs = draft_prob_rows[row]
                    assert draft_probs is not None
                    decision = acceptance_sampled(
                        draft_probs,
                        warped_probs(
                            logits[row],
                            sampling.temperature,
                            sampling.top_k,
                            sampling.top_p,
                        ),
                        block[row, 1:],
                        generator=sampling.generator,
                    )
                    accept = decision.accept
                    committed = committed_tokens_sampled(
                        block[row : row + 1],
                        decision.accept,
                        decision.final_token,
                    )
                else:
                    accept = greedy_accepts[row]
                    committed = greedy_committed[row]
                step_tokens = [
                    int(token) for token in committed.emitted[0].tolist()
                ]
                new_tokens = step_tokens[pending - 1 :]
                keep = len(new_tokens)
                stop = False
                stop_ids = stop_sets[row]
                if stop_ids:
                    for index, token in enumerate(new_tokens):
                        if token in stop_ids:
                            keep = index + 1
                            stop = True
                            break
                remaining = budgets[row] - len(emitted[row])
                if keep > remaining:
                    keep = remaining
                    stop = True
                emitted[row].extend(new_tokens[:keep])
                step_cc[row] = min(pending - 1 + keep, accept) + 1
                if stop or len(emitted[row]) >= budgets[row]:
                    finished[row] = True

            continuing = [row for row in step_cc if active(row)]
            min_cc = (
                min(step_cc[row] for row in continuing)
                if continuing
                else min(step_cc.values())
            )
            spec._rollback_target_cache(
                target_kv,
                cache_snapshot,
                prev_start=prev_start,
                committed=min_cc,
                block_size=block_size,
            )
            start = prev_start + min_cc
            assert int(target_kv.get_seq_length()) == start

            if spec.route_ahead_stats is not None:
                spec.route_ahead_stats.commit_step(kept_rows=min_cc)

            for row, committed_count in step_cc.items():
                step = NativeStepTrace(
                    prev_start=prev_start,
                    accept=committed_count - 1,
                    start=start,
                    emitted_len=len(emitted[row]),
                    target_cache_len=int(target_kv.get_seq_length()),
                    draft_cache_len=(
                        int(draft_kv.get_seq_length())
                        if draft_kv is not None
                        else None
                    ),
                )
                step_trace.append(step)
                session_traces[row].append(step)
            if not continuing:
                break

            suffix = spec._extract_context_feature(hidden_states, layer_ids).to(
                spec.device
            )[:, :min_cc, :]
            if spec._drafter_has_kv_cache:
                context_feature = suffix
            else:
                context_feature = torch.cat([context_feature, suffix], dim=1)

        generated = tuple(
            tuple(emitted[row][: budgets[row]]) for row in range(batch)
        )
        for row, trace in enumerate(session_traces):
            trace.emitted = len(generated[row])
            trace.finish_reason = (
                "stop"
                if generated[row] and generated[row][-1] in stop_sets[row]
                else "length"
            )
        return BareHFCohortResult(
            generated_token_ids=generated,
            step_trace=tuple(step_trace),
            target_cache=target_kv,
            draft_cache=draft_kv,
            session_traces=tuple(session_traces),
        )


__all__ = ["BareHFCohortResult", "BatchedBareHFBackend"]
