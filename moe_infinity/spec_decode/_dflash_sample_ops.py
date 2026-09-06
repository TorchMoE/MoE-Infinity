"""Sampled (non-greedy) DFlash accept rule: lossless speculative sampling for
a block-parallel (diffusion) proposal.

The greedy ops in ``_dflash_ops`` decide acceptance by argmax agreement. This
module is the temperature/top-k/top-p counterpart: per-slot rejection
sampling with residual correction -- the block-parallel generalization of
Leviathan et al. 2023 (Sec. 3.2) / Chen et al. 2023 speculative sampling,
i.e. Stern-style blockwise parallel proposals verified with the lossless
per-position accept test also used by Medusa (rejection-sampling variant) and
tree-based verifiers (SpecInfer/EAGLE).

Setup (one draft->verify step, block ``[anchor, d_1..d_{B-1}]``):

* ``Q_i`` -- the distribution the drafter ACTUALLY sampled draft ``d_i``
  from: the warped softmax of the drafter's slot-``i`` logits. The drafter is
  non-causal (block diffusion): every ``Q_i`` is produced in parallel,
  conditioned on the anchor + context feature, never on the other drafts.
* ``P_i`` -- the target's true conditional for slot ``i`` given the tokens
  actually preceding it: the warped softmax of the verify-forward logits at
  slot ``i - 1`` (causal attention over ``[anchor, d_1..d_{i-1}]``).

Accept rule: for ``i = 1..B-1`` accept ``d_i`` with probability
``min(1, P_i(d_i) / Q_i(d_i))``; on the first rejection at slot ``n`` emit a
correction token drawn from the residual ``norm(max(0, P_n - Q_n))`` and end
the step; if every draft is accepted, emit a bonus token from ``P_B`` (the
verify logits at the last slot).

Losslessness (Track-B B0 gate). Lemma (Leviathan Sec. 3.2): for any two
distributions P, Q, a draw ``x ~ Q`` accepted with probability
``min(1, P(x)/Q(x))`` -- and redrawn on rejection from
``norm(max(0, P - Q))`` -- is distributed EXACTLY as P, because
``P(out=t) = min(P(t),Q(t)) + max(0, P(t)-Q(t)) = P(t)``. Applying the lemma
per slot, conditioned on the committed prefix, yields a joint emitted stream
identical to autoregressive sampling from the (warped) target: the lemma
requires only that (a) ``d_i`` is a genuine draw from the named ``Q_i`` and
(b) ``P_i`` is the target's true conditional given the actual preceding
tokens. The proposal's non-autoregressive structure is irrelevant -- ``Q_i``
may depend on anything fixed before verification (here the anchor and the
context feature). The warp (temperature -> top-k -> top-p, mirroring
``GenerationEngine._sample``) is applied identically to P, to Q, and to the
plain sampler, so the preserved distribution is exactly what plain sampled
generation produces; EOS / max-token truncation is the same deterministic
function of the emitted stream on both paths.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

from moe_infinity.spec_decode._dflash_ops import Committed


class SampledAcceptance(NamedTuple):
    accept: int  # accepted leading drafts, in [0, block_size - 1]
    final_token: int  # residual correction (reject) or bonus (full accept)


def _validate_generator_device(
    generator: Optional[torch.Generator], probability_device: torch.device
) -> None:
    """Require an explicit request generator on the probability tensor device."""
    if generator is None:
        return
    generator_device = torch.device(generator.device)
    probability_device = torch.device(probability_device)
    same_device_type = generator_device.type == probability_device.type
    compatible_index = (
        generator_device.index is None
        or probability_device.index is None
        or generator_device.index == probability_device.index
    )
    if not (same_device_type and compatible_index):
        raise ValueError(
            f"generator device {generator_device} does not match "
            f"probability device {probability_device}"
        )


def warped_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Row-wise softmax with the ``GenerationEngine._sample`` warp.

    Order matters and matches the engine sampler exactly: temperature scale,
    top-k filter, top-p (nucleus) filter, softmax. Applied identically to the
    draft and target distributions, the rejection rule then preserves the
    warped target -- precisely the distribution plain sampled generation
    draws from. ``logits`` is ``[..., vocab]``; each row is warped
    independently (the top-p "keep at least one token" guard is per row).
    """
    if float(temperature) <= 0:
        raise ValueError(
            "warped_probs requires temperature > 0; greedy (temperature == 0)"
            " uses the argmax path in _dflash_ops"
        )
    if float(temperature) != 1.0:
        logits = logits / float(temperature)
    if int(top_k) > 0:
        k = min(int(top_k), int(logits.shape[-1]))
        topk_idx = torch.topk(logits, k, dim=-1).indices
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, topk_idx, logits.gather(-1, topk_idx))
        logits = filtered
    if float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, sorted_idx, sorted_logits)
        logits = filtered
    return F.softmax(logits, dim=-1)


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``norm(max(0, p - q))`` -- the rejection-resampling distribution.

    Sums to 1 by construction. An all-zero residual only arises when
    ``p == q`` (both rows are distributions), where rejection has probability
    0; floating-point noise can still land here, in which case we fall back
    to ``p`` itself -- sampling the exact target row is lossless.
    """
    residual = (p - q).clamp_min(0)
    total = residual.sum()
    if float(total) <= 0:
        return p
    return residual / total


def acceptance_sampled(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    drafts: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> SampledAcceptance:
    """Per-slot rejection-sampling accept rule (see module docstring).

    ``draft_probs``  -- ``[B - 1, V]`` rows ``Q_1..Q_{B-1}``: the warped
                        drafter slot distributions that produced ``drafts``;
    ``target_probs`` -- ``[B, V]`` rows ``P_1..P_B``: the warped verify
                        logits at slots ``0..B-1`` (the last row is the bonus
                        distribution);
    ``drafts``       -- ``[B - 1]`` token ids ``d_1..d_{B-1}`` sampled from
                        the ``Q_i`` rows (NOT argmax).

    Returns the accepted leading-draft count and the step's final token: the
    residual correction at the first rejected slot, or -- when every draft is
    accepted -- a bonus drawn from ``P_B``. ``generator`` isolates the draws
    for seeded determinism; ``None`` uses the global torch RNG (seed it with
    ``torch.manual_seed``).
    """
    _validate_generator_device(generator, target_probs.device)
    num_drafts = int(drafts.shape[0])
    for i in range(num_drafts):
        token = int(drafts[i])
        q = float(draft_probs[i, token])
        p = float(target_probs[i, token])
        accept_prob = min(1.0, p / q) if q > 0 else 0.0
        if (
            float(
                torch.rand((), device=target_probs.device, generator=generator)
            )
            < accept_prob
        ):
            continue
        correction = torch.multinomial(
            residual_distribution(target_probs[i], draft_probs[i]),
            num_samples=1,
            generator=generator,
        )
        return SampledAcceptance(accept=i, final_token=int(correction))
    bonus = torch.multinomial(
        target_probs[-1], num_samples=1, generator=generator
    )
    return SampledAcceptance(accept=num_drafts, final_token=int(bonus))


def committed_tokens_sampled(
    block: torch.Tensor, accept: int, final_token: int
) -> Committed:
    """Sampled-mode counterpart of ``committed_tokens``.

    The emitted/cached split is identical to greedy -- accepted drafts are
    emitted and cached, the final token is emitted-but-NOT-cached (it becomes
    the next anchor) -- but the final token is the accept rule's residual
    correction / sampled bonus, not an argmax posterior row.
    """
    accept = int(accept)
    accepted_drafts = block[:, 1 : accept + 1]
    bonus = torch.as_tensor(
        int(final_token), dtype=block.dtype, device=block.device
    ).reshape(1, 1)
    return Committed(
        emitted=torch.cat([accepted_drafts, bonus], dim=1),
        block_prefix=block[:, : accept + 1],
        bonus=bonus,
    )


__all__ = [
    "SampledAcceptance",
    "acceptance_sampled",
    "committed_tokens_sampled",
    "residual_distribution",
    "warped_probs",
]
