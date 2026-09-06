"""Track C: batch > 1 support for native DFlash (greedy, bare-HF target).

Pins the Track-C contract on the tiny CPU fixtures:

(a) batched == per-sequence-looped greedy, token-identical, for prompts of
    DIFFERENT lengths (left-padded, ``attention_mask``) and for equal-length
    prompts with no mask; both also equal ``plain_greedy_decode`` (the
    absolute oracle). Batching is a throughput change, never a correctness
    change.
(b) mixed accept lengths in ONE block: a scripted drafter makes row 0
    full-accept (``accept == 9``) and row 1 full-reject (``accept == 0``) in
    the same steps, forcing the lockstep-min rollback (cache advances by 1)
    and the known-prefix re-feed (row 0's block saturates to all-known
    tokens). Both rows still equal plain greedy.
(c) per-sequence EOS mid-block (row 0 stops at EOS, inclusive; row 1 runs to
    budget) and per-sequence ``max_new_tokens`` budgets -- ragged completion.
(d) batch==1 through the BATCHED path (``_generate_batched`` directly) is
    token-identical to the legacy single path (``generate`` dispatch) and to
    plain greedy.

Plus the C0 batched pure ops (``acceptance_lengths`` /
``committed_tokens_ragged`` / ``build_block_with_prefixes``) against their
per-row v1 counterparts, and the dispatch guard rails (sampled bare-HF
batch>1 uses physical execution while MoE-rich batch>1 uses independent
request sessions; right-padded/non-monotone masks raise ``ValueError``).

Trace convention (batched): one ``NativeStepTrace`` per ACTIVE row per step;
``accept`` is that row's effective accept (``cc_b - 1``) and ``start`` the
uniform post-rollback cache length, so per step ``start == prev_start +
min(accept over the step's rows) + 1``.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    TINY_BLOCK_SIZE,
    TINY_HIDDEN,
    TINY_VOCAB,
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
    plain_greedy_decode,
)

from moe_infinity.spec_decode import (  # noqa: E402
    DFlashSpeculator,
    read_dflash_config,
)
from moe_infinity.spec_decode._dflash_ops import (  # noqa: E402
    acceptance_length,
    acceptance_lengths,
    build_block_with_prefixes,
    committed_tokens,
    committed_tokens_ragged,
)

PROMPT_A = [3, 7, 11, 2, 5]
PROMPT_B = [1, 2, 3]
PROMPT_C = [10, 20, 30, 40]
EOS_ID = 62  # absent from the tiny target's greedy continuations used here

_TARGET = None
_DRAFTER = None


def _tiny_spec():
    """Fresh speculator per test (cheap ``from_models``) over shared models."""
    global _TARGET, _DRAFTER
    if _TARGET is None:
        _TARGET = build_tiny_target(seed=0)
        _DRAFTER = build_tiny_drafter(_TARGET, seed=1)
    config = read_dflash_config(make_tiny_drafter_config(_TARGET.config))
    spec = DFlashSpeculator.from_models(
        _TARGET, _DRAFTER, config=config, device="cpu"
    )
    return spec, _TARGET


def _left_pad(prompts, pad_id=0):
    width = max(len(p) for p in prompts)
    ids = torch.tensor([[pad_id] * (width - len(p)) + list(p) for p in prompts])
    mask = torch.tensor(
        [[0] * (width - len(p)) + [1] * len(p) for p in prompts]
    )
    return ids, mask, width


def _batched_new_tokens(out, spec, width):
    lengths = spec.last_generated_lengths
    assert lengths is not None
    return [
        out[b, width : width + lengths[b]].tolist() for b in range(out.shape[0])
    ]


def _plain_new(target, prompt, max_new):
    return plain_greedy_decode(
        target, torch.tensor([prompt]), max_new_tokens=max_new
    )[0, len(prompt) :].tolist()


def _greedy_streams(target, prompts, max_new):
    """Absolute (prompt ++ greedy) ids per row, long enough for draft scripting."""
    return [
        plain_greedy_decode(
            target,
            torch.tensor([p]),
            max_new_tokens=max_new + TINY_BLOCK_SIZE + 1,
        )[0].tolist()
        for p in prompts
    ]


class _ScriptedBatchedHead:
    """Per-row scripted drafter argmax: ``draft_fn(start, row)`` returns the
    ``block_size - 1`` draft ids for block positions 1..9 of that row."""

    def __init__(self, draft_fn) -> None:
        self.draft_fn = draft_fn
        self.start = None

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.start is not None
        batch, length = hidden.shape[0], hidden.shape[1]
        logits = torch.zeros(
            batch, length, TINY_VOCAB, dtype=hidden.dtype, device=hidden.device
        )
        for b in range(batch):
            drafts = [int(t) for t in self.draft_fn(self.start, b)]
            assert len(drafts) == TINY_BLOCK_SIZE - 1
            for i, tok in enumerate(drafts):
                logits[b, length - (TINY_BLOCK_SIZE - 1) + i, tok] = 1.0
        return logits


def _install_scripted_batched_drafter(monkeypatch, spec, draft_fn, batch):
    """Bypass drafter compute; feed per-row scripted drafts into the accept
    rule. The verify forward stays REAL (the genuine target)."""
    head = _ScriptedBatchedHead(draft_fn)

    def fake_run(block, context_feature, start, draft_kv):
        head.start = start
        return torch.zeros(batch, TINY_BLOCK_SIZE, TINY_HIDDEN)

    monkeypatch.setattr(spec, "_run_drafter", fake_run)
    monkeypatch.setattr(spec, "lm_head", head)


def _true_continuation_drafts(streams, pads):
    """Draft the row's true greedy token at every block position (full accept).

    Block position ``s`` sits at absolute stream index ``start - pads[b] + s``.
    """

    def draft_fn(start, b):
        base = start - int(pads[b])
        stream = streams[b]
        return [
            stream[base + s] if base + s < len(stream) else 0
            for s in range(1, TINY_BLOCK_SIZE)
        ]

    return draft_fn


def _force_target_argmax_rows(monkeypatch, spec, verify_rows):
    """One-hot override of the verify posterior: ``{row: {position: token}}``."""
    orig = spec._forward_target

    def wrapped(
        input_ids,
        past_key_values=None,
        logits_to_keep=0,
        attention_mask=None,
        position_ids=None,
    ):
        logits, hidden, kv = orig(
            input_ids,
            past_key_values=past_key_values,
            logits_to_keep=logits_to_keep,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if int(logits_to_keep) == 0 and verify_rows:
            logits = logits.clone()
            for row, rowspec in verify_rows.items():
                for pos, tok in rowspec.items():
                    logits[row, pos, :] = -1e9
                    logits[row, pos, int(tok)] = 1e9
        return logits, hidden, kv

    monkeypatch.setattr(spec, "_forward_target", wrapped)


def _grouped_trace(spec):
    by_step = defaultdict(list)
    for rec in spec.step_trace:
        by_step[rec.prev_start].append(rec)
    return by_step


def _assert_batched_trace_invariants(spec):
    by_step = _grouped_trace(spec)
    for prev_start, records in by_step.items():
        advance = records[0].start - prev_start
        assert all(rec.start == records[0].start for rec in records)
        # The uniform cache advance is the SMALLEST per-row commit among the
        # rows that CONTINUE past this step (a row that stops here is excluded
        # -- its cache row is never read again). The continuing set is a
        # subset of the step's records, hence the two-sided bound; when no row
        # stops mid-step it collapses to advance == min(accepts) + 1.
        assert min(rec.accept for rec in records) + 1 <= advance
        assert advance <= max(rec.accept for rec in records) + 1
        assert advance >= 1
        for rec in records:
            assert rec.target_cache_len == rec.start


# ---------------------------------------------------------------------------
# C0: batched pure ops
# ---------------------------------------------------------------------------


def _accept_case_row(k: int):
    """One block/posterior row pair with hand-checked accept ``k``."""
    block = [100] + [11 + i for i in range(TINY_BLOCK_SIZE - 1)]
    posterior = [0] * TINY_BLOCK_SIZE
    for i in range(k):
        posterior[i] = 11 + i
    if k < TINY_BLOCK_SIZE - 1:
        posterior[k] = 900 + k
        for i in range(k + 1, TINY_BLOCK_SIZE - 1):
            posterior[i] = 800 + i
    posterior[TINY_BLOCK_SIZE - 1] = 999
    return block, posterior


def test_acceptance_lengths_matches_per_row_v1_op():
    ks = [0, 4, TINY_BLOCK_SIZE - 1]
    rows, posts = zip(*[_accept_case_row(k) for k in ks])
    block = torch.tensor(list(rows), dtype=torch.long)
    posterior = torch.tensor(list(posts), dtype=torch.long)
    accepts = acceptance_lengths(block, posterior)
    assert accepts == ks
    for b, k in enumerate(ks):
        assert acceptance_length(block[b : b + 1], posterior[b : b + 1]) == k


def test_committed_tokens_ragged_matches_per_row_v1_op():
    ks = [0, 3, TINY_BLOCK_SIZE - 1]
    rows, posts = zip(*[_accept_case_row(k) for k in ks])
    block = torch.tensor(list(rows), dtype=torch.long)
    posterior = torch.tensor(list(posts), dtype=torch.long)
    ragged = committed_tokens_ragged(block, posterior, ks)
    assert len(ragged) == 3
    for b, k in enumerate(ks):
        ref = committed_tokens(block[b : b + 1], posterior[b : b + 1], k)
        assert ragged[b].emitted.shape == (1, k + 1)
        assert torch.equal(ragged[b].emitted, ref.emitted)
        assert torch.equal(ragged[b].block_prefix, ref.block_prefix)
        assert torch.equal(ragged[b].bonus, ref.bonus)
    # Raggedness a dense Committed cannot express:
    assert ragged[0].emitted.shape[1] != ragged[2].emitted.shape[1]


def test_committed_tokens_ragged_rejects_row_count_mismatch():
    block = torch.tensor([_accept_case_row(1)[0]] * 2, dtype=torch.long)
    posterior = torch.tensor([_accept_case_row(1)[1]] * 2, dtype=torch.long)
    with pytest.raises(ValueError, match="rows"):
        committed_tokens_ragged(block, posterior, [1])


def test_build_block_with_prefixes_contents_and_shape():
    block = build_block_with_prefixes(
        [[5], [1, 2, 3], []], mask_token_id=200, block_size=TINY_BLOCK_SIZE
    )
    assert block.shape == (3, TINY_BLOCK_SIZE)
    assert block.dtype == torch.long
    assert block[0].tolist() == [5] + [200] * (TINY_BLOCK_SIZE - 1)
    assert block[1].tolist() == [1, 2, 3] + [200] * (TINY_BLOCK_SIZE - 3)
    assert block[2].tolist() == [200] * TINY_BLOCK_SIZE


def test_build_block_with_prefixes_rejects_overlong_prefix():
    with pytest.raises(ValueError, match="exceeds block_size"):
        build_block_with_prefixes(
            [[1] * (TINY_BLOCK_SIZE + 1)],
            mask_token_id=200,
            block_size=TINY_BLOCK_SIZE,
        )


# ---------------------------------------------------------------------------
# (a) batched == looped singles == plain greedy (token-identical)
# ---------------------------------------------------------------------------


def test_batched_matches_looped_singles_token_identical():
    spec, target = _tiny_spec()
    # Six prompts of differing lengths (the test_native_e2e set), left-padded.
    prompts = [
        PROMPT_A,
        PROMPT_B,
        PROMPT_C,
        [5],
        [8, 16, 24, 32, 40, 48],
        [42, 17, 33, 9],
    ]
    ids, mask, width = _left_pad(prompts)
    max_new = 24

    out = spec.generate(ids, max_new_tokens=max_new, attention_mask=mask)
    batched = _batched_new_tokens(out, spec, width)
    assert spec.last_generated_lengths == [max_new] * len(prompts)

    for b, prompt in enumerate(prompts):
        single = spec.generate(torch.tensor([prompt]), max_new_tokens=max_new)[
            0, len(prompt) :
        ].tolist()
        plain = _plain_new(target, prompt, max_new)
        assert batched[b] == single == plain, (
            f"row {b} diverged:\n  batched={batched[b]}\n  single ={single}\n"
            f"  plain  ={plain}"
        )

    _assert_batched_trace_invariants(spec)
    # Non-degenerate: the real drafter is genuinely exercised in batched mode
    # (accepted drafts somewhere in the batch -- not a trivial accept-0 loop).
    accepts = [rec.accept for rec in spec.step_trace]
    assert sum(accepts) > 0


def test_batched_equal_length_prompts_without_mask():
    spec, target = _tiny_spec()
    prompts = [PROMPT_A, [42, 17, 33, 9, 21]]
    ids = torch.tensor(prompts)
    max_new = 16

    out = spec.generate(ids, max_new_tokens=max_new)
    batched = _batched_new_tokens(out, spec, len(prompts[0]))
    for b, prompt in enumerate(prompts):
        assert batched[b] == _plain_new(target, prompt, max_new)


# ---------------------------------------------------------------------------
# (b) mixed accept lengths in one block (lockstep-min rollback + re-feed)
# ---------------------------------------------------------------------------


def test_mixed_accept_lengths_in_one_block(monkeypatch):
    spec, target = _tiny_spec()
    prompts = [PROMPT_A, PROMPT_B]
    pads = [0, len(PROMPT_A) - len(PROMPT_B)]
    max_new = 21
    streams = _greedy_streams(target, prompts, max_new)
    true_drafts = _true_continuation_drafts(streams, pads)

    def draft_fn(start, b):
        if b == 0:
            return true_drafts(start, b)  # full accept every step
        # Full reject: first draft mismatches the target's argmax.
        base = start - pads[1]
        wrong = (streams[1][base + 1] + 1) % TINY_VOCAB
        return [wrong] * (TINY_BLOCK_SIZE - 1)

    _install_scripted_batched_drafter(monkeypatch, spec, draft_fn, batch=2)
    ids, mask, width = _left_pad(prompts)

    out = spec.generate(ids, max_new_tokens=max_new, attention_mask=mask)
    batched = _batched_new_tokens(out, spec, width)
    for b, prompt in enumerate(prompts):
        assert batched[b] == _plain_new(target, prompt, max_new)

    by_step = _grouped_trace(spec)
    # Row 0 accepts 9 (cc 10) while row 1 accepts 0 (cc 1): every cache
    # advance is the lockstep minimum of 1, so row 0's un-cached accepted
    # tokens are re-fed as the known prefix of its next block.
    two_row_steps = [recs for recs in by_step.values() if len(recs) == 2]
    assert two_row_steps, "expected steps with both rows active"
    for records in two_row_steps:
        assert sorted(rec.accept for rec in records) == [0, TINY_BLOCK_SIZE - 1]
    assert all(recs[0].start - prev == 1 for prev, recs in by_step.items())
    _assert_batched_trace_invariants(spec)


# ---------------------------------------------------------------------------
# (c) per-sequence EOS mid-block + ragged completion
# ---------------------------------------------------------------------------


def test_per_seq_eos_mid_block_ragged_completion(monkeypatch):
    spec, target = _tiny_spec()
    prompts = [PROMPT_A, PROMPT_B]
    pads = [0, len(PROMPT_A) - len(PROMPT_B)]
    max_new = 20
    streams = _greedy_streams(target, prompts, max_new)

    # Row 0 drafts its true continuation; the verify posterior at block
    # position 2 is forced to EOS, so accept == 2 and the bonus IS the stop
    # token (emitted, never cached). Row 1 is left untouched.
    _install_scripted_batched_drafter(
        monkeypatch, spec, _true_continuation_drafts(streams, pads), batch=2
    )
    _force_target_argmax_rows(monkeypatch, spec, {0: {2: EOS_ID}})
    ids, mask, width = _left_pad(prompts)

    out = spec.generate(
        ids,
        max_new_tokens=max_new,
        attention_mask=mask,
        stop_token_ids=[EOS_ID],
    )
    batched = _batched_new_tokens(out, spec, width)

    stream0 = streams[0]
    expected0 = [stream0[5], stream0[6], stream0[7], EOS_ID]
    assert batched[0] == expected0
    assert spec.last_generated_lengths == [4, max_new]
    # Row 1 never saw the override: full budget, token-identical to plain.
    assert batched[1] == _plain_new(target, prompts[1], max_new)
    # Row 0 stopped at step 1, so the step-1 cache advance is the CONTINUING
    # row's commit (row 1 full-accepted: cc == block_size), not the finishing
    # row's smaller one -- the lockstep minimum is taken over live rows only.
    step1 = _grouped_trace(spec)[width]
    assert len(step1) == 2
    assert step1[0].start - width == TINY_BLOCK_SIZE
    _assert_batched_trace_invariants(spec)


def test_per_seq_eos_as_accepted_draft_stops_only_that_row(monkeypatch):
    spec, target = _tiny_spec()
    prompts = [PROMPT_A, PROMPT_B]
    pads = [0, len(PROMPT_A) - len(PROMPT_B)]
    max_new = 20
    streams = _greedy_streams(target, prompts, max_new)

    def draft_fn(start, b):
        if b == 0:
            base = start - pads[0]
            stream = streams[0]
            # d1, d2 true; d3 = EOS (accepted via the forced posterior).
            drafts = [stream[base + 1], stream[base + 2], EOS_ID]
            drafts += [0] * (TINY_BLOCK_SIZE - 1 - len(drafts))
            return drafts
        return _true_continuation_drafts(streams, pads)(start, b)

    _install_scripted_batched_drafter(monkeypatch, spec, draft_fn, batch=2)
    _force_target_argmax_rows(monkeypatch, spec, {0: {2: EOS_ID, 3: 40}})
    ids, mask, width = _left_pad(prompts)

    out = spec.generate(
        ids,
        max_new_tokens=max_new,
        attention_mask=mask,
        stop_token_ids=[EOS_ID],
    )
    batched = _batched_new_tokens(out, spec, width)

    stream0 = streams[0]
    assert batched[0] == [stream0[5], stream0[6], stream0[7], EOS_ID]
    assert spec.last_generated_lengths == [4, max_new]
    assert batched[1] == _plain_new(target, prompts[1], max_new)


def test_per_sequence_max_new_tokens_ragged_completion():
    spec, target = _tiny_spec()
    prompts = [PROMPT_A, PROMPT_B]
    ids, mask, width = _left_pad(prompts)
    budgets = [6, 14]

    out = spec.generate(ids, max_new_tokens=budgets, attention_mask=mask)
    batched = _batched_new_tokens(out, spec, width)

    assert spec.last_generated_lengths == budgets
    for b, prompt in enumerate(prompts):
        assert batched[b] == _plain_new(target, prompt, budgets[b])
    _assert_batched_trace_invariants(spec)


# ---------------------------------------------------------------------------
# (d) batch==1 through the batched path == legacy single path
# ---------------------------------------------------------------------------


def test_batch_one_via_batched_path_equals_legacy():
    spec, target = _tiny_spec()
    prompt = torch.tensor([PROMPT_A])
    max_new = 32

    legacy = spec.generate(prompt, max_new_tokens=max_new)
    batched = spec._generate_batched(
        prompt, max_new_tokens=max_new, stop_token_ids=None, attention_mask=None
    )
    plain = plain_greedy_decode(target, prompt, max_new_tokens=max_new)

    assert torch.equal(batched, legacy)
    assert torch.equal(batched, plain)
    assert spec.last_generated_lengths == [max_new]


def test_batch_one_via_batched_path_with_stop_ids_equals_legacy():
    spec, _ = _tiny_spec()
    prompt = torch.tensor([PROMPT_A])
    max_new = 24

    legacy = spec.generate(
        prompt, max_new_tokens=max_new, stop_token_ids=[EOS_ID]
    )
    batched = spec._generate_batched(
        prompt,
        max_new_tokens=max_new,
        stop_token_ids=[EOS_ID],
        attention_mask=torch.ones_like(prompt),
    )
    assert torch.equal(batched, legacy)


# ---------------------------------------------------------------------------
# Dispatch guard rails
# ---------------------------------------------------------------------------


def test_batched_sampled_bare_hf_is_supported():
    spec, _ = _tiny_spec()
    ids = torch.tensor([PROMPT_A, PROMPT_A])
    output = spec.generate(
        ids,
        max_new_tokens=4,
        temperature=0.7,
        generator=torch.Generator().manual_seed(17),
    )
    assert output.shape == (2, len(PROMPT_A) + 4)
    assert spec.last_generated_lengths == [4, 4]


def test_batched_moe_rich_target_uses_independent_request_sessions():
    from moe_infinity.entrypoints.big_modeling import MoE

    spec, target = _tiny_spec()
    shell = MoE.__new__(MoE)
    shell.model = target
    shell._configure_hook = lambda _input_ids: None
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    shell._resolve_native_input_device = lambda: torch.device("cpu")
    spec = DFlashSpeculator.from_models(
        shell, spec.draft, config=spec.config, device="cpu"
    )

    ids = torch.tensor([PROMPT_A, PROMPT_A])
    output = spec.generate(ids, max_new_tokens=[2, 4])

    assert output.shape == (2, ids.shape[1] + 4)
    assert torch.equal(output[:, : ids.shape[1]], ids)
    assert spec.last_generated_lengths == [2, 4]
    assert [trace.request_id for trace in spec.last_session_traces] == [
        "direct-0",
        "direct-1",
    ]


def test_right_padded_attention_mask_rejected():
    spec, _ = _tiny_spec()
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    with pytest.raises(ValueError, match="LEFT-padded"):
        spec.generate(ids, max_new_tokens=4, attention_mask=mask)


def test_nonmonotone_attention_mask_rejected():
    spec, _ = _tiny_spec()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[1, 1, 1], [1, 0, 1]])
    with pytest.raises(ValueError, match="LEFT-padded"):
        spec.generate(ids, max_new_tokens=4, attention_mask=mask)


def test_per_sequence_budget_length_mismatch_rejected():
    spec, _ = _tiny_spec()
    ids = torch.tensor([PROMPT_A, PROMPT_A])
    with pytest.raises(ValueError, match="batch size"):
        spec.generate(ids, max_new_tokens=[4])
