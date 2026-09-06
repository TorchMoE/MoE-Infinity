"""Task 8: wire the native DFlash speculator as ``GenerationEngine.spec_strategy``.

Pins the two Task-8 QA scenarios on the tiny CPU fixtures (T5), driving the
REAL sync path end to end: ``MoE.generate(..., speculative_draft=...)`` ->
``GenerationEngine.generate`` -> ``spec_strategy.run`` ->
``DFlashSpeculator.generate`` (via ``MoE._native_model_forward_rich``).

(a) happy: a greedy, batch==1 ``generate`` with a drafter configured routes
    through the native strategy -- ``strategy.run`` is invoked exactly once and
    the emitted ids are non-empty and identical to the standalone native loop.
(b) negative: no drafter configured -> the ``_generate_standard`` path is used
    and the output is byte-identical to the plain-engine baseline.

Also pinned: omitting the kwarg detaches a previously attached strategy (the
kwarg is per-call, never sticky), non-greedy params with a drafter configured
still use the standard path (T1 gate), and rich batch>1 uses independent
request sessions without pretending the singleton engine is physically batched.
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from fixtures_tiny import (  # noqa: E402
    build_tiny_drafter,
    build_tiny_target,
    make_tiny_drafter_config,
    set_determinism,
)

from moe_infinity.engine.generation_loop import GenerationEngine  # noqa: E402
from moe_infinity.engine.types import SamplingParams  # noqa: E402
from moe_infinity.entrypoints.big_modeling import MoE  # noqa: E402
from moe_infinity.memory.kv_cache_manager import KVCacheManager  # noqa: E402
from moe_infinity.runtime.attention_types import KVCacheSpec  # noqa: E402
from moe_infinity.spec_decode import (  # noqa: E402
    DFlashSpeculator,
    read_dflash_config,
)
from moe_infinity.spec_decode.dflash import _resolve_stop_ids  # noqa: E402

PROMPT = [3, 7, 11, 2, 5]
MAX_NEW_TOKENS = 16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _tiny_moe_shell(seed: int = 0):
    """An MoE instance shell around the tiny target with a real engine.

    Mirrors the production wiring of ``_build_native_components`` (engine
    holds ``model_forward_fn=shell._native_model_forward``) without the
    offload runtime; ``_configure_hook`` is a no-op like in
    ``test_native_step.py``.
    """
    set_determinism(seed)
    target = build_tiny_target(seed=seed).to(DEVICE)
    shell = MoE.__new__(MoE)
    shell.model = target
    shell.use_native_engine = True
    shell.max_seq_length = 64
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    shell._configure_hook = lambda input_ids: None

    stop_ids = _resolve_stop_ids(target, None)
    eos_token_id = stop_ids[0] if stop_ids else -1

    engine = GenerationEngine(
        kv_cache_manager=KVCacheManager(
            num_gpu_blocks=64, num_cpu_blocks=16, block_size=4
        ),
        kv_spec=KVCacheSpec(
            num_kv_heads=2, head_dim=8, dtype=torch.float32, block_size=4
        ),
        num_layers=int(target.config.num_hidden_layers),
        vocab_size=int(target.config.vocab_size),
        model_forward_fn=shell._native_model_forward,
        eos_token_id=eos_token_id,
        max_seq_length=64,
    )
    shell._native_generation_engine = engine
    return shell, target


def _tiny_speculator(shell, target):
    drafter = build_tiny_drafter(target, seed=1).to(DEVICE)
    config = read_dflash_config(make_tiny_drafter_config(target.config))
    return DFlashSpeculator.from_models(
        shell, drafter, config=config, device=DEVICE
    )


def _spy_on_run(spec: DFlashSpeculator):
    calls = []
    orig_run = spec.run

    def run_spy(*, engine, prompt_token_ids, sampling_params, request_id=None):
        calls.append(
            {
                "engine": engine,
                "prompt_token_ids": list(prompt_token_ids),
                "sampling_params": sampling_params,
                "request_id": request_id,
            }
        )
        return orig_run(
            engine=engine,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        )

    spec.run = run_spy
    return calls


def _moe_generate(shell, **kwargs):
    input_ids = torch.tensor([PROMPT], dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return shell.generate(input_ids, **kwargs)


def test_engine_routes_greedy_through_native_strategy():
    """QA scenario (a): greedy batch==1 + drafter -> strategy.run once."""
    shell, target = _tiny_moe_shell()
    engine = shell._native_generation_engine
    spec = _tiny_speculator(shell, target)
    run_calls = _spy_on_run(spec)

    standard_calls = []
    orig_standard = engine._generate_standard

    def standard_spy(*args, **kwargs):
        standard_calls.append((args, kwargs))
        return orig_standard(*args, **kwargs)

    engine._generate_standard = standard_spy

    out = _moe_generate(
        shell,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        speculative_draft=spec,
    )

    assert len(run_calls) == 1
    assert standard_calls == []
    call = run_calls[0]
    assert call["engine"] is engine
    assert call["prompt_token_ids"] == PROMPT
    assert call["sampling_params"].temperature == 0.0
    assert call["sampling_params"].max_tokens == MAX_NEW_TOKENS
    assert engine.spec_strategy is spec

    # run() strips the prompt; MoE.generate re-prepends it exactly once.
    new_ids = out[0, len(PROMPT) :].tolist()
    assert 0 < len(new_ids) <= MAX_NEW_TOKENS

    standalone = spec.generate(
        torch.tensor([PROMPT], dtype=torch.long, device=DEVICE),
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        stop_token_ids=[engine.eos_token_id],
    )
    assert new_ids == standalone[0, len(PROMPT) :].tolist()

    print(
        f"engine-routed: run_calls={len(run_calls)} "
        f"new_tokens={len(new_ids)} ids={new_ids}"
    )


def test_no_drafter_uses_standard_path():
    """QA scenario (b): no drafter -> _generate_standard, baseline-equal."""
    shell, _ = _tiny_moe_shell()
    engine = shell._native_generation_engine

    shell._cached_past_key_values = None
    baseline = engine.generate(
        prompt_token_ids=list(PROMPT),
        sampling_params=SamplingParams(
            temperature=0.0, top_p=1.0, top_k=0, max_tokens=MAX_NEW_TOKENS
        ),
    )
    assert engine.spec_strategy is None

    standard_calls = []
    orig_standard = engine._generate_standard

    def standard_spy(*args, **kwargs):
        standard_calls.append((args, kwargs))
        return orig_standard(*args, **kwargs)

    engine._generate_standard = standard_spy

    out = _moe_generate(shell, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)

    assert engine.spec_strategy is None
    assert len(standard_calls) == 1
    assert out[0].tolist() == PROMPT + baseline.output_token_ids

    print(
        f"no-drafter standard: new_tokens={len(baseline.output_token_ids)} "
        f"ids={baseline.output_token_ids}"
    )


def test_omitted_kwarg_detaches_previously_attached_strategy():
    """The speculative_draft kwarg is per-call: omitting it restores the
    standard path (never sticky), so baseline/spec comparisons interleave
    safely."""
    shell, target = _tiny_moe_shell()
    engine = shell._native_generation_engine
    spec = _tiny_speculator(shell, target)

    _moe_generate(
        shell,
        do_sample=False,
        max_new_tokens=4,
        speculative_draft=spec,
    )
    assert engine.spec_strategy is spec

    shell._cached_past_key_values = None
    baseline = engine.generate(
        prompt_token_ids=list(PROMPT),
        sampling_params=SamplingParams(
            temperature=0.0, top_p=1.0, top_k=0, max_tokens=4
        ),
    )
    out = _moe_generate(shell, do_sample=False, max_new_tokens=4)
    assert engine.spec_strategy is None
    assert out[0].tolist() == PROMPT + baseline.output_token_ids


def test_non_greedy_with_drafter_uses_standard_path():
    """T1 gate through the MoE API: temperature>0 bypasses the strategy."""
    shell, target = _tiny_moe_shell()
    engine = shell._native_generation_engine
    spec = _tiny_speculator(shell, target)
    run_calls = _spy_on_run(spec)

    torch.manual_seed(123)
    out = _moe_generate(
        shell,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=8,
        speculative_draft=spec,
    )
    assert run_calls == []
    assert out.shape[1] > len(PROMPT)


def test_batch_larger_than_one_with_drafter_preserves_every_row():
    """The facade drives independent rich sessions and preserves row order."""
    shell, target = _tiny_moe_shell()
    spec = _tiny_speculator(shell, target)
    input_ids = torch.tensor([PROMPT, PROMPT], dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        output = shell.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=[2, 4],
            speculative_draft=spec,
        )

    assert output.shape == (2, len(PROMPT) + 4)
    assert torch.equal(output[:, : len(PROMPT)], input_ids)
    assert spec.last_generated_lengths == [2, 4]
