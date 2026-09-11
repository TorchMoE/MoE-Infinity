from __future__ import annotations

import os

import pytest
import torch

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
]

_QUALITY_ENABLED = os.environ.get("MOE_KV_QUANT_QUALITY") == "1"


@pytest.mark.skipif(
    not _QUALITY_ENABLED,
    reason="set MOE_KV_QUANT_QUALITY=1 to run the perplexity release gate",
)
def test_int8_wikitext2_perplexity_within_release_gate() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the perplexity gate")
    model_name = os.environ.get("MOE_TEST_MODEL")
    if not model_name:
        pytest.skip("set MOE_TEST_MODEL to a pinned local GQA checkpoint")
    dataset_path = os.environ.get("MOE_WIKITEXT2_PATH")
    if not dataset_path or not os.path.exists(dataset_path):
        pytest.skip("set MOE_WIKITEXT2_PATH to a pinned WikiText-2 snapshot")

    from tests.python.e2e._perplexity_harness import (  # type: ignore
        evaluate_perplexity,
    )

    native_ppl = evaluate_perplexity(
        model_name, dataset_path, kv_cache_format="native"
    )
    int8_ppl, effective_format = evaluate_perplexity(
        model_name,
        dataset_path,
        kv_cache_format="int8_sym",
        return_effective_format=True,
    )
    assert effective_format == "int8_sym", (
        "INT8 run must not silently fall back to native; "
        f"effective_format={effective_format}"
    )

    absolute_delta = int8_ppl - native_ppl
    relative_delta = absolute_delta / native_ppl
    print(
        f"model={model_name} native_ppl={native_ppl:.4f} "
        f"int8_ppl={int8_ppl:.4f} abs_delta={absolute_delta:.4f} "
        f"rel_delta={relative_delta:.4%} effective={effective_format}"
    )
    assert absolute_delta <= 0.10
    assert relative_delta <= 0.01
