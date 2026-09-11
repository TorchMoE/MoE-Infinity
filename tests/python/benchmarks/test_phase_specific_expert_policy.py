from __future__ import annotations

from typing import Any, cast

import pytest

from benchmarks.serving.phase_specific_expert_policy import (
    BenchmarkCell,
    build_matrix,
    compare_reports,
    summarize,
    summarize_rows,
)


def test_matrix_covers_prefill_decode_and_mixed_pressure() -> None:
    cells = build_matrix()
    assert {
        (cell.prompt_tokens, cell.output_tokens, cell.concurrency)
        for cell in cells
    } == {
        (128, 16, 1),
        (2048, 16, 1),
        (128, 256, 1),
        (2048, 256, 1),
        (128, 256, 8),
        (2048, 256, 8),
    }


def test_summary_uses_stream_timestamps_for_ttft_and_tpot() -> None:
    row = summarize(submitted=1.0, token_times=[2.0, 2.2, 2.5])
    assert row["ttft_s"] == 1.0
    assert row["tpot_s"] == 0.25
    assert row["e2e_s"] == 1.5
    assert row["output_tokens"] == 3


def test_summary_rejects_a_stream_without_tokens() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        summarize(submitted=1.0, token_times=[])


def test_summary_rows_reports_required_percentiles() -> None:
    rows = [
        {"ttft_s": 1.0, "tpot_s": 0.2, "e2e_s": 2.0},
        {"ttft_s": 2.0, "tpot_s": 0.4, "e2e_s": 4.0},
    ]
    summary = summarize_rows(rows)
    assert summary["ttft_s"] == {"p50": 1.5, "p90": 1.9, "p99": 1.99}
    assert summary["tpot_s"] == {"p50": 0.3, "p90": 0.38, "p99": 0.398}
    assert summary["e2e_s"] == {"p50": 3.0, "p90": 3.8, "p99": 3.98}


def _report(policy: str) -> dict[str, object]:
    cell = BenchmarkCell(128, 16, 1)
    return {
        "policy": policy,
        "environment": {"fingerprint": "same"},
        "cells": [
            {
                "cell": cell.as_dict(),
                "requests": [
                    {
                        "prompt_tokens": 128,
                        "requested_output_tokens": 16,
                        "output_tokens": 16,
                        "token_ids": [3, 4],
                    }
                ],
                "summary": {
                    "ttft_s": {"p50": 2.0},
                    "tpot_s": {"p50": 0.5},
                },
            }
        ],
    }


def test_compare_reports_emits_deltas_only_after_parity_checks() -> None:
    off = _report("off")
    on = _report("on")
    on_cell = cast(list[dict[str, Any]], on["cells"])[0]
    summary = cast(dict[str, dict[str, float]], on_cell["summary"])
    summary["ttft_s"]["p50"] = 1.5
    summary["tpot_s"]["p50"] = 0.4

    comparison = compare_reports(off, on)

    assert comparison == [
        {
            "cell": BenchmarkCell(128, 16, 1).as_dict(),
            "ttft_s_delta": {"p50": -0.5},
            "tpot_s_delta": {"p50": -0.1},
        }
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["environment"].update(fingerprint="other"),
            "environment",
        ),
        (
            lambda report: report["cells"][0]["requests"][0].update(
                token_ids=[9]
            ),
            "token IDs",
        ),
        (
            lambda report: report["cells"][0]["requests"][0].update(
                prompt_tokens=127
            ),
            "prompt length",
        ),
    ],
)
def test_compare_reports_rejects_noncomparable_runs(
    mutation, message: str
) -> None:
    off = _report("off")
    on = _report("on")
    mutation(on)
    with pytest.raises(ValueError, match=message):
        compare_reports(off, on)
