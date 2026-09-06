import pytest

from benchmarks.serving.chunked_prefill_latency import (
    build_prompt_token_ids,
    summarize_requests,
)


def test_summarize_requests_reports_ttft_and_tpot_tails() -> None:
    summary = summarize_requests(
        [
            {"started_at": 1.0, "token_times": [1.1, 1.2, 1.4]},
            {"started_at": 2.0, "token_times": [2.2, 2.5, 2.9]},
        ]
    )

    assert summary["request_count"] == 2
    assert summary["ttft_p50_ms"] == pytest.approx(150.0)
    assert summary["ttft_p99_ms"] == pytest.approx(199.0)
    assert summary["tpot_p50_ms"] == pytest.approx(250.0)
    assert summary["tpot_p99_ms"] == pytest.approx(397.0)


def test_summarize_requests_keeps_single_token_tpot_null() -> None:
    summary = summarize_requests([{"started_at": 1.0, "token_times": [1.1]}])
    assert summary["ttft_p99_ms"] == pytest.approx(100.0)
    assert summary["tpot_p99_ms"] is None


def test_build_prompt_token_ids_is_tokenizer_verified_exact_length() -> None:
    class Tokenizer:
        vocab_size = 32

        def encode(
            self, text: str, add_special_tokens: bool = False
        ) -> list[int]:
            assert add_special_tokens is False
            return [3, 5, 7]

    prompt = build_prompt_token_ids(Tokenizer(), target_length=8)

    assert prompt == [3, 5, 7, 3, 5, 7, 3, 5]
    assert len(prompt) == 8
