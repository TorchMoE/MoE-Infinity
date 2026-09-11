from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_adaptive_memory_docs_cover_configuration_rollout_and_rollback() -> (
    None
):
    configuration = (ROOT / "docs/configuration.md").read_text("utf-8")
    serving = (ROOT / "docs/serving.md").read_text("utf-8")
    benchmarking = (ROOT / "docs/benchmarking.md").read_text("utf-8")

    for text in (configuration, serving, benchmarking):
        assert "adaptive_memory_enabled" in text
        assert "free-memory reserve" in text.lower()
    assert 'POST /v1/config {"adaptive_memory_enabled": false}' in serving
    assert "Stage 0" in serving and "Stage 4" in serving
    assert "drain" in serving.lower() and "CUDA completion event" in serving
    assert "partial_donor_committed" in serving.lower()
    assert "_fi_prefill" in configuration and "_fi_decode" in configuration
    assert "device_id" in serving and "cuda:0" in serving
    assert "requested_config" in benchmarking
    assert "effective_config" in benchmarking
    assert "no assumed performance gain" in benchmarking.lower()
    assert "https://arxiv.org/abs/2606.21868" in benchmarking
