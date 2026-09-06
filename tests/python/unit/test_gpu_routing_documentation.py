from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_gpu_routing_configuration_and_observability_are_documented():
    configuration = _read("docs/configuration.md")
    environment = _read("docs/environment-variables.md")

    assert "`gpu_only_expert_routing`" in configuration
    assert "single-host `dispatch_local`" in configuration
    assert "`speculative_prefetch_overlap=true`" in configuration
    assert (
        "invalid combinations raise before engine construction" in configuration
    )
    assert "CPU masks" in configuration
    assert "older native extensions" in configuration
    assert "DFlash route-ahead" in configuration

    for stage in (
        "`gpu_route_submit`",
        "`gpu_route_fallback`",
        "`gpu_route_handoff`",
        "`expert_completion_handoff`",
    ):
        assert stage in environment
    assert "route_failures" in environment
    assert "gpu_only_expert_routing=false" in environment


def test_gpu_routing_benchmark_runbook_documents_keep_and_rollback():
    for relative_path in (
        "docs/benchmarking.md",
        "benchmarks/expert_io_microbench/README.md",
    ):
        runbook = " ".join(_read(relative_path).split())
        assert "concurrency 1, 2, and 4" in runbook
        assert "TPOT p50 regression <=2%" in runbook
        assert "TPOT p99 regression <=5%" in runbook
        assert "zero route failures" in runbook
        assert "zero unexpected eager fallbacks" in runbook
        assert "gpu_only_expert_routing=false" in runbook
        assert "speculative_prefetch_overlap=false" in runbook
        assert "single-host personal-machine offloading" in runbook
        assert "not multi-node results" in runbook
