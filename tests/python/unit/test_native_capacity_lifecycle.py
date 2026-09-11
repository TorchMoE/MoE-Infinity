from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "core"
SOURCE = (CORE / "prefetch/archer_prefetch_handle.cpp").read_text()


def _between(start: str, end: str) -> str:
    start_index = SOURCE.index(start)
    end_index = SOURCE.index(end, start_index + len(start))
    return SOURCE[start_index:end_index]


def test_constructor_leaves_expert_capacity_unconfigured() -> None:
    constructor = _between(
        "ArcherPrefetchHandle::ArcherPrefetchHandle(",
        "ArcherPrefetchHandle::~ArcherPrefetchHandle()",
    )
    assert "ConfigureCapacity" not in constructor
    assert "GetSparseCacheLimit" not in constructor


def test_topology_setters_configure_only_after_initialization() -> None:
    legacy = _between(
        "void ArcherPrefetchHandle::SetTopology(",
        "void ArcherPrefetchHandle::SetTopologyV2(",
    )
    v2 = _between(
        "void ArcherPrefetchHandle::SetTopologyV2(",
        "ArcherPrefetchHandle::GetTopologySnapshot()",
    )
    assert legacy.index("InitializeTopology(topology)") < legacy.index(
        "ConfigureExpertCapacityAfterTopology()"
    )
    assert v2.index("InitializeTopologyV2(topology)") < v2.index(
        "ConfigureExpertCapacityAfterTopology()"
    )


def test_capacity_helper_reads_sparse_limit_before_update() -> None:
    helper = _between(
        "void ArcherPrefetchHandle::ConfigureExpertCapacityAfterTopology()",
        "void ArcherPrefetchHandle::SetTopology(",
    )
    assert helper.index("GetSparseCacheLimit(device)") < helper.index(
        "ConfigureCapacity(gpu_id, bytes)"
    )


def test_only_topology_completion_installs_production_capacity() -> None:
    callsites = {
        str(path.relative_to(CORE)): text.count("->ConfigureCapacity(")
        for path in CORE.rglob("*.cpp")
        if (text := path.read_text()).count("->ConfigureCapacity(")
    }
    assert callsites == {"prefetch/archer_prefetch_handle.cpp": 1}
