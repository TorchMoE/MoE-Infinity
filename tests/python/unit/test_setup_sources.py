import ast
from pathlib import Path


def test_store_sources_link_expert_residency_implementation() -> None:
    setup_path = Path(__file__).resolve().parents[3] / "setup.py"
    module = ast.parse(setup_path.read_text())
    store_sources = None
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "_STORE_SOURCES"
            for target in statement.targets
        ):
            store_sources = ast.literal_eval(statement.value)
            break
    assert store_sources is not None
    assert "core/prefetch/expert_residency.cpp" in store_sources
    assert store_sources.count("core/prefetch/expert_residency.cpp") == 1
