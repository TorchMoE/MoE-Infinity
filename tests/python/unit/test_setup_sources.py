import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


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


def read_literal_list(path: Path, name: str) -> list[str]:
    module = ast.parse(path.read_text())
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            value = ast.literal_eval(statement.value)
            assert isinstance(value, list)
            return value
    raise AssertionError(f"{name} not found in {path}")


def test_store_sources_link_one_residency_authority() -> None:
    sources = read_literal_list(ROOT / "setup.py", "_STORE_SOURCES")
    residency = [p for p in sources if "residency" in p]
    assert residency == ["core/prefetch/expert_residency.cpp"]


def test_cmake_links_one_residency_authority() -> None:
    text = (ROOT / "core/CMakeLists.txt").read_text()
    assert text.count("prefetch/expert_residency.cpp") == 1
    assert text.count("residency.cpp") == 1


def test_testing_definition_is_consistent_across_build_systems() -> None:
    setup = (ROOT / "setup.py").read_text()
    root_cmake = (ROOT / "CMakeLists.txt").read_text()
    core_cmake = (ROOT / "core/CMakeLists.txt").read_text()
    extensions_cmake = (ROOT / "extensions/CMakeLists.txt").read_text()

    assert 'os.environ.get("MOE_INFINITY_TESTING") == "1"' in setup
    assert setup.count('"-DMOE_INFINITY_TESTING=1"') == 2
    assert (
        'option(MOE_INFINITY_TESTING "Expose native test-only bindings" OFF)'
        in root_cmake
    )
    assert "target_compile_definitions(archer_core" in core_cmake
    assert "target_compile_definitions(prefetch_op" in extensions_cmake
