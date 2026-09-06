from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PR_WORKFLOW = ROOT / ".github/workflows/ci-pr.yml"


def test_python_matrix_runs_dflash_unified_validation_e2e() -> None:
    workflow = PR_WORKFLOW.read_text(encoding="utf-8")
    unit_job = workflow.split("  unit-tests:\n", maxsplit=1)[1].split(
        "\n  build:", maxsplit=1
    )[0]

    unit_test_position = unit_job.index("- name: Run CPU unit tests")
    dflash_position = unit_job.index(
        "- name: Run DFlash unified validation E2E"
    )
    contextpilot_position = unit_job.index("- name: Install contextpilot")

    assert 'python-version: ["3.10", "3.12"]' in unit_job
    assert (
        "pytest tests/python/integration/"
        "test_dflash_unified_validation_e2e.py" in unit_job
    )
    assert unit_test_position < dflash_position < contextpilot_position
