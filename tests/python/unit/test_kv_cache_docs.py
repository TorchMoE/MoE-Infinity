from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_kv_cache_docs_state_opt_in_and_precision_contract() -> None:
    text = (ROOT / "docs" / "configuration.md").read_text()
    assert "`kv_cache_format`" in text
    assert "default: `native`" in text
    assert "Storage precision" in text
    assert "Transfer precision" in text
    assert "Attention execution precision" in text
    assert "MLA" in text and "fallback" in text


def test_serving_docs_include_one_command_rollback() -> None:
    text = (ROOT / "docs" / "serving.md").read_text()
    assert "--kv-cache-format native" in text
    assert "effective_kv_cache_format" in text
