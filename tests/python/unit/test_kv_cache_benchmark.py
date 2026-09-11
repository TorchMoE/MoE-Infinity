import json

from benchmarks.serving.kv_cache_quantization import (
    BenchmarkResult,
    parse_args,
    write_json,
)


def test_default_matrix_covers_long_context_and_concurrency() -> None:
    args = parse_args(
        [
            "--model",
            "/models/qwen3",
            "--offload-dir",
            "/tmp/offload",
        ]
    )
    assert args.context_lengths == [128, 2048, 8192, 32768]
    assert args.batch_sizes == [1, 4, 16]
    assert args.formats == ["native", "int8_sym"]


def test_result_schema_separates_precisions() -> None:
    result = BenchmarkResult.example()
    payload = result.to_dict()
    assert payload["storage_format"] == "int8_sym"
    assert payload["transfer_precision"] == "int8+fp16_scale"
    assert payload["execution_dtype"] == "float16"


def test_cli_exposes_expected_flags() -> None:
    args = parse_args(
        [
            "--model",
            "/models/qwen3",
            "--offload-dir",
            "/tmp/offload",
            "--formats",
            "native",
            "int8_sym",
            "--context-lengths",
            "128",
            "2048",
            "--batch-sizes",
            "1",
            "--strict-fallback",
            "--output-json",
            "/tmp/out.json",
        ]
    )
    assert args.formats == ["native", "int8_sym"]
    assert args.context_lengths == [128, 2048]
    assert args.batch_sizes == [1]
    assert args.strict_fallback is True
    assert args.output_json == "/tmp/out.json"


def test_write_json_creates_parent_directory(tmp_path) -> None:
    out = tmp_path / "nested" / "dir" / "result.json"
    write_json(str(out), [BenchmarkResult.example().to_dict()])
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert data[0]["storage_format"] == "int8_sym"
