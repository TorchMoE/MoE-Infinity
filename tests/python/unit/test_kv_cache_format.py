import pytest
import torch

from moe_infinity.runtime.kv_cache_format import (
    KVCacheBackendCapabilities,
    KVCacheFormat,
    KVCacheModelInfo,
    allocate_layered_paged_kv_store,
    quantize_tokenwise_symmetric,
    resolve_kv_cache_format,
)


def _make_layered_store(
    *,
    format_name: str = "int8_sym",
    num_layers: int = 2,
    num_blocks: int = 4,
    owner_id: str = "unit-store",
    block_size: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    execution_dtype: torch.dtype = torch.float32,
):
    return allocate_layered_paged_kv_store(
        owner_id=owner_id,
        format_name=format_name,
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        execution_dtype=execution_dtype,
        device=torch.device("cpu"),
    )


def test_int8_sym_calibration_round_trip_bound() -> None:
    x = torch.tensor([[[0.0, -1.0, 0.25, 2.0, 127.03125]]], dtype=torch.float32)
    q, scale = quantize_tokenwise_symmetric(x)
    restored = q.float() * scale.float().unsqueeze(-1)
    assert q.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert int(q.min()) >= -127
    assert torch.all(torch.isfinite(scale))
    assert torch.all(127 * scale.float() >= x.abs().amax(dim=-1))
    assert torch.all(
        (restored - x).abs() <= scale.float().unsqueeze(-1) / 2 + 1e-6
    )


def test_int8_sym_zero_and_fp16_rounding_adversaries() -> None:
    raw_scales = torch.tensor([0.0, 1.0001, 63.999, 65504.0])
    x = torch.stack((raw_scales * 127, -raw_scales * 127), dim=-1)
    q, scale = quantize_tokenwise_symmetric(x)
    restored = q.float() * scale.float().unsqueeze(-1)
    assert torch.equal(restored[0], torch.zeros_like(restored[0]))
    assert torch.all(127 * scale.float() >= x.abs().amax(dim=-1))
    assert torch.all(
        (restored - x).abs() <= scale.float().unsqueeze(-1) / 2 + 1e-6
    )


def test_int8_sym_rejects_nonfinite_or_unrepresentable_scale() -> None:
    with pytest.raises(ValueError, match="finite"):
        quantize_tokenwise_symmetric(torch.tensor([[float("inf")]]))
    with pytest.raises(ValueError, match="FP16 scale"):
        quantize_tokenwise_symmetric(torch.tensor([[65504.0 * 128.0]]))


def test_int8_sym_page_bytes_include_scales() -> None:
    fmt = KVCacheFormat.parse("int8_sym")
    assert (
        fmt.page_size_bytes(block_size=16, num_kv_heads=8, head_dim=128)
        == 33280
    )


@pytest.mark.parametrize(
    "device,backend,capabilities,expected_backend,expected_reason",
    [
        (
            torch.device("cuda"),
            "native",
            KVCacheBackendCapabilities(False, True, True, None),
            "native_int8",
            None,
        ),
        (
            torch.device("cpu"),
            "auto",
            KVCacheBackendCapabilities(
                False, False, True, "native_int8_module_unavailable"
            ),
            "sdpa_dequant",
            "cpu_sdpa_dequant",
        ),
        (
            torch.device("cuda"),
            "auto",
            KVCacheBackendCapabilities(
                False, False, True, "native_int8_binding_missing"
            ),
            "sdpa_dequant",
            "native_int8_binding_missing",
        ),
        (
            torch.device("cuda"),
            "flashinfer",
            KVCacheBackendCapabilities(True, True, True, None),
            "native_int8",
            "flashinfer_no_int8_sym_contract",
        ),
    ],
)
def test_resolver_consumes_device_kernel_and_backend_capability(
    device, backend, capabilities, expected_backend, expected_reason
) -> None:
    decision = resolve_kv_cache_format(
        requested="int8_sym",
        model=KVCacheModelInfo(32, 8, 128, False),
        device=device,
        backend_preference=backend,
        capabilities=capabilities,
        allow_fallback=True,
    )
    assert decision.effective_format.name == "int8_sym"
    assert decision.execution_backend == expected_backend
    assert decision.reason == expected_reason


def test_mla_request_falls_back_visibly() -> None:
    decision = resolve_kv_cache_format(
        requested="int8_sym",
        model=KVCacheModelInfo(
            num_attention_heads=16, num_kv_heads=1, head_dim=512, is_mla=True
        ),
        device=torch.device("cuda"),
        backend_preference="auto",
        capabilities=KVCacheBackendCapabilities(False, True, True, None),
        allow_fallback=True,
    )
    assert decision.effective_format.name == "native"
    assert decision.reason == "mla_not_validated"


def test_unknown_format_fails_before_allocation() -> None:
    with pytest.raises(ValueError, match="unsupported KV cache format"):
        KVCacheFormat.parse("int4")


@pytest.mark.parametrize("format_name", ["native", "int8_sym"])
def test_layered_store_writes_chunk_and_reads_logical_prefix(
    format_name: str,
) -> None:
    store = allocate_layered_paged_kv_store(
        owner_id="unit-store-prefix",
        format_name=format_name,
        num_layers=2,
        num_blocks=3,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        execution_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert store.owner_id == "unit-store-prefix"
    key = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8) / 31
    value = key + 0.5
    store.write_chunk(
        layer_idx=1,
        key_chunk=key,
        value_chunk=value,
        slot_mapping=torch.tensor([0, 5, 11]),
    )
    k_hat, v_hat = store.read_prefix(
        layer_idx=1,
        block_table=torch.tensor([0, 1, 2]),
        seq_len=12,
        execution_dtype=torch.float32,
    )
    tolerance = 0.0 if format_name == "native" else 2e-2
    torch.testing.assert_close(
        k_hat[[0, 5, 11]], key, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        v_hat[[0, 5, 11]], value, atol=tolerance, rtol=tolerance
    )
    assert k_hat.shape == (12, 2, 8)
    assert v_hat.shape == (12, 2, 8)


def test_int8_store_bytes_match_layers_times_pages() -> None:
    store = allocate_layered_paged_kv_store(
        owner_id="unit-store-bytes",
        format_name="int8_sym",
        num_layers=2,
        num_blocks=3,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        execution_dtype=torch.float16,
        device=torch.device("cpu"),
    )
    assert store.owner_id == "unit-store-bytes"
    assert store.nbytes == 2 * 3 * KVCacheFormat.parse(
        "int8_sym"
    ).page_size_bytes(block_size=4, num_kv_heads=2, head_dim=8)


def test_page_chunk_snapshot_preserves_layer_and_page_axes() -> None:
    store = _make_layered_store(
        format_name="int8_sym", num_layers=2, num_blocks=4
    )
    chunk = store.snapshot_page_chunk([1, 3], target_device=torch.device("cpu"))
    assert chunk.page_ids == (1, 3)
    assert chunk.payload.shape[:2] == (2, 2)  # layers, selected pages
    assert chunk.scales is not None and chunk.scales.shape[:2] == (2, 2)
