import pytest
import torch

from moe_infinity.kernel.marlin_gemm import marlin_supports_shape
from moe_infinity.utils.fp8 import dequant_fp8_blockwise, quant_fp8_blockwise


def test_fp8_blockwise_roundtrip_is_deterministic_and_owns_scale():
    torch.manual_seed(7)
    weight = torch.randn(257, 513, dtype=torch.bfloat16)
    q1, s1 = quant_fp8_blockwise(weight, block_size=128)
    q2, s2 = quant_fp8_blockwise(weight, block_size=128)
    assert q1.dtype == torch.float8_e4m3fn
    assert s1.dtype == torch.float32
    assert q1.shape == weight.shape
    assert s1.shape == (3, 5)
    assert torch.equal(q1.view(torch.uint8), q2.view(torch.uint8))
    assert torch.equal(s1, s2)
    restored = dequant_fp8_blockwise(q1, s1, dtype=torch.bfloat16)
    assert torch.isfinite(restored).all()


@pytest.mark.parametrize(
    ("k", "n", "available", "expected"),
    [
        (256, 512, True, True),
        (255, 512, True, False),
        (256, 511, True, False),
        (256, 512, False, False),
    ],
)
def test_marlin_capability_requires_extension_and_layout(
    k, n, available, expected
):
    assert (
        marlin_supports_shape(
            k, n, groupsize=128, extension_available=available
        )
        is expected
    )
