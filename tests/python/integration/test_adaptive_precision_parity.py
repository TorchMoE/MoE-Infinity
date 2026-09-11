import torch

from moe_infinity.runtime.expert_quality import (
    build_fp8_expert_variant,
    run_bf16_expert,
    run_fp8_expert,
    validate_fp8_expert_variant,
)


def test_fp8_expert_variant_meets_tensor_and_forward_gates():
    torch.manual_seed(7)
    source = tuple(
        torch.randn(256, 256, dtype=torch.bfloat16) / 8 for _ in range(3)
    )
    x = torch.randn(8, 256, dtype=torch.bfloat16) / 4
    variant = build_fp8_expert_variant(source)
    validate_fp8_expert_variant(source, variant)
    torch.testing.assert_close(
        run_fp8_expert(x, variant),
        run_bf16_expert(x, source),
        rtol=0.08,
        atol=0.08,
    )
