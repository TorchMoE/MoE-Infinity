import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_round_trip_fp16():
    from moe_infinity.utils.async_transfer import (
        async_d2h,
        async_h2d,
        wait_transfer,
    )

    stream = torch.cuda.Stream()
    original = torch.randn(64, 128, dtype=torch.float16, device="cuda:0")
    cpu_tensor = async_d2h(original, stream)
    wait_transfer(stream)
    restored = async_h2d(cpu_tensor, torch.device("cuda:0"), stream)
    wait_transfer(stream)

    assert torch.allclose(original, restored)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_round_trip_bf16():
    from moe_infinity.utils.async_transfer import (
        async_d2h,
        async_h2d,
        wait_transfer,
    )

    stream = torch.cuda.Stream()
    original = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda:0")
    cpu_tensor = async_d2h(original, stream)
    wait_transfer(stream)
    restored = async_h2d(cpu_tensor, torch.device("cuda:0"), stream)
    wait_transfer(stream)

    assert torch.allclose(original, restored)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_dtype_preserved_d2h():
    from moe_infinity.utils.async_transfer import async_d2h, wait_transfer

    stream = torch.cuda.Stream()
    original = torch.randn(8, 16, dtype=torch.float16, device="cuda:0")
    cpu_tensor = async_d2h(original, stream)
    wait_transfer(stream)

    assert cpu_tensor.dtype == original.dtype


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_dtype_preserved_h2d():
    from moe_infinity.utils.async_transfer import (
        async_d2h,
        async_h2d,
        wait_transfer,
    )

    stream = torch.cuda.Stream()
    original = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda:0")
    cpu_tensor = async_d2h(original, stream)
    wait_transfer(stream)
    restored = async_h2d(cpu_tensor, torch.device("cuda:0"), stream)
    wait_transfer(stream)

    assert restored.dtype == original.dtype
