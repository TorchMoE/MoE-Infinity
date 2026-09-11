import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def test_import_device_utils():
    from moe_infinity.utils.device import (
        DeviceConfig,
        get_default_device,
        get_device,
        get_num_devices,
        get_pinned_memory_device,
        is_cuda_available,
        to_device,
    )

    assert get_default_device is not None
    assert get_device is not None
    assert get_num_devices is not None
    assert get_pinned_memory_device is not None
    assert is_cuda_available is not None
    assert to_device is not None
    assert DeviceConfig is not None


def test_is_cuda_available_returns_bool():
    from moe_infinity.utils.device import is_cuda_available

    result = is_cuda_available()
    assert isinstance(result, bool)


def test_get_num_devices_returns_nonneg_int():
    from moe_infinity.utils.device import get_num_devices

    n = get_num_devices()
    assert isinstance(n, int)
    assert n >= 0


def test_get_default_device_returns_valid_string():
    from moe_infinity.utils.device import get_default_device

    device = get_default_device()
    assert isinstance(device, str)
    assert device.startswith("cuda") or device == "cpu"


def test_get_default_device_cpu_when_no_cuda():
    from moe_infinity.utils.device import get_default_device

    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.cuda.device_count", return_value=0
    ):
        device = get_default_device()
        assert device == "cpu"


def test_get_device_with_int_id():
    from moe_infinity.utils.device import get_device

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=4
    ):
        device = get_device(0)
        assert device == "cuda:0"
        device = get_device(2)
        assert device == "cuda:2"


def test_get_device_none_returns_default():
    from moe_infinity.utils.device import get_device

    device = get_device(None)
    assert isinstance(device, str)


def test_get_device_out_of_range_returns_cpu():
    from moe_infinity.utils.device import get_device

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=2
    ):
        device = get_device(5)
        assert device == "cpu"


def test_get_pinned_memory_device_always_cpu():
    from moe_infinity.utils.device import get_pinned_memory_device

    assert get_pinned_memory_device() == "cpu"


def test_to_device_moves_tensor():
    from moe_infinity.utils.device import to_device

    t = torch.zeros(3, 4)
    result = to_device(t, "cpu")
    assert result.device.type == "cpu"
    assert result.shape == (3, 4)


def test_device_config_has_required_fields():
    from moe_infinity.utils.device import DeviceConfig

    config = DeviceConfig(
        default_device="cpu", offload_device="cpu", num_devices=0
    )
    assert config.default_device == "cpu"
    assert config.offload_device == "cpu"
    assert config.num_devices == 0


def test_device_config_offload_always_cpu():
    from moe_infinity.utils.device import DeviceConfig

    config = DeviceConfig(
        default_device="cuda:0", offload_device="cpu", num_devices=1
    )
    assert config.offload_device == "cpu"
