from types import SimpleNamespace
from typing import Union, cast

import torch

from moe_infinity.engine.kv_cache_offload_coordinator import (
    KVCacheOffloadCoordinator,
)
from moe_infinity.engine.transfer_types import (
    TransferPriority,
    TransferRequest,
    TransferType,
)

KVCachedData = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


class _DummyScheduler:
    def __init__(self) -> None:
        self._handlers: dict[TransferType, object] = {}

    def register_handler(
        self, transfer_type: TransferType, handler: object
    ) -> None:
        self._handlers[transfer_type] = handler


def _request(
    transfer_id: str,
    transfer_type: TransferType,
    block_ids: list[int],
    target_device: str = "cpu",
) -> TransferRequest:
    return TransferRequest(
        transfer_id=transfer_id,
        transfer_type=transfer_type,
        priority=TransferPriority.NORMAL,
        source_device="cuda:0",
        target_device=target_device,
        tensor_id="kv",
        block_ids=block_ids,
    )


def test_handlers_registered_when_enabled() -> None:
    scheduler = _DummyScheduler()
    config = SimpleNamespace(enable_kv_cache_offload=True)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=torch.zeros(2, 4, 3),
        block_pool=None,
        config=config,
    )

    coordinator.register_with_scheduler(scheduler)

    handlers = cast(dict[TransferType, object], getattr(scheduler, "_handlers"))
    assert TransferType.KV_SWAP_OUT in handlers
    assert TransferType.KV_SWAP_IN in handlers


def test_handlers_not_registered_when_disabled() -> None:
    scheduler = _DummyScheduler()
    config = SimpleNamespace(enable_kv_cache_offload=False)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=torch.zeros(2, 4, 3),
        block_pool=None,
        config=config,
    )

    coordinator.register_with_scheduler(scheduler)

    handlers = cast(dict[TransferType, object], getattr(scheduler, "_handlers"))
    assert TransferType.KV_SWAP_OUT not in handlers
    assert TransferType.KV_SWAP_IN not in handlers


def test_handle_swap_out_copies_tensors() -> None:
    kv_tensors = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=kv_tensors,
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    swap_out = _request(
        transfer_id="transfer-1",
        transfer_type=TransferType.KV_SWAP_OUT,
        block_ids=[0, 1],
    )

    coordinator.handle_swap_out(swap_out)

    cpu_cache = cast(
        dict[str, KVCachedData], getattr(coordinator, "_cpu_cache")
    )
    assert "transfer-1" in cpu_cache
    cached = cpu_cache["transfer-1"]
    assert isinstance(cached, torch.Tensor)
    torch.testing.assert_close(cached, kv_tensors[:, [0, 1], ...])


def test_handle_swap_in_restores_tensors() -> None:
    kv_tensors = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    original = kv_tensors.clone()
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=kv_tensors,
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    swap_out = _request(
        transfer_id="transfer-2",
        transfer_type=TransferType.KV_SWAP_OUT,
        block_ids=[0, 1],
    )
    coordinator.handle_swap_out(swap_out)

    kv_tensors[:, [0, 1], ...] = -1.0

    swap_in = _request(
        transfer_id="transfer-2",
        transfer_type=TransferType.KV_SWAP_IN,
        block_ids=[0, 1],
        target_device="cpu",
    )
    coordinator.handle_swap_in(swap_in)

    torch.testing.assert_close(
        kv_tensors[:, [0, 1], ...], original[:, [0, 1], ...]
    )
    cpu_cache = cast(
        dict[str, KVCachedData], getattr(coordinator, "_cpu_cache")
    )
    assert "transfer-2" not in cpu_cache


def test_handle_swap_out_returns_payload_bytes_stacked() -> None:
    kv_tensors = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=kv_tensors,
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    swap_out = _request(
        transfer_id="stacked-bytes",
        transfer_type=TransferType.KV_SWAP_OUT,
        block_ids=[0, 1],
    )

    result_bytes = coordinator.handle_swap_out(swap_out)

    selected = kv_tensors[:, [0, 1], ...]
    assert result_bytes == selected.numel() * selected.element_size()


def test_handle_swap_out_returns_summed_bytes_for_tuple_layout() -> None:
    k_cache = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    v_cache = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=(k_cache, v_cache),
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    swap_out = _request(
        transfer_id="tuple-bytes",
        transfer_type=TransferType.KV_SWAP_OUT,
        block_ids=[0, 2],
    )

    result_bytes = coordinator.handle_swap_out(swap_out)

    k_sel = k_cache[[0, 2], ...]
    v_sel = v_cache[[0, 2], ...]
    expected = (
        k_sel.numel() * k_sel.element_size()
        + v_sel.numel() * v_sel.element_size()
    )
    assert result_bytes == expected


def test_handle_swap_out_missing_kv_tensors_raises() -> None:
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=None,
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    swap_out = _request(
        transfer_id="missing",
        transfer_type=TransferType.KV_SWAP_OUT,
        block_ids=[0],
    )

    try:
        coordinator.handle_swap_out(swap_out)
    except RuntimeError as exc:
        assert "KV tensors are not initialized" in str(exc)
    else:
        raise AssertionError("missing KV tensors must raise, not silently pass")


def test_handle_swap_in_missing_host_copy_raises() -> None:
    kv_tensors = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=kv_tensors,
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    swap_in = _request(
        transfer_id="missing",
        transfer_type=TransferType.KV_SWAP_IN,
        block_ids=[0, 1],
        target_device="cpu",
    )

    try:
        coordinator.handle_swap_in(swap_in)
    except RuntimeError as exc:
        assert "missing host KV for transfer missing" in str(exc)
    else:
        raise AssertionError("missing host copy must raise, not silently pass")


def test_handle_swap_in_returns_payload_bytes() -> None:
    kv_tensors = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    coordinator = KVCacheOffloadCoordinator(
        kv_tensors=kv_tensors,
        block_pool=None,
        config=SimpleNamespace(enable_kv_cache_offload=True),
    )

    coordinator.handle_swap_out(
        _request(
            transfer_id="rt",
            transfer_type=TransferType.KV_SWAP_OUT,
            block_ids=[0, 1],
        )
    )
    restored_bytes = coordinator.handle_swap_in(
        _request(
            transfer_id="rt",
            transfer_type=TransferType.KV_SWAP_IN,
            block_ids=[0, 1],
            target_device="cpu",
        )
    )

    selected = kv_tensors[:, [0, 1], ...]
    assert restored_bytes == selected.numel() * selected.element_size()
