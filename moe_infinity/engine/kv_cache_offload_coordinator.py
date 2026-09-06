from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Union, cast

import torch

from moe_infinity.engine.kv_transfer import (
    CopyTicket,
    KVTransferBackend,
    SyncKVTransferBackend,
)
from moe_infinity.engine.transfer_types import TransferRequest, TransferType

KVTensors = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]


class _SchedulerLike(Protocol):
    def register_handler(
        self, transfer_type: TransferType, handler: object
    ) -> None: ...


class KVCacheOffloadCoordinator:
    def __init__(
        self,
        kv_tensors: KVTensors | None,
        block_pool: object,
        config: Mapping[str, object] | object | None,
        transfer_backend: KVTransferBackend | None = None,
    ) -> None:
        self._kv_tensors: KVTensors | None = kv_tensors
        self._block_pool: object = block_pool
        self._config: Mapping[str, object] | object | None = config
        self._transfer_backend = transfer_backend or SyncKVTransferBackend()
        self._transfer_scheduler: _SchedulerLike | None = None
        self._cpu_cache: dict[str, KVTensors] = {}

    def set_kv_tensors(self, kv_tensors: KVTensors) -> None:
        self._kv_tensors = kv_tensors

    def register_with_scheduler(self, scheduler: _SchedulerLike) -> None:
        self._transfer_scheduler = scheduler
        if not self._is_enabled():
            return
        scheduler.register_handler(
            TransferType.KV_SWAP_OUT, self.handle_swap_out
        )
        scheduler.register_handler(TransferType.KV_SWAP_IN, self.handle_swap_in)

    def handle_swap_out(self, request: TransferRequest) -> int:
        if self._kv_tensors is None:
            raise RuntimeError("KV tensors are not initialized")

        block_ids = list(request.block_ids)
        tickets: list[CopyTicket] = []

        if isinstance(self._kv_tensors, tuple):
            k_cache, v_cache = self._kv_tensors
            k_blocks = self._make_host_destination(k_cache, block_ids, 0)
            v_blocks = self._make_host_destination(v_cache, block_ids, 0)
            cpu_data: KVTensors = (k_blocks, v_blocks)
            try:
                tickets.append(
                    self._transfer_backend.submit_d2h(
                        k_cache, k_blocks, block_ids=block_ids, block_dim=0
                    )
                )
                tickets.append(
                    self._transfer_backend.submit_d2h(
                        v_cache, v_blocks, block_ids=block_ids, block_dim=0
                    )
                )
            except Exception:
                self._retire_tickets(tickets)
                raise
        else:
            cpu_data = self._make_host_destination(
                self._kv_tensors, block_ids, 1
            )
            tickets.append(
                self._transfer_backend.submit_d2h(
                    self._kv_tensors,
                    cpu_data,
                    block_ids=block_ids,
                    block_dim=1,
                )
            )

        bytes_transferred = self._retire_tickets(tickets)
        self._cpu_cache[request.transfer_id] = cpu_data
        return bytes_transferred

    def handle_swap_in(self, request: TransferRequest) -> int:
        if self._kv_tensors is None:
            raise RuntimeError("KV tensors are not initialized")

        cpu_data = self._cpu_cache.get(request.transfer_id)
        if cpu_data is None:
            raise RuntimeError(
                f"missing host KV for transfer {request.transfer_id}"
            )

        block_ids = list(request.block_ids)
        tickets: list[CopyTicket] = []

        if isinstance(cpu_data, tuple):
            if not isinstance(self._kv_tensors, tuple):
                raise RuntimeError("host and device KV layouts do not match")
            k_blocks_cpu, v_blocks_cpu = cpu_data
            k_cache, v_cache = self._kv_tensors
            try:
                tickets.append(
                    self._transfer_backend.submit_h2d(
                        k_blocks_cpu,
                        k_cache,
                        block_ids=block_ids,
                        block_dim=0,
                    )
                )
                tickets.append(
                    self._transfer_backend.submit_h2d(
                        v_blocks_cpu,
                        v_cache,
                        block_ids=block_ids,
                        block_dim=0,
                    )
                )
            except Exception:
                self._retire_tickets(tickets)
                raise
        else:
            if isinstance(self._kv_tensors, tuple):
                raise RuntimeError("host and device KV layouts do not match")

            tickets.append(
                self._transfer_backend.submit_h2d(
                    cpu_data,
                    self._kv_tensors,
                    block_ids=block_ids,
                    block_dim=1,
                )
            )

        bytes_transferred = self._retire_tickets(tickets)
        del self._cpu_cache[request.transfer_id]
        return bytes_transferred

    def _is_enabled(self) -> bool:
        if isinstance(self._config, Mapping):
            config_map = cast(Mapping[str, object], self._config)
            return bool(config_map.get("enable_kv_cache_offload", False))
        return bool(getattr(self._config, "enable_kv_cache_offload", False))

    def _make_host_destination(
        self, source: torch.Tensor, block_ids: list[int], block_dim: int
    ) -> torch.Tensor:
        shape = list(source.shape)
        shape[block_dim] = len(block_ids)
        return torch.empty(
            tuple(shape),
            dtype=source.dtype,
            device="cpu",
            pin_memory=self._transfer_backend.asynchronous,
        )

    @staticmethod
    def _retire_tickets(tickets: list[CopyTicket]) -> int:
        for ticket in tickets:
            if not ticket.retire(synchronize=True):
                raise RuntimeError("KV copy ticket did not retire")
        return sum(ticket.nbytes for ticket in tickets)


__all__ = ["KVCacheOffloadCoordinator"]
