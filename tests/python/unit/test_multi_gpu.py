"""Tests for multi-GPU support in the Python layer.

All tests mock CUDA so they run on any machine, including CI without GPUs.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# MemoryCoordinator multi-GPU
# ---------------------------------------------------------------------------


class TestMemoryCoordinatorMultiGPU:
    @staticmethod
    def _cls():
        mod = importlib.import_module("moe_infinity.memory.memory_coordinator")
        return getattr(mod, "MemoryCoordinator")

    def test_total_gpu_memory_bytes_per_device(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.75, kv_cache_memory_ratio=0.15)

        gpu_memories = {0: 24 * 1024**3, 1: 48 * 1024**3}

        def fake_props(device_id):
            m = MagicMock()
            m.total_memory = gpu_memories[device_id]
            return m

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_device_properties", side_effect=fake_props),
        ):
            assert mc.total_gpu_memory_bytes(0) == 24 * 1024**3
            assert mc.total_gpu_memory_bytes(1) == 48 * 1024**3

    def test_total_gpu_memory_bytes_default_is_device_0(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.75, kv_cache_memory_ratio=0.15)

        def fake_props(device_id):
            m = MagicMock()
            m.total_memory = (device_id + 1) * 16 * 1024**3
            return m

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_device_properties", side_effect=fake_props),
        ):
            assert mc.total_gpu_memory_bytes() == mc.total_gpu_memory_bytes(0)

    def test_aggregate_gpu_memory_bytes(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.75, kv_cache_memory_ratio=0.15)

        mem_per_gpu = 24 * 1024**3

        def fake_props(device_id):
            m = MagicMock()
            m.total_memory = mem_per_gpu
            return m

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=4),
            patch("torch.cuda.get_device_properties", side_effect=fake_props),
        ):
            assert mc.aggregate_gpu_memory_bytes() == 4 * mem_per_gpu

    def test_expert_cache_bytes_per_device(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.5, kv_cache_memory_ratio=0.1)

        def fake_props(device_id):
            m = MagicMock()
            m.total_memory = (device_id + 1) * 10 * 1024**3
            return m

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_device_properties", side_effect=fake_props),
        ):
            assert mc.expert_cache_bytes(0) == int(10 * 1024**3 * 0.5)
            assert mc.expert_cache_bytes(1) == int(20 * 1024**3 * 0.5)

    def test_expert_cache_bytes_total(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.5, kv_cache_memory_ratio=0.1)

        def fake_props(device_id):
            m = MagicMock()
            m.total_memory = 24 * 1024**3
            return m

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=3),
            patch("torch.cuda.get_device_properties", side_effect=fake_props),
        ):
            per_gpu = int(24 * 1024**3 * 0.5)
            assert mc.expert_cache_bytes_total() == 3 * per_gpu

    def test_kv_cache_bytes_per_device(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.5, kv_cache_memory_ratio=0.2)

        def fake_props(device_id):
            m = MagicMock()
            m.total_memory = 32 * 1024**3
            return m

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_device_properties", side_effect=fake_props),
        ):
            expected = int(32 * 1024**3 * 0.2)
            assert mc.kv_cache_bytes(0) == expected
            assert mc.kv_cache_bytes(1) == expected

    def test_remaining_bytes_per_device(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.6, kv_cache_memory_ratio=0.2)

        def fake_props(device_id):
            m = MagicMock()
            m.total_memory = 100 * 1024**3
            return m

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", side_effect=fake_props),
        ):
            total = 100 * 1024**3
            used = int(total * 0.6) + int(total * 0.2)
            assert mc.remaining_bytes(0) == max(0, total - used)

    def test_num_gpu_devices(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.75, kv_cache_memory_ratio=0.15)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=8),
        ):
            assert mc.num_gpu_devices() == 8

    def test_num_gpu_devices_no_cuda(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.75, kv_cache_memory_ratio=0.15)

        with patch("torch.cuda.is_available", return_value=False):
            assert mc.num_gpu_devices() == 0

    def test_budget_status_includes_multi_gpu_fields(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.75, kv_cache_memory_ratio=0.15)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            status = mc.get_budget_status()
            assert "num_gpu_devices" in status
            assert "aggregate_gpu_bytes" in status
            assert status["num_gpu_devices"] == 2

    def test_out_of_range_device_id_returns_fallback(self):
        MC = self._cls()
        mc = MC(device_memory_ratio=0.75, kv_cache_memory_ratio=0.15)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            result = mc.total_gpu_memory_bytes(99)
            assert result == 24 * 1024**3


# ---------------------------------------------------------------------------
# ExpertOffloadCoordinator multi-GPU
# ---------------------------------------------------------------------------


class TestExpertOffloadCoordinatorMultiGPU:
    @staticmethod
    def _cls():
        mod = importlib.import_module(
            "moe_infinity.engine.expert_offload_coordinator"
        )
        return getattr(mod, "ExpertOffloadCoordinator")

    @staticmethod
    def _transfer_type_cls():
        mod = importlib.import_module("moe_infinity.engine.transfer_types")
        return getattr(mod, "TransferType")

    def test_target_device_round_robin(self):
        Coordinator = self._cls()
        coord = Coordinator(num_devices=4)
        assert coord._target_device_for_expert(0) == "cuda:0"
        assert coord._target_device_for_expert(1) == "cuda:1"
        assert coord._target_device_for_expert(2) == "cuda:2"
        assert coord._target_device_for_expert(3) == "cuda:3"
        assert coord._target_device_for_expert(4) == "cuda:0"
        assert coord._target_device_for_expert(7) == "cuda:3"

    def test_single_device_always_cuda_0(self):
        Coordinator = self._cls()
        coord = Coordinator(num_devices=1)
        for i in range(10):
            assert coord._target_device_for_expert(i) == "cuda:0"

    def test_prefetch_uses_correct_target_device(self):
        Coordinator = self._cls()
        TransferType = self._transfer_type_cls()

        coord = Coordinator(num_devices=2)

        mock_scheduler = MagicMock()
        mock_scheduler.register_handler = MagicMock()
        mock_scheduler.enqueue = MagicMock(return_value="req-1")
        coord.register_with_scheduler(mock_scheduler)

        coord.prefetch_experts(layer_id=3, expert_ids=[5])

        call_args = mock_scheduler.enqueue.call_args[0][0]
        assert call_args.target_device == "cuda:1"
        assert call_args.source_device == "cpu"
        assert call_args.transfer_type == TransferType.EXPERT_FETCH

    def test_prefetch_empty_expert_ids_defaults_cuda_0(self):
        Coordinator = self._cls()
        coord = Coordinator(num_devices=4)

        mock_scheduler = MagicMock()
        mock_scheduler.register_handler = MagicMock()
        mock_scheduler.enqueue = MagicMock(return_value="req-1")
        coord.register_with_scheduler(mock_scheduler)

        coord.prefetch_experts(layer_id=0, expert_ids=[])

        call_args = mock_scheduler.enqueue.call_args[0][0]
        assert call_args.target_device == "cuda:0"

    def test_default_num_devices_is_1(self):
        Coordinator = self._cls()
        coord = Coordinator()
        assert coord._num_devices == 1


# ---------------------------------------------------------------------------
# dispatch_local multi-GPU distribution
# ---------------------------------------------------------------------------


class TestDispatchLocalMultiGPU:
    @staticmethod
    def _cls():
        mod = importlib.import_module(
            "moe_infinity.distributed.expert_executor"
        )
        return getattr(mod, "DistributedExpertExecutor")

    def test_dispatch_local_distributes_across_gpus(self):
        import numpy as np
        import torch

        Executor = self._cls()

        mock_config = MagicMock()
        executor = Executor(archer_config=mock_config)

        mock_dispatcher = MagicMock()
        enqueued = []

        def capture_enqueue(layer_id, expert_id, gpu_id, remote, phase):
            enqueued.append((layer_id, expert_id, gpu_id, remote))

        mock_dispatcher.enqueue_expert = capture_enqueue
        mock_dispatcher.set_inputs = MagicMock()
        mock_dispatcher.set_expected_queue = MagicMock()
        mock_dispatcher.notify_fetch_start = MagicMock()
        executor.set_expert_dispatcher(mock_dispatcher)

        num_tokens = 4
        num_experts = 8
        hidden_dim = 16
        hidden_states = torch.randn(num_tokens, hidden_dim)

        router_mask = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
        router_mask[0, 0] = True
        router_mask[1, 3] = True
        router_mask[2, 5] = True
        router_mask[3, 7] = True

        router_weights = torch.zeros(
            num_tokens, num_experts, dtype=torch.float32
        )
        router_weights[0, 0] = 1.0
        router_weights[1, 3] = 1.0
        router_weights[2, 5] = 1.0
        router_weights[3, 7] = 1.0

        with patch("torch.cuda.device_count", return_value=4):
            executor.dispatch_local(
                0, hidden_states, router_mask, router_weights
            )

        gpu_ids = {e[1]: e[2] for e in enqueued}
        assert gpu_ids[0] == 0 % 4
        assert gpu_ids[3] == 3 % 4
        assert gpu_ids[5] == 5 % 4
        assert gpu_ids[7] == 7 % 4

    def test_dispatch_local_single_gpu(self):
        import torch

        Executor = self._cls()

        mock_config = MagicMock()
        executor = Executor(archer_config=mock_config)

        mock_dispatcher = MagicMock()
        enqueued = []

        def capture_enqueue(layer_id, expert_id, gpu_id, remote, phase):
            enqueued.append((layer_id, expert_id, gpu_id, remote))

        mock_dispatcher.enqueue_expert = capture_enqueue
        mock_dispatcher.set_inputs = MagicMock()
        mock_dispatcher.set_expected_queue = MagicMock()
        mock_dispatcher.notify_fetch_start = MagicMock()
        executor.set_expert_dispatcher(mock_dispatcher)

        num_tokens = 2
        num_experts = 8
        hidden_dim = 16
        hidden_states = torch.randn(num_tokens, hidden_dim)

        router_mask = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
        router_mask[0, 2] = True
        router_mask[1, 6] = True

        router_weights = torch.zeros(
            num_tokens, num_experts, dtype=torch.float32
        )
        router_weights[0, 2] = 1.0
        router_weights[1, 6] = 1.0

        with patch("torch.cuda.device_count", return_value=1):
            executor.dispatch_local(
                0, hidden_states, router_mask, router_weights
            )

        for _, _, gpu_id, _ in enqueued:
            assert gpu_id == 0


# ---------------------------------------------------------------------------
# device utility multi-GPU
# ---------------------------------------------------------------------------


class TestDeviceUtilsMultiGPU:
    def test_get_device_returns_correct_cuda_device(self):
        from moe_infinity.utils.device import get_device

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=4),
        ):
            assert get_device(0) == "cuda:0"
            assert get_device(1) == "cuda:1"
            assert get_device(2) == "cuda:2"
            assert get_device(3) == "cuda:3"

    def test_get_device_out_of_range_returns_cpu(self):
        from moe_infinity.utils.device import get_device

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            assert get_device(2) == "cpu"
            assert get_device(99) == "cpu"

    def test_get_num_devices_multi(self):
        from moe_infinity.utils.device import get_num_devices

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=8),
        ):
            assert get_num_devices() == 8


# ---------------------------------------------------------------------------
# ArcherConfig multi-GPU
# ---------------------------------------------------------------------------


class TestArcherConfigMultiGPU:
    def test_device_per_node_reflects_gpu_count(self):
        from moe_infinity.utils.config import ArcherConfig

        with patch("torch.cuda.device_count", return_value=4):
            config = ArcherConfig.load_from_json(
                {"offload_path": "/tmp/test", "device_memory_ratio": 0.75}
            )
            assert config.device_per_node == 4
