"""CPU-only smoke tests for moe_infinity/distributed.

These tests do not exercise the full expert-dispatch path (which requires
CUDA, a native ExpertDispatcher, and torch.distributed). They cover the
Python-side constructors, state setters, delegation helpers, and public
aliases so that refactors to the distributed module have a safety net in
CI.

Anything that touches torch.cuda, torch.distributed, or the native
engine extension should go in a GPU-gated test instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from transformers import PretrainedConfig

from moe_infinity.distributed import (
    DistributedExpertExecutor,
    DistributedExpertPrefetcher,
)
from moe_infinity.distributed.expert_executor import ExpertExecutor
from moe_infinity.memory.expert_policy import ExpertPhase
from moe_infinity.utils import ArcherConfig


def _make_archer_config() -> ArcherConfig:
    return ArcherConfig.load_from_json(
        {
            "offload_path": "/tmp/moe-infinity-smoke",
            "trace_capacity": 16,
            "prefetch": False,
        }
    )


def _make_moe_config() -> PretrainedConfig:
    config = PretrainedConfig()
    config.architectures = ["MixtralForCausalLM"]
    config.num_hidden_layers = 4
    config.num_local_experts = 8
    return config


class TestDistributedExpertExecutor:
    def test_construct_with_archer_config(self):
        config = _make_archer_config()
        executor = DistributedExpertExecutor(config)

        assert executor.archer_config is config
        assert executor.expert_dispatcher is None
        assert executor.device_map_manager is None
        assert executor.prefetcher is None

    def test_set_expert_dispatcher_stores_reference(self):
        executor = DistributedExpertExecutor(_make_archer_config())
        dispatcher = MagicMock(name="ExpertDispatcher")

        executor.set_expert_dispatcher(dispatcher)

        assert executor.expert_dispatcher is dispatcher

    def test_set_device_map_manager_stores_reference(self):
        executor = DistributedExpertExecutor(_make_archer_config())
        manager = MagicMock(name="DeviceMapManager")

        executor.set_device_map_manager(manager)

        assert executor.device_map_manager is manager

    def test_set_prefetcher_stores_reference(self):
        executor = DistributedExpertExecutor(_make_archer_config())
        prefetcher = MagicMock(name="ExpertPrefetcher")

        executor.set_prefetcher(prefetcher)

        assert executor.prefetcher is prefetcher

    def test_trigger_speculative_prefetch_with_prefetcher(self):
        executor = DistributedExpertExecutor(_make_archer_config())
        prefetcher = MagicMock(name="ExpertPrefetcher")
        executor.set_prefetcher(prefetcher)
        router_logits = MagicMock(name="router_logits")

        executor.trigger_speculative_prefetch(
            layer_id=3, router_logits=router_logits
        )

        prefetcher.speculative_prefetch.assert_called_once_with(
            3, router_logits, phase=ExpertPhase.MIXED
        )

    def test_trigger_speculative_prefetch_without_prefetcher_is_noop(self):
        executor = DistributedExpertExecutor(_make_archer_config())

        # Must not raise even though no prefetcher has been set.
        executor.trigger_speculative_prefetch(layer_id=0, router_logits=None)

    def test_expert_executor_alias_points_to_distributed_class(self):
        assert ExpertExecutor is DistributedExpertExecutor


class TestDistributedExpertPrefetcher:
    def test_construct_parses_moe_params(self):
        config = _make_moe_config()
        prefetcher = DistributedExpertPrefetcher(config)

        assert prefetcher.num_layers == 4
        assert prefetcher.num_experts == 8
        assert prefetcher.num_encoder_layers == 0

    def test_set_archer_engine_stores_reference(self):
        prefetcher = DistributedExpertPrefetcher(_make_moe_config())
        archer_engine = MagicMock(name="ArcherEngine")

        prefetcher.set_archer_engine(archer_engine)

        assert prefetcher.archer_engine is archer_engine

    def test_set_device_map_manager_stores_reference(self):
        prefetcher = DistributedExpertPrefetcher(_make_moe_config())
        manager = MagicMock(name="DeviceMapManager")

        prefetcher.set_device_map_manager(manager)

        assert prefetcher.device_map_manager is manager

    def test_set_archer_prefetch_stores_reference(self):
        prefetcher = DistributedExpertPrefetcher(_make_moe_config())
        archer_prefetch = MagicMock(name="ArcherPrefetch")

        prefetcher.set_archer_prefetch(archer_prefetch)

        assert prefetcher.archer_prefetch is archer_prefetch


class TestDistributedPackageExports:
    def test_package_exposes_executor_and_prefetcher(self):
        import moe_infinity.distributed as distributed

        assert (
            distributed.DistributedExpertExecutor is DistributedExpertExecutor
        )
        assert (
            distributed.DistributedExpertPrefetcher
            is DistributedExpertPrefetcher
        )
