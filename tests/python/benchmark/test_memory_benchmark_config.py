from argparse import Namespace
from unittest.mock import Mock

from benchmarks.serving import memory


def test_arm_ratios_and_feature_flag_reach_model_load(monkeypatch) -> None:
    seen: list[dict[str, object]] = []

    class FakeMoE:
        def __init__(self, model_name: str, config: dict[str, object]) -> None:
            _ = model_name
            seen.append(dict(config))
            self.model = Mock(config=Mock())
            self.engine = Mock(
                config=Mock(
                    device_memory_ratio=config["device_memory_ratio"],
                    kv_cache_ratio=config["kv_cache_ratio"],
                    adaptive_memory_enabled=config["adaptive_memory_enabled"],
                )
            )

    monkeypatch.setattr(memory, "_moe_class", lambda: FakeMoE)
    monkeypatch.setattr(
        memory, "_load_tokenizer", lambda _: Mock(pad_token_id=0)
    )
    arm = memory.ArmConfig("adaptive", 0.61, 0.37, True)
    model, _ = memory.load_model_and_tokenizer("m", "/offload", arm)
    assert seen == [
        {
            "offload_path": "/offload",
            "device_memory_ratio": 0.61,
            "kv_cache_ratio": 0.37,
            "adaptive_memory_enabled": True,
        }
    ]
    assert memory.effective_arm_config(model, arm) == {
        "arm": "adaptive",
        "device_memory_ratio": 0.61,
        "kv_cache_ratio": 0.37,
        "adaptive_memory_enabled": True,
    }


def test_report_separates_requested_and_effective_config(monkeypatch) -> None:
    arms = [
        memory.ArmConfig("fixed", 0.55, 0.20, False),
        memory.ArmConfig("adaptive", 0.55, 0.20, True),
    ]
    monkeypatch.setattr(
        memory,
        "run_arm",
        lambda arm, args: {
            "requested_config": arm.as_dict(),
            "effective_config": {**arm.as_dict(), "kv_cache_ratio": 0.19},
            "output_token_ids": [1, 2],
            "safety": {"violations": 0},
        },
    )
    report = memory.compare_arms(arms, Namespace(seed=7))
    assert report["arms"][0]["requested_config"]["kv_cache_ratio"] == 0.20
    assert report["arms"][0]["effective_config"]["kv_cache_ratio"] == 0.19
    assert (
        report["arms"][1]["effective_config"]["adaptive_memory_enabled"] is True
    )
