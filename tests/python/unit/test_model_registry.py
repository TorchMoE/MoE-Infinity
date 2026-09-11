import warnings
from types import SimpleNamespace
from typing import Any, cast


def test_all_models_registered():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.common.constants import (
        MODEL_MAPPING_NAMES,
        MODEL_MAPPING_TYPES,
    )

    expected_models = {
        "nllb",
        "mixtral",
        "opt",
        "deepseek",
        "deepseek_v3",
        "gptoss",
        "qwen3",
        "dbrx",
        "olmoe",
        "jamba",
    }
    if "deepseekv4" in MODEL_MAPPING_NAMES:
        expected_models.add("deepseekv4")
    if "qwen3_5" in MODEL_MAPPING_NAMES:
        expected_models.add("qwen3_5")
    if "glmmoedsa" in MODEL_MAPPING_NAMES:
        expected_models.add("glmmoedsa")
    if "glm5next" in MODEL_MAPPING_NAMES:
        expected_models.add("glm5next")
    actual_models = set(MODEL_MAPPING_NAMES.keys())
    assert expected_models == actual_models, (
        f"Missing: {expected_models - actual_models}, "
        f"Extra: {actual_models - expected_models}"
    )

    for key, cls in MODEL_MAPPING_NAMES.items():
        assert cls is not None, f"MODEL_MAPPING_NAMES['{key}'] is None"
        assert hasattr(
            cls, "__name__"
        ), f"MODEL_MAPPING_NAMES['{key}'] has no __name__"


def test_model_types_complete():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.common.constants import (
        MODEL_MAPPING_NAMES,
        MODEL_MAPPING_TYPES,
    )

    for key in MODEL_MAPPING_NAMES:
        assert (
            key in MODEL_MAPPING_TYPES
        ), f"'{key}' missing from MODEL_MAPPING_TYPES"
        assert isinstance(
            MODEL_MAPPING_TYPES[key], int
        ), f"MODEL_MAPPING_TYPES['{key}'] is not int"


def test_parse_expert_type_new_models():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.common.constants import parse_expert_type

    for arch_prefix, expected_type in [
        ("DbrxForCausalLM", 4),
        ("OlmoeForCausalLM", 4),
        ("JambaForCausalLM", 4),
    ]:
        config = SimpleNamespace(architectures=[arch_prefix])
        result = parse_expert_type(cast(Any, config))
        assert (
            result == expected_type
        ), f"{arch_prefix}: expected {expected_type}, got {result}"


def test_deepseek_uses_hf_classes():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from moe_infinity.common.constants import MODEL_MAPPING_NAMES

    deepseek_cls = MODEL_MAPPING_NAMES["deepseek"]
    assert (
        "transformers" in deepseek_cls.__module__
    ), f"deepseek should use HF class, got module: {deepseek_cls.__module__}"

    deepseek_v3_cls = MODEL_MAPPING_NAMES["deepseek_v3"]
    assert "transformers" in deepseek_v3_cls.__module__, (
        "deepseek_v3 should use HF class, "
        f"got module: {deepseek_v3_cls.__module__}"
    )
