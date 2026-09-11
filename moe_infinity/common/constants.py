from transformers import (
    DbrxForCausalLM,
    DeepseekV2ForCausalLM,
    DeepseekV3ForCausalLM,
    GptOssForCausalLM,
    JambaForCausalLM,
    MixtralForCausalLM,
    NllbMoeForConditionalGeneration,
    OlmoeForCausalLM,
    OPTForCausalLM,
    PretrainedConfig,
    Qwen3MoeForCausalLM,
)

try:
    from transformers import DeepseekV4ForCausalLM
except ImportError:
    DeepseekV4ForCausalLM = None

try:
    from transformers import Qwen3_5MoeForConditionalGeneration
except ImportError:
    Qwen3_5MoeForConditionalGeneration = None

try:
    from transformers import GlmMoeDsaForCausalLM
except ImportError:
    GlmMoeDsaForCausalLM = None

try:
    from transformers import Glm5NextForConditionalGeneration
except ImportError:
    Glm5NextForConditionalGeneration = None

MODEL_MAPPING_NAMES = {
    "nllb": NllbMoeForConditionalGeneration,
    "mixtral": MixtralForCausalLM,
    "opt": OPTForCausalLM,
    "deepseek_v3": DeepseekV3ForCausalLM,
    "deepseek": DeepseekV2ForCausalLM,
    "gptoss": GptOssForCausalLM,
    "qwen3": Qwen3MoeForCausalLM,
    "dbrx": DbrxForCausalLM,
    "olmoe": OlmoeForCausalLM,
    "jamba": JambaForCausalLM,
}

MODEL_MAPPING_TYPES = {
    "nllb": 2,
    "mixtral": 4,
    "opt": 3,
    "deepseek_v3": 5,
    "deepseek": 5,
    "gptoss": 6,
    "qwen3": 5,
    "dbrx": 4,
    "olmoe": 4,
    "jamba": 4,
}

# DeepSeek-V4 support depends on a transformers build that ships
# DeepseekV4ForCausalLM. Only register it when the class is importable so the
# registry never maps an architecture to a None class.
if DeepseekV4ForCausalLM is not None:
    MODEL_MAPPING_NAMES["deepseekv4"] = DeepseekV4ForCausalLM
    MODEL_MAPPING_TYPES["deepseekv4"] = 5

# Qwen3.5-MoE (arch "Qwen3_5MoeForConditionalGeneration") uses per-expert
# gate_proj/up_proj/down_proj weights (expert-type 5, like Qwen3/DeepSeek); the
# v5 packed checkpoint tensors are expanded to per-expert on load. Registered
# only when the HF class is importable (mirrors the V4 guard above).
if Qwen3_5MoeForConditionalGeneration is not None:
    MODEL_MAPPING_NAMES["qwen3_5"] = Qwen3_5MoeForConditionalGeneration
    MODEL_MAPPING_TYPES["qwen3_5"] = 5

# GLM-MoE-DSA (arch "GlmMoeDsaForCausalLM") uses per-expert gate_proj/up_proj/
# down_proj weights (expert-type 5, like Qwen3/DeepSeek). Requires transformers
# >= 5.12 which ships GlmMoeDsaForCausalLM. Registered only when the HF class
# is importable (mirrors the V4 and Qwen3.5 guards above).
if GlmMoeDsaForCausalLM is not None:
    MODEL_MAPPING_NAMES["glmmoedsa"] = GlmMoeDsaForCausalLM
    MODEL_MAPPING_TYPES["glmmoedsa"] = 5

# GLM-5.3-Flash (arch "Glm5NextForConditionalGeneration", model_type
# "glm5_next") nests its MoE fields under text_config and routes 288
# per-expert gate_proj/up_proj/down_proj experts with the same sigmoid/
# noaux_tc router family as GlmMoeDsa (expert-type 5). The hybrid KDA
# linear-attention layers, DSA indexer, mHC hyper-connections, and vision
# tower stay resident. Requires transformers >= 5.16 which ships the class;
# registered only when it is importable (mirrors the guards above).
if Glm5NextForConditionalGeneration is not None:
    MODEL_MAPPING_NAMES["glm5next"] = Glm5NextForConditionalGeneration
    MODEL_MAPPING_TYPES["glm5next"] = 5


def parse_expert_type(config: PretrainedConfig) -> int:
    architecture = (
        config.architectures[0].lower() if config.architectures else ""
    )
    arch = None
    # Match the most specific key first: "qwen3_5" and "deepseek_v3" both
    # contain shorter keys ("qwen3", "deepseek") as substrings, so longest-key
    # order prevents a shorter key from shadowing a more specific architecture.
    for supp_arch in sorted(MODEL_MAPPING_NAMES, key=len, reverse=True):
        if supp_arch in architecture:
            arch = supp_arch
            break
    if arch is None:
        raise RuntimeError(
            f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
            f"Please provide a model that is supported by the function. "
            f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
        )

    return MODEL_MAPPING_TYPES[arch]
