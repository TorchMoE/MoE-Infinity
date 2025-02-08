from transformers import (
    MixtralForCausalLM,
    NllbMoeForConditionalGeneration,
    OPTForCausalLM,
    PretrainedConfig,
    SwitchTransformersForConditionalGeneration,
)

from ..models.modeling_arctic import (
    ArcticForCausalLM,
)  # TODO: Replace this with huggingface transformers
from ..models.modeling_deepseek import DeepseekV2ForCausalLM
from ..models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ..models.modeling_grok.modeling_grok1 import (
    Grok1ModelForCausalLM,
)  # TODO: Replace this with huggingface transformers

MODEL_MAPPING_NAMES = {
    "switch": SwitchTransformersForConditionalGeneration,
    "nllb": NllbMoeForConditionalGeneration,
    "mixtral": MixtralForCausalLM,
    "opt": OPTForCausalLM,
    "grok": Grok1ModelForCausalLM,
    "arctic": ArcticForCausalLM,
    "deepseek": DeepseekV2ForCausalLM,
    "deepseek_v3": DeepseekV3ForCausalLM,
}

MODEL_MAPPING_TYPES = {
    "switch": 0,
    "nllb": 2,
    "mixtral": 4,
    "grok": 4,
    "arctic": 4,
    "deepseek": 5,
    "deepseek_v3": 5,
}


def parse_expert_type(config: PretrainedConfig) -> int:
    architecture = config.architectures[0].lower()
    arch = None
    for supp_arch in MODEL_MAPPING_NAMES:
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
