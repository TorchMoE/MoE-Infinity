from transformers import (
    SwitchTransformersForConditionalGeneration,
    NllbMoeForConditionalGeneration,
    MixtralForCausalLM,
    OPTForCausalLM,
)

MODEL_MAPPING_NAMES = {
    "switch": SwitchTransformersForConditionalGeneration,
    "nllb": NllbMoeForConditionalGeneration,
    "mixtral": MixtralForCausalLM,
    "opt": OPTForCausalLM
}


#define SWITCH_TRANSFORMERS_DENSE_ACT_DENSE 0
#define SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE 1
#define NLLB_MOE_DENSE_ACT_DENSE 2
#define FSGPT_MOE_DENSE_ACT_DENSE 3
#define MIXTRAL_MOE_DENSE_ACT_DENSE 4

MODEL_MAPPING_TYPES = {
    "switch": 0,
    "nllb": 2,
    "mixtral": 4,
}