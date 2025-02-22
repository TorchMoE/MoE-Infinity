from typing import Any, Dict

import torch
from sllm_store.device_map_utils import _compute_device_placement_from_map_fast
from sllm_store.utils import get_no_split_modules, get_tied_no_split_modules
from transformers import PretrainedConfig, PreTrainedModel

from moe_infinity.utils.hf_config import parse_expert_id


def partition_offloading_state_dict(
    state_dict: Dict[str, Any], config: PretrainedConfig
):
    non_offloading_state_dict = {}
    offloading_state_dict = {}

    for key, value in state_dict.items():
        layer_id, expert_id = parse_expert_id(key)
        if layer_id is None:
            non_offloading_state_dict[key] = value
        else:
            offloading_state_dict[key] = value

    return non_offloading_state_dict, offloading_state_dict


def load_non_offloading_state_dict(
    model: PreTrainedModel, state_dict: Dict[str, Any]
):
    config = model.config
    no_split_modules = get_no_split_modules(model, model._no_split_modules)
    tied_no_split_modules = get_tied_no_split_modules(model, no_split_modules)

    device_map = _compute_device_placement_from_map_fast(
        no_split_modules, tied_no_split_modules, "auto"
    )

    # model.load_state_dict(state_dict, strict=False)

    for key, param in state_dict.items():
        levels = key.split(".")
        # If the key cannot be found in the model, skip it
        if not hasattr(model, levels[0]):
            continue
        weight = model.__getattr__(levels[0])
        for l in levels[1:]:
            if not hasattr(weight, l):
                weight = None
                break
            weight = weight.__getattr__(l)
        if weight is not None:
            weight.data = param.to(config.torch_dtype)
    byte_size = (
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    )
    print(f"Model non offloading size: {byte_size:.2f} GB", flush=True)
