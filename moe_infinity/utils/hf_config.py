from moe_infinity.utils.constants import MODEL_MAPPING_TYPES, MODEL_MAPPING_NAMES
from transformers import PretrainedConfig, MixtralForCausalLM
from typing import Tuple
import re
import torch
from transformers import PretrainedConfig

def parse_expert_dtype(config: PretrainedConfig) -> int:
    dtype = config.torch_dtype
    if dtype == torch.bfloat16:
        dtype = 0
    elif dtype == torch.float32:
        dtype = 1
    elif dtype == torch.float16:
        dtype = 2
    else:
        assert False, "Unknown dtype %s" % dtype

    return dtype

def parse_expert_type(config: PretrainedConfig) -> int:
    arch = config.architectures[0].lower()
    if "switch" in arch:
        return MODEL_MAPPING_TYPES["switch"]
    elif "nllb" in arch:
        return MODEL_MAPPING_TYPES["nllb"]
    elif "mixtral" in arch:
        return MODEL_MAPPING_TYPES["mixtral"]
    elif "grok" in arch:
        return MODEL_MAPPING_TYPES["grok"]
    # elif "opt" in arch:
    #     return 0
    else:
        raise RuntimeError(f"Unsupported architecture {arch}")

    return MODEL_MAPPING_TYPES[arch]


def parse_moe_param(config: PretrainedConfig) -> Tuple[int, int, int]:
    arch = config.architectures[0].lower()

    if "switch" in arch:
        num_encoder_layers = config.num_sparse_encoder_layers
        num_decoder_layers = config.num_sparse_decoder_layers
        num_layers = num_encoder_layers + num_decoder_layers
        num_experts = config.num_experts
    elif "nllb" in arch:
        num_encoder_layers = config.encoder_layers // config.encoder_sparse_step
        num_decoder_layers = config.decoder_layers // config.decoder_sparse_step
        num_layers = num_encoder_layers + num_decoder_layers
        num_experts = config.num_experts
    elif "mixtral" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
    # elif "opt" in arch:
    #     num_encoder_layers = 0
    #     num_decoder_layers = config.num_hidden_layers
    #     num_layers = num_encoder_layers + num_decoder_layers
    #     num_experts = 0
    elif "grok" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
    else:
        raise RuntimeError(f"Unsupported architecture {arch}")

    return num_layers, num_experts, num_encoder_layers


def parse_expert_id(param_name: str, config: PretrainedConfig) -> Tuple[int, int]:
    arch = config.architectures[0].lower()
    _, _, num_encoder_layers = parse_moe_param(config)

    if "switch" in arch or "nllb" in arch:
        # example "decoder.block.1.layer.2.mlp.experts.expert_100.wi.weight"
        encoder_sparse_step = config.encoder_sparse_step
        decoder_sparse_step = config.decoder_sparse_step

        result = re.findall(
            r"(encoder|decoder)\.[a-z]+\.(\d+).*expert_(\d+)", param_name
        )

        if result:
            layer_type, layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)

    elif "mixtral" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        result = re.findall(
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "grok" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.0.moe_block.experts.0.linear_1.weight"
        result = re.findall(
            r"layers\.(\d+)\.moe_block\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            # print(f"layer_id: {layer_id}, expert_id: {expert_id}")
            layer_id = int(layer_id)
            expert_id = int(expert_id)

    if result:
        if layer_type == "decoder":
            layer_id = layer_id // decoder_sparse_step + num_encoder_layers
        elif layer_type == "encoder":
            layer_id = layer_id // encoder_sparse_step
        else:
            raise ValueError(f"Unsupported layer type {layer_type}")

        return layer_id, expert_id

    return None, None
