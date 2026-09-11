import re
from typing import Optional, Tuple

import torch
from transformers import PretrainedConfig

_DEEPSEEK_V2_DEFAULTS = {
    "head_dim": None,
    "mlp_bias": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "aux_loss_alpha": 0.001,
    "seq_aux": True,
    "norm_topk_prob": False,
}


def _apply_missing_attrs(cfg, defaults):
    changed = False
    for key, default in defaults.items():
        if key not in cfg.__dict__:
            if key == "head_dim":
                val = getattr(cfg, "qk_rope_head_dim", 0)
                if val == 0 and hasattr(cfg, "num_attention_heads"):
                    val = cfg.hidden_size // cfg.num_attention_heads
            else:
                val = default
            cfg.__dict__[key] = val
            changed = True
    return changed


def ensure_config_compat(config: PretrainedConfig) -> PretrainedConfig:
    arch_name = getattr(config, "model_type", "")
    if "deepseek" not in arch_name:
        return config

    if _apply_missing_attrs(config, _DEEPSEEK_V2_DEFAULTS):
        cfg_cls = type(config)
        _orig_init = cfg_cls.__init__

        def _patched_init(self, *a, **kw):
            _orig_init(self, *a, **kw)
            _apply_missing_attrs(self, _DEEPSEEK_V2_DEFAULTS)

        cfg_cls.__init__ = _patched_init
    return config


def resolve_config_dtype(config: object) -> Optional[torch.dtype]:
    # transformers v5 renamed config.torch_dtype -> config.dtype. Read the new
    # name first, fall back to the old one so both 4.x and 5.x configs work.
    dtype = getattr(config, "dtype", None)
    if dtype is None:
        dtype = getattr(config, "torch_dtype", None)
    return dtype


def parse_expert_dtype(config: PretrainedConfig) -> int:
    dtype = resolve_config_dtype(config)
    if dtype is None:
        dtype = torch.bfloat16
    if dtype == torch.bfloat16:
        return 0
    elif dtype == torch.float32:
        return 1
    elif dtype == torch.float16:
        return 2
    else:
        assert False, "Unknown dtype %s" % dtype


def moe_text_config(config: PretrainedConfig) -> PretrainedConfig:
    # Qwen3.5-MoE is a vision-language wrapper: its MoE fields (num_experts,
    # num_hidden_layers, ...) live under the nested text sub-config, exposed via
    # get_text_config() in transformers v5. Flat models return themselves.
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        text = getter()
    else:
        text = getattr(config, "text_config", config)
    if isinstance(text, dict):
        # A generic PretrainedConfig (used when the model class is not
        # importable, e.g. glm5_next on transformers < 5.16) keeps nested
        # sub-configs as plain dicts; normalize so MoE-shape parsing works
        # without the modeling class.
        text = PretrainedConfig.from_dict(text)
    return text


def parse_moe_param(config: PretrainedConfig) -> Tuple[int, int, int]:
    arch = (config.architectures or [""])[0].lower()

    if "nllb" in arch:
        encoder_sparse_step = int(config.encoder_sparse_step)
        decoder_sparse_step = int(config.decoder_sparse_step)
        num_encoder_layers = config.encoder_layers // encoder_sparse_step
        num_decoder_layers = config.decoder_layers // decoder_sparse_step
        num_layers = num_encoder_layers + num_decoder_layers
        num_experts = config.num_experts
    elif "mixtral" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
    elif "qwen3_5" in arch:
        text = moe_text_config(config)
        num_encoder_layers = 0
        num_decoder_layers = text.num_hidden_layers
        num_layers = text.num_hidden_layers
        num_experts = text.num_experts
    elif "qwen3" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
    elif "glmmoedsa" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.n_routed_experts
    elif "glm5next" in arch:
        text = moe_text_config(config)
        num_encoder_layers = 0
        num_decoder_layers = text.num_hidden_layers
        num_layers = text.num_hidden_layers
        num_experts = text.n_routed_experts
    elif "deepseek" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.n_routed_experts
    elif "gpt_oss" in arch or "gptoss" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_local_experts
    else:
        raise RuntimeError(f"Unsupported architecture {arch}")

    return num_layers, num_experts, num_encoder_layers


def parse_expert_id(
    param_name: str, config: PretrainedConfig
) -> Tuple[Optional[int], Optional[int]]:
    arch = (config.architectures or [""])[0].lower()
    num_layers, _, num_encoder_layers = parse_moe_param(config)
    result = None
    layer_type = ""
    layer_id = 0
    expert_id = 0
    encoder_sparse_step = 1
    decoder_sparse_step = 1

    if "nllb" in arch:
        # example "decoder.block.1.layer.2.mlp.experts.expert_100.wi.weight"
        encoder_sparse_step = int(config.encoder_sparse_step)
        decoder_sparse_step = int(config.decoder_sparse_step)

        result = re.findall(
            r"(encoder|decoder)\.[a-z]+\.(\d+).*expert_(\d+)", param_name
        )

        if result:
            layer_type, layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)

    elif "mixtral" in arch:
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
    elif "deepseekv4" in arch:
        decoder_sparse_step = 1
        layer_type = "decoder"

        # native key: "layers.1.ffn.experts.0.w1.weight"
        result = re.findall(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.", param_name)
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "qwen3_5" in arch:
        decoder_sparse_step = 1
        layer_type = "decoder"

        # per-expert keys after the v5 batched-expert expand; anchored on
        # language_model to exclude MTP (`mtp.*`) and vision (`model.visual.*`).
        # e.g. "model.language_model.layers.13.mlp.experts.7.gate_proj.weight"
        result = re.findall(
            r"language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "glmmoedsa" in arch:
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.10.mlp.experts.3.gate_proj.weight"
        result = re.findall(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
            # MTP layer guard: GLM has a MTP layer at index num_hidden_layers (78)
            if layer_id >= num_layers:
                return None, None
    elif "glm5next" in arch:
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.language_model.layers.10.mlp.experts.3.gate_proj.weight";
        # anchored on language_model to exclude vision keys (`model.visual.*`).
        result = re.findall(
            r"language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
            # MTP layer guard, same convention as the GlmMoeDsa branch above
            if layer_id >= num_layers:
                return None, None
    elif "deepseek" in arch or "qwen3" in arch:
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.1.mlp.experts.0.gate_proj.weight"
        result = re.findall(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name)
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    elif "gpt_oss" in arch or "gptoss" in arch:
        layer_type = "decoder"
        result = re.findall(
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\."
            r"(?:gate_up_proj|down_proj)_(?:blocks|scales|bias)$",
            param_name,
        )
        if result:
            layer_id, expert_id = (int(value) for value in result[0])
            if layer_id >= num_layers or expert_id >= config.num_local_experts:
                return None, None

    if result:
        if layer_type == "decoder":
            layer_id = layer_id // decoder_sparse_step + num_encoder_layers
        elif layer_type == "encoder":
            layer_id = layer_id // encoder_sparse_step
        else:
            raise ValueError(f"Unsupported layer type {layer_type}")

        return layer_id, expert_id

    return None, None
