import pytest

from moe_infinity.models.deepseek_v4.vision_exp import (
    TensorClass,
    classify_vision_exp_tensor,
    is_vision_exp_config,
    should_skip_resident_load,
)


class _VisionExpCfg:
    num_hidden_layers = 43
    num_nextn_predict_layers = 3
    n_routed_experts = 256
    vision_n_layers = 32
    dspark_target_layer_ids = [40, 41, 42]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("layers.5.ffn.experts.17.w1.weight", TensorClass.ROUTED_EXPERT),
        ("layers.5.ffn.experts.17.w1.scale", TensorClass.ROUTED_EXPERT),
        ("layers.0.ffn.experts.0.w2.weight", TensorClass.ROUTED_EXPERT),
        ("layers.42.ffn.experts.255.w3.scale", TensorClass.ROUTED_EXPERT),
        ("layers.5.ffn.shared_experts.w1.weight", TensorClass.RESIDENT_TEXT),
        ("layers.5.ffn.gate.weight", TensorClass.RESIDENT_TEXT),
        ("layers.5.ffn.gate.bias_vl", TensorClass.RESIDENT_TEXT),
        ("layers.5.attn.wkv.weight", TensorClass.RESIDENT_TEXT),
        ("layers.2.attn.compressor.ape", TensorClass.RESIDENT_TEXT),
        ("embed.weight", TensorClass.RESIDENT_TEXT),
        ("head.weight", TensorClass.RESIDENT_TEXT),
        ("norm.weight", TensorClass.RESIDENT_TEXT),
        ("hc_head_base", TensorClass.RESIDENT_TEXT),
        ("mtp.0.attn.wkv.weight", TensorClass.MTP_NEXTN),
        ("mtp.2.ffn.experts.0.w1.weight", TensorClass.MTP_NEXTN),
        ("vision.blocks.0.attn.wqkv.weight", TensorClass.RESIDENT_VISION),
        ("vision.patch_embed.proj.weight", TensorClass.RESIDENT_VISION),
        ("aligner.w1.weight", TensorClass.RESIDENT_VISION),
        ("image_newline", TensorClass.RESIDENT_VISION),
        ("image_start", TensorClass.RESIDENT_VISION),
    ],
)
def test_classify(name, expected):
    assert classify_vision_exp_tensor(name, _VisionExpCfg()) == expected


def test_is_vision_exp_config():
    assert is_vision_exp_config(_VisionExpCfg())


def test_base_v4_flash_is_not_vision_exp():
    class _TextOnlyCfg:
        num_hidden_layers = 43
        num_nextn_predict_layers = 1
        n_routed_experts = 256

    assert not is_vision_exp_config(_TextOnlyCfg())


class _BaseV4Cfg:
    num_hidden_layers = 43
    num_nextn_predict_layers = 1
    n_routed_experts = 256


@pytest.mark.parametrize(
    "name,text_only,expected_skip",
    [
        ("layers.5.ffn.experts.17.w1.weight", True, True),
        ("layers.5.ffn.experts.17.w1.weight", False, True),
        ("mtp.0.attn.wkv.weight", True, True),
        ("mtp.0.attn.wkv.weight", False, False),
        ("mtp.2.ffn.experts.0.w1.weight", True, True),
        ("vision.blocks.0.attn.wqkv.weight", True, False),
        ("aligner.w1.weight", True, False),
        ("image_newline", True, False),
        ("layers.5.ffn.shared_experts.w1.weight", True, False),
        ("layers.5.attn.wkv.weight", True, False),
        ("embed.weight", True, False),
    ],
)
def test_should_skip_resident_load_vision_exp(name, text_only, expected_skip):
    assert (
        should_skip_resident_load(name, _VisionExpCfg(), text_only=text_only)
        is expected_skip
    )


@pytest.mark.parametrize(
    "name,expected_skip",
    [
        ("layers.5.ffn.experts.17.w1.weight", True),
        ("layers.5.ffn.shared_experts.w1.weight", False),
        ("layers.5.attn.wkv.weight", False),
        ("embed.weight", False),
    ],
)
def test_should_skip_resident_load_base_v4(name, expected_skip):
    assert should_skip_resident_load(name, _BaseV4Cfg()) is expected_skip
