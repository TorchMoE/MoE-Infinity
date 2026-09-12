"""DeepSeek-V4-Flash-Vision-Exp tensor classification.

Vision-Exp keeps the exact V4-Flash text backbone (43 layers x 256 FP4 routed
experts, DeepSeek-native key scheme) and adds a resident vision encoder +
aligner, learned image-token embeddings, and 3 MTP/nextn draft layers under a
separate top-level ``mtp.<i>.`` prefix (each with its own routed experts, which
must stay OUT of the text-layer host store). See VISION_EXP_NOTES.md for the
key-by-key inventory this classification is derived from.
"""

import re
from enum import Enum

_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")

_VISION_PREFIXES = ("vision.", "aligner.")
_IMAGE_TOKEN_KEYS = ("image_start", "image_end", "image_newline", "image_pad")


class TensorClass(str, Enum):
    ROUTED_EXPERT = "routed_expert"
    RESIDENT_TEXT = "resident_text"
    RESIDENT_VISION = "resident_vision"
    MTP_NEXTN = "mtp_nextn"


def is_vision_exp_config(config) -> bool:
    return getattr(config, "vision_n_layers", None) is not None


def classify_vision_exp_tensor(name: str, config) -> TensorClass:
    if name.startswith(_VISION_PREFIXES) or name in _IMAGE_TOKEN_KEYS:
        return TensorClass.RESIDENT_VISION
    if name.startswith("mtp."):
        return TensorClass.MTP_NEXTN
    expert_match = _EXPERT_RE.match(name)
    if expert_match is not None and int(expert_match.group(1)) < int(
        config.num_hidden_layers
    ):
        return TensorClass.ROUTED_EXPERT
    return TensorClass.RESIDENT_TEXT


def should_skip_resident_load(
    name: str, config, text_only: bool = True
) -> bool:
    if not is_vision_exp_config(config):
        return ".ffn.experts." in name
    cls = classify_vision_exp_tensor(name, config)
    if cls is TensorClass.ROUTED_EXPERT:
        return True
    return text_only and cls is TensorClass.MTP_NEXTN
