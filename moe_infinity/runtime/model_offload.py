# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnannotatedClassAttribute=false, reportUninitializedInstanceVariable=false, reportPrivateUsage=false, reportPrivateLocalImportUsage=false, reportUnusedImport=false, reportUnusedCallResult=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportExplicitAny=false, reportAny=false, reportArgumentType=false, reportOperatorIssue=false, reportImplicitStringConcatenation=false, reportUnnecessaryComparison=false, reportUnreachable=false, reportMissingTypeArgument=false, reportDeprecated=false, reportGeneralTypeIssues=false
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import functools
import gc
import hashlib
import importlib
import json
import logging
import os
import re
import tempfile
import time
import warnings
from typing import Callable, Dict, Optional, Type, Union

import torch
import transformers

try:
    from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
        QuantLinear as QuantLinearOld,
    )
except ImportError:

    class QuantLinear:
        pass

    class QuantLinearOld:
        pass


from safetensors import safe_open
from tqdm import tqdm
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from moe_infinity.common import parse_expert_type
from moe_infinity.distributed import DistributedExpertExecutor
from moe_infinity.memory import ExpertPredictor, ExpertPrefetcher, ExpertTracer
from moe_infinity.models import (
    Qwen3MoEBlock,
    Qwen3PagedAttention,
    SyncDbrxFFNBlock,
    SyncDeepseekV2MoEBlock,
    SyncDeepseekV3MoEBlock,
    SyncGlmMoeDsaMoEBlock,
    SyncGptOssMLP,
    SyncJambaMoEBlock,
    SyncMixtralSparseMoeBlock,
    SyncNllbMoeSparseMLP,
    SyncOlmoeMoEBlock,
    SyncQwen3_5MoeSparseMoeBlock,
)
from moe_infinity.runtime.hooks import *
from moe_infinity.utils import (
    ArcherConfig,
    parse_expert_dtype,
    parse_expert_id,
    parse_moe_param,
    resolve_config_dtype,
)
from moe_infinity.utils.arguments import (
    copy_args_to_device,
    copy_kwargs_to_device,
)
from moe_infinity.utils.async_transfer import (
    async_d2h,
    async_h2d,
    wait_transfer,
)
from moe_infinity.utils.device import get_default_device, get_device
from moe_infinity.utils.gptq import is_gptq_packed_tensor, is_gptq_quantized
from moe_infinity.utils.mxfp4 import identify_mxfp4_pairs, is_mxfp4_quantized
from moe_infinity.utils.quantization import (
    detect_quantization,
    should_cast_tensor,
    validate_quantization_support,
)

_prefetch_lib = None
# Alias for compatibility
prefetch_op = None

logger = logging.getLogger(__name__)


def _arch_name(config: object) -> str:
    archs = getattr(config, "architectures", None) or [""]
    name = (archs[0] or "").lower()
    if not name:
        name = (getattr(config, "model_type", "") or "").lower()
    return name


def _remap_v5_batched_experts(
    state_dict: Dict[str, torch.Tensor], config: object
) -> None:
    # transformers v5 stores MoE experts as batched 3D tensors under ".mlp":
    #   mlp.experts.gate_up_proj [E, 2*inter, H], mlp.experts.down_proj [E, *, *].
    # MoE-Infinity offloads per-expert, so expand them back to the per-expert keys
    # each arch's parse_expert_id expects. Two naming schemes:
    #   - Mixtral: block ".mlp" -> ".block_sparse_moe", experts.{E}.w1/w3/w2.weight
    #     (w1=gate, w3=up, w2=down), gate moved to block_sparse_moe.gate.weight.
    #   - Qwen3 / DeepSeek / OLMoE: keep ".mlp", experts.{E}.gate_proj/up_proj/
    #     down_proj.weight.
    # gate_up_proj[e] splits on dim 0 into gate and up; down_proj[e] is down.
    # Numerically equivalent to the v5 expert forward. No-op on v4 checkpoints.
    # GPT-OSS keeps its own batched path (regex + MXFP4), so it is excluded.
    arch = _arch_name(config)
    if "gpt_oss" in arch or "gptoss" in arch:
        return

    is_mixtral = "mixtral" in arch
    if is_mixtral:
        gate_name, up_name, down_name = "w1", "w3", "w2"
    else:
        gate_name, up_name, down_name = "gate_proj", "up_proj", "down_proj"

    def _out_block(experts_prefix: str) -> str:
        block_prefix = experts_prefix[: -len(".experts")]
        if is_mixtral and block_prefix.endswith(".mlp"):
            out_block = block_prefix[: -len(".mlp")] + ".block_sparse_moe"
            gate_key = block_prefix + ".gate.weight"
            out_gate_key = out_block + ".gate.weight"
            if gate_key in state_dict and out_gate_key not in state_dict:
                state_dict[out_gate_key] = state_dict.pop(gate_key)
            return out_block
        return block_prefix

    # In sharded checkpoints, a layer's gate_up_proj and down_proj can live in
    # different shards, so expand each independently instead of requiring both
    # in the same per-shard state_dict slice.
    for gate_up_key in [
        k for k in list(state_dict) if k.endswith("experts.gate_up_proj")
    ]:
        gate_up = state_dict[gate_up_key]
        if gate_up.dim() != 3:
            continue
        out_block = _out_block(gate_up_key[: -len(".gate_up_proj")])
        for expert_idx in range(gate_up.shape[0]):
            gate_w, up_w = gate_up[expert_idx].chunk(2, dim=0)
            base = f"{out_block}.experts.{expert_idx}"
            state_dict[f"{base}.{gate_name}.weight"] = gate_w.contiguous()
            state_dict[f"{base}.{up_name}.weight"] = up_w.contiguous()
        del state_dict[gate_up_key]

    for down_key in [
        k for k in list(state_dict) if k.endswith("experts.down_proj")
    ]:
        down = state_dict[down_key]
        if down.dim() != 3:
            continue
        out_block = _out_block(down_key[: -len(".down_proj")])
        for expert_idx in range(down.shape[0]):
            base = f"{out_block}.experts.{expert_idx}"
            state_dict[f"{base}.{down_name}.weight"] = down[
                expert_idx
            ].contiguous()
        del state_dict[down_key]


_GPT_OSS_EXPERT_FIELDS = (
    "gate_up_proj_blocks",
    "gate_up_proj_scales",
    "gate_up_proj_bias",
    "down_proj_blocks",
    "down_proj_scales",
    "down_proj_bias",
)

_GPT_OSS_RESIDENT_RATIO = 0.9


def _gpt_oss_offload_enabled(model_type, archer_config):
    if model_type != "gpt_oss":
        return True
    return archer_config.device_memory_ratio < _GPT_OSS_RESIDENT_RATIO


def _wire_expert_collaborators(
    module,
    enabled,
    *,
    expert_executor,
    expert_prefetcher,
    expert_tracer,
    expert_predictor,
    expert_tensor_map,
):
    if not enabled:
        return
    module.expert_executor = expert_executor
    module.expert_prefetcher = expert_prefetcher
    module.expert_tracer = expert_tracer
    module.expert_predictor = expert_predictor
    module.expert_tensor_map = expert_tensor_map


def _expand_gpt_oss_packed_experts(state_dict, config):
    if getattr(config, "model_type", "") != "gpt_oss":
        return

    # gpt-oss checkpoints can split a layer's six packed expert components across
    # safetensors shards, and the loader expands one per-shard state_dict slice
    # at a time. Expand each packed component independently so a slice missing
    # some of the six still expands the ones it holds; global per-(layer, expert)
    # completeness is enforced later over the merged name_id_map by
    # _gpt_oss_expert_groups.
    expected_experts = int(config.num_local_experts)
    field_set = set(_GPT_OSS_EXPERT_FIELDS)
    packed_keys = [
        key
        for key in list(state_dict)
        if ".mlp.experts." in key and key.rsplit(".", 1)[-1] in field_set
    ]
    for key in packed_keys:
        prefix, field = key.rsplit(".", 1)
        tensor = state_dict[key]
        if tensor.shape[0] != expected_experts:
            raise ValueError(
                f"{key} has {tensor.shape[0]} experts; "
                f"expected {expected_experts}"
            )
        flatten_blocks = field.endswith("_blocks")
        for expert_idx in range(expected_experts):
            view = tensor[expert_idx]
            if flatten_blocks and view.dim() == 3:
                # MXFP4 blocks ship as [out, n_blocks, 16]; the native MoEMLP
                # offload path (DequantMxfp4Params/SetTensorsFromIds) treats
                # blocks as 2D [out, packed_bytes] and derives hidden as
                # packed_bytes*2, so collapse the trailing block dims here.
                view = view.reshape(view.shape[0], -1)
            if not view.is_contiguous():
                raise ValueError(
                    f"Non-contiguous GPT-OSS slice {prefix}.{expert_idx}.{field}"
                )
            state_dict[f"{prefix}.{expert_idx}.{field}"] = view
        del state_dict[key]


def _gpt_oss_expert_groups(name_id_map, config):
    fields = {name: index for index, name in enumerate(_GPT_OSS_EXPERT_FIELDS)}
    grouped = {}
    for name, tensor_id in name_id_map.items():
        layer_id, expert_id = parse_expert_id(name, config)
        if layer_id is None or expert_id is None:
            continue
        field = name.rsplit(".", 1)[-1]
        if field not in fields:
            continue
        slots = grouped.setdefault(
            (layer_id, expert_id), [None] * len(_GPT_OSS_EXPERT_FIELDS)
        )
        slots[fields[field]] = tensor_id

    topology = []
    for layer_id in range(config.num_hidden_layers):
        experts = []
        for expert_id in range(config.num_local_experts):
            ids = grouped.get((layer_id, expert_id))
            if ids is None or any(tensor_id is None for tensor_id in ids):
                raise ValueError(
                    f"Missing GPT-OSS expert tensors for layer={layer_id}, "
                    f"expert={expert_id}"
                )
            experts.append(ids)
        topology.append((f"model.layers.{layer_id}.mlp.experts", experts))
    return topology


def _make_expert_tensor_map(name_id_map, config):
    if getattr(config, "model_type", "") == "gpt_oss":
        return {
            (layer_id, expert_id): tensor_ids[0]
            for layer_id, (_, experts) in enumerate(
                _gpt_oss_expert_groups(name_id_map, config)
            )
            for expert_id, tensor_ids in enumerate(experts)
        }
    result = {}
    for name, tensor_id in name_id_map.items():
        layer_id, expert_id = parse_expert_id(name, config)
        if expert_id is not None:
            result[(layer_id, expert_id)] = tensor_id
    return result


def _make_expert_nbytes_map(model, config):
    """Sum stored payload bytes per ``(layer_id, expert_id)`` from live params.

    Read while the routed-expert weights still carry their true shape/dtype
    (before ``setup_archer_hooks`` installs offload placeholders), so
    ``numel * element_size`` is the exact stored FP4/FP8 payload the route-ahead
    prefetch fetches -- even a meta-init param preserves shape and dtype. This
    is read-only measurement metadata for the A5 recorder; it never changes
    routing, prefetch, or offload placement. gpt-oss stacks its experts into one
    tensor and never reaches the executor route-ahead seam, so it is skipped.
    """
    if getattr(config, "model_type", "") == "gpt_oss":
        return {}
    result: dict[tuple[int, int], int] = {}
    for name, param in model.named_parameters(recurse=True):
        layer_id, expert_id = parse_expert_id(name, config)
        if expert_id is None:
            continue
        nbytes = int(param.numel()) * int(param.element_size())
        result[(layer_id, expert_id)] = (
            result.get((layer_id, expert_id), 0) + nbytes
        )
    return result


def _identify_fp8_blockwise_pairs(keys):
    key_set = set(keys)
    pairs = []
    for k in keys:
        if k.endswith("_scale_inv"):
            base = k[: -len("_scale_inv")]
            if base in key_set:
                pairs.append((base, k))
    return pairs


_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")


def _is_routed_expert_key(key: str) -> bool:
    # Routed experts (including the MTP layer's, which parse_expert_id drops via
    # its layer_id >= num_layers guard) are dequantized on-device by the
    # dispatcher, so they stay FP8 in the store. Matches ".mlp.experts.<id>."
    # but not ".mlp.shared_experts." or dense ".mlp.<proj>".
    # NOTE(#142): Mixtral names routed experts ".block_sparse_moe.experts.<id>."
    # (see _remap_v5_batched_experts), which this regex does NOT match, so its
    # routed experts are currently misclassified as non-routed. Tracked in #142.
    return _ROUTED_EXPERT_RE.search(key) is not None


def _has_fp8_blockwise(config: object) -> bool:
    qcfg = getattr(config, "quantization_config", None)
    if qcfg is None:
        return False
    if isinstance(qcfg, dict):
        method = qcfg.get("quant_method", "") or qcfg.get("fmt", "")
    else:
        method = getattr(qcfg, "quant_method", "") or getattr(qcfg, "fmt", "")
    return "fp8" in str(method).lower()


def _compute_config_fingerprint(config: object) -> str:
    fields = [
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "vocab_size",
        "intermediate_size",
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "num_experts_per_tok",
        "torch_dtype",
    ]
    fingerprint_dict = {}
    for field in fields:
        val = getattr(config, field, None)
        if val is not None:
            fingerprint_dict[field] = str(val)
    serialized = json.dumps(fingerprint_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _write_model_signature(
    offload_path: str, model_name: str, config: object
) -> None:
    signature = {
        "model_name": model_name,
        "config_fingerprint": _compute_config_fingerprint(config),
        "signature_version": 1,
    }
    sig_path = os.path.join(offload_path, "model_signature.json")
    fd, tmp_path = tempfile.mkstemp(dir=offload_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(signature, f)
        os.replace(tmp_path, sig_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info(
        "Created model signature for '%s' at %s", model_name, offload_path
    )


def _validate_model_signature(
    offload_path: str, model_name: str, config: object
) -> None:
    sig_path = os.path.join(offload_path, "model_signature.json")

    if not os.path.exists(sig_path):
        logger.warning(
            "No model signature found in %s. This appears to be a legacy cache. "
            "Stamping with current model '%s'.",
            offload_path,
            model_name,
        )
        _write_model_signature(offload_path, model_name, config)
        return

    try:
        with open(sig_path, "r") as f:
            stored = json.load(f)
        stored_model_name = stored["model_name"]
        stored_fingerprint = stored["config_fingerprint"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(
            f"Corrupted model signature file at {sig_path}. Delete the file or the "
            + f"entire offload directory and retry. (Detail: {e})"
        ) from e

    if stored_model_name != model_name:
        raise ValueError(
            f"Model name mismatch: offload cache at '{offload_path}' was created for "
            + f"model '{stored_model_name}', but you are loading '{model_name}'. Use a "
            + f"different offload_path or delete the existing cache."
        )

    current_fingerprint = _compute_config_fingerprint(config)
    if stored_fingerprint != current_fingerprint:
        raise ValueError(
            f"Model config mismatch: offload cache at '{offload_path}' has a different "
            + f"model configuration than '{model_name}'. The model architecture or version "
            + f"may have changed. Delete the existing cache and retry."
        )


def _load_prefetch_lib():
    global _prefetch_lib, prefetch_op
    if _prefetch_lib is None:
        try:
            prefetch_lib = importlib.import_module("moe_infinity._store")
        except ImportError as exc:
            raise ImportError(
                "moe_infinity._store extension is required. Install with CUDA enabled."
            ) from exc

        _prefetch_lib = prefetch_lib
        prefetch_op = prefetch_lib

    return _prefetch_lib


class OffloadEngine(object):
    param_id = 0
    request_id = 0
    config = {}

    @property
    def num_offloaded_experts(self) -> int:
        prefetcher = getattr(self, "expert_prefetcher", None)
        if prefetcher is None:
            return 0
        return int(getattr(prefetcher, "num_offloaded_experts", 0))

    @property
    def expert_cache_hit_rate(self) -> float:
        prefetcher = getattr(self, "expert_prefetcher", None)
        getter = getattr(prefetcher, "get_hit_rate", None)
        return float(getter()) if callable(getter) else 0.0

    @property
    def expert_occupancy_bytes(self) -> float:
        prefetcher = getattr(self, "expert_prefetcher", None)
        getter = getattr(prefetcher, "expert_occupancy_bytes", None)
        return float(getter()) if callable(getter) else 0.0

    def _fetch_tensors_timed(self, tensors) -> None:
        """Issue the blocking on-demand fetch and accumulate its stall time.

        This synchronous call is the exposed-fetch window: compute cannot
        proceed until every un-prefetched expert is resident. Read-only BM3
        instrumentation (plan Task 9) -- the fetch is unchanged, only timed.
        """
        start = time.perf_counter()
        self.archer_engine.fetch_tensors(self.request_id, tensors)
        self._exposed_fetch_seconds += time.perf_counter() - start

    def get_exposed_fetch_seconds(self) -> float:
        """Total wall time compute stalled on on-demand expert fetches."""
        return float(getattr(self, "_exposed_fetch_seconds", 0.0))

    def reset_exposed_fetch_seconds(self) -> None:
        """Zero the exposed-fetch accumulator (per BM3 ablation arm)."""
        self._exposed_fetch_seconds = 0.0

    @property
    def kv_occupancy_bytes(self) -> Optional[float]:
        manager = getattr(self, "kv_cache_manager", None)
        if manager is None:
            return None
        for name in ("occupancy_bytes", "resident_bytes", "kv_cache_bytes"):
            value = getattr(manager, name, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    def __init__(
        self,
        capacity,
        config: PretrainedConfig,
        attention_backend=None,
        enable_attention_offload: bool = False,
        kv_cache_manager=None,
        enable_kv_cache_offload: bool = False,
    ):
        self.offload_exemption = set()
        self.expert_modules = []
        self.kv_cache_manager = kv_cache_manager
        self._exposed_fetch_seconds = 0.0

        self.ckpt_files = []

        self.expert_tracer = ExpertTracer(capacity, config)
        self.expert_predictor = ExpertPredictor(config)
        self.expert_predictor.add_tracer(self.expert_tracer)

        self.config = config

        self.quant_method = None
        self._quant_info = None

        # AttentionBackend scaffolding (no-op by default)
        # Set enable_attention_offload=True to activate (future work)
        self._attention_backend = None
        self._enable_attention_offload: bool = enable_attention_offload
        if attention_backend is not None:
            self.set_attention_backend(attention_backend)

        # KVCacheManager scaffolding (no-op by default)
        self._kv_cache_manager = kv_cache_manager
        self._enable_kv_cache_offload: bool = enable_kv_cache_offload
        self._captured_kv: dict[int, tuple[object, ...]] = {}
        self.model = None

    def decode_graph_capability(self):
        from moe_infinity.runtime.attention_types import DecodeGraphCapability

        if getattr(self, "forward_hooks", None) or getattr(
            self, "backward_hooks", None
        ):
            return DecodeGraphCapability(False, "active_model_hooks")
        if getattr(self, "archer_engine", None) is not None:
            return DecodeGraphCapability(False, "archer_callbacks")
        if getattr(self, "expert_dispatcher", None) is not None:
            return DecodeGraphCapability(False, "expert_dispatcher")
        if getattr(self, "expert_prefetcher", None) is not None:
            return DecodeGraphCapability(False, "transfer_scheduler")
        if (
            self._enable_kv_cache_offload
            or getattr(self, "_kv_cache_manager", None) is not None
        ):
            return DecodeGraphCapability(False, "kv_offload")
        return DecodeGraphCapability(True, "eligible")

    def get_attention_backend(self):
        """Return the registered attention backend, or None if not configured.

        Future: when enable_attention_offload=True, this backend will be used
        to intercept attention computation for CPU offloading.
        See moe_infinity/runtime/attention_backend.py for the interface.
        """
        return self._attention_backend

    def set_attention_backend(self, backend):
        """Register an AttentionBackend implementation.

        Args:
            backend: An object implementing the AttentionBackend Protocol.
                     Use PlaceholderAttentionBackend for testing.
        """
        from moe_infinity.runtime.attention_backend import AttentionBackend

        if not isinstance(backend, AttentionBackend):
            raise TypeError(
                f"backend must implement AttentionBackend Protocol, got {type(backend)}"
            )
        self._attention_backend = backend

    def get_kv_cache_manager(self):
        """Return the registered KVCacheManager, or None if not configured.

        Future: when enable_kv_cache_offload=True, this manager handles
        swapping KV cache blocks between GPU and CPU pinned memory.
        See moe_infinity/memory/kv_cache_manager.py for the interface.
        """
        return self._kv_cache_manager

    def set_kv_cache_manager(self, manager):
        """Register a KVCacheManager for KV cache offloading."""
        self._kv_cache_manager = manager

    def deliver_fp8_scales_to_dispatcher(self):
        scales = getattr(self, "_glm_fp8_scales", None)
        if not scales:
            return
        dispatcher = getattr(self, "expert_dispatcher", None)
        set_scales = getattr(dispatcher, "set_scales", None)
        if callable(set_scales):
            set_scales(scales)
        else:
            warnings.warn(
                "GLM-5.2-FP8 routed experts are kept FP8 in the store but the "
                "native set_scales dispatcher API is unavailable; build the "
                "moe_infinity._v4_fp4 extension with fp8-in-store support. "
                "Routed experts will not be dequantized correctly.",
                RuntimeWarning,
                stacklevel=2,
            )

    def init(
        self,
        cls: Type[PreTrainedModel],
        ar_config: Union[str, Dict, ArcherConfig],
    ):
        self.cls = cls
        self.name_id_map = {}
        self.tensor_id_map = {}
        self.registered_tensors = set()
        self.forward_hooks = []
        self.backward_hooks = []

        self.offload_set = set()

        if isinstance(ar_config, str):
            _archer_config = ArcherConfig.load_from_file(ar_config)
        elif isinstance(ar_config, dict):
            _archer_config = ArcherConfig.load_from_json(ar_config)
        elif isinstance(ar_config, ArcherConfig):
            _archer_config = ar_config
        else:
            raise ValueError(
                "ArcherConfig is not provided. Please provide a path to a config file or a dict."
            )

        # TODO: get trace from trace_path

        self.checkpoint = _archer_config.offload_path

        os.makedirs(self.checkpoint, exist_ok=True)

        self.prefetch_lib = _load_prefetch_lib()

        self.archer_engine = self.prefetch_lib.prefetch_handle(
            self.checkpoint, _archer_config.device_memory_ratio
        )

        self.archer_config = _archer_config
        self.gpt_oss_offload_enabled = _gpt_oss_offload_enabled(
            getattr(self.config, "model_type", ""), self.archer_config
        )
        if _archer_config.trace_path is not None:
            self.expert_tracer.load_trace(_archer_config.trace_path)

        self.expert_executor = DistributedExpertExecutor(
            archer_config=_archer_config
        )

        return self

    def __enter__(self):
        def torch_index_select_decorator(orig_torch_index_select: Callable):
            @functools.wraps(orig_torch_index_select)
            def archer_torch_index_select(input, dim, index):
                return orig_torch_index_select(
                    input, dim, index.to(input.device)
                ).to(get_default_device())

            return archer_torch_index_select

        def apply_to_model_decorator(orig_apply_to_model: Callable) -> Callable:
            @functools.wraps(orig_apply_to_model)
            def archer_apply_to_model(cls, fn):
                for name, param in cls.named_parameters(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    param.data = torch.zeros(
                        1,
                        dtype=param.dtype,
                        device=param.device,
                        pin_memory=True,
                    )

                for name, buffer in cls.named_buffers(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    buffer.data = torch.zeros(
                        1,
                        dtype=buffer.dtype,
                        device=buffer.device,
                        pin_memory=True,
                    )

            return archer_apply_to_model

        def cast_classifier_decorator(
            orig_cast_classifier: Callable,
        ) -> Callable:
            @functools.wraps(orig_cast_classifier)
            def archer_cast_classifier(cls, *args, **kwargs):
                orig_data_ptr = cls.classifier.weight.data.data_ptr()
                if orig_data_ptr in self.offload_set:
                    self.offload_set.remove(
                        cls.classifier.weight.data.data_ptr()
                    )
                    orig_cast_classifier(cls, *args, **kwargs)
                    new_data_ptr = cls.classifier.weight.data.data_ptr()
                    self.offload_set.add(cls.classifier.weight.data.data_ptr())
                    self.archer_engine.update_tensor_map(
                        orig_data_ptr, new_data_ptr
                    )
                else:
                    orig_cast_classifier(cls, *args, **kwargs)
                    self.offload_set.add(cls.classifier.weight.data.data_ptr())

            return archer_cast_classifier

        # GPTQ Override
        QuantLinear._old_init = QuantLinear.__init__
        QuantLinear.__init__ = empty_param_init_decorator(QuantLinear.__init__)
        QuantLinearOld._old_init = QuantLinearOld.__init__
        QuantLinearOld.__init__ = empty_param_init_decorator(
            QuantLinearOld.__init__
        )

        # GPTQ Override
        QuantLinear._old_init = QuantLinear.__init__
        QuantLinear.__init__ = empty_param_init_decorator(QuantLinear.__init__)
        QuantLinearOld._old_init = QuantLinearOld.__init__
        QuantLinearOld.__init__ = empty_param_init_decorator(
            QuantLinearOld.__init__
        )

        self.cls._old_init = self.cls.__init__
        self.cls.__init__ = do_nothing_decorator(self.cls._old_init)
        torch.nn.modules.module.Module._old_apply = (
            torch.nn.modules.module.Module.apply
        )
        torch.nn.modules.module.Module.apply = apply_to_model_decorator(
            torch.nn.modules.module.Module._old_apply
        )

        torch._old_index_select = torch.index_select
        torch.index_select = torch_index_select_decorator(
            torch._old_index_select
        )
        torch.Tensor._old_index_select = torch.Tensor.index_select
        torch.Tensor.index_select = torch_index_select_decorator(
            torch.Tensor._old_index_select
        )

        self.cls._old_post_init = self.cls.post_init
        self.cls.post_init = do_nothing_decorator(self.cls._old_post_init)
        PreTrainedModel._old_post_init = PreTrainedModel.post_init
        PreTrainedModel.post_init = do_nothing_decorator(
            PreTrainedModel._old_post_init
        )

        activate_empty_init()

        transformers.models.nllb_moe.modeling_nllb_moe._old_sparse_mlp = (
            transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeSparseMLP
        )
        transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeSparseMLP = (
            SyncNllbMoeSparseMLP
        )
        transformers.models.mixtral.modeling_mixtral._old_sparse_mlp = (
            transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
        )
        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = (
            SyncMixtralSparseMoeBlock
        )

        transformers.models.qwen3_moe.modeling_qwen3_moe._old_sparse_mlp = transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = Qwen3MoEBlock

        transformers.models.qwen3_moe.modeling_qwen3_moe._old_qwen3_attention = transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention
        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention = (
            Qwen3PagedAttention
        )

        transformers.models.dbrx.modeling_dbrx._old_dbrx_ffn = (
            transformers.models.dbrx.modeling_dbrx.DbrxFFN
        )
        transformers.models.dbrx.modeling_dbrx.DbrxFFN = SyncDbrxFFNBlock

        transformers.models.olmoe.modeling_olmoe._old_olmoe_moe = (
            transformers.models.olmoe.modeling_olmoe.OlmoeSparseMoeBlock
        )
        transformers.models.olmoe.modeling_olmoe.OlmoeSparseMoeBlock = (
            SyncOlmoeMoEBlock
        )

        transformers.models.jamba.modeling_jamba._old_jamba_moe = (
            transformers.models.jamba.modeling_jamba.JambaSparseMoeBlock
        )
        transformers.models.jamba.modeling_jamba.JambaSparseMoeBlock = (
            SyncJambaMoEBlock
        )

        _dsv2_mod = transformers.models.deepseek_v2.modeling_deepseek_v2
        _dsv2_cls = getattr(_dsv2_mod, "DeepseekV2MoE", None) or getattr(
            _dsv2_mod, "DeepseekV2Moe", None
        )
        _dsv2_mod._old_deepseek_v2_moe = _dsv2_cls
        _dsv2_attr = (
            "DeepseekV2MoE"
            if hasattr(_dsv2_mod, "DeepseekV2MoE")
            else "DeepseekV2Moe"
        )
        setattr(_dsv2_mod, _dsv2_attr, SyncDeepseekV2MoEBlock)
        transformers.models.deepseek_v3.modeling_deepseek_v3._old_deepseek_v3_moe = transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3MoE
        transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3MoE = (
            SyncDeepseekV3MoEBlock
        )

        transformers.models.gpt_oss.modeling_gpt_oss._old_gpt_oss_mlp = (
            transformers.models.gpt_oss.modeling_gpt_oss.GptOssMLP
        )
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssMLP = SyncGptOssMLP

        _q35_mod = transformers.models.qwen3_5_moe.modeling_qwen3_5_moe
        _q35_mod._old_qwen3_5_sparse_moe = _q35_mod.Qwen3_5MoeSparseMoeBlock
        _q35_mod.Qwen3_5MoeSparseMoeBlock = SyncQwen3_5MoeSparseMoeBlock

        try:
            import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as _glm_mod

            _glm_mod._old_glm_moe_dsa_moe = _glm_mod.GlmMoeDsaMoE
            _glm_mod.GlmMoeDsaMoE = SyncGlmMoeDsaMoEBlock
        except (ImportError, AttributeError):
            pass

        def from_pretrained_decorator(
            orig_from_pretrained: Callable,
        ) -> Callable:
            @functools.wraps(orig_from_pretrained)
            def archer_from_pretrained(cls, *args, **kwargs):
                name_id_map_file = os.path.join(
                    self.checkpoint, "name_id_map.json"
                )

                self.model_name = model_name = args[0]

                checkpoint_path = self.ckpt_files[0] if self.ckpt_files else ""
                checkpoint_dir = (
                    os.path.dirname(checkpoint_path) if checkpoint_path else ""
                )
                self._quant_info = detect_quantization(
                    self.config, checkpoint_dir
                )
                if self._quant_info is not None:
                    validate_quantization_support(self._quant_info, model_name)

                self.num_layers, self.num_experts, self.num_encoder_layers = (
                    parse_moe_param(self.config)
                )

                if (
                    "qwen" in model_name.lower()
                    and self.config.model_type != "qwen3_5_moe"
                ):
                    self.prefetch_lib.init_moe_layer(
                        self.num_experts,
                        self.config.num_experts_per_tok,
                        1024,
                        self.config.hidden_size,
                        self.config.moe_intermediate_size,
                    )

                self.dtype = parse_expert_dtype(self.config)
                self.dtype_cls = resolve_config_dtype(self.config)
                if self.dtype_cls is None:
                    self.dtype_cls = torch.bfloat16

                if self.config.model_type == "deepseek_v3":
                    self.dtype_cls = torch.float8_e4m3fn
                    self.dtype = 3

                if (
                    not self.archer_engine.is_tensor_index_initialized()
                    or not os.path.exists(name_id_map_file)
                ):
                    print("Creating model from scratch ...", flush=True)

                    self.cls.__init__ = self.cls._old_init

                    empty_state_dict = {}
                    self.name_id_map = {}
                    for ckpt in tqdm(
                        self.ckpt_files,
                        desc="Loading checkpoint files",
                        smoothing=0,
                    ):
                        state_dict = {}
                        is_mxfp4_ckpt = False
                        if "safetensors" in ckpt:
                            with safe_open(
                                ckpt, framework="pt", device="cpu"
                            ) as f:
                                weight_keys = list(f.keys())
                                mxfp4_pairs = identify_mxfp4_pairs(weight_keys)
                                is_mxfp4_ckpt = bool(mxfp4_pairs)
                                if not is_mxfp4_ckpt:
                                    try:
                                        is_mxfp4_ckpt = is_mxfp4_quantized(
                                            self.config
                                        )
                                    except Exception:
                                        is_mxfp4_ckpt = False

                                for k in weight_keys:
                                    state_dict[k] = f.get_tensor(k)
                        else:
                            state_dict = torch.load(ckpt)

                        _remap_v5_batched_experts(state_dict, self.config)
                        _expand_gpt_oss_packed_experts(state_dict, self.config)

                        is_gptq_ckpt = is_gptq_quantized(self.config)
                        _arch0_cast = (
                            getattr(self.config, "architectures", None) or [""]
                        )[0]
                        is_glm_fp8_ckpt = (
                            "GlmMoeDsa" in _arch0_cast
                            and _has_fp8_blockwise(self.config)
                        )
                        self._cast_state_dict_tensors(
                            state_dict,
                            is_gptq_ckpt=is_gptq_ckpt,
                            is_mxfp4_ckpt=is_mxfp4_ckpt,
                            is_glm_fp8_ckpt=is_glm_fp8_ckpt,
                        )

                        if (
                            is_mxfp4_ckpt
                            and self.config.model_type != "gpt_oss"
                            and os.environ.get("MOE_INFINITY_MXFP4_DEQUANT", "")
                            == "1"
                        ):
                            from moe_infinity.kernel.mxfp4_gemm import (
                                mxfp4_dequantize,
                            )

                            dequant_pairs = identify_mxfp4_pairs(
                                list(state_dict.keys())
                            )
                            for blocks_key, scales_key in dequant_pairs:
                                base = blocks_key[: -len("_blocks")]
                                if (
                                    blocks_key not in state_dict
                                    or scales_key not in state_dict
                                ):
                                    continue
                                blocks = state_dict[blocks_key]
                                scales = state_dict[scales_key]
                                if blocks.numel() == 0 or scales.numel() == 0:
                                    continue
                                if blocks.dim() == 4:
                                    E, R, G, B = blocks.shape
                                    blocks = blocks.reshape(E, R, G * B)
                                flat_b = blocks.reshape(-1, blocks.shape[-1])
                                flat_s = scales.reshape(-1, scales.shape[-1])
                                unpacked_k = flat_b.shape[-1] * 2
                                bs = unpacked_k // max(flat_s.shape[-1], 1)
                                bf16 = mxfp4_dequantize(
                                    flat_b,
                                    flat_s,
                                    dtype=self.dtype_cls,
                                    block_size=bs,
                                )
                                E_dim = blocks.shape[0]
                                N_dim = blocks.shape[1]
                                K_dim = bf16.shape[-1]
                                bf16_3d = bf16.reshape(E_dim, N_dim, K_dim)
                                # Checkpoint: [E, N, K]. Model expects [E, K, N].
                                state_dict[base] = bf16_3d.transpose(
                                    1, 2
                                ).contiguous()
                                del state_dict[blocks_key]
                                del state_dict[scales_key]

                        arch0 = (
                            getattr(self.config, "architectures", None) or [""]
                        )[0]
                        is_glm_fp8 = (
                            "GlmMoeDsa" in arch0
                            and _has_fp8_blockwise(self.config)
                        )
                        if is_glm_fp8:
                            from moe_infinity.utils.fp8 import (
                                dequant_fp8_blockwise,
                            )

                            if not hasattr(self, "_glm_fp8_scales"):
                                self._glm_fp8_scales = {}
                            fp8_pairs = _identify_fp8_blockwise_pairs(
                                list(state_dict.keys())
                            )
                            for base_key, scale_key in fp8_pairs:
                                w = state_dict.get(base_key)
                                s = state_dict.get(scale_key)
                                if w is None or s is None:
                                    continue
                                if w.dtype != torch.float8_e4m3fn:
                                    continue
                                if _is_routed_expert_key(base_key):
                                    # Routed expert: keep FP8 in the host store;
                                    # the dispatcher dequantizes on-device.
                                    self._glm_fp8_scales[base_key] = s
                                else:
                                    # Required by the model: attention, dense MLP
                                    # and shared experts run in PyTorch (not the
                                    # dispatcher), so dequantize them to BF16.
                                    state_dict[base_key] = (
                                        dequant_fp8_blockwise(
                                            w,
                                            s,
                                            dtype=torch.bfloat16,
                                            block_size=128,
                                        )
                                    )
                                del state_dict[scale_key]

                        self._offload_state_dict(state_dict, empty_state_dict)

                        del state_dict
                        gc.collect()
                        torch.cuda.empty_cache()

                    if is_mxfp4_quantized(self.config):
                        blocks_aliases = {}
                        for key in list(self.name_id_map.keys()):
                            if key.endswith("_blocks"):
                                base_name = key[: -len("_blocks")]
                                blocks_aliases[base_name] = self.name_id_map[
                                    key
                                ]
                        self.name_id_map.update(blocks_aliases)

                    with open(name_id_map_file, "w") as f:
                        json.dump(self.name_id_map, f)
                    _write_model_signature(
                        self.checkpoint, model_name, self.config
                    )
                else:
                    print("Loading model from offload_path ...", flush=True)
                    _validate_model_signature(
                        self.checkpoint, model_name, self.config
                    )
                    self.cls.__init__ = self.cls._old_init
                    # load the name_id_map
                    with open(name_id_map_file, "r") as f:
                        self.name_id_map = json.load(f)

                    _arch0_reload = (
                        getattr(self.config, "architectures", None) or [""]
                    )[0]
                    if "GlmMoeDsa" in _arch0_reload and _has_fp8_blockwise(
                        self.config
                    ):
                        self._rebuild_glm_fp8_scales_from_ckpt()

                is_flash_attn_available = kwargs.get(
                    "is_flash_attn_available", False
                )
                _model_type = getattr(self.config, "model_type", "")
                _force_eager = _model_type == "glm_moe_dsa"
                model = cls._from_config(
                    self.config,
                    torch_dtype=self.dtype_cls
                    if self.config.model_type != "deepseek_v3"
                    else torch.bfloat16,
                    attn_implementation=(
                        "eager"
                        if _force_eager or not is_flash_attn_available
                        else "flash_attention_2"
                    ),
                )

                if self.config.model_type == "deepseek_v3":
                    model = model.to(torch.float8_e4m3fn)

                self.model = model

                base_model_prefix = model.base_model_prefix

                model = self._apply_quantized_model_conversion(model)

                self.expert_prefetcher = ExpertPrefetcher(self.config)
                self.expert_prefetcher.set_archer_engine(self.archer_engine)
                self.expert_dispatcher = self.prefetch_lib.expert_dispatcher(
                    self.num_experts,
                    self.num_layers,
                    self.dtype,
                    parse_expert_type(self.config),
                    self.archer_config.num_threads,
                )
                self.expert_prefetcher.expert_dispatcher = (
                    self.expert_dispatcher
                )

                for name, param in model.named_parameters(recurse=True):
                    # remove base_model_prefix from self.name_id_map
                    if name.startswith(base_model_prefix):
                        name_without_prefix = name[
                            (len(base_model_prefix) + 1) :
                        ]
                        if name_without_prefix in self.name_id_map:
                            self.name_id_map[name] = self.name_id_map[
                                name_without_prefix
                            ]
                            self.name_id_map.pop(name_without_prefix)
                    param.ar_id = self.name_id_map.get(name, None)

                # Tied embeddings: lm_head.weight shares storage with the
                # input embeddings, so it never got its own tensor id in the
                # loop above. Register it under whichever embedding submodule
                # this architecture actually has -- encoder/decoder for
                # encoder-decoder models (NLLB-MoE), a single embed_tokens
                # for decoder-only ones (Mixtral, Qwen3-MoE, etc.).
                if "lm_head.weight" not in self.name_id_map:
                    print(
                        "lm_head.weight not in name_id_map, add it as embed_tokens"
                    )
                    self.name_id_map["lm_head.weight"] = 0
                    model.lm_head.weight.ar_id = 0

                    if hasattr(model.model, "encoder") and hasattr(
                        model.model, "decoder"
                    ):
                        self.name_id_map["encoder.embed_tokens.weight"] = 0
                        self.name_id_map["decoder.embed_tokens.weight"] = 0
                        model.model.encoder.embed_tokens.weight.ar_id = 0
                        model.model.decoder.embed_tokens.weight.ar_id = 0
                    else:
                        self.name_id_map["embed_tokens.weight"] = 0
                        model.model.embed_tokens.weight.ar_id = 0

                self.expert_tensor_map = _make_expert_tensor_map(
                    self.name_id_map, self.config
                )
                self.expert_prefetcher.expert_tensor_map = (
                    self.expert_tensor_map
                )
                self.expert_prefetcher.expert_nbytes_map = (
                    _make_expert_nbytes_map(model, self.config)
                )

                # for deepseek and glm, we need to set the expert_tensor_map for the model
                first_k_dense_replace = 0
                if "deepseek" in model_name or "glm" in model_name.lower():
                    self.expert_prefetcher.first_k_dense_replace = (
                        self.config.first_k_dense_replace
                    )
                    first_k_dense_replace = self.config.first_k_dense_replace

                self.expert_executor.set_expert_dispatcher(
                    self.expert_dispatcher
                )
                if self.archer_config.speculative_prefetch:
                    self.expert_executor.set_prefetcher(self.expert_prefetcher)

                module_idx = 0
                self.expert_layer_modules = []
                # ATTENTION BACKEND INJECTION POINT (T15)
                # When self._enable_attention_offload is True, replace attention modules here.
                # Pattern: same as how SyncMixtralSparseMoeBlock replaces MixtralSparseMoeBlock.
                # Future: self._attention_backend.forward() called in place of module.forward()
                if (
                    self._enable_attention_offload
                    and self._attention_backend is not None
                ):
                    pass  # No-op: attention replacement not yet implemented
                for module in model.modules():
                    if (
                        isinstance(module, SyncNllbMoeSparseMLP)
                        or isinstance(module, SyncMixtralSparseMoeBlock)
                        or isinstance(module, SyncDeepseekV2MoEBlock)
                        or isinstance(module, SyncDeepseekV3MoEBlock)
                        or isinstance(module, Qwen3MoEBlock)
                        or isinstance(module, SyncDbrxFFNBlock)
                        or isinstance(module, SyncGptOssMLP)
                        or isinstance(module, SyncQwen3_5MoeSparseMoeBlock)
                        or isinstance(module, SyncOlmoeMoEBlock)
                        or isinstance(module, SyncJambaMoEBlock)
                        or isinstance(module, SyncGlmMoeDsaMoEBlock)
                    ):
                        module.archer_engine = self.archer_engine
                        module.archer_config = self.archer_config
                        module.is_gptq = is_gptq_quantized(self.config)
                        self.expert_modules.append(module)

                        use_dispatcher = (
                            not isinstance(module, SyncGptOssMLP)
                            or self.gpt_oss_offload_enabled
                        )
                        _wire_expert_collaborators(
                            module,
                            use_dispatcher,
                            expert_executor=self.expert_executor,
                            expert_prefetcher=self.expert_prefetcher,
                            expert_tracer=self.expert_tracer,
                            expert_predictor=self.expert_predictor,
                            expert_tensor_map=self.expert_tensor_map,
                        )

                        module.lib = self.prefetch_lib

                        self.expert_layer_modules.append(module)
                        module.layer_id = module_idx + first_k_dense_replace

                        module_idx += 1

                if getattr(self.config, "model_type", "") in (
                    "glm_moe_dsa",
                    "qwen3_5_moe",
                ):
                    self._load_resident_shared_experts(model)
                    for _name in list(self.name_id_map.keys()):
                        if self._is_shared_expert_param(_name):
                            del self.name_id_map[_name]

                if (
                    getattr(self.config, "model_type", "") == "gpt_oss"
                    and not self.gpt_oss_offload_enabled
                ):
                    self._load_resident_gpt_oss(model)

                self.setup_archer_hooks(model)
                self.deliver_fp8_scales_to_dispatcher()
                return model

            return archer_from_pretrained

        self.cls._old_from_pretrained = self.cls.from_pretrained
        self.cls.from_pretrained = classmethod(
            from_pretrained_decorator(self.cls.from_pretrained)
        )

        return self

    # clean up initialization hooks
    def __exit__(self, exc_type, exc_value, traceback):
        # GPTQ Override
        QuantLinear.__init__ = QuantLinear._old_init
        QuantLinearOld.__init__ = QuantLinearOld._old_init

        self.cls.__init__ = self.cls._old_init
        self.cls.from_pretrained = self.cls._old_from_pretrained
        torch.nn.modules.module.Module.apply = (
            torch.nn.modules.module.Module._old_apply
        )
        torch.index_select = torch._old_index_select
        torch.Tensor.index_select = torch.Tensor._old_index_select

        self.cls.post_init = self.cls._old_post_init
        PreTrainedModel.post_init = PreTrainedModel._old_post_init

        deactivate_empty_init()

        transformers.models.gpt_oss.modeling_gpt_oss.GptOssMLP = (
            transformers.models.gpt_oss.modeling_gpt_oss._old_gpt_oss_mlp
        )
        _q35_mod = transformers.models.qwen3_5_moe.modeling_qwen3_5_moe
        if hasattr(_q35_mod, "_old_qwen3_5_sparse_moe"):
            _q35_mod.Qwen3_5MoeSparseMoeBlock = _q35_mod._old_qwen3_5_sparse_moe

        try:
            import transformers.models.glm_moe_dsa.modeling_glm_moe_dsa as _glm_mod

            if hasattr(_glm_mod, "_old_glm_moe_dsa_moe"):
                _glm_mod.GlmMoeDsaMoE = _glm_mod._old_glm_moe_dsa_moe
        except (ImportError, AttributeError):
            pass

    def _is_shared_expert_param(self, name: str) -> bool:
        # DeepSeek names shared experts ".shared_experts." (plural); Qwen3.5-MoE
        # uses singular ".shared_expert." plus ".shared_expert_gate.". The
        # substring "shared_expert" matches all of them and no routed-expert key.
        if "shared_expert" in name:
            return True
        # Qwen3.5-MoE: keep the small text backbone (embed, hybrid attention,
        # norms, lm_head) resident and offload ONLY the routed experts. This
        # avoids the per-module begin/end offload path for the VLM embedding and
        # the GatedDeltaNet linear-attention layers, which are not compatible
        # with the native offload engine's begin/end lifecycle.
        if getattr(self.config, "model_type", "") == "qwen3_5_moe":
            if "language_model." in name:
                _, expert_id = parse_expert_id(name, self.config)
                return expert_id is None
            if name.endswith("lm_head.weight"):
                return True
        return False

    @torch.no_grad()
    def _load_resident_shared_experts(self, model):
        # Shared experts run inside the Python MoE block (not the C++ expert
        # executor) and are used on every MoE layer. The Archer offload path
        # leaves their live param as a [1] placeholder that is never
        # materialized before shared_experts(x) runs, so load their real
        # weights once and keep them resident instead.
        wanted = {
            name: param
            for name, param in model.named_parameters(recurse=True)
            if self._is_shared_expert_param(name)
        }
        if not wanted:
            return

        remaining = set(wanted)
        from moe_infinity.utils.fp8 import dequant_fp8_blockwise

        def _resolve(param, name, weight, scale):
            if weight.dtype == torch.float8_e4m3fn:
                if scale is None:
                    raise RuntimeError(
                        f"FP8 shared-expert weight {name!r} is missing its "
                        "_scale_inv; cannot dequantize resident shared experts."
                    )
                weight = dequant_fp8_blockwise(
                    weight, scale, dtype=param.dtype, block_size=128
                )
            return weight.to(dtype=param.dtype, device="cpu").contiguous()

        for ckpt in self.ckpt_files:
            if not remaining:
                break
            if ckpt.endswith(".safetensors"):
                with safe_open(ckpt, framework="pt", device="cpu") as f:
                    keys = set(f.keys())
                    for name in list(remaining):
                        if name not in keys:
                            continue
                        param = wanted[name]
                        scale_key = name + "_scale_inv"
                        scale = (
                            f.get_tensor(scale_key)
                            if scale_key in keys
                            else None
                        )
                        param.data = _resolve(
                            param, name, f.get_tensor(name), scale
                        )
                        param.requires_grad_(False)
                        param._moe_infinity_resident = True
                        remaining.remove(name)
            else:
                state = torch.load(ckpt, map_location="cpu")
                for name in list(remaining):
                    if name not in state:
                        continue
                    param = wanted[name]
                    param.data = _resolve(
                        param, name, state[name], state.get(name + "_scale_inv")
                    )
                    param.requires_grad_(False)
                    param._moe_infinity_resident = True
                    remaining.remove(name)
                del state

        if remaining:
            raise RuntimeError(
                f"Missing shared_experts weights: {sorted(remaining)[:5]}"
            )

    @torch.no_grad()
    def _load_resident_gpt_oss(self, model):
        # GPT-OSS is not wired into the C++ expert dispatcher (see the
        # SyncGptOssMLP guards in setup_archer_hooks / register_expert): its
        # SyncGptOssMLP.forward runs a resident Python expert loop reading the
        # _PackedExperts params directly. The Archer empty-init replaces those
        # params with [1] placeholders and never materializes them, so load the
        # real MXFP4 blocks/scales, biases, router, and attention sinks here and
        # keep them resident. MXFP4 packed weights stay uint8 output-major
        # ([E, N, K//2] blocks, [E, N, K//32] scales) - the layout the fused
        # kernel consumes without transposition.
        modules = [m for m in model.modules() if isinstance(m, SyncGptOssMLP)]
        if not modules:
            return

        num_devices = torch.cuda.device_count()
        target = get_device(num_devices - 1) if num_devices else "cpu"

        def _set(param, tensor, *, cast_dtype=None):
            data = tensor
            if cast_dtype is not None:
                data = data.to(cast_dtype)
            param.requires_grad_(False)
            param.data = data.to(device=target).contiguous()
            param._moe_infinity_resident = True

        loaded_names = set()

        def _reshape_blocks(t):
            if t.dim() == 4:
                E, N, G, B = t.shape
                return t.reshape(E, N, G * B)
            return t

        for module in modules:
            layer_id = module.layer_id
            prefix = f"model.layers.{layer_id}"
            experts = f"{prefix}.mlp.experts"
            wanted = {
                f"{experts}.gate_up_proj_blocks": (
                    module.experts.gate_up_proj,
                    _reshape_blocks,
                    None,
                ),
                f"{experts}.gate_up_proj_scales": (
                    module.experts.gate_up_proj_scales,
                    None,
                    None,
                ),
                f"{experts}.gate_up_proj_bias": (
                    module.experts.gate_up_proj_bias,
                    None,
                    self.dtype_cls,
                ),
                f"{experts}.down_proj_blocks": (
                    module.experts.down_proj,
                    _reshape_blocks,
                    None,
                ),
                f"{experts}.down_proj_scales": (
                    module.experts.down_proj_scales,
                    None,
                    None,
                ),
                f"{experts}.down_proj_bias": (
                    module.experts.down_proj_bias,
                    None,
                    self.dtype_cls,
                ),
                f"{prefix}.mlp.router.weight": (
                    module.router.weight,
                    None,
                    self.dtype_cls,
                ),
                f"{prefix}.mlp.router.bias": (
                    module.router.bias,
                    None,
                    self.dtype_cls,
                ),
            }
            remaining = dict(wanted)
            for ckpt in self.ckpt_files:
                if not remaining or not ckpt.endswith(".safetensors"):
                    continue
                with safe_open(ckpt, framework="pt", device="cpu") as f:
                    keys = set(f.keys())
                    for name in list(remaining):
                        if name not in keys:
                            continue
                        param, transform, cast_dtype = remaining[name]
                        t = f.get_tensor(name)
                        if transform is not None:
                            t = transform(t)
                        _set(param, t, cast_dtype=cast_dtype)
                        loaded_names.add(name)
                        del remaining[name]

            if remaining:
                raise RuntimeError(
                    f"Missing gpt-oss resident weights: {sorted(remaining)[:5]}"
                )

        sink_params = {
            name: param
            for name, param in model.named_parameters(recurse=True)
            if name.endswith("self_attn.sinks")
        }
        sink_remaining = dict(sink_params)
        for ckpt in self.ckpt_files:
            if not sink_remaining or not ckpt.endswith(".safetensors"):
                continue
            with safe_open(ckpt, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                for name in list(sink_remaining):
                    lookup = name
                    if lookup not in keys and name.startswith(
                        model.base_model_prefix + "."
                    ):
                        lookup = name[len(model.base_model_prefix) + 1 :]
                    if lookup not in keys:
                        continue
                    _set(
                        sink_remaining[name],
                        f.get_tensor(lookup),
                        cast_dtype=self.dtype_cls,
                    )
                    loaded_names.add(name)
                    loaded_names.add(lookup)
                    del sink_remaining[name]
        if sink_remaining:
            raise RuntimeError(
                f"Missing gpt-oss attention sinks: {sorted(sink_remaining)[:5]}"
            )

        for name in loaded_names:
            for candidate in (name, f"{model.base_model_prefix}.{name}"):
                self.name_id_map.pop(candidate, None)

    def _rebuild_glm_fp8_scales_from_ckpt(self):
        # Reload-from-store path: FP8 block scales are NOT persisted in the store,
        # so rebuild them from the checkpoint. Deliver every scale (superset): the
        # dispatcher only applies a scale to a weight that is actually stored FP8,
        # so scales for BF16-dequantized weights are simply ignored. This keeps
        # reload compatible with stores whose non-routed weights are still FP8.
        self._glm_fp8_scales = {}
        for ckpt in self.ckpt_files:
            if not ckpt.endswith(".safetensors"):
                continue
            with safe_open(ckpt, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                for k in keys:
                    if not k.endswith("_scale_inv"):
                        continue
                    base = k[: -len("_scale_inv")]
                    if base in keys:
                        self._glm_fp8_scales[base] = f.get_tensor(k)

    def get_topology(self, model):
        name_lst = []
        ret_dict = {}

        if getattr(self.config, "model_type", "") == "gpt_oss":
            gpt_oss_topology = _gpt_oss_expert_groups(
                self.name_id_map, self.config
            )
            name_lst.extend(name for name, _ in gpt_oss_topology)
            ret_dict.update(dict(gpt_oss_topology))

        for name, _ in model.named_parameters(recurse=True):
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                continue
            if match:
                if _is_routed_expert_key(name):
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    # Use negative indexing to get the component between the last third and second dot
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(
                                self.name_id_map[name]
                            )
                        else:
                            ret_dict[stored_name][expert_name] = [
                                self.name_id_map[name]
                            ]
                    else:
                        ret_dict[stored_name] = {
                            expert_name: [self.name_id_map[name]]
                        }
                        name_lst.append(stored_name)

                else:
                    match = re.match(r"(.*\.\d+\.)", name)
                    if match:
                        last_number_position = match.end() - 2
                    else:
                        matches = [
                            each_match
                            for each_match in re.finditer(r"\d", name)
                        ]
                        last_number_position = (
                            matches[-1].start() if matches else -1
                        )
                    stored_name = name[: last_number_position + 1]

                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)

            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]

                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)

        for name, _ in model.named_buffers(recurse=True):
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                continue
            if match:
                if _is_routed_expert_key(name):
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    # Use negative indexing to get the component between the last third and second dot
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(
                                self.name_id_map[name]
                            )
                        else:
                            ret_dict[stored_name][expert_name] = [
                                self.name_id_map[name]
                            ]
                    else:
                        ret_dict[stored_name] = {
                            expert_name: [self.name_id_map[name]]
                        }
                        name_lst.append(stored_name)

                else:
                    matches = [match for match in re.finditer(r"\d", name)]
                    last_number_position = (
                        matches[-1].start() if matches else -1
                    )
                    stored_name = name[: last_number_position + 1]

                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)
            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]

                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)

        for i in ret_dict.keys():
            if isinstance(ret_dict[i], dict):
                ret_dict[i] = list(ret_dict[i].values())

        topology = list(ret_dict.items())
        # Canonicalize DENSE nodes' tensor_ids to ascending id == store/offload
        # order so the C++ per-slot views match the physical store layout (fixes
        # q_a_layernorm[2048] <- router[256] mis-map when a dense node spans a
        # partition). Expert nodes carry positional [gate,up,down] tensors the
        # fused MoE kernel reads by slot, so they must KEEP their order.
        for _stored_name, id_groups in topology:
            if len(id_groups) != 1:
                continue
            id_groups[0].sort()
        return topology

    def setup_archer_hooks(self, model):
        for name, param in model.named_parameters(recurse=True):
            if name not in self.name_id_map:
                continue
            self.archer_engine.register(param.data, self.name_id_map[name])
            param.ar_id = self.name_id_map[name]
            self.offload_set.add(param.data.data_ptr())

            if "shared" in name:
                self.offload_exemption.add(param.data.data_ptr())

        for name, buffer in model.named_buffers(recurse=True):
            if name not in self.name_id_map:
                continue
            self.archer_engine.register(buffer.data, self.name_id_map[name])
            buffer.ar_id = self.name_id_map[name]
            self.offload_set.add(buffer.data.data_ptr())

        from moe_infinity.utils.topology import build_topology_specs

        topo = self.get_topology(model)
        sparse_count = sum(
            1 for _, t in topo if isinstance(t, list) and len(t) > 1
        )
        print(
            f"TOPO: {len(topo)} stages, {sparse_count} sparse",
            flush=True,
        )
        self.archer_engine.set_topology_v2(build_topology_specs(topo))
        print("TOPO: set_topology done", flush=True)

        @torch.no_grad()
        def _pre_forward_input_hook(module, input, kwargs, device, tensors):
            self._fetch_tensors_timed(tensors)
            new_args = copy_args_to_device(device, input)
            new_kwargs = copy_kwargs_to_device(device, kwargs)
            return new_args, new_kwargs

        @torch.no_grad()
        def _post_forward_output_hook(module, input, output, device, tensors):
            if isinstance(output, tuple):
                new_args = copy_args_to_device(device, output)
            elif isinstance(output, dict):
                new_args = copy_kwargs_to_device(device, output)
            else:
                new_args = output.to(device)
            return new_args

        def gen_args_hook(
            key, input_device_index, output_device_index, tensors
        ):
            keys = key.split(".")
            m = model
            for k in keys:
                if k.isdigit():
                    m = m[int(k)]
                else:
                    m = getattr(m, k)

            m.register_forward_pre_hook(
                functools.partial(
                    _pre_forward_input_hook,
                    device=input_device_index,
                    tensors=tensors,
                ),
                prepend=True,
                with_kwargs=True,
            )
            if "lm_head" in key:
                m.register_forward_hook(
                    functools.partial(
                        _post_forward_output_hook, device=0, tensors=tensors
                    ),
                    prepend=False,
                )

        expert_layer_id = 0
        if "deepseek" in self.model_name or "glm" in self.model_name.lower():
            expert_layer_id = self.config.first_k_dense_replace

        output_device_index = None
        for key, tensors in topo:
            if "shared" in key or "lm_head" in key:
                key = key.split(".")[0]
                output_device_index = 0

            if "expert" in key:
                for expert_idx, expert_tensors in enumerate(tensors):
                    input_device_index = (
                        self.archer_engine.get_node_default_device(
                            expert_tensors
                        )
                    )

                    if not is_gptq_quantized(self.config):
                        self.expert_dispatcher.register_expert(
                            expert_layer_id,
                            expert_idx,
                            expert_tensors,
                            os.path.join(self.checkpoint, f"expert.pt"),
                        )
                expert_layer_id += 1
            else:
                input_device_index = self.archer_engine.get_node_default_device(
                    tensors[0]
                )
                gen_args_hook(
                    key, input_device_index, output_device_index, tensors[0]
                )
                output_device_index = input_device_index

        # likely one of them should be enough but just to be safe
        self._register_hooks_recursively(model)

    def _generate_param_id(self):
        param_id = self.param_id
        self.param_id += 1
        return param_id

    def _generate_request_id(self):
        request_id = self.request_id
        self.request_id += 1
        return request_id

    def _offload_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        empty_state_dict: Dict[str, torch.Tensor],
    ) -> None:
        param_names = list(state_dict.keys())

        for param_name in param_names:
            self.name_id_map[param_name] = self._generate_param_id()
            if not self.archer_engine.is_tensor_offloaded(
                self.name_id_map[param_name]
            ):
                self.archer_engine.offload(
                    state_dict[param_name], self.name_id_map[param_name]
                )

        gc.collect()
        torch.cuda.empty_cache()

    def _cast_state_dict_tensors(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        is_gptq_ckpt: bool = False,
        is_mxfp4_ckpt: bool = False,
        is_glm_fp8_ckpt: bool = False,
    ) -> None:
        quant_info = getattr(self, "_quant_info", None)

        for k, v in state_dict.items():
            try:
                if is_mxfp4_ckpt and (
                    k.endswith("_blocks") or k.endswith("_scales")
                ):
                    state_dict[k] = v.to("cpu")
                    continue

                if is_glm_fp8_ckpt and (
                    k.endswith("_scale_inv") or v.dtype == torch.float8_e4m3fn
                ):
                    state_dict[k] = v.to("cpu")
                    continue

                if (is_gptq_ckpt and is_gptq_packed_tensor(k)) or (
                    not should_cast_tensor(k, quant_info)
                ):
                    state_dict[k] = v.to("cpu")
                    continue

                state_dict[k] = v.to(self.dtype_cls).to("cpu")

            except (RuntimeError, TypeError) as e:
                if k.endswith("_blocks") or k.endswith("_scales"):
                    warnings.warn(
                        f"Could not convert tensor {k} (dtype={v.dtype}) "
                        f"to {self.dtype_cls}: {e}. Keeping original dtype.",
                        UserWarning,
                        stacklevel=2,
                    )
                    state_dict[k] = v.to("cpu")
                    continue

                print(
                    f"Error converting {k} (device={v.device}) to {self.dtype_cls} on CPU: {e}",
                    flush=True,
                )
                raise

            except Exception as e:
                print(
                    f"Error converting {k} (device={v.device}) to {self.dtype_cls} on CPU: {e}",
                    flush=True,
                )
                raise

    def _apply_quantized_model_conversion(self, model):
        if self._quant_info is None:
            return model

        if self._quant_info.method == "gptq":
            self.quant_method = "gptq"
            quantization_config = getattr(
                self.config, "quantization_config", None
            )
            if not isinstance(quantization_config, dict):
                quantization_config = dict(self._quant_info.config_dict)
                self.config.quantization_config = quantization_config

            quantization_config.setdefault("quant_method", "gptq")
            quantization_config["use_exllama"] = False
            quantization_config["disable_exllama"] = True

            try:
                gptq_module = importlib.import_module("optimum.gptq")
                GPTQQuantizer = getattr(gptq_module, "GPTQQuantizer")
            except ImportError as e:
                raise ImportError(
                    "GPTQ model detected but 'optimum' is not installed. "
                    "Install with: pip install optimum[gptq]"
                ) from e

            optimum_quantizer = GPTQQuantizer.from_dict(quantization_config)
            return optimum_quantizer.convert_model(model)

        if self._quant_info.method == "awq":
            self.quant_method = "awq"
            return self._convert_model_for_awq(model)

        return model

    def _convert_model_for_awq(self, model):
        try:
            awq_module = importlib.import_module("awq")
        except ImportError as e:
            raise ImportError(
                "AWQ model detected but 'autoawq' is not installed. "
                "Install with: pip install autoawq"
            ) from e

        replace_fn = getattr(awq_module, "replace_linear_modules", None)
        if callable(replace_fn):
            converted_model = replace_fn(model)
            return converted_model if converted_model is not None else model

        autoawq_cls = getattr(awq_module, "AutoAWQForCausalLM", None)
        if autoawq_cls is not None:
            for fn_name in ("replace_linear_modules", "convert_model"):
                autoawq_fn = getattr(autoawq_cls, fn_name, None)
                if callable(autoawq_fn):
                    converted_model = autoawq_fn(model)
                    return (
                        converted_model
                        if converted_model is not None
                        else model
                    )

        try:
            awq_linear_module = importlib.import_module("awq.modules.linear")
        except Exception:
            awq_linear_module = None

        if awq_linear_module is not None:
            for fn_name in (
                "replace_linear_modules",
                "replace_with_awq_linear",
            ):
                linear_replace_fn = getattr(awq_linear_module, fn_name, None)
                if callable(linear_replace_fn):
                    converted_model = linear_replace_fn(model)
                    return (
                        converted_model
                        if converted_model is not None
                        else model
                    )

        return model

    @torch.no_grad()
    def _capture_kv_cache(self, seq_id: int, past_key_values):
        if not getattr(self, "_enable_kv_cache_offload", True):
            return
        if past_key_values is None:
            return

        captured = []
        stream_map: dict[str, object] = {}

        def _get_stream(device: torch.device):
            key = str(device)
            if key not in stream_map:
                stream_map[key] = torch.cuda.Stream(device=device)
            return stream_map[key]

        for layer_kv in past_key_values:
            if (
                isinstance(layer_kv, (list, tuple))
                and len(layer_kv) >= 2
                and isinstance(layer_kv[0], torch.Tensor)
                and isinstance(layer_kv[1], torch.Tensor)
            ):
                k, v = layer_kv[0], layer_kv[1]
                if k.is_cuda:
                    k_cpu = async_d2h(k, _get_stream(k.device))
                else:
                    k_cpu = k.to("cpu", non_blocking=True)
                if v.is_cuda:
                    v_cpu = async_d2h(v, _get_stream(v.device))
                else:
                    v_cpu = v.to("cpu", non_blocking=True)
                captured.append((k_cpu, v_cpu, k.device, v.device))
            else:
                captured.append(layer_kv)

        for stream in stream_map.values():
            wait_transfer(stream)

        self._captured_kv[seq_id] = tuple(captured)

    @torch.no_grad()
    def _reload_kv_cache(self, seq_id: int):
        if not getattr(self, "_enable_kv_cache_offload", True):
            return None
        captured = self._captured_kv.pop(seq_id, None)
        if captured is None:
            return None

        model_device = None
        model = getattr(self, "model", None)
        if model is not None:
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = None

        restored = []
        stream_map: dict[str, object] = {}

        def _get_stream(device: torch.device):
            key = str(device)
            if key not in stream_map:
                stream_map[key] = torch.cuda.Stream(device=device)
            return stream_map[key]

        for layer_kv in captured:
            if (
                isinstance(layer_kv, tuple)
                and len(layer_kv) == 4
                and isinstance(layer_kv[0], torch.Tensor)
                and isinstance(layer_kv[1], torch.Tensor)
            ):
                k_cpu, v_cpu, k_device, v_device = layer_kv
                k_target = (
                    k_device
                    if isinstance(k_device, torch.device)
                    else model_device
                )
                v_target = (
                    v_device
                    if isinstance(v_device, torch.device)
                    else model_device
                )

                if k_target is None:
                    k_target = k_cpu.device
                if v_target is None:
                    v_target = v_cpu.device

                if k_target.type == "cuda":
                    k = async_h2d(k_cpu, k_target, _get_stream(k_target))
                else:
                    k = k_cpu.to(k_target, non_blocking=True)

                if v_target.type == "cuda":
                    v = async_h2d(v_cpu, v_target, _get_stream(v_target))
                else:
                    v = v_cpu.to(v_target, non_blocking=True)

                restored.append((k, v))
            elif (
                isinstance(layer_kv, tuple)
                and len(layer_kv) == 2
                and isinstance(layer_kv[0], torch.Tensor)
                and isinstance(layer_kv[1], torch.Tensor)
            ):
                k_cpu, v_cpu = layer_kv
                target = (
                    model_device if model_device is not None else k_cpu.device
                )
                if target.type == "cuda":
                    stream = _get_stream(target)
                    restored.append(
                        (
                            async_h2d(k_cpu, target, stream),
                            async_h2d(v_cpu, target, stream),
                        )
                    )
                else:
                    restored.append(
                        (
                            k_cpu.to(target, non_blocking=True),
                            v_cpu.to(target, non_blocking=True),
                        )
                    )
            else:
                restored.append(layer_kv)

        for stream in stream_map.values():
            wait_transfer(stream)

        return tuple(restored)

    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count

        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively(child, count=count)

        @torch.no_grad()
        def _pre_forward_module_hook(module, args, kwargs):
            device_list = []

            for name, param in module.named_parameters(recurse=False):
                if param.data.data_ptr() not in self.offload_set:
                    num_devices = torch.cuda.device_count()
                    param.data = param.data.to(get_device(num_devices - 1))
                    continue

                self.offload_set.remove(param.data.data_ptr())
                self.archer_engine.begin(
                    self.request_id, param, getattr(param, "ar_id", 0xFFFFFFFF)
                )
                self.offload_set.add(param.data.data_ptr())

                device_list.append(param.data.device)

            for name, buf in module.named_buffers(recurse=False):
                if buf.data.data_ptr() not in self.offload_set:
                    num_devices = torch.cuda.device_count()
                    buf.data = buf.data.to(get_device(num_devices - 1))
                    continue

                self.offload_set.remove(buf.data_ptr())
                self.archer_engine.begin(
                    self.request_id, buf, getattr(buf, "ar_id", 0xFFFFFFFF)
                )
                self.offload_set.add(buf.data_ptr())

                device_list.append(buf.data.device)

            # KV CACHE RELOAD POINT (T16)
            # When enable_kv_cache_offload=True, reload swapped-out KV cache here.
            # Future: self._kv_cache_manager.swap_in(block_ids_for_this_layer)  # noqa: ERA001
            # This fires BEFORE each module's forward(), giving time to H2D transfer KV blocks.
            if (
                self._enable_kv_cache_offload
                and self._kv_cache_manager is not None
            ):
                _ = self._reload_kv_cache(seq_id=self.request_id)

        @torch.no_grad()
        def _post_forward_module_hook(module, input, output):
            device_list = []
            param_not_offload = set()
            for param in module.parameters(recurse=False):
                if param.data.data_ptr() not in self.offload_set:
                    param_not_offload.add(param.data.data_ptr())
                    continue

                self.offload_set.remove(param.data.data_ptr())
                self.archer_engine.end(
                    self.request_id, param, getattr(param, "ar_id", 0xFFFFFFFF)
                )
                self.offload_set.add(param.data.data_ptr())

                device_list.append(param.data.device)

            for buf in module.buffers(recurse=False):
                if buf.data_ptr() not in self.offload_set:
                    continue

                self.offload_set.remove(buf.data_ptr())
                self.archer_engine.end(
                    self.request_id, buf, getattr(buf, "ar_id", 0xFFFFFFFF)
                )
                self.offload_set.add(buf.data_ptr())

                device_list.append(buf.device)

            # KV CACHE CAPTURE POINT (T16)
            # When enable_kv_cache_offload=True, capture past_key_values here.
            # Future: extract output[1] (present_key_value) and call
            #         self._kv_cache_manager.allocate_blocks(seq_id, num_blocks)  # noqa: ERA001
            # This fires AFTER each module's forward(), when KV output is available.
            if (
                self._enable_kv_cache_offload
                and self._kv_cache_manager is not None
            ):
                past_key_values = None
                if isinstance(output, (list, tuple)):
                    if len(output) > 1:
                        past_key_values = output[1]
                else:
                    past_key_values = getattr(output, "past_key_values", None)
                self._capture_kv_cache(
                    seq_id=self.request_id, past_key_values=past_key_values
                )

            if param_not_offload:
                if device_list:
                    target = device_list[0]
                else:
                    num_devices = torch.cuda.device_count()
                    target = torch.device(get_device(num_devices - 1))
                if isinstance(output, torch.Tensor):
                    return output.to(target)

                return copy_args_to_device(target, output)

        # Pre forward hook
        self.forward_hooks.append(
            module.register_forward_pre_hook(
                _pre_forward_module_hook, with_kwargs=True
            )
        )

        # Post forward hook
        self.forward_hooks.append(
            module.register_forward_hook(_post_forward_module_hook)
        )

    # clean runtime hooks
    def clean_up(self):
        transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeSparseMLP = (
            transformers.models.nllb_moe.modeling_nllb_moe._old_sparse_mlp
        )

        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = (
            transformers.models.mixtral.modeling_mixtral._old_sparse_mlp
        )

        transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock = transformers.models.qwen3_moe.modeling_qwen3_moe._old_sparse_mlp

        if hasattr(
            transformers.models.qwen3_moe.modeling_qwen3_moe,
            "_old_qwen3_attention",
        ):
            transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeAttention = transformers.models.qwen3_moe.modeling_qwen3_moe._old_qwen3_attention

        transformers.models.dbrx.modeling_dbrx.DbrxFFN = (
            transformers.models.dbrx.modeling_dbrx._old_dbrx_ffn
        )

        transformers.models.olmoe.modeling_olmoe.OlmoeSparseMoeBlock = (
            transformers.models.olmoe.modeling_olmoe._old_olmoe_moe
        )

        transformers.models.jamba.modeling_jamba.JambaSparseMoeBlock = (
            transformers.models.jamba.modeling_jamba._old_jamba_moe
        )

        _dsv2_mod2 = transformers.models.deepseek_v2.modeling_deepseek_v2
        _dsv2_attr2 = (
            "DeepseekV2MoE"
            if hasattr(_dsv2_mod2, "DeepseekV2MoE")
            else "DeepseekV2Moe"
        )
        setattr(_dsv2_mod2, _dsv2_attr2, _dsv2_mod2._old_deepseek_v2_moe)
        transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3MoE = transformers.models.deepseek_v3.modeling_deepseek_v3._old_deepseek_v3_moe
        transformers.models.gpt_oss.modeling_gpt_oss.GptOssMLP = (
            transformers.models.gpt_oss.modeling_gpt_oss._old_gpt_oss_mlp
        )
        _q35_mod = transformers.models.qwen3_5_moe.modeling_qwen3_5_moe
        if hasattr(_q35_mod, "_old_qwen3_5_sparse_moe"):
            _q35_mod.Qwen3_5MoeSparseMoeBlock = _q35_mod._old_qwen3_5_sparse_moe
