# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportUnannotatedClassAttribute=false, reportUninitializedInstanceVariable=false, reportPrivateUsage=false, reportPrivateLocalImportUsage=false, reportUnusedImport=false, reportUnusedCallResult=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportExplicitAny=false, reportAny=false, reportArgumentType=false, reportOperatorIssue=false, reportImplicitStringConcatenation=false, reportUnnecessaryComparison=false, reportUnreachable=false, reportMissingTypeArgument=false, reportDeprecated=false, reportGeneralTypeIssues=false

import os
import warnings
from typing import Any, Dict, Optional, Union

import torch

warnings.filterwarnings("ignore")


def extract_context_feature(hidden_states, layer_ids=(1, 9, 17, 25, 33)):
    """Concatenate post-layer hidden states for DFlash drafter conditioning.

    Args:
        hidden_states: Per-layer hidden states from an HF forward with
            ``output_hidden_states=True``. Index 0 is the embedding output, so
            decoder layer ``layer_id``'s output lives at ``layer_id + 1``.
        layer_ids: Decoder layers to concatenate; defaults to the gpt-oss-120b
            DFlash contract ``(1, 9, 17, 25, 33)``.

    Returns:
        Tensor ``[batch, seq_len, len(layer_ids) * hidden_size]`` on the same
        device/dtype as the inputs (e.g. ``[B, L, 14400]`` for 120B).
    """
    if hidden_states is None or len(hidden_states) == 0:
        raise ValueError(
            "hidden_states must be a non-empty tuple; "
            "run the forward with output_hidden_states=True"
        )
    selected = []
    for layer_id in layer_ids:
        idx = int(layer_id) + 1
        if idx >= len(hidden_states):
            raise ValueError(
                f"layer_id {layer_id} maps to hidden_states[{idx}], but only "
                f"{len(hidden_states)} entries were returned"
            )
        selected.append(hidden_states[idx])
    # Multi-GPU (TP>1): the target's layers can span devices, so the selected
    # per-layer states may live on different GPUs. Gather them onto one device
    # before concatenating (no-op on a single device).
    gather_device = selected[0].device
    selected = [state.to(gather_device) for state in selected]
    return torch.cat(selected, dim=-1)


class MoE:
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model_name_or_path (`str` or `os.PathLike`): The model to load. It can be:
            - a name of HuggingFace Transformers model
            - a path to a file containing a whole model state dict
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        config (`Dict` or `os.PathLike`): The MoE-Infinity configuration. It can be:
            - a Python dictionary containing the configuration
            - a path to a JSON file containing the configuration

    Example:

    ```python
    >>> from moe_infinity import MoE

    >>> checkpoint = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    >>> config = "config.json"
    >>> model = MoE(checkpoint, config)

    >>> # You can now use the model as usual
    >>> input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids
    >>> outputs = model.generate(input_ids)
    ```
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        config: Union[str, os.PathLike, Dict] = None,
    ) -> None:
        try:
            from accelerate.utils.versions import is_torch_version
        except Exception:
            is_torch_version = None

        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required to load model checkpoints"
            ) from exc

        from moe_infinity.common.constants import MODEL_MAPPING_NAMES
        from moe_infinity.runtime import OffloadEngine
        from moe_infinity.utils import ArcherConfig, get_checkpoint_paths
        from moe_infinity.utils.hf_config import ensure_config_compat
        from moe_infinity.utils.quantization import (
            detect_quantization,
            validate_quantization_support,
        )

        # TODO: remove the torch version check once older versions are supported
        if is_torch_version is not None and not is_torch_version(">=", "2.0"):
            raise RuntimeError(
                "The `load_checkpoint_and_dispatch` function requires PyTorch >= 2.0. "
                "Please update PyTorch."
            )

        if config is None:
            default_config_path = os.path.join(
                os.path.dirname(__file__), "config.json"
            )
            if not os.path.exists(default_config_path):
                raise RuntimeError(
                    "The `load_checkpoint_and_dispatch` function requires a configuration file. "
                    f"Please provide a configuration file or create a default one at {default_config_path}."
                )
            config = default_config_path

        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        model_config = ensure_config_compat(model_config)

        quant_info = detect_quantization(model_config, "")
        if quant_info is not None:
            validate_quantization_support(quant_info, model_name_or_path)

        architectures = getattr(model_config, "architectures", None)
        if not architectures or not isinstance(architectures, list):
            raise RuntimeError("Unable to resolve model architecture")
        architecture = str(architectures[0]).lower()

        arch = None
        # Longest key first so specific arches ("qwen3_5", "deepseek_v3") win
        # over the shorter keys they contain ("qwen3", "deepseek").
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
        self.arch = arch
        model_cls = MODEL_MAPPING_NAMES[arch]
        if os.path.exists(model_name_or_path):
            model_path = model_name_or_path
            quant_info = detect_quantization(model_config, model_path)
            if quant_info is not None:
                validate_quantization_support(quant_info, model_name_or_path)
            checkpoint_paths = get_checkpoint_paths(model_path)
        else:
            checkpoint_paths = None
            # get the checkpoint download path from huggingface hub
            model_path = snapshot_download(
                model_name_or_path,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
                ignore_patterns=["flax*", "tf*"],
            )
            if model_path is None:
                raise RuntimeError(
                    f"The `snapshot_download` function could not find the checkpoint {model_name_or_path}. "
                    f"Please provide a valid checkpoint."
                )

            quant_info = detect_quantization(model_config, model_path)
            if quant_info is not None:
                validate_quantization_support(quant_info, model_name_or_path)

            checkpoint_paths = get_checkpoint_paths(model_path)

        if isinstance(config, dict):
            engine_config = ArcherConfig.load_from_json(config)
        else:
            engine_config = ArcherConfig.load_from_file(config)

        self.use_native_engine = bool(
            getattr(engine_config, "use_native_engine", True)
        )
        # GLM-DSA remains unsupported by the native engine. Qwen3.5 builds the
        # native components, but generate() below admits them only for greedy,
        # batch-1 DFlash; ordinary generation still uses HF's hybrid-cache path.
        if getattr(model_config, "model_type", "") == "glm_moe_dsa":
            self.use_native_engine = False
        default_max_seq_length = getattr(
            model_config, "max_position_embeddings", None
        )
        self.max_seq_length = (
            int(default_max_seq_length)
            if isinstance(default_max_seq_length, int)
            else 4096
        )

        native_components = self._build_native_components(
            model_config=model_config,
            engine_config=engine_config,
        )
        attention_backend = native_components.get("attention_backend")
        kv_cache_manager = native_components.get("kv_cache_manager")

        self.engine = OffloadEngine(
            engine_config.trace_capacity,
            model_config,
            attention_backend=attention_backend,
            enable_attention_offload=attention_backend is not None,
            kv_cache_manager=kv_cache_manager,
            enable_kv_cache_offload=(
                bool(
                    getattr(
                        engine_config,
                        "enable_kv_cache_offload",
                        False,
                    )
                )
                and kv_cache_manager is not None
            ),
        )
        self.engine.ckpt_files = checkpoint_paths
        is_flash_attn_available = False
        try:
            import flash_attn  # noqa: F401  # availability probe; import side effect only

            is_flash_attn_available = True

            if (
                arch == "deepseek"
                or arch == "deepseek_v3"
                or arch == "nllb"
                or arch == "gptoss"
                or arch == "qwen3_5"
            ):
                is_flash_attn_available = False
        except ImportError:
            print(
                "[WARNING] FlashAttention is not available in the current environment. Using default attention."
            )
        with self.engine.init(cls=model_cls, ar_config=engine_config):
            self.model = model_cls.from_pretrained(
                model_name_or_path,
                config=model_config,
                attn_implementation=(
                    "flash_attention_2" if is_flash_attn_available else "eager"
                ),
                is_flash_attn_available=is_flash_attn_available,
                trust_remote_code=True,
            )
        mla_cache = native_components.get("mla_cache")
        if mla_cache is not None:
            from moe_infinity.models.deepseek_mla_attention import (
                adapt_deepseek_model,
            )

            adapted = adapt_deepseek_model(
                self.model,
                mla_cache,
                enabled=True,
            )
            if not adapted:
                self._native_mla_cache = None

        self._native_rich_batch_capable = self._supports_native_rich_batch()

        self._ensure_generation_mixin()
        self.engine_config = engine_config

    def _ensure_generation_mixin(self):
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin

        if not hasattr(self.model, "generate"):
            cls = self.model.__class__
            self.model.__class__ = type(
                cls.__name__,
                (GenerationMixin, cls),
                {},
            )
        if getattr(self.model, "generation_config", None) is None:
            self.model.generation_config = GenerationConfig.from_model_config(
                self.model.config
            )

    @staticmethod
    def _resolve_model_int_attr(
        model_config: object, *names: str
    ) -> Optional[int]:
        get_text = getattr(model_config, "get_text_config", None)
        text_config = (
            get_text()
            if callable(get_text)
            else getattr(model_config, "text_config", None)
        )
        for cfg in (model_config, text_config):
            if cfg is None:
                continue
            for name in names:
                value = getattr(cfg, name, None)
                if isinstance(value, int):
                    return value
        return None

    @staticmethod
    def _resolve_torch_dtype(model_config: object) -> torch.dtype:
        torch_dtype = getattr(model_config, "dtype", None)
        if torch_dtype is None:
            torch_dtype = getattr(model_config, "torch_dtype", None)
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        if isinstance(torch_dtype, str):
            mapping = {
                "float16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "float": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            return mapping.get(torch_dtype.replace("torch.", ""), torch.float16)
        return torch.float16

    def _build_native_components(
        self, model_config: object, engine_config: object
    ) -> dict[str, object]:
        if not self.use_native_engine:
            self._native_memory_coordinator = None
            self._native_kv_cache_manager = None
            self._native_attention_backend = None
            self._native_paged_kv_storage = None
            self._native_mla_cache = None
            self._native_transfer_scheduler = None
            self._native_scheduler = None
            self._native_generation_engine = None
            self._native_kv_offload_coordinator = None
            self._native_expert_offload_coordinator = None
            return {}

        from moe_infinity.engine.generation_loop import GenerationEngine
        from moe_infinity.engine.scheduler import Scheduler
        from moe_infinity.engine.unified_transfer_scheduler import (
            UnifiedTransferScheduler,
        )
        from moe_infinity.memory.kv_cache_manager import KVCacheManager
        from moe_infinity.memory.memory_coordinator import MemoryCoordinator
        from moe_infinity.runtime.attention_backend import PagedAttentionBackend
        from moe_infinity.runtime.attention_types import KVCacheSpec

        model_num_layers = self._resolve_model_int_attr(
            model_config,
            "num_hidden_layers",
            "num_layers",
            "n_layer",
        )
        num_layers = model_num_layers if model_num_layers is not None else 1

        num_attention_heads = self._resolve_model_int_attr(
            model_config,
            "num_attention_heads",
            "n_head",
        )
        if num_attention_heads is None:
            num_attention_heads = 1

        num_kv_heads = self._resolve_model_int_attr(
            model_config,
            "num_key_value_heads",
            "num_kv_heads",
            "n_head_kv",
        )
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads

        head_dim = self._resolve_model_int_attr(model_config, "head_dim")
        if head_dim is None:
            hidden_size = self._resolve_model_int_attr(
                model_config, "hidden_size", "n_embd"
            )
            if hidden_size is None:
                hidden_size = max(1, num_attention_heads * 128)
            head_dim = max(1, hidden_size // max(1, num_attention_heads))

        vocab_size = self._resolve_model_int_attr(model_config, "vocab_size")
        if vocab_size is None:
            vocab_size = 32000

        eos_token_id = getattr(model_config, "eos_token_id", 2)
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else 2
        if not isinstance(eos_token_id, int):
            eos_token_id = 2

        if (
            self.use_native_engine
            and getattr(engine_config, "kv_cache_memory_ratio", 0.0) == 0.0
        ):
            setattr(engine_config, "kv_cache_memory_ratio", 0.15)
            warnings.warn(
                "kv_cache_memory_ratio was 0.0 with use_native_engine=True; auto-set to 0.15.",
                UserWarning,
                stacklevel=2,
            )

        memory_coordinator = MemoryCoordinator.from_config(
            {
                "device_memory_ratio": float(
                    getattr(engine_config, "device_memory_ratio", 0.75)
                ),
                "kv_cache_memory_ratio": float(
                    getattr(engine_config, "kv_cache_memory_ratio", 0.15)
                ),
                "use_native_engine": True,
            }
        )

        kv_spec = KVCacheSpec(
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            dtype=self._resolve_torch_dtype(model_config),
            block_size=16,
        )

        block_size_bytes = kv_spec.page_size_bytes * max(1, int(num_layers))
        num_gpu_blocks = max(
            1, memory_coordinator.compute_num_kv_blocks(block_size_bytes)
        )
        num_cpu_blocks = max(32, num_gpu_blocks * 2)

        from moe_infinity.models.deepseek_mla_attention import (
            is_deepseek_mla_eligible,
        )

        mla_enabled = is_deepseek_mla_eligible(
            model_config,
            enabled=bool(
                getattr(engine_config, "enable_deepseek_mla_paging", False)
            ),
        )

        kv_cache_manager = KVCacheManager(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=kv_spec.block_size,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mla_cache = None
        if mla_enabled:
            from moe_infinity.serving.mla_cache import MLAPagedKVCache

            mla_cache = MLAPagedKVCache(
                num_blocks=num_gpu_blocks,
                block_size=kv_spec.block_size,
                num_layers=int(num_layers),
                latent_dim=int(getattr(model_config, "kv_lora_rank")),
                rope_dim=int(getattr(model_config, "qk_rope_head_dim")),
                dtype=kv_spec.dtype,
                device=device,
            )
            attention_backend = None
        else:
            try:
                attention_backend = PagedAttentionBackend(
                    spec=kv_spec,
                    num_gpu_blocks=num_gpu_blocks,
                    device=device,
                )
            except Exception:
                attention_backend = None
        native_paged_kv_storage = self._build_native_paged_kv_storage(
            kv_spec=kv_spec,
            num_layers=int(num_layers),
            num_gpu_blocks=num_gpu_blocks,
            device=device,
        )
        transfer_scheduler = UnifiedTransferScheduler()
        kv_offload_coordinator = None
        if getattr(engine_config, "enable_kv_cache_offload", False):
            from moe_infinity.engine.kv_cache_offload_coordinator import (
                KVCacheOffloadCoordinator,
            )

            # kv_tensors=None: The actual KV tensor storage (PagedAttentionBackend
            # k_cache/v_cache) is not yet available at coordinator construction time.
            # Call coordinator.set_kv_tensors(tensors) after model initialisation
            # to enable real tensor copies. Until then, handlers are registered but
            # copy operations are no-ops (guarded by `if self._kv_tensors is None`).
            # This is intentional engine-path scaffolding per the plan scope.
            kv_offload_coordinator = KVCacheOffloadCoordinator(
                kv_tensors=None,
                block_pool=None,
                config=engine_config,
            )
            kv_offload_coordinator.register_with_scheduler(transfer_scheduler)

        expert_offload_coordinator = None
        if getattr(engine_config, "enable_expert_offload", False):
            from moe_infinity.engine.expert_offload_coordinator import (
                ExpertOffloadCoordinator,
            )

            # Opt-in engine-path scaffolding mirroring the KV coordinator above;
            # default-off (absent flag => False) leaves the standard path
            # unchanged. Registers EXPERT_FETCH/EXPERT_EVICT handlers; with no
            # real expert_prefetcher supplied the coordinator uses a stub, so
            # transfers are no-ops until a prefetcher is wired.
            expert_offload_coordinator = ExpertOffloadCoordinator(
                config=engine_config,
                num_devices=max(1, torch.cuda.device_count()),
            )
            expert_offload_coordinator.register_with_scheduler(
                transfer_scheduler
            )

        scheduler = Scheduler(
            kv_cache_manager=kv_cache_manager,
            transfer_scheduler=transfer_scheduler,
        )

        generation_engine = GenerationEngine(
            kv_cache_manager=kv_cache_manager,
            kv_spec=kv_spec,
            num_layers=int(num_layers),
            vocab_size=int(vocab_size),
            model_forward_fn=self._native_model_forward,
            eos_token_id=int(eos_token_id),
            max_seq_length=self.max_seq_length,
        )

        max_seq_length = self._resolve_model_int_attr(
            model_config,
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "seq_length",
        )
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
            generation_engine.max_seq_length = max_seq_length

        self._native_memory_coordinator = memory_coordinator
        self._native_kv_cache_manager = kv_cache_manager
        self._native_attention_backend = attention_backend
        self._native_paged_kv_storage = native_paged_kv_storage
        self._native_mla_cache = mla_cache
        self._native_transfer_scheduler = transfer_scheduler
        self._native_scheduler = scheduler
        self._native_generation_engine = generation_engine
        self._native_kv_offload_coordinator = kv_offload_coordinator
        self._native_expert_offload_coordinator = expert_offload_coordinator

        return {
            "memory_coordinator": memory_coordinator,
            "kv_cache_manager": kv_cache_manager,
            "attention_backend": attention_backend,
            "paged_kv_storage": native_paged_kv_storage,
            "mla_cache": mla_cache,
            "transfer_scheduler": transfer_scheduler,
            "kv_offload_coordinator": kv_offload_coordinator,
            "expert_offload_coordinator": expert_offload_coordinator,
            "scheduler": scheduler,
            "generation_engine": generation_engine,
        }

    def _build_native_paged_kv_storage(
        self,
        *,
        kv_spec: "KVCacheSpec",
        num_layers: int,
        num_gpu_blocks: int,
        device: torch.device,
    ) -> "Optional[PagedKVStorage]":
        from moe_infinity.runtime.paged_kv_storage import (
            PagedKVStorage,
            PagedKVStorageSpec,
        )

        try:
            spec = PagedKVStorageSpec(
                num_layers=num_layers,
                num_blocks=num_gpu_blocks,
                block_size=kv_spec.block_size,
                num_kv_heads=kv_spec.num_kv_heads,
                head_dim=kv_spec.head_dim,
                dtype=kv_spec.dtype,
                device=device,
            )
            return PagedKVStorage(spec)
        except Exception:
            return None

    def decode_graph_capability(self):
        from moe_infinity.runtime.attention_types import DecodeGraphCapability

        engine = getattr(self, "engine", None)
        engine_capability_fn = getattr(engine, "decode_graph_capability", None)
        if callable(engine_capability_fn):
            engine_capability = engine_capability_fn()
            if not engine_capability.safe:
                return engine_capability

        transfer_scheduler = getattr(self, "_native_transfer_scheduler", None)
        if transfer_scheduler is not None:
            return DecodeGraphCapability(False, "transfer_scheduler")

        if getattr(self, "_native_kv_offload_coordinator", None) is not None:
            return DecodeGraphCapability(False, "kv_offload")

        storage = getattr(self, "_native_paged_kv_storage", None)
        if storage is None:
            return DecodeGraphCapability(False, "native_paged_required")

        return DecodeGraphCapability(
            True, "eligible", storage_owner_id=storage.owner_id
        )

    def _resolve_native_input_device(self) -> torch.device:
        """Input device for the native forward (mirrors engine._resolve_device).

        The OffloadEngine-managed backbone is resident on the LAST visible GPU,
        so hard-coding ``cuda:0`` mismatches ``embed_tokens`` on multi-GPU runs.
        """
        if not torch.cuda.is_available():
            model_device = getattr(self.model, "device", None)
            if isinstance(
                model_device, torch.device
            ) and model_device.type not in ("meta", "cpu"):
                return model_device
            return torch.device("cpu")

        get_embed = getattr(self.model, "get_input_embeddings", None)
        if callable(get_embed):
            try:
                weight = getattr(get_embed(), "weight", None)
                embed_device = getattr(weight, "device", None)
            except Exception:
                embed_device = None
            if (
                isinstance(embed_device, torch.device)
                and embed_device.type == "cuda"
            ):
                return embed_device

        model_device = getattr(self.model, "device", None)
        if (
            isinstance(model_device, torch.device)
            and model_device.type == "cuda"
        ):
            return model_device

        return torch.device(f"cuda:{torch.cuda.device_count() - 1}")

    def _native_model_forward(
        self, token_ids: list[int], _attention_metadata: object
    ) -> torch.Tensor:
        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        input_tensor = input_tensor.to(self._resolve_native_input_device())

        is_prefill = True
        if _attention_metadata is not None:
            is_prefill = bool(getattr(_attention_metadata, "is_prefill", True))

        paged_attention_classes = self._get_paged_attention_classes()
        use_paged_context = bool(
            paged_attention_classes
            and _attention_metadata is not None
            and getattr(self, "_native_attention_backend", None) is not None
        )

        extra_kwargs: dict = {}

        if not use_paged_context:
            # Non-paged path: use HF's KV cache for autoregressive decode.
            # During prefill, use_cache=True captures past_key_values for
            # subsequent decode steps.  During decode, the cached KV provides
            # context from all prior tokens and lets HF auto-compute correct
            # position_ids.
            extra_kwargs["use_cache"] = True
            if not is_prefill:
                cached_kv = getattr(self, "_cached_past_key_values", None)
                if cached_kv is not None:
                    extra_kwargs["past_key_values"] = cached_kv
        else:
            # Paged path: the paged attention backend manages its own KV
            # cache, but the model still needs correct position_ids for
            # rotary embeddings during decode steps.
            if not is_prefill and _attention_metadata is not None:
                seq_lens = getattr(_attention_metadata, "seq_lens", None)
                if seq_lens is not None and seq_lens.numel() > 0:
                    total_len = int(seq_lens[0].item())
                    start_pos = total_len - len(token_ids)
                    extra_kwargs["position_ids"] = torch.arange(
                        start_pos,
                        total_len,
                        device=input_tensor.device,
                        dtype=torch.long,
                    ).unsqueeze(0)

        with torch.no_grad():
            if not use_paged_context:
                outputs = self.model(input_tensor, **extra_kwargs)
            else:
                backend = self._native_attention_backend
                for attn_cls in paged_attention_classes:
                    attn_cls.set_paged_context(backend, _attention_metadata)
                try:
                    outputs = self.model(input_tensor, **extra_kwargs)
                finally:
                    for attn_cls in paged_attention_classes:
                        attn_cls.clear_paged_context()

        # Capture HF KV cache for next decode step (non-paged path only).
        if not use_paged_context:
            past_kv = getattr(outputs, "past_key_values", None)
            if past_kv is not None:
                self._cached_past_key_values = past_kv

        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, tuple) and outputs:
            logits = outputs[0]
        if logits is None:
            raise RuntimeError("model forward did not return logits")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("model logits must be a torch.Tensor")
        if logits.ndim == 3:
            return logits[0, -len(token_ids) :, :].detach().to("cpu")
        if logits.ndim == 2:
            return logits.detach().to("cpu")
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")

    def _native_model_forward_rich(
        self,
        token_ids: list[int] | torch.Tensor,
        _attention_metadata: object = None,
        logits_to_keep: int = 0,
    ) -> tuple[torch.Tensor, tuple, object] | object:
        """On-device forward for speculative decoding: hidden-state capture.

        Single HF forward with ``output_hidden_states=True`` returning
        ``(logits, hidden_states, past_key_values)`` on the model device —
        unlike `_native_model_forward`, nothing is detached to CPU. The cache
        contract mirrors the baseline (non-paged path reads/writes
        ``self._cached_past_key_values``) so callers can roll the returned
        ``DynamicCache`` back via ``crop()``.

        ``logits_to_keep`` passthrough: ``1`` for the anchor/prefill step
        (last-position logits only); ``0`` (the default) keeps full logits and
        MUST be used for the verify step. Experts flow through the exact same
        ``self.model(...)`` call as the baseline path, so the standard
        ExpertExecutor dispatch (and its ``speculative_prefetch`` hook) is
        preserved — nothing here bypasses expert dispatch.
        """
        batched = isinstance(token_ids, torch.Tensor)
        if batched:
            if token_ids.ndim != 2:
                raise ValueError(
                    "batched rich token_ids must have shape [batch, seq]"
                )
            input_tensor = token_ids.to(dtype=torch.long)
        else:
            input_tensor = torch.tensor([token_ids], dtype=torch.long)
        input_tensor = input_tensor.to(self._resolve_native_input_device())

        is_prefill = True
        if _attention_metadata is not None:
            is_prefill = bool(getattr(_attention_metadata, "is_prefill", True))

        mla_attention_modules = self._get_mla_attention_modules()
        use_mla_context = bool(
            mla_attention_modules
            and _attention_metadata is not None
            and getattr(self, "_native_mla_cache", None) is not None
            and all(
                hasattr(_attention_metadata, name)
                for name in (
                    "block_tables",
                    "seq_lens",
                    "slot_mapping",
                    "is_prefill",
                )
            )
        )
        paged_attention_classes = self._get_paged_attention_classes()
        use_paged_context = bool(
            not use_mla_context
            and paged_attention_classes
            and _attention_metadata is not None
            and getattr(self, "_native_attention_backend", None) is not None
        )

        extra_kwargs: dict = {"output_hidden_states": True}
        if logits_to_keep:
            extra_kwargs["logits_to_keep"] = int(logits_to_keep)
        if batched and _attention_metadata is not None:
            attention_mask = getattr(
                _attention_metadata, "attention_mask", None
            )
            position_ids = getattr(_attention_metadata, "position_ids", None)
            if attention_mask is not None:
                extra_kwargs["attention_mask"] = attention_mask.to(
                    input_tensor.device
                )
            if position_ids is not None:
                extra_kwargs["position_ids"] = position_ids.to(
                    input_tensor.device
                )

        if not use_paged_context and not use_mla_context:
            # Same HF KV-cache contract as the baseline: prefill captures
            # past_key_values; decode steps consume the cached KV.
            extra_kwargs["use_cache"] = True
            if not is_prefill or batched:
                cached_kv = None
                handles = getattr(_attention_metadata, "cache_handles", ())
                if handles and all(handle is handles[0] for handle in handles):
                    cached_kv = handles[0]
                if cached_kv is None:
                    cached_kv = getattr(self, "_cached_past_key_values", None)
                if cached_kv is not None:
                    extra_kwargs["past_key_values"] = cached_kv
        else:
            extra_kwargs["use_cache"] = False
            if not is_prefill and _attention_metadata is not None:
                seq_lens = getattr(_attention_metadata, "seq_lens", None)
                if seq_lens is not None and seq_lens.numel() > 0:
                    total_len = int(seq_lens[0].item())
                    start_pos = total_len - len(token_ids)
                    extra_kwargs["position_ids"] = torch.arange(
                        start_pos,
                        total_len,
                        device=input_tensor.device,
                        dtype=torch.long,
                    ).unsqueeze(0)

        with torch.no_grad():
            if not use_paged_context and not use_mla_context:
                outputs = self.model(input_tensor, **extra_kwargs)
            elif use_mla_context:
                from moe_infinity.models.deepseek_mla_attention import (
                    clear_deepseek_mla_context,
                    set_deepseek_mla_context,
                )

                for module in mla_attention_modules:
                    set_deepseek_mla_context(module, _attention_metadata)
                try:
                    outputs = self.model(input_tensor, **extra_kwargs)
                finally:
                    for module in mla_attention_modules:
                        clear_deepseek_mla_context(module)
            else:
                backend = self._native_attention_backend
                for attn_cls in paged_attention_classes:
                    attn_cls.set_paged_context(backend, _attention_metadata)
                try:
                    outputs = self.model(input_tensor, **extra_kwargs)
                finally:
                    for attn_cls in paged_attention_classes:
                        attn_cls.clear_paged_context()

        if not use_paged_context and not use_mla_context:
            past_kv = getattr(outputs, "past_key_values", None)
            if past_kv is not None:
                self._cached_past_key_values = past_kv

        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, tuple) and outputs:
            logits = outputs[0]
        if logits is None:
            raise RuntimeError("model forward did not return logits")
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("model logits must be a torch.Tensor")

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "model forward did not return hidden_states; "
                "output_hidden_states=True is required"
            )

        if use_mla_context or batched:
            from moe_infinity.spec_decode.protocols import RichForwardResult

            cache_handle = (
                self._native_mla_cache
                if use_mla_context
                else getattr(outputs, "past_key_values", None)
            )
            supplied_handles = tuple(
                getattr(_attention_metadata, "cache_handles", ())
            )
            if use_mla_context and supplied_handles:
                row_handles = supplied_handles
            else:
                row_handles = (cache_handle,) * int(input_tensor.shape[0])

            return RichForwardResult(
                logits=logits,
                hidden_states=tuple(hidden_states),
                cache_handle=cache_handle,
                cache_handles=row_handles,
                row_offsets=tuple(
                    getattr(_attention_metadata, "row_offsets", ())
                ),
                row_lengths=tuple(
                    getattr(_attention_metadata, "row_lengths", ())
                ),
            )
        past_key_values = getattr(outputs, "past_key_values", None)
        return logits, hidden_states, past_key_values

    def _get_mla_attention_modules(self) -> list[torch.nn.Module]:
        names = {"DeepseekV2MLAPagedAttention", "DeepseekV3MLAPagedAttention"}
        modules_fn = getattr(self.model, "modules", None)
        if not callable(modules_fn):
            return []
        return [
            module
            for module in modules_fn()
            if module.__class__.__name__ in names
        ]

    def _get_paged_attention_classes(self) -> list[type[Any]]:
        paged_class_names = {
            "DeepseekV2PagedAttention",
            "DeepseekV3PagedAttention",
        }
        classes: list[type[Any]] = []
        seen: set[type[Any]] = set()

        modules_fn = getattr(self.model, "modules", None)
        if not callable(modules_fn):
            return classes

        for module in modules_fn():
            cls = module.__class__
            if cls in seen:
                continue
            if cls.__name__ not in paged_class_names:
                continue
            if not hasattr(cls, "set_paged_context") or not hasattr(
                cls, "clear_paged_context"
            ):
                continue
            seen.add(cls)
            classes.append(cls)

        return classes

    def _supports_native_rich_batch(self) -> bool:
        """Fail-closed declaration for the dense row-aware rich contract."""
        config = getattr(self.model, "config", None)
        if any(
            bool(getattr(config, name, False))
            for name in (
                "hybrid_attention",
                "sliding_window_pattern",
                "recurrent_chunk_size",
            )
        ):
            return False
        # Existing paged wrappers retain engine ownership and are driven by
        # ModelRunner; MLA and Qwen/hybrid rollback are intentionally not
        # widened through the dense direct-generation backend.
        return (
            not self._get_paged_attention_classes()
            and not self._get_mla_attention_modules()
        )

    def _configure_hook(self, input_ids: torch.LongTensor):
        if self.arch == "mixtral":
            import moe_infinity.models.mixtral  # noqa: F401

        batch_size = input_ids.shape[0]
        self.seq_id_list = [
            self.engine.expert_tracer.create_entry() for _ in range(batch_size)
        ]
        for module in self.engine.expert_layer_modules:
            module.seq_id_list = self.seq_id_list

    def _resolve_spec_strategy(self, speculative_draft):
        """Attach/detach the DFlash speculator on the native engine per call.

        ``speculative_draft`` may be a drafter checkpoint path (loaded via
        ``DFlashSpeculator``), an already-built ``DFlashSpeculator``, or an
        instantiated draft module (wrapped via ``from_models``). Construction
        is memoized by the passed value; ``None``/``False`` detaches so the
        standard path runs, without discarding the memoized speculator.
        """
        engine = self._native_generation_engine
        if engine is None:
            raise RuntimeError(
                "cannot configure speculative decoding without a native "
                "generation engine"
            )
        if not speculative_draft:
            engine.spec_strategy = None
            return
        cached = getattr(self, "_dflash_speculator", None)
        source = getattr(self, "_dflash_speculator_source", None)
        same = source is speculative_draft or (
            isinstance(source, str)
            and isinstance(speculative_draft, (str, os.PathLike))
            and source == str(speculative_draft)
        )
        if cached is None or not same:
            import importlib

            DFlashSpeculator = getattr(
                importlib.import_module("moe_infinity.spec_decode.dflash"),
                "DFlashSpeculator",
            )

            if isinstance(speculative_draft, DFlashSpeculator):
                cached = speculative_draft
            elif isinstance(speculative_draft, (str, os.PathLike)):
                cached = DFlashSpeculator(self, str(speculative_draft))
            else:
                cached = DFlashSpeculator.from_models(self, speculative_draft)
            self._dflash_speculator = cached
            self._dflash_speculator_source = speculative_draft
        engine.spec_strategy = cached

    def generate(self, input_ids: torch.LongTensor, **kwargs) -> Any:
        """
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation. If `past` is used, only `bos_token_id` is used as
                prompt.
            speculative_draft: optional DFlash drafter (checkpoint path,
                `DFlashSpeculator`, or draft module). When given, greedy
                batch-1 decoding routes through the native speculative
                strategy on the engine; omitted/`None`/`False` uses the
                standard path (per-call, never sticky).
            **kwargs: Additional arguments for the generation method. Check the HuggingFace documentation of the model's
                `generate` method for the supported arguments.

        Returns:
            `torch.LongTensor` of shape `(batch_size, sequence_length)`:
                The generated sequences. Sequences shorter than `min_length` are padded with `pad_token_id`.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("default", DeprecationWarning)
            warnings.warn(
                "MoE.generate() is deprecated. Use MoE.serve() for continuous batching "
                "with higher throughput. MoE.generate() will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        speculative_draft = kwargs.pop("speculative_draft", None)

        generation_config = kwargs.get("generation_config")

        def generation_value(name: str, default: Any) -> Any:
            if name in kwargs:
                return kwargs[name]
            if generation_config is not None:
                value = getattr(generation_config, name, None)
                if value is not None:
                    return value
            return default

        model_type = getattr(
            getattr(self.model, "config", None), "model_type", ""
        )
        is_qwen35 = model_type == "qwen3_5_moe"
        do_sample = generation_value("do_sample", None)
        sampling_temperature = (
            0.0
            if do_sample is False
            else float(generation_value("temperature", 1.0))
        )
        top_p = float(generation_value("top_p", 1.0))
        top_k = int(generation_value("top_k", 0) or 0)
        is_greedy = sampling_temperature == 0.0 and top_p == 1.0 and top_k == 0
        qwen35_dflash = bool(speculative_draft) and is_greedy
        native_engine = self._native_generation_engine
        native_for_call = (
            self.use_native_engine
            and native_engine is not None
            and input_ids.ndim == 2
            and input_ids.shape[0] == 1
            and (not is_qwen35 or qwen35_dflash)
        )

        if speculative_draft and is_qwen35 and not is_greedy:
            raise ValueError(
                "qwen3_5_moe speculative_draft (DFlash) requires "
                "greedy decoding (do_sample=False or temperature=0, "
                "top_p=1, top_k=0)"
            )

        if (
            speculative_draft
            and is_greedy
            and input_ids.ndim == 2
            and input_ids.shape[0] > 1
        ):
            if not self.use_native_engine or native_engine is None:
                raise ValueError(
                    "speculative_draft (DFlash) requires the MoE-Infinity "
                    "native engine (use_native_engine=True)"
                )
            self._resolve_spec_strategy(speculative_draft)
            try:
                speculator = getattr(self, "_dflash_speculator", None)
                if speculator is None:
                    raise RuntimeError("DFlash speculator was not configured")

                attention_mask = kwargs.get("attention_mask")
                if attention_mask is None:
                    prompt_lengths = [int(input_ids.shape[1])] * int(
                        input_ids.shape[0]
                    )
                else:
                    if tuple(attention_mask.shape) != tuple(input_ids.shape):
                        raise ValueError(
                            f"attention_mask shape {tuple(attention_mask.shape)} != "
                            f"input_ids shape {tuple(input_ids.shape)}"
                        )
                    prompt_lengths = [
                        int(value)
                        for value in attention_mask.to(dtype=torch.long)
                        .sum(dim=1)
                        .tolist()
                    ]
                for prompt_length in prompt_lengths:
                    if prompt_length > self.max_seq_length:
                        raise ValueError(
                            f"prompt length {prompt_length} exceeds max_seq_length "
                            f"{self.max_seq_length}"
                        )

                max_tokens = generation_value("max_new_tokens", None)
                if max_tokens is None:
                    max_tokens = generation_value("max_tokens", 256)
                stop_token_ids = kwargs.get("stop_token_ids")
                if stop_token_ids is None:
                    stop_token_ids = generation_value("eos_token_id", None)
                if isinstance(stop_token_ids, bool):
                    raise ValueError(
                        "eos_token_id/stop_token_ids cannot be boolean"
                    )
                if isinstance(stop_token_ids, int):
                    stop_token_ids = [stop_token_ids]
                generator = kwargs.get("generator")
                self._configure_hook(input_ids)
                self._cached_past_key_values = None
                self.model.eval()
                output = speculator.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=sampling_temperature,
                    stop_token_ids=stop_token_ids,
                    top_k=top_k,
                    top_p=top_p,
                    attention_mask=attention_mask,
                    generator=generator,
                )
                self.last_dflash_traces = getattr(
                    speculator, "last_session_traces", ()
                )
                return output
            finally:
                self._cached_past_key_values = None
                native_engine.spec_strategy = None

        if not native_for_call:
            if speculative_draft:
                if input_ids.ndim == 2 and input_ids.shape[0] != 1:
                    raise NotImplementedError(
                        "speculative_draft (DFlash) v1 supports batch==1 "
                        f"only; got batch size {input_ids.shape[0]}"
                    )
                raise ValueError(
                    "speculative_draft (DFlash) requires the MoE-Infinity "
                    "native engine (use_native_engine=True)"
                )
            self._configure_hook(input_ids)
            self.model.eval()
            with torch.no_grad():
                return self.model.generate(input_ids, **kwargs)

        if native_engine is None:
            raise RuntimeError(
                "native generation engine unexpectedly unavailable"
            )

        from moe_infinity.engine.types import SamplingParams

        self._resolve_spec_strategy(speculative_draft)
        self._configure_hook(input_ids)
        self._cached_past_key_values = None
        self.model.eval()

        prompt_token_ids = [int(token) for token in input_ids[0].tolist()]
        if len(prompt_token_ids) > self.max_seq_length:
            raise ValueError(
                f"prompt length {len(prompt_token_ids)} exceeds max_seq_length {self.max_seq_length}"
            )

        max_tokens = generation_value("max_new_tokens", None)
        if max_tokens is None:
            max_tokens = generation_value("max_tokens", 256)
        sampling_params = SamplingParams(
            temperature=sampling_temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=int(max_tokens) if max_tokens is not None else 256,
        )
        try:
            result = native_engine.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
        finally:
            self._cached_past_key_values = None

        output_ids = prompt_token_ids + result.output_token_ids
        return torch.tensor(
            [output_ids],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        device_memory_ratio: float = 0.75,
        kv_cache_ratio: float = 0.25,
        max_batch_size: int = 32,
        enable_prefix_caching: bool = False,
        offload_dir: Optional[str] = None,
        speculative_draft: Optional[object] = None,
        enable_deepseek_mla_paging: bool = False,
        max_resident_paged_speculative_sessions: int = 1,
        min_free_mla_blocks_after_admission: int = 1,
    ) -> None:
        """
        Start the OpenAI-compatible continuous batching server.

        This is the recommended entry point for production serving.
        Replaces the deprecated MoE.generate() for serving use cases.

        Args:
            host: Server host (default: 0.0.0.0)
            port: Server port (default: 8000)
            device_memory_ratio: GPU memory fraction for caching (default: 0.75)
            kv_cache_ratio: Fraction of device_memory_ratio for KV cache (default: 0.25)
            max_batch_size: Maximum concurrent sequences (default: 32)
            enable_prefix_caching: Enable hash-based prefix caching (default: False)
            enable_deepseek_mla_paging: Enable default-off DeepSeek V2/V3 MLA paging.
            max_resident_paged_speculative_sessions: Resident paged-session cap.
            min_free_mla_blocks_after_admission: Free blocks retained after admission.
            offload_dir: Path to offload directory (required)
            speculative_draft: Optional DFlash checkpoint, speculator, or draft
                module for greedy batch-1 serving.
        """

        if offload_dir is None:
            raise ValueError("offload_dir is required for MoE.serve()")

        import importlib

        from moe_infinity.entrypoints.openai import api_server_v2

        serving_speculator = None
        if speculative_draft:
            self._resolve_spec_strategy(speculative_draft)
            serving_speculator = getattr(self, "_dflash_speculator", None)

        model_name = getattr(
            getattr(self.model, "config", None), "_name_or_path", None
        )
        api_server_v2.initialize_with_model(
            moe_model=self,
            model_name=model_name,
            tok=getattr(self, "tokenizer", None),
            max_seq_length=self.max_seq_length,
            device_memory_ratio=device_memory_ratio,
            kv_cache_ratio=kv_cache_ratio,
            max_batch_size=max_batch_size,
            enable_prefix_caching=enable_prefix_caching,
            speculative_draft=serving_speculator,
            enable_deepseek_mla_paging=enable_deepseek_mla_paging,
            max_resident_paged_speculative_sessions=(
                max_resident_paged_speculative_sessions
            ),
            min_free_mla_blocks_after_admission=(
                min_free_mla_blocks_after_admission
            ),
        )

        uvicorn = importlib.import_module("uvicorn")
        uvicorn.run(api_server_v2.app, host=host, port=port)

    def forward(self, input_ids: torch.LongTensor, *args, **kwargs) -> Any:
        """
        Forwards the input through the model.

        Args:
            *args: Additional positional arguments for the model's forward method.
            **kwargs: Additional keyword arguments for the model's forward method.

        Returns:
            Any: The output of the model.
        """

        self._configure_hook(input_ids)

        return self.model(input_ids, *args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Forwards the input through the model.

        Args:
            *args: Additional positional arguments for the model's forward method.
            **kwargs: Additional keyword arguments for the model's forward method.

        Returns:
            Any: The output of the model.
        """
        return self.forward(*args, **kwargs)
