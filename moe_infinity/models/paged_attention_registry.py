from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from moe_infinity.runtime.paged_kv_storage import PagedKVStorage

_QWEN3_MODULE = "moe_infinity.models.qwen3_paged_attention"
_QWEN3_CLASS = "Qwen3PagedAttention"


@dataclass(frozen=True)
class PagedClassSpec:
    module_path: str
    class_name: str


SUPPORTED_PAGED_CLASS_SPECS: dict[tuple[str, str], PagedClassSpec] = {
    (_QWEN3_MODULE, _QWEN3_CLASS): PagedClassSpec(_QWEN3_MODULE, _QWEN3_CLASS),
}


def _resolve_supported_qwen3_type() -> type:
    module = importlib.import_module(_QWEN3_MODULE)
    return getattr(module, _QWEN3_CLASS)


def _resolve_unsupported_mla_types() -> tuple[type, ...]:
    candidates = (
        (
            "moe_infinity.models.deepseek_v2_paged_attention",
            "DeepseekV2PagedAttention",
        ),
        (
            "moe_infinity.models.deepseek_v3_paged_attention",
            "DeepseekV3PagedAttention",
        ),
        (
            "moe_infinity.models.modeling_deepseek_v2",
            "DeepseekV2PagedAttention",
        ),
        (
            "moe_infinity.models.modeling_deepseek_v3",
            "DeepseekV3PagedAttention",
        ),
    )
    resolved: list[type] = []
    for module_path, class_name in candidates:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        cls = getattr(module, class_name, None)
        if isinstance(cls, type):
            resolved.append(cls)
    return tuple(resolved)


UNSUPPORTED_MLA_CLASS_TYPES: tuple[type, ...] = _resolve_unsupported_mla_types()


class LayerBoundPagedBackend:
    """Immutable per-layer proxy binding an attention backend to one
    ``layer_idx`` and one authoritative storage owner."""

    def __init__(
        self, backend: Any, layer_idx: int, storage_owner_id: str
    ) -> None:
        self._backend = backend
        self._layer_idx = int(layer_idx)
        self._storage_owner_id = storage_owner_id

    @property
    def backend(self) -> Any:
        return self._backend

    @property
    def layer_idx(self) -> int:
        return self._layer_idx

    @property
    def storage_owner_id(self) -> str:
        return self._storage_owner_id

    @property
    def storage(self) -> Any:
        return getattr(self._backend, "storage", None)

    def forward(
        self, *args: Any, layer_idx: Optional[int] = None, **kwargs: Any
    ) -> Any:
        if layer_idx is not None and int(layer_idx) != self._layer_idx:
            raise ValueError(
                f"layer_idx {layer_idx} does not match bound layer "
                f"{self._layer_idx}"
            )
        return self._backend.forward(*args, layer_idx=self._layer_idx, **kwargs)


@dataclass
class PagedLayerBinding:
    module: Any
    layer_idx: int
    base_class: type
    bound_class: type
    backend: LayerBoundPagedBackend
    class_fqn: str
    storage_owner_id: str
    has_write_proof: bool = False


@dataclass
class InspectResult:
    binding: Optional[PagedLayerBinding]
    reason: str


@dataclass
class PagedAttentionLayerRegistry:
    bindings: list[PagedLayerBinding] = field(default_factory=list)
    reason: str = "eligible"

    @classmethod
    def empty(cls, reason: str) -> "PagedAttentionLayerRegistry":
        return cls(bindings=[], reason=reason)

    @classmethod
    def inspect_module(cls, module: Any) -> InspectResult:
        module_type = type(module)
        if module_type in UNSUPPORTED_MLA_CLASS_TYPES:
            return InspectResult(binding=None, reason="mla_layout_unsupported")
        supported = _resolve_supported_qwen3_type()
        if module_type is supported:
            return InspectResult(binding=None, reason="eligible")
        return InspectResult(binding=None, reason="paged_class_unregistered")

    @classmethod
    def register(
        cls,
        model: Any,
        backend: Any,
        storage: "PagedKVStorage",
    ) -> "PagedAttentionLayerRegistry":
        supported = _resolve_supported_qwen3_type()
        class_fqn = f"{_QWEN3_MODULE}.{_QWEN3_CLASS}"
        bindings: list[PagedLayerBinding] = []
        seen_layers: set[int] = set()

        for module in model.modules():
            module_type = type(module)
            if module_type in UNSUPPORTED_MLA_CLASS_TYPES:
                raise ValueError(
                    "DeepSeek MLA paged attention is unsupported "
                    "(mla_layout_unsupported)"
                )
            if module_type is not supported:
                continue

            layer_idx = getattr(module, "layer_idx", None)
            if not isinstance(layer_idx, int) or isinstance(layer_idx, bool):
                raise ValueError("Qwen3 paged module missing integer layer_idx")
            if not 0 <= layer_idx < storage.spec.num_layers:
                raise ValueError(
                    f"layer_idx {layer_idx} out of range "
                    f"[0, {storage.spec.num_layers})"
                )
            if layer_idx in seen_layers:
                raise ValueError(f"duplicate layer_idx {layer_idx}")
            seen_layers.add(layer_idx)

            bound_class = type(
                f"{supported.__name__}Layer{layer_idx}",
                (supported,),
                {},
            )
            module.__class__ = bound_class
            proxy = LayerBoundPagedBackend(backend, layer_idx, storage.owner_id)
            bindings.append(
                PagedLayerBinding(
                    module=module,
                    layer_idx=layer_idx,
                    base_class=supported,
                    bound_class=bound_class,
                    backend=proxy,
                    class_fqn=class_fqn,
                    storage_owner_id=storage.owner_id,
                    has_write_proof=True,
                )
            )

        bindings.sort(key=lambda binding: binding.layer_idx)
        return cls(bindings=bindings, reason="eligible")

    register_qwen3 = register

    def install_metadata(self, metadata: Any) -> None:
        for binding in self.bindings:
            binding.bound_class.set_paged_context(binding.backend, metadata)

    def clear_metadata(self) -> None:
        for binding in self.bindings:
            binding.bound_class.clear_paged_context()


__all__ = [
    "InspectResult",
    "LayerBoundPagedBackend",
    "PagedAttentionLayerRegistry",
    "PagedClassSpec",
    "PagedLayerBinding",
    "SUPPORTED_PAGED_CLASS_SPECS",
    "UNSUPPORTED_MLA_CLASS_TYPES",
]
