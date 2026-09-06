from __future__ import annotations

import inspect
from dataclasses import dataclass
from numbers import Integral
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Union,
    cast,
)

import torch

from moe_infinity.spec_decode._dflash_ops import (
    acceptance_length,
    build_block,
    committed_tokens,
)
from moe_infinity.spec_decode._dflash_sample_ops import (
    _validate_generator_device,
    acceptance_sampled,
    committed_tokens_sampled,
    warped_probs,
)
from moe_infinity.spec_decode._nvtx import nvtx_phase
from moe_infinity.spec_decode._prefetch_route import union_experts_from_mask
from moe_infinity.spec_decode._route_ahead_ctx import route_ahead_context
from moe_infinity.spec_decode._route_ahead_stats import RouteAheadStats
from moe_infinity.spec_decode.protocols import (
    CacheAdapter,
    ExecutorEvidence,
    NativeStepTrace,
    PairingEvidence,
    RequestSpec,
    SamplingContext,
    SessionTrace,
)

if TYPE_CHECKING:
    from moe_infinity.engine.generation_loop import GenerationEngine
    from moe_infinity.engine.types import SamplingParams


@dataclass
class DFlashConfig:
    block_size: int
    mask_token_id: int
    target_layer_ids: List[int]
    num_target_layers: int
    hidden_size: int
    vocab_size: int


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_context_feature(
    hidden_states: Sequence[torch.Tensor], layer_ids: Sequence[int]
) -> torch.Tensor:
    # Lazy import avoids the package cycle big_modeling -> spec_decode -> dflash
    # while preserving big_modeling.extract_context_feature as the one contract.
    from moe_infinity.entrypoints.big_modeling import extract_context_feature

    return extract_context_feature(hidden_states, layer_ids)


def read_dflash_config(draft_hf_config: Any) -> DFlashConfig:
    dflash = _get(draft_hf_config, "dflash_config", {}) or {}
    text_config = _get(draft_hf_config, "text_config", draft_hf_config)

    block_size = _get(draft_hf_config, "block_size", _get(dflash, "block_size"))
    mask_token_id = _get(
        dflash, "mask_token_id", _get(draft_hf_config, "mask_token_id")
    )
    target_layer_ids = _get(
        dflash, "target_layer_ids", _get(draft_hf_config, "target_layer_ids")
    )
    num_target_layers = _get(
        draft_hf_config, "num_target_layers", _get(dflash, "num_target_layers")
    )

    if block_size is None or mask_token_id is None or target_layer_ids is None:
        raise ValueError(
            "draft config is missing required DFlash fields "
            "(block_size, dflash_config.mask_token_id, dflash_config.target_layer_ids)"
        )

    return DFlashConfig(
        block_size=int(block_size),
        mask_token_id=int(mask_token_id),
        target_layer_ids=[int(x) for x in target_layer_ids],
        num_target_layers=int(num_target_layers)
        if num_target_layers is not None
        else -1,
        hidden_size=int(_get(text_config, "hidden_size")),
        vocab_size=int(_get(text_config, "vocab_size")),
    )


DFLASH_BLOCK_SIZE = 10
DFLASH_TARGET_LAYER_IDS = [1, 9, 17, 25, 33]


def validate_pairing(
    draft_cfg: DFlashConfig, target_hf_config: Any
) -> PairingEvidence:
    target_text = _get(target_hf_config, "text_config", target_hf_config)
    t_hidden = int(_get(target_text, "hidden_size"))
    t_vocab = int(_get(target_text, "vocab_size"))
    t_layers = int(_get(target_text, "num_hidden_layers"))

    if draft_cfg.hidden_size != t_hidden:
        raise ValueError(
            f"DFlash drafter hidden_size {draft_cfg.hidden_size} != target hidden_size {t_hidden}"
        )
    if draft_cfg.vocab_size != t_vocab:
        raise ValueError(
            f"DFlash drafter vocab_size {draft_cfg.vocab_size} != target vocab_size {t_vocab}"
        )
    if draft_cfg.mask_token_id >= t_vocab:
        raise ValueError(
            f"DFlash mask_token_id {draft_cfg.mask_token_id} is outside target vocab_size {t_vocab}"
        )
    if draft_cfg.block_size < 2:
        raise ValueError(
            f"DFlash block_size {draft_cfg.block_size} must be >= 2 (anchor + >=1 draft token)"
        )
    if not draft_cfg.target_layer_ids or any(
        i < 0 for i in draft_cfg.target_layer_ids
    ):
        raise ValueError(
            f"DFlash target_layer_ids {draft_cfg.target_layer_ids} must be non-empty and non-negative"
        )
    highest_capture = max(draft_cfg.target_layer_ids) + 1
    if highest_capture > t_layers:
        raise ValueError(
            f"DFlash target_layer_ids reference target layer {max(draft_cfg.target_layer_ids)} "
            f"(capture index {highest_capture}) but target has only {t_layers} layers"
        )
    return PairingEvidence(
        valid=True,
        config_valid=True,
        dimensions_valid=True,
        vocab_valid=True,
        mask_valid=True,
        layers_valid=True,
        block_valid=True,
        module_valid=None,
    )


def validate_drafter_module(draft_model: Any, draft_cfg: DFlashConfig) -> None:
    fc = _get(draft_model, "fc")
    if fc is None:
        raise ValueError(
            "DFlash drafter is missing the required `fc` projection layer "
            "(expected a Linear consuming the concatenated 5-layer hidden feature)"
        )
    in_features = _get(fc, "in_features")
    expected = len(draft_cfg.target_layer_ids) * draft_cfg.hidden_size
    if in_features != expected:
        raise ValueError(
            f"DFlash drafter fc.in_features {in_features} != expected {expected} "
            f"({len(draft_cfg.target_layer_ids)} * hidden_size {draft_cfg.hidden_size})"
        )


def validate_drafter(
    draft_model: Any,
    target_hf_config: Any,
    draft_cfg: Optional[DFlashConfig] = None,
) -> DFlashConfig:
    if draft_cfg is None:
        draft_cfg = read_dflash_config(_get(draft_model, "config"))
    validate_pairing(draft_cfg, target_hf_config)
    validate_drafter_module(draft_model, draft_cfg)
    return draft_cfg


_PUBLISHED_DFLASH_PAIRS = frozenset(
    {
        ("openai/gpt-oss-20b", "z-lab/gpt-oss-20b-DFlash"),
        ("openai/gpt-oss-120b", "z-lab/gpt-oss-120b-DFlash"),
    }
)


def _validated_checkpoint_scope(
    target_hf_config: Any,
    draft_hf_config: Any,
    draft_model_path: str | None = None,
) -> tuple[str, ...]:
    """Return checkpoint identities only for explicitly published pairs."""
    target_name = _get(target_hf_config, "_name_or_path")
    draft_name = draft_model_path or _get(draft_hf_config, "_name_or_path")
    pair = (str(target_name), str(draft_name))
    return pair if pair in _PUBLISHED_DFLASH_PAIRS else ()


def build_pairing_evidence(
    draft_model: Any,
    target_hf_config: Any,
    draft_cfg: DFlashConfig,
    *,
    validated_checkpoint_scope: tuple[str, ...] = (),
) -> PairingEvidence:
    """Run the authoritative structural checks and return their evidence."""
    base = validate_pairing(draft_cfg, target_hf_config)
    validate_drafter_module(draft_model, draft_cfg)
    return PairingEvidence(
        valid=base.valid,
        config_valid=base.config_valid,
        dimensions_valid=base.dimensions_valid,
        vocab_valid=base.vocab_valid,
        mask_valid=base.mask_valid,
        layers_valid=base.layers_valid,
        block_valid=base.block_valid,
        module_valid=True,
        validated_checkpoint_scope=validated_checkpoint_scope,
    )


def executor_wiring_reachable(moe: Any) -> bool:
    """Whether a target can reach ``DistributedExpertExecutor`` dispatch."""
    roots = (moe, getattr(moe, "model", None))
    for root in roots:
        if root is None:
            continue
        if getattr(root, "expert_executor", None) is not None:
            return True
        if getattr(root, "_executor", None) is not None:
            return True
        modules = getattr(root, "modules", None)
        if not callable(modules):
            continue
        for module in modules():
            if getattr(module, "expert_executor", None) is not None:
                return True
    return False


def _resolve_input_embeddings(target: Any) -> Any:
    getter = getattr(target, "get_input_embeddings", None)
    if callable(getter):
        emb = getter()
        if emb is not None:
            return emb
    inner = getattr(target, "model", target)
    emb = getattr(inner, "embed_tokens", None)
    if emb is not None:
        return emb
    raise ValueError(
        "could not resolve target embed_tokens for DFlash drafter weight sharing"
    )


def _resolve_output_embeddings(target: Any) -> Any:
    getter = getattr(target, "get_output_embeddings", None)
    if callable(getter):
        head = getter()
        if head is not None:
            return head
    head = getattr(target, "lm_head", None)
    if head is not None:
        return head
    raise ValueError(
        "could not resolve target lm_head for DFlash drafter weight sharing"
    )


def bind_shared_weights(draft_model: Any, target: Any) -> tuple[Any, Any]:
    embed_tokens = _resolve_input_embeddings(target)
    lm_head = _resolve_output_embeddings(target)
    draft_model.embed_tokens = embed_tokens
    draft_model.lm_head = lm_head
    return embed_tokens, lm_head


def _infer_cuda_device(model: Any) -> str:
    dev = getattr(model, "device", None)
    if isinstance(dev, torch.device) and dev.type == "cuda":
        return str(dev)
    # Match the bound shared embed_tokens/lm_head device: an offloaded backbone
    # is resident on the LAST visible GPU, so first-cuda-param/cuda:0 would put
    # the drafter's block on a different GPU than its shared weights.
    try:
        embed = _resolve_input_embeddings(model)
        embed_device = getattr(getattr(embed, "weight", None), "device", None)
        if (
            isinstance(embed_device, torch.device)
            and embed_device.type == "cuda"
        ):
            return str(embed_device)
    except Exception:
        pass
    try:
        for param in model.parameters():
            if param.device.type == "cuda":
                return str(param.device)
    except Exception:
        pass
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.device_count() - 1}"
    return "cpu"


def _resolve_stop_ids(target: Any, stop_token_ids: Any) -> List[int]:
    value = (
        _get(_get(target, "config", target), "eos_token_id")
        if stop_token_ids is None
        else stop_token_ids
    )
    if value is None:
        return []
    if isinstance(value, Integral) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if all(
            isinstance(token, Integral) and not isinstance(token, bool)
            for token in values
        ):
            return [int(token) for token in values]
    raise ValueError(
        "stop_token_ids must be an integer token ID or a sequence of integer token IDs"
    )


def _normalize_per_row(
    name: str, value: Any, batch: int, convert: Callable[[Any], Any]
) -> tuple[Any, ...]:
    """Expand one scalar or validate one explicit value per batch row."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = tuple(value)
        if len(rows) != batch:
            raise ValueError(
                f"per-row {name} has {len(rows)} entries for batch size {batch}"
            )
        return tuple(convert(row) for row in rows)
    return tuple(convert(value) for _ in range(batch))


def _clone_generator(generator: torch.Generator) -> torch.Generator:
    clone = torch.Generator(device=generator.device)
    clone.set_state(generator.get_state())
    return clone


def _normalize_sampling_contexts(
    *,
    batch: int,
    probability_device: Union[str, torch.device],
    temperature: Union[float, Sequence[float]],
    top_k: Union[int, Sequence[int]],
    top_p: Union[float, Sequence[float]],
    generator: Union[
        torch.Generator,
        Sequence[Optional[torch.Generator]],
        None,
    ],
) -> tuple[SamplingContext, ...]:
    """Normalize scalar-or-per-row sampling inputs before model prefill.

    Numeric scalars broadcast. A scalar generator is an immutable template
    for independent row-local clones; an explicit sequence is retained so
    each request-owned stream visibly advances. In a physical batch, omitted
    generators are replaced by independently seeded generators on the model
    device so sampled rows never share the ambient RNG stream.
    """
    temperatures = _normalize_per_row("temperature", temperature, batch, float)
    top_ks = _normalize_per_row(
        "top_k", top_k, batch, lambda value: max(0, int(value))
    )
    top_ps = _normalize_per_row("top_p", top_p, batch, float)

    if isinstance(generator, Sequence):
        generators = tuple(generator)
        if len(generators) != batch:
            raise ValueError(
                f"per-row generator has {len(generators)} entries for batch size {batch}"
            )
        if any(
            item is not None and not isinstance(item, torch.Generator)
            for item in generators
        ):
            raise TypeError(
                "per-row generator values must be torch.Generator or None"
            )
    elif generator is None:
        generators = (None,) * batch
    elif isinstance(generator, torch.Generator):
        generators = (
            (generator,)
            if batch == 1
            else tuple(_clone_generator(generator) for _ in range(batch))
        )
    else:
        raise TypeError(
            "generator must be torch.Generator, a per-row sequence, or None"
        )

    if batch > 1:
        missing_sampled = [
            row
            for row in range(batch)
            if temperatures[row] > 0 and generators[row] is None
        ]
        if missing_sampled:
            seeds = torch.randint(
                0,
                torch.iinfo(torch.int64).max,
                (len(missing_sampled),),
                dtype=torch.int64,
            ).tolist()
            mutable_generators = list(generators)
            for row, seed in zip(missing_sampled, seeds):
                mutable_generators[row] = torch.Generator(
                    device=torch.device(probability_device)
                ).manual_seed(seed)
            generators = tuple(mutable_generators)

    sampled_generator_ids = [
        id(generators[row])
        for row in range(batch)
        if temperatures[row] > 0 and generators[row] is not None
    ]
    if len(set(sampled_generator_ids)) != len(sampled_generator_ids):
        raise ValueError(
            "the same explicit generator object cannot be shared by sampled "
            "rows; pass distinct per-row generators or one scalar generator "
            "to request cloned streams"
        )

    return tuple(
        SamplingContext(
            temperature=temperatures[row],
            top_k=top_ks[row],
            top_p=top_ps[row],
            generator=generators[row],
        )
        for row in range(batch)
    )


def _normalize_stop_rows(
    target: Any,
    stop_token_ids: Any,
    batch: int,
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    """Disambiguate a shared flat stop set from nested per-row stop sets."""
    if stop_token_ids is None or (
        isinstance(stop_token_ids, Integral)
        and not isinstance(stop_token_ids, bool)
    ):
        shared = tuple(_resolve_stop_ids(target, stop_token_ids))
        return (shared,) * batch, False
    if not isinstance(stop_token_ids, Sequence) or isinstance(
        stop_token_ids, (str, bytes)
    ):
        raise ValueError(
            "stop_token_ids must be an integer, a shared sequence of integers, "
            "or one sequence per row"
        )

    values = tuple(stop_token_ids)
    scalar_rows = tuple(
        isinstance(value, Integral) and not isinstance(value, bool)
        for value in values
    )
    nested_rows = tuple(
        value is None
        or (isinstance(value, Sequence) and not isinstance(value, (str, bytes)))
        for value in values
    )
    if all(scalar_rows):
        shared = tuple(int(token) for token in values)
        return (shared,) * batch, False
    if not all(nested_rows):
        raise ValueError(
            "stop_token_ids cannot mix shared scalar token IDs with per-row sequences"
        )
    if len(values) != batch:
        raise ValueError(
            f"per-row stop_token_ids has {len(values)} entries for batch size {batch}"
        )
    default = tuple(_resolve_stop_ids(target, None))
    rows: list[tuple[int, ...]] = []
    for row in values:
        if row is None:
            rows.append(default)
            continue
        try:
            rows.append(tuple(_resolve_stop_ids(target, row)))
        except ValueError as exc:
            raise ValueError(
                "each per-row stop_token_ids value must be None or a sequence "
                "of integer token IDs"
            ) from exc
    return tuple(rows), True


@dataclass(frozen=True)
class SlidingWindowCacheSnapshot:
    keys: torch.Tensor
    values: torch.Tensor
    cumulative_length: int


@dataclass(frozen=True)
class LinearAttentionCacheSnapshot:
    conv_states: Optional[torch.Tensor]
    recurrent_states: Optional[torch.Tensor]
    is_conv_states_initialized: bool
    is_recurrent_states_initialized: bool
    has_previous_state: bool


@dataclass(frozen=True)
class TargetCacheSnapshot:
    sliding: dict[int, SlidingWindowCacheSnapshot]
    linear: dict[int, LinearAttentionCacheSnapshot]


_LINEAR_CACHE_FIELDS = (
    "conv_states",
    "recurrent_states",
    "is_conv_states_initialized",
    "is_recurrent_states_initialized",
    "has_previous_state",
)


def snapshot_target_cache(target_kv: Any) -> TargetCacheSnapshot:
    """Clone rollback-sensitive state before a speculative verify forward."""
    try:
        from transformers.cache_utils import (
            DynamicSlidingWindowLayer,
            LinearAttentionCacheLayerMixin,
        )
    except ImportError as exc:
        raise RuntimeError(
            "unsupported transformers cache API: DFlash hybrid rollback "
            "requires DynamicSlidingWindowLayer and "
            "LinearAttentionCacheLayerMixin"
        ) from exc

    sliding: dict[int, SlidingWindowCacheSnapshot] = {}
    linear: dict[int, LinearAttentionCacheSnapshot] = {}
    for index, layer in enumerate(target_kv.layers):
        present_linear_fields = [
            field for field in _LINEAR_CACHE_FIELDS if hasattr(layer, field)
        ]
        is_linear = isinstance(layer, LinearAttentionCacheLayerMixin)
        if is_linear or present_linear_fields:
            if len(present_linear_fields) != len(_LINEAR_CACHE_FIELDS):
                missing = sorted(
                    field
                    for field in _LINEAR_CACHE_FIELDS
                    if field not in present_linear_fields
                )
                raise RuntimeError(
                    "unsupported transformers linear-attention cache contract "
                    f"for layer {index}: missing fields {missing}"
                )
            conv_states = layer.conv_states
            recurrent_states = layer.recurrent_states
            linear[index] = LinearAttentionCacheSnapshot(
                conv_states=(
                    conv_states.clone() if conv_states is not None else None
                ),
                recurrent_states=(
                    recurrent_states.clone()
                    if recurrent_states is not None
                    else None
                ),
                is_conv_states_initialized=bool(
                    layer.is_conv_states_initialized
                ),
                is_recurrent_states_initialized=bool(
                    layer.is_recurrent_states_initialized
                ),
                has_previous_state=bool(layer.has_previous_state),
            )

        if isinstance(layer, DynamicSlidingWindowLayer):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            cumulative_length = getattr(layer, "cumulative_length", None)
            if keys is None or values is None or cumulative_length is None:
                raise RuntimeError(
                    "unsupported transformers sliding-window cache contract "
                    f"for layer {index}: expected initialized keys, values, "
                    "and cumulative_length"
                )
            sliding[index] = SlidingWindowCacheSnapshot(
                keys=keys.clone(),
                values=values.clone(),
                cumulative_length=int(cumulative_length),
            )

    return TargetCacheSnapshot(sliding=sliding, linear=linear)


def rollback_target_cache(
    target_kv: Any,
    snapshot: TargetCacheSnapshot,
    prev_start: int,
    committed: int,
    block_size: int,
    *,
    block: Optional[torch.Tensor] = None,
    replay: Optional[Callable[[torch.Tensor, Any], Any]] = None,
) -> None:
    """Restore a partial verify and replay its committed prefix exactly."""
    if committed < 0 or committed > block_size:
        raise ValueError(
            f"committed must be in [0, {block_size}], got {committed}"
        )
    if committed == block_size:
        return

    # Legacy full/sliding-only callers (the batched path) remain slice based.
    # Hybrid linear attention cannot use this branch because its recurrent
    # state is not croppable; batch-1 callers provide replay below.
    if replay is None:
        if snapshot.linear:
            raise RuntimeError(
                "hybrid linear-attention cache rollback requires committed-"
                "prefix replay; no replay callback was provided"
            )
        new_len = prev_start + committed
        for index, layer in enumerate(target_kv.layers):
            sliding = snapshot.sliding.get(index)
            if sliding is None:
                layer.crop(new_len)
                continue
            block_k = layer.keys[:, :, -block_size:, :]
            block_v = layer.values[:, :, -block_size:, :]
            full_k = torch.cat(
                [sliding.keys, block_k[:, :, :committed, :]], dim=-2
            )
            full_v = torch.cat(
                [sliding.values, block_v[:, :, :committed, :]], dim=-2
            )
            layer.keys = full_k[:, :, -layer.sliding_window + 1 :, :]
            layer.values = full_v[:, :, -layer.sliding_window + 1 :, :]
            layer.cumulative_length = new_len
        return

    if block is None or block.ndim != 2 or int(block.shape[1]) != block_size:
        raise ValueError(
            "partial cache replay requires block with shape "
            f"[batch, {block_size}]"
        )

    for index, layer in enumerate(target_kv.layers):
        linear = snapshot.linear.get(index)
        if linear is not None:
            for field in ("conv_states", "recurrent_states"):
                saved = getattr(linear, field)
                if saved is None:
                    setattr(layer, field, None)
                else:
                    # Reassign a fresh clone rather than in-place copy_: the
                    # GatedDeltaNet state is an inference tensor (the target
                    # forward runs under inference/no_grad), and an in-place
                    # copy_ on it raises a version-counter bump error, silently
                    # aborting the restore and breaking greedy losslessness.
                    setattr(layer, field, saved.clone())
            layer.is_conv_states_initialized = linear.is_conv_states_initialized
            layer.is_recurrent_states_initialized = (
                linear.is_recurrent_states_initialized
            )
            layer.has_previous_state = linear.has_previous_state

        sliding = snapshot.sliding.get(index)
        if sliding is not None:
            if sliding.cumulative_length != prev_start:
                raise RuntimeError(
                    f"sliding cache layer {index} snapshot length "
                    f"{sliding.cumulative_length} != expected {prev_start}"
                )
            layer.keys = sliding.keys.clone()
            layer.values = sliding.values.clone()
            layer.cumulative_length = sliding.cumulative_length
        else:
            crop = getattr(layer, "crop", None)
            if callable(crop):
                crop(prev_start)
            elif linear is None:
                raise RuntimeError(
                    "unsupported transformers cache layer for DFlash rollback: "
                    f"layer {index} is {type(layer).__name__}"
                )

    replayed_cache = replay(block[:, :committed], target_kv)
    if replayed_cache is not None and replayed_cache is not target_kv:
        raise RuntimeError(
            "target replay replaced the DynamicCache object; in-place hybrid "
            "rollback requires the original cache instance"
        )


# ---------------------------------------------------------------------------
# Single-round control seam (PD-DFlash Task 6 Step 5)
#
# ``SpecSession`` + ``begin_session``/``draft_round``/``verify_round`` own the
# canonical single-sequence state machine. ``_generate_single`` is only its
# synchronous driver; serving can interpose the 2-D verify scheduler between
# DRAFT and VERIFY while registering each pending verify's EXACT token/byte
# demand without maintaining a second decode algorithm.
# ---------------------------------------------------------------------------


class RouteUnionCollector:
    """Read-only per-verify routed-union recorder for the Step-5 projection.

    Duck-types the ``RouteAheadStats`` recorder slot of ``route_ahead_context``
    so it plugs into the EXISTING route-ahead seam with zero executor changes:
    ``DistributedExpertExecutor._maybe_route_ahead_prefetch`` calls
    ``observe_layer(layer_id, predicted_ids, router_mask, ...)`` for every
    executor-backed MoE dispatch of the verify forward. This recorder ignores
    the pinned ``predicted_ids`` and unions the layer's ACTUAL routed experts
    straight from ``router_mask`` via the SAME ``union_experts_from_mask``
    primitive the verify's dispatch uses -- so the projection IS the verify's
    own gate/top-k routing, observed, never re-run. It NEVER pins, prefetches,
    or changes routing/outputs. Bare-HF / gpt-oss / resident targets never
    reach the seam, so the union stays empty (byte-free admission).
    """

    def __init__(self) -> None:
        self._pending: set[tuple[int, int]] = set()
        self.union: set[tuple[int, int]] = set()

    def begin_step(self) -> None:
        """Start a verify step; drop any uncommitted prior-layer records."""
        self._pending = set()

    def observe_layer(
        self,
        layer_id: int,
        predicted_ids: Sequence[int],
        router_mask: Any,
        expert_nbytes: Optional[Any] = None,
    ) -> None:
        """Union this dispatched layer's ACTUAL routed experts (read-only)."""
        del predicted_ids, expert_nbytes  # read the routed union, not the pin
        for expert_id in union_experts_from_mask(router_mask):
            self._pending.add((int(layer_id), int(expert_id)))

    def commit_step(self, kept_rows: int = 0) -> RouteAheadStats | None:
        """Finalize the step's union as the projection for the next round.

        ``kept_rows`` is accepted for recorder-interface parity but ignored:
        the admission projection intentionally covers the FULL block's routing
        (the pending verify reads every block row before the accept rule fixes
        the kept prefix). Returns ``None``.
        """
        del kept_rows
        self.union = set(self._pending)
        self._pending = set()
        return None


def project_expert_bytes(
    expert_union: Any, expert_nbytes_map: Optional[Any]
) -> int:
    """Exact Sum of registration-time payload bytes over a routed union.

    ``expert_nbytes_map`` is ``ExpertPrefetcher.expert_nbytes_map`` -- the real
    FP4 payload sizes keyed by ``(layer_id, expert_id)``. ``(layer, expert)``
    pairs absent from the map contribute nothing; the demand is NEVER
    fabricated from an expert count or an average size. A missing / non-dict /
    empty map yields ``0`` (resident / bare-HF / mock -> byte-free admission).
    """
    if (
        not expert_union
        or not isinstance(expert_nbytes_map, dict)
        or not expert_nbytes_map
    ):
        return 0
    total = 0
    for key in expert_union:
        nbytes = expert_nbytes_map.get(key)
        if nbytes is not None:
            total += int(nbytes)
    return total


@dataclass(frozen=True)
class DraftResult:
    """One DRAFT round's projected verify demand (engine Step-5 admission).

    ``tokens`` is the verify block width (``block_size``). ``expert_union`` is
    the projected ``(layer_id, expert_id)`` set the pending verify is expected
    to route to -- captured read-only from the SAME gate/top-k path the verify
    uses (see ``RouteUnionCollector``), NOT a second router. ``expert_bytes`` is
    the EXACT summed registration-time payload over that union (never an
    expert-count estimate). An empty union / no offload prefetcher yields
    ``expert_bytes == 0`` (byte-free admission, e.g. the first warm-up round or
    resident / bare-HF targets).
    """

    tokens: int
    expert_union: frozenset[tuple[int, int]]
    expert_bytes: int


@dataclass(frozen=True)
class VerifyResult:
    """One VERIFY round's committed outcome.

    ``accepted_token_ids`` are the tokens emitted this round (accepted drafts
    then the bonus, stop/budget-truncated); ``accept`` is the accepted drafts
    committed to KV (cache advance minus one), while ``verified_accept`` is the
    untruncated acceptance-rule result. ``committed_count`` is the number of
    emitted tokens appended to the sequence; ``finished`` is set once a stop id
    or ``max_new_tokens`` ends the session.
    """

    accepted_token_ids: list[int]
    accept: int
    verified_accept: int
    committed_count: int
    finished: bool


@dataclass
class SpecSession:
    """Externalized per-sequence round state for the DFlash draft/verify loop.

    ``begin_session`` seeds this from the prefill/anchor forward; ``draft_round``
    and ``verify_round`` advance it one DRAFT and one VERIFY round respectively,
    reproducing singleton ``generate()`` under explicit engine control (Task 6
    Step 5). The synchronous singleton path drives this same state machine.
    """

    input_ids: torch.Tensor
    max_new_tokens: int
    sampling: SamplingContext
    block_size: int
    layer_ids: list[int]
    mask_token_id: int
    stop_ids: set[int]
    num_prompt_tokens: int
    target_kv: Any
    draft_kv: Any
    context_feature: torch.Tensor
    anchor: int
    start: int
    emitted: list[int]
    step_trace: list[NativeStepTrace]
    finished: bool = False
    round_index: int = 0
    projected_expert_union: frozenset[tuple[int, int]] = frozenset()
    projected_expert_bytes: int = 0
    collector: Optional[RouteUnionCollector] = None
    _pending: bool = False
    _pending_block: Optional[torch.Tensor] = None
    _pending_prev_start: int = 0
    pending_draft_probs: Optional[torch.Tensor] = None
    _pending_cache_snapshot: Any = None

    @property
    def temperature(self) -> float:
        return self.sampling.temperature

    @property
    def top_k(self) -> int:
        return self.sampling.top_k

    @property
    def top_p(self) -> float:
        return self.sampling.top_p

    @property
    def sampled(self) -> bool:
        return self.sampling.is_sampled

    @property
    def output_ids(self) -> list[int]:
        """All tokens emitted so far (anchor ++ per-round commits), capped."""
        return list(self.emitted[: self.max_new_tokens])

    @property
    def has_pending_draft(self) -> bool:
        """Whether this session owns a tentative block awaiting verification."""
        return self._pending

    def clear_pending(self) -> None:
        """Discard tentative draft metadata after verification or abort."""
        self._pending = False
        self._pending_block = None
        self.pending_draft_probs = None
        self._pending_cache_snapshot = None


class DFlashSpeculator:
    def __init__(
        self,
        moe: Any,
        draft_model_path: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from transformers import AutoModel

        self.moe = moe
        self.target = getattr(moe, "model", moe)
        self.device = device or _infer_cuda_device(self.target)

        self.draft = (
            AutoModel.from_pretrained(
                draft_model_path, trust_remote_code=True, dtype=dtype
            )
            .to(self.device)
            .eval()
        )

        self.config = read_dflash_config(self.draft.config)
        checkpoint_scope = _validated_checkpoint_scope(
            self.target.config, self.draft.config, draft_model_path
        )
        self.pairing_evidence = build_pairing_evidence(
            self.draft,
            self.target.config,
            self.config,
            validated_checkpoint_scope=checkpoint_scope,
        )
        self.embed_tokens, self.lm_head = bind_shared_weights(
            self.draft, self.target
        )
        self.executor_evidence = self._base_executor_evidence()
        self._init_native_runtime()

    @classmethod
    def from_models(
        cls,
        moe: Any,
        draft_model: Any,
        config: Optional[DFlashConfig] = None,
        device: Optional[str] = None,
    ) -> "DFlashSpeculator":
        """Build a speculator from already-instantiated models (no checkpoint load).

        ``moe`` may be the MoE wrapper -- every target forward then routes
        through ``moe._native_model_forward_rich`` so experts stay on the
        standard ExpertExecutor dispatch -- or a bare HF causal LM, which is
        called directly with ``output_hidden_states=True``. The 120B pairing
        constants are not applied here; the module-level ``fc`` contract is.
        """
        self = cls.__new__(cls)
        self.moe = moe
        on_moe_engine = callable(
            getattr(moe, "_native_model_forward_rich", None)
        )
        self.target = moe.model if on_moe_engine else moe
        if device is None:
            if on_moe_engine:
                device = _infer_cuda_device(self.target)
            else:
                dev = getattr(self.target, "device", None)
                device = (
                    str(dev)
                    if isinstance(dev, torch.device)
                    else str(next(self.target.parameters()).device)
                )
        self.device = device
        self.draft = draft_model
        if config is None:
            config = read_dflash_config(_get(self.draft, "config"))
        self.config = config
        checkpoint_scope = _validated_checkpoint_scope(
            self.target.config, _get(self.draft, "config")
        )
        self.pairing_evidence = build_pairing_evidence(
            self.draft,
            self.target.config,
            self.config,
            validated_checkpoint_scope=checkpoint_scope,
        )
        self.embed_tokens, self.lm_head = bind_shared_weights(
            self.draft, self.target
        )
        self.executor_evidence = self._base_executor_evidence()
        self._init_native_runtime()
        return self

    def _init_native_runtime(self) -> None:
        # The reference DFlashDraftModel.forward takes ``noise_embedding`` and
        # maintains a DynamicCache of projected context KV; stateless drafters
        # (tiny fixtures) take ``(block_ids, context_feature)`` instead.
        self._drafter_has_kv_cache = (
            "noise_embedding"
            in inspect.signature(self.draft.forward).parameters
        )
        # Sliding-window targets retain only the last ``sliding_window - 1``
        # tokens per cache update; the verify block must fit in that span or
        # the snapshot/rebuild rollback cannot recover the block's K/V.
        sliding_window = getattr(
            getattr(self.target, "config", None), "sliding_window", None
        )
        if sliding_window:
            assert int(self.config.block_size) <= int(sliding_window) - 1, (
                f"DFlash block_size {self.config.block_size} must be <= "
                f"target sliding_window - 1 ({int(sliding_window) - 1})"
            )
        self.step_trace: List[NativeStepTrace] = []
        self.last_target_cache: Any = None
        self.last_draft_cache: Any = None
        # Per-row new-token counts set by the batched path (its ragged rows
        # are right-padded in the returned rectangle). None on the batch==1
        # path, whose output is never padded.
        self.last_generated_lengths: Optional[List[int]] = None
        self.last_session_traces: tuple[SessionTrace, ...] = ()
        self.last_session_results: tuple[Any, ...] = ()
        self.rich_forward_batched = False
        self.rich_forward_batch_sizes: list[int] = []
        # Track A5 metrics handle; None (default) = instrumentation off, zero
        # overhead. Set via ``enable_route_ahead_stats``.
        self.route_ahead_stats: Optional[RouteAheadStats] = None

    def enable_route_ahead_stats(self) -> RouteAheadStats:
        """Opt in to Track A5 route-ahead coverage/waste instrumentation.

        Creates (or resets) the ``RouteAheadStats`` recorder consumed by the
        verify-time route-ahead context; read the counters back from
        ``self.route_ahead_stats`` after ``generate()``. Pure observer --
        enabling it never changes routing, prefetch, or emitted tokens.
        """
        if self.route_ahead_stats is None:
            self.route_ahead_stats = RouteAheadStats()
        else:
            self.route_ahead_stats.reset()
        return self.route_ahead_stats

    def _configure_target_hooks(self, input_ids: torch.Tensor) -> None:
        configure = getattr(self.moe, "_configure_hook", None)
        if callable(configure):
            configure(input_ids)
        eval_fn = getattr(self.target, "eval", None)
        if callable(eval_fn):
            eval_fn()

    @staticmethod
    def _extract_context_feature(
        hidden_states: Sequence[torch.Tensor], layer_ids: Sequence[int]
    ) -> torch.Tensor:
        return _extract_context_feature(hidden_states, layer_ids)

    @staticmethod
    def _snapshot_target_cache(target_kv: Any) -> TargetCacheSnapshot:
        return snapshot_target_cache(target_kv)

    @staticmethod
    def _rollback_target_cache(
        target_kv: Any,
        snapshot: TargetCacheSnapshot,
        *,
        prev_start: int,
        committed: int,
        block_size: int,
    ) -> None:
        rollback_target_cache(
            target_kv,
            snapshot,
            prev_start=prev_start,
            committed=committed,
            block_size=block_size,
        )

    def _forward_target(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        logits_to_keep: int = 0,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_metadata: Any = None,
    ) -> tuple[torch.Tensor, Any, Any]:
        """Rich target forward -> on-device (logits, hidden_states, past_key_values).

        ``logits_to_keep=1`` slices to the last position and is for the
        prefill/anchor step only; the verify step MUST pass ``0`` (full
        logits) or the acceptance rule breaks. ``attention_mask`` /
        ``position_ids`` are the Track-C batched-path plumbing (left-padded
        rows need explicit per-row RoPE positions); the batch==1 call sites
        never pass them, so the single-sequence forward is byte-identical.
        """
        rich = getattr(self.moe, "_native_model_forward_rich", None)
        if callable(rich):
            from moe_infinity.spec_decode.protocols import RichBatchMetadata

            metadata = attention_metadata
            if metadata is None and past_key_values is not None:
                metadata = SimpleNamespace(is_prefill=False)
            if int(input_ids.shape[0]) > 1:
                batch = int(input_ids.shape[0])
                query_lengths = (
                    tuple(
                        int(value)
                        for value in attention_mask.sum(dim=1).tolist()
                    )
                    if attention_mask is not None
                    and int(attention_mask.shape[1]) == int(input_ids.shape[1])
                    else (int(input_ids.shape[1]),) * batch
                )
                offsets = [0]
                for length in query_lengths:
                    offsets.append(offsets[-1] + length)
                metadata = RichBatchMetadata(
                    row_offsets=tuple(offsets),
                    row_lengths=query_lengths,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_handles=(past_key_values,) * batch,
                    is_prefill=past_key_values is None,
                )
                token_ids: list[int] | torch.Tensor = input_ids
                self.rich_forward_batch_sizes.append(batch)
            else:
                token_ids = [int(t) for t in input_ids[0].tolist()]
            result = rich(token_ids, metadata, logits_to_keep=logits_to_keep)
            from moe_infinity.spec_decode.protocols import RichForwardResult

            if isinstance(result, RichForwardResult):
                logits = result.logits
                hidden_states = result.hidden_states
                past_key_values = result.cache_handle
            elif isinstance(result, tuple) and len(result) == 3:
                logits, hidden_states, past_key_values = result
            else:
                raise RuntimeError(
                    "_native_model_forward_rich must return "
                    "(logits, hidden_states, past_key_values)"
                )
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError(
                    "native rich forward logits must be a Tensor"
                )
            return logits, hidden_states, past_key_values
        kwargs: dict[str, Any] = {
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_hidden_states": True,
        }
        if logits_to_keep:
            kwargs["logits_to_keep"] = int(logits_to_keep)
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        outputs = self.target(input_ids, **kwargs)
        return outputs.logits, outputs.hidden_states, outputs.past_key_values

    def _resolve_route_ahead_prefetcher(self) -> Any:
        """Prefetcher handle bound to the verify-time route-ahead context.

        Production MoE shell: ``moe.engine.expert_prefetcher``
        (``runtime/model_offload.py:865``). Tiny fixtures / bare-HF targets
        have no offload engine -> ``None``; the executor seam then falls back
        to its own configured prefetcher and no-ops when none exists
        (resident mode / ``speculative_prefetch`` config off).
        """
        engine = getattr(self.moe, "engine", None)
        return getattr(engine, "expert_prefetcher", None)

    def _base_executor_evidence(self) -> ExecutorEvidence:
        reachable = executor_wiring_reachable(self.moe)
        prefetcher_present = self._resolve_route_ahead_prefetcher() is not None
        reason = None
        if not reachable:
            reason = "executor_unreachable"
        elif not prefetcher_present:
            reason = "prefetcher_absent"
        return ExecutorEvidence(
            wiring_reachable=reachable,
            prefetcher_present=prefetcher_present,
            fallback_reason=reason,
        )

    def _verify_target_block(
        self,
        block: torch.Tensor,
        target_kv: Any,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Any, Any]:
        """Verify forward under the Track A3 route-ahead context.

        The context flags executor-backed MoE layers to pin + prefetch their
        ACTUAL routed expert union before reading any expert weight
        (cache warming only; routing/outputs unchanged). Token reset in
        ``finally`` makes the activation exception-safe, so a failed verify
        cannot leak route-ahead state into non-spec decode.

        A4 audit: NO resident-only guard exists anywhere on the spec path --
        ``_init_native_runtime``/``generate`` here,
        ``big_modeling._resolve_spec_strategy`` (big_modeling.py:721), and
        ``generation_loop._spec_strategy_applies`` (generation_loop.py:113)
        gate only on batch size and greedy sampling, never on expert
        residency -- so offloaded executor-backed models (DeepSeek/Qwen/
        Mixtral) already run DFlash with this seam active. gpt-oss stays
        excluded structurally, not by assertion: ``model_offload.py:954``
        never wires ``expert_executor`` into ``SyncGptOssMLP`` (its forward
        runs a resident Python expert loop), ``parse_expert_id`` yields no
        per-expert ids for it (hf_config.py:216-223), and its experts are
        force-resident (model_offload.py:1114-1123).

        A5: when ``self.route_ahead_stats`` is set, the step's observations
        are opened here (``begin_step``) and the handle is carried by the
        context; ``generate`` commits the step after the accept rule.
        """
        stats = getattr(self, "route_ahead_stats", None)
        if stats is not None:
            stats.begin_step()
        # The mask/position kwargs are forwarded only when set so the batch==1
        # call keeps its exact pre-Track-C signature (test doubles wrap it).
        kwargs: dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        with nvtx_phase("target_verify"):
            row_width = int(block.shape[1])
            row_offsets = tuple(
                row * row_width for row in range(int(block.shape[0]) + 1)
            )
            with route_ahead_context(
                self._resolve_route_ahead_prefetcher(),
                stats=stats,
                row_offsets=row_offsets,
            ):
                return self._forward_target(
                    block, past_key_values=target_kv, logits_to_keep=0, **kwargs
                )

    def _run_drafter(
        self,
        block: torch.Tensor,
        context_feature: torch.Tensor,
        start: int,
        draft_kv: Any,
    ) -> torch.Tensor:
        """One non-causal drafter pass over ``block`` -> hidden [1, block_size, H].

        Both drafter contracts KV-inject the 5-layer target feature into
        EVERY drafter layer internally. The KV drafter is fed suffix-only
        features and its cache is cropped back to ``start`` after the pass:
        the cache accumulates projected context KV only, so this block's
        noise KV is discarded (mirrors the reference ``spec_generate``).
        """
        with nvtx_phase("dflash_draft"):
            if self._drafter_has_kv_cache:
                noise_embedding = self.embed_tokens(block)
                position_ids = torch.arange(
                    draft_kv.get_seq_length(),
                    start + block.shape[1],
                    device=block.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                drafter_out = self.draft(
                    target_hidden=context_feature,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids,
                    past_key_values=draft_kv,
                    use_cache=True,
                    is_causal=False,
                )
                draft_kv.crop(start)
                return drafter_out
            return self.draft(block, context_feature)

    @torch.no_grad()
    def begin_session(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop_token_ids: Optional[List[int]] = None,
        top_k: int = 0,
        top_p: float = 1.0,
        generator: Optional[torch.Generator] = None,
        collect_route_union: bool = False,
        target_cache_adapter: CacheAdapter | None = None,
    ) -> SpecSession:
        """Prefill + anchor forward; seed a ``SpecSession`` for engine rounds.

        Mirrors the setup half of ``_generate_single`` (batch==1, greedy or the
        lossless sampled path) but stops after the anchor, handing the per-round
        loop to ``draft_round``/``verify_round``. ``collect_route_union`` opts in
        to the read-only route projection (``RouteUnionCollector``); it is off
        on ``generate()``'s path, so nothing there changes. A prefill anchor
        that is itself a stop id yields an immediately ``finished`` session
        whose only emitted token is that anchor (never cached), matching
        ``_generate_single``'s early return.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "begin_session expects input_ids of shape [1, seq], got "
                f"{tuple(input_ids.shape)}"
            )
        if float(temperature) < 0:
            raise ValueError(
                f"begin_session: temperature must be >= 0, got {temperature}"
            )

        from transformers import DynamicCache

        normalized_top_k = max(0, int(top_k))
        sampling = SamplingContext(
            temperature=float(temperature),
            top_k=normalized_top_k,
            top_p=float(top_p),
            generator=generator,
        )
        sampled = sampling.is_sampled
        if target_cache_adapter is not None and sampled:
            raise ValueError("paged target cache requires greedy sampling")
        input_ids = input_ids.to(self.device)
        if sampled:
            _validate_generator_device(generator, input_ids.device)
        self._configure_target_hooks(input_ids)

        num_prompt_tokens = int(input_ids.shape[1])
        block_size = int(self.config.block_size)
        layer_ids = list(self.config.target_layer_ids)
        max_new_tokens = int(max_new_tokens)

        if target_cache_adapter is None:
            logits, hidden_states, target_kv = self._forward_target(
                input_ids, past_key_values=None, logits_to_keep=1
            )
        else:
            build_metadata = getattr(
                target_cache_adapter, "build_attention_metadata", None
            )
            if not callable(build_metadata):
                raise TypeError(
                    "paged target cache adapter must build attention metadata"
                )
            metadata = build_metadata(
                query_length=num_prompt_tokens, is_prefill=True
            )
            logits, hidden_states, returned_handle = self._forward_target(
                input_ids,
                past_key_values=target_cache_adapter,
                logits_to_keep=1,
                attention_metadata=metadata,
            )
            engine_cache = getattr(target_cache_adapter, "cache", None)
            if returned_handle is not engine_cache:
                raise RuntimeError(
                    "paged target forward did not return the engine-owned cache"
                )
            target_kv = target_cache_adapter
        if sampled:
            anchor = int(
                torch.multinomial(
                    warped_probs(
                        logits[0, -1],
                        sampling.temperature,
                        sampling.top_k,
                        sampling.top_p,
                    ),
                    num_samples=1,
                    generator=generator,
                ).item()
            )
        else:
            anchor = int(logits[:, -1, :].argmax(dim=-1).item())
        context_feature = _extract_context_feature(hidden_states, layer_ids).to(
            self.device
        )
        stop_ids = set(_resolve_stop_ids(self.target, stop_token_ids))

        session = SpecSession(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            block_size=block_size,
            layer_ids=layer_ids,
            mask_token_id=int(self.config.mask_token_id),
            stop_ids=stop_ids,
            num_prompt_tokens=num_prompt_tokens,
            target_kv=target_kv,
            draft_kv=(DynamicCache() if self._drafter_has_kv_cache else None),
            context_feature=context_feature,
            anchor=anchor,
            start=num_prompt_tokens,
            emitted=[anchor],
            step_trace=[],
            collector=(RouteUnionCollector() if collect_route_union else None),
        )
        self.step_trace = session.step_trace
        self.last_generated_lengths = None
        if stop_ids and anchor in stop_ids and max_new_tokens >= 1:
            self.last_target_cache = target_kv
            self.last_draft_cache = session.draft_kv
            session.finished = True
        return session

    @torch.no_grad()
    def draft_round(self, session: SpecSession) -> DraftResult:
        """One drafter pass over the next block + the read-only verify demand.

        Reproduces the DRAFT half of the ``_generate_single`` loop body (build
        the ``[anchor, MASK...]`` block, one ``_run_drafter`` pass, fill the
        drafts greedily or by lossless sampling), snapshots the target cache for
        the pending verify's rollback, and returns the projected demand for
        admission. The projection is the union captured read-only from the
        PRIOR round's verify routing (``session.projected_*``); the first round
        projects the empty set (byte-free warm-up) since no verify has routed
        yet. No expert executes and no second router runs here.
        """
        if session.finished:
            raise RuntimeError("draft_round called on a finished session")
        if session.has_pending_draft:
            raise RuntimeError(
                "draft_round called with an un-verified pending block; "
                "call verify_round first"
            )

        prev_start = session.start
        block = build_block(
            session.anchor, session.mask_token_id, session.block_size
        ).to(self.device)
        drafter_out = self._run_drafter(
            block, session.context_feature, session.start, session.draft_kv
        )
        draft_logits = self.lm_head(drafter_out)[
            :, -(session.block_size - 1) :, :
        ]
        draft_probs: Optional[torch.Tensor] = None
        if session.sampled:
            draft_probs = warped_probs(
                draft_logits[0],
                session.temperature,
                session.top_k,
                session.top_p,
            )
            block[:, 1:] = torch.multinomial(
                draft_probs,
                num_samples=1,
                generator=session.sampling.generator,
            ).squeeze(-1)
        else:
            block[:, 1:] = draft_logits.argmax(dim=-1)

        session._pending_block = block
        session._pending_prev_start = prev_start
        session.pending_draft_probs = draft_probs
        if isinstance(session.target_kv, CacheAdapter):
            session._pending_cache_snapshot = session.target_kv.snapshot()
        else:
            session._pending_cache_snapshot = snapshot_target_cache(
                session.target_kv
            )
        session._pending = True

        return DraftResult(
            tokens=session.block_size,
            expert_union=session.projected_expert_union,
            expert_bytes=session.projected_expert_bytes,
        )

    @torch.no_grad()
    def verify_round(self, session: SpecSession) -> VerifyResult:
        """One verify forward + accept/commit/rollback for the pending block.

        Reproduces the VERIFY half of the ``_generate_single`` loop body exactly
        (verify under ``route_ahead_context``, accept rule, emitted/cached
        split, hybrid rollback, step trace, next anchor/context feature). When
        the session opted into the route projection, this run's ACTUAL routed
        union is collected read-only and turned into the EXACT
        ``expert_nbytes``-summed demand carried to the next ``draft_round``.
        """
        if not session.has_pending_draft or session._pending_block is None:
            raise RuntimeError(
                "verify_round called without a pending draft; "
                "call draft_round first"
            )

        block = session._pending_block
        prev_start = session._pending_prev_start
        draft_probs = session.pending_draft_probs
        cache_snapshot = session._pending_cache_snapshot
        paged_target = (
            session.target_kv
            if isinstance(session.target_kv, CacheAdapter)
            else None
        )

        collector = session.collector
        stats = (
            collector
            if collector is not None
            else getattr(self, "route_ahead_stats", None)
        )
        if stats is not None:
            stats.begin_step()
        with nvtx_phase("target_verify"):
            with route_ahead_context(
                self._resolve_route_ahead_prefetcher(),
                stats=stats,
                row_offsets=(0, int(block.numel())),
            ):
                if paged_target is None:
                    logits, hidden_states, session.target_kv = (
                        self._forward_target(
                            block,
                            past_key_values=session.target_kv,
                            logits_to_keep=0,
                        )
                    )
                else:
                    paged_target.append(session.block_size)
                    build_metadata = getattr(
                        paged_target, "build_attention_metadata", None
                    )
                    if not callable(build_metadata):
                        raise TypeError(
                            "paged target cache adapter must build attention metadata"
                        )
                    metadata = build_metadata(
                        query_length=session.block_size, is_prefill=False
                    )
                    logits, hidden_states, returned_handle = (
                        self._forward_target(
                            block,
                            past_key_values=paged_target,
                            logits_to_keep=0,
                            attention_metadata=metadata,
                        )
                    )
                    if returned_handle is not getattr(
                        paged_target, "cache", None
                    ):
                        raise RuntimeError(
                            "paged target forward replaced the engine-owned cache"
                        )

        if session.sampled:
            assert draft_probs is not None
            decision = acceptance_sampled(
                draft_probs,
                warped_probs(
                    logits[0],
                    session.temperature,
                    session.top_k,
                    session.top_p,
                ),
                block[0, 1:],
                generator=session.sampling.generator,
            )
            accept = decision.accept
            committed = committed_tokens_sampled(
                block, decision.accept, decision.final_token
            )
        else:
            posterior = logits.argmax(dim=-1).to(self.device)
            accept = acceptance_length(block, posterior)
            committed = committed_tokens(block, posterior, accept)

        step_tokens = [int(t) for t in committed.emitted[0].tolist()]
        keep = accept + 1
        stop = False
        if session.stop_ids:
            for j, tok in enumerate(step_tokens):
                if tok in session.stop_ids:
                    keep = j + 1
                    stop = True
                    break
        remaining = max(0, session.max_new_tokens - len(session.emitted))
        if keep > remaining:
            keep = remaining
            stop = True

        session.emitted.extend(step_tokens[:keep])
        cache_committed = min(keep, accept) + 1
        session.start = prev_start + cache_committed
        if paged_target is None:
            rollback_target_cache(
                session.target_kv,
                cache_snapshot,
                prev_start=prev_start,
                committed=cache_committed,
                block_size=session.block_size,
                block=block,
                replay=(
                    (
                        lambda prefix, cache: self._forward_target(
                            prefix, past_key_values=cache, logits_to_keep=0
                        )[2]
                    )
                    if cache_snapshot.linear
                    else None
                ),
            )
            target_cache_len = int(session.target_kv.get_seq_length())
        else:
            paged_target.truncate(session.start)
            target_cache_len = paged_target.logical_length()
        if target_cache_len != session.start:
            raise RuntimeError(
                "DFlash target cache length invariant violated: "
                f"expected {session.start}, got {target_cache_len}"
            )

        if collector is not None:
            collector.commit_step(kept_rows=cache_committed)
            prefetcher = self._resolve_route_ahead_prefetcher()
            union = frozenset(collector.union)
            session.projected_expert_union = union
            session.projected_expert_bytes = project_expert_bytes(
                union, getattr(prefetcher, "expert_nbytes_map", None)
            )
        elif stats is not None:
            stats.commit_step(kept_rows=cache_committed)

        session.step_trace.append(
            NativeStepTrace(
                prev_start=prev_start,
                accept=cache_committed - 1,
                start=session.start,
                emitted_len=len(session.emitted),
                target_cache_len=target_cache_len,
                draft_cache_len=(
                    int(session.draft_kv.get_seq_length())
                    if session.draft_kv is not None
                    else None
                ),
            )
        )
        self.step_trace = session.step_trace

        finished = stop or len(session.emitted) >= session.max_new_tokens
        if not finished:
            suffix = _extract_context_feature(
                hidden_states, session.layer_ids
            ).to(self.device)[:, : accept + 1, :]
            if self._drafter_has_kv_cache:
                session.context_feature = suffix
            else:
                session.context_feature = torch.cat(
                    [session.context_feature, suffix], dim=1
                )
            session.anchor = int(committed.bonus[0, 0].item())

        session.clear_pending()
        session.round_index += 1

        if finished:
            session.finished = True
            self.last_target_cache = session.target_kv
            self.last_draft_cache = session.draft_kv

        return VerifyResult(
            accepted_token_ids=step_tokens[:keep],
            accept=cache_committed - 1,
            verified_accept=accept,
            committed_count=keep,
            finished=finished,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Union[int, Sequence[int]] = 256,
        temperature: Union[float, Sequence[float]] = 0.0,
        stop_token_ids: Optional[
            Union[
                int,
                Sequence[int],
                Sequence[Optional[Sequence[int]]],
            ]
        ] = None,
        top_k: Union[int, Sequence[int]] = 0,
        top_p: Union[float, Sequence[float]] = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
        generator: Union[
            torch.Generator,
            Sequence[Optional[torch.Generator]],
            None,
        ] = None,
    ) -> torch.Tensor:
        """Native DFlash draft->verify->rollback loop (RFC 1.2).

        Per step: draft a block of ``block_size`` candidates with the
        non-causal drafter, verify the whole block in ONE target forward with
        full logits, accept the leading drafts the target's argmax agrees
        with, emit the accepted drafts plus the target's bonus token, and roll
        both KV caches back to the committed prefix. The bonus token is
        emitted but NOT cached -- it becomes the next step's anchor, so
        ``cached_len`` (``start``) always trails the emitted count by one.

        Sampling: ``temperature == 0`` is the greedy path above (byte-for-byte
        the v1 contract). ``temperature > 0`` (optionally with ``top_k`` /
        ``top_p``) runs the LOSSLESS speculative-sampling accept rule of
        ``_dflash_sample_ops``: drafts are drawn from the drafter's warped
        slot distributions and verified per slot against the identically
        warped target conditionals (rejection sampling + residual
        correction), so the emitted stream follows the exact distribution of
        plain sampled generation from the target. Greedy-vs-sampled only
        changes how tokens are CHOSEN; the commit/rollback bookkeeping below
        is shared.

        Stop handling: ``stop_token_ids`` (falling back to the target
        config's ``eos_token_id``) truncates the emitted output at the first
        stop id, inclusive, and ends the loop; ``max_new_tokens`` truncates a
        block that would overshoot the remaining budget. Neither cache is
        allowed past the last kept token: keeping ``k`` of a step's
        ``accept + 1`` emitted tokens (``k - 1`` is the stop index) commits
        ``min(k, accept) + 1`` cached tokens -- the anchor plus the first
        ``min(k, accept)`` drafts -- because the verify forward only
        produced KV for block tokens, never for the bonus.

        Batching (Track C): ``input_ids`` with batch > 1 dispatches to
        ``_generate_batched`` for bare-HF targets and wrappers that explicitly
        satisfy the row-aware rich-forward capability. Unsupported rich,
        MLA, and hybrid wrappers retain independent per-request sessions.
        ``temperature``, ``top_k``, ``top_p``, generators, budgets, and nested
        stop-id sets accept one value per row; numeric scalars and flat stop-id
        sets retain their shared/broadcast meaning.
        For batch > 1, one scalar generator is cloned to the same initial
        state for every row, so identical requests have correlated streams.
        Callers wanting independent explicit streams should pass one generator
        per row or omit the generator. Seed-exact outputs are not guaranteed
        across different batch shapes; the batched API guarantees per-request
        order/composition invariance for a fixed request and row-local stream.
        Prompts are LEFT-padded according to ``attention_mask`` (omit it when
        all prompts share one length). Ragged outputs are right-padded in the
        returned rectangle and their true new-token counts are exposed as
        ``self.last_generated_lengths``. ``attention_mask`` is ignored on the
        batch==1 path.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"DFlashSpeculator.generate expects input_ids of shape [batch, seq], got {tuple(input_ids.shape)}"
            )
        batch = int(input_ids.shape[0])
        sampling_contexts = _normalize_sampling_contexts(
            batch=batch,
            probability_device=self.device,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        stop_rows, has_per_row_stops = _normalize_stop_rows(
            self.target, stop_token_ids, batch
        )
        budgets = self._normalize_budgets(max_new_tokens, batch)
        rich_forward = callable(
            getattr(self.moe, "_native_model_forward_rich", None)
        )
        if batch == 1:
            self.rich_forward_batched = False
            return self._generate_per_request(
                input_ids,
                budgets=budgets,
                stop_rows=stop_rows,
                sampling_contexts=sampling_contexts,
                attention_mask=attention_mask,
            )
        if rich_forward:
            from moe_infinity.spec_decode.backends_rich import (
                BatchedRichBackend,
            )

            if not BatchedRichBackend(self).wrapper_supported:
                self.rich_forward_batched = False
                return self._generate_per_request(
                    input_ids,
                    budgets=budgets,
                    stop_rows=stop_rows,
                    sampling_contexts=sampling_contexts,
                    attention_mask=attention_mask,
                )
        self.rich_forward_batched = rich_forward
        self.rich_forward_batch_sizes = []
        return self._generate_batched(
            input_ids,
            max_new_tokens=budgets,
            stop_token_ids=(None if has_per_row_stops else list(stop_rows[0])),
            attention_mask=attention_mask,
            sampling_contexts=sampling_contexts,
            stop_token_ids_by_row=(stop_rows if has_per_row_stops else None),
        )

    @staticmethod
    def _normalize_budgets(
        max_new_tokens: Union[int, Sequence[int]], batch: int
    ) -> tuple[int, ...]:
        if isinstance(max_new_tokens, Sequence):
            budgets = tuple(int(value) for value in max_new_tokens)
            if len(budgets) != batch:
                raise ValueError(
                    f"per-sequence max_new_tokens has {len(budgets)} entries "
                    f"for batch size {batch}"
                )
        else:
            budgets = (int(max_new_tokens),) * batch
        if any(value < 0 for value in budgets):
            raise ValueError(
                f"max_new_tokens must be >= 0, got {list(budgets)}"
            )
        return budgets

    @torch.no_grad()
    def _generate_per_request(
        self,
        input_ids: torch.Tensor,
        *,
        budgets: tuple[int, ...],
        stop_rows: tuple[tuple[int, ...], ...],
        sampling_contexts: tuple[SamplingContext, ...],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Adapt tensor rows to canonical request sessions in row order."""
        from moe_infinity.spec_decode.backends import DFlashExecutionBackend
        from moe_infinity.spec_decode.session_driver import (
            SessionDriver,
            UnsupportedRequestError,
        )

        batch = int(input_ids.shape[0])
        if attention_mask is not None and batch > 1:
            if tuple(attention_mask.shape) != tuple(input_ids.shape):
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} != "
                    f"input_ids shape {tuple(input_ids.shape)}"
                )
            binary = (attention_mask == 0) | (attention_mask == 1)
            if not bool(torch.all(binary).item()):
                raise ValueError("attention_mask must be 0/1 valued")

        requests: list[RequestSpec] = []
        for row in range(batch):
            row_ids = input_ids[row]
            if attention_mask is not None and batch > 1:
                row_ids = row_ids[attention_mask[row].to(dtype=torch.bool)]
            requests.append(
                RequestSpec(
                    request_id=f"direct-{row}",
                    prompt_token_ids=tuple(
                        int(token) for token in row_ids.tolist()
                    ),
                    max_new_tokens=budgets[row],
                    stop_token_ids=frozenset(stop_rows[row]),
                    sampling=sampling_contexts[row],
                )
            )

        backend = DFlashExecutionBackend(
            self,
            retain_diagnostics=True,
            collect_route_union=False,
        )
        for request in requests:
            if (
                request.is_sampled
                and not backend.capabilities.supports_sampling
            ):
                raise UnsupportedRequestError(
                    f"request {request.request_id!r} has no compatible sampled backend"
                )
            if not backend.supports(request):
                mode = "sampled" if request.is_sampled else "greedy"
                raise UnsupportedRequestError(
                    f"request {request.request_id!r} has no compatible {mode} backend"
                )

        # The rich MoE seam stores its dense cache on the shell, so only one
        # request may be live at a time.  Each row still uses the canonical
        # driver/backend lifecycle; this is semantic batching, not a claim of
        # physical model batching.
        results = tuple(
            SessionDriver([backend]).run(request)[0] for request in requests
        )
        self.last_session_results = results
        self.last_session_traces = tuple(result.trace for result in results)
        lengths = [len(result.output_token_ids) for result in results]
        self.last_generated_lengths = None if batch == 1 else lengths

        pad_id = _get(_get(self.target, "config", self.target), "pad_token_id")
        pad_id = 0 if pad_id is None else int(pad_id)
        width = max(lengths, default=0)
        new_ids = torch.full(
            (batch, width),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        for row, result in enumerate(results):
            if result.output_token_ids:
                new_ids[row, : lengths[row]] = torch.tensor(
                    result.output_token_ids,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
        return torch.cat([input_ids, new_ids], dim=1)

    @torch.no_grad()
    def _generate_single(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        stop_token_ids: Optional[List[int]],
        top_k: int,
        top_p: float,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Synchronously drive the canonical single-sequence session state."""
        budget = max(0, int(max_new_tokens))
        stop_ids = _resolve_stop_ids(self.target, stop_token_ids)
        session = self.begin_session(
            input_ids,
            max_new_tokens=budget,
            temperature=float(temperature),
            stop_token_ids=stop_ids,
            top_k=int(top_k),
            top_p=float(top_p),
            generator=generator,
        )
        if len(session.output_ids) >= budget:
            session.finished = True
        while not session.finished:
            self.draft_round(session)
            self.verify_round(session)

        self.last_target_cache = session.target_kv
        self.last_draft_cache = session.draft_kv
        new_ids = torch.tensor(
            [session.output_ids],
            dtype=torch.long,
            device=session.input_ids.device,
        )
        return torch.cat([session.input_ids, new_ids], dim=1)

    @torch.no_grad()
    def _generate_batched(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Union[int, Sequence[int]],
        stop_token_ids: Optional[List[int]],
        attention_mask: Optional[torch.Tensor],
        sampling_contexts: Optional[tuple[SamplingContext, ...]] = None,
        stop_token_ids_by_row: Optional[tuple[tuple[int, ...], ...]] = None,
    ) -> torch.Tensor:
        """Adapt the legacy tensor API to a driver-owned physical cohort."""
        from moe_infinity.spec_decode.backends import PhysicalCohortBackend
        from moe_infinity.spec_decode.backends_bare_hf import (
            BatchedBareHFBackend,
        )
        from moe_infinity.spec_decode.backends_rich import (
            BatchedRichBackend,
        )
        from moe_infinity.spec_decode.session_driver import SessionDriver

        if input_ids.ndim != 2:
            raise ValueError(
                "DFlashSpeculator._generate_batched expects input_ids of shape "
                f"[batch, seq], got {tuple(input_ids.shape)}"
            )
        batch = int(input_ids.shape[0])
        budgets = self._normalize_budgets(max_new_tokens, batch)
        cohort_input_ids = input_ids.to(self.device)
        if attention_mask is None:
            cohort_attention_mask = torch.ones_like(cohort_input_ids)
        else:
            cohort_attention_mask = attention_mask.to(device=self.device)
        if tuple(cohort_attention_mask.shape) != tuple(cohort_input_ids.shape):
            raise ValueError(
                f"attention_mask shape {tuple(cohort_attention_mask.shape)} != "
                f"input_ids shape {tuple(cohort_input_ids.shape)}"
            )
        binary = (cohort_attention_mask == 0) | (cohort_attention_mask == 1)
        if not bool(torch.all(binary).item()):
            raise ValueError("attention_mask must be 0/1 valued")
        cohort_attention_mask = cohort_attention_mask.to(dtype=torch.long)

        if sampling_contexts is None:
            normalized_sampling = tuple(SamplingContext() for _ in range(batch))
        else:
            normalized_sampling = tuple(sampling_contexts)
            if len(normalized_sampling) != batch:
                raise ValueError(
                    f"sampling_contexts has {len(normalized_sampling)} entries "
                    f"for batch size {batch}"
                )
        if stop_token_ids_by_row is None:
            shared_stops = tuple(_resolve_stop_ids(self.target, stop_token_ids))
            stop_rows = tuple(shared_stops for _ in range(batch))
        else:
            stop_rows = tuple(tuple(row) for row in stop_token_ids_by_row)
            if len(stop_rows) != batch:
                raise ValueError(
                    f"stop_token_ids_by_row has {len(stop_rows)} entries "
                    f"for batch size {batch}"
                )

        requests: list[RequestSpec] = []
        for row in range(batch):
            row_ids = cohort_input_ids[row][
                cohort_attention_mask[row].to(dtype=torch.bool)
            ]
            requests.append(
                RequestSpec(
                    request_id=f"direct-{row}",
                    prompt_token_ids=tuple(
                        int(token) for token in row_ids.tolist()
                    ),
                    max_new_tokens=budgets[row],
                    stop_token_ids=frozenset(stop_rows[row]),
                    sampling=normalized_sampling[row],
                )
            )

        backend = (
            BatchedRichBackend(self)
            if callable(getattr(self.moe, "_native_model_forward_rich", None))
            else BatchedBareHFBackend(self)
        )
        run = SessionDriver(
            [cast(PhysicalCohortBackend, backend)]
        ).run_physical_cohort(
            cohort_input_ids,
            requests=tuple(requests),
            attention_mask=cohort_attention_mask,
        )
        result = run.backend_result

        self.step_trace = list(result.step_trace)
        self.last_target_cache = result.target_cache
        self.last_draft_cache = result.draft_cache
        self.last_session_results = run.results
        self.last_session_traces = tuple(row.trace for row in run.results)
        new_lengths = [len(row.output_token_ids) for row in run.results]
        self.last_generated_lengths = new_lengths
        pad_id = _get(_get(self.target, "config", self.target), "pad_token_id")
        pad_id = 0 if pad_id is None else int(pad_id)
        width = max(new_lengths) if new_lengths else 0
        new_ids = torch.full(
            (batch, width),
            pad_id,
            dtype=cohort_input_ids.dtype,
            device=cohort_input_ids.device,
        )
        for row, driver_result in enumerate(run.results):
            generated = driver_result.output_token_ids
            if generated:
                new_ids[row, : len(generated)] = torch.tensor(
                    generated,
                    dtype=cohort_input_ids.dtype,
                    device=cohort_input_ids.device,
                )
        return torch.cat([cohort_input_ids, new_ids], dim=1).to(
            input_ids.device
        )

    def run(
        self,
        *,
        engine: GenerationEngine,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        request_id: Optional[str] = None,
    ) -> List[int]:
        """``SpecDecodeStrategy`` adapter: engine list contract -> native loop.

        The engine calls this only after its greedy gate applied (batch==1,
        temperature==0, top_p==1, top_k==0); the sampling params are
        forwarded verbatim, so every production call takes the greedy path,
        while direct ``generate()`` callers may opt into the lossless sampled
        path. Target forwards route through this speculator's OWN
        ``self.moe`` rich helper -- never through ``engine`` -- so the
        standard expert dispatch is preserved. ``max_new_tokens`` comes from
        ``sampling_params.max_tokens`` and the stop ids from
        ``engine.eos_token_id`` (the native loop falls back to the target
        config's eos when the engine has none). Returns ONLY the newly
        generated token ids (prompt stripped); the engine wraps them in a
        ``GenerationResult`` and ``MoE.generate`` re-prepends the prompt.
        """
        input_ids = torch.tensor([list(prompt_token_ids)], dtype=torch.long)
        max_new_tokens = int(getattr(sampling_params, "max_tokens", 256))
        eos = getattr(engine, "eos_token_id", None)
        stop_ids = [int(eos)] if isinstance(eos, int) and eos >= 0 else None
        output = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=float(getattr(sampling_params, "temperature", 0.0)),
            stop_token_ids=stop_ids,
            top_k=int(getattr(sampling_params, "top_k", 0) or 0),
            top_p=float(getattr(sampling_params, "top_p", 1.0)),
        )
        return [int(t) for t in output[0, len(prompt_token_ids) :].tolist()]

    def supports_engine_request(
        self, sampling_params: SamplingParams, *, batch_size: int
    ) -> bool:
        """Return whether the compatibility engine may select DFlash."""
        if batch_size != 1:
            return False
        sampling = SamplingContext(
            temperature=float(getattr(sampling_params, "temperature", 0.0)),
            top_k=int(getattr(sampling_params, "top_k", 0) or 0),
            top_p=float(getattr(sampling_params, "top_p", 1.0)),
        )
        if sampling.is_sampled or getattr(sampling_params, "do_sample", False):
            return False
        from moe_infinity.spec_decode.backends import DFlashExecutionBackend

        request = RequestSpec(
            request_id="engine-capability-check",
            prompt_token_ids=(0,),
            max_new_tokens=int(getattr(sampling_params, "max_tokens", 256)),
            sampling=sampling,
        )
        return DFlashExecutionBackend(self).supports(request)


__all__ = [
    "DFlashConfig",
    "DFlashSpeculator",
    "DraftResult",
    "RouteUnionCollector",
    "SpecSession",
    "VerifyResult",
    "bind_shared_weights",
    "build_pairing_evidence",
    "executor_wiring_reachable",
    "project_expert_bytes",
    "read_dflash_config",
    "validate_drafter",
    "validate_drafter_module",
    "validate_pairing",
]
