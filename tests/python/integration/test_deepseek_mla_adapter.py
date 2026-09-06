from __future__ import annotations

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from moe_infinity.models.deepseek_mla_attention import (  # noqa: E402
    adapt_deepseek_attention,
    adapt_deepseek_model,
    clear_deepseek_mla_context,
    is_deepseek_mla_eligible,
    set_deepseek_mla_context,
)
from moe_infinity.runtime.attention_types import AttentionMetadata  # noqa: E402
from moe_infinity.serving.mla_cache import MLAPagedKVCache  # noqa: E402
from moe_infinity.spec_decode.protocols import RichForwardResult  # noqa: E402


def _metadata(
    *,
    seq_len: int,
    slots: list[int],
    prefill: bool,
    block_table: list[int] | None = None,
    seq_id: int | None = None,
) -> AttentionMetadata:
    metadata = AttentionMetadata(
        block_tables=torch.tensor(
            [block_table if block_table is not None else [0, 1]],
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        max_seq_len=seq_len,
        num_prefill_tokens=len(slots) if prefill else 0,
        num_decode_tokens=0 if prefill else len(slots),
        slot_mapping=torch.tensor(slots, dtype=torch.int64),
        is_prefill=prefill,
    )
    metadata.seq_id = seq_id
    return metadata


def _case(
    version: str,
    *,
    q_lora_rank: int | None = None,
    rope_interleave: bool = True,
):
    if version == "v2":
        modeling = pytest.importorskip(
            "transformers.models.deepseek_v2.modeling_deepseek_v2"
        )
        config_cls = transformers.DeepseekV2Config
        attention_cls = modeling.DeepseekV2Attention
        rotary_cls = modeling.DeepseekV2RotaryEmbedding
        config = config_cls(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            head_dim=8,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
        )
    else:
        modeling = pytest.importorskip(
            "transformers.models.deepseek_v3.modeling_deepseek_v3"
        )
        config_cls = transformers.DeepseekV3Config
        attention_cls = modeling.DeepseekV3Attention
        rotary_cls = modeling.DeepseekV3RotaryEmbedding
        config = config_cls(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
            n_group=1,
            topk_group=1,
            rope_interleave=rope_interleave,
        )
    return modeling, config, attention_cls, rotary_cls


def _positions(
    version: str,
    rotary: torch.nn.Module,
    hidden: torch.Tensor,
    positions: torch.Tensor,
):
    if version == "v2":
        return rotary(hidden, positions)
    return rotary(hidden, positions)


def _call(
    version: str,
    module: torch.nn.Module,
    hidden: torch.Tensor,
    positions: object,
    cache: object = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs = dict(
        hidden_states=hidden,
        attention_mask=attention_mask,
        past_key_values=cache,
        position_embeddings=positions,
    )
    return module(**kwargs)[0]


def _validation_case():
    _, config, attention_cls, rotary_cls = _case("v3")
    module = attention_cls(config, layer_idx=0).eval()
    cache = MLAPagedKVCache(4, 2, 1, 4, 4, torch.float32, torch.device("cpu"))
    cache.allocate_sequence(17, 3)
    adapt_deepseek_attention(module, cache, enabled=True)
    hidden = torch.randn(1, 3, config.hidden_size)
    positions = rotary_cls(config)(hidden, torch.arange(3).unsqueeze(0))
    return module, cache, hidden, positions


def test_mla_attention_rejects_total_len_shorter_than_query() -> None:
    module, _, hidden, positions = _validation_case()
    set_deepseek_mla_context(
        module, _metadata(seq_len=2, slots=[0, 1, 2], prefill=True)
    )
    try:
        with pytest.raises(ValueError, match="total_len .* query_len"):
            _call("v3", module, hidden, positions)
    finally:
        clear_deepseek_mla_context(module)


def test_mla_attention_rejects_short_attention_mask() -> None:
    module, _, hidden, positions = _validation_case()
    set_deepseek_mla_context(
        module, _metadata(seq_len=3, slots=[0, 1, 2], prefill=True)
    )
    try:
        with pytest.raises(ValueError, match="attention_mask last dimension"):
            _call(
                "v3",
                module,
                hidden,
                positions,
                attention_mask=torch.zeros(1, 1, 3, 2),
            )
    finally:
        clear_deepseek_mla_context(module)


def test_mla_attention_requires_seq_id_for_engine_owned_cache_access() -> None:
    module, _, hidden, positions = _validation_case()
    set_deepseek_mla_context(
        module, _metadata(seq_len=3, slots=[0, 1, 2], prefill=True)
    )
    try:
        with pytest.raises(ValueError, match="requires metadata.seq_id"):
            _call("v3", module, hidden, positions)
    finally:
        clear_deepseek_mla_context(module)


@pytest.mark.parametrize(
    ("block_table", "slots", "message"),
    [
        ([1, 0], [0, 1, 2], "block_tables do not match allocated sequence"),
        (
            [0, 1],
            [0, 1, 4],
            "slot_mapping references pages not owned by sequence",
        ),
    ],
)
def test_mla_attention_rejects_incoherent_owned_pages(
    block_table: list[int], slots: list[int], message: str
) -> None:
    module, cache, hidden, positions = _validation_case()
    set_deepseek_mla_context(
        module,
        _metadata(
            seq_len=3,
            slots=slots,
            prefill=True,
            block_table=block_table,
            seq_id=17,
        ),
    )
    try:
        with pytest.raises(ValueError, match=message):
            _call("v3", module, hidden, positions)
    finally:
        clear_deepseek_mla_context(module)


@pytest.mark.parametrize(
    ("version", "q_lora_rank", "rope_interleave"),
    [
        pytest.param("v2", None, True, id="v2-direct-q"),
        pytest.param("v2", 16, True, id="v2-lora-q"),
        pytest.param("v3", None, True, id="v3-direct-q-interleaved"),
        pytest.param("v3", 16, True, id="v3-lora-q-interleaved"),
        pytest.param("v3", None, False, id="v3-direct-q-split-rope"),
        pytest.param("v3", 16, False, id="v3-lora-q-split-rope"),
    ],
)
def test_real_upstream_attention_is_adapted_in_place_and_matches_dense(
    version: str,
    q_lora_rank: int | None,
    rope_interleave: bool,
) -> None:
    modeling, config, attention_cls, rotary_cls = _case(
        version,
        q_lora_rank=q_lora_rank,
        rope_interleave=rope_interleave,
    )
    torch.manual_seed(4)
    dense = attention_cls(config, layer_idx=0).eval()
    paged = copy.deepcopy(dense)
    parameter_ids = {
        name: id(value) for name, value in paged.named_parameters()
    }
    cache = MLAPagedKVCache(4, 2, 1, 4, 4, torch.float32, torch.device("cpu"))
    cache.allocate_sequence(17, 3)

    adapted = adapt_deepseek_attention(paged, cache, enabled=True)

    assert adapted is paged
    assert adapted.layer_idx == 0
    assert parameter_ids == {
        name: id(value) for name, value in adapted.named_parameters()
    }
    assert (
        adapted.__class__.__name__
        == f"Deepseek{version.upper()}MLAPagedAttention"
    )

    rotary = rotary_cls(config)
    prompt = torch.randn(1, 3, config.hidden_size)
    prompt_pos = torch.arange(3).unsqueeze(0)
    prompt_embeddings = _positions(version, rotary, prompt, prompt_pos)
    prompt_mask = torch.zeros(1, 1, 3, 3)
    prompt_mask.masked_fill_(
        torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1),
        torch.finfo(prompt.dtype).min,
    )
    dense_prompt = _call(
        version, dense, prompt, prompt_embeddings, attention_mask=prompt_mask
    )

    set_deepseek_mla_context(
        adapted,
        _metadata(
            seq_len=3,
            slots=[0, 1, 2],
            prefill=True,
            seq_id=17,
        ),
    )
    try:
        paged_prompt = _call(
            version,
            adapted,
            prompt,
            prompt_embeddings,
            attention_mask=prompt_mask,
        )
    finally:
        clear_deepseek_mla_context(adapted)

    assert torch.allclose(paged_prompt, dense_prompt, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(cache.get_mla_cache_tensors()[0]) > 0

    from transformers import DynamicCache

    dense_cache = DynamicCache()
    _ = _call(
        version,
        dense,
        prompt,
        prompt_embeddings,
        dense_cache,
        prompt_mask,
    )
    token = torch.randn(1, 1, config.hidden_size)
    decode_pos = torch.tensor([[3]])
    decode_embeddings = _positions(version, rotary, token, decode_pos)
    dense_decode = _call(version, dense, token, decode_embeddings, dense_cache)

    cache.append_tokens(17, 1)
    set_deepseek_mla_context(
        adapted,
        _metadata(seq_len=4, slots=[3], prefill=False, seq_id=17),
    )
    try:
        paged_decode = _call(version, adapted, token, decode_embeddings)
    finally:
        clear_deepseek_mla_context(adapted)

    assert torch.allclose(paged_decode, dense_decode, atol=2e-5, rtol=2e-5)
    assert cache is adapted._mla_cache


def test_selection_is_default_off_and_rejects_hybrid_models() -> None:
    from moe_infinity.utils.config import ArcherConfig

    assert ArcherConfig().enable_deepseek_mla_paging is False
    _, config, attention_cls, _ = _case("v3")
    module = attention_cls(config, layer_idx=0)
    cache = MLAPagedKVCache(2, 2, 1, 4, 4, torch.float32, torch.device("cpu"))

    assert adapt_deepseek_attention(module, cache) is module
    assert module.__class__ is attention_cls
    assert not is_deepseek_mla_eligible(config, enabled=False)
    config.sliding_window = 32
    assert not is_deepseek_mla_eligible(config, enabled=True)


def test_native_rich_forward_returns_engine_cache_handle_without_dynamic_cache() -> (
    None
):
    from moe_infinity.entrypoints.big_modeling import MoE

    _, config, _, _ = _case("v3")
    config.vocab_size = 32
    config.first_k_dense_replace = 1
    model_cls = transformers.DeepseekV3ForCausalLM
    model = model_cls(config).eval()
    cache = MLAPagedKVCache(4, 2, 1, 4, 4, torch.float32, torch.device("cpu"))
    modules = adapt_deepseek_model(model, cache, enabled=True)
    assert len(modules) == 1

    shell = MoE.__new__(MoE)
    shell.model = model
    shell._cached_past_key_values = None
    shell._native_attention_backend = None
    shell._native_mla_cache = cache
    shell._resolve_native_input_device = lambda: torch.device("cpu")
    cache.allocate_sequence(23, 3)
    metadata = _metadata(seq_len=3, slots=[0, 1, 2], prefill=True, seq_id=23)
    draft_cache = object()

    result = shell._native_model_forward_rich([1, 2, 3], metadata)

    assert isinstance(result, RichForwardResult)
    assert result.logits.shape == (1, 3, config.vocab_size)
    assert len(result.hidden_states) == 2
    assert result.cache_handle is cache
    assert result.cache_handle is not draft_cache
    assert shell._cached_past_key_values is None
