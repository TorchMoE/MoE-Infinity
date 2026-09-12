"""Return-contract test: the wrapper must satisfy transformers v5 decoder layers.

transformers v5's ``JambaDecoderLayer.forward`` does
``hidden_states = residual + self.feed_forward(hidden_states)`` — a bare
tensor. The v4 contract returned ``(hidden_states, router_logits)`` and would
raise ``TypeError: unsupported operand type(s) for +: 'Tensor' and 'tuple'``.
Registration-layout tests cannot see this; this test can.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
modeling = pytest.importorskip("transformers.models.jamba.modeling_jamba")

if not hasattr(modeling, "JambaMLP"):
    pytest.skip(
        "transformers does not expose JambaMLP", allow_module_level=True
    )


def test_forward_returns_bare_tensor():
    """SyncJambaMoEBlock.forward must return a bare (batch, seq, hidden) tensor.

    A stubbed expert_executor isolates the wrapper's own contract from the
    distributed execution layer: dispatch_local is a no-op and
    wait_dispatch_local returns an identity-shaped payload.
    """
    from transformers import JambaConfig

    from moe_infinity.models.jamba import SyncJambaMoEBlock

    config = JambaConfig(
        hidden_size=16,
        intermediate_size=40,
        num_experts=2,
        num_experts_per_tok=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        architectures=["JambaForCausalLM"],
    )
    block = SyncJambaMoEBlock(config)

    hidden_states = torch.randn(2, 5, config.hidden_size)
    flat = hidden_states.view(-1, config.hidden_size).clone()

    block.expert_executor = SimpleNamespace(
        dispatch_local=lambda *args, **kwargs: None,
        wait_dispatch_local=lambda: flat,
    )

    out = block(hidden_states)

    assert isinstance(out, torch.Tensor), (
        "feed_forward must return a bare tensor for transformers v5 "
        f"'residual + hidden_states'; got {type(out)}"
    )
    assert out.shape == hidden_states.shape
    # And the result must survive the decoder layer's own arithmetic.
    _ = hidden_states + out
