"""Return-contract test: the wrapper must satisfy transformers v5 decoder layers.

transformers v5's ``OlmoeDecoderLayer.forward`` does
``hidden_states = residual + self.mlp(hidden_states)`` — a bare tensor. The v4
contract returned ``(hidden_states, router_logits)``, which makes that addition
raise ``TypeError: unsupported operand type(s) for +: 'Tensor' and 'tuple'``.

A registration-layout test cannot see this: the expert weights are laid out
correctly either way, and the failure is one frame further along, in the
decoder layer. This test can, and it fails on the tuple return.

Mirrors ``test_jamba_return_contract.py`` (#206), which covers the same defect
in the Jamba wrapper.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
modeling = pytest.importorskip("transformers.models.olmoe.modeling_olmoe")

if not hasattr(modeling, "OlmoeMLP"):
    pytest.skip(
        "transformers does not expose OlmoeMLP", allow_module_level=True
    )


def test_forward_returns_bare_tensor():
    """SyncOlmoeMoEBlock.forward must return a bare (batch, seq, hidden) tensor.

    A stubbed expert_executor isolates the wrapper's own contract from the
    distributed execution layer: dispatch_local is a no-op and
    wait_dispatch_local returns an identity-shaped payload, so what is under
    test is only what forward hands back.
    """
    from transformers import OlmoeConfig

    from moe_infinity.models.olmoe import SyncOlmoeMoEBlock

    config = OlmoeConfig(
        hidden_size=16,
        intermediate_size=40,
        num_experts=2,
        num_experts_per_tok=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        architectures=["OlmoeForCausalLM"],
    )
    block = SyncOlmoeMoEBlock(config)

    hidden_states = torch.randn(2, 5, config.hidden_size)
    flat = hidden_states.view(-1, config.hidden_size).clone()

    block.expert_executor = SimpleNamespace(
        dispatch_local=lambda *args, **kwargs: None,
        wait_dispatch_local=lambda: flat,
    )

    out = block(hidden_states)

    assert isinstance(out, torch.Tensor), (
        f"forward must return a bare tensor for the transformers v5 decoder "
        f"contract, got {type(out)}"
    )
    assert out.shape == hidden_states.shape

    # The actual operation the decoder layer performs. With a tuple return this
    # raises TypeError, which is the failure this test exists to catch.
    residual = torch.randn_like(hidden_states)
    summed = residual + out
    assert summed.shape == hidden_states.shape
