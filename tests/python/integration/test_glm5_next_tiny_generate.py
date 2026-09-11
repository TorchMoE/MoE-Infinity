import os

import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip(
    "transformers.models.glm5_next",
    reason="transformers >= 5.16 with glm5_next required",
)


@pytest.mark.skipif(
    os.environ.get("MOE_GLM5_NEXT_TINY") != "1",
    reason="set MOE_GLM5_NEXT_TINY=1 to run tiny-glm5_next GPU harness",
)
def test_tiny_glm5_next_generates(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from moe_infinity import MoE
    from moe_infinity.models.glm5_next import SyncGlm5NextMoEBlock
    from tests.python.integration._glm5_next_tiny import build_tiny_glm5_next

    d = build_tiny_glm5_next(str(tmp_path / "tiny"))
    model = MoE(
        d, {"offload_path": str(tmp_path / "off"), "device_memory_ratio": 0.8}
    )

    engine = model.engine if hasattr(model, "engine") else None
    if engine is not None and hasattr(engine, "expert_layer_modules"):
        assert engine.expert_layer_modules, "no expert modules were wired"
        assert all(
            isinstance(m, SyncGlm5NextMoEBlock)
            for m in engine.expert_layer_modules
        )

    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    out = model.generate(ids, max_new_tokens=8)
    assert out.shape[1] >= ids.shape[1] + 1
