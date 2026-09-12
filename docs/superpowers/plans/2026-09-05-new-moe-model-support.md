# New MoE Model Support Implementation Plan (GLM-5.3 / DeepSeek-V4-Flash-Vision-Exp / GLM-5.3-Flash)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support in MoE-Infinity for three new HuggingFace releases — validate `zai-org/GLM-5.3` (drop-in), extend the DeepSeek-V4 Path-B offload loader to `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (text-only first), and onboard the new `glm5_next` family for `zai-org/GLM-5.3-Flash`. `Qwen/Qwen3.8-Flash-Next` is explicitly deferred (prereq gates at the end).

**Architecture:** MoE-Infinity registers model families in `moe_infinity/common/constants.py` (substring match on `config.architectures[0].lower()`, guarded imports for classes newer than the `transformers>=5.3.0,<6` floor), monkey-patches each family's HF MoE block with a `Sync*MoEBlock` in `moe_infinity/runtime/model_offload.py`, and parses MoE shape/expert-id info in `moe_infinity/utils/hf_config.py`. DeepSeek-V4 additionally has a separate "Path B" official-loader flow in `moe_infinity/models/deepseek_v4/` (FP4 expert host store + `patch_moe_with_offload`), which is the seam for Vision-Exp.

**Tech Stack:** Python 3.12, PyTorch, transformers (floor 5.3.0; installed venv 5.12.1; GLM-5.3-Flash requires >= 5.16 via guarded import), pytest, safetensors, existing FP8 blockwise / FP4 E2M1 kernels in-repo.

**Delivery — draft PRs (required, PR BEFORE work):** Each phase is developed on its own feature branch and delivered as a **draft PR** against the default branch (verify with `git remote show origin | grep "HEAD branch"`; assumed `main` below). Protocol:
1. Before the phase's first task: `git checkout main && git pull && git checkout -b <phase-branch>`.
   - Phase 1: `feat/glm-5.3-family-docs`
   - Phase 2: `feat/dsv4-flash-vision-exp`
   - Phase 3: `feat/glm5-next-family`
2. Open the draft PR **before any implementation work**: the branch's initial commit is the plan document itself, committed at `docs/superpowers/plans/2026-09-05-new-moe-model-support.md` (`.sisyphus/` is gitignored; `docs/superpowers/plans/` is the repo's committed-plan convention). Push and `gh pr create --draft` immediately, so the plan is reviewable on GitHub and CI runs against every subsequent task commit. Phases 2 and 3 reference the plan committed by the Phase-1 PR instead of re-committing it (their initial commit is the phase's first task).
3. Keep the PR in draft until the phase's completion-checklist items pass; then mark ready with `gh pr ready`. Never merge from this plan; merging is a human decision.
4. GPU/docker-gated tasks (1.3, 2.4, 3.6) that cannot run in CI must have their local command output pasted into the PR description under a "Local validation evidence" heading.

**Verified facts this plan relies on (checked 2026-09-05):**
- `zai-org/GLM-5.3` config: `architectures=["GlmMoeDsaForCausalLM"]`, `model_type="glm_moe_dsa"`, 78 layers + 1 MTP, 256 routed experts, 8/tok, `first_k_dense_replace=3`, FP8 block 128x128 — byte-compatible with the already-supported GLM-5.2 family. Installed transformers 5.12.1 imports `GlmMoeDsaForCausalLM` OK.
- `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` config: `architectures=["DeepseekV4ForCausalLM"]`, `model_type="deepseek_v4"`; identical text backbone to supported `DeepSeek-V4-Flash` (43 layers, 256 FP4 routed experts, 6/tok); NEW: `vision_*` fields (32-layer encoder), `num_nextn_predict_layers=3` (was 1), `dspark_*` speculative fields, `rms_norm_eps=1e-20`, `compress_ratios` length 46 (43+3).
- `zai-org/GLM-5.3-Flash` config: `architectures=["Glm5NextForConditionalGeneration"]`, `model_type="glm5_next"` with nested `text_config` (`model_type="glm5_next_text"`): 45 layers (+1 MTP), first 3 dense, `n_routed_experts=288`, `n_shared_experts=1`, 8/tok, `moe_intermediate_size=2048`, `hidden_size=4096`, sigmoid + `noaux_tc` + `routed_scaling_factor=2.5` (same router family as GlmMoeDsa), hybrid KDA linear attention (34 layers) + DSA sparse attention (11 layers, every 4th), mHC hyper-connections (`hc_mult=4`), 24-layer vision tower, FP8 block 128x128, saved by transformers 5.16.0. Class **missing** in installed 5.12.1.
- Text tensor names in GLM-5.3-Flash use the flat `model.layers.N.*` scheme (per `quantization_config.modules_to_not_convert`); routed experts are per-expert `mlp.experts.<id>.{gate,up,down}_proj` like GLM-5.2.
- `Qwen/Qwen3.8-Flash-Next`: `architectures=["Qwen4ExpForConditionalGeneration"]`, `model_type="qwen4_exp"`; class not present in any confirmed stable transformers; 51.2B n-gram PLE table (~102 GB BF16) has no offload path in MoE-Infinity → deferred.

**Repo integration points (exact locations, current HEAD):**
- Registry: `moe_infinity/common/constants.py` — guarded imports lines 15–28, `MODEL_MAPPING_NAMES`/`MODEL_MAPPING_TYPES` lines 30–54, conditional registrations lines 59–77, `parse_expert_type` longest-key-first matching lines 80–99.
- Runtime wiring: `moe_infinity/runtime/model_offload.py` — MoE-block patching lines 828–838, GLM FP8 cast detection line 927–939 (`"GlmMoeDsa" in _arch0_cast` at 931; same pattern at ~993 and ~1065), `isinstance` expert-module list lines 1181–1198, resident-shared-experts model_type list lines 1221–1228, `_is_shared_expert_param` lines 1283–1299, un-patching lines 1268–1281.
- Config parsing: `moe_infinity/utils/hf_config.py` — `parse_expert_dtype` line 59, `moe_text_config` line 73 (uses `get_text_config()`), `parse_moe_param` line 83 (arch-substring branches; `"qwen3_5"` nested branch lines 98–103), `parse_expert_id` line 130 (arch branches; `"qwen3_5"` at 178).
- Sync block templates: `moe_infinity/models/glm_moe_dsa.py` (`SyncGlmMoeDsaMoEBlock`, routing at `_route_tokens_to_experts` lines 76–105), `moe_infinity/models/qwen3_5_moe.py` (`SyncQwen3_5MoeSparseMoeBlock`).
- DeepSeek-V4 Path B: `moe_infinity/models/deepseek_v4/` (`load_offloaded_v4_flash`, `OfficialExpertHostStore`, `patch_moe_with_offload`, `convert.py`), tests in `tests/python/v4/`, e2e harness `tests/python/v4/e2e_mp4_offload.py`.
- Docs: `README.md` Supported Models table, `docs/model-compatibility.md` matrix, `docs/glm-5.2.md`, `moe_infinity/models/deepseek_v4/README.md`, `CHANGELOG.md`.

**Global conventions for every task:**
- Run unit tests with the repo venv: `.venv/bin/python -m pytest <path> -q`.
- New families are documented as `implemented/experimental` in `docs/model-compatibility.md` until a real-checkpoint harness passes (repo legend).
- Never lower or raise the `transformers>=5.3.0,<6` floor; use guarded imports + `pytest.importorskip`/`skipif` for newer-class dependencies.
- Commit after each green step; never commit checkpoints or downloaded configs outside `tests/fixtures/`.

---

## Phase 1 — GLM-5.3 (base): registry regression test + docs (no runtime code)

GLM-5.3 shares GLM-5.2's exact architecture; it must work through the existing `glmmoedsa` path. This phase pins that with a config-level regression test and documents the checkpoint.

### Task 1.1: GLM-5.3 config regression test

**Files:**
- Create: `tests/fixtures/glm_5_3/config.json`
- Create: `tests/python/unit/test_glm53_registry.py`

- [ ] **Step 1: Add the GLM-5.3 config fixture**

Download once and strip to metadata only (no weights involved):

```bash
mkdir -p tests/fixtures/glm_5_3
.venv/bin/python - <<'EOF'
import json, urllib.request
url = "https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json"
cfg = json.load(urllib.request.urlopen(url))
cfg.pop("quantization_config", None)  # keep fixture small; quant is tested via _has_fp8_blockwise fixtures elsewhere
with open("tests/fixtures/glm_5_3/config.json", "w") as f:
    json.dump(cfg, f, indent=2)
EOF
```

Expected fixture keys (sanity-check by eye): `"architectures": ["GlmMoeDsaForCausalLM"]`, `"model_type": "glm_moe_dsa"`, `"num_hidden_layers": 78`, `"n_routed_experts": 256`, `"num_experts_per_tok": 8`, `"first_k_dense_replace": 3`, `"num_nextn_predict_layers": 1`.

- [ ] **Step 2: Write the failing-or-passing regression test**

```python
# tests/python/unit/test_glm53_registry.py
"""GLM-5.3 must resolve through the existing GlmMoeDsa registry path.

GLM-5.3 (zai-org/GLM-5.3) reuses the GLM-5.2 base architecture
(GlmMoeDsaForCausalLM, model_type=glm_moe_dsa); this test pins that the
registry, expert-type parser, and MoE-shape parser all treat the real
GLM-5.3 config exactly like GLM-5.2.
"""

import json
import os

import pytest

transformers = pytest.importorskip("transformers")
from transformers import PretrainedConfig

from moe_infinity.common.constants import (
    MODEL_MAPPING_NAMES,
    parse_expert_type,
)
from moe_infinity.utils.hf_config import parse_moe_param

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "fixtures", "glm_5_3", "config.json"
)


@pytest.fixture()
def glm53_config() -> PretrainedConfig:
    with open(FIXTURE) as f:
        payload = json.load(f)
    return PretrainedConfig.from_dict(payload)


def test_glm53_arch_resolves_to_glmmoedsa(glm53_config):
    if "glmmoedsa" not in MODEL_MAPPING_NAMES:
        pytest.skip("GlmMoeDsaForCausalLM not available in this transformers")
    assert parse_expert_type(glm53_config) == 5


def test_glm53_moe_shape(glm53_config):
    num_layers, num_experts, num_encoder_layers = parse_moe_param(glm53_config)
    assert num_layers == 78
    assert num_experts == 256
    assert num_encoder_layers == 0
```

- [ ] **Step 3: Run the test**

Run: `.venv/bin/python -m pytest tests/python/unit/test_glm53_registry.py -q`
Expected: `2 passed` (installed transformers 5.12.1 ships `GlmMoeDsaForCausalLM`). If `test_glm53_moe_shape` fails, inspect `parse_moe_param`'s glm branch in `moe_infinity/utils/hf_config.py:83` — fix the test's expectation only if the discrepancy is a deliberate off-by-MTP convention already used by GLM-5.2 tests (`tests/python/unit/test_glm_hf_config.py` is the source of truth for that convention; mirror it).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/glm_5_3/config.json tests/python/unit/test_glm53_registry.py
git commit -m "test: pin GLM-5.3 config resolution through GlmMoeDsa registry path"
```

### Task 1.2: GLM-5.3 documentation

**Files:**
- Modify: `README.md` (Supported Models table, GLM-5.2 note block at lines 58 and 83)
- Modify: `docs/model-compatibility.md` (matrix row after the GLM-5.2 row, line 25)
- Modify: `CHANGELOG.md` (Unreleased section)

- [ ] **Step 1: README — extend the GLM row**

In the Supported Models table change the GLM row to:

```markdown
| [GLM-5.2 / GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | `zai-org/GLM-5.2-FP8`, `zai-org/GLM-5.3` |
```

In the GLM note block (line 83) append one sentence:

```markdown
> GLM-5.3 (`zai-org/GLM-5.3`, native FP8) reuses the same `GlmMoeDsaForCausalLM` base as GLM-5.2 — all gains are post-training — so it runs through the identical registry entry and FP8 expert-offload path. Its chat template adds `reasoning_effort` (`low`/`high`/`max`, default `max`) and `clear_thinking`; pass them through the tokenizer's chat template as needed.
```

- [ ] **Step 2: docs/model-compatibility.md — add a row after GLM-5.2**

```markdown
| GLM-5.3 (`GlmMoeDsaForCausalLM`) | `zai-org/GLM-5.3` | `>= 5.12` | implemented/experimental | implemented/experimental | implemented/experimental (same FP8 path as GLM-5.2) | built-in MTP only | Not recorded | Same base as GLM-5.2; no GLM-5.3 real-checkpoint harness in repo yet; config-resolution pinned by `tests/python/unit/test_glm53_registry.py` |
```

- [ ] **Step 3: CHANGELOG — Unreleased note**

```markdown
- Documented GLM-5.3 (`zai-org/GLM-5.3`) as running through the existing GlmMoeDsa path; added config-resolution regression test.
```

- [ ] **Step 4: Verify docs consistency and commit**

Run: `.venv/bin/python -m pytest tests/python/unit/test_glm53_registry.py -q` (still green)

```bash
git add README.md docs/model-compatibility.md CHANGELOG.md
git commit -m "docs: register GLM-5.3 as a GLM-5.2-family checkpoint"
```

### Task 1.3 (optional, GPU + 760 GB disk gated): GLM-5.3 real-checkpoint smoke

- [ ] **Step 1: Run the existing GLM smoke harness against GLM-5.3**

Only on a host with a >= 48 GB GPU, ~760 GB free SSD, and transformers >= 5.12:

```bash
GLM_SMOKE_MODEL=zai-org/GLM-5.3 GLM_SMOKE_OFFLOAD=/ssd/moe-infinity/glm-5.3 \
  .venv/bin/python -m pytest tests/python/integration/test_glm_smoke.py -q
```

Expected: pass. (Check `tests/python/integration/test_glm_smoke.py` for the exact env-var names it honors; if it hardcodes `zai-org/GLM-5.2-FP8`, parametrize it with an env override in the same commit.) On success, upgrade the matrix row from `implemented/experimental` to `validated` for sync generation only.

- [ ] **Step 2: Commit**

```bash
git add tests/python/integration/test_glm_smoke.py docs/model-compatibility.md
git commit -m "test: allow GLM smoke harness to target GLM-5.3 via env override"
```

### Task 1.4: Draft PR for Phase 1

- [ ] **Step 1: Push the branch and open the draft PR** (skip `gh pr create` if it was already opened after Task 1.1 per the Delivery protocol; then just verify with `gh pr view`)

```bash
git push -u origin feat/glm-5.3-family-docs
gh pr create --draft \
  --title "GLM-5.3: register as GLM-5.2-family checkpoint (docs + config regression test)" \
  --body "Phase 1 of .sisyphus/plans/2026-09-05-new-moe-model-support.md

- [x] Task 1.1 config regression test (tests/python/unit/test_glm53_registry.py)
- [x] Task 1.2 docs (README, model-compatibility, CHANGELOG)
- [ ] Task 1.3 optional GPU smoke (evidence below if run)

## Local validation evidence
(paste pytest output; paste GLM smoke output here if Task 1.3 was run)"
```

- [ ] **Step 2: Confirm CI is green on the draft PR**

Run: `gh pr checks --watch`
Expected: all required checks pass. Fix failures with follow-up commits on the same branch before proceeding to the next phase.

---

## Phase 2 — DeepSeek-V4-Flash-Vision-Exp (Path B extension, text-only first)

The Vision-Exp checkpoint keeps the exact V4-Flash text backbone (same FP4 expert shapes: 43 layers x 256 experts x 3 projections), so `OfficialExpertHostStore`, the FP4 kernels, and `patch_moe_with_offload` are reused unchanged. The deltas are: (a) an official inference module from the Vision-Exp HF repo (covers vision encoder/aligner + DSpark), (b) 3 MTP/nextn layers instead of 1, (c) vision weights that must load resident (text-only mode can skip running them), (d) DSpark is out of scope (serve without speculative decoding).

### Task 2.1: Delta inventory (discovery, produces a committed notes file)

**Files:**
- Create: `moe_infinity/models/deepseek_v4/VISION_EXP_NOTES.md`

- [ ] **Step 1: Download metadata only and diff the weight maps**

```bash
mkdir -p /tmp/opencode/dsv4-vision
.venv/bin/python - <<'EOF'
import json, urllib.request
base = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/raw/main/"
for name in ("config.json", "model.safetensors.index.json"):
    urllib.request.urlretrieve(base + name, f"/tmp/opencode/dsv4-vision/{name}")
idx = json.load(open("/tmp/opencode/dsv4-vision/model.safetensors.index.json"))
keys = sorted(idx["weight_map"].keys())
prefixes = {}
for k in keys:
    p = ".".join(k.split(".")[:2])
    prefixes[p] = prefixes.get(p, 0) + 1
for p, n in sorted(prefixes.items()):
    print(f"{n:6d}  {p}")
EOF
```

- [ ] **Step 2: Classify every top-level prefix and write the notes file**

Write `moe_infinity/models/deepseek_v4/VISION_EXP_NOTES.md` containing:
1. The prefix histogram from Step 1.
2. A table assigning each prefix to exactly one class: `routed_expert` (host store), `resident_text` (loads via existing non-expert path), `resident_vision` (new: vision encoder + aligner), `mtp_nextn` (3 layers; loaded resident, used only when MTP enabled), `dspark_only` (skipped in this phase).
3. The download instructions for the Vision-Exp official `inference/` + `encoding/` modules (they live inside the HF repo itself, not a separate GitHub release):
   `hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --include "inference/*" "encoding/*" --local-dir <DIR>`.
4. Whether the repo's `inference/convert.py` (or our `moe_infinity/models/deepseek_v4/convert.py`) already handles vision + 3 nextn shards for mp-sharding, and the exact command used.

- [ ] **Step 3: Commit**

```bash
git add moe_infinity/models/deepseek_v4/VISION_EXP_NOTES.md
git commit -m "docs(v4): inventory DeepSeek-V4-Flash-Vision-Exp weight-map deltas"
```

### Task 2.2: Tensor classifier for Vision-Exp weight names

A small pure function keeps the load-path change reviewable and unit-testable without any checkpoint.

**Files:**
- Create: `moe_infinity/models/deepseek_v4/vision_exp.py`
- Create: `tests/python/v4/test_vision_exp_classify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/python/v4/test_vision_exp_classify.py
import pytest

from moe_infinity.models.deepseek_v4.vision_exp import (
    TensorClass,
    classify_vision_exp_tensor,
    is_vision_exp_config,
)


class _Cfg:
    """Minimal stand-in for the parsed Vision-Exp config."""

    num_hidden_layers = 43
    num_nextn_predict_layers = 3
    n_routed_experts = 256
    vision_n_layers = 32
    dspark_target_layer_ids = [40, 41, 42]


@pytest.mark.parametrize(
    "name,expected",
    [
        # routed experts -> host store (identical naming to V4-Flash)
        ("model.layers.5.mlp.experts.17.gate_proj.weight", TensorClass.ROUTED_EXPERT),
        ("model.layers.42.mlp.experts.255.down_proj.weight", TensorClass.ROUTED_EXPERT),
        # shared expert + gate + attention -> resident text
        ("model.layers.5.mlp.shared_experts.gate_proj.weight", TensorClass.RESIDENT_TEXT),
        ("model.layers.5.mlp.gate.weight", TensorClass.RESIDENT_TEXT),
        ("model.layers.5.self_attn.q_a_proj.weight", TensorClass.RESIDENT_TEXT),
        ("model.embed_tokens.weight", TensorClass.RESIDENT_TEXT),
        ("lm_head.weight", TensorClass.RESIDENT_TEXT),
        # nextn/MTP layers 43,44,45 (num_hidden_layers .. +num_nextn-1)
        ("model.layers.43.mlp.experts.0.gate_proj.weight", TensorClass.MTP_NEXTN),
        ("model.layers.45.eh_proj.weight", TensorClass.MTP_NEXTN),
        # vision tower / aligner -> resident vision
        ("model.vision.blocks.0.attn.qkv.weight", TensorClass.RESIDENT_VISION),
    ],
)
def test_classify(name, expected):
    assert classify_vision_exp_tensor(name, _Cfg()) == expected


def test_is_vision_exp_config():
    assert is_vision_exp_config(_Cfg())

    class _Text(_Cfg):
        vision_n_layers = None
        num_nextn_predict_layers = 1

    assert not is_vision_exp_config(_Text())
```

NOTE: the vision-prefix literal (`model.vision.` above) MUST be corrected to the real prefix recorded in `VISION_EXP_NOTES.md` (Task 2.1) before implementing — update the test first if the histogram shows e.g. `vision_model.` or `model.visual.`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/python/v4/test_vision_exp_classify.py -q`
Expected: FAIL with `ModuleNotFoundError: ... vision_exp`

- [ ] **Step 3: Implement the classifier**

```python
# moe_infinity/models/deepseek_v4/vision_exp.py
"""DeepSeek-V4-Flash-Vision-Exp support helpers.

Vision-Exp keeps the exact V4-Flash text backbone (43 layers x 256 FP4
routed experts) and adds: a resident vision encoder + aligner, 3 MTP/nextn
layers (indices num_hidden_layers .. num_hidden_layers+2), and DSpark
speculative-decoding weights that this phase does not execute.
"""

import re
from enum import Enum

# Keep in sync with VISION_EXP_NOTES.md (Task 2.1 histogram).
VISION_PREFIX = "model.vision."

_EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class TensorClass(str, Enum):
    ROUTED_EXPERT = "routed_expert"
    RESIDENT_TEXT = "resident_text"
    RESIDENT_VISION = "resident_vision"
    MTP_NEXTN = "mtp_nextn"


def is_vision_exp_config(config) -> bool:
    return (
        getattr(config, "vision_n_layers", None) is not None
        and getattr(config, "num_nextn_predict_layers", 1) > 1
    )


def classify_vision_exp_tensor(name: str, config) -> TensorClass:
    if name.startswith(VISION_PREFIX):
        return TensorClass.RESIDENT_VISION
    layer_match = _LAYER_RE.match(name)
    if layer_match is not None:
        layer_id = int(layer_match.group(1))
        if layer_id >= config.num_hidden_layers:
            return TensorClass.MTP_NEXTN
        if _EXPERT_RE.match(name) is not None:
            return TensorClass.ROUTED_EXPERT
    return TensorClass.RESIDENT_TEXT
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/python/v4/test_vision_exp_classify.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/models/deepseek_v4/vision_exp.py tests/python/v4/test_vision_exp_classify.py
git commit -m "feat(v4): tensor classifier for DeepSeek-V4-Flash-Vision-Exp weight maps"
```

### Task 2.3: Wire the classifier into `load_offloaded_v4_flash`

**Files:**
- Modify: `moe_infinity/models/deepseek_v4/load.py` (or wherever `load_offloaded_v4_flash` lives — locate with `rg -n "def load_offloaded_v4_flash" moe_infinity/models/deepseek_v4/`)
- Test: `tests/python/v4/test_vision_exp_loader.py`

- [ ] **Step 1: Write the failing loader-behavior test**

The test drives the non-expert weight-load path with a synthetic state dict; it does not need a checkpoint or GPU. Adapt the module import to the actual file found above.

```python
# tests/python/v4/test_vision_exp_loader.py
"""Vision-Exp loader must route vision weights resident, keep 3 nextn layers,
and leave the routed-expert host-store path untouched."""

import pytest

from moe_infinity.models.deepseek_v4.vision_exp import (
    TensorClass,
    classify_vision_exp_tensor,
)


class _Cfg:
    num_hidden_layers = 43
    num_nextn_predict_layers = 3
    n_routed_experts = 256
    vision_n_layers = 32
    dspark_target_layer_ids = [40, 41, 42]


def test_partition_counts():
    # Synthetic mini weight map: 2 text layers x 2 experts, 1 nextn layer,
    # 2 vision tensors, embeddings.
    names = (
        [f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight" for l in (3, 4) for e in (0, 1)]
        + ["model.layers.3.mlp.shared_experts.gate_proj.weight"]
        + ["model.layers.43.eh_proj.weight"]
        + ["model.vision.blocks.0.attn.qkv.weight", "model.vision.aligner.proj.weight"]
        + ["model.embed_tokens.weight", "lm_head.weight"]
    )
    buckets = {}
    for n in names:
        buckets.setdefault(classify_vision_exp_tensor(n, _Cfg()), []).append(n)
    assert len(buckets[TensorClass.ROUTED_EXPERT]) == 4
    assert len(buckets[TensorClass.RESIDENT_VISION]) == 2
    assert len(buckets[TensorClass.MTP_NEXTN]) == 1
    assert len(buckets[TensorClass.RESIDENT_TEXT]) == 3
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/python/v4/test_vision_exp_loader.py -q`
Expected: PASS (this pins bucket semantics before touching the loader; it fails only if Task 2.2 regressed).

- [ ] **Step 3: Extend `load_offloaded_v4_flash`**

In the loader, at the point where non-expert weights are loaded into the official module (identified in Task 2.1 Step 2 item 4), add:

```python
from moe_infinity.models.deepseek_v4.vision_exp import (
    TensorClass,
    classify_vision_exp_tensor,
    is_vision_exp_config,
)
```

and inside the weight-load loop, when `is_vision_exp_config(config)`:
- `TensorClass.ROUTED_EXPERT` for `layer_id < num_hidden_layers` -> existing host-store path (unchanged).
- `TensorClass.RESIDENT_VISION` -> load onto `device` into the official module's vision submodule (the official `inference/model.py` from the Vision-Exp repo defines it; the exact attribute name comes from Task 2.1's notes).
- `TensorClass.MTP_NEXTN` -> load resident when the official module builds nextn layers, else skip with a one-line `logger.info` (text-only greedy serving does not require them).
- `TensorClass.RESIDENT_TEXT` -> existing path (unchanged).

Also thread a `text_only: bool = True` kwarg through `load_offloaded_v4_flash`; when `text_only=True` and the official module requires vision weights at construction, load them but never call the vision forward.

- [ ] **Step 4: Run the full v4 unit suite**

Run: `.venv/bin/python -m pytest tests/python/v4/ -q -k "not e2e"`
Expected: all previously-passing tests still pass + the two new files pass.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/models/deepseek_v4/ tests/python/v4/test_vision_exp_loader.py
git commit -m "feat(v4): load DeepSeek-V4-Flash-Vision-Exp via Path B (text-only, vision resident, 3 nextn layers)"
```

### Task 2.4: E2E text-only parity (GPU + docker gated)

**Files:**
- Create: `tests/python/v4/e2e_vision_exp_text_only.py` (modeled on `tests/python/v4/e2e_mp4_offload.py`)
- Modify: `moe_infinity/models/deepseek_v4/README.md`
- Modify: `docs/model-compatibility.md`, `CHANGELOG.md`

- [ ] **Step 1: Convert the checkpoint** (inside the v4flash docker image, 4 GPUs)

Use the convert command recorded in `VISION_EXP_NOTES.md` (Task 2.1). Baseline expectation, subject to that note:

```bash
python convert.py --hf-ckpt-path <HF_SNAPSHOT of DeepSeek-V4-Flash-Vision-Exp> \
  --save-path <OUT> --n-experts 256 --model-parallel 4
```

- [ ] **Step 2: Write the e2e harness**

Copy `tests/python/v4/e2e_mp4_offload.py` to `tests/python/v4/e2e_vision_exp_text_only.py`; change: checkpoint env var to `DSV4_VISION_CKPT`, official module import to the Vision-Exp `inference/model.py`, prompts to the text-only examples under the repo's `inference/examples/` TXT format, and assert greedy decode matches the official implementation run side-by-side (same golden-parity method the mp4 harness uses: prefill argmax + N-token greedy match).

- [ ] **Step 3: Run it**

```bash
torchrun --nproc-per-node 4 tests/python/v4/e2e_vision_exp_text_only.py
```

Expected: prefill argmax + greedy tokens match the official reference for the smoke prompts.

- [ ] **Step 4: Docs**

- `moe_infinity/models/deepseek_v4/README.md`: add a "Vision-Exp" section — text-only validated scope, vision-input and DSpark explicitly not wired, `num_nextn_predict_layers=3` handling, convert + e2e commands.
- `docs/model-compatibility.md`: add row `DeepSeek-V4-Flash-Vision-Exp (DeepseekV4ForCausalLM)` — Path B only, text-only, `implemented/experimental` until Step 3 passes, then `validated for the official offload path (text-only)`.
- `CHANGELOG.md` Unreleased: one line.
- Also add a fail-fast guard for Path A: in `moe_infinity/runtime/model_offload.py`, where `deepseekv4` configs are handled, raise a clear `RuntimeError("DeepSeek-V4-Flash-Vision-Exp requires the official Path B loader; see moe_infinity/models/deepseek_v4/README.md")` when `is_vision_exp_config(self.config)` — the registry substring match would otherwise let `MoE()` attempt an HF-native load that cannot parse FP4 experts.

- [ ] **Step 5: Commit**

```bash
git add tests/python/v4/e2e_vision_exp_text_only.py moe_infinity/models/deepseek_v4/README.md docs/model-compatibility.md CHANGELOG.md moe_infinity/runtime/model_offload.py
git commit -m "feat(v4): text-only e2e harness + docs + Path-A guard for Vision-Exp"
```

### Task 2.5: Draft PR for Phase 2

- [ ] **Step 1: Push the branch and open the draft PR** (skip `gh pr create` if already opened after Task 2.1; verify with `gh pr view`)

```bash
git push -u origin feat/dsv4-flash-vision-exp
gh pr create --draft \
  --title "DeepSeek-V4-Flash-Vision-Exp: Path-B text-only offload support" \
  --body "Phase 2 of .sisyphus/plans/2026-09-05-new-moe-model-support.md

- [x] Task 2.1 weight-map delta inventory (VISION_EXP_NOTES.md)
- [x] Task 2.2 tensor classifier + unit tests
- [x] Task 2.3 loader wiring (text-only, vision resident, 3 nextn layers)
- [ ] Task 2.4 GPU/docker e2e parity (evidence below)
- Out of scope: vision-input forward, DSpark speculative decoding (documented)

## Local validation evidence
(paste tests/python/v4 pytest output; paste torchrun e2e_vision_exp_text_only.py output when run on the 4-GPU v4flash image)"
```

- [ ] **Step 2: Confirm CI is green on the draft PR**

Run: `gh pr checks --watch`
Expected: all required checks pass (e2e is local-only; its evidence lives in the PR body).

---

## Phase 3 — GLM-5.3-Flash (`glm5_next` family)

New architecture family. Router semantics are identical to GlmMoeDsa (sigmoid, `noaux_tc`, `e_score_correction_bias`, `routed_scaling_factor=2.5`, `n_group=topk_group=1`), so the Sync block subclasses `SyncGlmMoeDsaMoEBlock`. Everything non-MoE (KDA linear attention, DSA indexer, mHC hyper-connections, vision tower, MTP) stays GPU-resident and runs through HF's `glm5_next` implementation — MoE-Infinity only swaps the sparse-MoE block, exactly as it does for GLM-5.2 and Qwen3.5.

**Hard prerequisite (Task 3.0):** a transformers build that ships `Glm5NextForConditionalGeneration` (config written by 5.16.0; installed venv has 5.12.1). All code uses guarded imports; all tests use skip-guards, so the branch stays green on 5.12.1.

### Task 3.0: Environment gate

- [ ] **Step 1: Create a secondary venv for glm5_next testing**

```bash
python3.12 -m venv .venv-tf516
.venv-tf516/bin/pip install -e . --no-build-isolation --no-deps
.venv-tf516/bin/pip install "transformers>=5.16,<6" pytest safetensors accelerate
.venv-tf516/bin/python -c "from transformers import Glm5NextForConditionalGeneration; print('ok')"
```

Expected: `ok`. If the class name differs in the stable release (check `transformers/models/glm5_next/`), record the real name in this plan file and use it consistently below. Do NOT change `requirements.txt`.

### Task 3.1: Registry entry

**Files:**
- Modify: `moe_infinity/common/constants.py` (guarded imports lines 15–28, conditional registration after line 77)
- Create: `tests/python/unit/test_glm5_next_registry.py`
- Create: `tests/fixtures/glm_5_3_flash/config.json` (same download-and-strip recipe as Task 1.1 Step 1, URL `https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json`, keep `text_config`, drop `quantization_config`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/python/unit/test_glm5_next_registry.py
import json
import os

import pytest

from transformers import PretrainedConfig

from moe_infinity.common.constants import (
    MODEL_MAPPING_NAMES,
    MODEL_MAPPING_TYPES,
    parse_expert_type,
)

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "fixtures", "glm_5_3_flash", "config.json"
)

_HAS_GLM5_NEXT = "glm5next" in MODEL_MAPPING_NAMES


@pytest.fixture()
def glm53_flash_config() -> PretrainedConfig:
    with open(FIXTURE) as f:
        return PretrainedConfig.from_dict(json.load(f))


@pytest.mark.skipif(not _HAS_GLM5_NEXT, reason="transformers lacks Glm5Next classes")
def test_registry_maps_glm5next_to_expert_type_5():
    assert MODEL_MAPPING_TYPES["glm5next"] == 5


@pytest.mark.skipif(not _HAS_GLM5_NEXT, reason="transformers lacks Glm5Next classes")
def test_parse_expert_type(glm53_flash_config):
    assert parse_expert_type(glm53_flash_config) == 5


def test_fail_fast_when_unregistered(glm53_flash_config):
    if _HAS_GLM5_NEXT:
        pytest.skip("registered in this environment")
    with pytest.raises(RuntimeError, match="glm5nextforconditionalgeneration"):
        parse_expert_type(glm53_flash_config)
```

- [ ] **Step 2: Run to verify current behavior**

Run: `.venv/bin/python -m pytest tests/python/unit/test_glm5_next_registry.py -q`
Expected on 5.12.1: `2 skipped, 1 passed` — the fail-fast test passes TODAY (this pins the current unsupported-model error message).

- [ ] **Step 3: Implement the registry entry**

Append to the guarded imports block in `moe_infinity/common/constants.py` (after line 28):

```python
try:
    from transformers import Glm5NextForConditionalGeneration
except ImportError:
    Glm5NextForConditionalGeneration = None
```

Append after the `glmmoedsa` registration (line 77):

```python
# GLM-5.3-Flash (arch "Glm5NextForConditionalGeneration", model_type
# "glm5_next") routes 288 per-expert gate_proj/up_proj/down_proj experts with
# the same sigmoid/noaux_tc router family as GlmMoeDsa (expert-type 5). The
# hybrid KDA linear-attention layers, DSA indexer, mHC hyper-connections, and
# vision tower stay resident. Requires transformers >= 5.16; registered only
# when the class is importable (mirrors the guards above).
if Glm5NextForConditionalGeneration is not None:
    MODEL_MAPPING_NAMES["glm5next"] = Glm5NextForConditionalGeneration
    MODEL_MAPPING_TYPES["glm5next"] = 5
```

(`"glm5next"` is a prefix of `"glm5nextforconditionalgeneration".lower()`, and longest-key-first matching in `parse_expert_type` keeps it from colliding with other keys.)

- [ ] **Step 4: Run in both venvs**

Run: `.venv/bin/python -m pytest tests/python/unit/test_glm5_next_registry.py tests/python/unit/test_model_registry.py -q` — expected `passed` + skips.
Run: `.venv-tf516/bin/python -m pytest tests/python/unit/test_glm5_next_registry.py -q` — expected `2 passed, 1 skipped`.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/common/constants.py tests/python/unit/test_glm5_next_registry.py tests/fixtures/glm_5_3_flash/config.json
git commit -m "feat: register glm5_next (GLM-5.3-Flash) family behind a transformers guard"
```

### Task 3.2: Config parsing (`parse_moe_param`, `parse_expert_id`, `parse_expert_dtype`)

**Files:**
- Modify: `moe_infinity/utils/hf_config.py` (branches at lines 83–127 and 130–190)
- Create: `tests/python/unit/test_glm5_next_hf_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/python/unit/test_glm5_next_hf_config.py
import json
import os

import pytest

from transformers import PretrainedConfig

from moe_infinity.utils.hf_config import parse_expert_id, parse_moe_param

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "fixtures", "glm_5_3_flash", "config.json"
)


@pytest.fixture()
def cfg() -> PretrainedConfig:
    with open(FIXTURE) as f:
        return PretrainedConfig.from_dict(json.load(f))


def test_parse_moe_param_reads_nested_text_config(cfg):
    num_layers, num_experts, num_encoder_layers = parse_moe_param(cfg)
    assert num_layers == 45
    assert num_experts == 288
    assert num_encoder_layers == 0


@pytest.mark.parametrize(
    "name,expected_layer,expected_expert",
    [
        ("model.layers.3.mlp.experts.0.gate_proj.weight", 3, 0),
        ("model.layers.44.mlp.experts.287.down_proj.weight", 44, 287),
    ],
)
def test_parse_expert_id_routed(cfg, name, expected_layer, expected_expert):
    layer_id, expert_id = parse_expert_id(name, cfg)
    assert layer_id == expected_layer
    assert expert_id == expected_expert


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.layers.3.mlp.gate.weight",
        "model.layers.0.self_attn.A_log",
        "model.embed_tokens.weight",
    ],
)
def test_parse_expert_id_non_expert(cfg, name):
    _, expert_id = parse_expert_id(name, cfg)
    assert expert_id is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/python/unit/test_glm5_next_hf_config.py -q`
Expected: FAIL (the `glm5_next` arch hits no branch — verify the actual failure mode: either wrong values or an exception from a fallthrough branch).

- [ ] **Step 3: Implement the branches**

In `parse_moe_param` (`moe_infinity/utils/hf_config.py:83`), add a branch BEFORE any broader `glm`/`deepseek` substring branch (order matters — inspect neighboring branches and insert so `glm5next` is tested before any branch whose substring also matches):

```python
elif "glm5next" in arch:
    text = moe_text_config(config)
    num_layers = text.num_hidden_layers
    num_experts = text.n_routed_experts
```

In `parse_expert_id` (`:130`), add the matching branch. The routed-expert naming is identical to GlmMoeDsa (`model.layers.<L>.mlp.experts.<E>.<proj>.weight`), so reuse that regex; only the config source differs (nested `text_config`). Copy the exact regex from the existing glmmoedsa branch in the same function rather than re-deriving it.

In `parse_expert_dtype` (`:59`): GLM-5.3-Flash's top-level config carries the same `quantization_config` FP8-blockwise shape as GlmMoeDsa; confirm the existing FP8 detection reads the TOP-LEVEL config (it does for GLM-5.2) and that the fixture-derived path returns the same dtype code as GLM-5.2. If detection reads a field that only exists flat, add `moe_text_config()` fallback the same way `parse_moe_param` does.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/python/unit/test_glm5_next_hf_config.py tests/python/unit/test_glm_hf_config.py -q`
Expected: all pass (GLM-5.2 tests must not regress).

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/utils/hf_config.py tests/python/unit/test_glm5_next_hf_config.py
git commit -m "feat: parse glm5_next nested text_config in MoE param/expert-id parsing"
```

### Task 3.3: `SyncGlm5NextMoEBlock`

**Files:**
- Create: `moe_infinity/models/glm5_next.py`
- Modify: `moe_infinity/models/__init__.py` (export)
- Create: `tests/python/unit/test_glm5_next_moe_block.py`

- [ ] **Step 1: Write the failing routing-parity test**

```python
# tests/python/unit/test_glm5_next_moe_block.py
"""Glm5Next routing must be numerically identical to GlmMoeDsa routing when
given the same router params (sigmoid + noaux_tc + e_score_correction_bias)."""

import pytest
import torch

pytest.importorskip("transformers")

from moe_infinity.models.glm5_next import SyncGlm5NextMoEBlock


class _TextCfg:
    hidden_size = 64
    moe_intermediate_size = 32
    n_routed_experts = 8
    n_shared_experts = 1
    num_experts_per_tok = 2
    n_group = 1
    topk_group = 1
    norm_topk_prob = True
    routed_scaling_factor = 2.5
    hidden_act = "silu"


class _Cfg:
    text_config = _TextCfg()

    def get_text_config(self):
        return self.text_config


def test_block_constructs_and_routes():
    torch.manual_seed(0)
    block = SyncGlm5NextMoEBlock(_Cfg())
    hidden = torch.randn(1, 4, 64)
    topk_idx, topk_w = block._route(hidden.view(-1, 64))
    assert topk_idx.shape == (4, 2)
    assert topk_w.shape == (4, 2)
    assert (topk_idx >= 0).all() and (topk_idx < 8).all()
    # noaux_tc + norm_topk_prob + routed_scaling_factor: weights positive,
    # per-token sum == routed_scaling_factor after normalization.
    torch.testing.assert_close(
        topk_w.sum(dim=-1), torch.full((4,), 2.5), rtol=1e-4, atol=1e-4
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/python/unit/test_glm5_next_moe_block.py -q`
Expected: FAIL with `ModuleNotFoundError: ... glm5_next`

- [ ] **Step 3: Implement the block**

```python
# moe_infinity/models/glm5_next.py
"""Sync MoE block for GLM-5.3-Flash (glm5_next).

Router semantics are identical to GlmMoeDsa (sigmoid scoring, noaux_tc
top-k with e_score_correction_bias, routed_scaling_factor renorm), so this
subclasses SyncGlmMoeDsaMoEBlock and only adapts config access: glm5_next
nests all MoE fields under config.text_config.
"""

from moe_infinity.models.glm_moe_dsa import SyncGlmMoeDsaMoEBlock
from moe_infinity.utils.hf_config import moe_text_config


class SyncGlm5NextMoEBlock(SyncGlmMoeDsaMoEBlock):
    def __init__(self, config):
        super().__init__(moe_text_config(config))
```

IMPORTANT: before relying on the subclass, open `moe_infinity/models/glm_moe_dsa.py:31-60` and verify `SyncGlmMoeDsaMoEBlock.__init__` reads ONLY these config fields: `hidden_size`, `moe_intermediate_size`, `n_routed_experts`, `n_group`, `topk_group`, `norm_topk_prob`, `routed_scaling_factor`, `num_experts_per_tok`, `n_shared_experts`, `hidden_act`. All exist in glm5_next's `text_config` with the same names (verified against the real config). If `__init__` also touches a field absent from `text_config` (e.g. `first_k_dense_replace` — present in glm5_next too, or `moe_router_dtype` — also present), add it to `_TextCfg` in the test and proceed; if it touches a genuinely missing field, override that access in the subclass instead of subclassing blindly.

Export it from `moe_infinity/models/__init__.py` next to the existing exports:

```python
from .glm5_next import SyncGlm5NextMoEBlock  # noqa: F401
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/python/unit/test_glm5_next_moe_block.py tests/python/unit/test_glm_routing.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/models/glm5_next.py moe_infinity/models/__init__.py tests/python/unit/test_glm5_next_moe_block.py
git commit -m "feat: SyncGlm5NextMoEBlock reusing GlmMoeDsa router semantics"
```

### Task 3.4: Runtime wiring in `model_offload.py`

**Files:**
- Modify: `moe_infinity/runtime/model_offload.py` (five locations listed below)
- Create: `tests/python/unit/test_glm5_next_offload_wiring.py` (model on `tests/python/unit/test_glm_offload_wiring.py` — read that file first and mirror its mocking style exactly)

- [ ] **Step 1: Write the failing wiring test**

Read `tests/python/unit/test_glm_offload_wiring.py` and clone its structure for glm5_next, asserting:
1. When transformers ships `glm5_next`, entering the offload context replaces `transformers.models.glm5_next.modeling_glm5_next.<MoE class>` with `SyncGlm5NextMoEBlock` and restores it on exit (mirror how that test asserts the GlmMoeDsa patch at lines 832–838 / 1275–1281).
2. `_is_shared_expert_param` returns True for `model.layers.0.self_attn.A_log` and `model.embed_tokens.weight` and False for `model.layers.3.mlp.experts.5.up_proj.weight` when `config.model_type == "glm5_next"`.

Gate the whole module with `pytest.importorskip("transformers.models.glm5_next")` so it runs only in `.venv-tf516`.

- [ ] **Step 2: Run to verify skip/failure**

Run: `.venv/bin/python -m pytest tests/python/unit/test_glm5_next_offload_wiring.py -q` — expected: `skipped` on 5.12.1.
Run: `.venv-tf516/bin/python -m pytest tests/python/unit/test_glm5_next_offload_wiring.py -q` — expected: FAIL.

- [ ] **Step 3: Implement the five wiring changes**

(a) Patch block — after the GlmMoeDsa patch at line 838, add (determine `<MoEClassName>` from `transformers/models/glm5_next/modeling_glm5_next.py` in the 5.16 venv; it is the module class holding `.experts` + `.gate` + `.shared_experts`):

```python
        try:
            import transformers.models.glm5_next.modeling_glm5_next as _glm5n_mod

            _glm5n_mod._old_glm5_next_moe = _glm5n_mod.<MoEClassName>
            _glm5n_mod.<MoEClassName> = SyncGlm5NextMoEBlock
        except (ImportError, AttributeError):
            pass
```

(b) Un-patch block — mirror at lines 1275–1281:

```python
        try:
            import transformers.models.glm5_next.modeling_glm5_next as _glm5n_mod

            if hasattr(_glm5n_mod, "_old_glm5_next_moe"):
                _glm5n_mod.<MoEClassName> = _glm5n_mod._old_glm5_next_moe
        except (ImportError, AttributeError):
            pass
```

(c) `isinstance` expert-module list (lines 1181–1194): add `or isinstance(module, SyncGlm5NextMoEBlock)`.

(d) Resident shared-expert model_type list (lines 1221–1224): add `"glm5_next"`:

```python
                if getattr(self.config, "model_type", "") in (
                    "glm_moe_dsa",
                    "qwen3_5_moe",
                    "glm5_next",
                ):
```

(e) `_is_shared_expert_param` (lines 1283–1299): add a glm5_next branch mirroring the qwen3_5 policy but for flat names — everything except routed experts stays resident (KDA layers, mHC, indexer, vision tower, MTP, embeddings, lm_head):

```python
        if getattr(self.config, "model_type", "") == "glm5_next":
            _, expert_id = parse_expert_id(name, self.config)
            return expert_id is None
```

(f) FP8 cast detection — extend the three arch checks (lines 931, ~993, ~1065) from `"GlmMoeDsa" in _arch0_cast` to `("GlmMoeDsa" in _arch0_cast or "Glm5Next" in _arch0_cast)` (keep variable names per site). This reuses the GLM-5.2 rule: routed experts stay FP8 in the host store; non-routed FP8 weights dequant to BF16 on load.

Import `SyncGlm5NextMoEBlock` next to the `SyncGlmMoeDsaMoEBlock` import at line 53.

- [ ] **Step 4: Run**

Run: `.venv-tf516/bin/python -m pytest tests/python/unit/test_glm5_next_offload_wiring.py tests/python/unit/test_glm_offload_wiring.py -q` — expected: pass.
Run: `.venv/bin/python -m pytest tests/python/unit -q` — expected: no regressions (glm5_next tests skip).

- [ ] **Step 5: Commit**

```bash
git add moe_infinity/runtime/model_offload.py tests/python/unit/test_glm5_next_offload_wiring.py
git commit -m "feat: wire glm5_next into offload runtime (patch, resident policy, FP8 cast)"
```

### Task 3.5: Tiny-fixture integration test

**Files:**
- Create: `tests/python/integration/test_glm5_next_tiny_generate.py`

- [ ] **Step 1: Read the template**

Read `tests/python/integration/test_glm_tiny_generate.py` end-to-end. It builds a tiny random GlmMoeDsa checkpoint on disk and runs `MoE(...).generate()` on CPU-scale shapes. Clone its builder for a tiny `Glm5NextConfig` (2 KDA layers + 1 DSA layer + 1 dense layer, 4 experts, hidden 64), including the nested `text_config` and a minimal `vision_config` (the smallest the HF class accepts).

- [ ] **Step 2: Write the test** (guarded by `pytest.importorskip("transformers.models.glm5_next")`), asserting: model loads through `MoE`, routed experts are offloaded (assert `expert_layer_modules` non-empty and every module is a `SyncGlm5NextMoEBlock`), and an 8-token greedy `generate()` completes and is deterministic across two runs.

- [ ] **Step 3: Run**

Run: `.venv-tf516/bin/python -m pytest tests/python/integration/test_glm5_next_tiny_generate.py -q`
Expected: pass. Debug notes: `attn_implementation="eager"` must be forced for glm5_next exactly as the GLM-5.2 path does (find where `model_offload.py` forces eager for `glm_moe_dsa` — `rg -n "eager" moe_infinity/runtime/model_offload.py` — and extend that condition to `glm5_next`; if this was missed, this test is where it surfaces).

- [ ] **Step 4: Commit**

```bash
git add tests/python/integration/test_glm5_next_tiny_generate.py moe_infinity/runtime/model_offload.py
git commit -m "test: tiny glm5_next end-to-end generate through the offload runtime"
```

### Task 3.6: Real-checkpoint smoke (GPU gated) + docs

- [ ] **Step 1: Real-checkpoint smoke** — on a >= 48 GB GPU host with ~330 GB free SSD and the 5.16 venv:

```python
# scripts/smoke_glm5_next.py (new, argparse: --model zai-org/GLM-5.3-Flash --offload-dir ...)
from moe_infinity import MoE
model = MoE(args.model, {"offload_path": args.offload_dir, "device_memory_ratio": 0.5})
# text-only prompt through the tokenizer chat template; assert non-empty completion.
```

Run: `CUDA_VISIBLE_DEVICES=0 .venv-tf516/bin/python scripts/smoke_glm5_next.py --model zai-org/GLM-5.3-Flash --offload-dir /ssd/moe-infinity/glm-5.3-flash`
Expected: coherent greedy completion; resident GPU footprint roughly ~34 GB BF16 backbone + expert cache.

- [ ] **Step 2: Docs** — `README.md` Supported Models row + note block (text-only; vision weights resident-unused; requires transformers >= 5.16; FP8 experts stay FP8 in host store), `docs/model-compatibility.md` row (status per what actually passed), new `docs/glm-5.3-flash.md` modeled on `docs/glm-5.2.md`, `CHANGELOG.md` line.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/model-compatibility.md docs/glm-5.3-flash.md CHANGELOG.md scripts/smoke_glm5_next.py
git commit -m "docs: GLM-5.3-Flash (glm5_next) support notes and smoke script"
```

### Task 3.7: Draft PR for Phase 3

- [ ] **Step 1: Push the branch and open the draft PR** (skip `gh pr create` if already opened after Task 3.1; verify with `gh pr view`)

```bash
git push -u origin feat/glm5-next-family
gh pr create --draft \
  --title "GLM-5.3-Flash: new glm5_next family (registry, parsing, Sync MoE block, runtime wiring)" \
  --body "Phase 3 of .sisyphus/plans/2026-09-05-new-moe-model-support.md

Requires transformers >= 5.16 (guarded import; repo floor unchanged, CI on 5.12.1 skips the new tests cleanly).

- [x] Task 3.1 registry entry (glm5next, expert-type 5)
- [x] Task 3.2 hf_config parsing (nested text_config)
- [x] Task 3.3 SyncGlm5NextMoEBlock
- [x] Task 3.4 model_offload.py wiring (patch/unpatch, resident policy, FP8 cast)
- [x] Task 3.5 tiny-fixture integration test
- [ ] Task 3.6 real-checkpoint GPU smoke (evidence below)

## Local validation evidence
(paste .venv-tf516 pytest output for the glm5_next unit + tiny integration tests; paste scripts/smoke_glm5_next.py output when run)"
```

- [ ] **Step 2: Confirm CI is green on the draft PR**

Run: `gh pr checks --watch`
Expected: all required checks pass on transformers 5.12.1 (glm5_next tests skip); the 5.16 evidence lives in the PR body.

---

## Phase 4 — Qwen3.8-Flash-Next: DEFERRED (prereq gates only — do not implement from this plan)

Blocked; revisit when ALL gates pass. No tasks below are executable now, by design.

- **Gate A (transformers):** `Qwen4ExpForConditionalGeneration` importable from a pinned stable `transformers<6` release (absent from 5.12.1; config written by `5.8.0.dev0`, suggesting a vendor branch).
- **Gate B (PLE design):** a written design for hosting the 51.2B-param n-gram PLE table (~102 GB BF16, 128 shards at layer 2) in pinned host memory with row-granular async gather. This is a NEW engine capability (MoE-Infinity currently offloads whole expert tensors, not embedding rows) and must be its own plan.
- **Gate C (stability):** Qwen explicitly labels the checkpoint an experimental Qwen4-architecture preview; wait for a non-"Exp" sibling or confirm the architecture is frozen.
- Notes for the future plan: experts are PACKED `model.language_model.layers.N.mlp.experts.{gate_up_proj,down_proj}` (the `_remap_v5_batched_experts` expansion at `model_offload.py:923` is the precedent); 24,576 experts of ~9.8 MB each stress per-expert bookkeeping — add a registry-scale test; GDN/QSA/gated-residual/vision all stay resident (qwen3_5 precedent).

---

## Cross-cutting completion checklist

- [ ] `.venv/bin/python -m pytest tests/python/unit -q` green on transformers 5.12.1 (new glm5_next tests skip cleanly).
- [ ] `.venv-tf516/bin/python -m pytest tests/python/unit tests/python/integration/test_glm5_next_tiny_generate.py -q` green on transformers >= 5.16.
- [ ] `docs/model-compatibility.md` statuses match what was actually executed (repo legend: no `validated` without a harness run recorded).
- [ ] No change to `requirements.txt` transformers floor.
- [ ] `git log` shows one commit per green step; no checkpoint/weight files committed.
- [ ] One **draft PR per phase** exists (Tasks 1.4 / 2.5 / 3.7), CI green, GPU-gated evidence pasted in the PR body; PRs stay draft until their phase checklist passes (`gh pr ready` only then; merging is a human decision).

## Execution order & independence

Phases 1, 2, 3 are independent and individually shippable; recommended order 1 → 2 → 3 (ascending risk). Each phase lives on its own branch and its own draft PR (see Delivery protocol in the header). Within each phase, tasks are strictly ordered. Phase 4 must not be started from this plan.
