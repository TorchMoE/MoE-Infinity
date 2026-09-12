# DeepSeek-V4-Flash-Vision-Exp weight-map delta inventory

Source of truth: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` `model.safetensors.index.json`
(72,633 tensors) and `config.json`, fetched 2026-09-05. The checkpoint uses the
**DeepSeek-native key scheme** (`layers.N.*`, `embed.weight`, `head.weight`), the
same scheme the existing V4-Flash Path B loader and the `deepseekv4` branch of
`parse_expert_id` already consume — not HF-style `model.layers.*` names.

## Top-level prefix histogram

| Count | Prefix | Meaning |
|---|---|---|
| 43 x ~1570 | `layers.0` .. `layers.42` | text decoder layers; every layer (incl. 0) carries 256 routed experts |
| 1569/1566/1573 | `mtp.0`, `mtp.1`, `mtp.2` | 3 nextn/MTP draft layers (`num_nextn_predict_layers=3`), layer-shaped incl. their own routed experts |
| 256 | `vision.blocks` | 32-block vision encoder |
| 1 + 2 | `vision.norm`, `vision.patch_embed` | vision encoder head/stem |
| 2 + 2 | `aligner.w1`, `aligner.w2` | vision-to-text aligner |
| 4 | `image_start`, `image_end`, `image_newline`, `image_pad` | learned image-token embeddings |
| 3 | `hc_head_base`, `hc_head_fn`, `hc_head_scale` | hyper-connection head (present in V4-Flash too) |
| 1 each | `embed.weight`, `head.weight`, `norm.weight` | token embedding, LM head, final norm |

## Tensor classification (drives `vision_exp.py`)

| Class | Match rule (native keys) | Handling |
|---|---|---|
| `ROUTED_EXPERT` | `layers.<L>.ffn.experts.<E>.` with `L < num_hidden_layers` (43); tensors `w1/w2/w3` x `weight` (FP4 packed) + `scale` | `OfficialExpertHostStore` host store + streaming, unchanged from V4-Flash |
| `MTP_NEXTN` | `mtp.<i>.` prefix, `i` in 0..2 | load resident only when the official module builds nextn layers; skipped for text-only greedy serving (DSpark/MTP out of scope) |
| `RESIDENT_VISION` | `vision.` / `aligner.` / `image_start|image_end|image_newline|image_pad` | load resident on `device`; never executed in text-only mode |
| `RESIDENT_TEXT` | everything else (`layers.<L>.attn.*`, `layers.<L>.ffn.gate*`, shared expert, `embed.`, `head.`, `norm.`, `hc_head_*`) | existing non-expert resident load path, unchanged |

Notes:
- Every decoder layer (including layer 0) is MoE in this checkpoint; there is no
  `first_k_dense_replace` in the config.
- The MTP layers contain their own `ffn.experts.*` tensors; classifying by the
  `mtp.` prefix (not by layer index) keeps them out of the routed-expert host
  store. The plan's original `model.layers.43+` assumption was wrong and is
  superseded by this table.
- `is_vision_exp_config` keys off the presence of `vision_n_layers` in the
  config (absent from base V4-Flash). `num_nextn_predict_layers=3` also differs
  (V4-Flash has 1) but is treated as informational, not the discriminator.

## Official inference module

The reference implementation ships inside the HF repo itself:

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --include "inference/*" "encoding/*" \
  --local-dir <DIR>
```

`inference/` covers the vision encoder + aligner, DFlash attention, MoE,
hyper-connections, and the DSpark forward path; `encoding/` maps OpenAI-style
messages (or `<image>path</image>` TXT notation) to prompt token IDs.

## Checkpoint conversion

The HF repo publishes native-format shards plus `inference/` weight-conversion
tooling (`inference/README.md` there documents it). The mp-sharding step mirrors
the V4-Flash flow in this directory's README:

```bash
python convert.py --hf-ckpt-path <HF_SNAPSHOT> --save-path <OUT> \
  --n-experts 256 --model-parallel 4
```

Whether the V4-Flash `convert.py` passes the `vision.*`/`aligner.*`/`mtp.*`
tensors through unchanged must be verified inside the v4flash docker image
during Task 2.4 (it predates this checkpoint); if it drops unknown prefixes,
extend its passthrough list before the e2e run.
