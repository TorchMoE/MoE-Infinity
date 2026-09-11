# Adaptive expert precision

Adaptive precision is opt-in: `adaptive_expert_precision` defaults to `false`. `adaptive_hbm_budget_bytes` is measured in bytes. Controls include `adaptive_policy_epoch_tokens`, `adaptive_hotness_decay`, `adaptive_promotion_threshold`, `adaptive_demotion_threshold`, `adaptive_min_residency_epochs`, `adaptive_transition_cooldown_epochs`, `adaptive_variant_build`, and `adaptive_derivative_root`.

`ExpertResidencyManager` is the sole authority. Checkpoint storage, cached representation, and execution dtype are separate. SliceMoE and DynaExq are motivation only; no speedup is guaranteed.

Candidate building writes `derivative-index.v1.json`, `quality-attestation.v1.json`, and `manifest.v1.json`, then publishes `CURRENT` last. Serving validates `checkpoint_fingerprint`, `converter_version`, `quality_attestation_sha256`, and an exact `ReleasedAdaptiveEntry`. `manifest_unapproved` causes canonical fallback.

Protected paths bypass adaptation: GPT-OSS MXFP4, GLM-5.2 FP8, DeepSeek-V4-Flash FP4, DeepSeek-V3 FP8, and GPTQ/AWQ.

| Phase policy | Adaptive precision | Path |
|---|---|---|
| off | off | legacy fallback |
| on | off | ExpertResidencyManager canonical generation 0 |
| off | on | ExpertResidencyManager variant transactions |
| on | on | ExpertResidencyManager variants plus phase utility |

Use `deterministic-v1` for canonical, static-low, and adaptive arms. Record H2D, `peak_accounted_bytes`, TTFT, TPOT p50, p90, p99, throughput, quality-attestation, and `release_gate` over five measured repetitions.

```bash
python benchmarks/adaptive_precision/bench_e2e.py --adaptive-variant-build --build-only
python benchmarks/adaptive_precision/bench_policy.py
python benchmarks/adaptive_precision/report.py results.jsonl --output report.json
```

Rollback with `{"adaptive_expert_precision": false}`, restart, and verify `disabled`; preserve `adaptive_derivatives`, `CURRENT`, manifests, and canonical files. Disabled serving does not load or mutate them.
