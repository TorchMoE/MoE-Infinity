# Changelog

All notable changes to MoE-Infinity will be documented in this file.

## [Unreleased]

### Added

- Documentation hub at `docs/README.md` for users, operators, contributors, and project-history readers.
- DFlash documentation that distinguishes direct batch-1 greedy and sampled draft/verify from the greedy-gated `MoE.generate` and serving integrations, explains the current batch>1 greedy-only constraint, and limits continuous-batching and route-ahead claims to validated paths.
- Opt-in correctness-preserving prefix KV reuse in the OpenAI continuous-batching path (`--enable-prefix-caching`, default off). Reuse is gated on a supported Qwen3 paged-attention layer registry plus real FlashInfer prefill/decode; unsupported runtimes execute the unchanged cold path with a stable disabled reason. Reuse requires exact namespace/parent-path/token identity, admits every sequence in a request group atomically with pinned leases before eviction, copies shared partial tails on write, publishes only successfully committed full prompt blocks, invalidates on `/v1/reload`, and excludes reused-prefix and non-cold requests from DFlash delegation. `--prefix-cache-max-entries` (default 1000) bounds the index; `moe_prefix_cache_*` metrics and `/admin/stats` expose lifecycle counters. Motivated by SGLang RadixAttention (<https://lmsys.org/blog/2024-01-17-sglang/>) and vLLM automatic prefix caching (<https://docs.vllm.ai/en/stable/examples/features/automatic_prefix_caching>); no universal speedup is claimed. Rollback is removing the flag and restarting.
- Unified DFlash `SessionDriver`/`SpecSession` protocol, capability-selected bare-HF and rich backends, shared trace evidence, and per-request sampling streams.
- Direct bare-HF batch-1/batch>1 greedy, sampled, and mixed-row execution with dense reconstruction, right-padded output, and `last_generated_lengths`.
- Stage 4a persistent serving sessions and default-off Stage 4b engine-owned DeepSeek V2/V3 MLA pages for eligible greedy batch-1 requests.
- No-download unified-execution benchmark/validator and compatibility assertions that fail closed on sampling, ordering, cache invariant, or ownership failures.

### Changed

- Qwen3.5-MoE keeps its text backbone resident and offloads routed experts only on the text-only path.
- GLM-5.2-FP8 keeps routed FP8 experts in the host store while non-routed FP8 weights dequantize to BF16 on load.
- Root README is now a concise discovery surface and points readers to the docs hub, model compatibility, DFlash, serving, troubleshooting, architecture, and changelog.
- Release notes are split out of README and tracked here instead of being presented as shipped releases.
- Package version is now derived from git tags by setuptools-scm and written to `moe_infinity/_version.py` at build time, replacing the manual `MOEINF_VERSION`, `NIGHTLY_BASE_VERSION`, and hardcoded `setup.py`/`__init__.py` version strings.

### Deprecated

- `MoE.generate()` emits `DeprecationWarning` in the current code and is scheduled for removal. It remains documented as the current in-process synchronous transition path; `MoE.serve()` is recommended for continuous-batching HTTP serving and is not a drop-in in-process return API. No removal version is announced.

### Fixed

- Serving-path DFlash now truncates KV cache to the committed prefix after each verify step, so emitted and cached tokens stay aligned.
- GPT-OSS resident-load path now materializes MXFP4 blocks, scales, router, biases, and attention sinks instead of leaving placeholder tensors in place.
- GLM FP8 store and reload parity now stays stable across fresh stores and reloads.
- PyPI publishing for both stable (`publish.yml`) and nightly (`publish-test.yml`): stable releases now take their version from the pushed git tag instead of always publishing `0.0.1`, and nightly sdists carry their version in `PKG-INFO` so `pip install --pre moe-infinity` no longer fails with a `MetadataInconsistent` version mismatch on rebuild.

### Known Limitations

- `MoE.generate()` retains narrower compatibility behavior than the direct API and does not generally expose sampled batch > 1.
- Stage 4b is default-off, greedy batch-1 DeepSeek V2/V3 only, resident-only, and non-preemptible; all ineligible and hybrid/Qwen paths use Stage 4a.
- No real DeepSeek or Qwen DFlash pair is claimed. GPT-OSS has named valid pairs but no executor route-ahead path.
- Required GPU fixture skips/unavailability are not rollout success; these Unreleased notes do not claim a release or unrun GPU validation.
- Multi-node distributed inference is still unsupported.
