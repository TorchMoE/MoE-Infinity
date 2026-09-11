# Troubleshooting

Use the entries below as symptom-first checks. Each one lists the likely cause, a quick confirmation step, and the exact fix.

## GPU OOM or cache budget mismatch

- **Symptom:** load or generate hits CUDA OOM, or the cache footprint is larger than expected.
- **Likely cause:** `device_memory_ratio` and `kv_cache_memory_ratio` leave too little room, or the offload directory is being reused for a different model.
- **How to confirm:** run `pytest tests/python/unit/test_utils_config.py -q` and print the loaded config, or inspect the live budget with `ArcherConfig.load_from_json({...})` and `MemoryCoordinator.from_config(...)`.
- **Resolution:** lower `device_memory_ratio`, set `kv_cache_memory_ratio` explicitly, and use a unique `offload_path` per model. The current prefix-cache flag and scaffolding do not participate in OpenAI request execution or change the expert cache budget; see [Serving](serving.md#prefix-caching).
- **Related guide:** [Configuration](configuration.md)

## Reused offload path or fingerprint mismatch

- **Symptom:** the loader raises `Model name mismatch`, `Model config mismatch`, or says the cache is legacy.
- **Likely cause:** `offload_path` points at a cache created for a different model or config, so `model_signature.json` no longer matches.
- **How to confirm:** open `offload_path/model_signature.json` and compare it with the current model name and config, or run `pytest tests/python/unit/test_model_signature.py -q`.
- **Resolution:** prefer a new `offload_path` for the model. If you must delete the old cache, remove only the generated MoE-Infinity offload cache after confirming it contains no user data or checkpoint source, then reload the model.
- **Related guide:** [Configuration](configuration.md)

## Missing or incompatible extension or CUDA runtime

- **Symptom:** import or load fails with `moe_infinity._store extension is required. Install with CUDA enabled.` or a similar missing-extension error.
- **Likely cause:** the source install did not build the CUDA extensions, or Torch and the CUDA toolkit do not match.
- **How to confirm:**

  ```bash
  python -c 'import importlib.util as u; print("moe_infinity._store: OK" if u.find_spec("moe_infinity._store") else "moe_infinity._store: missing")'
  ```

- **Resolution:** rebuild with the matching CUDA toolkit, reinstall with `pip install --no-build-isolation -e .`, and install the optional attention packages that you want to use.
- **Related guide:** [Environment variables](environment-variables.md)

## Unsupported SM120 or Blackwell build

- **Symptom:** the native DeepSeek-V4 path does not load on Blackwell, or the build fails around the SM120 path.
- **Likely cause:** `MOE_ENABLE_SM120` was not set during build, `CUTLASS_DIR` is missing, or the native path was intentionally disabled.
- **How to confirm:**

  ```bash
  python - <<'PY'
  import torch

  try:
      from moe_infinity.models.deepseek_v4.official_offload_adapter import (
          native_fp4_available,
      )
  except Exception as e:
      print(f'import_error={type(e).__name__}: {e}')
  else:
      print(f'native_fp4_available={native_fp4_available()}')

  print(f'cuda_capability={torch.cuda.get_device_capability() if torch.cuda.is_available() else None}')
  PY
  ```

- **Resolution:** rebuild with `MOE_ENABLE_SM120=1 CUTLASS_DIR=... pip install -e .` for the Blackwell path, or set `MOE_DSV4_FORCE_NATIVE=0` if you want the tilelang fallback.
- **Related guide:** [DeepSeek-V4-Flash](../moe_infinity/models/deepseek_v4/README.md)

## Attention backend fallback policy

- **Symptom:** the model loads with eager/other attention even when FlashAttention is installed, or FlashInfer-specific paths are absent.
- **Likely cause:** some model families intentionally choose eager; for example, the runtime forces `attn_implementation="eager"` for `glm_moe_dsa`, and otherwise selects eager whenever FlashAttention is unavailable. FlashInfer is a separate optional backend and may still be absent even if FlashAttention is present.
- **How to confirm FlashInfer availability:**

  ```bash
  python -c 'from moe_infinity.runtime.flashinfer_utils import HAS_FLASHINFER; print(f"HAS_FLASHINFER={HAS_FLASHINFER}")'
  ```

- **How to confirm FlashAttention selection:** inspect the model-family branch in `moe_infinity/runtime/model_offload.py` or print the same condition the loader uses before model construction; the stock code switches to eager for `glm_moe_dsa` and otherwise uses FlashAttention only when the runtime reports it available.
- **Resolution:** keep the fallback if correctness matters more than backend selection, or install the matching optional package and model-family path if you are validating that specific branch.
- **Related guide:** [Serving](serving.md)

## Startup or decode timeout

- **Symptom:** `/health` returns 503 with `starting` or `unhealthy`, or the logs mention a startup or decode watchdog timeout.
- **Likely cause:** model init exceeded `--startup-timeout`, or decode stalled longer than `--decode-step-timeout`.
- **How to confirm:** `curl http://localhost:8000/health` and check for the exact reason, or run `pytest tests/python/unit/test_watchdog_integration.py -q` to see the same state transitions.
- **Resolution:** raise or disable the timeout flags, then check model load, extension import, and GPU availability. The current watchdog path does not emit a py-spy dump, so use logs and timeout repros instead.
- **Related guide:** [Serving](serving.md)

## Auth rejects requests or the queue is full

- **Symptom:** requests return 401, 429, or 503 with `queue_full`.
- **Likely cause:** `MOE_API_KEYS` or `--api-key` does not match the Bearer token, the per-key rate limit is hit, or `_max_waiting_requests` is full.
- **How to confirm:** send a request with and without the Bearer token, or run `pytest tests/python/serving/test_api_routes.py -q` and look for the same status codes.
- **Resolution:** set the API key and rate limit flags consistently, lower request pressure, or raise `--max-waiting-requests` if the queue is too small.
- **Related guide:** [Serving](serving.md)

## DFlash target and drafter pairing fails

- **Symptom:** the server says it cannot configure DFlash, or sampled requests take the standard decode path.
- **Likely cause:** the drafter checkpoint is wrong, `trust_remote_code` is missing, or the target/drafter pair fails validation; sampled requests are also routed away by design in the serving path.
- **How to confirm:** check that `MoE.serve(..., speculative_draft=...)` or `--speculative-draft` was passed, and compare the setup with `docs/dflash.md` and `tests/python/dflash/test_native_e2e.py`. If the request was sampled, compare it with the serving matrix in `docs/dflash.md`.
- **Resolution:** use the matching target and drafter pair, keep greedy decode for serving, keep batch size at 1, and pass the drafter through the server hook. Deprecated `MoE.generate(..., speculative_draft=...)` remains the current in-process transition path. For sampled batch-1 DFlash, the experimental direct speculator path is available without a stable API promise.
- **Related guide:** [DFlash](dflash.md)

## Unsupported model or Transformers version

- **Symptom:** model registration fails, or a checkpoint that works in one environment fails in another.
- **Likely cause:** the checkpoint is not on the supported model list, or the installed `transformers` version is too old for that architecture.
- **How to confirm:** check the supported model table in `README.md`, then print `transformers.__version__` and compare it with the model-specific notes.
- **Resolution:** upgrade `transformers`, switch to the model-specific loader, or choose a checkpoint that is already listed as supported.
- **Related guide:** [Top-level README](../README.md)

## Multi-GPU ownership or P2P topology looks wrong

- **Symptom:** one logical GPU stays idle, ownership looks shifted, or multi-GPU transfers are slower than expected.
- **Likely cause:** `CUDA_VISIBLE_DEVICES` order does not match the intended ownership, or the topology does not give you a useful peer path.
- **How to confirm:** run `nvidia-smi topo -m`, print `torch.cuda.device_count()` after setting `CUDA_VISIBLE_DEVICES`, and compare the memory deltas in `tests/python/integration/test_multi_gpu_live.py`.
- **Resolution:** reorder `CUDA_VISIBLE_DEVICES` so the desired card is logical `cuda:0`, keep all devices on one host, and do not assume universal peer transfers.
- **Related guide:** [Single-server multi-GPU](multi-gpu.md)

## Phase-specific expert policy regression or rollback

- **Symptom:** TTFT or TPOT regresses, `prefetch_rejected` rises, or `starvation_promotions` grows after enabling the phase policy.
- **Likely cause:** the configured admission, top-k, priority, eviction weights, or starvation bound does not match the workload. The cache itself is still shared; there are no phase-specific pools.
- **How to confirm:** capture the effective config, `/admin/stats`, `/metrics`, and both benchmark JSON files before changing settings. Keep the model, offload directory, GPU visibility, seed, greedy sampling, prompt/output lengths, concurrency, and `device_memory_ratio` identical.
- **Resolution:** restart with `--no-phase-specific-expert-policy`, or set `"phase_specific_expert_policy": false` for in-process use. Leave subordinate keys in place if desired; they are inert while disabled. Reuse the existing offload directory without deleting or migrating it because phase is not persisted in tensor IDs, topology, or checkpoint metadata.
- **Compatibility check:** disabled non-paged mixed batches use one combined forward; disabled paged mixed batches run prefill then decode. `moe_expert_phase_policy_enabled` reports `0`, phase-policy counters remain zero, and legacy cache occupancy/hit-rate telemetry remains available.
- **Related guides:** [Serving](serving.md#phase-specific-expert-policy), [Configuration](configuration.md#phase-specific-expert-policy-fields), and [Benchmarking](benchmarking.md#phase-specific-expert-policy-matrix).

Do not parallel-merge adaptive expert precision with the fixed-size phase-policy
manager. Rebase it onto the shared residency lease/accounting transactions and
add combined tests before enabling both features.

## Related guides

- [Configuration](configuration.md)
- [Single-server multi-GPU](multi-gpu.md)
- [Serving](serving.md)
- [Environment variables](environment-variables.md)
- [DFlash](dflash.md)
- [Phase-policy benchmarking](benchmarking.md#phase-specific-expert-policy-matrix)
