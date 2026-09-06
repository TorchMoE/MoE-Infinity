# Paged MLA Admission Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound resident, non-preemptible `paged_mla` sessions and preserve an MLA free-block reserve by immediately routing rejected admissions to the existing Stage 4a path.

**Architecture:** `ArcherConfig` and serving-engine config expose validated block-based limits. `MLAPagedKVCache` exposes a read-only free-block count, while `SpecSessionDriver` makes an atomic synchronous admission decision before allocation using the block-rounded declared budget plus transient verify peak, records a structured decision, and maintains counters surfaced by engine stats. Existing session release naturally removes the active-session count and makes later admissions eligible.

**Tech Stack:** Python dataclasses, PyTorch cache allocator, pytest.

---

### Task 1: Configuration and cache introspection

**Files:**
- Modify: `moe_infinity/utils/config.py`
- Modify: `moe_infinity/serving/mla_cache.py`
- Test: `tests/python/unit/test_utils_config.py`
- Test: `tests/python/serving/test_mla_paged_cache.py`

- [ ] Add failing tests asserting defaults of one resident session and one reserved free MLA block, rejection of booleans/negative session caps/reserves below one, and a public `free_block_count` that tracks allocation and release.
- [ ] Run `pytest -q tests/python/unit/test_utils_config.py tests/python/serving/test_mla_paged_cache.py` and confirm the new assertions fail for missing fields/API.
- [ ] Add `max_resident_paged_speculative_sessions: int = 1` and `min_free_mla_blocks_after_admission: int = 1`, validate exact integer types with cap `>= 0` and reserve `>= 1`, and expose `MLAPagedKVCache.free_block_count` by reading the allocator count.

### Task 2: Admission, fallback, lifecycle, and statistics

**Files:**
- Modify: `moe_infinity/serving/spec_session_driver.py`
- Modify: `moe_infinity/serving/engine.py`
- Test: `tests/python/serving/test_dflash_stage4b.py`

- [ ] Add failing tests for the concurrent-session cap, insufficient post-allocation reserve, Stage 4a fallback without sampling changes, admission after release, structured per-record decisions/counters, and default-off unchanged behavior.
- [ ] Run `pytest -q tests/python/serving/test_dflash_stage4b.py` and confirm failures identify the missing admission policy.
- [ ] Pass validated engine limits into `SpecSessionDriver`; before creating `PagedCacheAdapter`, count live `paged_mla` records and compute peak demand as `ceil((prompt_tokens + max_new_tokens + dflash_block_size - 1) / cache.block_size)`. Admit only when below the cap and `free_block_count - demand >= reserve`; otherwise create the existing temporary-dynamic context immediately.
- [ ] Store reason codes (`admitted`, `session_cap`, `free_block_reserve`, or `ineligible`) on records, increment admission counters, and expose policy, active count, free blocks, and counters from `ContinuousBatchingEngine.get_stats()`.
- [ ] Run `pytest -q tests/python/serving/test_dflash_stage4b.py tests/python/serving/test_dflash_stage4a.py` and confirm all pass.

### Task 3: Serving configuration and documentation

**Files:**
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py`
- Modify: `docs/serving.md`
- Modify: `docs/dflash.md`
- Modify: `docs/superpowers/specs/2026-08-17-dflash-unified-execution-design.md`
- Modify: `docs/superpowers/plans/2026-08-17-dflash-unified-execution.md`

- [ ] Add serving CLI/config defaults for both guard fields and pass them into the engine config.
- [ ] Replace “guard required” wording with the implemented cap/reserve/fallback behavior and retain limitations: resident/no swap, no preemption, block-based estimate, no general fairness proof.

### Task 4: Verification

**Files:** All modified Python files.

- [ ] Run language-server diagnostics on every modified Python file and require zero errors.
- [ ] Run `pytest -q tests/python/unit/test_utils_config.py tests/python/serving/test_mla_paged_cache.py tests/python/serving/test_dflash_stage4b.py tests/python/serving/test_dflash_stage4a.py` once; stop after the first successful verification.

### Follow-up: Full-budget reservation and serving configuration

**Files:**
- Modify: `moe_infinity/serving/spec_session_driver.py`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py`
- Modify: `moe_infinity/entrypoints/big_modeling.py`
- Test: `tests/python/serving/test_dflash_stage4b.py`
- Test: `tests/python/serving/test_api_routes.py`

- [ ] Add failing exact-fit and one-block-short tests using
  `ceil((prompt_len + max_new_tokens + dflash_block_size - 1) / mla_block_size)`, including active
  sessions' declared-but-not-yet-allocated headroom.
- [ ] Add a failing backend-begin test proving `admitted` is not incremented
  until adapter retention and session construction succeed, while
  `begin_failed` is incremented and prompt pages are released.
- [ ] Implement per-session block budgets and dynamically subtract active
  unallocated headroom before deciding whether the candidate plus configured
  reserve fits.
- [ ] Add failing CLI/default/config-builder/programmatic-initializer tests for
  `enable_deepseek_mla_paging`,
  `max_resident_paged_speculative_sessions`, and
  `min_free_mla_blocks_after_admission`.
- [ ] Forward the three fields through `parse_args`, `_build_engine_config`, CLI
  `moe_config`, `initialize_with_model`, and `MoE.serve`, preserving defaults
  `False`, `1`, and `1`.
- [ ] Add an engine `add_request` -> `step` paged-selection test and a forced
  external MLA exhaustion test proving draft failure releases the request's
  pages, records `speculative_draft_failed`, and re-raises the allocator error.
- [ ] Update serving/DFlash docs to state that declared prompt-plus-output
  capacity is reserved and Stage 4a dense fallback can increase total GPU
  memory.
- [ ] Run the requested admission/API/serving/DFlash tests and language-server
  diagnostics once after implementation.

### Final polish: transient verify peak

**Files:**
- Modify: `moe_infinity/serving/spec_session_driver.py`
- Modify: `moe_infinity/utils/config.py`
- Modify: `moe_infinity/entrypoints/openai/api_server_v2.py`
- Test: `tests/python/serving/test_dflash_stage4b.py`
- Test: `tests/python/serving/test_api_routes.py`

- [ ] Add failing exact-fit and one-block-short tests where the DFlash verify
  block is larger than the MLA cache block. Reserve
  `ceil((prompt + max_new_tokens + dflash_block_size - 1) / mla_block_size)`.
- [ ] Add a test that appends the declared committed budget plus maximum
  transient verify tokens without allocator exhaustion after admission.
- [ ] Change standalone `SpecSessionDriver` paging to default off and make every
  direct paged test opt in explicitly; keep the engine's explicit flag wiring.
- [ ] Store transient-inclusive peak block budgets for active sessions so their
  unallocated headroom remains reserved.
- [ ] Update CLI/config help and docs to describe full declared capacity plus
  transient verify headroom, then run Ruff, LSP, and focused tests.
- [ ] Cover missing, non-integer, zero, and one-valued DFlash block sizes as
  ineligible Stage 4a fallbacks with no MLA allocation. Help text defines the
  reserve as blocks remaining after all active/new declared budgets and
  transient verify peaks.
