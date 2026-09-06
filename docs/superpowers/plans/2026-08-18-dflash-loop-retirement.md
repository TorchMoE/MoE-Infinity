# DFlash Legacy Batch Loop Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `_generate_batched` a compatibility adapter whose requests execute only through a capability-selected `SessionDriver` physical cohort.

**Architecture:** Add a physical-cohort run result and `SessionDriver.run_physical_cohort` beside the existing request-scoped `run` lifecycle. The driver validates every request and selects one compatible batch backend before model execution, calls the backend once, validates the complete returned cohort, then atomically publishes row-level `DriverResult` objects and traces. The legacy tensor method retains its signature and handles only normalization, request construction, driver invocation, diagnostics, and rectangular output adaptation.

**Tech Stack:** Python, PyTorch, pytest, Python `ast`, basedpyright/LSP.

---

### Task 1: Lock Down Loop Retirement and Driver Semantics

**Files:**
- Create: `tests/python/dflash/test_loop_retirement.py`
- Test: `tests/python/dflash/test_loop_retirement.py`

- [ ] **Step 1: Write failing structural tests**

Parse `DFlashSpeculator._generate_batched` with `ast`, resolve called attribute/name tails, and assert it contains no decode `while` loop and no calls to acceptance, committed-token, warped-probability, target/draft forward, cache snapshot/rollback, or route-stat operations. Assert that it constructs `RequestSpec`, `BatchedBareHFBackend`, and `SessionDriver`, and invokes `run_physical_cohort`.

- [ ] **Step 2: Write failing physical-cohort driver tests**

Use a fake `PhysicalCohortBackend` to prove all `supports`/capability checks precede `execute_cohort`, sampled requests never downgrade to a non-sampling backend, unsupported or incompatible rows fail before execution, returned row count/budgets/traces are validated atomically, successful rows become ordered `DriverResult` values, and backend exceptions leave `last_results` empty.

- [ ] **Step 3: Write failing adapter parity tests**

Cover greedy, sampled, and mixed requests with scalar/per-row budgets, sampling contexts/generators, stop sets, and masks. Verify one physical driver invocation, right-padded output, batch-one shape, `last_generated_lengths`, caches, aggregate step trace, row `DriverResult` values, session traces, and route/pairing/executor evidence.

- [ ] **Step 4: Verify RED**

Run: `pytest -q tests/python/dflash/test_loop_retirement.py`

Expected: failures because `SessionDriver.run_physical_cohort` and the physical run result do not exist and `_generate_batched` still invokes `execute_cohort` directly.

### Task 2: Add the Physical Cohort Driver Entry

**Files:**
- Modify: `moe_infinity/spec_decode/session_driver.py`
- Modify: `moe_infinity/spec_decode/__init__.py`
- Test: `tests/python/dflash/test_loop_retirement.py`

- [ ] **Step 1: Define the physical run result**

Add an immutable `PhysicalCohortDriverResult` containing ordered `DriverResult` rows plus the opaque backend cohort result so compatibility adapters can retain backend-owned cache diagnostics without duplicating execution logic.

- [ ] **Step 2: Implement capability-first physical selection**

Normalize and uniquely identify requests, filter to runtime `PhysicalCohortBackend` implementations with `supports_batch`, preserve sampling capability checks, validate hashable/common cohort keys, and reject a physical cohort that would require backend/key splitting before calling any backend execution method.

- [ ] **Step 3: Implement one atomic physical execution**

Validate tensor/mask rank and row counts, derive shared or per-row stops and row sampling contexts, call `execute_cohort` once, validate generated rows and session traces for every request, convert them to `DriverResult` values, and only then assign `last_results` and return the wrapper. On every exception retain `last_results == ()`.

- [ ] **Step 4: Verify focused driver tests pass**

Run: `pytest -q tests/python/dflash/test_loop_retirement.py -k 'driver or physical'`

Expected: PASS.

### Task 3: Reduce `_generate_batched` to a Legacy Adapter

**Files:**
- Modify: `moe_infinity/spec_decode/dflash.py`
- Update: `tests/python/dflash/test_bare_hf_backend.py`
- Test: `tests/python/dflash/test_loop_retirement.py`

- [ ] **Step 1: Construct request rows from legacy arguments**

Keep the existing method signature. Validate shape/budgets/mask without model execution, strip left padding for each `RequestSpec.prompt_token_ids`, preserve each `SamplingContext`, budget, and stop set, and retain legacy shared-stop resolution.

- [ ] **Step 2: Invoke the driver and adapt the complete result**

Construct `BatchedBareHFBackend(self)` and `SessionDriver([backend])`, call `run_physical_cohort`, copy backend cache/step diagnostics and driver row results/traces only after success, compute generated lengths, right-pad new tokens with the target pad id, and concatenate with the original padded prompt tensor.

- [ ] **Step 3: Verify adapter and structural tests pass**

Run: `pytest -q tests/python/dflash/test_loop_retirement.py tests/python/dflash/test_bare_hf_backend.py`

Expected: PASS.

### Task 4: Regression and Static Verification

**Files:**
- Verify: `moe_infinity/spec_decode/session_driver.py`
- Verify: `moe_infinity/spec_decode/dflash.py`
- Verify: `moe_infinity/spec_decode/__init__.py`
- Verify: `tests/python/dflash/test_loop_retirement.py`

- [ ] **Step 1: Run all DFlash tests**

Run: `pytest -q tests/python/dflash`

Expected: PASS, with only environment-declared skips.

- [ ] **Step 2: Run relevant public/serving regression tests**

Run: `pytest -q tests/python/dflash/test_public_api_compat.py tests/python/dflash/test_engine_wire.py tests/python/serving/test_dflash_stage4a.py tests/python/serving/test_dflash_stage4b.py`

Expected: PASS, with only environment-declared skips.

- [ ] **Step 3: Run static diagnostics**

Run LSP diagnostics on every changed Python file, then run the repository's configured basedpyright command over the changed implementation and test files.

Expected: no errors.

- [ ] **Step 4: Confirm scope and worktree state**

Inspect the diff and status without staging or committing. Confirm no serving/MLA files were changed by Task11 and stop after the first complete successful verification.
