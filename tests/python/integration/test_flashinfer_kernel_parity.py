"""FlashInfer full-prefill vs append-path numerical parity on identical paged KV.

Evidence base for the prefix-reuse warm-path verdict (PR #190): the two plan
geometries (cold: q=kv, warm: q=suffix with reused context) execute different
kernel schedules, so bitwise equality between them is NOT an invariant. The
enforceable invariants are: (a) each path stays within ULP-scale error of an
fp32 reference, (b) each path is deterministic, and (c) slots past kv_len in
the last partial page never influence the output.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

flashinfer = pytest.importorskip("flashinfer")

H_QO, H_KV, D, PAGE = 32, 4, 128, 16
DTYPE = torch.bfloat16
TOTAL, CTX = 72, 64
NUM_PAGES = (TOTAL + PAGE - 1) // PAGE
FP32_ATOL = 8e-3


@pytest.fixture(scope="module")
def device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return "cuda:0"


@pytest.fixture(scope="module")
def wrapper(device):
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")


def plan_and_run(wrapper, device, query, kv_data, qo_len, kv_len):
    qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, NUM_PAGES], dtype=torch.int32, device=device)
    kv_indices = torch.arange(NUM_PAGES, dtype=torch.int32, device=device)
    rem = kv_len % PAGE
    kv_last_page_len = torch.tensor(
        [PAGE if rem == 0 else rem], dtype=torch.int32, device=device
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        H_QO,
        H_KV,
        D,
        PAGE,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    return wrapper.run(query, kv_data)


def fp32_reference(query, kv_data, kv_len, qo_len, device):
    flat_k = kv_data[:, 0].reshape(-1, H_KV, D)[:kv_len].float()
    flat_v = kv_data[:, 1].reshape(-1, H_KV, D)[:kv_len].float()
    group = H_QO // H_KV
    k = flat_k.repeat_interleave(group, dim=1)
    v = flat_v.repeat_interleave(group, dim=1)
    qf = query.float()
    scale = D**-0.5
    out = torch.empty(qo_len, H_QO, D, device=device, dtype=torch.float32)
    offset = kv_len - qo_len
    for i in range(qo_len):
        allowed = offset + i + 1
        scores = torch.einsum("hd,khd->hk", qf[i], k[:allowed]) * scale
        probs = torch.softmax(scores, dim=-1)
        out[i] = torch.einsum("hk,khd->hd", probs, v[:allowed])
    return out


@pytest.fixture(scope="module")
def tensors(device):
    generator = torch.Generator(device=device).manual_seed(0)
    kv_data = torch.randn(
        NUM_PAGES,
        2,
        PAGE,
        H_KV,
        D,
        device=device,
        dtype=DTYPE,
        generator=generator,
    )
    query = torch.randn(
        TOTAL, H_QO, D, device=device, dtype=DTYPE, generator=generator
    )
    return kv_data, query


def test_both_kernel_paths_match_fp32_reference(wrapper, device, tensors):
    kv_data, query = tensors
    out_cold = plan_and_run(wrapper, device, query, kv_data, TOTAL, TOTAL)
    suffix_query = query[CTX:].contiguous()
    out_warm = plan_and_run(
        wrapper, device, suffix_query, kv_data, TOTAL - CTX, TOTAL
    )
    reference = fp32_reference(
        suffix_query, kv_data, TOTAL, TOTAL - CTX, device
    )
    cold_err = (out_cold[CTX:].float() - reference).abs().max().item()
    warm_err = (out_warm.float() - reference).abs().max().item()
    assert cold_err < FP32_ATOL
    assert warm_err < FP32_ATOL


def test_each_kernel_path_is_deterministic(wrapper, device, tensors):
    kv_data, query = tensors
    suffix_query = query[CTX:].contiguous()
    cold_a = plan_and_run(wrapper, device, query, kv_data, TOTAL, TOTAL)
    cold_b = plan_and_run(wrapper, device, query, kv_data, TOTAL, TOTAL)
    warm_a = plan_and_run(
        wrapper, device, suffix_query, kv_data, TOTAL - CTX, TOTAL
    )
    warm_b = plan_and_run(
        wrapper, device, suffix_query, kv_data, TOTAL - CTX, TOTAL
    )
    assert torch.equal(cold_a, cold_b)
    assert torch.equal(warm_a, warm_b)


def test_partial_page_tail_never_leaks(wrapper, device, tensors):
    kv_data, query = tensors
    suffix_query = query[CTX:].contiguous()
    poisoned = kv_data.clone()
    tail = TOTAL % PAGE
    poisoned[-1, :, tail:] = 1e4
    for qo_len, run_query in ((TOTAL, query), (TOTAL - CTX, suffix_query)):
        clean_out = plan_and_run(
            wrapper, device, run_query, kv_data, qo_len, TOTAL
        )
        poisoned_out = plan_and_run(
            wrapper, device, run_query, poisoned, qo_len, TOTAL
        )
        assert torch.equal(clean_out, poisoned_out)
