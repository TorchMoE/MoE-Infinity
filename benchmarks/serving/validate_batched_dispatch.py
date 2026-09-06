from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, cast

import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TARGET_FILES = {
    "expert_executor": os.path.join(
        PROJECT_ROOT, "moe_infinity", "distributed", "expert_executor.py"
    ),
    "mixtral": os.path.join(
        PROJECT_ROOT, "moe_infinity", "models", "mixtral.py"
    ),
    "deepseek_v2_wrapper": os.path.join(
        PROJECT_ROOT, "moe_infinity", "models", "deepseek_v2_wrapper.py"
    ),
    "dispatcher_h": os.path.join(
        PROJECT_ROOT, "core", "parallel", "expert_dispatcher.h"
    ),
    "dispatcher_cpp": os.path.join(
        PROJECT_ROOT, "core", "parallel", "expert_dispatcher.cpp"
    ),
}


class Blocked(RuntimeError):
    pass


@dataclass
class Result:
    name: str
    status: str
    detail: str


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _find_line(text: str, needle: str) -> int:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return -1


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _extract_transformers_hf_constraint() -> str:
    try:
        requires = importlib.metadata.requires("transformers") or []
    except importlib.metadata.PackageNotFoundError:
        return "unknown (transformers not installed)"

    for req in requires:
        if req.lower().startswith("huggingface-hub"):
            return req
    return "unknown (no huggingface-hub requirement found)"


def check_store_import() -> str:
    try:
        _ = importlib.import_module("moe_infinity._store")
        return "moe_infinity._store import succeeded"
    except Exception as exc:
        hf_hub_ver = _pkg_version("huggingface-hub")
        transformers_ver = _pkg_version("transformers")
        hf_req = _extract_transformers_hf_constraint()

        conflict_hint = ""
        if hf_hub_ver != "not-installed":
            major_match = re.match(r"(\d+)\.", hf_hub_ver)
            if major_match and int(major_match.group(1)) >= 1:
                conflict_hint = (
                    " Potential conflict: huggingface-hub major version is >=1; "
                    "many transformers releases require huggingface-hub<1.0."
                )

        message = (
            "moe_infinity._store import failed gracefully. "
            f"Reason: {type(exc).__name__}: {exc}. "
            f"Detected versions: transformers={transformers_ver}, "
            f"huggingface-hub={hf_hub_ver}. "
            f"transformers requirement: {hf_req}."
            f"{conflict_hint}"
        )
        raise Blocked(message)


def analyze_dispatch_interface_payload() -> dict[str, object]:
    expert_executor_text = _read_text(TARGET_FILES["expert_executor"])
    mixtral_text = _read_text(TARGET_FILES["mixtral"])
    deepseek_text = _read_text(TARGET_FILES["deepseek_v2_wrapper"])
    dispatcher_h_text = _read_text(TARGET_FILES["dispatcher_h"])
    dispatcher_cpp_text = _read_text(TARGET_FILES["dispatcher_cpp"])

    markers = {
        "dispatch_signature_line": _find_line(
            expert_executor_text,
            "def dispatch_local(",
        ),
        "dispatch_set_inputs_line": _find_line(
            expert_executor_text,
            "self.expert_dispatcher.set_inputs(",
        ),
        "mixtral_flatten_line": _find_line(
            mixtral_text,
            "hidden_states = hidden_states.view(-1, hidden_dim)",
        ),
        "mixtral_dispatch_line": _find_line(
            mixtral_text,
            "self.expert_executor.dispatch_local(",
        ),
        "deepseek_flatten_line": _find_line(
            deepseek_text,
            "hidden_states = hidden_states.view(-1, hidden_states.shape[-1])",
        ),
        "deepseek_dispatch_line": _find_line(
            deepseek_text,
            "self.expert_executor.dispatch_local(",
        ),
        "cpp_set_inputs_line": _find_line(
            dispatcher_h_text,
            "void SetInputs(const torch::Tensor& hidden_states",
        ),
        "cpp_wait_line": _find_line(
            dispatcher_h_text,
            "torch::Tensor WaitHiddenStates();",
        ),
        "cpp_batch_size_line": _find_line(
            dispatcher_cpp_text,
            "int64_t batch_size = hidden_states_.size(0);",
        ),
        "cpp_token_mask_line": _find_line(
            dispatcher_cpp_text,
            'auto token_mask = router_mask_.index({"...", expert_idx});',
        ),
        "cpp_index_select_line": _find_line(
            dispatcher_cpp_text,
            "hidden_states_.index({token_mask}).to(device)",
        ),
        "cpp_index_add_line": _find_line(
            dispatcher_cpp_text,
            "final_hidden_states_.index_add_(0, token_indices, weighted_output);",
        ),
        "cpp_zeros_like_line": _find_line(
            dispatcher_cpp_text,
            "final_hidden_states_ = torch::zeros_like(hidden_states, options);",
        ),
    }

    native_branch = ""
    native_start = expert_executor_text.find("if use_native_routing:")
    if native_start >= 0:
        native_end = expert_executor_text.find("else:", native_start)
        native_branch = expert_executor_text[native_start:native_end]

    native_markers = {
        "native_branch_has_blocking_host_extract": any(
            needle in native_branch
            for needle in (".cpu(", ".numpy(", ".item(", ".tolist(")
        ),
        "native_dispatch_binding_line": _find_line(
            expert_executor_text,
            "self.expert_dispatcher.dispatch_experts(layer_id)",
        ),
        "native_take_active_line": _find_line(
            expert_executor_text,
            "self.expert_dispatcher.take_last_active_experts()",
        ),
    }
    markers.update(native_markers)

    missing = [
        k
        for k, v in markers.items()
        if isinstance(v, int) and not isinstance(v, bool) and v < 0
    ]
    if missing:
        raise RuntimeError(f"Missing expected source markers: {missing}")
    if native_markers["native_branch_has_blocking_host_extract"]:
        raise RuntimeError(
            "native gpu routing branch performs a blocking host extract"
        )
    for line_key in ("native_dispatch_binding_line", "native_take_active_line"):
        if native_markers[line_key] < 0:
            raise RuntimeError(f"Missing expected native marker: {line_key}")

    return {
        "supports_token_batched_input": True,
        "inferred_interface": {
            "hidden_states": "[total_tokens, hidden_dim]",
            "router_mask": "[total_tokens, num_experts] bool",
            "router_weights": "[total_tokens, num_experts] float",
            "return": "[total_tokens, hidden_dim]",
        },
        "markers": markers,
    }


def synthetic_batched_example_payload() -> dict[str, object]:
    seq_lens = [5, 2, 7]
    hidden_dim = 32
    num_experts = 8
    top_k = 2

    total_tokens = int(sum(seq_lens))
    hidden_states = torch.randn(total_tokens, hidden_dim, dtype=torch.float16)

    selected_experts = torch.randint(
        low=0,
        high=num_experts,
        size=(total_tokens, top_k),
        dtype=torch.int64,
    )
    gate_scores = torch.rand(total_tokens, top_k, dtype=torch.float32)
    gate_scores = gate_scores / gate_scores.sum(dim=-1, keepdim=True)

    router_mask = torch.zeros(total_tokens, num_experts, dtype=torch.bool)
    _ = router_mask.scatter_(1, selected_experts, True)

    router_weights = torch.zeros(total_tokens, num_experts, dtype=torch.float32)
    _ = router_weights.scatter_add_(1, selected_experts, gate_scores)

    expert_count = torch.sum(router_mask, dim=0)
    active_expert_indices = torch.nonzero(
        expert_count > 0, as_tuple=False
    ).reshape(-1)
    expert_list = [int(idx.item()) for idx in active_expert_indices]

    token_to_seq: list[int] = []
    for seq_id, seq_len in enumerate(seq_lens):
        token_to_seq.extend([seq_id] * seq_len)

    per_expert: dict[int, dict[str, object]] = {}
    for expert_id in expert_list:
        token_mask = router_mask[:, expert_id]
        token_indices = torch.nonzero(token_mask, as_tuple=False).reshape(-1)
        token_index_list = [int(idx.item()) for idx in token_indices]
        source_sequences = sorted(
            {token_to_seq[idx] for idx in token_index_list}
        )
        per_expert[expert_id] = {
            "token_count": len(token_index_list),
            "source_sequences": source_sequences,
        }

    return {
        "sequence_lengths": seq_lens,
        "total_tokens": total_tokens,
        "hidden_states_shape": list(hidden_states.shape),
        "router_mask_shape": list(router_mask.shape),
        "router_weights_shape": list(router_weights.shape),
        "active_experts": expert_list,
        "expected_wait_cnt": len(expert_list),
        "per_expert_routing": per_expert,
    }


def recommendation(analysis: dict[str, object]) -> str:
    supports = cast(bool, analysis["supports_token_batched_input"])
    if not supports:
        return (
            "RECOMMENDATION: Option B (extend C++ interface). "
            "Current dispatcher markers do not confirm token-batched behavior."
        )

    return (
        "RECOMMENDATION: Option A (Python-layer integration; no C++ API change). "
        "Current C++ dispatcher already consumes token-batched tensors "
        "[T,H]/[T,E] and routes by token_mask.\n"
        "Required Python changes for continuous batching:\n"
        "  1) Pack all active sequences' hidden states into one contiguous [T,H] tensor per MoE layer call.\n"
        "  2) Build aligned router_mask/router_weights as [T,E] over the same packed token order.\n"
        "  3) Track sequence offsets so wait_dispatch_local() output [T,H] can be scattered back to per-request buffers.\n"
        "  4) Keep one dispatch_local() call per layer per scheduler step (do not force per-sequence calls).\n"
        "  5) Preserve decode fast-path semantics: T==1 remains single-token path; T>1 uses token_mask path."
    )


def capture_result(name: str, fn: Callable[[], str]) -> Result:
    try:
        detail = fn()
        return Result(name=name, status="PASS", detail=detail)
    except Blocked as exc:
        return Result(name=name, status="BLOCKED", detail=str(exc))
    except Exception as exc:
        return Result(
            name=name,
            status="FAIL",
            detail=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    print("=== Expert Dispatcher Batched Dispatch Feasibility ===")
    print(f"Project root: {PROJECT_ROOT}")
    environment_line = (
        f"Environment: torch={getattr(torch, '__version__', 'unknown')}, "
        + f"cuda_available={torch.cuda.is_available()}, "
        + f"cuda_device_count={torch.cuda.device_count()}"
    )
    print(environment_line)
    print()

    results: list[Result] = []
    analysis_payload: dict[str, object] | None = None
    synthetic_payload: dict[str, object] | None = None

    results.append(
        capture_result("Import moe_infinity._store", check_store_import)
    )

    try:
        analysis_payload = analyze_dispatch_interface_payload()
        results.append(
            Result(
                name="Static interface analysis (Python/C++)",
                status="PASS",
                detail=json.dumps(analysis_payload, ensure_ascii=False),
            )
        )
    except Exception as exc:
        results.append(
            Result(
                name="Static interface analysis (Python/C++)",
                status="FAIL",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    try:
        synthetic_payload = synthetic_batched_example_payload()
        results.append(
            Result(
                name="Synthetic 3-sequence concatenated routing example",
                status="PASS",
                detail=json.dumps(synthetic_payload, ensure_ascii=False),
            )
        )
    except Exception as exc:
        results.append(
            Result(
                name="Synthetic 3-sequence concatenated routing example",
                status="FAIL",
                detail=f"{type(exc).__name__}: {exc}",
            )
        )

    for result in results:
        print(f"[{result.status}] {result.name}")
        print(result.detail)
        print()

    if analysis_payload is not None and synthetic_payload is not None:
        print("=== Dispatch Interface Documentation ===")
        iface = cast(dict[str, str], analysis_payload["inferred_interface"])
        print(
            "dispatch_local(layer_id, hidden_states, router_mask, router_weights)"
        )
        print(f"  hidden_states : {iface['hidden_states']}")
        print(f"  router_mask   : {iface['router_mask']}")
        print(f"  router_weights: {iface['router_weights']}")
        print(f"  return        : {iface['return']}")
        print()
        print("=== Design Recommendation ===")
        print(recommendation(analysis_payload))
    else:
        print("=== Design Recommendation ===")
        print(
            "Insufficient analysis data for recommendation due to earlier failure(s)."
        )

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    blocked = sum(1 for r in results if r.status == "BLOCKED")
    print()
    print(f"Summary: PASS={passed}, FAIL={failed}, BLOCKED={blocked}")

    if failed > 0:
        print("Overall verdict: NO-GO (analysis failure)")
    elif blocked > 0:
        print(
            "Overall verdict: GO (design feasible), RUNTIME BLOCKED for _store import"
        )
    else:
        print("Overall verdict: GO")


if __name__ == "__main__":
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    main()
