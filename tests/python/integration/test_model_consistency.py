import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
from transformers import AutoTokenizer

from moe_infinity.utils import ArcherConfig


def _get_attn_impl(model_name: str) -> str:
    if "deepseek" in model_name.lower():
        return "eager"
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "eager"


_HF_WORKER = r"""
import pickle, sys, torch
from transformers import AutoModelForCausalLM, AutoConfig
from moe_infinity.utils.hf_config import ensure_config_compat

model_name, attn_impl, input_ids_path, out_path = sys.argv[1:]
input_ids = torch.load(input_ids_path, weights_only=True)

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
ensure_config_compat(config)

if "deepseek" in model_name.lower():
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2ForCausalLM,
    )
    model = DeepseekV2ForCausalLM.from_pretrained(
        model_name, config=config, device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl, trust_remote_code=True,
    )

pad_token_id = int(open(out_path + ".pad").read())
model.eval()
with torch.no_grad():
    logits = model(input_ids).logits.detach().cpu()
    gen_ids = model.generate(
        input_ids, max_new_tokens=32, do_sample=False,
        pad_token_id=pad_token_id,
    ).detach().cpu()

with open(out_path, "wb") as f:
    pickle.dump({"logits": logits, "gen_ids": gen_ids}, f)
"""

_MOE_WORKER = r"""
import pickle, sys, torch
from moe_infinity import MoE

model_name, input_ids_path, offload_path, out_path = sys.argv[1:]
input_ids = torch.load(input_ids_path, weights_only=True)
pad_token_id = int(open(out_path + ".pad").read())

config = {"offload_path": offload_path, "device_memory_ratio": 0.75}
default_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
try:
    model = MoE(model_name, config)
finally:
    torch.set_default_dtype(default_dtype)

with torch.no_grad():
    _ = model(input_ids)

with torch.no_grad():
    logits = model(input_ids).logits.detach().cpu()
    gen_ids = model.generate(
        input_ids, max_new_tokens=32, do_sample=False,
        pad_token_id=pad_token_id,
    ).detach().cpu()

with open(out_path, "wb") as f:
    pickle.dump({"logits": logits, "gen_ids": gen_ids}, f)
"""

_PHASE_POLICY_WORKER = r"""
import pickle, sys, torch
from moe_infinity import MoE

model_name, input_ids_path, offload_path, enabled_flag, out_path = sys.argv[1:]
input_ids = torch.load(input_ids_path, weights_only=True)
pad_token_id = int(open(out_path + ".pad").read())

# Off/on runs share the same greedy seed so any output divergence is
# attributable to the phase policy, not RNG state.
torch.manual_seed(0)

config = {
    "offload_path": offload_path,
    "use_native_engine": True,
    "device_memory_ratio": 0.75,
    "kv_cache_memory_ratio": 0.15,
    "phase_specific_expert_policy": enabled_flag == "1",
}
default_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
try:
    model = MoE(model_name, config)
finally:
    torch.set_default_dtype(default_dtype)

with torch.no_grad():
    gen_ids = model.generate(
        input_ids, max_new_tokens=16, do_sample=False,
        pad_token_id=pad_token_id,
    ).detach().cpu()

with open(out_path, "wb") as f:
    pickle.dump({"gen_ids": gen_ids}, f)
"""


def _run_worker(script: str, args: list, label: str) -> Dict[str, torch.Tensor]:
    import time

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    out_path = args[-1]
    worker_env = os.environ.copy()
    repository_root = str(Path(__file__).resolve().parents[3])
    inherited_pythonpath = worker_env.get("PYTHONPATH")
    worker_env["PYTHONPATH"] = (
        repository_root
        if not inherited_pythonpath
        else os.pathsep.join((repository_root, inherited_pythonpath))
    )
    try:
        proc = subprocess.Popen(
            [sys.executable, script_path] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=worker_env,
        )
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                time.sleep(2)
                break
            ret = proc.poll()
            if ret is not None and ret != 0:
                _, stderr = proc.communicate(timeout=10)
                pytest.fail(
                    f"[{label}] Worker crashed (exit {ret}):\n"
                    f"STDERR:\n{stderr.decode(errors='replace')[-2000:]}"
                )
            time.sleep(3)
        else:
            proc.kill()
            pytest.fail(f"[{label}] Worker timed out after 3600s")
        proc.kill()
        with open(out_path, "rb") as f:
            return pickle.load(f)
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


@pytest.fixture(
    scope="session",
    params=[
        "deepseek-ai/DeepSeek-V2-Lite",
        "Qwen/Qwen3-30B-A3B",
    ],
)
def consistency_outputs(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — consistency test requires GPU")

    model_name = request.param
    prompt = os.environ.get(
        "MOE_CONSISTENCY_PROMPT", "The capital of France is"
    )
    attn_impl = _get_attn_impl(model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError(f"[{model_name}] tokenizer.eos_token_id is required")

    work_dir = tmp_path_factory.mktemp("consistency")
    input_ids_path = str(work_dir / "input_ids.pt")
    torch.save(input_ids, input_ids_path)

    hf_out_path = str(work_dir / "hf_out.pkl")
    moe_out_path = str(work_dir / "moe_out.pkl")
    offload_path = str(work_dir / model_name.replace("/", "_"))

    with open(hf_out_path + ".pad", "w") as f:
        f.write(str(pad_token_id))
    with open(moe_out_path + ".pad", "w") as f:
        f.write(str(pad_token_id))

    hf_data = _run_worker(
        _HF_WORKER,
        [model_name, attn_impl, input_ids_path, hf_out_path],
        f"HF:{model_name}",
    )

    moe_data = _run_worker(
        _MOE_WORKER,
        [model_name, input_ids_path, offload_path, moe_out_path],
        f"MoE:{model_name}",
    )

    return {
        "model_name": model_name,
        "tokenizer": tokenizer,
        "hf_logits": hf_data["logits"],
        "hf_gen_ids": hf_data["gen_ids"],
        "moe_logits": moe_data["logits"],
        "moe_gen_ids": moe_data["gen_ids"],
    }


@pytest.mark.cuda
def test_forward_consistency(consistency_outputs: Dict[str, Any]) -> None:
    model_name = consistency_outputs["model_name"]
    hf_logits = consistency_outputs["hf_logits"]
    moe_logits = consistency_outputs["moe_logits"]

    abs_diff = (moe_logits - hf_logits).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    hf_top = hf_logits.argmax(dim=-1)
    moe_top = moe_logits.argmax(dim=-1)
    print(
        f"\n[{model_name}] Forward — max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    )
    print(f"[{model_name}] Argmax match: {torch.equal(hf_top, moe_top)}")
    print(
        f"[{model_name}] Logits shape: hf={hf_logits.shape}, moe={moe_logits.shape}"
    )

    assert (
        hf_logits.shape == moe_logits.shape
    ), f"Shape mismatch: {hf_logits.shape} vs {moe_logits.shape}"
    assert torch.equal(hf_top, moe_top), (
        f"[{model_name}] Argmax mismatch at positions: "
        f"{(hf_top != moe_top).nonzero(as_tuple=True)}"
    )
    torch.testing.assert_close(
        moe_logits,
        hf_logits,
        rtol=1e-3,
        atol=1e-3,
        msg=lambda msg: (
            f"[{model_name}] Logits not close. Max diff: {max_diff:.6f}\n{msg}"
        ),
    )


@pytest.mark.cuda
def test_generation_consistency(consistency_outputs: Dict[str, Any]) -> None:
    model_name = consistency_outputs["model_name"]
    tokenizer = consistency_outputs["tokenizer"]
    hf_gen_ids = consistency_outputs["hf_gen_ids"]
    moe_gen_ids = consistency_outputs["moe_gen_ids"]

    hf_text = tokenizer.decode(hf_gen_ids[0], skip_special_tokens=True)
    moe_text = tokenizer.decode(moe_gen_ids[0], skip_special_tokens=True)
    print(f"\n[{model_name}] HF  output: {hf_text!r}")
    print(f"[{model_name}] MoE output: {moe_text!r}")

    assert torch.equal(hf_gen_ids, moe_gen_ids), (
        f"[{model_name}] Generation mismatch!\n"
        f"HF  tokens: {hf_gen_ids[0].tolist()}\n"
        f"MoE tokens: {moe_gen_ids[0].tolist()}\n"
        f"First diff: {(hf_gen_ids != moe_gen_ids).nonzero(as_tuple=True)}"
    )


@pytest.mark.cuda
def test_model_consistency_with_explicit_phase_policy(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA not available — phase-policy consistency requires GPU"
        )

    model_name = os.environ.get(
        "MOE_PHASE_POLICY_MODEL", "deepseek-ai/DeepSeek-V2-Lite"
    )
    prompt = os.environ.get(
        "MOE_CONSISTENCY_PROMPT", "The capital of France is"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pytest.skip(f"[{model_name}] tokenizer.eos_token_id is required")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    work_dir = tmp_path_factory.mktemp("phase_policy_consistency")
    input_ids_path = str(work_dir / "input_ids.pt")
    torch.save(input_ids, input_ids_path)

    outputs = []
    for enabled in (False, True):
        offload_path = str(
            work_dir / f"{model_name.replace('/', '_')}_{int(enabled)}"
        )
        out_path = str(work_dir / f"out_{int(enabled)}.pkl")
        with open(out_path + ".pad", "w") as pad_file:
            pad_file.write(str(pad_token_id))

        _ = ArcherConfig.load_from_json(
            {
                "offload_path": offload_path,
                "use_native_engine": True,
                "device_memory_ratio": 0.75,
                "kv_cache_memory_ratio": 0.15,
                "phase_specific_expert_policy": enabled,
            }
        )

        data = _run_worker(
            _PHASE_POLICY_WORKER,
            [
                model_name,
                input_ids_path,
                offload_path,
                "1" if enabled else "0",
                out_path,
            ],
            f"MoE-phase-policy[{'on' if enabled else 'off'}]:{model_name}",
        )
        outputs.append(data["gen_ids"])

    assert torch.equal(outputs[0], outputs[1]), (
        "Phase policy off/on produced different tokens:\n"
        f"off: {outputs[0][0].tolist()}\n"
        f"on:  {outputs[1][0].tolist()}"
    )
