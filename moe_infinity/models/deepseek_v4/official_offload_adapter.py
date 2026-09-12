# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from glob import glob
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open

from moe_infinity.models.deepseek_v4.vision_exp import (
    should_skip_resident_load,
)


class OfficialExpertHostStore:
    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        max_resident_experts: int = 16,
        shard_file: str = None,
    ):
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)
        self.max_resident_experts = max_resident_experts
        self.shard_file = shard_file

        self._shard_for: Dict[str, str] = {}
        self._open_files: Dict[str, object] = {}
        self._host: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self._gpu: "OrderedDict[Tuple[int, int], Dict[str, torch.Tensor]]" = (
            OrderedDict()
        )
        self._copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self._build_key_index()

    def _build_key_index(self) -> None:
        shards = (
            [self.shard_file]
            if self.shard_file
            else sorted(glob(os.path.join(self.ckpt_path, "*.safetensors")))
        )
        for shard in shards:
            with open(shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            for key in header:
                self._shard_for[key] = shard

    def _opener(self, shard: str):
        if shard not in self._open_files:
            self._open_files[shard] = safe_open(
                shard, framework="pt", device="cpu"
            )
        return self._open_files[shard]

    def _load_host(
        self, layer_id: int, expert_id: int
    ) -> Dict[str, torch.Tensor]:
        key = (layer_id, expert_id)
        cached = self._host.get(key)
        if cached is not None:
            return cached
        prefix = f"layers.{layer_id}.ffn.experts.{expert_id}"
        tensors: Dict[str, torch.Tensor] = {}
        for proj in ("w1", "w2", "w3"):
            for part in ("weight", "scale"):
                k = f"{prefix}.{proj}.{part}"
                t = self._opener(self._shard_for[k]).get_tensor(k)
                try:
                    t = t.pin_memory()
                except Exception:
                    pass
                tensors[f"{proj}.{part}"] = t
        self._host[key] = tensors
        return tensors

    def resident_experts(self) -> List[Tuple[int, int]]:
        return list(self._gpu.keys())

    def is_resident(self, layer_id: int, expert_id: int) -> bool:
        return (layer_id, expert_id) in self._gpu

    def _to_device(
        self, host: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        gpu = {}
        for name, t in host.items():
            if t.dtype == torch.float4_e2m1fn_x2:
                g = torch.empty_like(t, device=self.device)
                g.view(torch.uint8).copy_(
                    t.view(torch.uint8), non_blocking=True
                )
            else:
                g = t.to(self.device, non_blocking=True)
            gpu[name] = g
        return gpu

    def prefetch(self, layer_id: int, expert_ids) -> None:
        if self._copy_stream is None:
            return
        with torch.cuda.stream(self._copy_stream):
            for e in expert_ids:
                key = (layer_id, e)
                if key in self._gpu:
                    self._gpu.move_to_end(key)
                    continue
                gpu = self._to_device(self._load_host(layer_id, e))
                self._gpu[key] = gpu
        self._evict()

    def _evict(self) -> None:
        while len(self._gpu) > self.max_resident_experts:
            self._gpu.popitem(last=False)

    def get(self, layer_id: int, expert_id: int) -> Dict[str, torch.Tensor]:
        key = (layer_id, expert_id)
        if self._copy_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(
                self._copy_stream
            )
        if key in self._gpu:
            self._gpu.move_to_end(key)
            return self._gpu[key]
        gpu = self._to_device(self._load_host(layer_id, expert_id))
        self._gpu[key] = gpu
        self._evict()
        return gpu


_V4_FP4 = None


def _load_native_fp4():
    global _V4_FP4
    if _V4_FP4 is None:
        from moe_infinity import _v4_fp4

        _V4_FP4 = _v4_fp4
    return _V4_FP4


def _is_blackwell(device) -> bool:
    if not torch.cuda.is_available():
        return False
    index = torch.device(device).index if device is not None else None
    major, _ = torch.cuda.get_device_capability(index)
    return major >= 12


def native_fp4_available() -> bool:
    try:
        _load_native_fp4()
        return True
    except Exception:
        return False


def resolve_use_native(use_native, device) -> bool:
    if use_native is not None:
        return bool(use_native)
    if os.environ.get("MOE_DSV4_FORCE_NATIVE") == "0":
        return False
    return _is_blackwell(device) and native_fp4_available()


def _expert_forward(x, bundle, swiglu_limit, official_module):
    linear = official_module.linear

    def proj(name):
        w = bundle[f"{name}.weight"]
        w.scale = bundle[f"{name}.scale"]
        return w

    dtype = x.dtype
    gate = linear(x, proj("w1")).float()
    up = linear(x, proj("w3")).float()
    if swiglu_limit and swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    h = F.silu(gate) * up
    return linear(h.to(dtype), proj("w2"))


def _expert_forward_native(x, bundle, swiglu_limit):
    ext = _load_native_fp4()
    return ext.v4_expert_forward(
        x,
        bundle["w1.weight"],
        bundle["w1.scale"],
        bundle["w2.weight"],
        bundle["w2.scale"],
        bundle["w3.weight"],
        bundle["w3.scale"],
        float(swiglu_limit),
    )


def patch_moe_with_offload(model, store, official_module, use_native=None):
    use_native = resolve_use_native(use_native, store.device)
    if use_native:
        _load_native_fp4()

    def make_forward(moe, layer_id):
        gate = moe.gate
        shared = moe.shared_experts
        dim = moe.dim
        n_routed = moe.n_routed_experts
        swiglu_limit = getattr(shared, "swiglu_limit", 0.0)

        start_idx = moe.experts_start_idx
        end_idx = moe.experts_end_idx

        def forward(x, input_ids):
            shape = x.size()
            x = x.view(-1, dim)
            weights, indices = gate(x, input_ids.flatten())
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(
                indices.flatten(), minlength=n_routed
            ).tolist()
            active = [e for e in range(start_idx, end_idx) if counts[e] > 0]
            store.prefetch(layer_id, active)
            for e in active:
                idx, top = torch.where(indices == e)
                bundle = store.get(layer_id, e)
                if use_native:
                    out = _expert_forward_native(x[idx], bundle, swiglu_limit)
                else:
                    out = _expert_forward(
                        x[idx], bundle, swiglu_limit, official_module
                    )
                y[idx] += out.float() * weights[idx, top, None].float()
            if official_module.world_size > 1:
                official_module.dist.all_reduce(y)
            y += shared(x)
            return y.type_as(x).view(shape)

        return forward

    expert_layer = 0
    for block in model.layers:
        if hasattr(block, "ffn") and hasattr(block.ffn, "gate"):
            moe = block.ffn
            moe.forward = make_forward(moe, expert_layer)
            for e in range(len(moe.experts)):
                moe.experts[e] = None
            expert_layer += 1
    return model


def load_offloaded_v4_flash(
    official_module,
    ckpt_path: str,
    config_path: str,
    device: torch.device,
    shard_file: str,
    max_resident_experts: int = 8,
    use_native: bool = None,
    text_only: bool = True,
):
    import json as _json

    M = official_module
    with open(config_path) as f:
        margs = M.ModelArgs(**_json.load(f))
    margs.max_batch_size = 1

    orig_moe_init = M.MoE.__init__

    def _null_routed_moe_init(self, layer_id, args):
        import torch.nn as nn

        orig_expert = M.Expert

        class _NullExpert(nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, x, weights=None):
                return x

        M.Expert = _NullExpert
        try:
            orig_moe_init(self, layer_id, args)
        finally:
            M.Expert = orig_expert
        for i in range(len(self.experts)):
            self.experts[i] = None
        self.shared_experts = orig_expert(
            args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
        )

    M.MoE.__init__ = _null_routed_moe_init
    try:
        with torch.device(device):
            model = M.Transformer(margs)
    finally:
        M.MoE.__init__ = orig_moe_init
    torch.set_grad_enabled(False)

    store = OfficialExpertHostStore(
        ckpt_path,
        device,
        max_resident_experts=max_resident_experts,
        shard_file=shard_file,
    )
    patch_moe_with_offload(model, store, M, use_native=use_native)
    torch.cuda.empty_cache()

    _load_non_expert_weights(
        model, shard_file, device, config=margs, text_only=text_only
    )
    return model, store


def _load_non_expert_weights(
    model, shard_file, device, config=None, text_only=True
):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with safe_open(shard_file, framework="pt", device="cpu") as f:
        for k in f.keys():
            if config is not None:
                if should_skip_resident_load(k, config, text_only=text_only):
                    continue
            elif ".ffn.experts." in k:
                continue
            if k.endswith(".scale"):
                wp = params.get(k[:-6] + ".weight")
                if wp is not None and getattr(wp, "scale", None) is not None:
                    t = f.get_tensor(k).to(device)
                    if wp.scale.shape == t.shape:
                        wp.scale.copy_(t)
                continue
            target = params.get(k)
            if target is None:
                target = buffers.get(k)
            if target is None:
                continue
            t = f.get_tensor(k).to(device)
            if target.shape == t.shape:
                target.copy_(t)
    return model
