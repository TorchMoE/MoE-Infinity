from contextlib import nullcontext

from transformers import PretrainedConfig

from moe_infinity.memory.expert_tracer import ExpertTracer
from moe_infinity.utils import parse_moe_param

try:
    import nvtx  # type: ignore[reportMissingTypeStubs]
except ImportError:
    nvtx = None

HAS_NVTX = nvtx is not None

try:
    from moe_infinity.profiling.io_profiler import (  # pyright: ignore[reportMissingImports]
        IOProfiler,
    )
except Exception:
    IOProfiler = None


class ExpertPredictor:
    def __init__(self, config: PretrainedConfig) -> None:
        self.num_layers, self.num_experts, self.num_encoder_layers = (
            parse_moe_param(config)
        )
        self.layer_decay_func = lambda x, l, L: -1 / (L + 1) * (x - l) + 1
        self.tracer = None

    def add_tracer(self, tracer: ExpertTracer):
        self.tracer = tracer

    def ranked_candidates(self, expert_matrix, layer_idx):
        row = expert_matrix[layer_idx]
        scored = [
            (int(expert_id), float(row[expert_id]))
            for expert_id in range(self.num_experts)
            if row[expert_id] > 0
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def predict(self, seq_id, expert_list, layer_idx):
        if self.tracer is None:
            raise RuntimeError("tracer must be added before predict")
        tracer = self.tracer

        profiler = IOProfiler.instance() if IOProfiler is not None else None
        nvtx_cm = nullcontext()
        if HAS_NVTX and nvtx is not None:
            nvtx_cm = nvtx.annotate("prefetch_predict", color="blue")
        profiler_cm = (
            profiler.time("prefetch_predict")
            if profiler is not None
            else nullcontext()
        )
        with profiler_cm:
            with nvtx_cm:
                tracer.update_entry(seq_id, expert_list, layer_idx)
                current_entry = tracer.get_entry(seq_id)

                expert_matrix = tracer.find_most_similar(
                    current_entry.matrix, layer_idx
                )
                expert_matrix[:layer_idx, :] = 0

                for l in range(layer_idx, self.num_layers):
                    expert_matrix[l] = (
                        expert_matrix[l] + 1e-8
                    ) * self.layer_decay_func(l, layer_idx, self.num_layers)

                return expert_matrix
