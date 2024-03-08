import sys
from moe_infinity.memory.expert_priority_score import *
from moe_infinity.memory.expert_entry import ExpertCacheEntry

import numpy as np
from collections import Counter

import logging


class ExpertCache:
    def __init__(
        self, num_gpu: int, gpu_size: int, memory_ratio: float, expert_size: int
    ) -> None:
        # get GPU memory size
        self.memory_ratio = memory_ratio
        self.num_gpu = num_gpu
        self.gpu_memory_size = gpu_size * memory_ratio
        self.cpu_memory_size = self.gpu_memory_size * 4

        self.gpu_expert_capacity = int(
            self.gpu_memory_size * self.num_gpu / expert_size
        )
        self.cpu_expert_capacity = int(
            self.cpu_memory_size * self.num_gpu / expert_size
        )

        self.gpu_expert_cache = {}
        self.cpu_expert_cache = {}

        self.experts_protected_ondemand = {}
        self.experts_protected_prefetch = {}
        self.experts_protected_by_layer = {}

        self.total_visit = 0
        self.total_gpu_cache_hit = 0
        self.total_cpu_cache_hit = 0

        self.total_encoder_visit = 0
        self.total_decoder_visit = 0
        self.total_encoder_gpu_cache_hit = 0
        self.total_decoder_gpu_cache_hit = 0

        self.expert_frequency = Counter()

        self.logical_timestamp = 0
        self.cache_policy = "priority"

        self.logger = logging.getLogger()
        self.handler = logging.StreamHandler(sys.stdout)
        self.formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def set_log_level(self, level):
        self.logger.setLevel(level)
        self.handler.setLevel(level)

        logging.basicConfig(stream=sys.stdout, level=level)

    def set_cache_policy(self, policy: str):
        self.cache_policy = policy

    def get_gpu_sorted_candidates(self):
        if type == "lfu":
            expert_freq = {
                (entry.expert_idx, entry.layer_idx): self.expert_frequency[key]
                for key, entry in self.gpu_expert_cache.items()
            }
            return lfu_score(expert_freq)
        elif type == "priority":
            self.score_func = priority_score
        else:
            assert False, "Should not reach here"

    def add_tracer(self, tracer):
        self.tracer = tracer

    def unprotect_expert(self, expert_idx: int, layer_idx: int):
        try:
            del self.experts_protected_ondemand[(expert_idx, layer_idx)]
        except KeyError:
            self.logger.error(
                f"Expert {expert_idx} at layer {layer_idx} is not protected on demand, hash {hash(cache_entry)}"
            )
            self.logger.error("=====================================")
            for entry in self.experts_protected_ondemand:
                self.logger.error(
                    f"Expert {entry.expert_idx} at layer {entry.layer_idx}, hash {hash(entry)}"
                )
            raise KeyError

    def gpu_evict(self, seq_id, layer_idx):
        if len(self.gpu_expert_cache) <= self.gpu_expert_capacity:
            return True

        # now try to evict one expert from GPU
        cache_entries = set(self.gpu_expert_cache.values())
        trace_entries = set(self.tracer.trace.values())
        decoder_entry = self.tracer.get_entry_decoder(seq_id)
        expert_freq = {
            (entry.expert_idx, entry.layer_idx): self.expert_frequency[key]
            for key, entry in self.gpu_expert_cache.items()
        }
        expert_visit_freq = {
            (entry.expert_idx, entry.layer_idx): entry.visit
            for key, entry in self.gpu_expert_cache.items()
        }

        if self.cache_policy == "lru":
            cache_candidates = lru_score(cache_entries)
        elif self.cache_policy == "lru_ds":
            cache_candidates = lru_score_with_layers(cache_entries, layer_idx)
        elif self.cache_policy == "lfu":
            cache_candidates = lfu_score(expert_visit_freq)
        elif self.cache_policy == "priority":
            cache_candidates = priority_score(
                expert_visit_freq,
                cache_entries,
                trace_entries,
                decoder_entry,
                layer_idx,
                self.tracer.num_layers,
            )
        else:
            assert False, "Should not reach here"

        cache_candidates.sort(key=lambda x: x.r)  # sort by r acending
        cache_candidates_in_gpu = [
            x
            for x in cache_candidates
            if (x.expert_idx, x.layer_idx) in self.gpu_expert_cache
        ]
        self.logger.debug(f"cache_candidates {cache_candidates_in_gpu[:5]}")

        if self.cache_policy == "priority":
            for candidate in cache_candidates:
                candidate_key = (candidate.expert_idx, candidate.layer_idx)
                if (
                    not candidate_key in self.experts_protected_ondemand
                    and not candidate_key in self.experts_protected_prefetch
                    and not candidate_key in self.experts_protected_by_layer
                ):
                    if candidate_key in self.gpu_expert_cache:
                        del self.gpu_expert_cache[candidate_key]
                        self.logger.debug(f"Evicting expert {candidate_key}")
                        return True

        # ForceEvict
        for candidate in cache_candidates:
            candidate_key = (candidate.expert_idx, candidate.layer_idx)
            if candidate_key in self.gpu_expert_cache:
                if self.cache_policy == "priority":
                    if not candidate_key in self.experts_protected_ondemand:
                        del self.gpu_expert_cache[candidate_key]
                        self.logger.debug(f"Force evicting expert {candidate_key}")
                        return True
                else:
                    del self.gpu_expert_cache[candidate_key]
                    self.logger.debug(f"Force evicting expert {candidate_key}")
                    return True

        return False

    def cpu_evict(self, seq_id, layer_idx):
        if len(self.cpu_expert_cache) <= self.cpu_expert_capacity:
            return True

        # now try to evict one expert from GPU
        cache_entries = self.cpu_expert_cache
        trace_entries = set(self.tracer.trace.values())
        decoder_entry = self.tracer.get_entry_decoder_matrix(seq_id)

        cache_candidates = priority_score(
            cache_entries,
            trace_entries,
            decoder_entry,
            layer_idx,
            self.tracer.num_layers,
        )
        cache_candidates.sort(key=lambda x: x.r)  # sort by r acending

        for candidate in cache_candidates:
            candidate_key = (candidate.expert_idx, candidate.layer_idx)
            if (
                not candidate_key in self.experts_protected_ondemand
                and not candidate_key in self.experts_protected_prefetch
            ):
                if candidate_key in self.cpu_expert_cache:
                    del self.cpu_expert_cache[candidate_key]
                    return True

        return False

    def cache_ondemand(self, expert_list, layer_idx, total_tokens):
        expert_counter = Counter(expert_list.flatten().tolist())

        self.protect_experts_on_demand(expert_list, layer_idx)

        for expert_idx, count in expert_counter.items():
            r = count / total_tokens
            cache_gpu_success = self.cache_gpu(expert_idx, layer_idx, r)
            assert cache_gpu_success, "GPU cache is full"

            # self.unprotect_expert(expert_idx, layer_idx)

    def cache_gpu(self, seq_id, expert_idx: int, layer_idx: int, r):
        cache_key = (expert_idx, layer_idx)
        cache_entry = ExpertCacheEntry(expert_idx, layer_idx, r)
        cache_entry.timestamp = self.logical_timestamp
        self.logical_timestamp += 1

        if cache_key in self.gpu_expert_cache:
            return True
        evict_gpu_success = self.gpu_evict(seq_id, layer_idx)
        # self.logger.debug(evict_gpu_success, cache_key in self.gpu_expert_cache)
        if evict_gpu_success:
            self.gpu_expert_cache[cache_key] = cache_entry
            return True

        return False

    def cache_cpu(self, seq_id, expert_idx, layer_idx, r):
        cache_key = (expert_idx, layer_idx)
        if cache_key in self.cpu_expert_cache:
            return True
        evict_cpu_success = self.cpu_evict(seq_id, layer_idx)
        cache_entry = ExpertCacheEntry(expert_idx, layer_idx, r)
        if evict_cpu_success:
            self.cpu_expert_cache[cache_key] = cache_entry
            return True

        return False

    def visit(self, expert_idx, layer_idx):
        cache_key = (expert_idx, layer_idx)
        self.total_visit += 1
        self.expert_frequency[cache_key] += 1
        # self.logger.debug(cache_key)
        # self.logger.debug(self.gpu_expert_cache.keys())
        if cache_key in self.gpu_expert_cache:
            self.gpu_expert_cache[cache_key].visit += 1
            self.total_gpu_cache_hit += 1

        if cache_key in self.cpu_expert_cache:
            self.cpu_expert_cache[cache_key].visit += 1
            self.total_cpu_cache_hit += 1

        if layer_idx < self.tracer.num_encoder_layers:
            self.total_encoder_visit += 1
            if cache_key in self.gpu_expert_cache:
                self.total_encoder_gpu_cache_hit += 1
        else:
            self.total_decoder_visit += 1
            if cache_key in self.gpu_expert_cache:
                self.total_decoder_gpu_cache_hit += 1

        self.logger.debug(
            f"Visit expert {expert_idx} at layer {layer_idx} cache hit {cache_key in self.gpu_expert_cache} encoder hit {(layer_idx < self.tracer.num_encoder_layers) & (cache_key in self.gpu_expert_cache)}"
        )

    def protect_experts_by_layer(self, layer_idx: int):
        self.experts_protected_by_layer = {
            (expert_idx, layer_idx): ExpertCacheEntry(expert_idx, layer_idx) for expert_idx in range(self.tracer.num_experts)
        }

    def protect_experts_on_demand(
        self, expert_list: np.ndarray, layer_idx, total_tokens
    ):
        cache_entries = []
        # total_tokens = np.prod(expert_list.shape)
        expert_counter = Counter(expert_list.flatten().tolist())
        for expert_idx, count in expert_counter.items():
            # self.logger.debug("Protecting expert", expert_idx, "at layer", layer_idx, "with count", count, "total tokens", total_tokens)
            cache_entries.append(
                ExpertCacheEntry(expert_idx, layer_idx, count / total_tokens)
            )

        self.experts_protected_ondemand = {
            (entry.expert_idx, entry.layer_idx): entry for entry in cache_entries
        }

    def protect_experts_prefetch(self, matrix, layer_idx: int):
        cache_entries = []
        for l in range(layer_idx + 1, matrix.shape[0]):
            for e in range(matrix.shape[1]):
                if matrix[l, e] > 0:
                    cache_entries.append(ExpertCacheEntry(e, l, matrix[l, e]))

        self.experts_protected_prefetch = {
            (entry.expert_idx, entry.layer_idx): entry for entry in cache_entries
        }


if __name__ == "__main__":
    pass
