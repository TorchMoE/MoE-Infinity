#pragma once

#include <algorithm>
#include <cstdint>
#include <tuple>

enum class ExpertPhase : std::uint8_t { PREFILL = 0, DECODE = 1, MIXED = 2 };
enum class AdmissionMode : std::uint8_t {
  CACHE = 0,
  TRANSIENT_ON_PRESSURE = 1,
};

struct PhasePolicyConfig {
  bool enabled = false;
  AdmissionMode prefill_admission = AdmissionMode::TRANSIENT_ON_PRESSURE;
  AdmissionMode decode_admission = AdmissionMode::CACHE;
  double prefill_eviction_weight = 1.0;
  double decode_eviction_weight = 4.0;
  std::uint32_t starvation_limit = 8;
};

struct ExpertPolicyMetadata {
  std::uint64_t prefill_accesses = 0;
  std::uint64_t decode_accesses = 0;
  std::uint64_t last_prefill_sequence = 0;
  std::uint64_t last_decode_sequence = 0;
};

struct VictimCandidate {
  std::int64_t layer_id;
  std::int64_t expert_id;
  double utility;
  std::uint64_t last_sequence;
};

inline ExpertPhase EffectivePhase(ExpertPhase phase) {
  return phase == ExpertPhase::MIXED ? ExpertPhase::DECODE : phase;
}

inline double VictimUtility(const ExpertPolicyMetadata& m, ExpertPhase phase,
                            const PhasePolicyConfig& cfg) {
  const auto active = EffectivePhase(phase);
  const double prefill_weight =
      active == ExpertPhase::PREFILL ? cfg.prefill_eviction_weight : 1.0;
  const double decode_weight =
      active == ExpertPhase::DECODE ? cfg.decode_eviction_weight : 1.0;
  return prefill_weight * m.prefill_accesses +
         decode_weight * m.decode_accesses;
}

inline bool VictimLess(const VictimCandidate& a, const VictimCandidate& b) {
  return std::tie(a.utility, a.last_sequence, a.layer_id, a.expert_id) <
         std::tie(b.utility, b.last_sequence, b.layer_id, b.expert_id);
}

inline std::uint32_t ServiceClass(std::uint32_t priority,
                                  std::uint32_t bypasses, std::uint32_t limit) {
  if (priority == 0) return 0;
  return bypasses >= limit ? 1 : priority;
}
