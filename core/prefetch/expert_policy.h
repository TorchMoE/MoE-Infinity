// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>

enum class ExpertPhase : std::uint8_t { PREFILL = 0, DECODE = 1, MIXED = 2 };

enum class AdmissionMode : std::uint8_t {
  CACHE = 0,
  TRANSIENT_ON_PRESSURE = 1
};

inline std::uint32_t ServiceClass(std::uint32_t priority,
                                  std::uint32_t bypasses, std::uint32_t limit) {
  if (priority == 0) return 0;
  return bypasses >= limit ? 1 : priority;
}

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

inline ExpertPhase EffectivePhase(ExpertPhase phase) {
  return phase == ExpertPhase::MIXED ? ExpertPhase::DECODE : phase;
}
