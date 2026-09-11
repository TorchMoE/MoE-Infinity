#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "prefetch/expert_policy.h"
#include "prefetch/expert_residency.h"

TEST(ExpertPolicy, DecodeWeightsDecodeReuse) {
  PhasePolicyConfig cfg{true,
                        AdmissionMode::TRANSIENT_ON_PRESSURE,
                        AdmissionMode::CACHE,
                        1.0,
                        4.0,
                        8};
  ExpertPolicyMetadata prefill_hot{10, 0, 10, 0};
  ExpertPolicyMetadata decode_hot{0, 3, 0, 9};
  EXPECT_LT(VictimUtility(prefill_hot, ExpertPhase::DECODE, cfg),
            VictimUtility(decode_hot, ExpertPhase::DECODE, cfg));
}

TEST(ExpertPolicy, StableVictimTieBreakUsesLayerThenExpert) {
  VictimCandidate a{2, 7, 1.0, 4};
  VictimCandidate b{3, 0, 1.0, 4};
  EXPECT_TRUE(VictimLess(a, b));
}

TEST(ExpertPolicy, BypassPromotionNeverPassesDemand) {
  EXPECT_EQ(ServiceClass(0, 99, 8), 0);
  EXPECT_EQ(ServiceClass(2, 8, 8), 1);
  EXPECT_EQ(ServiceClass(1, 0, 8), 1);
}

TEST(ExpertResidency, EligibilityAcceptsOnlyIdleUnprotectedResident) {
  EvictionState eligible{/*is_cuda=*/true,
                         /*is_prefetching=*/false,
                         NodeExecState::IDLE,
                         /*pending_dispatches=*/0,
                         /*is_overflow=*/false,
                         /*lease_count=*/0,
                         /*protected_candidate=*/false};
  EXPECT_TRUE(IsEvictionEligible(eligible));

  auto state = eligible;
  state.is_cuda = false;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible;
  state.is_prefetching = true;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible;
  state.exec_state = NodeExecState::FETCHING;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible;
  state.exec_state = NodeExecState::EXECUTING;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible;
  state.pending_dispatches = 1;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible;
  state.is_overflow = true;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible;
  state.lease_count = 1;
  EXPECT_FALSE(IsEvictionEligible(state));
  state = eligible;
  state.protected_candidate = true;
  EXPECT_FALSE(IsEvictionEligible(state));
}

TEST(ExpertResidency, RandomizedCandidateOrderAlwaysSelectsStableVictim) {
  const std::vector<VictimCandidate> original{
      {4, 2, 1.0, 9}, {1, 3, 1.0, 9}, {2, 1, 2.0, 1}, {0, 7, 3.0, 2}};
  for (std::uint32_t seed = 0; seed < 100; ++seed) {
    auto candidates = original;
    std::shuffle(candidates.begin(), candidates.end(), std::mt19937(seed));
    const auto victim = SelectStableVictim(candidates);
    ASSERT_TRUE(victim.has_value());
    EXPECT_EQ(victim->layer_id, 1);
    EXPECT_EQ(victim->expert_id, 3);
  }
}
