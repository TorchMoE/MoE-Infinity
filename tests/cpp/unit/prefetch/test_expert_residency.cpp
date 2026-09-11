// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_residency_test_fixture.h"

TEST_F(ExpertResidencyManagerTest, AdmissionRejectsUntilCapacityConfigured) {
  auto node = MakeNode(1, 40);
  auto rejected =
      manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.outcome, AdmissionOutcome::REJECTED);
  EXPECT_EQ(manager->ResidentBytes(0), 0);
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto admitted =
      manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_TRUE(admitted.valid);
  EXPECT_TRUE(manager->AbortAdmission(admitted));
}

TEST_F(ExpertResidencyManagerTest, CapacityAndVictimTransactionAreAtomic) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 60);
  auto first_ticket =
      manager->BeginAdmission(first, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(first_ticket));
  auto second = MakeNode(2, 60);
  auto second_ticket =
      manager->BeginAdmission(second, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::PREFETCH);
  ASSERT_EQ(second_ticket.reserved_victim, first);
  ASSERT_TRUE(manager->EvictReserved(second_ticket));
  second->device = second->default_device;
  ASSERT_TRUE(manager->CommitAdmission(second_ticket));
  EXPECT_EQ(manager->ResidentBytes(0), 60);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest, LeaseBlocksEvictionUntilReleased) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeNode(1, 100);
  auto ticket =
      manager->BeginAdmission(first, 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  first->device = first->default_device;
  ASSERT_TRUE(manager->CommitAdmission(ticket));
  const auto lease = manager->AcquireLease(first, LeaseKind::DEMAND);
  EXPECT_FALSE(manager
                   ->BeginAdmission(MakeNode(2, 100), 0, ExpertPhase::DECODE,
                                    AdmissionMode::CACHE,
                                    AdmissionSource::DEMAND)
                   .valid);
  EXPECT_TRUE(manager->ReleaseLease(lease));
  EXPECT_FALSE(manager->ReleaseLease(lease));
  auto retry =
      manager->BeginAdmission(MakeNode(3, 100), 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_EQ(retry.reserved_victim, first);
  EXPECT_TRUE(manager->AbortAdmission(retry));
}

TEST_F(ExpertResidencyManagerTest, DemandAndPrefetchClientsShareAccounting) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  ExpertResidencyClient demand(manager, AdmissionSource::DEMAND);
  ExpertResidencyClient prefetch(manager, AdmissionSource::PREFETCH);
  auto node = MakeNode(1, 40);
  auto admitted =
      demand.BeginAdmission(node, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  node->device = node->default_device;
  ASSERT_TRUE(manager->CommitAdmission(admitted));
  auto duplicate = prefetch.BeginAdmission(node, 0, ExpertPhase::PREFILL,
                                           AdmissionMode::CACHE);
  EXPECT_EQ(duplicate.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_EQ(demand.manager().get(), prefetch.manager().get());
  EXPECT_EQ(manager->ResidentBytes(0), 40);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}
