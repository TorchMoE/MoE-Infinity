// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_residency_test_fixture.h"

TEST_F(ExpertResidencyManagerTest, HostReadyVariantsConsumeNoCapacity) {
  auto fp8 = MakeVariant(0, 1, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 12);
  manager->RegisterVariant(fp8);
  EXPECT_EQ(manager->ResidentBytes(0), 0);
  EXPECT_EQ(manager->Snapshot().at("registered_variants"), 1);
}

TEST_F(ExpertResidencyManagerTest, TransitionReservesDestinationAndWorkspace) {
  auto low = MakeVariant(0, 1, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 12);
  auto high = MakeVariant(0, 1, ExpertFormat::BF16, 2, 80, 16);
  CommitResident(low, ExpertPhase::DECODE);
  auto tx = manager->BeginTransition(low.key, high, 0, ExpertPhase::DECODE,
                                     AdmissionSource::DEMAND);
  ASSERT_TRUE(tx.valid);
  auto snapshot = manager->Snapshot();
  EXPECT_EQ(snapshot.at("resident_bytes"), 40);
  EXPECT_EQ(snapshot.at("transition_reserved_bytes"), 80);
  EXPECT_EQ(snapshot.at("workspace_bytes"), 16);
  EXPECT_LE(40 + 80 + 16, snapshot.at("capacity_bytes"));
  EXPECT_TRUE(manager->AbortTransaction(tx));
  EXPECT_EQ(manager->Snapshot().at("workspace_bytes"), 0);
}

TEST_F(ExpertResidencyManagerTest, CommitTransitionRetiresOldGeneration) {
  auto low = MakeVariant(0, 1, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 0);
  auto high = MakeVariant(0, 1, ExpertFormat::BF16, 2, 80, 0);
  CommitResident(low, ExpertPhase::PREFILL);
  auto lease = manager->AcquireLease(low.key, LeaseKind::EXECUTION);
  auto tx = manager->BeginTransition(low.key, high, 0, ExpertPhase::DECODE,
                                     AdmissionSource::DEMAND);
  ASSERT_TRUE(manager->CommitTransaction(tx));
  EXPECT_EQ(manager->Snapshot().at("retiring_generations"), 1);
  EXPECT_EQ(manager->ReapRetired(0), 0);
  EXPECT_TRUE(manager->ReleaseLease(lease));
  EXPECT_EQ(manager->ReapRetired(0), 1);
}

TEST_F(ExpertResidencyManagerTest, DemandAndPrefetchClientsShareVariantCharge) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1024));
  auto variant = MakeVariant(0, 2, ExpertFormat::BF16, 1, 64, 0);
  manager->RegisterVariant(variant);
  auto prefetch_tx = prefetch_client->BeginAdmission(
      variant, 0, ExpertPhase::PREFILL, AdmissionMode::CACHE);
  ASSERT_TRUE(manager->CommitTransaction(prefetch_tx));
  auto demand_tx = demand_client->BeginAdmission(
      variant, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_EQ(demand_tx.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_EQ(manager->ResidentBytes(0), 64);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest,
       ConcurrentTransitionForLogicalExpertIsRejected) {
  auto low = MakeVariant(0, 3, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 0);
  auto high = MakeVariant(0, 3, ExpertFormat::BF16, 2, 80, 0);
  CommitResident(low, ExpertPhase::DECODE);
  auto first = manager->BeginTransition(low.key, high, 0, ExpertPhase::DECODE,
                                        AdmissionSource::DEMAND);
  auto second = manager->BeginTransition(low.key, high, 0, ExpertPhase::DECODE,
                                         AdmissionSource::PREFETCH);
  EXPECT_TRUE(first.valid);
  EXPECT_FALSE(second.valid);
  EXPECT_TRUE(manager->AbortTransaction(first));
}

TEST_F(ExpertResidencyManagerTest, WorkspaceRemainsChargedUntilCompletion) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 1024));
  auto variant = MakeVariant(0, 4, ExpertFormat::FP8_E4M3_BLOCK128, 1, 40, 16);
  manager->RegisterVariant(variant);
  auto tx = demand_client->BeginAdmission(variant, 0, ExpertPhase::DECODE,
                                          AdmissionMode::CACHE);
  ASSERT_TRUE(manager->CommitTransaction(tx));
  EXPECT_EQ(manager->Snapshot().at("workspace_bytes"), 16);
  ASSERT_TRUE(manager->RecordWorkspaceUse(tx.id, nullptr));
  EXPECT_EQ(manager->ReapWorkspace(0), 1);
  EXPECT_EQ(manager->Snapshot().at("workspace_bytes"), 0);
}

TEST_F(ExpertResidencyManagerTest,
       VictimReservationNeverOverchargesCapacityOrUnderflowsOnAbort) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto first = MakeVariant(0, 5, ExpertFormat::FP8_E4M3_BLOCK128, 1, 60, 0);
  auto second = MakeVariant(0, 6, ExpertFormat::FP8_E4M3_BLOCK128, 1, 60, 0);
  manager->RegisterVariant(first);
  manager->RegisterVariant(second);
  auto first_tx = demand_client->BeginAdmission(first, 0, ExpertPhase::DECODE,
                                                AdmissionMode::CACHE);
  ASSERT_TRUE(manager->CommitTransaction(first_tx));

  auto replacement = demand_client->BeginAdmission(
      second, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  ASSERT_TRUE(replacement.valid);
  auto pending = manager->Snapshot();
  EXPECT_LE(pending.at("resident_bytes") +
                pending.at("transition_reserved_bytes") +
                pending.at("workspace_bytes"),
            pending.at("capacity_bytes"));
  ASSERT_TRUE(manager->AbortTransaction(replacement));
  EXPECT_EQ(manager->Snapshot().at("transition_reserved_bytes"), 0);
}
