#include "expert_residency_test_fixture.h"

#include <thread>

TEST_F(ExpertResidencyManagerTest, AdmissionIsRejectedUntilDeviceConfigured) {
  auto rejected = Begin(MakeNode(1, 40), AdmissionSource::DEMAND);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.outcome, AdmissionOutcome::REJECTED);
  EXPECT_EQ(manager->ResidentBytes(0), 0);
  EXPECT_EQ(manager->ResidentCount(0), 0);
  EXPECT_EQ(manager->Snapshot().at("decode_admissions"), 0);

  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto allowed = Begin(MakeNode(2, 40), AdmissionSource::DEMAND);
  EXPECT_TRUE(allowed.valid);
  EXPECT_EQ(allowed.outcome, AdmissionOutcome::ADMIT);
  EXPECT_TRUE(manager->AbortAdmission(allowed));
}

TEST_F(ExpertResidencyManagerTest, DevicesConfigureIndependently) {
  ASSERT_TRUE(manager->ConfigureCapacity(1, 200));
  auto gpu0 =
      manager->BeginAdmission(MakeNode(1, 40), 0, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  auto gpu1 =
      manager->BeginAdmission(MakeNode(2, 40), 1, ExpertPhase::DECODE,
                              AdmissionMode::CACHE, AdmissionSource::DEMAND);
  EXPECT_FALSE(gpu0.valid);
  EXPECT_TRUE(gpu1.valid);
  EXPECT_TRUE(manager->AbortAdmission(gpu1));
}

TEST_F(ExpertResidencyManagerTest, AllowsOnlyOneTransientOverflowPerDevice) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 40));
  auto resident = MakeNode(1, 40);
  Admit(resident, AdmissionSource::DEMAND);

  auto first = manager->BeginAdmission(MakeNode(2, 40), 0, ExpertPhase::PREFILL,
                                       AdmissionMode::TRANSIENT_ON_PRESSURE,
                                       AdmissionSource::DEMAND);
  auto second = manager->BeginAdmission(
      MakeNode(3, 40), 0, ExpertPhase::PREFILL,
      AdmissionMode::TRANSIENT_ON_PRESSURE, AdmissionSource::DEMAND);

  ASSERT_TRUE(first.valid);
  EXPECT_EQ(first.outcome, AdmissionOutcome::TRANSIENT);
  EXPECT_FALSE(second.valid);
  EXPECT_EQ(second.outcome, AdmissionOutcome::REJECTED);
  EXPECT_TRUE(manager->AbortAdmission(first));

  auto after_release = manager->BeginAdmission(
      MakeNode(4, 40), 0, ExpertPhase::PREFILL,
      AdmissionMode::TRANSIENT_ON_PRESSURE, AdmissionSource::DEMAND);
  EXPECT_TRUE(after_release.valid);
  EXPECT_EQ(after_release.outcome, AdmissionOutcome::TRANSIENT);
  EXPECT_TRUE(manager->AbortAdmission(after_release));
}

TEST_F(ExpertResidencyManagerTest, CapacityUpdateIsAtomicAndPerDevice) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto resident = MakeNode(1, 60);
  Admit(resident, AdmissionSource::DEMAND);

  EXPECT_TRUE(manager->ConfigureCapacity(0, 80));
  EXPECT_EQ(manager->CapacityBytes(0), 80);
  EXPECT_FALSE(manager->ConfigureCapacity(0, 50));
  EXPECT_EQ(manager->CapacityBytes(0), 80);
  EXPECT_FALSE(manager->ConfigureCapacity(0, -1));
  EXPECT_EQ(manager->CapacityBytes(0), 80);

  auto pressure = Begin(MakeNode(2, 30), AdmissionSource::DEMAND);
  ASSERT_TRUE(pressure.valid);
  EXPECT_EQ(pressure.reserved_victim, resident);
  EXPECT_FALSE(manager->ConfigureCapacity(0, 120));
  EXPECT_EQ(manager->CapacityBytes(0), 80);
  EXPECT_TRUE(manager->AbortAdmission(pressure));
  EXPECT_TRUE(manager->ConfigureCapacity(0, 120));
  EXPECT_EQ(manager->CapacityBytes(0), 120);
}

TEST_F(ExpertResidencyManagerTest, CapacityEvictsReservedVictimBeforeCommit) {
  auto first = MakeNode(1, 60);
  Admit(first, AdmissionSource::DEMAND);
  auto second = MakeNode(2, 60);
  auto ticket = Begin(second, AdmissionSource::DEMAND);
  ASSERT_TRUE(ticket.valid);
  ASSERT_EQ(ticket.reserved_victim, first);
  EXPECT_TRUE(manager->EvictReserved(ticket));
  second->device = second->default_device;
  EXPECT_TRUE(manager->CommitAdmission(ticket));
  EXPECT_EQ(manager->ResidentBytes(0), 60);
  EXPECT_EQ(manager->ResidentCount(0), 1);
  ASSERT_EQ(transfer_ops->moved_ids.size(), 1);
  EXPECT_EQ(transfer_ops->moved_ids[0], first->id);
}

TEST_F(ExpertResidencyManagerTest, DuplicateAdmissionIsNoOp) {
  auto node = MakeNode(1, 40);
  Admit(node, AdmissionSource::DEMAND);
  auto duplicate = Begin(node, AdmissionSource::PREFETCH);
  EXPECT_TRUE(duplicate.valid);
  EXPECT_EQ(duplicate.outcome, AdmissionOutcome::ALREADY_RESIDENT);
  EXPECT_FALSE(manager->CommitAdmission(duplicate));
  EXPECT_EQ(manager->ResidentBytes(0), 40);
  EXPECT_EQ(manager->ResidentCount(0), 1);
}

TEST_F(ExpertResidencyManagerTest, VictimReservationAllowsOnlyOneRacingTicket) {
  auto victim = MakeNode(1, 100);
  Admit(victim, AdmissionSource::DEMAND);
  std::atomic<bool> go{false};
  ResidencyTicket tickets[2];
  std::thread threads[2];
  for (int i = 0; i < 2; ++i) {
    threads[i] = std::thread([&, i] {
      while (!go.load(std::memory_order_acquire)) {
      }
      tickets[i] = Begin(MakeNode(10 + i, 100), AdmissionSource::DEMAND);
    });
  }
  go.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();
  const int reservations = int(tickets[0].reserved_victim != nullptr) +
                           int(tickets[1].reserved_victim != nullptr);
  EXPECT_EQ(reservations, 1);
  for (const auto& ticket : tickets) {
    if (ticket.valid) EXPECT_TRUE(manager->AbortAdmission(ticket));
  }
}

TEST_F(ExpertResidencyManagerTest, CommitAndAbortAreIdempotent) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  auto committed = MakeNode(1, 40);
  auto commit_ticket = Begin(committed, AdmissionSource::DEMAND);
  committed->device = committed->default_device;
  EXPECT_TRUE(manager->CommitAdmission(commit_ticket));
  EXPECT_FALSE(manager->CommitAdmission(commit_ticket));

  auto aborted = MakeNode(2, 40);
  auto abort_ticket = Begin(aborted, AdmissionSource::PREFETCH);
  EXPECT_TRUE(manager->AbortAdmission(abort_ticket));
  EXPECT_FALSE(manager->AbortAdmission(abort_ticket));
  EXPECT_EQ(manager->ResidentBytes(0), 40);
}

TEST_F(ExpertResidencyManagerTest, LeaseReleaseRestoresEvictionEligibility) {
  auto resident = MakeNode(1, 100);
  Admit(resident, AdmissionSource::DEMAND);
  const auto lease = manager->AcquireLease(resident, LeaseKind::DEMAND);
  auto blocked = Begin(MakeNode(2, 100), AdmissionSource::DEMAND);
  EXPECT_FALSE(blocked.valid);
  EXPECT_TRUE(manager->ReleaseLease(lease));
  EXPECT_FALSE(manager->ReleaseLease(lease));
  auto allowed = Begin(MakeNode(3, 100), AdmissionSource::DEMAND);
  EXPECT_TRUE(allowed.valid);
  EXPECT_EQ(allowed.reserved_victim, resident);
  EXPECT_TRUE(manager->AbortAdmission(allowed));
}

TEST_F(ExpertResidencyManagerTest, DemandAndPrefetchShareByteAccounting) {
  ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
  ExpertResidencyClient dispatcher_client(manager, AdmissionSource::DEMAND);
  ExpertResidencyClient prefetch_client(manager, AdmissionSource::PREFETCH);
  auto demand = MakeNode(1, 40);
  auto demand_ticket = dispatcher_client.BeginAdmission(
      demand, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  demand->device = demand->default_device;
  ASSERT_TRUE(manager->CommitAdmission(demand_ticket));
  auto prefetched = MakeNode(2, 50);
  auto prefetch_ticket = prefetch_client.BeginAdmission(
      prefetched, 0, ExpertPhase::DECODE, AdmissionMode::CACHE);
  prefetched->device = prefetched->default_device;
  ASSERT_TRUE(manager->CommitAdmission(prefetch_ticket));
  const auto stats = manager->Snapshot();
  EXPECT_EQ(manager->ResidentBytes(0), 90);
  EXPECT_EQ(manager->ResidentCount(0), 2);
  EXPECT_EQ(stats.at("resident_bytes"), 90);
  EXPECT_EQ(stats.at("resident_experts"), 2);
  EXPECT_EQ(stats.at("decode_admissions"), 1);
  EXPECT_EQ(stats.at("decode_prefetch_completed"), 1);
}
