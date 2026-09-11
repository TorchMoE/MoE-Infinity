#include <gtest/gtest.h>

#include <chrono>
#include <future>

#include "parallel/expert_dispatcher.h"
#include "prefetch/task_scheduler.h"
#include "sparse_cache_resize_fixture.h"

TEST(SparseCacheResize, ProductionApiSignaturesExist) {
  auto reserve = &ArcherTaskPool::ReserveSparseCacheVictims;
  auto cancel = &ArcherTaskPool::CancelSparseCacheReservation;
  auto commit = &ArcherTaskPool::CommitSparseCacheReservation;
  auto begin = &ExpertDispatcher::BeginMemoryResize;
  auto end = &ExpertDispatcher::EndMemoryResize;
  EXPECT_NE(reserve, nullptr);
  EXPECT_NE(cancel, nullptr);
  EXPECT_NE(commit, nullptr);
  EXPECT_NE(begin, nullptr);
  EXPECT_NE(end, nullptr);
}

TEST(SparseCacheResize, RejectsPinnedAndExecutingNodes) {
  FakeSparseCache cache({{0, 64}, {1, 64}, {2, 64}});
  cache.node(0).pending_dispatches.store(1);
  cache.node(1).exec_state.store(NodeExecState::FETCHING);
  cache.ReplaceCacheCandidates({cache.node_ptr(2)});
  auto result = cache.TrimSparseCache(/*device=*/0, /*target_bytes=*/64);
  EXPECT_EQ(result.outcome, "rejected");
  EXPECT_EQ(result.resident_bytes, 192);
  EXPECT_EQ(result.reason, "pinned_or_in_flight");
}

TEST(SparseCacheResize, EvictsOnlyIdleUnprotectedNodes) {
  FakeSparseCache cache({{0, 64}, {1, 64}, {2, 64}});
  cache.ReplaceCacheCandidates({cache.node_ptr(2)});
  auto result = cache.TrimSparseCache(0, 128);
  EXPECT_EQ(result.outcome, "committed");
  EXPECT_LE(result.resident_bytes, 128);
  EXPECT_TRUE(cache.node(2).device.is_cuda());
}

TEST(SparseCacheResize, ReservationRollbackDoesNotEvictVictims) {
  FakeSparseCache cache({{0, 64}, {1, 64}, {2, 64}});
  auto reservation = cache.ReserveSparseCacheVictims(0, 128);
  ASSERT_TRUE(reservation.ready);
  EXPECT_EQ(cache.ResidentBytes(0), 192);
  cache.CancelSparseCacheReservation(reservation.id);
  EXPECT_EQ(cache.ResidentBytes(0), 192);
  EXPECT_TRUE(cache.AllNodesResident(0));
}

TEST(SparseCacheResize, DispatcherDrainWaitsForQueuesWorkersAndStreams) {
  FakeDispatcher dispatcher(/*devices=*/2);
  dispatcher.EnqueueFetch(/*device=*/1);
  dispatcher.EnqueueExec(/*device=*/1);
  dispatcher.SetFetchEventComplete(/*device=*/1, false);
  auto blocked = dispatcher.BeginMemoryResize(/*device=*/1, /*timeout_ms=*/1);
  EXPECT_FALSE(blocked.ready);
  EXPECT_EQ(blocked.reason, "dispatcher_drain_timeout");
  dispatcher.CompleteQueuesWorkersAndEvents(/*device=*/1);
  auto ready = dispatcher.BeginMemoryResize(/*device=*/1, /*timeout_ms=*/1000);
  EXPECT_TRUE(ready.ready);
  EXPECT_TRUE(dispatcher.AdmissionsPaused(1));
  dispatcher.EndMemoryResize(ready);
  EXPECT_FALSE(dispatcher.AdmissionsPaused(1));
}

TEST(SparseCacheResize, ReservationNeverNestsExecAndCandidateLocks) {
  auto* pool = ArcherTaskPool::GetInstance();
  std::promise<void> exec_snapshot_released;
  std::promise<void> resume_snapshot;
  auto resume = resume_snapshot.get_future().share();
  pool->SetAfterExecSnapshotHookForTest([&] {
    exec_snapshot_released.set_value();
    resume.wait();
  });

  auto snapshot = std::async(std::launch::async, [&] {
    return pool->SnapshotResizeExclusionsForTest(/*device_id=*/0);
  });
  ASSERT_EQ(
      exec_snapshot_released.get_future().wait_for(std::chrono::seconds(1)),
      std::future_status::ready);
  auto replace = std::async(
      std::launch::async, [&] { pool->ReplaceCacheCandidates(NodePtrList{}); });
  EXPECT_EQ(replace.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  resume_snapshot.set_value();
  EXPECT_EQ(snapshot.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  snapshot.get();
  pool->SetAfterExecSnapshotHookForTest({});
}
