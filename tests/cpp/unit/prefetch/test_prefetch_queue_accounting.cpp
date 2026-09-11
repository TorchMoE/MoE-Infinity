// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <gtest/gtest.h>

#include <stdexcept>

#include "prefetch/task_scheduler.h"

TEST(PrefetchQueueAccounting, AdmissionCancellationAndCompletionAreDisjoint) {
  PrefetchQueueAccounting a;
  EXPECT_TRUE(a.TryAdmit(10, 100, 250));
  EXPECT_TRUE(a.TryAdmit(11, 100, 250));
  EXPECT_FALSE(a.TryAdmit(12, 100, 250));
  EXPECT_EQ(a.queued_bytes(), 200);
  a.MarkStarted(10);
  EXPECT_EQ(a.CancelQueued(11), 100);
  EXPECT_EQ(a.CancelQueued(10), 0);
  a.MarkCompleted(10);
  EXPECT_EQ(a.inflight_bytes(), 0);
  EXPECT_EQ(a.completed_bytes(), 100);
  EXPECT_EQ(a.canceled_bytes(), 100);
  EXPECT_EQ(a.rejected_bytes(), 100);
  EXPECT_TRUE(a.InvariantHolds());
}

TEST(PrefetchQueueAccounting, EveryQueueRemovalReasonRetiresInflightBytesOnce) {
  for (auto reason :
       {RemovalReason::kClear, RemovalReason::kReplaceCandidates,
        RemovalReason::kFetchSweep, RemovalReason::kDeduplicate,
        RemovalReason::kObsoleteLayer, RemovalReason::kExplicitCancel,
        RemovalReason::kPopDuplicate, RemovalReason::kShutdown}) {
    PrefetchQueueAccounting a;
    ASSERT_TRUE(a.TryAdmit(10, 128, 1024));
    EXPECT_EQ(a.RetireQueued(10, reason), 128);
    EXPECT_EQ(a.RetireQueued(10, reason), 0);
    EXPECT_EQ(a.inflight_bytes(), 0);
    EXPECT_EQ(a.removed_bytes(reason), 128);
    EXPECT_TRUE(a.InvariantHolds());
  }
}

TEST(PrefetchQueueAccounting, PopMovesQueuedToRunningThenAllTerminalsBalance) {
  for (auto outcome :
       {RunningOutcome::kCompleted, RunningOutcome::kEvictionFailed,
        RunningOutcome::kStateConflict, RunningOutcome::kAlreadyResident,
        RunningOutcome::kTransferFailed}) {
    PrefetchQueueAccounting a;
    ASSERT_TRUE(a.TryAdmit(10, 256, 1024));
    ASSERT_TRUE(a.MarkStarted(10));
    a.RetireRunning(10, outcome,
                    outcome == RunningOutcome::kCompleted ? 256 : 0);
    EXPECT_EQ(a.inflight_bytes(), 0);
    EXPECT_TRUE(a.InvariantHolds());
  }
}

TEST(PrefetchQueueAccounting, WorkerBoundaryRecordsStdExceptionAsFailure) {
  PrefetchQueueAccounting a;
  ASSERT_TRUE(a.TryAdmit(10, 256, 1024));
  ASSERT_TRUE(a.MarkStarted(10));
  EXPECT_NO_THROW(RunPrefetchTaskNoThrow(
      10, &a, [] { throw std::runtime_error("SetNodeDevice failed"); }));
  EXPECT_EQ(a.inflight_bytes(), 0);
  EXPECT_EQ(a.failed_bytes(), 256);
  EXPECT_TRUE(a.InvariantHolds());
}

TEST(PrefetchQueueAccounting, WorkerBoundaryRecordsUnknownExceptionAsFailure) {
  PrefetchQueueAccounting a;
  ASSERT_TRUE(a.TryAdmit(10, 256, 1024));
  ASSERT_TRUE(a.MarkStarted(10));
  EXPECT_NO_THROW(RunPrefetchTaskNoThrow(10, &a, [] { throw 7; }));
  EXPECT_EQ(a.inflight_bytes(), 0);
  EXPECT_EQ(a.failed_bytes(), 256);
  EXPECT_TRUE(a.InvariantHolds());
}
