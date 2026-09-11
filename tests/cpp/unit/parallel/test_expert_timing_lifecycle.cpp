// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_set>

#include "parallel/expert_timing.h"

namespace {

class FakeCuda : public CudaTimingAdapter {
 public:
  cudaError_t stop_query = cudaSuccess;
  cudaError_t fence_query = cudaSuccess;
  cudaError_t stream_sync = cudaSuccess;

  int create_timing_calls = 0;
  int create_fence_calls = 0;
  int record_calls = 0;
  int query_calls = 0;
  int stop_query_calls = 0;
  int elapsed_calls = 0;
  int destroy_timing_calls = 0;
  int destroy_fence_calls = 0;

  cudaError_t CreateTimingEvent(cudaEvent_t* event) override {
    ++create_timing_calls;
    *event = MakeHandle();
    timing_events_.insert(*event);
    return cudaSuccess;
  }

  cudaError_t CreateFenceEvent(cudaEvent_t* event) override {
    ++create_fence_calls;
    *event = MakeHandle();
    fence_events_.insert(*event);
    return cudaSuccess;
  }

  cudaError_t RecordEvent(cudaEvent_t, cudaStream_t) override {
    ++record_calls;
    return cudaSuccess;
  }

  cudaError_t QueryEvent(cudaEvent_t event) override {
    ++query_calls;
    if (fence_events_.count(event) > 0) {
      return fence_query;
    }
    ++stop_query_calls;
    return stop_query;
  }

  cudaError_t ElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) override {
    ++elapsed_calls;
    *ms = 0.0f;
    return cudaSuccess;
  }

  cudaError_t SynchronizeStream(cudaStream_t) override { return stream_sync; }

  cudaError_t DestroyTimingEvent(cudaEvent_t) override {
    ++destroy_timing_calls;
    return cudaSuccess;
  }

  cudaError_t DestroyFenceEvent(cudaEvent_t) override {
    ++destroy_fence_calls;
    return cudaSuccess;
  }

 private:
  cudaEvent_t MakeHandle() { return reinterpret_cast<cudaEvent_t>(++counter_); }
  std::uintptr_t counter_ = 0;
  std::unordered_set<cudaEvent_t> timing_events_;
  std::unordered_set<cudaEvent_t> fence_events_;
};

cudaStream_t FakeStream(int v) {
  return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(v));
}

}  // namespace

TEST(ExpertTimingLifecycle, NormalStopNotReadyIsPendingAndNeverReused) {
  FakeCuda cuda;
  ExpertTimingLifecycle lifecycle(/*pairs=*/2, &cuda);
  auto first = lifecycle.Acquire();
  lifecycle.MarkStartRecorded(first);
  lifecycle.MarkStopRecorded(first);
  cuda.stop_query = cudaErrorNotReady;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(first), TimingPairState::kPending);
  EXPECT_NE(lifecycle.Acquire().id, first.id);
  EXPECT_EQ(cuda.elapsed_calls, 0);
}

TEST(ExpertTimingLifecycle, ForwardExceptionQuarantinesUntilFenceCompletes) {
  FakeCuda cuda;
  ExpertTimingLifecycle lifecycle(/*pairs=*/1, &cuda);
  auto pair = lifecycle.Acquire();
  lifecycle.MarkStartRecorded(pair);
  lifecycle.QuarantineAfterException(pair, /*stream=*/FakeStream(7));
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kQuarantined);
  EXPECT_FALSE(lifecycle.TryAcquire().has_value());
  EXPECT_EQ(cuda.stop_query_calls, 0);
  cuda.fence_query = cudaErrorNotReady;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kQuarantined);
  cuda.fence_query = cudaSuccess;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kFree);
  EXPECT_EQ(cuda.elapsed_calls, 0);
}

TEST(ExpertTimingLifecycle, FenceErrorRequiresSuccessfulStreamProof) {
  FakeCuda cuda;
  ExpertTimingLifecycle lifecycle(/*pairs=*/1, &cuda);
  auto pair = lifecycle.Acquire();
  lifecycle.QuarantineAfterException(pair, /*stream=*/FakeStream(7));
  cuda.fence_query = cudaErrorInvalidResourceHandle;
  cuda.stream_sync = cudaErrorLaunchFailure;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kQuarantined);
  cuda.stream_sync = cudaSuccess;
  lifecycle.Poll();
  EXPECT_EQ(lifecycle.state(pair), TimingPairState::kFree);
}

TEST(ExpertTimingLifecycle, DestructorDoesNotDestroyUnprovenEvents) {
  FakeCuda cuda;
  cuda.stream_sync = cudaErrorLaunchFailure;
  {
    ExpertTimingLifecycle lifecycle(/*pairs=*/1, &cuda);
    auto pair = lifecycle.Acquire();
    lifecycle.QuarantineAfterException(pair, /*stream=*/FakeStream(7));
  }
  EXPECT_EQ(cuda.destroy_timing_calls, 0);
  EXPECT_EQ(cuda.destroy_fence_calls, 0);
}

TEST(ExpertTimingLifecycle, DisabledPolicyCreatesNoCudaTimingObjects) {
  FakeCuda cuda;
  ExpertTimingLifecycle lifecycle(/*enabled=*/false, /*pairs=*/4, &cuda);
  EXPECT_FALSE(lifecycle.TryAcquire().has_value());
  lifecycle.Poll();
  EXPECT_EQ(cuda.create_timing_calls, 0);
  EXPECT_EQ(cuda.create_fence_calls, 0);
  EXPECT_EQ(cuda.record_calls, 0);
  EXPECT_EQ(cuda.query_calls, 0);
  EXPECT_EQ(cuda.elapsed_calls, 0);
}
