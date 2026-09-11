// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <cuda_runtime_api.h>

// Testable free/pending/quarantined timing-pair lifecycle for dispatcher-owned,
// timing-enabled CUDA kernel events. All CUDA calls go through an adapter so
// the state machine can be exercised on the CPU with a fake status source: no
// GPU is required to prove that normal pairs retire only after a successful
// stop query, exception pairs are quarantined behind a same-stream fence until
// a query or stream synchronization proves completion, and unproven handles are
// never reused, queried, or destroyed.

enum class TimingPairState {
  kFree,
  kActive,
  kPending,
  kQuarantined,
};

struct TimingTicket {
  int id = -1;
};

// Adapter over the subset of the CUDA runtime the lifecycle needs. The real
// dispatcher supplies a thin implementation; tests supply a fake that returns
// scripted status codes and counts calls.
class CudaTimingAdapter {
 public:
  virtual ~CudaTimingAdapter() = default;
  virtual cudaError_t CreateTimingEvent(cudaEvent_t* event) = 0;
  virtual cudaError_t CreateFenceEvent(cudaEvent_t* event) = 0;
  virtual cudaError_t RecordEvent(cudaEvent_t event, cudaStream_t stream) = 0;
  virtual cudaError_t QueryEvent(cudaEvent_t event) = 0;
  virtual cudaError_t ElapsedTime(float* ms, cudaEvent_t start,
                                  cudaEvent_t stop) = 0;
  virtual cudaError_t SynchronizeStream(cudaStream_t stream) = 0;
  virtual cudaError_t DestroyTimingEvent(cudaEvent_t event) = 0;
  virtual cudaError_t DestroyFenceEvent(cudaEvent_t event) = 0;
};

class ExpertTimingLifecycle {
 public:
  ExpertTimingLifecycle(int pairs, CudaTimingAdapter* cuda)
      : ExpertTimingLifecycle(true, pairs, cuda) {}

  ExpertTimingLifecycle(bool enabled, int pairs, CudaTimingAdapter* cuda)
      : enabled_(enabled), cuda_(cuda) {
    if (!enabled_) return;
    pairs_.reserve(pairs);
    for (int i = 0; i < pairs; ++i) {
      Pair pair;
      pair.id = i;
      pair.state = TimingPairState::kFree;
      cuda_->CreateTimingEvent(&pair.start);
      cuda_->CreateTimingEvent(&pair.stop);
      pairs_.push_back(pair);
    }
  }

  ~ExpertTimingLifecycle() {
    if (!enabled_) return;
    for (auto& pair : pairs_) {
      bool proven = pair.state == TimingPairState::kFree ||
                    pair.state == TimingPairState::kActive;
      if (!proven) {
        if (cuda_->SynchronizeStream(kQuarantineStream) != cudaSuccess) {
          continue;
        }
      }
      if (pair.start != nullptr) cuda_->DestroyTimingEvent(pair.start);
      if (pair.stop != nullptr) cuda_->DestroyTimingEvent(pair.stop);
      if (pair.fence != nullptr) cuda_->DestroyFenceEvent(pair.fence);
    }
  }

  std::optional<TimingTicket> TryAcquire() {
    if (!enabled_) return std::nullopt;
    for (auto& pair : pairs_) {
      if (pair.state == TimingPairState::kFree) {
        pair.state = TimingPairState::kActive;
        pair.start_recorded = false;
        pair.stop_recorded = false;
        return TimingTicket{pair.id};
      }
    }
    return std::nullopt;
  }

  TimingTicket Acquire() {
    auto ticket = TryAcquire();
    return ticket.has_value() ? *ticket : TimingTicket{-1};
  }

  void MarkStartRecorded(const TimingTicket& ticket) {
    Pair* pair = Find(ticket);
    if (pair == nullptr) return;
    pair->start_recorded = true;
  }

  void MarkStopRecorded(const TimingTicket& ticket) {
    Pair* pair = Find(ticket);
    if (pair == nullptr) return;
    pair->stop_recorded = true;
    pair->state = TimingPairState::kPending;
  }

  void QuarantineAfterException(const TimingTicket& ticket,
                                cudaStream_t stream) {
    Pair* pair = Find(ticket);
    if (pair == nullptr) return;
    pair->state = TimingPairState::kQuarantined;
    pair->fence_recorded = false;
    if (cuda_->CreateFenceEvent(&pair->fence) == cudaSuccess) {
      if (cuda_->RecordEvent(pair->fence, stream) == cudaSuccess) {
        pair->fence_recorded = true;
      }
    }
  }

  void Poll() {
    if (!enabled_) return;
    for (auto& pair : pairs_) {
      if (pair.state == TimingPairState::kPending) {
        PollPending(pair);
      } else if (pair.state == TimingPairState::kQuarantined) {
        PollQuarantined(pair);
      }
    }
  }

  TimingPairState state(const TimingTicket& ticket) {
    Pair* pair = Find(ticket);
    return pair == nullptr ? TimingPairState::kFree : pair->state;
  }

 private:
  struct Pair {
    int id = -1;
    TimingPairState state = TimingPairState::kFree;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEvent_t fence = nullptr;
    bool start_recorded = false;
    bool stop_recorded = false;
    bool fence_recorded = false;
  };

  Pair* Find(const TimingTicket& ticket) {
    for (auto& pair : pairs_) {
      if (pair.id == ticket.id) return &pair;
    }
    return nullptr;
  }

  void PollPending(Pair& pair) {
    cudaError_t status = cuda_->QueryEvent(pair.stop);
    if (status == cudaErrorNotReady) return;
    if (status == cudaSuccess) {
      float ms = 0.0f;
      cuda_->ElapsedTime(&ms, pair.start, pair.stop);
    }
    Reset(pair);
  }

  void PollQuarantined(Pair& pair) {
    if (pair.fence_recorded) {
      cudaError_t status = cuda_->QueryEvent(pair.fence);
      if (status == cudaErrorNotReady) return;
      if (status == cudaSuccess) {
        cuda_->DestroyFenceEvent(pair.fence);
        pair.fence = nullptr;
        Reset(pair);
        return;
      }
    }
    if (cuda_->SynchronizeStream(kQuarantineStream) == cudaSuccess) {
      if (pair.fence != nullptr) {
        cuda_->DestroyFenceEvent(pair.fence);
        pair.fence = nullptr;
      }
      Reset(pair);
    }
  }

  void Reset(Pair& pair) {
    pair.state = TimingPairState::kFree;
    pair.start_recorded = false;
    pair.stop_recorded = false;
    pair.fence_recorded = false;
  }

  static constexpr cudaStream_t kQuarantineStream = nullptr;

  bool enabled_;
  CudaTimingAdapter* cuda_;
  std::vector<Pair> pairs_;
};
