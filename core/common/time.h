// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <chrono>

typedef std::chrono::high_resolution_clock::time_point TimePoint;

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define MCIROSECONDS std::chrono::microseconds
#define MILLISECONDS std::chrono::milliseconds
#define SECONDS std::chrono::seconds

#define MCIROSECONDS_SINCE_EPOCH \
    std::chrono::duration_cast<MCIROSECONDS>(TIME_NOW.time_since_epoch()).count()
#define MILLISECONDS_SINCE_EPOCH \
    std::chrono::duration_cast<MILLISECONDS>(TIME_NOW.time_since_epoch()).count()
#define SECONDS_SINCE_EPOCH std::chrono::duration_cast<SECONDS>(TIME_NOW.time_since_epoch()).count()
