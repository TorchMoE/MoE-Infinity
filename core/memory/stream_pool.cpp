// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "stream_pool.h"

// Stream0 is used for H2D, Stream1 is used for Kernel, Stream2 is used for D2H
// TorchStreamPool* kTorchStreamPool = TorchStreamPool::GetInstance();
std::unique_ptr<TorchStreamPool> kTorchStreamPool =
    std::make_unique<TorchStreamPool>();
