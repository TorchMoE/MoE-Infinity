// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "stream_pool.h"

// Stream0 is used for H2D, Stream1 is used for Kernel, Stream2 is used for D2H
// CUDAStreamPool* kCUDAStreamPool = CUDAStreamPool::GetInstance();
std::unique_ptr<CUDAStreamPool> kCUDAStreamPool = std::make_unique<CUDAStreamPool>();
