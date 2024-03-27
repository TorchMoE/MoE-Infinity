// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <vector>

#include "archer_aio_thread.h"

class ArcherAioThreadPool {
public:
    explicit ArcherAioThreadPool(int num_threads);
    ~ArcherAioThreadPool();

    void Start();
    void Stop();

    void Enqueue(AioCallback& callback, int thread_id = -1);
    void Wait();

private:
    int num_threads_;
    std::vector<std::unique_ptr<ArcherAioThread>> threads_;
};
