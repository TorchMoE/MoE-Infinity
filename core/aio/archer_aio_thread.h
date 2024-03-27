// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <atomic>
#include <functional>
#include <list>
#include <mutex>
#include <thread>

#include "archer_aio_utils.h"

class ArcherAioThread {
public:
    explicit ArcherAioThread(int thread_id);
    ~ArcherAioThread();

    void Start();
    void Stop();

    void Enqueue(AioCallback& callback);
    void Wait();

private:
    void Run();

private:
    int thread_id_;
    std::thread thread_;
    bool is_running_;

    std::list<AioCallback> callbacks_;

    std::mutex mutex_;
    std::atomic<int> pending_callbacks_;
};
