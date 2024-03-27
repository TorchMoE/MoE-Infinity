// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_aio_thread.h"

#include "utils/archer_logger.h"

ArcherAioThread::ArcherAioThread(int thread_id) : thread_id_(thread_id), is_running_(false)
{
    ARCHER_LOG_INFO("Create ArcherAioThread for thread: ", thread_id_);
}

ArcherAioThread::~ArcherAioThread() { Stop(); }

void ArcherAioThread::Start()
{
    if (is_running_) { return; }

    is_running_ = true;
    pending_callbacks_ = 0;
    thread_ = std::thread(&ArcherAioThread::Run, this);
}

void ArcherAioThread::Stop()
{
    if (!is_running_) { return; }

    is_running_ = false;
    thread_.join();
}

void ArcherAioThread::Enqueue(AioCallback& callback)
{
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.push_back(std::move(callback));
    pending_callbacks_.fetch_add(1);
}

void ArcherAioThread::Wait()
{
    // while (!callbacks_.empty()) { usleep(1000); }
    while (pending_callbacks_.load() != 0) { usleep(1000); }
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.clear();
}

void ArcherAioThread::Run()
{

    while (is_running_) {
        std::function<void()> callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (callbacks_.empty()) { continue; }
            callback = std::move(callbacks_.front());
            callbacks_.pop_front();
        }
        callback();
        pending_callbacks_.fetch_sub(1);
    }

}
