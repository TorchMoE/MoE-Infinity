// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_prio_aio_handle.h"

#include <cuda_runtime_api.h>

#include "archer_aio_utils.h"
#include "utils/archer_logger.h"
#include "utils/cuda_utils.h"

const int kBlockSize = 1024 * 1024;
const int kQueueDepth = 32;

ArcherPrioAioHandle::ArcherPrioAioHandle(const std::string& prefix)
    : aio_context_(kBlockSize), time_to_exit_(false)
{
    InitLogger();
    thread_ = std::thread(&ArcherPrioAioHandle::Run, this);
}

void ArcherPrioAioHandle::Run()
{
    while (!time_to_exit_) { aio_context_.Schedule(); }
}

ArcherPrioAioHandle::~ArcherPrioAioHandle()
{
    time_to_exit_ = true;
    thread_.join();
    for (auto& file : file_set_) { close(file.second); }
}

std::int64_t ArcherPrioAioHandle::Read(const std::string& filename,
                                       void* buffer,
                                       const bool high_prio,
                                       const std::int64_t num_bytes,
                                       const std::int64_t offset)
{
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(file_set_mutex_);
        auto file = file_set_.find(filename);
        if (file == file_set_.end()) {
            fd = ArcherOpenFile(filename.c_str());
            file_set_.insert(std::make_pair(filename, fd));
        }
        fd = file_set_[filename];
    }

    std::int64_t num_bytes_aligned = (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);

    auto callbacks =
        aio_context_.PrepIocbs(true, buffer, fd, kBlockSize, offset, num_bytes_aligned);
    auto io_request = std::make_shared<struct AioRequest>();
    io_request->callbacks = std::move(callbacks);
    io_request->pending_callbacks.store(io_request->callbacks.size());
    aio_context_.AcceptRequest(io_request, high_prio);

    {
        std::unique_lock<std::mutex> lock(io_request->mutex);
        io_request->cv.wait(lock,
                            [&io_request] { return io_request->pending_callbacks.load() == 0; });
    }

    return num_bytes_aligned;
}

std::int64_t ArcherPrioAioHandle::Write(const std::string& filename,
                                        const void* buffer,
                                        const bool high_prio,
                                        const std::int64_t num_bytes,
                                        const std::int64_t offset)
{
    int fd = -1;
    {
        std::lock_guard<std::mutex> lock(file_set_mutex_);
        auto file = file_set_.find(filename);
        if (file == file_set_.end()) {
            fd = ArcherOpenFile(filename.c_str());
            file_set_.insert(std::make_pair(filename, fd));
        }
        fd = file_set_[filename];
    }

    std::int64_t num_bytes_aligned = (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);
    void* write_buffer = nullptr;

    auto mem_type = IsDevicePointer(buffer) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
    cudaHostAlloc(&write_buffer, num_bytes_aligned, cudaHostAllocDefault);
    cudaMemcpy(write_buffer, buffer, num_bytes, mem_type);
    auto callbacks =
        aio_context_.PrepIocbs(false, write_buffer, fd, kBlockSize, offset, num_bytes_aligned);
    auto io_request = std::make_shared<struct AioRequest>();
    io_request->callbacks = std::move(callbacks);
    io_request->pending_callbacks.store(io_request->callbacks.size());
    aio_context_.AcceptRequest(io_request, high_prio);

    {
        std::unique_lock<std::mutex> lock(io_request->mutex);
        io_request->cv.wait(lock,
                            [&io_request] { return io_request->pending_callbacks.load() == 0; });
    }

    cudaFreeHost(write_buffer);
    return num_bytes_aligned;
}

ArcherPrioAioContext::ArcherPrioAioContext(const int block_size)
    : block_size_(block_size)
{
    thread_pool_ = std::make_unique<ArcherAioThreadPool>(1);  // only one SSD device
    thread_pool_->Start();
}

ArcherPrioAioContext::~ArcherPrioAioContext() {}

void ArcherPrioAioContext::Schedule()
{
    std::shared_ptr<AioRequest> io_request = nullptr;

    {
        std::lock_guard<std::mutex> lock(io_queue_high_mutex_);
        if (!io_queue_high_.empty()) {
            io_request = io_queue_high_.front();
            io_queue_high_.pop_front();
        }
    }

    if (io_request != nullptr) {
        for (auto& cb : io_request->callbacks) {
            thread_pool_->Enqueue(cb, 0);  // only SSD device 0 is feasible here
        }
        thread_pool_->Wait();
        io_request->callbacks.clear();
        io_request->pending_callbacks.store(0);
        io_request->cv.notify_one();
        return;
    }

    AioCallback cb = nullptr;
    {
        std::lock_guard<std::mutex> lock(io_queue_low_mutex_);
        if (!io_queue_low_.empty()) {
            io_request = io_queue_low_.front();
            cb = std::move(io_request->callbacks.back());
            io_request->callbacks.pop_back();
            if (io_request->callbacks.empty()) { io_queue_low_.pop_front(); }
        }
    }

    if (cb == nullptr) { return; }

    thread_pool_->Enqueue(cb, 0);  // only SSD device 0 is feasible here
    thread_pool_->Wait();
    io_request->pending_callbacks.fetch_sub(1);

    if (io_request->pending_callbacks.load() == 0) { io_request->cv.notify_one(); }
}

void ArcherPrioAioContext::AcceptRequest(std::shared_ptr<AioRequest>& io_request, bool high_prio)
{
    std::lock_guard<std::mutex> lock(high_prio ? io_queue_high_mutex_ : io_queue_low_mutex_);
    if (high_prio) {
        io_queue_high_.push_back(io_request);
    } else {
        io_queue_low_.push_back(io_request);
    }
}

std::vector<AioCallback> ArcherPrioAioContext::PrepIocbs(const bool read_op,
                                                         void* buffer,
                                                         const int fd,
                                                         const int block_size,
                                                         const std::int64_t offset,
                                                         const std::int64_t total_size)
{
    const auto n_blocks = total_size / static_cast<std::int64_t>(block_size);
    const auto last_block_size = total_size % static_cast<std::int64_t>(block_size);
    const auto n_iocbs = n_blocks + (last_block_size > 0 ? 1 : 0);

    std::vector<AioCallback> callbacks;

    for (auto i = 0; i < n_iocbs; ++i) {
        const std::int64_t shift = i * static_cast<std::int64_t>(block_size);
        const auto xfer_buffer = static_cast<char*>(buffer) + shift;
        const std::int64_t xfer_offset = offset + shift;
        auto byte_count = static_cast<std::int64_t>(block_size);
        if ((shift + block_size) > total_size) { byte_count = total_size - shift; }
        if (read_op) {
            auto cb = std::bind(ArcherReadFile, fd, xfer_buffer, byte_count, xfer_offset);
            callbacks.push_back(std::move(cb));
        } else {
            auto cb = std::bind(ArcherWriteFile, fd, xfer_buffer, byte_count, xfer_offset);
            callbacks.push_back(std::move(cb));
        }
    }

    return callbacks;
}
