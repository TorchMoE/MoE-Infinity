// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_aio_utils.h"
#include <future>
#include "utils/archer_logger.h"
#include <string.h>
#include <cmath>

const size_t kBlockSize = 1 * 1024 * 1024;
const size_t kQueueDepth =
    std::thread::hardware_concurrency() / 4;  // set to 1/4 total number of cores in the system

int ArcherOpenFile(const char* filename)
{
    const int flags = (O_RDWR | O_CREAT | O_DIRECT);
    const int mode = 0660;
    const auto fd = open(filename, flags, mode);
    if (fd < 0) {
        ARCHER_LOG_FATAL("Failed to open file: ", filename);
        return -1;
    }
    return fd;
}

int ArcherCloseFile(const int fd)
{
    const auto ret = close(fd);
    if (ret < 0) {
        ARCHER_LOG_FATAL("Failed to close file: ", fd);
        return -1;
    }
    return 0;
}

int ArcherReadFileBatch(const int fd, void* buffer, const size_t num_bytes, const size_t offset)
{
    std::vector<std::future<ssize_t>> futures;
    const auto num_blocks = std::ceil(static_cast<double>(num_bytes) / kBlockSize);
    // const auto num_threads = std::thread::hardware_concurrency();

    for (auto i = 0; i < num_blocks; ++i) {
        const auto shift = i * kBlockSize;
        const auto xfer_buffer = (char*)buffer + shift;
        const auto xfer_offset = offset + shift;
        auto byte_count = kBlockSize;
        if ((shift + kBlockSize) > num_bytes) { byte_count = num_bytes - shift; }

        futures.emplace_back(
            std::async(std::launch::async, pread, fd, xfer_buffer, byte_count, xfer_offset));
    }

    for (auto& future : futures) {
        const auto ret = future.get();
        if (ret < 0) {
            ARCHER_LOG_FATAL("Failed to read file: ", fd);
            return -1;
        }
    }

    return 0;
}

int ArcherWriteFileBatch(const int fd,
                         const void* buffer,
                         const size_t num_bytes,
                         const size_t offset)
{
    std::vector<std::future<ssize_t>> futures;
    const auto num_blocks = std::ceil(static_cast<double>(num_bytes) / kBlockSize);

    for (auto i = 0; i < num_blocks; ++i) {
        const auto shift = i * kBlockSize;
        const auto xfer_buffer = (char*)buffer + shift;
        const auto xfer_offset = offset + shift;
        auto byte_count = kBlockSize;
        if ((shift + kBlockSize) > num_bytes) { byte_count = num_bytes - shift; }

        futures.emplace_back(
            std::async(std::launch::async, pwrite, fd, xfer_buffer, byte_count, xfer_offset));
    }

    for (auto& future : futures) {
        const auto ret = future.get();
        if (ret < 0) {
            ARCHER_LOG_FATAL(
                "Failed to write file: ", fd,", errno: ", errno,", msg: ", strerror(errno));
            return -1;
        }
    }

    return 0;
}

int ArcherReadFile(int fd, void* buffer, const size_t num_bytes, const size_t offset)
{
    const auto ret = pread(fd, buffer, num_bytes, offset);
    if (ret < 0) {
        ARCHER_LOG_FATAL("Failed to read file: ", fd,", errno: ", errno,", msg: ", strerror(errno));
        return -1;
    }

    return 0;
}

int ArcherWriteFile(int fd, const void* buffer, size_t num_bytes, size_t offset)
{
    const auto ret = pwrite(fd, buffer, num_bytes, offset);
    if (ret < 0) {
        ARCHER_LOG_FATAL(
            "Failed to write file: ", fd,", errno: ", errno,", msg: ", strerror(errno));
        return -1;
    }

    return 0;
}
