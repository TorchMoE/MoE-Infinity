// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <functional>

typedef std::function<int()> AioCallback;

int ArcherOpenFile(const char* filename);
int ArcherCloseFile(const int fd);
int ArcherReadFileBatch(const int fd, void* buffer, const size_t num_bytes, const size_t offset);
int ArcherWriteFileBatch(const int fd,
                         const void* buffer,
                         const size_t num_bytes,
                         const size_t offset);
int ArcherReadFile(int fd, void* buffer, const size_t num_bytes, const size_t offset);
int ArcherWriteFile(int fd, const void* buffer, size_t num_bytes, size_t offset);
