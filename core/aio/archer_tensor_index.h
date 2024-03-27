// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <torch/extension.h>
#include <torch/script.h>
#include <cstdint>
#include <istream>
#include <ostream>
#include <sstream>
#include <unordered_map>

#include "common/pytorch.h"
#include "common/types.h"
#include "utils/noncopyable.h"

static const std::uint32_t kTensorIndexVersion = 1;

struct TensorStorageMeta {
    std::uint32_t file_id;
    std::int64_t offset;
    std::size_t size;
    std::vector<std::int64_t> shape;
    torch::TensorOptions options;
    TensorID id;

    // not for serialization
    torch::Tensor tensor;
    torch::Device device = DISK_DEVICE;

    std::string DebugString() const;
};

std::ostream& operator<<(std::ostream& os, const TensorStorageMeta& obj);
std::istream& operator>>(std::istream& is, TensorStorageMeta& obj);
void write_options(std::ostream& os, const torch::TensorOptions& obj);
void read_options(std::istream& is, torch::TensorOptions& obj);

class ArcherTensorIndex : public std::unordered_map<uint32_t, TensorStorageMeta>,
                          public noncopyable {
public:
    void Serialize(const char* path);
    void Deserialize(const char* path);

    STATIC_GET_INSTANCE(ArcherTensorIndex)

private:
    ArcherTensorIndex() = default;
    ~ArcherTensorIndex() = default;
};

extern ArcherTensorIndex* kTensorIndex;
