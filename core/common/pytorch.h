// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <torch/extension.h>
#include "aio/archer_prio_aio_handle.h"
#include "utils/noncopyable.h"

#define CPU_DEVICE torch::Device(torch::kCPU)
#define CUDA_DEVICE(index) torch::Device(torch::kCUDA, index)
#define DISK_DEVICE torch::Device(torch::kMeta)
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)

#define FLOAT32_TENSOR_OPTIONS(target) torch::TensorOptions().dtype(torch::kFloat32).device(target)
#define FLOAT16_TENSOR_OPTIONS(target) torch::TensorOptions().dtype(torch::kFloat16).device(target)
#define FAKE_TENSOR_SIZES torch::IntArrayRef({1})

inline std::vector<uint32_t> list_to_vector(py::list list)
{
    std::vector<uint32_t> vec;
    for (auto item : list) { vec.push_back(item.cast<uint32_t>()); }
    return vec;
}

inline py::list vector_to_list(std::vector<uint32_t>& vec)
{
    py::list list;
    for (auto item : vec) { list.append(item); }
    return list;
}
