// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <torch/extension.h>
#include "aio/archer_prio_aio_handle.h"
#include "base/noncopyable.h"

#define CPU_DEVICE torch::Device(torch::kCPU)
#define CUDA_DEVICE(index) torch::Device(torch::kCUDA, index)
#define DISK_DEVICE torch::Device(torch::kMeta)
#define DEFAULT_CUDA_DEVICE torch::Device(torch::kCUDA, 0)

#define TENSOR_OPTIONS(dtype, target) \
  torch::TensorOptions()              \
      .dtype(dtype)                   \
      .device(target)                 \
      .requires_grad(false)           \
      .memory_format(torch::MemoryFormat::Contiguous)

#define FLOAT32_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kFloat32, target)
#define FLOAT16_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kFloat16, target)
#define INT32_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kInt32, target)
#define INT64_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kInt64, target)
#define BFLOAT16_TENSOR_OPTIONS(target) TENSOR_OPTIONS(torch::kBFloat16, target)

#define FAKE_TENSOR_SIZES torch::IntArrayRef({1})

inline std::vector<uint32_t> list_to_vector(py::list list) {
  std::vector<uint32_t> vec;
  for (auto item : list) {
    vec.push_back(item.cast<uint32_t>());
  }
  return vec;
}

inline py::list vector_to_list(std::vector<uint32_t>& vec) {
  py::list list;
  for (auto item : vec) {
    list.append(item);
  }
  return list;
}

inline size_t torch_shape_size(const std::vector<int64_t>& shape, int dtype) {
  auto torch_type = torch::ScalarType(dtype);
  auto itemsize = torch::empty({1}, torch_type).itemsize();
  size_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  size *= itemsize;
  return size;
}
