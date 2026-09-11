// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "archer_prio_aio_handle.h"
#include "archer_tensor_index.h"
#include "base/noncopyable.h"

namespace py = pybind11;

extern const char* ARCHER_PARAM_NAME;
extern const char* ARCHER_IHDEX_NAME;

class ArcherTensorHandle : public base::noncopyable {
 public:
  static constexpr int64_t kPartitionSize = 10LL * 1024 * 1024 * 1024;

  explicit ArcherTensorHandle(const std::string& prefix,
                              int num_io_threads = 0);
  ~ArcherTensorHandle() = default;

  void StoreTensor(const std::uint32_t tensor_id, torch::Tensor& buffer);
  void RegisterTensor(std::uint32_t tensor_id, torch::Tensor& buffer);
  void SetTensor(std::uint32_t tensor_id, torch::Tensor& buffer);
  void SetTensor(std::uint32_t tensor_id, torch::Tensor& buffer,
                 const torch::Device& device);

  void ReadTensor(const std::uint32_t tensor_id, void* memory_ptr,
                  bool on_demand = false);
  void ReadBulk(const std::string& filename, void* memory_ptr, bool on_demand,
                std::int64_t num_bytes, std::int64_t offset);

  void MoveTensor(const std::uint32_t tensor_id,
                  const torch::Device& src_device,
                  const torch::Device& dst_device);

  std::uint32_t GetTensorId(void* tensor) const;
  void UpdateTensorMap(void* old_data_ptr, void* new_data_ptr);

  bool IsTensorIndexInitialized() const { return is_serialized_; }

  int64_t GetTensorSizeAligned(const std::uint32_t tensor_id) const;
  torch::TensorOptions GetTensorOptions(const std::uint32_t tensor_id) const;
  std::string GetIndexFileName(const std::uint32_t file_id) const;

  std::vector<std::unordered_map<std::string, py::object>>
  GetCanonicalTensorIndexSnapshot() const;
  void BeginDerivativeOverlay(const std::string& generation,
                              std::int64_t canonical_max_tensor_id,
                              std::int64_t canonical_max_file_id);
  void RegisterDerivativeTensor(const std::string& generation,
                                std::int64_t tensor_id, std::int64_t file_id,
                                std::int64_t offset, std::int64_t size,
                                const std::vector<std::int64_t>& shape,
                                const std::string& dtype);
  void CommitDerivativeOverlay(const std::string& generation);
  void AbortDerivativeOverlay(const std::string& generation);

 private:
  torch::ScalarType DerivativeDtypeToScalarType(const std::string& dtype) const;

  std::string prefix_;
  ArcherPrioAioHandle prio_aio_handle_;
  std::uint32_t file_id_;
  std::int64_t file_offset_;
  std::unordered_map<void*, std::uint32_t> tensor_to_id_;

  bool overlay_active_ = false;
  std::string overlay_generation_;
  std::int64_t overlay_canonical_max_tensor_id_ = 0;
  std::int64_t overlay_canonical_max_file_id_ = 0;
  std::unordered_map<std::uint32_t, TensorStorageMeta> overlay_staged_;
  std::unordered_set<std::uint32_t> derivative_owned_ids_;

  mutable std::mutex mutex_;

  bool is_serialized_ = false;
};

extern std::unique_ptr<ArcherTensorHandle> kArcherTensorHandle;
