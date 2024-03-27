// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <torch/extension.h>

#include "archer_prio_aio_handle.h"
#include "archer_tensor_index.h"
#include "utils/noncopyable.h"

extern const char* ARCHER_PARAM_NAME;
extern const char* ARCHER_IHDEX_NAME;

class ArcherTensorHandle : public noncopyable {
public:
    explicit ArcherTensorHandle(const std::string& prefix);
    ~ArcherTensorHandle() = default;

    void StoreTensor(const std::uint32_t tensor_id, torch::Tensor& buffer);
    void RegisterTensor(std::uint32_t tensor_id, torch::Tensor& buffer);
    void SetTensor(std::uint32_t tensor_id, torch::Tensor& buffer);
    void SetTensor(std::uint32_t tensor_id, torch::Tensor& buffer, const torch::Device& device);

    void ReadTensor(const std::uint32_t tensor_id, void* memory_ptr, bool on_demand = false);

    void MoveTensor(const std::uint32_t tensor_id,
                    const torch::Device& src_device,
                    const torch::Device& dst_device);

    std::uint32_t GetTensorId(void* tensor) const;
    void UpdateTensorMap(void* old_data_ptr, void* new_data_ptr);

    bool IsTensorIndexInitialized() const { return is_serialized_; }

    int64_t GetTensorSizeAligned(const std::uint32_t tensor_id) const;
    torch::TensorOptions GetTensorOptions(const std::uint32_t tensor_id) const;

private:
    // bool ValidateTensorMove(const std::uint32_t tensor_id,
    //                         const torch::Device& src_device,
    //                         const torch::Device& dst_device);
    std::string GetIndexFileName(const std::uint32_t file_id) const;

private:
    std::string prefix_;
    ArcherPrioAioHandle prio_aio_handle_;
    std::uint32_t file_id_;
    std::int64_t file_offset_;
    std::unordered_map<void*, std::uint32_t> tensor_to_id_;

    std::mutex mutex_;

    bool is_serialized_ = false;
};

extern std::unique_ptr<ArcherTensorHandle> kArcherTensorHandle;
