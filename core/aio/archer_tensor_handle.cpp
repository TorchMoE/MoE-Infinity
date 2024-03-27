// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_tensor_handle.h"

#include <cuda_runtime_api.h>

#include <torch/script.h>
#include "common/pytorch.h"
#include "prefetch/task_scheduler.h"
#include "utils/archer_logger.h"

const int c_block_size = 128 * 1024;
const int c_io_queue_depth = 8;

const char* ARCHER_PARAM_NAME = "archer_param";
const char* ARCHER_IHDEX_NAME = "archer_index";

std::unique_ptr<ArcherTensorHandle> kArcherTensorHandle(nullptr);

ArcherTensorHandle::ArcherTensorHandle(const std::string& prefix)
    : prefix_(prefix), prio_aio_handle_(prefix), file_id_(0), file_offset_(0)
{
    InitLogger();

    if (prefix_.back() != '/') { prefix_ += '/'; }

    struct stat st;
    if (stat(prefix_.c_str(), &st) != -1 && !S_ISDIR(st.st_mode)) {
        ARCHER_LOG_FATAL("Invalid prefix: ", prefix_, " is not a directory");
    }
    if (stat(prefix_.c_str(), &st) == -1) {
        ARCHER_LOG_WARN("Invalid prefix: ", prefix_," does not exist, creating");
        mkdir(prefix_.c_str(), 0777);
    }

    ARCHER_LOG_DEBUG("Aio alignment size ", st.st_blksize);

    auto ckpt_index_path = prefix_ + std::string(ARCHER_IHDEX_NAME);
    if (access(ckpt_index_path.c_str(), F_OK) != -1) {
        ARCHER_LOG_INFO("Loading index file from ", ckpt_index_path);
        kTensorIndex->Deserialize(ckpt_index_path.c_str());
        is_serialized_ = true;
    } else {
        ARCHER_LOG_INFO("Index file", ckpt_index_path," does not exist, creating");
    }
    ARCHER_LOG_INFO("Index file size ", kTensorIndex->size());
}

void ArcherTensorHandle::StoreTensor(const std::uint32_t tensor_id, torch::Tensor& buffer)
{
    auto it = kTensorIndex->find(tensor_id);
    bool tensor_exists = (it != kTensorIndex->end());

    std::unique_lock<std::mutex> lock(mutex_);
    TensorStorageMeta tensor_meta{file_id_, file_offset_, buffer.nbytes(), buffer.sizes().vec()};
    tensor_meta.options = buffer.options();
    tensor_meta.id = tensor_id;

    auto num_bytes = buffer.nbytes();
    std::int64_t num_bytes_aligned = (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);

    if (tensor_exists) {
        // size must be the same if found
        if (it->second.size != buffer.nbytes()) {
            ARCHER_LOG_FATAL(
                "Tensor {} size mismatch {} != {}", tensor_id, it->second.size, buffer.nbytes());
        }
        tensor_meta = it->second;
    }

    file_offset_ += tensor_exists ? 0 : num_bytes_aligned;

    kTensorIndex->insert(std::make_pair(tensor_id, tensor_meta));

    auto filename = GetIndexFileName(tensor_meta.file_id);

    lock.unlock();
    prio_aio_handle_.Write(
        filename, buffer.data_ptr(), false, tensor_meta.size, tensor_meta.offset);
}

int64_t ArcherTensorHandle::GetTensorSizeAligned(const std::uint32_t tensor_id) const
{
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) { ARCHER_LOG_FATAL("Tensor not found", tensor_id); }
    auto num_bytes = it->second.size;
    std::int64_t num_bytes_aligned = (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);
    return num_bytes_aligned;
}

torch::TensorOptions ArcherTensorHandle::GetTensorOptions(const std::uint32_t tensor_id) const
{
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) { ARCHER_LOG_FATAL("Tensor not found", tensor_id); }
    return it->second.options;
}

void ArcherTensorHandle::SetTensor(std::uint32_t tensor_id,
                                   torch::Tensor& buffer,
                                   const torch::Device& device)
{
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) { ARCHER_LOG_FATAL("Tensor not found", tensor_id); }
    // FIXME: this is may creates extra copy of data, need to be confirmed optimized
    // CANNOT use shallow copy here, e.g., buffer = it->second.tensor.to(DEFAULT_CUDA_DEVICE);

    buffer.set_data(it->second.tensor.to(device).to(buffer.dtype()));
}

void ArcherTensorHandle::SetTensor(std::uint32_t tensor_id, torch::Tensor& buffer)
{
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) { ARCHER_LOG_FATAL("Tensor not found", tensor_id); }
    if (buffer.dtype() != it->second.tensor.dtype()) {
        std::ostringstream oss;
        oss << buffer.dtype() << " -> " << it->second.tensor.dtype();
        ARCHER_LOG_DEBUG("Tensor dtype mismatch", tensor_id, oss.str());
        buffer.set_data(it->second.tensor.to(buffer.dtype()));
    } else {
        buffer.set_data(it->second.tensor);
    }
    ARCHER_LOG_DEBUG("Set tensor to device", tensor_id, buffer.device().str());
}

void ArcherTensorHandle::RegisterTensor(const std::uint32_t tensor_id, torch::Tensor& buffer)
{
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) { ARCHER_LOG_FATAL("Tensor not found", tensor_id); }

    tensor_to_id_.insert(std::make_pair((void*)buffer.data_ptr(), tensor_id));

    kTensorIndex->find(tensor_id)->second.tensor = buffer;
}

std::string ArcherTensorHandle::GetIndexFileName(const std::uint32_t file_id) const
{
    return prefix_ + std::string(ARCHER_PARAM_NAME) + "_" + std::to_string(file_id);
}

std::uint32_t ArcherTensorHandle::GetTensorId(void* tensor) const
{
    auto it = tensor_to_id_.find(tensor);
    if (it == tensor_to_id_.end()) {
        ARCHER_LOG_FATAL("Tensor not found", (void*)tensor);
        return UINT32_MAX;
    }
    return it->second;
}

void ArcherTensorHandle::UpdateTensorMap(void* old_data_ptr, void* new_data_ptr)
{
    auto it = tensor_to_id_.find(old_data_ptr);
    if (it == tensor_to_id_.end()) {
        ARCHER_LOG_FATAL("Tensor ", (void*)old_data_ptr, " not found in tensor_to_id_");
        return;
    }
    auto tensor_id = it->second;
    tensor_to_id_.erase(it);

    auto it2 = kTensorIndex->find(tensor_id);
    if (it2 == kTensorIndex->end()) {
        ARCHER_LOG_FATAL("Tensor not found in tensor_index_", tensor_id);
        return;
    }
    tensor_to_id_.insert(std::make_pair(new_data_ptr, tensor_id));
    // ARCHER_LOG_DEBUG("Update tensor {} with address {} to {}",
    //                  tensor_id,
    //                  (void*)old_data_ptr,
    //                  (void*)new_data_ptr);
}

void ArcherTensorHandle::ReadTensor(const uint32_t tensor_id, void* memory_ptr, bool on_demand)
{
    auto it = kTensorIndex->find(tensor_id);
    if (it == kTensorIndex->end()) { ARCHER_LOG_FATAL("Tensor not found", tensor_id); }

    auto tensor_meta = it->second;
    auto filename = GetIndexFileName(tensor_meta.file_id);

    prio_aio_handle_.Read(filename, memory_ptr, on_demand, tensor_meta.size, tensor_meta.offset);
}
