// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "archer_tensor_handle.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>
#include <torch/script.h>
#include <utility>
#include "common/pytorch.h"
#include "prefetch/task_scheduler.h"
#include "utils/logger.h"

const int c_block_size = 128 * 1024;
const int c_io_queue_depth = 8;

const char* ARCHER_PARAM_NAME = "archer_param";
const char* ARCHER_IHDEX_NAME = "archer_index";

std::unique_ptr<ArcherTensorHandle> kArcherTensorHandle(nullptr);

ArcherTensorHandle::ArcherTensorHandle(const std::string& prefix,
                                       int num_io_threads)
    : prefix_(prefix),
      prio_aio_handle_(prefix, num_io_threads),
      file_id_(0),
      file_offset_(0) {
  // InitLogger();

  if (prefix_.back() != '/') {
    prefix_ += '/';
  }

  struct stat st;
  if (stat(prefix_.c_str(), &st) != -1 && !S_ISDIR(st.st_mode)) {
    DLOG_FATAL("Invalid prefix: ", prefix_, " is not a directory");
  }
  if (stat(prefix_.c_str(), &st) == -1) {
    DLOG_WARN("Invalid prefix: ", prefix_, " does not exist, creating");
    mkdir(prefix_.c_str(), 0777);
  }

  DLOG_TRACE("Aio alignment size ", st.st_blksize);

  auto ckpt_index_path = prefix_ + std::string(ARCHER_IHDEX_NAME);
  if (access(ckpt_index_path.c_str(), F_OK) != -1) {
    DLOG_INFO("Loading index file from ", ckpt_index_path);
    kTensorIndex->Deserialize(ckpt_index_path.c_str());
    is_serialized_ = true;
  } else {
    DLOG_INFO("Index file", ckpt_index_path, " does not exist, creating");
  }
  DLOG_INFO("Index file size ", kTensorIndex->size());
}

void ArcherTensorHandle::StoreTensor(const std::uint32_t tensor_id,
                                     torch::Tensor& buffer) {
  auto it = kTensorIndex->find(tensor_id);
  bool tensor_exists = (it != kTensorIndex->end());

  std::unique_lock<std::mutex> lock(mutex_);
  TensorStorageMeta tensor_meta{file_id_, file_offset_, buffer.nbytes(),
                                buffer.sizes().vec()};
  tensor_meta.options = buffer.options();
  tensor_meta.id = tensor_id;

  auto num_bytes = buffer.nbytes();
  std::int64_t num_bytes_aligned =
      (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);

  if (tensor_exists) {
    // size must be the same if found
    if (it->second.size != buffer.nbytes()) {
      DLOG_FATAL("Tensor {} size mismatch {} != {}", tensor_id, it->second.size,
                 buffer.nbytes());
    }
    tensor_meta = it->second;
  }

  if (!tensor_exists) {
    // Check if this tensor would exceed the current partition
    if (file_offset_ + num_bytes_aligned > kPartitionSize) {
      file_id_++;
      file_offset_ = 0;
      DLOG_INFO("Storage partition full, switching to file_id=", file_id_);
    }
    tensor_meta.file_id = file_id_;
    tensor_meta.offset = file_offset_;
    file_offset_ += num_bytes_aligned;
  }

  kTensorIndex->insert(std::make_pair(tensor_id, tensor_meta));

  auto filename = GetIndexFileName(tensor_meta.file_id);

  lock.unlock();
  prio_aio_handle_.Write(filename, buffer.data_ptr(), false, tensor_meta.size,
                         tensor_meta.offset);
}

int64_t ArcherTensorHandle::GetTensorSizeAligned(
    const std::uint32_t tensor_id) const {
  auto it = kTensorIndex->find(tensor_id);
  if (it == kTensorIndex->end()) {
    DLOG_FATAL("Tensor not found", tensor_id);
  }
  auto num_bytes = it->second.size;
  std::int64_t num_bytes_aligned =
      (num_bytes + kAioAlignment - 1) & ~(kAioAlignment - 1);
  return num_bytes_aligned;
}

torch::TensorOptions ArcherTensorHandle::GetTensorOptions(
    const std::uint32_t tensor_id) const {
  auto it = kTensorIndex->find(tensor_id);
  if (it == kTensorIndex->end()) {
    DLOG_FATAL("Tensor not found", tensor_id);
  }
  return it->second.options;
}

void ArcherTensorHandle::SetTensor(std::uint32_t tensor_id,
                                   torch::Tensor& buffer,
                                   const torch::Device& device) {
  auto it = kTensorIndex->find(tensor_id);
  if (it == kTensorIndex->end()) {
    DLOG_FATAL("Tensor not found", tensor_id);
  }
  // FIXME: this is may creates extra copy of data, need to be confirmed
  // optimized CANNOT use shallow copy here, e.g., buffer =
  // it->second.tensor.to(DEFAULT_CUDA_DEVICE);

  buffer.set_data(it->second.tensor.to(device).to(buffer.dtype()));
}

void ArcherTensorHandle::SetTensor(std::uint32_t tensor_id,
                                   torch::Tensor& buffer) {
  auto it = kTensorIndex->find(tensor_id);
  if (it == kTensorIndex->end()) {
    DLOG_FATAL("Tensor not found", tensor_id);
  }
  if (buffer.dtype() != it->second.tensor.dtype()) {
    std::ostringstream oss;
    oss << buffer.dtype() << " -> " << it->second.tensor.dtype();
    DLOG_TRACE("Tensor dtype mismatch", tensor_id, oss.str());
    buffer.set_data(it->second.tensor.to(buffer.dtype()));
  } else {
    buffer.set_data(it->second.tensor);
  }
  DLOG_TRACE("Set tensor to device", tensor_id, buffer.device().str());
}

void ArcherTensorHandle::RegisterTensor(const std::uint32_t tensor_id,
                                        torch::Tensor& buffer) {
  auto it = kTensorIndex->find(tensor_id);
  if (it == kTensorIndex->end()) {
    DLOG_FATAL("Tensor not found", tensor_id);
  }

  tensor_to_id_.insert(std::make_pair((void*)buffer.data_ptr(), tensor_id));

  kTensorIndex->find(tensor_id)->second.tensor = buffer;
}

std::string ArcherTensorHandle::GetIndexFileName(
    const std::uint32_t file_id) const {
  return prefix_ + std::string(ARCHER_PARAM_NAME) + "_" +
         std::to_string(file_id);
}

std::uint32_t ArcherTensorHandle::GetTensorId(void* tensor) const {
  auto it = tensor_to_id_.find(tensor);
  if (it == tensor_to_id_.end()) {
    DLOG_FATAL("Tensor not found", (void*)tensor);
    return UINT32_MAX;
  }
  return it->second;
}

void ArcherTensorHandle::UpdateTensorMap(void* old_data_ptr,
                                         void* new_data_ptr) {
  auto it = tensor_to_id_.find(old_data_ptr);
  if (it == tensor_to_id_.end()) {
    DLOG_FATAL("Tensor ", (void*)old_data_ptr, " not found in tensor_to_id_");
    return;
  }
  auto tensor_id = it->second;
  tensor_to_id_.erase(it);

  auto it2 = kTensorIndex->find(tensor_id);
  if (it2 == kTensorIndex->end()) {
    DLOG_FATAL("Tensor not found in tensor_index_", tensor_id);
    return;
  }
  tensor_to_id_.insert(std::make_pair(new_data_ptr, tensor_id));
  // DLOG_TRACE("Update tensor {} with address {} to {}",
  //                  tensor_id,
  //                  (void*)old_data_ptr,
  //                  (void*)new_data_ptr);
}

void ArcherTensorHandle::ReadTensor(const uint32_t tensor_id, void* memory_ptr,
                                    bool on_demand) {
  auto it = kTensorIndex->find(tensor_id);
  if (it == kTensorIndex->end()) {
    DLOG_FATAL("Tensor not found", tensor_id);
  }

  auto tensor_meta = it->second;
  auto filename = GetIndexFileName(tensor_meta.file_id);

  prio_aio_handle_.Read(filename, memory_ptr, on_demand, tensor_meta.size,
                        tensor_meta.offset);
}

void ArcherTensorHandle::ReadBulk(const std::string& filename, void* memory_ptr,
                                  bool on_demand, std::int64_t num_bytes,
                                  std::int64_t offset) {
  prio_aio_handle_.Read(filename, memory_ptr, on_demand, num_bytes, offset);
}

static const char* ScalarTypeToDerivativeDtype(torch::ScalarType type) {
  switch (type) {
    case torch::kFloat8_e4m3fn:
      return "float8_e4m3fn";
    case torch::kFloat:
      return "float32";
    case torch::kBFloat16:
      return "bfloat16";
    case torch::kHalf:
      return "float16";
    case torch::kInt:
      return "int32";
    case torch::kByte:
      return "uint8";
    default:
      throw std::invalid_argument(
          "unsupported canonical dtype for derivative snapshot");
  }
}

torch::ScalarType ArcherTensorHandle::DerivativeDtypeToScalarType(
    const std::string& dtype) const {
  if (dtype == "float8_e4m3fn") return torch::kFloat8_e4m3fn;
  if (dtype == "float32") return torch::kFloat;
  if (dtype == "bfloat16") return torch::kBFloat16;
  if (dtype == "float16") return torch::kHalf;
  if (dtype == "int32") return torch::kInt;
  if (dtype == "uint8") return torch::kByte;
  throw std::invalid_argument("unknown derivative dtype: " + dtype);
}

std::vector<std::unordered_map<std::string, py::object>>
ArcherTensorHandle::GetCanonicalTensorIndexSnapshot() const {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<std::pair<std::uint32_t, const TensorStorageMeta*>> rows;
  for (const auto& entry : *kTensorIndex) {
    if (derivative_owned_ids_.count(entry.first) > 0) {
      continue;
    }
    rows.emplace_back(entry.first, &entry.second);
  }
  std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });

  std::vector<std::unordered_map<std::string, py::object>> snapshot;
  snapshot.reserve(rows.size());
  for (const auto& row : rows) {
    const TensorStorageMeta& meta = *row.second;
    std::unordered_map<std::string, py::object> py_row;
    py_row["tensor_id"] = py::cast(static_cast<std::int64_t>(row.first));
    py_row["dtype"] = py::cast(std::string(ScalarTypeToDerivativeDtype(
        c10::typeMetaToScalarType(meta.options.dtype()))));
    py_row["shape"] = py::cast(meta.shape);
    py_row["size"] = py::cast(static_cast<std::int64_t>(meta.size));
    py_row["file_id"] = py::cast(static_cast<std::int64_t>(meta.file_id));
    py_row["offset"] = py::cast(static_cast<std::int64_t>(meta.offset));
    snapshot.push_back(std::move(py_row));
  }
  return snapshot;
}

void ArcherTensorHandle::BeginDerivativeOverlay(
    const std::string& generation, std::int64_t canonical_max_tensor_id,
    std::int64_t canonical_max_file_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (overlay_active_) {
    throw std::runtime_error("derivative overlay already active");
  }
  overlay_active_ = true;
  overlay_generation_ = generation;
  overlay_canonical_max_tensor_id_ = canonical_max_tensor_id;
  overlay_canonical_max_file_id_ = canonical_max_file_id;
  overlay_staged_.clear();
}

void ArcherTensorHandle::RegisterDerivativeTensor(
    const std::string& generation, std::int64_t tensor_id, std::int64_t file_id,
    std::int64_t offset, std::int64_t size,
    const std::vector<std::int64_t>& shape, const std::string& dtype) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!overlay_active_ || overlay_generation_ != generation) {
    throw std::runtime_error("derivative overlay is not active");
  }
  if (tensor_id <= overlay_canonical_max_tensor_id_) {
    throw std::invalid_argument("derivative tensor_id not above canonical max");
  }
  if (file_id <= overlay_canonical_max_file_id_) {
    throw std::invalid_argument("derivative file_id not above canonical max");
  }
  if (tensor_id > UINT32_MAX || file_id > UINT32_MAX) {
    throw std::invalid_argument("derivative id exceeds uint32 range");
  }
  auto scalar_type = DerivativeDtypeToScalarType(dtype);
  auto id = static_cast<std::uint32_t>(tensor_id);
  if (kTensorIndex->find(id) != kTensorIndex->end() ||
      overlay_staged_.find(id) != overlay_staged_.end()) {
    throw std::invalid_argument("duplicate derivative tensor_id");
  }
  if (offset < 0 || (offset % kAioAlignment) != 0) {
    throw std::invalid_argument("derivative offset is not aligned");
  }
  for (const auto& staged : overlay_staged_) {
    const TensorStorageMeta& other = staged.second;
    if (other.file_id != static_cast<std::uint32_t>(file_id)) {
      continue;
    }
    std::int64_t other_begin = other.offset;
    std::int64_t other_end =
        other.offset + static_cast<std::int64_t>(other.size);
    if (offset < other_end && other_begin < offset + size) {
      throw std::invalid_argument("derivative file intervals overlap");
    }
  }
  TensorStorageMeta meta{static_cast<std::uint32_t>(file_id), offset,
                         static_cast<std::size_t>(size), shape};
  meta.options = torch::TensorOptions().dtype(scalar_type);
  meta.id = id;
  overlay_staged_.emplace(id, std::move(meta));
}

void ArcherTensorHandle::CommitDerivativeOverlay(
    const std::string& generation) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!overlay_active_ || overlay_generation_ != generation) {
    throw std::runtime_error("derivative overlay is not active");
  }
  for (auto& staged : overlay_staged_) {
    kTensorIndex->emplace(staged.first, staged.second);
    derivative_owned_ids_.insert(staged.first);
  }
  overlay_staged_.clear();
  overlay_active_ = false;
  overlay_generation_.clear();
}

void ArcherTensorHandle::AbortDerivativeOverlay(const std::string& generation) {
  std::unique_lock<std::mutex> lock(mutex_);
  overlay_staged_.clear();
  overlay_active_ = false;
  overlay_generation_.clear();
}
