// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "expert_module.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "kernel/fused_moe_mlp.h"

void mxfp4_dequant_cuda(const void* packed, const void* scales, void* output,
                        int rows, int packed_cols, int scale_cols,
                        int block_size, cudaStream_t stream);
extern void fp8_dequant_blockwise_cuda(const void* weight, const void* scale,
                                       void* out, int N, int K,
                                       cudaStream_t stream);

static const int64_t kMaxTokens = 8192;

void ExpertNode::SetTensorsFromBlob(const torch::Device& device) {
  auto expert_type = static_cast<ExpertType>(this->expert_type);
  switch (expert_type) {
    case ExpertType::NllbMoeDenseActDense:
      reinterpret_cast<NllbMoeDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case ExpertType::FSGPTMoeDenseActDense:
      reinterpret_cast<FSGPTMoEDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case ExpertType::MixtralMoeDenseActDense:
      reinterpret_cast<MixtralMoEDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case ExpertType::DeepSeekMoeDenseActDense:
      reinterpret_cast<DeepSeekMoEDenseActDense*>(module)->SetTensorsFromBlob(
          node->device_memory_ptr, node->tensor_ids, device);
      break;
    case ExpertType::GptOssMoeDenseActDense:
      break;
    default:
      assert(false);
  }
}

MoEMLP::MoEMLP(int dtype, int expert_type) {
  auto tensor_dtype = dtype_to_torch(dtype);
  auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);

  expert_type_ = expert_type;
  dtype_ = dtype;

  for (int i = 0; i < 8; i++) {
    buffer_.push_back(torch::zeros({1}, options));
  }
  int num_params = expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE ? 6 : 4;
  for (int i = 0; i < num_params; i++) {
    param_.push_back(torch::zeros({1}, options));
  }
}

void MoEMLP::SetTensorsFromIds(const std::vector<std::uint32_t>& tensor_ids) {
  if (expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE) {
    DLOG_FATAL_IF(tensor_ids.size() != 6,
                  "GPT-OSS expert requires blocks/scales/bias for two "
                  "projections");
  }
  int device = at::cuda::current_device();
  auto options = torch::TensorOptions()
                     .dtype(dtype_to_torch(dtype_))
                     .device(CUDA_DEVICE(device));

  // Safety: exec_state is FETCHING/EXECUTING throughout forward(), so the
  // expert's GPU memory cannot be evicted until OutputFunc resets to IDLE.
  std::vector<std::vector<int64_t>> tensor_shapes;
  for (size_t i = 0; i < tensor_ids.size(); i++) {
    auto& tensor = kTensorIndex->find(tensor_ids[i])->second.tensor;
    tensor_shapes.push_back(tensor.sizes().vec());
    param_[i].set_data(tensor);
  }

  if (!param_init_) {
    auto allocator = c10::DeviceCachingAllocator::get(device);

    int64_t hdim;
    int64_t idim;
    if (expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE) {
      hdim = tensor_shapes[0][1] * 2;
      idim = tensor_shapes[0][0] / 2;
    } else {
      hdim = tensor_shapes[0][1];
      idim = tensor_shapes[0][0];
    }

    std::vector<std::vector<int64_t>> data_shapes;
    data_shapes.push_back({kMaxTokens, hdim});
    data_shapes.push_back({kMaxTokens, hdim});

    for (size_t i = 0; i < tensor_shapes.size(); i++) {
      data_shapes.push_back({kMaxTokens, idim});
    }

    for (size_t i = 0; i < data_shapes.size(); i++) {
      auto data_shape = data_shapes[i];
      auto data_size = torch_shape_size(data_shape, dtype_);
      void* buffer_ptr = allocator->allocate(data_size);
      buffer_[i].set_data(torch::from_blob(buffer_ptr, data_shape,
                                           DoNothingDeleter<void>{}, options));
      DLOG_TRACE("MoEMLP::SetTensorsFromIds: buffer_ tensor", i, "data_shape",
                 data_shape, "data_size", data_size, "device",
                 buffer_[i].device().str());
    }
    param_init_ = true;
  }

  if (!param_init_) {
    // Allocate computation buffers (input, output, intermediates) once.
    // These are reused across expert invocations.
    auto allocator = c10::DeviceCachingAllocator::get(device);

    // MLP tensor shape: weight is [intermediate, hidden], so
    //   hdim = tensor_shapes[0][1], idim = tensor_shapes[0][0]
    int64_t hdim;
    int64_t idim;
    if (expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE) {
      hdim = tensor_shapes[0][1] * 2;
      idim = tensor_shapes[0][0] / 2;
    } else {
      hdim = tensor_shapes[0][1];
      idim = tensor_shapes[0][0];
    }

    std::vector<std::vector<int64_t>> data_shapes;
    data_shapes.push_back({kMaxTokens, hdim});  // input buffer
    data_shapes.push_back({kMaxTokens, hdim});  // output buffer

    for (size_t i = 0; i < tensor_shapes.size(); i++) {
      data_shapes.push_back({kMaxTokens, idim});  // intermediate buffers
    }

    for (size_t i = 0; i < data_shapes.size(); i++) {
      auto data_shape = data_shapes[i];
      auto data_size = torch_shape_size(data_shape, dtype_);
      void* buffer_ptr = allocator->allocate(data_size);
      buffer_[i].set_data(torch::from_blob(buffer_ptr, data_shape,
                                           DoNothingDeleter<void>{}, options));
      DLOG_TRACE("MoEMLP::SetTensorsFromIds: buffer_ tensor", i, "data_shape",
                 data_shape, "data_size", data_size, "device",
                 buffer_[i].device().str());
    }
    param_init_ = true;
  }

  assert(param_init_ == true);
  assert(param_set_ == false);
  param_set_ = true;
}

void MoEMLP::DequantMxfp4Params(cudaStream_t stream) {
  if (expert_type_ != GPT_OSS_MOE_DENSE_ACT_DENSE) return;
  int device = at::cuda::current_device();
  gpt_oss_param_.clear();
  for (auto pair : {std::pair<int, int>{0, 1}, {3, 4}}) {
    auto packed = param_[pair.first].contiguous();
    auto scales = param_[pair.second].contiguous();
    DLOG_FATAL_IF(packed.scalar_type() != torch::kUInt8 ||
                      scales.scalar_type() != torch::kUInt8,
                  "GPT-OSS MXFP4 blocks/scales must be uint8");
    int rows = packed.size(0);
    int packed_cols = packed.size(1);
    int scale_cols = scales.size(1);
    int block_size = packed_cols * 2 / scale_cols;
    auto output =
        torch::empty({rows, packed_cols * 2}, torch::TensorOptions()
                                                  .dtype(torch::kBFloat16)
                                                  .device(CUDA_DEVICE(device)));
    mxfp4_dequant_cuda(packed.data_ptr(), scales.data_ptr(), output.data_ptr(),
                       rows, packed_cols, scale_cols, block_size, stream);
    gpt_oss_param_.push_back(output);
  }
  gpt_oss_param_.insert(gpt_oss_param_.begin() + 1, param_[2]);
  gpt_oss_param_.push_back(param_[5]);
}

torch::Tensor MoEMLP::forward(torch::Tensor hidden_states,
                              cudaStream_t stream) {
  DLOG_FATAL_IF(param_set_ == false, "param_set_ should be true");
  DLOG_FATAL_IF(param_init_ == false, "param_init_ should be true");

  auto& input_ = buffer_[0];
  auto& output_ = buffer_[1];

  int64_t batch_size = hidden_states.size(0);
  int64_t hdim = hidden_states.size(1);

  DLOG_FATAL_IF(batch_size > kMaxTokens || batch_size <= 0,
                "batch_size should be (0,", kMaxTokens, "] , but got",
                batch_size);
  TORCH_CHECK(hidden_states.scalar_type() == input_.scalar_type(),
              "hidden_states dtype must match expert input dtype");

  CUDA_CHECK(
      cudaMemcpyAsync(input_.data_ptr(), hidden_states.data_ptr(),
                      hidden_states.numel() * hidden_states.element_size(),
                      cudaMemcpyDeviceToDevice, stream));

  for (auto& buffer : buffer_) {
    auto shape_vec = buffer.sizes().vec();
    if (shape_vec.size() != 2) continue;
    int64_t row = buffer.size(0);
    int64_t col = buffer.size(1);
    auto dtype = buffer.dtype();
    if (row == kMaxTokens) {
      buffer.set_data(torch::from_blob(
          buffer.data_ptr(), {batch_size, col}, DoNothingDeleter<void>{},
          torch::TensorOptions().dtype(dtype).device(
              CUDA_DEVICE(at::cuda::current_device()))));
    }
  }

  ForwardHelper(stream);
  param_set_ = false;
  auto output = output_.clone();

  // Restore buffers to kMaxTokens shape for next invocation
  for (auto& buffer : buffer_) {
    auto shape_vec = buffer.sizes().vec();
    if (shape_vec.size() != 2) continue;
    int64_t row = buffer.size(0);
    int64_t col = buffer.size(1);
    auto dtype = buffer.dtype();

    if (row == batch_size && batch_size != kMaxTokens) {
      buffer.set_data(torch::from_blob(
          buffer.data_ptr(), {kMaxTokens, col}, DoNothingDeleter<void>{},
          torch::TensorOptions().dtype(dtype).device(
              CUDA_DEVICE(at::cuda::current_device()))));
    }
  }

  return output;
}

void MoEMLP::ForwardHelper(cudaStream_t stream) {
  torch::NoGradGuard no_grad;
  auto& input = buffer_[0];
  auto& output = buffer_[1];

  if (expert_type_ == GPT_OSS_MOE_DENSE_ACT_DENSE) {
    auto& gate_up_weight = gpt_oss_param_[0];
    auto& gate_up_bias = gpt_oss_param_[1];
    auto& down_weight = gpt_oss_param_[2];
    auto& down_bias = gpt_oss_param_[3];
    auto gate_up =
        torch::matmul(input, gate_up_weight.transpose(0, 1)) + gate_up_bias;
    auto gate =
        gate_up.index({torch::indexing::Slice(),
                       torch::indexing::Slice(0, torch::indexing::None, 2)});
    auto up =
        gate_up.index({torch::indexing::Slice(),
                       torch::indexing::Slice(1, torch::indexing::None, 2)});
    gate = torch::clamp_max(gate, 7.0);
    up = torch::clamp(up, -7.0, 7.0);
    auto activated = (up + 1.0) * (gate * torch::sigmoid(gate * 1.702));
    output.copy_(torch::matmul(activated, down_weight.transpose(0, 1)) +
                 down_bias);
    return;
  }

  if (expert_type_ == NLLB_MOE_DENSE_ACT_DENSE) {
    auto& fc1 = param_[0];
    auto& fc2 = param_[1];
    output.copy_(
        torch::matmul(torch::relu(torch::matmul(input, fc1.transpose(0, 1))),
                      fc2.transpose(0, 1)));
    return;
  }

  if (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE ||
      expert_type_ == MIXTRAL_MOE_DENSE_ACT_DENSE) {
    auto& gate_proj = param_[0];
    auto& up_proj =
        (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE) ? param_[1] : param_[2];
    auto& down_proj =
        (expert_type_ == DEEPSEEK_MOE_DENSE_ACT_DENSE) ? param_[2] : param_[1];

    auto& gate_out = buffer_[2];
    auto& fused_out = buffer_[3];

    fused_moe_ffn_into(input, gate_proj, up_proj, down_proj, gate_out,
                       fused_out, output, stream);
    return;
  }
  DLOG_FATAL("MoEMLP::forward: expert_type not supported", expert_type_);
}

void MoEMLP::SetFp8Scales(const std::vector<torch::Tensor>& scales) {
  fp8_scales_ = scales;
  has_fp8_scales_ = !scales.empty();
}

void MoEMLP::DequantFp8Params(cudaStream_t stream) {
  if (!has_fp8_scales_) return;
  int device = at::cuda::current_device();
  for (size_t i = 0; i < fp8_scales_.size() && i < param_.size(); ++i) {
    auto& w = param_[i];
    if (!w.defined()) continue;
    auto wdtype = w.scalar_type();
    if (wdtype != torch::kFloat8_e4m3fn && wdtype != torch::kUInt8) continue;
    auto& s = fp8_scales_[i];
    if (!s.defined()) continue;
    auto s_gpu = s.to(torch::kFloat32).to(CUDA_DEVICE(device));
    int N = w.size(0);
    int K = w.size(1);
    auto out = torch::empty({N, K}, torch::TensorOptions()
                                        .dtype(torch::kBFloat16)
                                        .device(CUDA_DEVICE(device)));
    auto w_u8 = w.view(torch::kUInt8).contiguous();
    fp8_dequant_blockwise_cuda(w_u8.data_ptr(), s_gpu.data_ptr(),
                               out.data_ptr(), N, K, stream);
    param_[i] = out;
  }
}
