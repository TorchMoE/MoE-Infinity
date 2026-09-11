// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>

#include "model/model_topology.h"
#include "aio/archer_tensor_handle.h"

// Expert type enum
enum class ExpertType {
  NllbMoeDenseActDense = 2,
  FSGPTMoeDenseActDense = 3,
  MixtralMoeDenseActDense = 4,
  DeepSeekMoeDenseActDense = 5,
  GptOssMoeDenseActDense = 6
};

// Activation functions enum
enum class ActivationType { ReLU, GELU, SiLU, Identity };

// Base traits for expert architectures
template <ExpertType T>
struct ExpertTraits;

template <>
struct ExpertTraits<ExpertType::NllbMoeDenseActDense> {
  static constexpr size_t num_weights = 2;
  static constexpr size_t num_biases = 2;
  static constexpr std::array<const char*, 2> weight_names = {"fc1", "fc2"};
  static constexpr std::array<const char*, 2> bias_names = {"fc1_bias",
                                                            "fc2_bias"};
};

template <>
struct ExpertTraits<ExpertType::FSGPTMoeDenseActDense> {
  static constexpr size_t num_weights = 2;
  static constexpr size_t num_biases = 2;
  static constexpr std::array<const char*, 2> weight_names = {"fc1", "fc2"};
  static constexpr std::array<const char*, 2> bias_names = {"fc1_bias",
                                                            "fc2_bias"};
};

template <>
struct ExpertTraits<ExpertType::MixtralMoeDenseActDense> {
  static constexpr size_t num_weights = 3;
  static constexpr size_t num_biases = 0;
  static constexpr std::array<const char*, 3> weight_names = {"w1", "w2", "w3"};
  static constexpr std::array<const char*, 0> bias_names = {};
};

template <>
struct ExpertTraits<ExpertType::DeepSeekMoeDenseActDense> {
  static constexpr size_t num_weights = 3;
  static constexpr size_t num_biases = 0;
  static constexpr std::array<const char*, 3> weight_names = {
      "gate_proj", "up_proj", "down_proj"};
  static constexpr std::array<const char*, 0> bias_names = {};
};

// Templated expert implementation
template <ExpertType Type>
class Expert : public torch::nn::Module {
 private:
  using Traits = ExpertTraits<Type>;
  std::array<torch::Tensor, Traits::num_weights> weights_;
  std::array<torch::Tensor, Traits::num_biases> biases_;

 public:
  explicit Expert(int dtype) {
    auto tensor_dtype = dtype_to_torch(dtype);
    auto options =
        torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);

    // Register weights
    for (size_t i = 0; i < Traits::num_weights; ++i) {
      weights_[i] = register_parameter(Traits::weight_names[i],
                                       torch::zeros({1}, options));
    }

    // Register biases if any
    for (size_t i = 0; i < Traits::num_biases; ++i) {
      biases_[i] =
          register_parameter(Traits::bias_names[i], torch::zeros({1}, options));
    }
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        cudaStream_t stream = nullptr);

  void SetTensorsFromBlob(void* ptr,
                          const std::vector<std::uint32_t>& tensor_ids,
                          const torch::Device& device) {
    size_t idx = 0;

    // Set weights
    for (size_t i = 0; i < Traits::num_weights; ++i) {
      weights_[i] = kTensorIndex->find(tensor_ids[idx++])->second.tensor;
    }

    // Set biases
    for (size_t i = 0; i < Traits::num_biases; ++i) {
      biases_[i] = kTensorIndex->find(tensor_ids[idx++])->second.tensor;
    }
  }

  void SetModuleFromBlob(torch::jit::script::Module* ptr) {
    for (auto it = ptr->parameters().begin(); it != ptr->parameters().end();
         ++it) {
      // Set weights
      for (size_t i = 0; i < Traits::num_weights; ++i) {
        if ((*it).name() == Traits::weight_names[i]) {
          (*it).set_data(weights_[i]);
        }
      }

      // Set biases
      for (size_t i = 0; i < Traits::num_biases; ++i) {
        if ((*it).name() == Traits::bias_names[i]) {
          (*it).set_data(biases_[i]);
        }
      }
    }
  }
};

// Forward specializations
template <>
inline torch::Tensor Expert<ExpertType::NllbMoeDenseActDense>::forward(
    torch::Tensor hidden_states, cudaStream_t stream) {
  return torch::matmul(torch::relu(torch::matmul(hidden_states,
                                                 weights_[0].transpose(0, 1)) +
                                   biases_[0]),
                       weights_[1].transpose(0, 1)) +
         biases_[1];
}

template <>
inline torch::Tensor Expert<ExpertType::FSGPTMoeDenseActDense>::forward(
    torch::Tensor hidden_states, cudaStream_t stream) {
  if (hidden_states.dtype() != weights_[0].dtype()) {
    hidden_states = hidden_states.to(weights_[0].dtype());
  }
  return torch::matmul(torch::relu(torch::matmul(hidden_states,
                                                 weights_[0].transpose(0, 1)) +
                                   biases_[0]),
                       weights_[1].transpose(0, 1)) +
         biases_[1];
}

template <>
inline torch::Tensor Expert<ExpertType::MixtralMoeDenseActDense>::forward(
    torch::Tensor hidden_states, cudaStream_t stream) {
  return torch::matmul(
      torch::silu(torch::matmul(hidden_states, weights_[0].transpose(0, 1))) *
          torch::matmul(hidden_states, weights_[2].transpose(0, 1)),
      weights_[1].transpose(0, 1));
}

template <>
inline torch::Tensor Expert<ExpertType::DeepSeekMoeDenseActDense>::forward(
    torch::Tensor hidden_states, cudaStream_t stream) {
  return torch::matmul(
      torch::silu(torch::matmul(hidden_states, weights_[0].transpose(0, 1))) *
          torch::matmul(hidden_states, weights_[1].transpose(0, 1)),
      weights_[2].transpose(0, 1));
}

// Type aliases for compatibility
using NllbMoeDenseActDense = Expert<ExpertType::NllbMoeDenseActDense>;
using FSGPTMoEDenseActDense = Expert<ExpertType::FSGPTMoeDenseActDense>;
using MixtralMoEDenseActDense = Expert<ExpertType::MixtralMoeDenseActDense>;
using DeepSeekMoEDenseActDense = Expert<ExpertType::DeepSeekMoeDenseActDense>;
struct GptOssMoeDenseActDense : public torch::nn::Module {};

#ifndef EXPERT_TYPE
  #define EXPERT_TYPE 0
#endif

#define NLLB_MOE_DENSE_ACT_DENSE 2
#define FSGPT_MOE_DENSE_ACT_DENSE 3
#define MIXTRAL_MOE_DENSE_ACT_DENSE 4
#define DEEPSEEK_MOE_DENSE_ACT_DENSE 5
#define GPT_OSS_MOE_DENSE_ACT_DENSE 6

// forward declarations
torch::Tensor launch_fused_moe_ffn(torch::Tensor hidden,  // [M, K]
                                   torch::Tensor w1,      // [N, K]
                                   torch::Tensor w2,      // [N, K]
                                   torch::Tensor w3,      // [K, N]
                                   cudaStream_t stream);  // CUDA stream

struct MoEMLP : public torch::nn::Module {
  explicit MoEMLP(int dtype, int expert_type);
  torch::Tensor forward(torch::Tensor hidden_states, cudaStream_t stream,
                        cudaEvent_t kernel_start = nullptr,
                        cudaEvent_t kernel_stop = nullptr);

  void SetTensorsFromIds(const std::vector<std::uint32_t>& tensor_ids);
  void DequantMxfp4Params(cudaStream_t stream);
  void SetFp8Scales(const std::vector<torch::Tensor>& scales);
  void DequantFp8Params(cudaStream_t stream);

 private:
  void ForwardHelper(cudaStream_t stream);

 private:
  std::vector<torch::Tensor> buffer_;
  std::vector<torch::Tensor> param_;
  std::vector<torch::Tensor> gpt_oss_param_;

  at::cuda::CUDAGraph graph_;
  int warmup_count_ = 5;
  bool graph_mode_ = false;
  bool param_init_ = false;
  bool param_set_ = false;

  int dtype_;
  int expert_type_;

  std::vector<torch::Tensor> fp8_scales_;
  bool has_fp8_scales_ = false;
};

struct ExpertNode {
  NodePtr node;
  torch::nn::Module* module;
  void SetTensorsFromBlob(const torch::Device& device);
  int layer_idx;
  int expert_idx;
  int expert_type;
  torch::jit::script::Module* jit_module = nullptr;
};

typedef std::shared_ptr<ExpertNode> ExpertNodePtr;
