// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include <torch/torch.h>
#include "model/model_topology.h"

#ifndef EXPERT_TYPE
#define EXPERT_TYPE 0
#endif

#define SWITCH_TRANSFORMERS_DENSE_ACT_DENSE 0
#define SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE 1
#define NLLB_MOE_DENSE_ACT_DENSE 2
#define FSGPT_MOE_DENSE_ACT_DENSE 3
#define MIXTRAL_MOE_DENSE_ACT_DENSE 4

#define DTYPE_BFLOAT16 0
#define DTYPE_FLOAT32 1
#define DTYPE_FLOAT16 2

struct ModuleUtils {
    virtual void SetTensorsFromBlob(void* ptr,
                                    const std::vector<std::uint32_t>& tensor_ids,
                                    const torch::Device& device) = 0;
};

struct SwitchTransformersDenseActDense : public torch::nn::Module, public ModuleUtils {
    SwitchTransformersDenseActDense(int dtype);
    torch::Tensor forward(torch::Tensor hidden_states);
    torch::Tensor wi, wo;

    void SetTensorsFromBlob(void* ptr,
                            const std::vector<std::uint32_t>& tensor_ids,
                            const torch::Device& device) override;
};

struct SwitchTransformersDenseGatedActDense : public torch::nn::Module, public ModuleUtils {
    SwitchTransformersDenseGatedActDense(int dtype);
    torch::Tensor forward(torch::Tensor hidden_states);
    torch::Tensor wi_0, wi_1, wo;

    void SetTensorsFromBlob(void* ptr,
                            const std::vector<std::uint32_t>& tensor_ids,
                            const torch::Device& device) override;
};

struct NllbMoeDenseActDense : public torch::nn::Module, public ModuleUtils {
    NllbMoeDenseActDense(int dtype);
    torch::Tensor forward(torch::Tensor hidden_states);
    torch::Tensor fc1, fc2;
    torch::Tensor fc1_bias, fc2_bias;

    void SetTensorsFromBlob(void* ptr,
                            const std::vector<std::uint32_t>& tensor_ids,
                            const torch::Device& device) override;
};

struct FSGPTMoEDenseActDense : public torch::nn::Module, public ModuleUtils {
    FSGPTMoEDenseActDense(int dtype);
    torch::Tensor forward(torch::Tensor hidden_states);
    torch::Tensor fc1, fc2;
    torch::Tensor fc1_bias, fc2_bias;

    void SetTensorsFromBlob(void* ptr,
                            const std::vector<std::uint32_t>& tensor_ids,
                            const torch::Device& device) override;
};

struct MixtralMoEDenseActDense : public torch::nn::Module, public ModuleUtils {
    MixtralMoEDenseActDense(int dtype);
    torch::Tensor forward(torch::Tensor hidden_states);
    torch::Tensor w1, w2, w3;

    void SetTensorsFromBlob(void* ptr,
                            const std::vector<std::uint32_t>& tensor_ids,
                            const torch::Device& device) override;
};

struct ExpertNode {
    NodePtr node;
    torch::nn::Module* module;
    void SetTensorsFromBlob(const torch::Device& device);
    int layer_idx;
    int expert_idx;
    int expert_type;
};

typedef std::shared_ptr<ExpertNode> ExpertNodePtr;

inline torch::ScalarType dtype_to_torch(int dtype)
{
    auto tensor_dtype = torch::kFloat32;
    switch (dtype) {
        case DTYPE_BFLOAT16: tensor_dtype = torch::kBFloat16; break;
        case DTYPE_FLOAT16: tensor_dtype = torch::kHalf; break;
        case DTYPE_FLOAT32: tensor_dtype = torch::kFloat32; break;
        default: assert(false);
    }
    return tensor_dtype;
}

inline int torch_dtype_to_int(torch::ScalarType dtype)
{
    auto tensor_dtype = DTYPE_FLOAT32;
    switch (dtype) {
        case torch::kBFloat16: tensor_dtype = DTYPE_BFLOAT16; break;
        case torch::kHalf: tensor_dtype = DTYPE_FLOAT16; break;
        case torch::kFloat32: tensor_dtype = DTYPE_FLOAT32; break;
        default: assert(false);
    }
    return tensor_dtype;
}
