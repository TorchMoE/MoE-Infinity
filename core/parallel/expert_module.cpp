// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "expert_module.h"
#include "aio/archer_tensor_handle.h"
#include "utils/archer_logger.h"

SwitchTransformersDenseActDense::SwitchTransformersDenseActDense(int dtype)
{
    // auto tensor_dtype = dtype_to_torch(dtype);
    auto options = torch::TensorOptions().device(torch::kCPU);
    wi = register_parameter("wi", torch::zeros({1}, options));
    wo = register_parameter("wo", torch::zeros({1}, options));
}

void SwitchTransformersDenseActDense::SetTensorsFromBlob(
    void *ptr,
    const std::vector<std::uint32_t> &tensor_ids,
    const torch::Device &device)
{
    wi = kTensorIndex->find(tensor_ids[0])->second.tensor;
    wo = kTensorIndex->find(tensor_ids[1])->second.tensor;
}

torch::Tensor SwitchTransformersDenseActDense::forward(torch::Tensor hidden_states)
{
    // ARCHER_LOG_DEBUG("SwitchTransformersDenseActDense wi {} wo {}, hidden_states {}",
    //                  torch_dtype_to_int(wi.dtype()),
    //                     torch_dtype_to_int(wo.dtype()),
    //                     torch_dtype_to_int(hidden_states.dtype()));
    // ARCHER_LOG_DEBUG("SwitchTransformersDenseActDense wi {} wo {}, hidden_states {}",
    return torch::matmul(
        torch::relu(torch::matmul(hidden_states, wi.transpose(0, 1).to(hidden_states.dtype()))),
        wo.transpose(0, 1).to(hidden_states.dtype()));
}

SwitchTransformersDenseGatedActDense::SwitchTransformersDenseGatedActDense(int dtype)
{
    auto tensor_dtype = dtype_to_torch(dtype);
    auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
    wi_0 = register_parameter("wi_0", torch::zeros({1}, options));
    wi_1 = register_parameter("wi_1", torch::zeros({1}, options));
    wo = register_parameter("wo", torch::zeros({1}));
}

void SwitchTransformersDenseGatedActDense::SetTensorsFromBlob(
    void *ptr,
    const std::vector<std::uint32_t> &tensor_ids,
    const torch::Device &device)
{
    wi_0 = kTensorIndex->find(tensor_ids[0])->second.tensor;
    wi_1 = kTensorIndex->find(tensor_ids[1])->second.tensor;
    wo = kTensorIndex->find(tensor_ids[2])->second.tensor;
}

torch::Tensor SwitchTransformersDenseGatedActDense::forward(torch::Tensor hidden_states)
{
    auto gate = torch::gelu(torch::matmul(hidden_states, wi_0.transpose(0, 1)));
    auto linear = torch::matmul(hidden_states, wi_1.transpose(0, 1));
    return torch::matmul(torch::mul(gate, linear), wo.transpose(0, 1));
}

NllbMoeDenseActDense::NllbMoeDenseActDense(int dtype)
{
    auto tensor_dtype = dtype_to_torch(dtype);
    auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
    fc1 = register_parameter("fc1", torch::zeros({1}, options));
    fc2 = register_parameter("fc2", torch::zeros({1}, options));
    fc1_bias = register_parameter("fc1_bias", torch::zeros({1}, options));
    fc2_bias = register_parameter("fc2_bias", torch::zeros({1}, options));
}

void NllbMoeDenseActDense::SetTensorsFromBlob(void *ptr,
                                              const std::vector<std::uint32_t> &tensor_ids,
                                              const torch::Device &device)
{
    fc1 = kTensorIndex->find(tensor_ids[0])->second.tensor;
    fc1_bias = kTensorIndex->find(tensor_ids[1])->second.tensor;
    fc2 = kTensorIndex->find(tensor_ids[2])->second.tensor;
    fc2_bias = kTensorIndex->find(tensor_ids[3])->second.tensor;
}

torch::Tensor NllbMoeDenseActDense::forward(torch::Tensor hidden_states)
{
    // ARCHER_LOG_DEBUG("NllbMoeDenseActDense fc1 {} fc1_bias {} fc2 {} fc2_bias {} hidden_states {}",
    //                  fc1.device().str(),
    //                  fc1_bias.device().str(),
    //                  fc2.device().str(),
    //                  fc2_bias.device().str(),
    //                  hidden_states.device().str());
    return torch::matmul(torch::relu(torch::matmul(hidden_states, fc1.transpose(0, 1)) + fc1_bias),
                         fc2.transpose(0, 1)) +
           fc2_bias;
}

FSGPTMoEDenseActDense::FSGPTMoEDenseActDense(int dtype)
{
    auto tensor_dtype = dtype_to_torch(dtype);
    auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
    fc1 = register_parameter("fc1", torch::zeros({1}, options));
    fc2 = register_parameter("fc2", torch::zeros({1}, options));
    fc1_bias = register_parameter("fc1_bias", torch::zeros({1}, options));
    fc2_bias = register_parameter("fc2_bias", torch::zeros({1}, options));
}

void FSGPTMoEDenseActDense::SetTensorsFromBlob(void *ptr,
                                               const std::vector<std::uint32_t> &tensor_ids,
                                               const torch::Device &device)
{
    fc1 = kTensorIndex->find(tensor_ids[0])->second.tensor;
    fc1_bias = kTensorIndex->find(tensor_ids[1])->second.tensor;
    fc2 = kTensorIndex->find(tensor_ids[2])->second.tensor;
    fc2_bias = kTensorIndex->find(tensor_ids[3])->second.tensor;
}

torch::Tensor FSGPTMoEDenseActDense::forward(torch::Tensor hidden_states)
{
    // ARCHER_LOG_DEBUG("FSGPTMoEDenseActDense fc1 {} fc1_bias {} fc2 {} fc2_bias {} hidden_states {}",
    //                  fc1.device().str(),
    //                  fc1_bias.device().str(),
    //                  fc2.device().str(),
    //                  fc2_bias.device().str(),
    //                  hidden_states.device().str());
    if (hidden_states.dtype() != fc1.dtype())
        hidden_states = hidden_states.to(fc1.dtype());
    return torch::matmul(torch::relu(torch::matmul(hidden_states, fc1.transpose(0, 1)) + fc1_bias),
                         fc2.transpose(0, 1)) +
           fc2_bias;
}

MixtralMoEDenseActDense::MixtralMoEDenseActDense(int dtype)
{
    auto tensor_dtype = dtype_to_torch(dtype);
    auto options = torch::TensorOptions().dtype(tensor_dtype).device(torch::kCPU);
    w1 = register_parameter("w1", torch::zeros({1}, options));
    w2 = register_parameter("w2", torch::zeros({1}, options));
    w3 = register_parameter("w3", torch::zeros({1}, options));
}

void MixtralMoEDenseActDense::SetTensorsFromBlob(void *ptr,
                                                 const std::vector<std::uint32_t> &tensor_ids,
                                                 const torch::Device &device)
{
    w1 = kTensorIndex->find(tensor_ids[0])->second.tensor;
    w2 = kTensorIndex->find(tensor_ids[1])->second.tensor;
    w3 = kTensorIndex->find(tensor_ids[2])->second.tensor;
}

torch::Tensor MixtralMoEDenseActDense::forward(torch::Tensor hidden_states)
{
    /*
    current_hidden_states = self.silu(self.w1(hidden_states)) * self.w3(hidden_states)
    current_hidden_states = self.w2(current_hidden_states)
    return current_hidden_states
    */
    int w1_nan = torch::sum(torch::isnan(w1)).item<int>();
    int w2_nan = torch::sum(torch::isnan(w2)).item<int>();
    int w3_nan = torch::sum(torch::isnan(w3)).item<int>();
    int hidden_states_nan = torch::sum(torch::isnan(hidden_states)).item<int>();
    // std::cout << "MixtralMoEDenseActDense w1 " << w1_nan << " w2 " << w2_nan << " w3 " << w3_nan << " hidden_states " << hidden_states_nan << std::endl;

    assert(w1_nan == 0);
    assert(w2_nan == 0);
    assert(w3_nan == 0);
    assert(hidden_states_nan == 0);

    return torch::matmul(torch::silu(torch::matmul(hidden_states, w1.transpose(0, 1))) * torch::matmul(hidden_states, w3.transpose(0, 1)), w2.transpose(0, 1));
}

void ExpertNode::SetTensorsFromBlob(const torch::Device &device)
{
    int expert_type = this->expert_type;
    switch (expert_type)
    {
    case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
        reinterpret_cast<SwitchTransformersDenseActDense *>(module)->SetTensorsFromBlob(
            node->device_memory_ptr, node->tensor_ids, device);
        break;
    case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
        reinterpret_cast<SwitchTransformersDenseGatedActDense *>(module)->SetTensorsFromBlob(
            node->device_memory_ptr, node->tensor_ids, device);
        break;
    case NLLB_MOE_DENSE_ACT_DENSE:
        reinterpret_cast<NllbMoeDenseActDense *>(module)->SetTensorsFromBlob(
            node->device_memory_ptr, node->tensor_ids, device);
        break;
    case FSGPT_MOE_DENSE_ACT_DENSE:
        reinterpret_cast<FSGPTMoEDenseActDense *>(module)->SetTensorsFromBlob(
            node->device_memory_ptr, node->tensor_ids, device);
        break;
    case MIXTRAL_MOE_DENSE_ACT_DENSE:
        reinterpret_cast<MixtralMoEDenseActDense *>(module)->SetTensorsFromBlob(
            node->device_memory_ptr, node->tensor_ids, device);
        break;
    default:
        assert(false);
    }
}
