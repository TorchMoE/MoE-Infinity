// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <unordered_map>
#include <vector>

class ArcherTensorTracer {
public:
    // ArcherTensorTracer(const std::string& prefix);

    void SetRequestID(const std::uint64_t& request_id) { request_id_ = request_id; }
    std::uint64_t GetRequestID() { return request_id_; }
    void ClearRequestID();

    void AddTrace(const std::uint32_t layer_id, const std::vector<std::uint32_t> buffer);
    std::vector<std::uint32_t> GetCandidates(std::uint32_t layer_id);

private:
    std::uint64_t request_id_;
    std::unordered_map<std::uint32_t,
                       std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>>>
        traces_;
    std::uint32_t max_layer_id_ = 0;
};
