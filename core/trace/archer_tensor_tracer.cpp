// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "archer_tensor_tracer.h"

#include <algorithm>
#include <sstream>

#include "utils/archer_logger.h"

void ArcherTensorTracer::AddTrace(const std::uint32_t layer_id,
                                  const std::vector<std::uint32_t> buffer)
{
    assert(request_id_ != UINT64_MAX);
    if (traces_.find(request_id_) == traces_.end()) {
        traces_[request_id_] = std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>>();
    }
    traces_[request_id_].push_back({layer_id, buffer});

    max_layer_id_ = std::max(max_layer_id_, layer_id);
}

void ArcherTensorTracer::ClearRequestID()
{
    if (traces_.size() > 1000) { traces_.erase(request_id_); }
    request_id_ = UINT64_MAX;
}

std::vector<std::uint32_t> ArcherTensorTracer::GetCandidates(std::uint32_t layer_id)
{
    if (traces_.size() == 1) { return {}; }

    std::vector<
        std::pair<std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>>, double>>
        candidates;

    for (auto& req_item : traces_[request_id_]) {
        auto req_layer = req_item.first;
        auto req_tensors = req_item.second;

        for (auto& hist_item : traces_) {
            auto hist_req_id = hist_item.first;
            auto value = hist_item.second;

            if (hist_req_id == request_id_) continue;
            if (layer_id < value[0].first) continue;

            for (auto& value_item : value) {
                auto hist_layer = value_item.first;
                auto hist_tensors = value_item.second;

                if (req_layer != hist_layer) continue;

                std::vector<std::uint32_t> overlap;
                std::set_intersection(req_tensors.begin(),
                                      req_tensors.end(),
                                      hist_tensors.begin(),
                                      hist_tensors.end(),
                                      std::back_inserter(overlap));

                std::vector<std::uint32_t> total;
                std::set_union(req_tensors.begin(),
                               req_tensors.end(),
                               hist_tensors.begin(),
                               hist_tensors.end(),
                               std::back_inserter(total));

                double prob =
                    static_cast<double>(overlap.size()) / total.size() / (layer_id - req_layer + 1);

                std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>> candidate_value;
                for (auto& pair : value) {
                    if (pair.first > layer_id) { candidate_value.push_back(pair); }
                }

                if (!candidate_value.empty()) { candidates.push_back({candidate_value, prob}); }
                break;
            }
        }
    }

    if (candidates.empty()) { return {}; }

    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, double>>
        tensor_probs;  // <layer_id, tensor_id>, prob
    for (auto& item : candidates) {
        auto layer_tensors = item.first;
        auto prob = item.second;

        for (auto& layer_item : layer_tensors) {
            auto layer = layer_item.first;
            auto tensors = layer_item.second;
            if (tensor_probs.find(layer) == tensor_probs.end()) {
                tensor_probs[layer] = std::unordered_map<std::uint32_t, double>();
            }
            auto& layer_tensor_probs = tensor_probs[layer];
            for (std::uint32_t tensor : tensors) {
                if (layer_tensor_probs.find(tensor) == layer_tensor_probs.end()) {
                    layer_tensor_probs[tensor] = 0;
                }
                layer_tensor_probs[tensor] += prob;
            }
        }
    }

    // find top 10 tensors for each layer id
    std::vector<std::uint32_t> tensor_ids;
    for (auto& item : tensor_probs) {
        // auto layer = item.first;
        auto& layer_tensor_probs = item.second;

        // if (layer < layer_id + 2) continue; // skip the layers that are too close to the current
        // layer

        std::vector<std::pair<std::uint32_t, double>> layer_tensor_probs_vec(
            layer_tensor_probs.begin(), layer_tensor_probs.end());

        layer_tensor_probs_vec.erase(
            std::remove_if(
                layer_tensor_probs_vec.begin(),
                layer_tensor_probs_vec.end(),
                [](const std::pair<std::uint32_t, double>& pair) { return pair.second < 0.01; }),
            layer_tensor_probs_vec.end());

        if (layer_tensor_probs_vec.empty()) { continue; }

        std::sort(layer_tensor_probs_vec.begin(),
                  layer_tensor_probs_vec.end(),
                  [](const std::pair<std::uint32_t, double>& a,
                     const std::pair<std::uint32_t, double>& b) { return a.second > b.second; });

        std::size_t width = 20;
        for (std::uint32_t i = 0; i < std::min(layer_tensor_probs_vec.size(), width); ++i) {
            tensor_ids.push_back(layer_tensor_probs_vec[i].first);
        }
    }
    return tensor_ids;
}
