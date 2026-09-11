// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "prefetch/expert_residency.h"

class RecordingTransferOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override {
    moved_ids.push_back(node->id);
    node->device = node->default_host;
    return true;
  }
  std::vector<std::size_t> moved_ids;
};

class ExpertResidencyManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    transfer_ops = std::make_shared<RecordingTransferOps>();
    manager = std::make_shared<ExpertResidencyManager>(transfer_ops);
    demand_client = std::make_unique<ExpertResidencyClient>(
        manager, AdmissionSource::DEMAND);
    prefetch_client = std::make_unique<ExpertResidencyClient>(
        manager, AdmissionSource::PREFETCH);
  }

  ResidencyVariant MakeVariant(std::uint32_t layer_id, std::uint32_t expert_id,
                               ExpertFormat format, std::uint64_t generation,
                               std::int64_t aligned_bytes,
                               std::int64_t workspace_bytes) {
    const auto logical_key = (static_cast<std::uint64_t>(layer_id) << 32) |
                             static_cast<std::uint64_t>(expert_id);
    auto node =
        MakeNode(static_cast<std::size_t>((logical_key << 8) | generation),
                 aligned_bytes);
    return ResidencyVariant{
        ResidencyVariantKey{logical_key, format, generation}, node,
        aligned_bytes, aligned_bytes, workspace_bytes};
  }

  void CommitResident(const ResidencyVariant& variant, ExpertPhase phase) {
    if (!manager->IsCapacityConfigured(0)) {
      ASSERT_TRUE(manager->ConfigureCapacity(0, 1024));
    }
    manager->RegisterVariant(variant);
    auto transaction =
        demand_client->BeginAdmission(variant, 0, phase, AdmissionMode::CACHE);
    ASSERT_TRUE(transaction.valid);
    variant.node->device = variant.node->default_device;
    ASSERT_TRUE(manager->CommitTransaction(transaction));
    if (variant.workspace_bytes > 0) {
      ASSERT_TRUE(manager->RecordWorkspaceUse(transaction.id, nullptr));
      ASSERT_EQ(manager->ReapWorkspace(0), 1);
    }
  }

  NodePtr MakeNode(std::size_t id, std::int64_t bytes) {
    auto node = std::make_shared<Node>();
    node->id = id;
    node->corr_id = id;
    node->byte_size = bytes;
    node->device = torch::Device(torch::kCPU);
    node->default_device = torch::Device(torch::kCUDA, 0);
    node->default_host = torch::Device(torch::kCPU);
    return node;
  }

  std::shared_ptr<RecordingTransferOps> transfer_ops;
  std::shared_ptr<ExpertResidencyManager> manager;
  std::unique_ptr<ExpertResidencyClient> demand_client;
  std::unique_ptr<ExpertResidencyClient> prefetch_client;
};
