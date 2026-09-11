#pragma once

#include <gtest/gtest.h>

#include <atomic>
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

  ResidencyTicket Begin(const NodePtr& node, AdmissionSource source) {
    return manager->BeginAdmission(node, 0, ExpertPhase::DECODE,
                                   AdmissionMode::CACHE, source);
  }

  void Admit(const NodePtr& node, AdmissionSource source) {
    if (!manager->IsCapacityConfigured(0)) {
      ASSERT_TRUE(manager->ConfigureCapacity(0, 100));
    }
    auto ticket = Begin(node, source);
    ASSERT_TRUE(ticket.valid);
    ASSERT_EQ(ticket.outcome, AdmissionOutcome::ADMIT);
    node->device = node->default_device;
    ASSERT_TRUE(manager->CommitAdmission(ticket));
  }

  std::shared_ptr<RecordingTransferOps> transfer_ops;
  std::shared_ptr<ExpertResidencyManager> manager;
};
