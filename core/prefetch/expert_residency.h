// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model/model_topology.h"
#include "prefetch/expert_policy.h"

enum class AdmissionSource : std::uint8_t { DEMAND = 0, PREFETCH = 1 };

enum class AdmissionOutcome : std::uint8_t {
  ADMIT = 0,
  ALREADY_RESIDENT = 1,
  TRANSIENT = 2,
  REJECTED = 3
};

enum class ExpertFormat : std::uint8_t {
  BF16 = 0,
  FP8_E4M3_BLOCK128 = 1,
  MARLIN_INT4_GROUP128 = 2,
  GPT_OSS_MXFP4 = 3,
  GLM_FP8_BLOCK128 = 4,
  DEEPSEEK_V4_FP4 = 5,
  GPTQ = 6,
  AWQ = 7,
};

struct ResidencyVariantKey {
  std::uint64_t logical_expert_key = 0;
  ExpertFormat format = ExpertFormat::BF16;
  std::uint64_t generation = 0;

  bool operator==(const ResidencyVariantKey& other) const {
    return logical_expert_key == other.logical_expert_key &&
           format == other.format && generation == other.generation;
  }
  bool operator<(const ResidencyVariantKey& other) const {
    return std::tie(logical_expert_key, format, generation) <
           std::tie(other.logical_expert_key, other.format, other.generation);
  }
};

struct ResidencyVariant {
  ResidencyVariantKey key;
  NodePtr node;
  std::int64_t payload_bytes = 0;
  std::int64_t aligned_bytes = 0;
  std::int64_t workspace_bytes = 0;
};

struct ExpertExecutionDescriptor {
  ResidencyVariantKey key;
  std::uint8_t execution_kind = 0;
  std::vector<TensorID> tensor_ids;
  std::vector<std::string> tensor_roles;
};

enum class ResidencyState : std::uint8_t {
  HOST_READY = 0,
  ACTIVE = 1,
  RETIRING = 2,
};

enum class LeaseKind : std::uint8_t {
  DEMAND = 0,
  PREFETCH = 1,
  TRANSFER = 2,
  EXECUTION = 3,
  TRANSITION = 4,
};

enum class ResidencyTransactionKind : std::uint8_t {
  ADMISSION = 0,
  TRANSITION = 1,
};

class ResidencyTransferOps {
 public:
  virtual ~ResidencyTransferOps() = default;
  virtual bool MoveToHost(const NodePtr& node) = 0;
};

struct ResidencyTicket {
  std::uint64_t id = 0;
  ResidencyTransactionKind kind = ResidencyTransactionKind::ADMISSION;
  NodePtr incoming;
  NodePtr reserved_victim;
  ResidencyVariant incoming_variant;
  std::optional<ResidencyVariantKey> replaced_key;
  std::optional<ResidencyVariantKey> reserved_victim_key;
  int gpu_id = -1;
  ExpertPhase phase = ExpertPhase::MIXED;
  AdmissionSource source = AdmissionSource::DEMAND;
  AdmissionOutcome outcome = AdmissionOutcome::REJECTED;
  std::int64_t reserved_aligned_bytes = 0;
  std::int64_t reserved_workspace_bytes = 0;
  bool transient = false;
  bool valid = false;
};

struct ResidencyEntry {
  NodePtr node;
  std::int64_t bytes = 0;
  ResidencyVariantKey key;
  std::int64_t payload_bytes = 0;
  ResidencyState state = ResidencyState::HOST_READY;
  std::uint32_t lease_count = 0;
  cudaEvent_t last_use_event = nullptr;
};

struct LeaseRecord {
  std::uint64_t id = 0;
  NodePtr node;
  ResidencyVariantKey key;
  LeaseKind kind = LeaseKind::DEMAND;
};

struct WorkspaceRecord {
  std::uint64_t transaction_id = 0;
  ResidencyVariantKey key;
  int gpu_id = -1;
  std::int64_t bytes = 0;
  cudaEvent_t completion_event = nullptr;
};

using ExpertPolicyStats = std::unordered_map<std::string, std::int64_t>;

class ExpertResidencyManager {
 public:
  explicit ExpertResidencyManager(
      std::shared_ptr<ResidencyTransferOps> transfer_ops);

  bool ConfigureCapacity(int gpu_id, std::int64_t capacity_bytes);
  bool IsCapacityConfigured(int gpu_id) const;
  std::int64_t CapacityBytes(int gpu_id) const;

  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode,
                                 AdmissionSource source);
  bool EvictReserved(const ResidencyTicket& ticket);
  bool CommitAdmission(const ResidencyTicket& ticket);
  bool AbortAdmission(const ResidencyTicket& ticket);

  void RegisterVariant(const ResidencyVariant& variant);
  std::optional<ResidencyVariant> RegisteredVariant(
      const ResidencyVariantKey& key) const;
  ResidencyTicket BeginAdmission(const ResidencyVariant& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode,
                                 AdmissionSource source);
  ResidencyTicket BeginTransition(const ResidencyVariantKey& current,
                                  const ResidencyVariant& incoming, int gpu_id,
                                  ExpertPhase phase, AdmissionSource source);
  bool CommitTransaction(const ResidencyTicket& transaction);
  bool AbortTransaction(const ResidencyTicket& transaction);

  std::uint64_t AcquireLease(const NodePtr& node, LeaseKind kind);
  std::uint64_t AcquireLease(const ResidencyVariantKey& key, LeaseKind kind);
  bool ReleaseLease(std::uint64_t lease_id);
  bool RecordLastUse(const ResidencyVariantKey& key, cudaEvent_t event);
  bool RecordWorkspaceUse(std::uint64_t transaction_id, cudaEvent_t event);
  std::size_t ReapWorkspace(int gpu_id);
  bool RequestRetirement(const ResidencyVariantKey& key);
  std::size_t ReapRetired(int gpu_id);
  std::vector<ResidencyEntry> ResidentGenerations(int gpu_id) const;
  std::size_t ResidentGenerationCount(int gpu_id) const;
  std::optional<ResidencyVariantKey> ActiveGeneration(
      int gpu_id, std::uint64_t logical_expert_key) const;
  void ReplaceProtectedVariants(
      const std::vector<ResidencyVariantKey>& candidates);

  void ReplaceProtectedCandidates(const NodePtrList& candidates);
  void RecordAccess(const NodePtr& node, ExpertPhase phase, bool hit);

  void ConfigurePolicy(const PhasePolicyConfig& config);
  bool PolicyEnabled() const;
  AdmissionMode AdmissionFor(ExpertPhase phase) const;
  std::uint32_t StarvationLimit() const;
  void RecordStarvationPromotion();

  ExpertPolicyStats Snapshot() const;
  std::int64_t ResidentBytes(int gpu_id) const;
  std::size_t ResidentCount(int gpu_id) const;

 private:
  struct GpuState {
    bool configured = false;
    std::int64_t capacity_bytes = 0;
    std::int64_t resident_bytes = 0;
    std::int64_t transition_reserved_bytes = 0;
    std::int64_t workspace_bytes = 0;
    std::vector<ResidencyEntry> entries;
  };

  GpuState& StateFor(int gpu_id);
  const GpuState* FindState(int gpu_id) const;
  ResidencyEntry* FindEntry(GpuState& state, std::size_t node_id);
  const ResidencyEntry* FindEntry(const GpuState& state,
                                  std::size_t node_id) const;
  ResidencyEntry* FindEntry(GpuState& state, const ResidencyVariantKey& key);
  const ResidencyEntry* FindEntry(const GpuState& state,
                                  const ResidencyVariantKey& key) const;
  ResidencyEntry* FindEntry(const ResidencyVariantKey& key, int* gpu_id);
  NodePtr SelectVictim(const GpuState& state, std::int64_t needed_bytes) const;
  std::uint32_t NodeLeaseCount(std::size_t node_id) const;

  mutable std::mutex mutex_;
  std::shared_ptr<ResidencyTransferOps> transfer_ops_;
  std::unordered_map<int, GpuState> gpu_states_;
  std::unordered_map<std::uint64_t, ResidencyTicket> pending_tickets_;
  std::unordered_map<std::uint64_t, LeaseRecord> leases_;
  std::vector<ResidencyVariant> registered_variants_;
  std::unordered_map<std::uint64_t, WorkspaceRecord> workspace_records_;
  std::unordered_map<std::size_t, ExpertPolicyMetadata> access_metadata_;
  std::unordered_set<std::size_t> protected_ids_;
  std::vector<ResidencyVariantKey> protected_variants_;
  PhasePolicyConfig config_;
  ExpertPolicyStats counters_;

  std::uint64_t next_ticket_id_ = 1;
  std::uint64_t next_lease_id_ = 1;
  std::uint64_t access_sequence_ = 0;
};

class ExpertResidencyClient {
 public:
  ExpertResidencyClient(std::shared_ptr<ExpertResidencyManager> manager,
                        AdmissionSource source)
      : manager_(std::move(manager)), source_(source) {}

  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode) {
    return manager_->BeginAdmission(incoming, gpu_id, phase, mode, source_);
  }

  ResidencyTicket BeginAdmission(const ResidencyVariant& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode) {
    return manager_->BeginAdmission(incoming, gpu_id, phase, mode, source_);
  }

  ResidencyTicket BeginTransition(const ResidencyVariantKey& current,
                                  const ResidencyVariant& incoming, int gpu_id,
                                  ExpertPhase phase) {
    return manager_->BeginTransition(current, incoming, gpu_id, phase, source_);
  }

  std::uint64_t AcquireLease(const ResidencyVariantKey& key, LeaseKind kind) {
    return manager_->AcquireLease(key, kind);
  }
  bool ReleaseLease(std::uint64_t lease_id) {
    return manager_->ReleaseLease(lease_id);
  }
  bool RecordLastUse(const ResidencyVariantKey& key, cudaEvent_t event) {
    return manager_->RecordLastUse(key, event);
  }
  bool RecordWorkspaceUse(std::uint64_t transaction_id, cudaEvent_t event) {
    return manager_->RecordWorkspaceUse(transaction_id, event);
  }
  std::size_t ReapWorkspace(int gpu_id) {
    return manager_->ReapWorkspace(gpu_id);
  }
  bool RequestRetirement(const ResidencyVariantKey& key) {
    return manager_->RequestRetirement(key);
  }
  std::size_t ReapRetired(int gpu_id) { return manager_->ReapRetired(gpu_id); }
  ExpertPolicyStats Snapshot() const { return manager_->Snapshot(); }

  std::shared_ptr<ExpertResidencyManager> manager() const { return manager_; }

 private:
  std::shared_ptr<ExpertResidencyManager> manager_;
  AdmissionSource source_;
};

class TopologyMoveToHostOps final : public ResidencyTransferOps {
 public:
  bool MoveToHost(const NodePtr& node) override;
};

extern std::shared_ptr<ExpertResidencyManager> kExpertResidencyManager;
extern std::unique_ptr<ExpertResidencyClient> kDemandResidencyClient;
extern std::unique_ptr<ExpertResidencyClient> kPrefetchResidencyClient;

void InitExpertResidency();
void ResetExpertResidency();
void ConfigureExpertResidencyCapacityFromTopology();
