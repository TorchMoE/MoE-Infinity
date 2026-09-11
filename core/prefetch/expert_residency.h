#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model/model_topology.h"
#include "prefetch/expert_policy.h"

// Sole production eligibility predicate; no other code may duplicate a subset.
struct EvictionState {
  bool is_cuda;
  bool is_prefetching;
  NodeExecState exec_state;
  int pending_dispatches;
  bool is_overflow;
  std::uint32_t lease_count;
  bool protected_candidate;
};

inline bool IsEvictionEligible(const EvictionState& state) {
  return state.is_cuda && !state.is_prefetching &&
         state.exec_state == NodeExecState::IDLE &&
         state.pending_dispatches == 0 && !state.is_overflow &&
         state.lease_count == 0 && !state.protected_candidate;
}

inline std::optional<VictimCandidate> SelectStableVictim(
    const std::vector<VictimCandidate>& candidates) {
  if (candidates.empty()) return std::nullopt;
  return *std::min_element(candidates.begin(), candidates.end(), VictimLess);
}

enum class AdmissionSource : std::uint8_t { DEMAND = 0, PREFETCH = 1 };

enum class AdmissionOutcome : std::uint8_t {
  ADMIT = 0,
  ALREADY_RESIDENT = 1,
  TRANSIENT = 2,
  REJECTED = 3,
};

enum class LeaseKind : std::uint8_t { DEMAND = 0, PREFETCH = 1, TRANSFER = 2 };

class ResidencyTransferOps {
 public:
  virtual ~ResidencyTransferOps() = default;
  virtual bool MoveToHost(const NodePtr& node) = 0;
};

struct ResidencyTicket {
  std::uint64_t id = 0;
  NodePtr incoming;
  NodePtr reserved_victim;
  int gpu_id = -1;
  ExpertPhase phase = ExpertPhase::MIXED;
  AdmissionSource source = AdmissionSource::DEMAND;
  AdmissionOutcome outcome = AdmissionOutcome::REJECTED;
  bool transient = false;
  bool valid = false;
};

struct ResidencyEntry {
  NodePtr node;
  std::int64_t bytes = 0;
  std::uint32_t lease_count = 0;
};

struct LeaseRecord {
  std::uint64_t id = 0;
  NodePtr node;
  LeaseKind kind = LeaseKind::DEMAND;
};

using ExpertPolicyStats = std::unordered_map<std::string, std::int64_t>;

// Sole authority mutating persistent residency, bytes, leases, and
// reservations.
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

  std::uint64_t AcquireLease(const NodePtr& node, LeaseKind kind);
  bool ReleaseLease(std::uint64_t lease_id);

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
  // Every *Locked helper requires mutex_ to be held by the caller.
  void EnsureDeviceLocked(int gpu_id);
  bool IsConfiguredLocked(int gpu_id) const;
  std::int64_t ResidentBytesLocked(int gpu_id) const;
  bool IsResidentLocked(int gpu_id, std::uint64_t key) const;
  std::optional<NodePtr> ReserveVictimLocked(int gpu_id,
                                             const NodePtr& incoming,
                                             ExpertPhase phase);
  EvictionState ProjectStateLocked(const NodePtr& node,
                                   std::uint32_t lease_count) const;
  VictimCandidate CandidateForLocked(const NodePtr& node,
                                     ExpertPhase phase) const;
  void ReleaseReservationLocked(const ResidencyTicket& ticket);

  static std::uint64_t KeyFor(const NodePtr& node);

  mutable std::mutex mutex_;
  std::shared_ptr<ResidencyTransferOps> transfer_ops_;

  std::vector<std::map<std::uint64_t, ResidencyEntry>> residents_;
  std::vector<std::optional<std::int64_t>> capacity_bytes_;
  std::vector<int> pending_ticket_counts_;
  std::vector<int> transient_ticket_counts_;

  std::unordered_map<std::uint64_t, LeaseRecord> leases_;
  std::unordered_map<std::uint64_t, ResidencyTicket> pending_tickets_;
  std::unordered_set<NodePtr> protected_candidates_;

  PhasePolicyConfig config_;

  std::uint64_t next_ticket_id_ = 1;
  std::uint64_t next_lease_id_ = 1;
  std::uint64_t sequence_counter_ = 0;

  ExpertPolicyStats counters_;
};

extern std::shared_ptr<ExpertResidencyManager> kExpertResidencyManager;

class ExpertResidencyClient {
 public:
  ExpertResidencyClient(std::shared_ptr<ExpertResidencyManager> manager,
                        AdmissionSource source)
      : manager_(std::move(manager)), source_(source) {}

  ResidencyTicket BeginAdmission(const NodePtr& incoming, int gpu_id,
                                 ExpertPhase phase, AdmissionMode mode) {
    return manager_->BeginAdmission(incoming, gpu_id, phase, mode, source_);
  }

  std::shared_ptr<ExpertResidencyManager> manager() const { return manager_; }

 private:
  std::shared_ptr<ExpertResidencyManager> manager_;
  AdmissionSource source_;
};
