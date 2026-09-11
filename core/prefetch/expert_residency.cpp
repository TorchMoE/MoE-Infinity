#include "prefetch/expert_residency.h"

#include <utility>

namespace {

const char* PhasePrefix(ExpertPhase phase) {
  switch (EffectivePhase(phase)) {
    case ExpertPhase::PREFILL:
      return "prefill";
    case ExpertPhase::DECODE:
      return "decode";
    default:
      return "decode";
  }
}

const char* RawPhasePrefix(ExpertPhase phase) {
  switch (phase) {
    case ExpertPhase::PREFILL:
      return "prefill";
    case ExpertPhase::DECODE:
      return "decode";
    default:
      return "mixed";
  }
}

}  // namespace

ExpertResidencyManager::ExpertResidencyManager(
    std::shared_ptr<ResidencyTransferOps> transfer_ops)
    : transfer_ops_(std::move(transfer_ops)) {
  static const char* kKeys[] = {
      "enabled",
      "resident_bytes",
      "resident_experts",
      "prefill_accesses",
      "prefill_hits",
      "prefill_misses",
      "prefill_admissions",
      "prefill_transient",
      "prefill_evictions",
      "prefill_prefetch_issued",
      "prefill_prefetch_completed",
      "prefill_prefetch_rejected",
      "decode_accesses",
      "decode_hits",
      "decode_misses",
      "decode_admissions",
      "decode_transient",
      "decode_evictions",
      "decode_prefetch_issued",
      "decode_prefetch_completed",
      "decode_prefetch_rejected",
      "mixed_accesses",
      "transition_hits",
      "starvation_promotions",
  };
  for (const char* key : kKeys) counters_[key] = 0;
}

std::uint64_t ExpertResidencyManager::KeyFor(const NodePtr& node) {
  return node->corr_id;
}

void ExpertResidencyManager::EnsureDeviceLocked(int gpu_id) {
  const std::size_t needed = static_cast<std::size_t>(gpu_id) + 1;
  if (residents_.size() < needed) residents_.resize(needed);
  if (capacity_bytes_.size() < needed) capacity_bytes_.resize(needed);
  if (pending_ticket_counts_.size() < needed)
    pending_ticket_counts_.resize(needed, 0);
  if (transient_ticket_counts_.size() < needed)
    transient_ticket_counts_.resize(needed, 0);
}

bool ExpertResidencyManager::IsConfiguredLocked(int gpu_id) const {
  if (gpu_id < 0) return false;
  const std::size_t idx = static_cast<std::size_t>(gpu_id);
  return idx < capacity_bytes_.size() && capacity_bytes_[idx].has_value();
}

std::int64_t ExpertResidencyManager::ResidentBytesLocked(int gpu_id) const {
  if (gpu_id < 0) return 0;
  const std::size_t idx = static_cast<std::size_t>(gpu_id);
  if (idx >= residents_.size()) return 0;
  std::int64_t total = 0;
  for (const auto& [key, entry] : residents_[idx]) total += entry.bytes;
  return total;
}

bool ExpertResidencyManager::IsResidentLocked(int gpu_id,
                                              std::uint64_t key) const {
  const std::size_t idx = static_cast<std::size_t>(gpu_id);
  if (idx >= residents_.size()) return false;
  return residents_[idx].find(key) != residents_[idx].end();
}

bool ExpertResidencyManager::ConfigureCapacity(int gpu_id,
                                               std::int64_t capacity_bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (gpu_id < 0 || capacity_bytes < 0) return false;
  EnsureDeviceLocked(gpu_id);
  const std::size_t idx = static_cast<std::size_t>(gpu_id);
  if (pending_ticket_counts_[idx] != 0) return false;
  if (capacity_bytes < ResidentBytesLocked(gpu_id)) return false;
  capacity_bytes_[idx] = capacity_bytes;
  return true;
}

bool ExpertResidencyManager::IsCapacityConfigured(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return IsConfiguredLocked(gpu_id);
}

std::int64_t ExpertResidencyManager::CapacityBytes(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!IsConfiguredLocked(gpu_id)) return -1;
  return capacity_bytes_[static_cast<std::size_t>(gpu_id)].value();
}

EvictionState ExpertResidencyManager::ProjectStateLocked(
    const NodePtr& node, std::uint32_t lease_count) const {
  return EvictionState{
      node->device.is_cuda(),
      node->is_prefetching.load(std::memory_order_acquire),
      node->exec_state.load(std::memory_order_acquire),
      node->pending_dispatches.load(std::memory_order_acquire),
      node->is_overflow,
      lease_count,
      protected_candidates_.find(node) != protected_candidates_.end(),
  };
}

VictimCandidate ExpertResidencyManager::CandidateForLocked(
    const NodePtr& node, ExpertPhase phase) const {
  const auto& m = node->policy_metadata;
  const ExpertPhase active = EffectivePhase(phase);
  const double utility = VictimUtility(m, active, config_);
  const std::uint64_t last_sequence = active == ExpertPhase::PREFILL
                                          ? m.last_prefill_sequence
                                          : m.last_decode_sequence;
  const std::int64_t layer_id =
      static_cast<std::int64_t>(node->corr_id & 0xFFFFFFFF);
  const std::int64_t expert_id = static_cast<std::int64_t>(node->corr_id >> 32);
  return VictimCandidate{layer_id, expert_id, utility, last_sequence};
}

std::optional<NodePtr> ExpertResidencyManager::ReserveVictimLocked(
    int gpu_id, const NodePtr& incoming, ExpertPhase phase) {
  const std::size_t idx = static_cast<std::size_t>(gpu_id);
  std::vector<VictimCandidate> candidates;
  std::unordered_map<std::int64_t, NodePtr> by_key;
  for (const auto& [key, entry] : residents_[idx]) {
    const auto& node = entry.node;
    const auto state = ProjectStateLocked(node, entry.lease_count);
    if (!IsEvictionEligible(state)) continue;
    auto candidate = CandidateForLocked(node, phase);
    by_key[static_cast<std::int64_t>(node->corr_id)] = node;
    candidates.push_back(candidate);
  }
  const auto victim = SelectStableVictim(candidates);
  if (!victim.has_value()) return std::nullopt;
  const std::int64_t corr_id =
      (victim->expert_id << 32) | (victim->layer_id & 0xFFFFFFFF);
  return by_key[corr_id];
}

ResidencyTicket ExpertResidencyManager::BeginAdmission(const NodePtr& incoming,
                                                       int gpu_id,
                                                       ExpertPhase phase,
                                                       AdmissionMode mode,
                                                       AdmissionSource source) {
  std::lock_guard<std::mutex> lock(mutex_);
  ResidencyTicket ticket;
  ticket.incoming = incoming;
  ticket.gpu_id = gpu_id;
  ticket.phase = phase;
  ticket.source = source;

  if (!IsConfiguredLocked(gpu_id)) {
    ticket.valid = false;
    ticket.outcome = AdmissionOutcome::REJECTED;
    return ticket;
  }

  const std::size_t idx = static_cast<std::size_t>(gpu_id);
  const std::uint64_t key = KeyFor(incoming);

  if (IsResidentLocked(gpu_id, key)) {
    ticket.valid = true;
    ticket.outcome = AdmissionOutcome::ALREADY_RESIDENT;
    return ticket;
  }

  const std::int64_t capacity = capacity_bytes_[idx].value();
  const std::int64_t resident = ResidentBytesLocked(gpu_id);
  const bool has_free_space = resident + incoming->byte_size <= capacity;

  ticket.id = next_ticket_id_++;

  if (source == AdmissionSource::PREFETCH) {
    counters_[std::string(PhasePrefix(phase)) + "_prefetch_issued"] += 1;
  }

  if (has_free_space) {
    ticket.valid = true;
    ticket.outcome = AdmissionOutcome::ADMIT;
    pending_ticket_counts_[idx] += 1;
    pending_tickets_[ticket.id] = ticket;
    return ticket;
  }

  if (mode == AdmissionMode::TRANSIENT_ON_PRESSURE) {
    if (transient_ticket_counts_[idx] != 0) {
      ticket.valid = false;
      ticket.outcome = AdmissionOutcome::REJECTED;
      if (source == AdmissionSource::PREFETCH) {
        counters_[std::string(PhasePrefix(phase)) + "_prefetch_rejected"] += 1;
      }
      return ticket;
    }
    ticket.valid = true;
    ticket.transient = true;
    ticket.outcome = AdmissionOutcome::TRANSIENT;
    counters_[std::string(PhasePrefix(phase)) + "_transient"] += 1;
    pending_ticket_counts_[idx] += 1;
    transient_ticket_counts_[idx] += 1;
    pending_tickets_[ticket.id] = ticket;
    return ticket;
  }

  auto victim = ReserveVictimLocked(gpu_id, incoming, phase);
  if (!victim.has_value()) {
    ticket.valid = false;
    ticket.outcome = AdmissionOutcome::REJECTED;
    if (source == AdmissionSource::PREFETCH) {
      counters_[std::string(PhasePrefix(phase)) + "_prefetch_rejected"] += 1;
    }
    return ticket;
  }

  const std::uint64_t victim_key = KeyFor(victim.value());
  residents_[idx][victim_key].lease_count += 1;
  ticket.reserved_victim = victim.value();
  ticket.valid = true;
  ticket.outcome = AdmissionOutcome::ADMIT;
  pending_ticket_counts_[idx] += 1;
  pending_tickets_[ticket.id] = ticket;
  return ticket;
}

void ExpertResidencyManager::ReleaseReservationLocked(
    const ResidencyTicket& ticket) {
  if (ticket.reserved_victim == nullptr) return;
  const std::size_t idx = static_cast<std::size_t>(ticket.gpu_id);
  const std::uint64_t victim_key = KeyFor(ticket.reserved_victim);
  auto it = residents_[idx].find(victim_key);
  if (it != residents_[idx].end() && it->second.lease_count > 0) {
    it->second.lease_count -= 1;
  }
}

bool ExpertResidencyManager::EvictReserved(const ResidencyTicket& ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_tickets_.find(ticket.id);
  if (it == pending_tickets_.end()) return false;
  ResidencyTicket& live = it->second;
  if (live.reserved_victim == nullptr) return false;

  const std::size_t idx = static_cast<std::size_t>(live.gpu_id);
  const std::uint64_t victim_key = KeyFor(live.reserved_victim);
  auto entry_it = residents_[idx].find(victim_key);
  if (entry_it == residents_[idx].end()) return false;

  NodePtr victim = live.reserved_victim;
  if (!transfer_ops_->MoveToHost(victim)) return false;
  residents_[idx].erase(entry_it);
  counters_[std::string(PhasePrefix(live.phase)) + "_evictions"] += 1;

  live.reserved_victim = nullptr;
  return true;
}

bool ExpertResidencyManager::CommitAdmission(const ResidencyTicket& ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_tickets_.find(ticket.id);
  if (it == pending_tickets_.end()) return false;
  ResidencyTicket live = it->second;
  if (live.outcome == AdmissionOutcome::ALREADY_RESIDENT) return false;

  const std::size_t idx = static_cast<std::size_t>(live.gpu_id);

  if (live.transient) {
    ReleaseReservationLocked(live);
    pending_tickets_.erase(it);
    pending_ticket_counts_[idx] -= 1;
    transient_ticket_counts_[idx] -= 1;
    return true;
  }

  ReleaseReservationLocked(live);

  ResidencyEntry entry;
  entry.node = live.incoming;
  entry.bytes = live.incoming->byte_size;
  entry.lease_count = 0;
  residents_[idx][KeyFor(live.incoming)] = entry;

  const std::string phase = PhasePrefix(live.phase);
  if (live.source == AdmissionSource::DEMAND) {
    counters_[phase + "_admissions"] += 1;
  } else {
    counters_[phase + "_prefetch_completed"] += 1;
  }

  pending_tickets_.erase(it);
  pending_ticket_counts_[idx] -= 1;
  return true;
}

bool ExpertResidencyManager::AbortAdmission(const ResidencyTicket& ticket) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_tickets_.find(ticket.id);
  if (it == pending_tickets_.end()) return false;
  ResidencyTicket live = it->second;

  ReleaseReservationLocked(live);
  const std::size_t idx = static_cast<std::size_t>(live.gpu_id);
  pending_tickets_.erase(it);
  pending_ticket_counts_[idx] -= 1;
  if (live.transient) transient_ticket_counts_[idx] -= 1;
  return true;
}

std::uint64_t ExpertResidencyManager::AcquireLease(const NodePtr& node,
                                                   LeaseKind kind) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::uint64_t lease_id = next_lease_id_++;
  leases_[lease_id] = LeaseRecord{lease_id, node, kind};
  for (auto& device_map : residents_) {
    auto it = device_map.find(KeyFor(node));
    if (it != device_map.end()) {
      it->second.lease_count += 1;
      break;
    }
  }
  return lease_id;
}

bool ExpertResidencyManager::ReleaseLease(std::uint64_t lease_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = leases_.find(lease_id);
  if (it == leases_.end()) return false;
  const NodePtr node = it->second.node;
  for (auto& device_map : residents_) {
    auto entry = device_map.find(KeyFor(node));
    if (entry != device_map.end() && entry->second.lease_count > 0) {
      entry->second.lease_count -= 1;
      break;
    }
  }
  leases_.erase(it);
  return true;
}

void ExpertResidencyManager::ReplaceProtectedCandidates(
    const NodePtrList& candidates) {
  std::lock_guard<std::mutex> lock(mutex_);
  protected_candidates_.clear();
  for (const auto& node : candidates) protected_candidates_.insert(node);
}

void ExpertResidencyManager::RecordAccess(const NodePtr& node,
                                          ExpertPhase phase, bool hit) {
  std::lock_guard<std::mutex> lock(mutex_);
  const ExpertPhase active = EffectivePhase(phase);
  auto& m = node->policy_metadata;
  const std::uint64_t seq = ++sequence_counter_;

  counters_[std::string(RawPhasePrefix(phase)) + "_accesses"] += 1;

  if (active == ExpertPhase::PREFILL) {
    m.prefill_accesses += 1;
    m.last_prefill_sequence = seq;
    counters_[hit ? "prefill_hits" : "prefill_misses"] += 1;
  } else {
    const bool is_transition =
        hit && m.decode_accesses == 0 && m.prefill_accesses > 0;
    m.decode_accesses += 1;
    m.last_decode_sequence = seq;
    counters_[hit ? "decode_hits" : "decode_misses"] += 1;
    if (is_transition) counters_["transition_hits"] += 1;
  }
}

void ExpertResidencyManager::ConfigurePolicy(const PhasePolicyConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  counters_["enabled"] = config.enabled ? 1 : 0;
}

bool ExpertResidencyManager::PolicyEnabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.enabled;
}

AdmissionMode ExpertResidencyManager::AdmissionFor(ExpertPhase phase) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return EffectivePhase(phase) == ExpertPhase::PREFILL
             ? config_.prefill_admission
             : config_.decode_admission;
}

std::uint32_t ExpertResidencyManager::StarvationLimit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_.starvation_limit;
}

void ExpertResidencyManager::RecordStarvationPromotion() {
  std::lock_guard<std::mutex> lock(mutex_);
  counters_["starvation_promotions"] += 1;
}

std::int64_t ExpertResidencyManager::ResidentBytes(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ResidentBytesLocked(gpu_id);
}

std::size_t ExpertResidencyManager::ResidentCount(int gpu_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (gpu_id < 0) return 0;
  const std::size_t idx = static_cast<std::size_t>(gpu_id);
  if (idx >= residents_.size()) return 0;
  return residents_[idx].size();
}

ExpertPolicyStats ExpertResidencyManager::Snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  ExpertPolicyStats stats = counters_;
  std::int64_t total_bytes = 0;
  std::int64_t total_experts = 0;
  for (const auto& device_map : residents_) {
    for (const auto& [key, entry] : device_map) {
      total_bytes += entry.bytes;
      total_experts += 1;
    }
  }
  stats["resident_bytes"] = total_bytes;
  stats["resident_experts"] = total_experts;
  return stats;
}
