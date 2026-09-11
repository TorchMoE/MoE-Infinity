// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include <torch/extension.h>
#include "parallel/expert_dispatcher.h"
#include "prefetch/archer_prefetch_handle.h"
#include "model/moe.h"
#include "kernel/ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_moe_layer", InitMoELayer,
        "Initialize the MoE layer with the specified parameters.");
  m.def("topk_softmax", TopKSoftmax,
        "Perform top-k softmax operation on the MoE layer.");

  py::class_<PrefetchAdmission>(m, "prefetch_admission")
      .def_readonly("accepted_tensor_ids",
                    &PrefetchAdmission::accepted_tensor_ids)
      .def_readonly("accepted_bytes", &PrefetchAdmission::accepted_bytes)
      .def_readonly("rejected_bytes", &PrefetchAdmission::rejected_bytes)
      .def_readonly("inflight_bytes", &PrefetchAdmission::inflight_bytes);

  py::class_<PrefetchSample>(m, "prefetch_sample")
      .def_readonly("generation", &PrefetchSample::generation)
      .def_readonly("layer_id", &PrefetchSample::layer_id)
      .def_readonly("tensor_id", &PrefetchSample::tensor_id)
      .def_readonly("bytes", &PrefetchSample::bytes)
      .def_readonly("queue_wait_ns", &PrefetchSample::queue_wait_ns)
      .def_readonly("transfer_ns", &PrefetchSample::transfer_ns)
      .def_readonly("source_device", &PrefetchSample::source_device)
      .def_readonly("outcome", &PrefetchSample::outcome);

  py::class_<ArcherPrefetchHandle>(m, "prefetch_handle")
      .def(py::init<const std::string&, const double>())

      .def("offload", &ArcherPrefetchHandle::OffloadTensor)
      .def("register", (void(ArcherPrefetchHandle::*)(torch::Tensor&,
                                                      const std::uint32_t)) &
                           ArcherPrefetchHandle::RegisterTensor)
      //    .def("register",
      //         (void(ArcherPrefetchHandle::*)(torch::nn::Module&)) &
      //             ArcherPrefetchHandle::RegisterModule)
      .def("register", (void(ArcherPrefetchHandle::*)(torch::Tensor*)) &
                           ArcherPrefetchHandle::RegisterTensor)
      .def("set_tensor_device",
           (void(ArcherPrefetchHandle::*)(torch::Tensor&, torch::Device)) &
               ArcherPrefetchHandle::SetTensorDevice)
      // .def("begin", (void (ArcherPrefetchHandle::*)(torch::nn::Module&))
      // &ArcherPrefetchHandle::AcquireTensor) .def("end", (void
      // (ArcherPrefetchHandle::*)(torch::nn::Module&))
      // &ArcherPrefetchHandle::ReleaseTensor)
      .def("begin", (void(ArcherPrefetchHandle::*)(
                        std::uint64_t&, torch::Tensor&, std::uint32_t)) &
                        ArcherPrefetchHandle::AcquireTensor)
      .def("end", (void(ArcherPrefetchHandle::*)(std::uint64_t&, torch::Tensor&,
                                                 std::uint32_t)) &
                      ArcherPrefetchHandle::ReleaseTensor)
      // .def("begin",
      //      (void (ArcherPrefetchHandle::*)(torch::Tensor&, const
      //      std::uint32_t)) &
      //          ArcherPrefetchHandle::AcquireTensor)
      // .def("end",
      //      (void (ArcherPrefetchHandle::*)(torch::Tensor&, const
      //      std::uint32_t)) &
      //          ArcherPrefetchHandle::ReleaseTensor)
      //    .def("get_trace",
      //    (torch::Tensor(ArcherPrefetchHandle::*)()) &
      //    ArcherPrefetchHandle::GetTrace)
      .def("get_hit_rate", (torch::Tensor(ArcherPrefetchHandle::*)()) &
                               ArcherPrefetchHandle::GetHitRate)
      .def("get_expert_occupancy_bytes",
           &ArcherPrefetchHandle::GetExpertOccupancyBytes)
      .def("get_wasted_prefetch_bytes",
           &ArcherPrefetchHandle::GetWastedPrefetchBytes)
      .def("set_trace", (void(ArcherPrefetchHandle::*)(const torch::Tensor&)) &
                            ArcherPrefetchHandle::SetTrace)
      //    .def("trace_request",
      //         (void(ArcherPrefetchHandle::*)(const std::uint64_t, const
      //         std::uint32_t)) &
      //             ArcherPrefetchHandle::TraceRequest)
      .def("set_topology",
           (void(ArcherPrefetchHandle::*)(
               const std::vector<std::tuple<
                   std::string, std::vector<std::vector<TensorID>>>>&)) &
               ArcherPrefetchHandle::SetTopology)
      .def("set_topology_v2",
           (void(ArcherPrefetchHandle::*)(
               const std::vector<std::tuple<std::string, bool,
                                            std::vector<std::vector<TensorID>>,
                                            std::vector<std::uint64_t>>>&)) &
               ArcherPrefetchHandle::SetTopologyV2)
      .def("get_topology_snapshot", &ArcherPrefetchHandle::GetTopologySnapshot)
      .def("get_expert_policy_stats",
           &ArcherPrefetchHandle::GetExpertPolicyStats)
      .def("get_policy_stats", &ArcherPrefetchHandle::GetExpertPolicyStats)
      .def("get_residency_manager_id",
           &ArcherPrefetchHandle::GetResidencyManagerId)
      .def("configure_residency_manager",
           &ArcherPrefetchHandle::ConfigureResidencyManager)
      .def("set_adaptive_hbm_budget_bytes",
           &ArcherPrefetchHandle::SetAdaptiveHbmBudgetBytes)
      .def("prefetch_expert_variants",
           &ArcherPrefetchHandle::PrefetchExpertVariants, py::arg("keys"),
           py::arg("priority") = kBackgroundPrefetchPriority,
           py::arg("phase") = "mixed")
      .def("update_tensor_map",
           (void(ArcherPrefetchHandle::*)(std::uint64_t, std::uint64_t)) &
               ArcherPrefetchHandle::UpdateTensorMap)
      .def("is_tensor_offloaded", &ArcherPrefetchHandle::IsTensorOffloaded)
      .def("is_tensor_index_initialized",
           &ArcherPrefetchHandle::IsTensorIndexInitialized)
      .def("is_tensor_on_device",
           (bool(ArcherPrefetchHandle::*)(const torch::Tensor&) const) &
               ArcherPrefetchHandle::IsTensorOnDevice)
      .def("is_tensor_on_device",
           (bool(ArcherPrefetchHandle::*)(const std::uint32_t) const) &
               ArcherPrefetchHandle::IsTensorOnDevice)
      .def("get_node_default_device",
           &ArcherPrefetchHandle::GetNodeDefaultDevice)
      .def("get_node_device", &ArcherPrefetchHandle::GetNodeDevice)
      .def("prefetch_tensors", &ArcherPrefetchHandle::EnqueuePrefetchTensors,
           py::arg("tensor_ids"), py::arg("priority") = kRouteAheadPriority,
           py::arg("phase") = static_cast<int>(ExpertPhase::MIXED))
      .def("schedule_prefetch_tensors",
           &ArcherPrefetchHandle::SchedulePrefetchTensors,
           py::arg("tensor_ids"), py::arg("priority"), py::arg("generation"),
           py::arg("layer_id"), py::arg("max_inflight_bytes"))
      .def("cancel_prefetch_generation",
           &ArcherPrefetchHandle::CancelPrefetchGeneration,
           py::arg("generation"), py::arg("layer_id"),
           py::arg("keep_tensor_ids"))
      .def("drain_prefetch_samples",
           &ArcherPrefetchHandle::DrainPrefetchSamples)
      .def("get_inflight_prefetch_bytes",
           &ArcherPrefetchHandle::GetInflightPrefetchBytes)
      .def("replace_cache_candidates",
           &ArcherPrefetchHandle::ReplaceCacheCandidates)
      .def("enqueue_prefetch", &ArcherPrefetchHandle::EnqueuePrefetch)
      .def("fetch_tensors", &ArcherPrefetchHandle::FetchTensors)
      .def("get_canonical_tensor_index_snapshot",
           &ArcherPrefetchHandle::GetCanonicalTensorIndexSnapshot)
      .def("begin_derivative_overlay",
           &ArcherPrefetchHandle::BeginDerivativeOverlay)
      .def("register_derivative_tensor",
           &ArcherPrefetchHandle::RegisterDerivativeTensor)
      .def("commit_derivative_overlay",
           &ArcherPrefetchHandle::CommitDerivativeOverlay)
      .def("abort_derivative_overlay",
           &ArcherPrefetchHandle::AbortDerivativeOverlay)
      .def("clean_up_resources", &ArcherPrefetchHandle::CleanUpResources)
      .def("configure_phase_policy",
           &ArcherPrefetchHandle::ConfigureExpertPolicy, py::arg("enabled"),
           py::arg("prefill_admission"), py::arg("decode_admission"),
           py::arg("prefill_weight"), py::arg("decode_weight"),
           py::arg("starvation_limit"))
      .def("get_expert_policy_stats",
           &ArcherPrefetchHandle::GetExpertPolicyStats)
      .def("reset_cache", &ArcherPrefetchHandle::ResetCache);
  //    .def("set_node_cache_priority",
  //    &ArcherPrefetchHandle::SetNodeCachePriority);

  m.def("silu_and_mul", &silu_and_mul, "Fused SiLU(gate) * up");
  m.def("gelu_and_mul", &gelu_and_mul, "Fused GeLU(gate) * up");
  m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "Fused GeLU-tanh(gate) * up");
  m.def("fatrelu_and_mul", &fatrelu_and_mul, "Fused FatReLU(gate) * up");

  py::class_<ExpertComputeSample>(m, "expert_compute_sample")
      .def_readonly("invocation_id", &ExpertComputeSample::invocation_id)
      .def_readonly("layer_id", &ExpertComputeSample::layer_id)
      .def_readonly("expert_id", &ExpertComputeSample::expert_id)
      .def_readonly("gpu_id", &ExpertComputeSample::gpu_id)
      .def_readonly("kernel_start_offset_ns",
                    &ExpertComputeSample::kernel_start_offset_ns)
      .def_readonly("kernel_end_offset_ns",
                    &ExpertComputeSample::kernel_end_offset_ns)
      .def_readonly("kernel_duration_ns",
                    &ExpertComputeSample::kernel_duration_ns)
      .def_readonly("forward_return_host_ns",
                    &ExpertComputeSample::forward_return_host_ns)
      .def_readonly("output_complete_host_ns",
                    &ExpertComputeSample::output_complete_host_ns)
      .def_readonly("output_delay_ns", &ExpertComputeSample::output_delay_ns);

  py::class_<ExpertDispatcher>(m, "expert_dispatcher")
      .def(py::init<int, int, int, int, int>())
      .def("register_expert", &ExpertDispatcher::RegisterExpert)
      .def("enqueue_expert", &ExpertDispatcher::EnqueueExpert,
           py::arg("layer_idx"), py::arg("expert_idx"), py::arg("gpu_id") = -1,
           py::arg("remote") = false,
           py::arg("phase") = static_cast<int>(ExpertPhase::MIXED))
      .def("set_inputs", &ExpertDispatcher::SetInputs)
      .def("set_inputs_with_invocation",
           &ExpertDispatcher::SetInputsWithInvocation)
      .def("set_overlap_compute_timing_enabled",
           &ExpertDispatcher::SetOverlapComputeTimingEnabled)
      .def("drain_compute_samples", &ExpertDispatcher::DrainComputeSamples)
      .def("set_expected_queue", &ExpertDispatcher::SetExpectedQueue)
      .def("wait_expert", &ExpertDispatcher::WaitHiddenStates)
      .def("dispatch_experts", &ExpertDispatcher::DispatchExperts)
      .def("take_last_active_experts", &ExpertDispatcher::TakeLastActiveExperts)
      .def("get_routing_stats", &ExpertDispatcher::GetRoutingStats)
      .def("_set_dispatch_fault_for_test",
           &ExpertDispatcher::SetDispatchFaultForTest)
      .def("_fail_dispatch_for_test", &ExpertDispatcher::FailDispatchForTest)
      .def("notify_fetch_start", &ExpertDispatcher::NotifyFetchStart)
      .def("clear_expert_cache_counts",
           &ExpertDispatcher::ClearExpertCacheCounts)
      .def("get_cache_occupancy_bytes",
           &ExpertDispatcher::GetCacheOccupancyBytes)
      .def("get_cache_hit_rate", &ExpertDispatcher::GetCacheHitRate)
      .def("register_expert_variant", &ExpertDispatcher::RegisterExpertVariant)
      .def("set_precision_targets", &ExpertDispatcher::SetPrecisionTargets,
           py::arg("targets"), py::arg("epoch"))
      .def("set_adaptive_hbm_budget_bytes",
           &ExpertDispatcher::SetAdaptiveHbmBudgetBytes)
      .def("get_precision_metrics",
           [](const ExpertDispatcher& dispatcher) {
             py::dict result;
             const auto metrics = dispatcher.GetPrecisionMetrics();
             for (const auto& metric : metrics)
               result[py::str(metric.first)] = metric.second;
             result["active_format"] = dispatcher.GetActiveFormat();
             py::dict fallback_counts;
             auto failed = metrics.find("transition_failed");
             fallback_counts["transition_failed"] =
                 failed == metrics.end() ? 0 : failed->second;
             result["fallback_counts"] = fallback_counts;
             py::dict leases_by_kind;
             for (const char* kind : {"demand", "prefetch", "transfer",
                                      "execution", "transition"}) {
               const auto found = metrics.find(std::string("leases_") + kind);
               leases_by_kind[kind] =
                   found == metrics.end() ? 0 : found->second;
             }
             result["leases_by_kind"] = leases_by_kind;
             static const char* formats[] = {"bf16",
                                             "fp8_e4m3_block128",
                                             "marlin_int4_group128",
                                             "gpt_oss_mxfp4",
                                             "glm_fp8_block128",
                                             "deepseek_v4_fp4",
                                             "gptq",
                                             "awq"};
             py::list entries;
             py::dict by_format;
             for (const char* format : formats) {
               py::dict aggregate;
               aggregate["resident_generations"] = 0;
               aggregate["resident_bytes"] = 0;
               by_format[format] = aggregate;
             }
             for (const auto& row : dispatcher.GetResidentGenerationEntries()) {
               const auto format_index = std::get<1>(row);
               py::dict item;
               item["logical_expert_key"] = std::get<0>(row);
               item["format"] = formats[format_index];
               item["generation"] = std::get<2>(row);
               item["payload_bytes"] = std::get<3>(row);
               item["aligned_bytes"] = std::get<4>(row);
               item["state"] = std::get<5>(row) == 1 ? "active" : "retiring";
               entries.append(item);
               auto aggregate =
                   by_format[formats[format_index]].cast<py::dict>();
               aggregate["resident_generations"] =
                   aggregate["resident_generations"].cast<std::int64_t>() + 1;
               aggregate["resident_bytes"] =
                   aggregate["resident_bytes"].cast<std::int64_t>() +
                   std::get<4>(row);
             }
             result["resident_generation_entries"] = entries;
             result["by_format"] = by_format;
             return result;
           })
      .def("get_policy_stats", &ExpertDispatcher::GetPrecisionMetrics)
      .def("get_residency_manager_id", &ExpertDispatcher::GetResidencyManagerId)
      .def("configure_residency_manager",
           &ExpertDispatcher::ConfigureResidencyManager)
#ifdef MOE_INFINITY_TESTING
      .def("inject_transition_failure_once_for_test",
           &ExpertDispatcher::InjectTransitionFailureOnceForTest)
#endif
      .def("set_scales", &ExpertDispatcher::SetScales,
           "Store fp8 block scales for dequant-on-copy (fp8-in-store path)");
}
