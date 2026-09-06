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
           py::arg("tensor_ids"), py::arg("priority") = kRouteAheadPriority)
      .def("replace_cache_candidates",
           &ArcherPrefetchHandle::ReplaceCacheCandidates)
      .def("enqueue_prefetch", &ArcherPrefetchHandle::EnqueuePrefetch)
      .def("fetch_tensors", &ArcherPrefetchHandle::FetchTensors)
      .def("clean_up_resources", &ArcherPrefetchHandle::CleanUpResources)
      .def("reset_cache", &ArcherPrefetchHandle::ResetCache);
  //    .def("set_node_cache_priority",
  //    &ArcherPrefetchHandle::SetNodeCachePriority);

  m.def("silu_and_mul", &silu_and_mul, "Fused SiLU(gate) * up");
  m.def("gelu_and_mul", &gelu_and_mul, "Fused GeLU(gate) * up");
  m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "Fused GeLU-tanh(gate) * up");
  m.def("fatrelu_and_mul", &fatrelu_and_mul, "Fused FatReLU(gate) * up");

  py::class_<ExpertDispatcher>(m, "expert_dispatcher")
      .def(py::init<int, int, int, int, int>())
      .def("register_expert", &ExpertDispatcher::RegisterExpert)
      .def("enqueue_expert", &ExpertDispatcher::EnqueueExpert)
      .def("set_inputs", &ExpertDispatcher::SetInputs)
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
      .def("set_scales", &ExpertDispatcher::SetScales,
           "Store fp8 block scales for dequant-on-copy (fp8-in-store path)");
}
