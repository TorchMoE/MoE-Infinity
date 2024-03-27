// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include <torch/extension.h>
#include "parallel/expert_dispatcher.h"
#include "prefetch/archer_prefetch_handle.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<ArcherPrefetchHandle>(m, "prefetch_handle")
        .def(py::init<const std::string&, const double>())

        .def("offload", &ArcherPrefetchHandle::OffloadTensor)
        .def("register",
             (void(ArcherPrefetchHandle::*)(torch::Tensor&, const std::uint32_t)) &
                 ArcherPrefetchHandle::RegisterTensor)
        //    .def("register",
        //         (void(ArcherPrefetchHandle::*)(torch::nn::Module&)) &
        //             ArcherPrefetchHandle::RegisterModule)
        .def("register",
             (void(ArcherPrefetchHandle::*)(torch::Tensor*)) & ArcherPrefetchHandle::RegisterTensor)
        .def("set_tensor_device",
             (void(ArcherPrefetchHandle::*)(torch::Tensor&, torch::Device)) &
                 ArcherPrefetchHandle::SetTensorDevice)
        // .def("begin", (void (ArcherPrefetchHandle::*)(torch::nn::Module&))
        // &ArcherPrefetchHandle::AcquireTensor) .def("end", (void
        // (ArcherPrefetchHandle::*)(torch::nn::Module&)) &ArcherPrefetchHandle::ReleaseTensor)
        .def("begin",
             (void(ArcherPrefetchHandle::*)(std::uint64_t&, torch::Tensor&)) &
                 ArcherPrefetchHandle::AcquireTensor)
        .def("end",
             (void(ArcherPrefetchHandle::*)(std::uint64_t&, torch::Tensor&)) &
                 ArcherPrefetchHandle::ReleaseTensor)
        // .def("begin",
        //      (void (ArcherPrefetchHandle::*)(torch::Tensor&, const std::uint32_t)) &
        //          ArcherPrefetchHandle::AcquireTensor)
        // .def("end",
        //      (void (ArcherPrefetchHandle::*)(torch::Tensor&, const std::uint32_t)) &
        //          ArcherPrefetchHandle::ReleaseTensor)
        //    .def("get_trace",
        //    (torch::Tensor(ArcherPrefetchHandle::*)()) & ArcherPrefetchHandle::GetTrace)
        .def("get_hit_rate",
             (torch::Tensor(ArcherPrefetchHandle::*)()) & ArcherPrefetchHandle::GetHitRate)
        .def("set_trace",
             (void(ArcherPrefetchHandle::*)(const torch::Tensor&)) & ArcherPrefetchHandle::SetTrace)
        //    .def("trace_request",
        //         (void(ArcherPrefetchHandle::*)(const std::uint64_t, const std::uint32_t)) &
        //             ArcherPrefetchHandle::TraceRequest)
        .def("set_topology",
             (void(ArcherPrefetchHandle::*)(
                 const std::vector<std::tuple<std::string, std::vector<std::vector<TensorID>>>>&)) &
                 ArcherPrefetchHandle::SetTopology)
        .def("update_tensor_map",
             (void(ArcherPrefetchHandle::*)(std::uint64_t, std::uint64_t)) &
                 ArcherPrefetchHandle::UpdateTensorMap)
        .def("is_tensor_offloaded", &ArcherPrefetchHandle::IsTensorOffloaded)
        .def("is_tensor_index_initialized", &ArcherPrefetchHandle::IsTensorIndexInitialized)
        .def("is_tensor_on_device",
             (bool(ArcherPrefetchHandle::*)(const torch::Tensor&) const) &
                 ArcherPrefetchHandle::IsTensorOnDevice)
        .def("is_tensor_on_device",
             (bool(ArcherPrefetchHandle::*)(const std::uint32_t) const) &
                 ArcherPrefetchHandle::IsTensorOnDevice)
        .def("get_node_default_device", &ArcherPrefetchHandle::GetNodeDefaultDevice)
        .def("get_node_device", &ArcherPrefetchHandle::GetNodeDevice)
        .def("prefetch_tensors", &ArcherPrefetchHandle::PrefetchTensors)
        .def("replace_cache_candidates", &ArcherPrefetchHandle::ReplaceCacheCandidates)
        .def("enqueue_prefetch", &ArcherPrefetchHandle::EnqueuePrefetch)
        .def("fetch_tensors", &ArcherPrefetchHandle::FetchTensors);
     //    .def("set_node_cache_priority", &ArcherPrefetchHandle::SetNodeCachePriority);

    py::class_<ExpertDispatcher>(m, "expert_dispatcher")
        .def(py::init<int, int, int, int>())
        .def("register_expert", &ExpertDispatcher::RegisterExpert)
        .def("enqueue_expert", &ExpertDispatcher::EnqueueExpert)
        .def("set_inputs", &ExpertDispatcher::SetInputs)
        .def("set_expected_queue", &ExpertDispatcher::SetExpectedQueue)
        .def("wait_expert", &ExpertDispatcher::WaitExpert);
}
