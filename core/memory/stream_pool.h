// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <c10/cuda/CUDAStream.h>

#include "utils/cuda_utils.h"
#include "utils/noncopyable.h"

class CUDAStreamPool : public noncopyable {
public:
    std::vector<c10::cuda::CUDAStream>& operator()(const int device_id)
    {
        return cuda_streams_[device_id];
    }

    CUDAStreamPool()
    {
        int num_devices = GetDeviceCount();
        for (int i = 0; i < num_devices; ++i) {
            std::vector<c10::cuda::CUDAStream> streams;
            for (int j = 0; j < 3; ++j) {
                streams.push_back(c10::cuda::getStreamFromPool(false, i));
            }
            cuda_streams_.push_back(std::move(streams));
        }
    }
    virtual ~CUDAStreamPool() = default;

private:
    std::vector<std::vector<c10::cuda::CUDAStream>> cuda_streams_;
};

extern std::unique_ptr<CUDAStreamPool> kCUDAStreamPool;
#define CUDA_STREAM_VIEW(device_id, stream_id) (*kCUDAStreamPool)(device_id)[stream_id]
#define CUDA_STREAM_H2D_VIEW(device_id) CUDA_STREAM_VIEW(device_id, 0)
#define CUDA_STREAM_D2H_VIEW(device_id) CUDA_STREAM_VIEW(device_id, 1)
#define CUDA_STREAM_COMPUTE_VIEW(device_id) CUDA_STREAM_VIEW(device_id, 2)
