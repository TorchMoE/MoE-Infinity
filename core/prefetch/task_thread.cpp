// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "task_thread.h"

#include <pthread.h>

#include <iostream>

#include "common/time.h"
#include "utils/archer_logger.h"

std::atomic_uint32_t kGPUCounter{0};

void SetThreadScheduling(std::thread& th, int policy, int priority)
{
    sched_param sch_params;
    sch_params.sched_priority = priority;
    if (pthread_setschedparam(th.native_handle(), policy, &sch_params)) {
        std::cerr << "Failed to set Thread scheduling : " << std::strerror(errno) << std::endl;
        assert(false);
    }
}

void SetThreadAffinity(std::thread& th, int cpu_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    if (pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset)) {
        std::cerr << "Failed to set Thread affinity : " << std::strerror(errno) << std::endl;
        assert(false);
    }
}

void SetThreadAffinity(std::thread& th)
{
    // get number of cpus
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    kCPUCounter++;
    int cpu_id = kCPUCounter % num_cpus;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    if (pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset)) {
        std::cerr << "Failed to set Thread affinity : " << std::strerror(errno) << std::endl;
        assert(false);
    }
}
