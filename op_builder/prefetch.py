#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# op_builder/async_io.py
#
# Part of the DeepSpeed Project, under the Apache-2.0 License.
# See https://github.com/microsoft/DeepSpeed/blob/master/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0

# MoE-Infinity: replaced AsyncIOBuilder with PrefetchBuilder

from .builder import OpBuilder
import distutils
import subprocess
import glob


class PrefetchBuilder(OpBuilder):
    BUILD_VAR = "MOE_BUILD_PREFETCH"
    NAME = "prefetch"

    def __init__(self):
        super().__init__(name=self.NAME)
    
    def absolute_name(self):
        return f'moe_infinity.ops.prefetch.{self.NAME}_op'

    def sources(self):
        return [
            'core/utils/archer_logger.cpp',
            'core/utils/cuda_utils.cpp',
            'core/model/model_topology.cpp',
            'core/prefetch/archer_prefetch_handle.cpp',
            'core/prefetch/task_scheduler.cpp',
            'core/prefetch/task_thread.cpp',
            'core/memory/memory_pool.cpp',
            'core/memory/stream_pool.cpp',
            'core/memory/host_caching_allocator.cpp',
            'core/python/py_archer_prefetch.cpp',
            'core/parallel/expert_dispatcher.cpp',
            'core/parallel/expert_module.cpp',
            'core/aio/archer_aio_thread.cpp',
            'core/aio/archer_prio_aio_handle.cpp',
            'core/aio/archer_aio_utils.cpp',
            'core/aio/archer_aio_threadpool.cpp',
            'core/aio/archer_tensor_handle.cpp',
            'core/aio/archer_tensor_index.cpp',
        ]

    def include_paths(self):
        return ['core']

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        return [
            '-g',
            '-Wall',
            '-O2',
            '-std=c++17',
            '-shared',
            '-fPIC',
            '-Wno-reorder',
            CPU_ARCH,
            '-fopenmp',
            SIMD_WIDTH,
            '-I/usr/local/cuda/include',
            '-L/usr/local/cuda/lib64',
            '-lcuda',
            '-lcudart',
            '-lcublas',
            '-lpthread',
        ]

    def extra_ldflags(self):
        return []

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)
