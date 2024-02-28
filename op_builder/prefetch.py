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
            '-lspdlog',
            '-lcuda',
            '-lcudart',
            '-lcublas',
            '-lpthread',
        ]

    def extra_ldflags(self):
        return ['-laio']

    def check_for_libaio_pkg(self):
        libs = dict(
            dpkg=["-l", "libaio-dev", "apt"],
            pacman=["-Q", "libaio", "pacman"],
            rpm=["-q", "libaio-devel", "yum"],
        )

        found = False
        for pkgmgr, data in libs.items():
            flag, lib, tool = data
            path = distutils.spawn.find_executable(pkgmgr)
            if path is not None:
                cmd = f"{pkgmgr} {flag} {lib}"
                result = subprocess.Popen(cmd,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          shell=True)
                if result.wait() == 0:
                    found = True
                else:
                    self.warning(
                        f"{self.NAME}: please install the {lib} package with {tool}"
                    )
                break
        return found

    def is_compatible(self, verbose=True):
        # Check for the existence of libaio by using distutils
        # to compile and link a test program that calls io_submit,
        # which is a function provided by libaio that is used in the async_io op.
        # If needed, one can define -I and -L entries in CFLAGS and LDFLAGS
        # respectively to specify the directories for libaio.h and libaio.so.
        aio_compatible = self.has_function('io_submit', ('aio', ))
        if verbose and not aio_compatible:
            self.warning(
                f"{self.NAME} requires the dev libaio .so object and headers but these were not found."
            )

            # Check for the libaio package via known package managers
            # to print suggestions on which package to install.
            self.check_for_libaio_pkg()

            self.warning(
                "If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found."
            )
        return super().is_compatible(verbose) and aio_compatible
