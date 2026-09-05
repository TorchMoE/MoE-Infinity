# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

import io
import os
import sys
from typing import Any, Optional

from setuptools import find_packages, setup

torch_available = True
cuda_available = False
torch: Any = None
try:
    import torch as _torch

    torch = _torch

    cuda_available = torch.version.cuda is not None
except ImportError:
    torch_available = False
    print(
        "[WARNING] Unable to import torch, pre-compiling ops will be disabled. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."
    )

ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, ROOT_DIR)

from torch.utils import cpp_extension

TORCH_LIB_DIR = (
    os.path.join(os.path.dirname(torch.__file__), "lib")
    if torch_available
    else ""
)

RED_START = "\033[31m"
RED_END = "\033[0m"
ERROR = f"{RED_START} [ERROR] {RED_END}"


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def abort(msg):
    print(f"{ERROR} {msg}")
    assert False, msg


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def _find_cuda_home() -> str:
    cuda_version = (
        torch.version.cuda if torch_available and torch.version.cuda else ""
    )
    cuda_major = cuda_version.split(".")[0] if cuda_version else ""
    if cuda_major == "12":
        candidates = [
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.2",
            os.environ.get("CUDA_HOME"),
            "/usr/local/cuda",
            "/usr/local/cuda-13.2",
            "/usr/local/cuda-13",
        ]
    elif cuda_major == "13":
        candidates = [
            "/usr/local/cuda-13.2",
            "/usr/local/cuda-13",
            os.environ.get("CUDA_HOME"),
            "/usr/local/cuda",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.2",
        ]
    else:
        candidates = [
            os.environ.get("CUDA_HOME"),
            "/usr/local/cuda",
            "/usr/local/cuda-13.2",
            "/usr/local/cuda-13",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.2",
        ]

    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.expanduser(candidate)
        if os.path.isfile(os.path.join(candidate, "bin", "nvcc")):
            return candidate

    return os.path.expanduser(os.environ.get("CUDA_HOME", "/usr/local/cuda"))


install_requires = fetch_requirements("requirements.txt")

# Get CUTLASS_DIR from environment or default to ~/cutlass
CUTLASS_DIR = os.path.expanduser(os.environ.get("CUTLASS_DIR", "~/cutlass"))

CUDA_HOME = _find_cuda_home()
os.environ["CUDA_HOME"] = CUDA_HOME
cpp_extension.CUDA_HOME = CUDA_HOME


def _find_nvtx_include_dir() -> Optional[str]:
    cuda_roots = [
        CUDA_HOME,
        "/usr/local/cuda",
        "/usr/local/cuda-13.2",
        "/usr/local/cuda-13",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.2",
    ]
    for root in cuda_roots:
        nvtx_header = os.path.join(
            os.path.expanduser(root), "include", "nvtx3", "nvtx3.hpp"
        )
        if os.path.isfile(nvtx_header):
            return os.path.dirname(os.path.dirname(nvtx_header))
    return None


COMMON_NVTX_INCLUDE_DIR = _find_nvtx_include_dir()

# Common include paths
COMMON_INCLUDE_PATHS = [
    get_path("core"),
    get_path("core", "include"),
    get_path("extensions"),
    os.path.join(CUTLASS_DIR, "include"),
    os.path.join(CUTLASS_DIR, "tools/util/include"),
]
if COMMON_NVTX_INCLUDE_DIR is not None:
    COMMON_INCLUDE_PATHS.append(COMMON_NVTX_INCLUDE_DIR)

# Common compile args
COMMON_NVCC_ARGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

COMMON_CXX_ARGS = [
    "-O3",
    "-Wall",
    "-Wno-reorder",
    "-fPIC",
    "-fopenmp",
]

if os.environ.get("NVTX_DISABLE", "0") == "1":
    COMMON_CXX_ARGS.append("-DNVTX_DISABLE")
    COMMON_NVCC_ARGS.append("-DNVTX_DISABLE")

# _store extension: IO/checkpoint and prefetch functionality
# Includes AIO, prefetch handle, tensor index, memory pools, model topology
_STORE_SOURCES = [
    # utils
    "core/utils/logger.cpp",
    "core/utils/cuda_utils.cpp",
    # model
    "core/model/model_topology.cpp",
    "core/model/moe.cpp",
    # prefetch
    "core/prefetch/archer_prefetch_handle.cpp",
    "core/prefetch/task_scheduler.cpp",
    "core/prefetch/task_thread.cpp",
    # memory
    "core/memory/caching_allocator.cpp",
    "core/memory/memory_pool.cpp",
    "core/memory/pinned_memory_pool.cpp",
    "core/memory/stream_pool.cpp",
    "core/memory/event_pool.cpp",
    "core/memory/host_caching_allocator.cpp",
    "core/memory/device_caching_allocator.cpp",
    # parallel
    "core/parallel/expert_dispatcher.cpp",
    "core/parallel/expert_module.cpp",
    # aio
    "core/aio/archer_aio_thread.cpp",
    "core/aio/archer_prio_aio_handle.cpp",
    "core/aio/archer_aio_utils.cpp",
    "core/aio/archer_aio_threadpool.cpp",
    "core/aio/archer_tensor_handle.cpp",
    "core/aio/archer_tensor_index.cpp",
    # base
    "core/base/thread.cc",
    "core/base/exception.cc",
    "core/base/date.cc",
    "core/base/process_info.cc",
    "core/base/logging.cc",
    "core/base/log_file.cc",
    "core/base/timestamp.cc",
    "core/base/file_util.cc",
    "core/base/countdown_latch.cc",
    "core/base/timezone.cc",
    "core/base/log_stream.cc",
    "core/base/thread_pool.cc",
    # CUDA kernels for store
    "core/model/fused_mlp.cu",
    "extensions/kernel/fused_moe_mlp.cu",
    "extensions/kernel/activation_kernels.cu",
    "extensions/kernel/topk_softmax_kernels.cu",
    "extensions/kernel/v4_fp4/mxfp4_dequant.cu",
    "extensions/kernel/v4_fp4/fp8_dequant.cu",
    # Python binding
    "core/python/py_archer_prefetch.cpp",
]

_STORE_EXTRA_LINK_ARGS = [
    "-luuid",
    "-lcublas",
    "-lcudart",
    "-lcuda",
    "-lpthread",
]

# NVTX v3 (nvtx3.hpp) is header-only and requires no link library; the legacy
# libnvToolsExt was removed in CUDA 13, so do not link it.

if TORCH_LIB_DIR:
    _STORE_EXTRA_LINK_ARGS.append(f"-Wl,-rpath,{TORCH_LIB_DIR}")

# libcuda (-lcuda) ships with the GPU driver, not the toolkit, so it is
# missing on driverless build machines (CI). Link against the toolkit's stub
# in lib64/stubs; its SONAME (libcuda.so.1) means the real driver is still
# loaded at runtime on GPUs.
_CUDA_STUBS_DIR = os.path.join(CUDA_HOME, "lib64", "stubs")
_STORE_LIBRARY_DIRS = (
    [_CUDA_STUBS_DIR] if os.path.isdir(_CUDA_STUBS_DIR) else []
)

# _engine extension: compute kernels (fused_glu + expert_gemm)
_ENGINE_SOURCES = [
    "core/python/fused_glu_cuda.cu",
]

_KV_CACHE_SOURCES = [
    "core/utils/logger.cpp",
    "core/utils/cuda_utils.cpp",
    "core/memory/stream_pool.cpp",
    "core/memory/kv_cache_pool.cpp",
    "core/base/thread.cc",
    "core/base/exception.cc",
    "core/base/date.cc",
    "core/base/process_info.cc",
    "core/base/logging.cc",
    "core/base/log_file.cc",
    "core/base/timestamp.cc",
    "core/base/file_util.cc",
    "core/base/countdown_latch.cc",
    "core/base/timezone.cc",
    "core/base/log_stream.cc",
    "core/base/thread_pool.cc",
    "core/python/py_kv_cache.cpp",
]

_PAGED_ATTN_SOURCES = [
    "extensions/kernel/paged_attention.cu",
]

_MARLIN_SOURCES = [
    "moe_infinity/kernel/marlin/marlin_cuda.cpp",
    "moe_infinity/kernel/marlin/marlin_cuda_kernel.cu",
]

# Note: _engine needs CUTLASS for fused_glu_cuda.cu

ext_modules = []

if cuda_available:
    _cuda_arch_flags = ["-gencode=arch=compute_80,code=sm_80"]
    if os.environ.get("MOE_ENABLE_SM90", "1") == "1":
        _cuda_arch_flags.append("-gencode=arch=compute_90,code=sm_90")
    if os.environ.get("MOE_ENABLE_SM120", "0") == "1":
        _cuda_arch_flags.append("-gencode=arch=compute_120,code=sm_120")

    # _store extension: IO and prefetch
    ext_modules.append(
        cpp_extension.CUDAExtension(
            name="moe_infinity._store",
            sources=_STORE_SOURCES,
            include_dirs=COMMON_INCLUDE_PATHS,
            library_dirs=_STORE_LIBRARY_DIRS,
            extra_compile_args={
                "cxx": COMMON_CXX_ARGS,
                "nvcc": COMMON_NVCC_ARGS + _cuda_arch_flags,
            },
            extra_link_args=_STORE_EXTRA_LINK_ARGS,
        )
    )

    # _engine extension: compute kernels (needs CUTLASS)
    ext_modules.append(
        cpp_extension.CUDAExtension(
            name="moe_infinity._engine",
            sources=_ENGINE_SOURCES,
            include_dirs=COMMON_INCLUDE_PATHS,
            extra_compile_args={
                "nvcc": COMMON_NVCC_ARGS
                + _cuda_arch_flags
                + ["-DBF16_AVAILABLE"],
            },
        )
    )

    ext_modules.append(
        cpp_extension.CUDAExtension(
            name="moe_infinity._kv_cache",
            sources=_KV_CACHE_SOURCES,
            include_dirs=COMMON_INCLUDE_PATHS,
            library_dirs=_STORE_LIBRARY_DIRS,
            extra_compile_args={
                "cxx": COMMON_CXX_ARGS,
                "nvcc": COMMON_NVCC_ARGS + _cuda_arch_flags,
            },
            extra_link_args=_STORE_EXTRA_LINK_ARGS,
        )
    )

    ext_modules.append(
        cpp_extension.CUDAExtension(
            name="moe_infinity._paged_attn",
            sources=_PAGED_ATTN_SOURCES,
            include_dirs=COMMON_INCLUDE_PATHS,
            extra_compile_args={
                "nvcc": COMMON_NVCC_ARGS + _cuda_arch_flags,
            },
        )
    )

    _v4fp4_arch_flags = [
        f
        for f in _cuda_arch_flags
        if "compute_80" not in f and "compute_90" not in f
    ]
    if not _v4fp4_arch_flags:
        _v4fp4_arch_flags = ["-gencode=arch=compute_120a,code=sm_120a"]
    else:
        _v4fp4_arch_flags = [
            f.replace("compute_120,code=sm_120", "compute_120a,code=sm_120a")
            for f in _v4fp4_arch_flags
        ]
    ext_modules.append(
        cpp_extension.CUDAExtension(
            name="moe_infinity._v4_fp4",
            sources=[
                "extensions/kernel/v4_fp4/v4_fp4_binding.cpp",
                "extensions/kernel/v4_fp4/v4_fp4_dequant.cu",
                "extensions/kernel/v4_fp4/mxfp4_dequant.cu",
                "extensions/kernel/v4_fp4/fp8_dequant.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"]
                + _v4fp4_arch_flags,
            },
        )
    )

    ext_modules.append(
        cpp_extension.CUDAExtension(
            name="moe_infinity._marlin",
            sources=_MARLIN_SOURCES,
            extra_compile_args={
                "nvcc": ["-O3", "--use_fast_math", "-std=c++17"]
                + _cuda_arch_flags,
            },
        )
    )

cmdclass = {
    "build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)
}

print(f"find_packages: {find_packages()}")

# install all files in the package, rather than just the egg
setup(
    name="moe_infinity",
    # version is supplied dynamically by setuptools-scm (see pyproject.toml
    # [tool.setuptools_scm]); it is derived from git tags at build time and
    # baked into the sdist's PKG-INFO so it survives a source reinstall.
    packages=find_packages(exclude=["extensions", "extensions.*"]),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "flashinfer": ["flashinfer-python"],
        "flash_attn": ["flash-attn>=2.5.2"],
        "contextpilot": ["contextpilot>=0.4.0"],
    },
    author="EfficientMoE Team",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/EfficientMoE/MoE-Infinity",
    project_urls={"Homepage": "https://github.com/EfficientMoE/MoE-Infinity"},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache License 2.0",
    python_requires=">=3.10",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
