# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import io
from typing import List
from setuptools import setup, find_packages
import os
import sys

torch_available = True
try:
    import torch  # noqa: F401
except ImportError:
    torch_available = False
    print('[WARNING] Unable to import torch, pre-compiling ops will be disabled. ' \
        'Please visit https://pytorch.org/ to see how to properly install torch on your system.')

ROOT_DIR = os.path.dirname(__file__)

sys.path.insert(0, ROOT_DIR)
# sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from op_builder.all_ops import ALL_OPS
from torch.utils import cpp_extension

RED_START = '\033[31m'
RED_END = '\033[0m'
ERROR = f"{RED_START} [ERROR] {RED_END}"


def fetch_requirements(path):
    with open(path, 'r') as fd:
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

install_requires = fetch_requirements('requirements.txt')

ext_modules = []

BUILD_OP_DEFAULT = int(os.environ.get('BUILD_OPS', 0))

if BUILD_OP_DEFAULT:
    assert torch_available, 'Unable to pre-compile ops without torch installed. Please install torch before attempting to pre-compile ops.'
    compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
    install_ops = dict.fromkeys(ALL_OPS.keys(), False)
    for op_name, builder in ALL_OPS.items():
        if builder is not None:
            op_compatible = builder.is_compatible()
            compatible_ops[op_name] = op_compatible
            if not op_compatible:
                abort(f"Unable to pre-compile {op_name}")
            ext_modules.append(builder.builder())

cmdclass = {
    'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=True)
}

print(f"find_packages: {find_packages()}")

# install all files in the package, rather than just the egg    
setup(
    name='test_proj_8e00a834c8',
    version='0.0.1',
    packages=find_packages(exclude=['op_builder', 'op_builder.*', 'moe_infinity.ops.core.*']),
    package_data={
        'moe_infinity.ops.prefetch': ['**/*.so'],
        'moe_infinity': ['ops/core/**']
    },
    include_package_data=True,
    install_requires=install_requires,
    author='TorchMoE Team',
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/TorchMoE/MoE-Infinity",
    project_urls={'Homepage': 'https://github.com/TorchMoE/MoE-Infinity'},
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license='Apache License 2.0',
    python_requires=">=3.8",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
