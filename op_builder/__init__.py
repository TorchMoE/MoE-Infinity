#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# op_builder/__init__.py
#
# Part of the DeepSpeed Project, under the Apache-2.0 License.
# See https://github.com/microsoft/DeepSpeed/blob/master/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0

# MoE-Infinity: replced builder_closure with PrefetchBuilder

import sys
import os
import pkgutil
import importlib

from .builder import get_default_compute_capabilities, OpBuilder
from .prefetch import PrefetchBuilder

# Do not remove, required for abstract accelerator to detect if we have a deepspeed or 3p op_builder
__deepspeed__ = True

# List of all available op builders from deepspeed op_builder
try:
    import moe_infinity.ops.op_builder  # noqa: F401
    op_builder_dir = "moe_infinity.ops.op_builder"
except ImportError:
    op_builder_dir = "op_builder"

this_module = sys.modules[__name__]


def builder_closure(member_name):
    return PrefetchBuilder


# reflect builder names and add builder closure, such as 'TransformerBuilder()' creates op builder wrt current accelerator
for _, module_name, _ in pkgutil.iter_modules(
    [os.path.dirname(this_module.__file__)]):
    if module_name != 'all_ops' and module_name != 'builder':
        module = importlib.import_module(f".{module_name}",
                                         package=op_builder_dir)
        for member_name in module.__dir__():
            if member_name.endswith(
                    'Builder'
            ) and member_name != "OpBuilder" and member_name != "CUDAOpBuilder":
                # assign builder name to variable with same name
                # the following is equivalent to i.e. TransformerBuilder = "TransformerBuilder"
                this_module.__dict__[member_name] = builder_closure(
                    member_name)
