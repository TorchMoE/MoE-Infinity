# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import torch


def copy_args_to_device(device, args):
    new_args = ()
    if isinstance(args, torch.Tensor):
        return args.to(device)
    for i in range(len(args)):
        if isinstance(args[i], torch.Tensor):
            new_args += (args[i].to(device, non_blocking=True), )
        elif isinstance(args[i], list) or isinstance(args[i], tuple):
            # move_args_to_device(device, *args[i])
            new_args += (copy_args_to_device(device, args[i]), )
        elif isinstance(args[i], dict):
            new_args += (copy_kwargs_to_device(device, args[i]), )
        else:
            new_args += (args[i], )
    # print("new_args", device, new_args)
    return new_args


def copy_kwargs_to_device(device, kwargs):
    new_kwargs = kwargs
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            new_kwargs[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list) or isinstance(value, tuple):
            new_kwargs[key] = copy_args_to_device(device, value)
        else:
            new_kwargs[key] = value
    return new_kwargs
