# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import gc
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from accelerate.utils.constants import WEIGHTS_NAME, SAFE_WEIGHTS_NAME


def get_checkpoint_paths(checkpoint: Union[str, os.PathLike]):
    """
    Returns the paths to the checkpoint files in a given folder.

    Args:
        checkpoint_folder (`str` or `os.PathLike`):
            The folder where we will look for the checkpoint files.
    """
    checkpoint_files = None
    index_filename = None
    if os.path.isfile(checkpoint):
        if str(checkpoint).endswith(".json"):
            index_filename = checkpoint
        else:
            checkpoint_files = [checkpoint]
    elif os.path.isdir(checkpoint):
        # check if the whole state dict is present
        potential_state_bin = [f for f in os.listdir(checkpoint) if f == WEIGHTS_NAME]
        potential_state_safetensor = [f for f in os.listdir(checkpoint) if f == SAFE_WEIGHTS_NAME]
        if len(potential_state_bin) == 1:
            checkpoint_files = [os.path.join(checkpoint, potential_state_bin[0])]
        elif len(potential_state_safetensor) == 1:
            checkpoint_files = [os.path.join(checkpoint, potential_state_safetensor[0])]
        else:
            # otherwise check for sharded checkpoints
            potential_index = [f for f in os.listdir(checkpoint) if f.endswith(".index.json")]
            if len(potential_index) == 0:
                raise ValueError(
                    f"{checkpoint} is not a folder containing a `.index.json` file or a {WEIGHTS_NAME} or a {SAFE_WEIGHTS_NAME} file"
                )
            elif len(potential_index) == 1:
                index_filename = os.path.join(checkpoint, potential_index[0])
            else:
                raise ValueError(
                    f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
                )
    else:
        raise ValueError(
            "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
            f"checkpoint, or a folder containing a sharded checkpoint or the whole state dict, but got {checkpoint}."
        )

    if index_filename is not None:
        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename, "r") as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]

    return checkpoint_files
