# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

from functools import partial
import os
import torch
import argparse
import datasets
import multiprocessing as mp
from transformers import AutoTokenizer, TextStreamer, LlamaTokenizerFast
from moe_infinity import MoE

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--offload_dir", type=str, required=True)
parser.add_argument("--device_memory_ratio", type=float, default=0.75)
args = parser.parse_args()

model_name = args.model_name_or_path.split("/")[-1]

if "grok" in model_name:
    tokenizer = LlamaTokenizerFast.from_pretrained("Xenova/grok-1-tokenizer", trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer)

dataset_name = "tasksource/bigbench"
names = datasets.get_dataset_config_names(dataset_name)

# remove empty entry in BIGBench dataset
names.remove("simple_arithmetic_json_multiple_choice")
names.remove("simple_arithmetic_multiple_targets_json")
names.remove("cifar10_classification")

pool = mp.Pool(mp.cpu_count())
all_inputs = [None] * len(names)
all_inputs = pool.map(partial(datasets.load_dataset, dataset_name), names)

all_inputs = [text for dataset in all_inputs for text in dataset["validation"]["inputs"] ]

config = {
    "offload_path": os.path.join(args.offload_dir, model_name),
    "device_memory_ratio": args.device_memory_ratio,
}
model = MoE(args.model_name_or_path, config)

max_seq_length = 512
custom_kwargs = {}
if "switch" in args.model_name_or_path.lower():
    custom_kwargs = {"decoder_start_token_id": 0}
elif "nllb" in args.model_name_or_path.lower():
    custom_kwargs = {"forced_bos_token_id": 256057} # translate to French
elif "mixtral" in args.model_name_or_path.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
elif "grok" in args.model_name_or_path.lower():
    custom_kwargs = {}
else:
    raise ValueError(f"Model {args.model_name_or_path} not supported")

tokenizer.pad_token = tokenizer.eos_token
for input_text in all_inputs:

    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="do_not_pad",
        max_length=max_seq_length,
        return_tensors="pt",
    )
    print("inputs ...")
    print(input_text)

    with torch.no_grad():
        print("outputs_text ...")
        outputs = model.generate(
            inputs.input_ids.to("cuda:0"),
            streamer=streamer,
            max_new_tokens=20,
            attention_mask=inputs.attention_mask,
            do_sample=False,
            **custom_kwargs,
        )
