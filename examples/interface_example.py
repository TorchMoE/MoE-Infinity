# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

from functools import partial
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
import argparse
import datasets
import multiprocessing as mp
from transformers import AutoTokenizer, TextStreamer, LlamaTokenizerFast
from moe_infinity import MoE
from moe_infinity.models.modeling_arctic import ArcticTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--offload_dir", type=str, required=True)
parser.add_argument("--device_memory_ratio", type=float, default=0.85)
args = parser.parse_args()


model_name = args.model_name_or_path.split("/")[-1]

tokenizer = None
if "grok" in model_name:
    tokenizer = LlamaTokenizerFast.from_pretrained("Xenova/grok-1-tokenizer", trust_remote_code=True)
elif "arctic" in args.model_name_or_path.lower():
    tokenizer = ArcticTokenizer.from_pretrained(args.model_name_or_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)

class StopWatch(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.start_decoding = time.time()
        self.decoding_iterations += 1
        return value

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding
        print(f"Prefilling time: {self.prefilling_time}")
        print(f"Decoding time: {self.decoding_time}")
        print(f"Decoding iter time: {self.decoding_time / self.decoding_iterations}")
        return

# streamer = StopWatch(tokenizer)

dataset_name = "cais/mmlu"
# names = datasets.get_dataset_config_names(dataset_name)

# remove empty entry in BIGBench dataset
# names.remove("simple_arithmetic_json_multiple_choice")
# names.remove("simple_arithmetic_multiple_targets_json")
# names.remove("cifar10_classification")

# pool = mp.Pool(mp.cpu_count())
# all_inputs = [None] * len(names)
# all_inputs = pool.map(partial(datasets.load_dataset, dataset_name), names)

# print(datasets.load_dataset(dataset_name))

# all_inputs = []
# for task_name in names:
#     dataset = datasets.load_dataset(dataset_name, task_name)
#     all_inputs.append(dataset)
dataset = datasets.load_dataset(dataset_name)
all_inputs = [text for text in dataset["validation"]["question"] ]

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
elif "arctic" in args.model_name_or_path.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
elif "deepseek" in args.model_name_or_path.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
else:
    raise ValueError(f"Model {args.model_name_or_path} not supported")

print("Cuda available: ", torch.cuda.is_available())
print("Cuda device count: ", torch.cuda.device_count())
print("Cuda current device: ", torch.cuda.current_device())

tokenizer.pad_token = tokenizer.eos_token
for input_text in all_inputs:
    streamer = StopWatch(tokenizer)
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

# CUDA_VISIBLE_DEVICES=0 sudo /mnt/data/xly/.conda/envs/moe-infinity/bin/python interface_example.py --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 --offload_dir /mnt/raid0nvme1/xly/test-data