# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

from functools import partial
import os
import time
import torch
import argparse
import datasets
import multiprocessing as mp
from transformers import AutoTokenizer, TextStreamer, LlamaTokenizerFast
from moe_infinity import MoE
from moe_infinity.models.arctic import ArcticTokenizer

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
        
        return super().put(value)

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding
        
        return super().end()

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--offload_dir", type=str, required=True)
parser.add_argument("--device_memory_ratio", type=float, default=0.75)
args = parser.parse_args()

model_name = args.model_name_or_path.split("/")[-1]

tokenizer = None
if "arctic" in args.model_name_or_path.lower():
    tokenizer = ArcticTokenizer.from_pretrained(args.model_name_or_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
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
elif "arctic" in args.model_name_or_path.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
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
    
    streamer = StopWatch(tokenizer)

    with torch.no_grad():
        print("outputs_text ...")
        start_time = time.time()
        outputs = model.generate(
            inputs.input_ids.to("cuda:0"),
            streamer=streamer,
            max_new_tokens=20,
            attention_mask=inputs.attention_mask,
            do_sample=False,
            **custom_kwargs,
        )
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Prefilling time: {streamer.prefilling_time} seconds")
        print(f"Decoding time: {streamer.decoding_time} seconds")
        print(f"Decoding iterations: {streamer.decoding_iterations}")
        print(f"Input tokens: {len(inputs.input_ids[0])}")
