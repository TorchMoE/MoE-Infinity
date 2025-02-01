# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import argparse
import multiprocessing as mp
import os
import time
import warnings
from functools import partial

warnings.filterwarnings("ignore")

import datasets
import torch
from transformers import AutoTokenizer, LlamaTokenizerFast, TextStreamer

from moe_infinity import MoE
from moe_infinity.models.modeling_arctic import ArcticTokenizer


class StopWatch(TextStreamer):
    def __init__(self, engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0
        self.engine = engine

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        elif self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.engine.expert_dispatcher.clear_expert_cache_counts()
            self.start_decoding = time.time()
        self.decoding_iterations += 1

        if self.decoding_iterations % 100 == 0:
            current_time = time.time()
            print(f"Prefilling time: {self.prefilling_time} seconds")
            print(f"Decoding time: {self.decoding_time} seconds")
            print(f"Decoding iterations: {self.decoding_iterations}")
            print(
                f"Decoding time per iteration: {(current_time-self.start_decoding) / self.decoding_iterations} seconds"
            )

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding

        return super().end()


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--offload_dir", type=str, required=True)
parser.add_argument("--device_memory_ratio", type=float, default=0.85)
parser.add_argument("--out_len", type=int, default=32)
args = parser.parse_args()

model_name = args.model_name_or_path.split("/")[-1]

tokenizer = None
if "grok" in model_name:
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "Xenova/grok-1-tokenizer", trust_remote_code=True
    )
elif "arctic" in args.model_name_or_path.lower():
    tokenizer = ArcticTokenizer.from_pretrained(args.model_name_or_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=False
    )

dataset_name = "tasksource/bigbench"
names = datasets.get_dataset_config_names(dataset_name)

# remove empty entry in BIGBench dataset
names.remove("simple_arithmetic_json_multiple_choice")
names.remove("simple_arithmetic_multiple_targets_json")
names.remove("cifar10_classification")

pool = mp.Pool(mp.cpu_count())
all_inputs = [None] * len(names)
all_inputs = pool.map(partial(datasets.load_dataset, dataset_name), names)

all_inputs = [
    text for dataset in all_inputs for text in dataset["validation"]["inputs"]
]

config = {
    "offload_path": os.path.join(args.offload_dir, model_name),
    "device_memory_ratio": args.device_memory_ratio,
}
model = MoE(args.model_name_or_path, config)


custom_kwargs = {}
if "switch" in args.model_name_or_path.lower():
    custom_kwargs = {"decoder_start_token_id": 0}
elif "nllb" in args.model_name_or_path.lower():
    custom_kwargs = {"forced_bos_token_id": 256057}  # translate to French
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

tokenizer.pad_token = tokenizer.eos_token
cnt = 0
max_seq_length = 512
for input_text in all_inputs:
    # repeat the input text 100 times to test the performance
    input_text = input_text * 1000
    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="do_not_pad",
        max_length=max_seq_length,
        return_tensors="pt",
    )
    print("inputs ...")
    print(inputs.input_ids.shape)

    streamer = StopWatch(model.engine, tokenizer)
    with torch.no_grad():
        print("outputs_text ...")
        outputs = model.generate(
            inputs.input_ids.to("cuda:0"),
            streamer=streamer,
            max_new_tokens=args.out_len,
            min_new_tokens=args.out_len,
            attention_mask=inputs.attention_mask,
            do_sample=False,
            # use_cache=False,
            **custom_kwargs,
        )

        print(f"Prefilling time: {streamer.prefilling_time} seconds")
        print(f"Decoding time: {streamer.decoding_time} seconds")
        print(f"Decoding iterations: {streamer.decoding_iterations}")
        print(
            f"Decoding time per iteration: {streamer.decoding_time / streamer.decoding_iterations} seconds"
        )
        print(f"Input tokens: {len(inputs.input_ids[0])}")
