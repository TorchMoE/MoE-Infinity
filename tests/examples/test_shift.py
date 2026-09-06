# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import time
import warnings
from functools import partial

import datasets
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer, TextStreamer

from moe_infinity import MoE


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
warnings.filterwarnings("ignore")


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


def load_dataset_parallel(dataset, names, split, key, limit=50):
    # pool = mp.Pool(mp.cpu_count())
    # dataset_list = [None] * len(names)
    # dataset_list = pool.map(partial(datasets.load_dataset, dataset), names)

    def load_single_dataset(dataset, name, split):
        try:
            return datasets.load_dataset(dataset, name, split=split)
        except Exception as e:
            print(f"Error loading dataset {dataset} with name {name}: {e}")
            return None

    dataset_list = Parallel(n_jobs=32)(
        delayed(load_single_dataset)(dataset, name, split)
        for name in tqdm(names)
    )
    dataset_list = [d for d in (dataset_list or []) if d is not None]

    trimmed_datasets = []
    for k, d in enumerate(dataset_list):
        # print(names[k], dataset)
        text_list = []
        for i, text in enumerate(d[key]):
            if i >= limit:
                break
            text_list.append(text)
        trimmed_datasets.extend(text_list)
    while type(trimmed_datasets[0]) is list:
        tmp_list = []
        for x in trimmed_datasets:
            tmp_list.extend(x)
        trimmed_datasets = tmp_list
    print(dataset, len(trimmed_datasets), trimmed_datasets[0])
    return trimmed_datasets

    # all_inputs = [text for dataset in all_inputs for text in enumerate(dataset[split][key])]


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--offload_dir", type=str, default="/home/xly/test_data")
parser.add_argument("--device_memory_ratio", type=float, default=0.75)
parser.add_argument("--rev", action="store_true", default=False)
args = parser.parse_args()

MATH_dir = "/home/xly/MATH/test"
MATH_files = list(
    glob.glob(os.path.join(MATH_dir, "**", "*.json"), recursive=True)
)

# read all json files and get problem
MATH_data = []
for file in MATH_files:
    with open(file, "r") as f:
        data = json.load(f)
        MATH_data.append(data["problem"])
print("MATH_data", len(MATH_data))

model_name = args.model_name_or_path.split("/")[-1]

names_c4 = datasets.get_dataset_config_names("allenai/c4")
names_c4 = ["fy"]
datasets_c4 = load_dataset_parallel(
    "allenai/c4", names_c4, "validation", "text"
)


# load_bigbench_dataset
names_bigbench = datasets.get_dataset_config_names("hails/bigbench")
# print(names_bigbench)
# remove empty entry in BIGBench dataset
# names_bigbench.remove("simple_arithmetic_json_multiple_choice")
# names_bigbench.remove("simple_arithmetic_multiple_targets_json")
# names_bigbench.remove("cifar10_classification")
# names_bigbench.remove("abstract_narrative_understanding")

datasets_bigbench = load_dataset_parallel(
    "hails/bigbench", names_bigbench, "train", "inputs"
)
print("datasets_bigbench", len(datasets_bigbench))

names_mmlu = datasets.get_dataset_config_names("cais/mmlu")
names_mmlu.remove("auxiliary_train")
# print(names_mmlu)

datasets_mmlu = load_dataset_parallel(
    "cais/mmlu", names_mmlu, "test", "question"
)
print("datasets_mmlu", len(datasets_mmlu))

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, trust_remote_code=True
)
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
    custom_kwargs = {"forced_bos_token_id": 256057}  # translate to French
elif "mixtral" in args.model_name_or_path.lower():
    custom_kwargs = {"pad_token_id": tokenizer.eos_token_id}
elif "deepseek" in args.model_name_or_path.lower():
    custom_kwargs = {}
else:
    raise ValueError(f"Model {args.model_name_or_path} not supported")

# random shuffle
random.shuffle(datasets_mmlu)
random.shuffle(datasets_bigbench)
random.shuffle(datasets_c4)

chunk_size = 50
if args.rev:
    all_inputs = (
        datasets_mmlu[:chunk_size]
        + datasets_bigbench[:chunk_size]
        + datasets_c4[:chunk_size]
        + MATH_data[:chunk_size]
    )
else:
    all_inputs = (
        datasets_bigbench[:chunk_size]
        + datasets_mmlu[:chunk_size]
        + datasets_c4[:chunk_size]
        + MATH_data[:chunk_size]
    )

tokenizer.pad_token = tokenizer.eos_token
for i, input_text in enumerate(all_inputs):
    # print(f"dataset {i} with length {len(input_text)}")
    # for input_text in dataset:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Continue writing if the text is incomplete, other wise answer question.",
        },
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer.encode(
        prompt,
        truncation=True,
        padding="do_not_pad",
        max_length=max_seq_length,
        return_tensors="pt",
    )
    print("inputs ...")
    print(prompt)

    streamer = StopWatch(tokenizer)

    with torch.no_grad():
        print("outputs_text ...")
        start_time = time.time()
        outputs = model.generate(
            inputs.to("cuda:0"),
            streamer=streamer,
            max_new_tokens=128,
            # attention_mask=inputs.attention_mask,
            do_sample=False,
            **custom_kwargs,
        )
        end_time = time.time()
        decoding_start = streamer.start_decoding or start_time
        tpot = (end_time - decoding_start) / max(
            streamer.decoding_iterations, 1
        )
        print(f"Time taken: {end_time - start_time} seconds")
        print(f"Prefilling time: {streamer.prefilling_time} seconds")
        print(f"Decoding time: {streamer.decoding_time} seconds")
        print(f"Decoding iterations: {streamer.decoding_iterations}")
        print(f"TPOP: {tpot}")
        print(f"Input tokens: {len(inputs[0])}")
