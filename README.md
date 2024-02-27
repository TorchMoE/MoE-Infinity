# MoE-Infinity

MoE-Infinity is a cost-efficient mixture-of-expert (MoE) serving system that realizes activation-aware expert offloading. MoE-Infinity features sequence-level expert activation tracing, a new approach adept at identifying sparse activations and capturing the temporal locality of MoE inference. By analyzing these traces, MoE-Infinity performs novel activation-aware expert prefetching and caching, substantially reducing the latency overheads usually associated with offloading experts for improved cost performance. Extensive experiments in a cluster show that MoE-Infinity outperforms numerous existing systems and approaches, reducing latency by 4 - 20X and decreasing deployment costs by over 8X for various MoEs.

For more details, please refer to our paper: [MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving](https://arxiv.org/abs/2401.14361).

## Contents
- [Performance](#performance)
- [Installation](#installation)
     - [Prerequisites](#prerequisites)
     - [Install from PyPI](#install-from-pypi)
     - [Install from Source](#install-from-source)
- [Usage and Examples](#usage-and-examples)
     - [Sample Code of Huggingface LLM Inference](#sample-code-of-huggingface-llm-inference)
    - [Running Inference](#running-inference)
- [Roadmap](#roadmap)

## Performance

Single GPU A5000, per-token-latency (seconds) for generation on a mixture of [FLAN](https://huggingface.co/datasets/Muennighoff/flan), [BIG-Bench](https://huggingface.co/datasets/bigbench) and [MMLU](https://huggingface.co/datasets/lukaemon/mmlu) datasets.

|  | switch-large-128 | NLLB-MoE-54B | Mixtral-7x8b |
| :---: | :---: | :---: | :---: |
| *MoE-Infinity* | *0.230*	| *0.239* | *0.895* |
| Accelerate | 1.043 | 3.071 | 6.633 |
|DeepSpeed | 4.578 | 8.381 | 2.486 |
|Mixtral Offloading| X | X | 1.752 | 
|Ollama | X | X | 0.903 |

Single GPU A5000, throughput (token/s) for generation at batch size 32.

|  | switch-large-128 | NLLB-MoE-54B | Mixtral-7x8b |
| :---: | :---: | :---: | :---: |
| *MoE-Infinity* | *69.105*	| *30.300* | *12.579* |
| Accelerate | 5.788 | 4.344 | 1.245 |
|DeepSpeed | 7.416 | 4.334 | 7.727 |
|Mixtral Offloading| X | X | 7.684 | 
|Ollama | X | X | 1.107 |

> The Mixtral Offloading experiment was carried out with a batch size of 16, as utilizing a batch size of 32 would result in Out of Memory errors on the GPU.

## Installation

We recommend installing MoE-Infinity in a virtual environment. To install MoE-Infinity, you can either install it from PyPI or build it from source.

### Prerequisites
MoE-Infinity is currently only supported on Linux, Ensure the following dependencies are installed on your system:

```bash
# example of installing dependencies on ubuntu
sudo apt install build-essential curl libaio-dev libspdlog-dev
```

Pytorch (>=2.0), libstdcxx-ng (>=12.0) and Python (>=3.8) required for MoE-Infinity, please refer to [Pytorch](https://pytorch.org/get-started/locally/) for installation instructions.

### Install from PyPI

```bash
pip install moe-infinity
```

### Install from Source

```bash
git clone https://github.com/TorchMoE/MoE-Infinity.git
cd MoE-Infinity
pip install -e .
```

## Usage and Examples

We provide a intuitive and consistent API for diverse setups, including single GPU, multiple GPUs, and multiple nodes. The following examples show how to use MoE-Infinity to run generation on a Huggingface LLM model.

### Sample Code of Huggingface LLM Inference

```python
import torch
import os
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
from moe_infinity import MoE

user_home = os.path.expanduser('~')

checkpoint = 'TheBloke/Mixtral-8x7B-v0.1-GPTQ'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

config = {
    "offload_path": os.path.join(user_home, "moe-infinity"),
    "device_memory_ratio": 0.75, # 75% of the device memory is used for caching, change the value according to your device memory size on OOM
}

model = MoE(checkpoint, config)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda:0")

output_ids = model.generate(input_ids)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)
```

### Running Inference

This command runs the script on selected GPUs.
```bash
CUDA_VISIBLE_DEVICES=0,1 python script.py
```

We provide a simple example to run inference on a Huggingface LLM model. The script will download the model checkpoint and run inference on the specified input text. The output will be printed to the console.

```bash
CUDA_VISIBLE_DEVICES=0 python example/interface_example.py --model_name_or_path "mistralai/Mixtral-8x7B-Instruct-v0.1" --offload_dir <your local path on SSD> 
```

## Roadmap

- [ ] Open LLM Leaderboard
- [ ] User-defined device-map for expert parallelism
- [ ] Multinode expert offloading
- [ ] Automatic memory size management
- [ ] vLLM KV cache offloading
