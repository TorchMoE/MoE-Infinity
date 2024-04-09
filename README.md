# MoE-Infinity

MoE-Infinity is a cost-effective, fast, and easy-to-use library for Mixture-of-Experts (MoE) inference and serving.

MoE-Infinity is cost-effective yet fast:

- Offloading MoE's experts to host memory, allowing memory-constrained GPUs to serve MoE models.
- Minimizing the expert offloading overheads through several novel techniques: expert activation tracing, activation-aware expert prefetching, and activation-aware expert caching.
- Supporting LLM acceleration techniques (such as [FlashAttention](https://github.com/Dao-AILab/flash-attention)).
- Supporting multi-GPU environments with numeorous OS-level performance optimizations. 
- Achieving SOTA latency and throughput performance when serving MoEs in a resource-constrained GPU environment (in comparison with HuggingFace [Accelerate](https://github.com/huggingface/accelerate), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [Mixtral-Offloading](https://github.com/dvmazur/mixtral-offloading), and [Ollama/LLama.cpp](https://github.com/ollama/ollama)).

MoE-Infinity is easy-to-use:

- HuggingFace model compatible, and HuggingFace programmer friendly.
- Supporting all available MoE checkpoints (including [Google Switch Transformers](https://huggingface.co/google/switch-large-128), [Meta NLLB-MoE](https://huggingface.co/facebook/nllb-moe-54b), and [Mixtral](mistralai/Mixtral-8x7B-Instruct-v0.1)).

Note that: The open-sourced MoE-Infinity has been redesigned for making it HuggingFace-users friendly. This version is different from the version reported in the paper, which takes extreme performance as the top priority. As a result, distributed inference is currently not supported in this open-sourced version.

## Contents
- [Performance](#performance)
- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from conda environment](#install-from-conda-environment)
    - [Install from PyPI](#install-from-pypi)
    - [Install from Source](#install-from-source)
    - [Enable FlashAttention (Optional)](#enable-flashattention-optional)
- [Usage and Examples](#usage-and-examples)
    - [Sample Code of Huggingface LLM Inference](#sample-code-of-huggingface-llm-inference)
    - [Running Inference](#running-inference)
- [Release Plan](#release-plan)
- [Citation](#citation)

## Performance

Single GPU A5000 (24GB Memory), per-token-latency (seconds) for generation with a mixed dataset that includes [FLAN](https://huggingface.co/datasets/Muennighoff/flan), [BIG-Bench](https://huggingface.co/datasets/bigbench) and [MMLU](https://huggingface.co/datasets/lukaemon/mmlu) datasets.
Lower per-token-latency is preferable.

|  | switch-large-128 | NLLB-MoE-54B | Mixtral-7x8b |
| :---: | :---: | :---: | :---: |
| <ins>MoE-Infinity</ins> | <ins>*0.230*</ins>	| <ins>*0.239*</ins> | <ins>*0.895*</ins> |
| Accelerate | 1.043 | 3.071 | 6.633 |
|DeepSpeed | 4.578 | 8.381 | 2.486 |
|Mixtral Offloading| X | X | 1.752 | 
|Ollama | X | X | 0.903 |


Single GPU A5000, throughput (token/s) for generation with batch size 32.
Higher throughput is preferable.

|  | switch-large-128 | NLLB-MoE-54B | Mixtral-7x8b |
| :---: | :---: | :---: | :---: |
| <ins>MoE-Infinity</ins> | <ins>*69.105*</ins>	| <ins>*30.300*</ins> | <ins>*12.579*</ins> |
| Accelerate | 5.788 | 4.344 | 1.245 |
|DeepSpeed | 7.416 | 4.334 | 7.727 |
|Mixtral Offloading| X | X | 7.684 | 
|Ollama | X | X | 1.107 |

> The Mixtral Offloading experiment was carried out with a batch size of 16, as utilizing a batch size of 32 would result in Out of Memory errors on the GPU.

> Ollama does not support batching for generation, so the throughput is calculated with a batch size of 1.

## Installation

We recommend installing MoE-Infinity in a virtual environment. To install MoE-Infinity, you can either install it from PyPI or build it from source.

### Install from conda environment

```bash
conda env create --file environment.yml
conda activate moe-infinity
```

### Install from PyPI

```bash
pip install moe-infinity
conda install -c conda-forge libstdcxx-ng=12 # assume using conda, otherwise install libstdcxx-ng=12 using your package manager or gcc=12
```

### Install from Source

```bash
git clone https://github.com/TorchMoE/MoE-Infinity.git
cd MoE-Infinity
pip install -e .
```

### Enable FlashAttention (Optional)

Install FlashAttention (>=2.5.2) for faster inference with the following command.
```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
```
Post-installation, MoE-Infinity will automatically integrate with FlashAttention to enhance performance.

## Usage and Examples

We provide a simple API for diverse setups, including single GPU, multiple GPUs, and multiple nodes. The following examples show how to use MoE-Infinity to run generation on a Huggingface LLM model.

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

## Release Plan

We plan to release two functions in the following months:

* We currently support PyTorch as the default inference engine, and we are in the process of supporting vLLM as another inference runtime, which includes the support of KV cache offloading. 
* Supporting expert parallelism for distributed MoE inference.
* More (We welcome contributors to join us!)

## Citation

If you use MoE-Inifity for your research, please cite our [paper](https://arxiv.org/abs/2401.14361):
```bibtex
@inproceedings{moe-infinity2024,
  title={MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving},
  author={Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, Mahesh Marina},
  booktitle={https://arxiv.org/abs/2401.14361},
  year={2024}
}
```
