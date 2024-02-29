# Copyright 2024 TorchMoE Team

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file includes source code adapted from vLLM 
# (https://github.com/vllm-project/vllm),
# which is also licensed under the Apache License, Version 2.0.

import argparse
import asyncio
import json
import os
import time
from typing import Tuple
from queue import Queue

from transformers import AutoTokenizer, TextStreamer
# from moe_infinity import MoE
from transformers import OPTForCausalLM
import torch

import fastapi
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse, Response

from moe_infinity.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    CompletionRequest, CompletionResponse,
    CompletionResponseChoice,
    ModelPermission, ModelCard, ModelList,
    UsageInfo)
from moe_infinity.entrypoints.openai.protocol import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds
# logger = init_logger(__name__)
model_name = None
model = None
tokenizer = None
model_queue = None

app = fastapi.FastAPI()


class TokenStreamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.token_cache = []
        self.encoded = False

    def put(self, value):
        if self.encoded and value is not None:
            self.token_cache.append(value)
        else:
            self.encoded = True
    
    def end(self):
        pass

    def get_tokens(self):
        return self.token_cache


def parse_prompt_format(prompt) -> Tuple[bool, list]:
    # get the prompt, openai supports the following
    # "a string, array of strings, array of tokens, or array of token arrays."
    prompt_is_tokens = False
    prompts = [prompt]  # case 1: a string
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")
        elif isinstance(prompt[0], str):
            prompt_is_tokens = False
            prompts = prompt  # case 2: array of strings
        elif isinstance(prompt[0], int):
            prompt_is_tokens = True
            prompts = [prompt]  # case 3: array of tokens
        elif isinstance(prompt[0], list) and isinstance(prompt[0][0], int):
            prompt_is_tokens = True
            prompts = prompt  # case 4: array of token arrays
        else:
            raise ValueError(
                "prompt must be a string, array of strings, array of tokens, or array of token arrays"
            )
    return prompt_is_tokens, prompts


def get_available_models() -> ModelList:
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=model_name,
                    root=model_name,
                    permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoE-Infinity OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--offload-dir", type=str, required=True)
    parser.add_argument("--device-memory-ratio", type=float, default=0.75)

    return parser.parse_args()


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


# @app.get("/v1/models")
# async def show_available_models():
#     models = get_available_models()
#     return JSONResponse(content=models.model_dump())


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    model_name = request.model
    created_time = int(time.monotonic())
    request_id = random_uuid()

    prompt = tokenizer.apply_chat_template(
        conversation=request.messages,
        tokenize=False,
        add_generation_prompt=request.add_generation_prompt
    )
    print(f"prompt: {prompt}")

    token_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"token_ids: {token_ids}")
    num_prompt_tokens = token_ids.size(1)

    streamer = TokenStreamer(tokenizer)

    token = model_queue.get()
    _ = model.generate(
        token_ids,
        streamer=streamer,
        **request.to_hf_params()
    )
    model_queue.put(token)

    outputs = torch.tensor(streamer.get_tokens())
    print(f"outputs: {outputs}")
    num_generated_tokens = len(outputs)

    final_res = tokenizer.decode(outputs, skip_special_tokens=True)
    print(f"final_res: {final_res}")

    choices = []
    # role = self.get_chat_request_role(request)
    role = "assistant" # FIXME: hardcoded
    # for output in final_res.outputs:
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role=role, content=final_res),
        finish_reason="stop",
    )
    choices.append(choice_data)

    if request.echo:
        last_msg_content = ""
        if request.messages and isinstance(
                request.messages, list) and request.messages[-1].get(
                    "content") and request.messages[-1].get(
                        "role") == role:
            last_msg_content = request.messages[-1]["content"]

        for choice in choices:
            full_message = last_msg_content + choice.message.content
            choice.message.content = full_message

    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    return response


@app.post("/v1/completions")
async def completion(request: CompletionRequest, raw_request: Request):
    model_name = request.model
    created_time = int(time.monotonic())
    request_id = random_uuid()

    prompt_is_tokens, prompts = parse_prompt_format(request.prompt)

    choices = []
    num_prompt_tokens = 0
    num_generated_tokens = 0
    for i, prompt in enumerate(prompts):
        if prompt_is_tokens:
            input_ids = prompt
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

        streamer = TokenStreamer(tokenizer)

        token = model_queue.get()
        _ = model.generate(
            input_ids,
            streamer=streamer,
            **request.to_hf_params()
        )
        model_queue.put(token)

        outputs = torch.tensor(streamer.get_tokens())
        final_res = tokenizer.decode(outputs, skip_special_tokens=True)

        output_text = final_res

        choice_data = CompletionResponseChoice(
            index=i,
            text=output_text,
            logprobs=None,
            finish_reason="stop",
        )
        choices.append(choice_data)

        num_prompt_tokens += input_ids.size(1)
        num_generated_tokens += len(outputs)
        
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )

    return CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )


if __name__ == "__main__":
    args = parse_args()

    print(f"args: {args}")

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = {
        "offload_path": os.path.join(args.offload_dir, model_name),
        "device_memory_ratio": args.device_memory_ratio,
    }
    # model = MoE(args.model_name_or_path, config)
    model = OPTForCausalLM.from_pretrained(model_name)
    model_queue = Queue()
    model_queue.put("token")

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
