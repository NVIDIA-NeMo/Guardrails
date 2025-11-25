# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import logging
import time
from typing import Annotated, Union

from fastapi import Depends, FastAPI, HTTPException, Request

from nemoguardrails.benchmark.mock_llm_server.config import ModelSettings, get_settings
from nemoguardrails.benchmark.mock_llm_server.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Message,
    Model,
    ModelsResponse,
    Usage,
)
from nemoguardrails.benchmark.mock_llm_server.response_data import (
    calculate_tokens,
    generate_id,
    get_latency_seconds,
    get_response,
)

# Create a console logging handler
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # TODO Control this from the CLi args

# Create a formatter to define the log message format
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Create a console handler to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # DEBUG and higher will go to the console
console_handler.setFormatter(formatter)

# Add console handler to logs
log.addHandler(console_handler)


ModelSettingsDep = Annotated[ModelSettings, Depends(get_settings)]


def _validate_request_model(
    config: ModelSettingsDep,
    request: Union[CompletionRequest, ChatCompletionRequest],
) -> None:
    """Check the Completion or Chat Completion `model` field is in our supported model list"""
    if request.model != config.model:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available models: {config.model}",
        )


app = FastAPI(
    title="Mock LLM Server",
    description="OpenAI-compatible mock LLM server for testing and benchmarking",
    version="0.0.1",
)


@app.middleware("http")
async def log_http_duration(request: Request, call_next):
    """
    Middleware to log incoming requests and their responses.
    """
    request_time = time.time()
    response = await call_next(request)
    response_time = time.time()

    duration_seconds = response_time - request_time
    log.info(
        "Request finished: %s, took %.3f seconds",
        response.status_code,
        duration_seconds,
    )
    return response


@app.get("/")
async def root(config: ModelSettingsDep):
    """Root endpoint with basic server information."""
    return {
        "message": "Mock LLM Server",
        "version": "0.0.1",
        "description": f"OpenAI-compatible mock LLM server for model: {config.model}",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions"],
        "model_configuration": config,
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models(config: ModelSettingsDep):
    """List available models."""
    log.debug("/v1/models request")

    model = Model(id=config.model, object="model", created=int(time.time()), owned_by="system")
    response = ModelsResponse(object="list", data=[model])
    log.debug("/v1/models response: %s", response)
    return response


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, config: ModelSettingsDep) -> ChatCompletionResponse:
    """Create a chat completion."""

    log.debug("/v1/chat/completions request: %s", request)

    # Validate model exists
    _validate_request_model(config, request)

    # Generate dummy response
    response_content = get_response(config)
    response_latency_seconds = get_latency_seconds(config)

    # Calculate token usage
    prompt_text = " ".join([msg.content for msg in request.messages])
    prompt_tokens = calculate_tokens(prompt_text)
    completion_tokens = calculate_tokens(response_content)

    # Create response
    completion_id = generate_id("chatcmpl")
    created_timestamp = int(time.time())

    choices = []
    for i in range(request.n or 1):
        choice = ChatCompletionChoice(
            index=i,
            message=Message(role="assistant", content=response_content),
            finish_reason="stop",
        )
        choices.append(choice)

    response = ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created_timestamp,
        model=request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    await asyncio.sleep(response_latency_seconds)
    log.debug("/v1/chat/completions response: %s", response)
    return response


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest, config: ModelSettingsDep) -> CompletionResponse:
    """Create a text completion."""

    log.debug("/v1/completions request: %s", request)

    # Validate model exists
    _validate_request_model(config, request)

    # Handle prompt (can be string or list)
    if isinstance(request.prompt, list):
        prompt_text = " ".join(request.prompt)
    else:
        prompt_text = request.prompt

    # Generate dummy response
    response_text = get_response(config)
    response_latency_seconds = get_latency_seconds(config)

    # Calculate token usage
    prompt_tokens = calculate_tokens(prompt_text)
    completion_tokens = calculate_tokens(response_text)

    # Create response
    completion_id = generate_id("cmpl")
    created_timestamp = int(time.time())

    choices = []
    for i in range(request.n or 1):
        choice = CompletionChoice(text=response_text, index=i, logprobs=None, finish_reason="stop")
        choices.append(choice)

    response = CompletionResponse(
        id=completion_id,
        object="text_completion",
        created=created_timestamp,
        model=request.model,
        choices=choices,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    await asyncio.sleep(response_latency_seconds)
    log.debug("/v1/completions response: %s", response)
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    log.debug("/health request")
    response = {"status": "healthy", "timestamp": int(time.time())}
    log.debug("/health response: %s", response)
    return response
