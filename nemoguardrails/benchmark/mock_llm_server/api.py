# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import time
from typing import Annotated, Optional, Union

from fastapi import Depends, FastAPI, HTTPException

from nemoguardrails.benchmark.mock_llm_server.config import AppModelConfig, get_config
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
    DUMMY_MODELS,
    calculate_tokens,
    generate_id,
    get_dummy_chat_response,
    get_dummy_completion_response,
)


def _validate_request_model(
    request: Union[CompletionRequest, ChatCompletionRequest],
) -> None:
    """Check the Completion or Chat Completion `model` field is in our supported model list"""
    available_models = set([model["id"] for model in DUMMY_MODELS])
    if request.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available models: {available_models}",
        )


app = FastAPI(
    title="Mock LLM Server",
    description="OpenAI-compatible mock LLM server for testing and benchmarking",
    version="0.0.1",
)


ModelConfigDep = Annotated[AppModelConfig, Depends(get_config)]


@app.get("/")
async def root(current_config: ModelConfigDep):
    """Root endpoint with basic server information."""
    return {
        "message": "Mock LLM Server",
        "version": "0.0.1",
        "description": "OpenAI-compatible mock LLM server for testing and benchmarking",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions"],
        "model_configuration": current_config,
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models."""
    return ModelsResponse(
        object="list", data=[Model(**model) for model in DUMMY_MODELS]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest, config: ModelConfigDep
) -> ChatCompletionResponse:
    """Create a chat completion."""
    # Validate model exists
    _validate_request_model(request)

    # Generate dummy response
    response_content = get_dummy_chat_response(config)

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

    return response


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    request: CompletionRequest, config: ModelConfigDep
) -> CompletionResponse:
    """Create a text completion."""

    # Validate model exists
    _validate_request_model(request)

    # Handle prompt (can be string or list)
    if isinstance(request.prompt, list):
        prompt_text = " ".join(request.prompt)
    else:
        prompt_text = request.prompt

    # Generate dummy response
    response_text = get_dummy_completion_response(config)

    # Calculate token usage
    prompt_tokens = calculate_tokens(prompt_text)
    completion_tokens = calculate_tokens(response_text)

    # Create response
    completion_id = generate_id("cmpl")
    created_timestamp = int(time.time())

    choices = []
    for i in range(request.n or 1):
        choice = CompletionChoice(
            text=response_text, index=i, logprobs=None, finish_reason="stop"
        )
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
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": int(time.time())}
