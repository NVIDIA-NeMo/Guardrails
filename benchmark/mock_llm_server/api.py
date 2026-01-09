# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Annotated, AsyncGenerator, Union

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from benchmark.mock_llm_server.config import ModelSettings, get_settings
from benchmark.mock_llm_server.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    Message,
    Model,
    ModelsResponse,
    Usage,
)
from benchmark.mock_llm_server.response_data import (
    calculate_tokens,
    generate_chunk_latencies,
    generate_id,
    get_latency_seconds,
    get_response,
    split_response_into_chunks,
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
    log.debug(
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


async def stream_chat_completion(
    completion_id: str,
    model: str,
    response_content: str,
    config: ModelSettings,
    n_choices: int = 1,
) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events for streaming chat completions.

    Args:
        completion_id: Unique ID for this completion
        model: Model name
        response_content: Full response text to stream
        config: Model settings for latency configuration
        n_choices: Number of choices to generate
    """
    created_timestamp = int(time.time())
    chunks = split_response_into_chunks(response_content)
    latencies = generate_chunk_latencies(config, len(chunks))

    # First chunk with role
    for i in range(n_choices):
        first_response = ChatCompletionStreamResponse(
            id=completion_id,
            object="chat.completion.chunk",
            created=created_timestamp,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=i,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_response.model_dump_json(exclude_none=True)}\n\n"

    # Stream content chunks
    for chunk_idx, chunk in enumerate(chunks):
        await asyncio.sleep(latencies[chunk_idx])

        for i in range(n_choices):
            chunk_response = ChatCompletionStreamResponse(
                id=completion_id,
                object="chat.completion.chunk",
                created=created_timestamp,
                model=model,
                choices=[
                    ChatCompletionStreamChoice(
                        index=i,
                        delta=DeltaMessage(content=chunk),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk_response.model_dump_json(exclude_none=True)}\n\n"

    # Final chunk with finish_reason
    for i in range(n_choices):
        final_response = ChatCompletionStreamResponse(
            id=completion_id,
            object="chat.completion.chunk",
            created=created_timestamp,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=i,
                    delta=DeltaMessage(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_response.model_dump_json(exclude_none=True)}\n\n"

    yield "data: [DONE]\n\n"


async def stream_completion(
    completion_id: str,
    model: str,
    response_text: str,
    config: ModelSettings,
    n: int = 1,
) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events for streaming text completions.

    Args:
        completion_id: Unique ID for this completion
        model: Model name
        response_text: Full response text to stream
        config: Model settings for latency configuration
        n: Number of choices to generate
    """
    created_timestamp = int(time.time())
    chunks = split_response_into_chunks(response_text)
    latencies = generate_chunk_latencies(config, len(chunks))

    # Stream content chunks
    for chunk_idx, chunk in enumerate(chunks):
        await asyncio.sleep(latencies[chunk_idx])

        for i in range(n):
            chunk_response = CompletionStreamResponse(
                id=completion_id,
                object="text_completion",
                created=created_timestamp,
                model=model,
                choices=[
                    CompletionStreamChoice(
                        text=chunk,
                        index=i,
                        logprobs=None,
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {chunk_response.model_dump_json(exclude_none=True)}\n\n"

    # Final chunk with finish_reason
    for i in range(n):
        final_response = CompletionStreamResponse(
            id=completion_id,
            object="text_completion",
            created=created_timestamp,
            model=model,
            choices=[
                CompletionStreamChoice(
                    text="",
                    index=i,
                    logprobs=None,
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_response.model_dump_json(exclude_none=True)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest, config: ModelSettingsDep
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """Create a chat completion."""

    log.debug("/v1/chat/completions request: %s", request)

    # Validate model exists
    _validate_request_model(config, request)

    # Generate dummy response
    response_content = get_response(config)
    completion_id = generate_id("chatcmpl")

    # Handle streaming response
    if request.stream:
        log.debug("/v1/chat/completions streaming response for id: %s", completion_id)
        return StreamingResponse(
            stream_chat_completion(
                completion_id=completion_id,
                model=request.model,
                response_content=response_content,
                config=config,
                n_choices=request.n or 1,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    response_latency_seconds = get_latency_seconds(config)

    # Calculate token usage
    prompt_text = " ".join([msg.content for msg in request.messages])
    prompt_tokens = calculate_tokens(prompt_text)
    completion_tokens = calculate_tokens(response_content)

    # Create response
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


@app.post("/v1/completions", response_model=None)
async def completions(
    request: CompletionRequest, config: ModelSettingsDep
) -> Union[CompletionResponse, StreamingResponse]:
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
    completion_id = generate_id("cmpl")

    # Handle streaming response
    if request.stream:
        log.debug("/v1/completions streaming response for id: %s", completion_id)
        return StreamingResponse(
            stream_completion(
                completion_id=completion_id,
                model=request.model,
                response_text=response_text,
                config=config,
                n=request.n or 1,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    response_latency_seconds = get_latency_seconds(config)

    # Calculate token usage
    prompt_tokens = calculate_tokens(prompt_text)
    completion_tokens = calculate_tokens(response_text)

    # Create response
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
