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

"""
Mock LLM FastAPI Server with OpenAI-compatible interface.

This server provides dummy implementations of OpenAI API endpoints for testing
and benchmarking purposes.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Mock LLM Server",
    description="OpenAI-compatible mock LLM server for testing and benchmarking",
    version="1.0.0",
)


# Pydantic Models for Request/Response validation


class Message(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="The role of the message author")
    content: str = Field(..., description="The content of the message")
    name: Optional[str] = Field(None, description="The name of the author")


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""

    model: str = Field(..., description="ID of the model to use")
    messages: List[Message] = Field(
        ..., description="List of messages comprising the conversation"
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate", ge=1
    )
    temperature: Optional[float] = Field(
        1.0, description="Sampling temperature", ge=0.0, le=2.0
    )
    top_p: Optional[float] = Field(
        1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0
    )
    n: Optional[int] = Field(
        1, description="Number of completions to generate", ge=1, le=128
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream back partial progress"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Sequences where the API will stop generating"
    )
    presence_penalty: Optional[float] = Field(
        0.0, description="Presence penalty", ge=-2.0, le=2.0
    )
    frequency_penalty: Optional[float] = Field(
        0.0, description="Frequency penalty", ge=-2.0, le=2.0
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Modify likelihood of specified tokens"
    )
    user: Optional[str] = Field(
        None, description="Unique identifier representing your end-user"
    )


class CompletionRequest(BaseModel):
    """Text completion request model."""

    model: str = Field(..., description="ID of the model to use")
    prompt: Union[str, List[str]] = Field(
        ..., description="The prompt(s) to generate completions for"
    )
    max_tokens: Optional[int] = Field(
        16, description="Maximum number of tokens to generate", ge=1
    )
    temperature: Optional[float] = Field(
        1.0, description="Sampling temperature", ge=0.0, le=2.0
    )
    top_p: Optional[float] = Field(
        1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0
    )
    n: Optional[int] = Field(
        1, description="Number of completions to generate", ge=1, le=128
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream back partial progress"
    )
    logprobs: Optional[int] = Field(
        None, description="Include log probabilities", ge=0, le=5
    )
    echo: Optional[bool] = Field(
        False, description="Echo back the prompt in addition to completion"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Sequences where the API will stop generating"
    )
    presence_penalty: Optional[float] = Field(
        0.0, description="Presence penalty", ge=-2.0, le=2.0
    )
    frequency_penalty: Optional[float] = Field(
        0.0, description="Frequency penalty", ge=-2.0, le=2.0
    )
    best_of: Optional[int] = Field(
        1, description="Number of completions to generate server-side", ge=1
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Modify likelihood of specified tokens"
    )
    user: Optional[str] = Field(
        None, description="Unique identifier representing your end-user"
    )


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(
        ..., description="Number of tokens in the completion"
    )
    total_tokens: int = Field(..., description="Total number of tokens used")


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int = Field(..., description="The index of this choice")
    message: Message = Field(..., description="The generated message")
    finish_reason: str = Field(
        ..., description="The reason the model stopped generating"
    )


class CompletionChoice(BaseModel):
    """Text completion choice."""

    text: str = Field(..., description="The generated text")
    index: int = Field(..., description="The index of this choice")
    logprobs: Optional[Dict[str, Any]] = Field(
        None, description="Log probability information"
    )
    finish_reason: str = Field(
        ..., description="The reason the model stopped generating"
    )


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(
        ..., description="Unix timestamp when the completion was created"
    )
    model: str = Field(..., description="The model used for completion")
    choices: List[ChatCompletionChoice] = Field(
        ..., description="List of completion choices"
    )
    usage: Usage = Field(..., description="Token usage information")


class CompletionResponse(BaseModel):
    """Text completion response."""

    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(
        ..., description="Unix timestamp when the completion was created"
    )
    model: str = Field(..., description="The model used for completion")
    choices: List[CompletionChoice] = Field(
        ..., description="List of completion choices"
    )
    usage: Usage = Field(..., description="Token usage information")


class Model(BaseModel):
    """Model information."""

    id: str = Field(..., description="Model identifier")
    object: str = Field("model", description="Object type")
    created: int = Field(..., description="Unix timestamp when the model was created")
    owned_by: str = Field(..., description="Organization that owns the model")


class ModelsResponse(BaseModel):
    """Models list response."""

    object: str = Field("list", description="Object type")
    data: List[Model] = Field(..., description="List of available models")


# Dummy data and helper functions

DUMMY_MODELS = [
    {
        "id": "gpt-3.5-turbo",
        "object": "model",
        "created": 1677610602,
        "owned_by": "openai",
    },
    {"id": "gpt-4", "object": "model", "created": 1687882411, "owned_by": "openai"},
    {
        "id": "gpt-4-turbo",
        "object": "model",
        "created": 1712361441,
        "owned_by": "openai",
    },
    {
        "id": "text-davinci-003",
        "object": "model",
        "created": 1669599635,
        "owned_by": "openai",
    },
]

DUMMY_CHAT_RESPONSES = [
    "This is a mock response from the LLM server.",
    "I'm a dummy AI assistant created for testing purposes.",
    "This response is generated by a mock OpenAI-compatible server.",
    "Hello! I'm responding with dummy data for benchmarking.",
    "This is a simulated conversation response for testing.",
]

DUMMY_COMPLETION_RESPONSES = [
    " This is a dummy text completion.",
    " Here's some mock generated text.",
    " This is a sample completion response.",
    " Mock completion text for testing purposes.",
    " Dummy text generated by the mock server.",
]


def generate_id(prefix: str = "chatcmpl") -> str:
    """Generate a unique ID for completions."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def calculate_tokens(text: str) -> int:
    """Rough token calculation (approximately 4 characters per token)."""
    return max(1, len(text) // 4)


def get_dummy_chat_response() -> str:
    """Get a dummy chat response."""
    import random

    return random.choice(DUMMY_CHAT_RESPONSES)


def get_dummy_completion_response() -> str:
    """Get a dummy completion response."""
    import random

    return random.choice(DUMMY_COMPLETION_RESPONSES)


# API Endpoints


@app.get("/")
async def root():
    """Root endpoint with basic server information."""
    return {
        "message": "Mock LLM Server",
        "version": "1.0.0",
        "description": "OpenAI-compatible mock LLM server for testing and benchmarking",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/completions"],
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models."""
    return ModelsResponse(
        object="list", data=[Model(**model) for model in DUMMY_MODELS]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Create a chat completion."""
    # Validate model exists
    available_models = [model["id"] for model in DUMMY_MODELS]
    if request.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available models: {available_models}",
        )

    # Generate dummy response
    response_content = get_dummy_chat_response()

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
            message=Message(role="assistant", content=response_content, name=None),
            finish_reason="stop",
        )
        choices.append(choice)

    return ChatCompletionResponse(
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


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """Create a text completion."""
    # Validate model exists
    available_models = [model["id"] for model in DUMMY_MODELS]
    if request.model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found. Available models: {available_models}",
        )

    # Handle prompt (can be string or list)
    if isinstance(request.prompt, list):
        prompt_text = " ".join(request.prompt)
    else:
        prompt_text = request.prompt

    # Generate dummy response
    response_text = get_dummy_completion_response()

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

    return CompletionResponse(
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    uvicorn.run(
        "mock_llm_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
