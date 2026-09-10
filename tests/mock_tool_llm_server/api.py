# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Minimal OpenAI-compatible mock LLM server for per-tool-rail server integration tests.

Unlike benchmark/mock_llm_server (built for content-safety load testing: text in, text
out), this mock is purpose-built to emit tool calls. Given a user turn, it returns a
tool call naming the configured tool, with the user's message as the sole argument
value -- so a test drives block/allow behavior purely by varying the message it sends,
without reconfiguring the running server per test case. Given a turn whose last message
is a tool result, it returns plain text, simulating the model's reply after the tool ran.
"""

import json
import os
import time
import uuid
from typing import Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

MOCK_MODEL = os.environ.get("MOCK_TOOL_LLM_MODEL", "mock-tool-model")
MOCK_TOOL_NAME = os.environ.get("MOCK_TOOL_LLM_TOOL_NAME", "run_sql")
MOCK_TOOL_ARGUMENT_NAME = os.environ.get("MOCK_TOOL_LLM_ARGUMENT_NAME", "query")
MOCK_TOOL_CALL_DELIMITER = os.environ.get("MOCK_TOOL_LLM_CALL_DELIMITER", "||")


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: Optional[bool] = False
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


app = FastAPI(title="Mock Tool LLM Server")


def _last_user_content(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return ""


def _tool_call_specs(messages: list[Message]) -> list[tuple[str, str]]:
    """One (tool_name, content) pair per tool call to emit.

    A user message containing MOCK_TOOL_CALL_DELIMITER ("||" by default) requests
    multiple tool calls in one response, one per delimited part -- so a test can drive
    the per-tool fan-out loop by sending e.g. "SELECT 1||DROP TABLE users" and getting
    back two tool calls, one safe and one that should block. A part may itself be
    prefixed "tool_name:content" to name a specific tool for that call (default
    MOCK_TOOL_NAME otherwise) -- so a test can prove only the tool actually configured
    for a per-tool check is the one that gets evaluated, even when a sibling call in the
    same response carries identical, otherwise-matching content under a different name.
    """
    content = _last_user_content(messages)
    parts = content.split(MOCK_TOOL_CALL_DELIMITER) if MOCK_TOOL_CALL_DELIMITER in content else [content]
    specs = []
    for part in parts:
        if not part:
            continue
        tool_name, sep, rest = part.partition(":")
        specs.append((tool_name, rest) if sep else (MOCK_TOOL_NAME, part))
    return specs


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MOCK_MODEL, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    if request.messages and request.messages[-1].role == "tool":
        response_message = Message(role="assistant", content="ok")
        finish_reason = "stop"
    else:
        response_message = Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=ToolCallFunction(name=tool_name, arguments=json.dumps({MOCK_TOOL_ARGUMENT_NAME: content})),
                )
                for tool_name, content in _tool_call_specs(request.messages)
            ],
        )
        finish_reason = "tool_calls"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(message=response_message, finish_reason=finish_reason)],
    )
