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

This mock is purpose-built to emit tool calls. Given a user turn, it returns a tool
call naming the configured tool, with the user's message as the primary argument
value, so a test drives block/allow behavior purely by varying the message it sends,
without reconfiguring the running server per test case. Message content can also be a
JSON object, used as the call's arguments directly, so a test can drive a multi-field
call. Given a turn whose last message is a tool result, it returns plain text,
simulating the model's reply after the tool ran.

A second, optional role, judge mode for LLM-judged rails like tool_safety_check, is
keyed on request.model rather than message shape, so both the main model and the
judge model can point at this same running instance under different configured model
names. In judge mode the response is plain safe/unsafe text (matching the
is_content_safe output parser), deterministic on a trigger substring in the prompt,
the same content-driven pattern as the tool-call branch above.
"""

import json
import os
import time
import uuid
from typing import Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MOCK_MODEL = os.environ.get("MOCK_TOOL_LLM_MODEL", "mock-tool-model")
MOCK_TOOL_NAME = os.environ.get("MOCK_TOOL_LLM_TOOL_NAME", "run_sql")
MOCK_TOOL_ARGUMENT_NAME = os.environ.get("MOCK_TOOL_LLM_ARGUMENT_NAME", "query")
MOCK_TOOL_CALL_DELIMITER = os.environ.get("MOCK_TOOL_LLM_CALL_DELIMITER", "||")
MOCK_JUDGE_MODEL = os.environ.get("MOCK_TOOL_LLM_JUDGE_MODEL", "")
MOCK_JUDGE_UNSAFE_TRIGGER = os.environ.get("MOCK_TOOL_LLM_JUDGE_UNSAFE_TRIGGER", "UNSAFE_TRIGGER")


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

# The last request received per model, so a real-subprocess test can assert on the exact
# rendered prompt it sent, not just the decision it produced.
_last_requests: dict[str, ChatCompletionRequest] = {}


def _last_user_content(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return ""


def _tool_call_specs(messages: list[Message]) -> list[tuple[str, dict]]:
    """One (tool_name, arguments) pair per tool call to emit.

    A user message containing MOCK_TOOL_CALL_DELIMITER ("||" by default) requests
    multiple tool calls in one response, one per delimited part, so a test can drive
    the per-tool fan-out loop by sending e.g. "SELECT 1||DROP TABLE users" and getting
    back two tool calls, one safe and one that should block. A part may itself be
    prefixed "tool_name:content" to name a specific tool for that call (default
    MOCK_TOOL_NAME otherwise), so a test can prove only the tool actually configured
    for a per-tool check is the one that gets evaluated, even when a sibling call in the
    same response carries identical, otherwise-matching content under a different name.

    If content (after the optional "tool_name:" prefix) is itself a JSON object, it is
    used as the arguments dict directly, matching the real OpenAI wire shape where
    `arguments` is just a JSON-serialized object, so a test can drive a multi-field
    call (e.g. for `$argument=` scoping) without any bespoke syntax. Otherwise content
    is wrapped as a single MOCK_TOOL_ARGUMENT_NAME field, as before.
    """
    content = _last_user_content(messages)
    parts = content.split(MOCK_TOOL_CALL_DELIMITER) if MOCK_TOOL_CALL_DELIMITER in content else [content]
    specs = []
    for part in parts:
        if not part:
            continue
        tool_name, sep, rest = part.partition(":")
        tool_name, body = (tool_name, rest) if sep else (MOCK_TOOL_NAME, part)
        if body.lstrip().startswith("{"):
            specs.append((tool_name, json.loads(body)))
        else:
            specs.append((tool_name, {MOCK_TOOL_ARGUMENT_NAME: body}))
    return specs


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MOCK_MODEL, "object": "model"}]}


def _judge_verdict(content: str) -> str:
    if MOCK_JUDGE_UNSAFE_TRIGGER in content:
        return f"unsafe: contains {MOCK_JUDGE_UNSAFE_TRIGGER}"
    return "safe"


@app.get("/debug/last-request/{model}")
async def last_request(model: str) -> ChatCompletionRequest:
    if model not in _last_requests:
        raise HTTPException(status_code=404, detail=f"no request captured for model {model!r}")
    return _last_requests[model]


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    _last_requests[request.model] = request
    if MOCK_JUDGE_MODEL and request.model == MOCK_JUDGE_MODEL:
        response_message = Message(role="assistant", content=_judge_verdict(_last_user_content(request.messages)))
        finish_reason = "stop"
    elif request.messages and request.messages[-1].role == "tool":
        response_message = Message(role="assistant", content="ok")
        finish_reason = "stop"
    else:
        response_message = Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=ToolCallFunction(name=tool_name, arguments=json.dumps(arguments)),
                )
                for tool_name, arguments in _tool_call_specs(request.messages)
            ],
        )
        finish_reason = "tool_calls"

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatCompletionChoice(message=response_message, finish_reason=finish_reason)],
    )
