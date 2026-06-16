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

"""End-to-end tests for the tool rails at the RailsManager level.

These exercise the full seam this PR builds: the real ModelEngine response
parsing (non-streaming ``chat_completion`` and streaming SSE assembly) feeds
the EngineRegistry tool helpers (``parse_tools`` / ``extract_tool_results``),
whose normalized output is validated by ``RailsManager.are_tool_calls_safe`` /
``are_tool_results_safe``.

The only mock is the aiohttp transport (the model's HTTP/SSE response body), so
ModelEngine's parsing runs for real. The "model response -> parse -> validate"
orchestration here stands in for the future IORails glue; this PR keeps the
RailsManager surface limited to the two validation methods.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemoguardrails.guardrails.engine_registry import EngineRegistry
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.guardrails.rails_manager import RailsManager
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.types import ChatMessage

STACK_CONFIG = {"models": [{"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"}]}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
LLM_PARAMS = {"tools": [WEATHER_TOOL], "tool_choice": "auto"}
MESSAGES = [{"role": "user", "content": "What's the weather in Paris?"}]


def _build_stack() -> tuple[EngineRegistry, RailsManager]:
    """Build an EngineRegistry + RailsManager with both tool rails enabled."""
    config = RailsConfig.from_content(config=STACK_CONFIG)
    engine_registry = EngineRegistry(config.models, config.rails.config)
    rails_manager = RailsManager(
        engine_registry=engine_registry,
        task_manager=LLMTaskManager(config),
        input_flows=[],
        output_flows=[],
        tool_call_flows=["tool call validation"],
        tool_result_flows=["tool result validation"],
    )
    return engine_registry, rails_manager


def _inject_json_response(registry: EngineRegistry, payload: dict) -> None:
    """Wire the 'main' engine's aiohttp client to return *payload* as JSON."""
    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=payload)
    mock_client = AsyncMock()
    mock_client.post = MagicMock(return_value=mock_response)
    mock_client.closed = False
    engine = registry._get_engine("main", ModelEngine)
    engine._client = mock_client
    engine._running = True


def _inject_sse_stream(registry: EngineRegistry, raw_lines: list) -> None:
    """Wire the 'main' engine's aiohttp client to stream *raw_lines* via readline()."""
    all_lines = []
    for raw in raw_lines:
        for part in raw.split(b"\n"):
            if part:
                all_lines.append(part + b"\n")
    line_iter = iter(all_lines)

    async def _readline():
        return next(line_iter, b"")

    mock_content = MagicMock()
    mock_content.readline = _readline

    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.status = 200
    mock_response.content = mock_content

    mock_client = AsyncMock()
    mock_client.post = MagicMock(return_value=mock_response)
    mock_client.closed = False
    engine = registry._get_engine("main", ModelEngine)
    engine._client = mock_client
    engine._running = True


def _tool_call_payload(name: str, arguments_json: str, call_id: str = "call_1") -> dict:
    """A non-streaming /v1/chat/completions response carrying a single tool call."""
    return {
        "id": "chatcmpl-1",
        "model": "meta/llama-3.3-70b-instruct",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments_json},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _sse(chunk: dict) -> bytes:
    return ("data: " + json.dumps(chunk) + "\n\n").encode("utf-8")


def _tool_call_sse_lines(name: str, arg_fragments: list, call_id: str = "call_1") -> list:
    """SSE lines streaming a single tool call: id/name first, then argument fragments, then finish."""
    chunks = [
        {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
    ]
    for fragment in arg_fragments:
        chunks.append(
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]},
                        "finish_reason": None,
                    }
                ],
            }
        )
    chunks.append({"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    lines = [_sse(chunk) for chunk in chunks]
    lines.append(b"data: [DONE]\n\n")
    return lines


def _result_messages(result_call_id: str = "call_1") -> list:
    """A conversation with an assistant tool call and a following tool result."""
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": result_call_id, "name": "get_weather", "content": "18C"},
    ]


class TestToolCallRailEndToEndNonStreaming:
    @pytest.mark.asyncio
    async def test_allowed_call_with_valid_arguments_passes(self):
        registry, manager = _build_stack()
        _inject_json_response(registry, _tool_call_payload("get_weather", '{"city": "Paris"}'))

        toolset = registry.parse_tools("main", LLM_PARAMS)
        response = await registry.model_call("main", MESSAGES, **LLM_PARAMS)

        assert response.tool_calls is not None
        result = await manager.are_tool_calls_safe(response.tool_calls, toolset)
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_undeclared_tool_call_is_blocked(self):
        registry, manager = _build_stack()
        _inject_json_response(registry, _tool_call_payload("rm_rf", "{}"))

        toolset = registry.parse_tools("main", LLM_PARAMS)
        response = await registry.model_call("main", MESSAGES, **LLM_PARAMS)

        assert response.tool_calls is not None
        result = await manager.are_tool_calls_safe(response.tool_calls, toolset)
        assert result.is_safe is False
        assert result.reason is not None
        assert "rm_rf" in result.reason

    @pytest.mark.asyncio
    async def test_invalid_arguments_are_blocked(self):
        registry, manager = _build_stack()
        _inject_json_response(registry, _tool_call_payload("get_weather", "{}"))

        toolset = registry.parse_tools("main", LLM_PARAMS)
        response = await registry.model_call("main", MESSAGES, **LLM_PARAMS)

        assert response.tool_calls is not None
        result = await manager.are_tool_calls_safe(response.tool_calls, toolset)
        assert result.is_safe is False
        assert result.reason is not None
        assert "get_weather" in result.reason


class TestToolResultRailEndToEnd:
    @pytest.mark.asyncio
    async def test_linked_result_passes(self):
        registry, manager = _build_stack()
        messages = _result_messages(result_call_id="call_1")

        tool_results = registry.extract_tool_results("main", messages)
        prior_calls = ChatMessage.from_dict(messages[1]).tool_calls

        assert prior_calls is not None
        result = await manager.are_tool_results_safe(tool_results, prior_calls)
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_unlinked_result_is_blocked(self):
        registry, manager = _build_stack()
        messages = _result_messages(result_call_id="call_999")

        tool_results = registry.extract_tool_results("main", messages)
        prior_calls = ChatMessage.from_dict(messages[1]).tool_calls

        assert prior_calls is not None
        result = await manager.are_tool_results_safe(tool_results, prior_calls)
        assert result.is_safe is False
        assert result.reason is not None
        assert "call_999" in result.reason


class TestToolCallRailEndToEndStreaming:
    @pytest.mark.asyncio
    async def test_streamed_allowed_call_passes(self):
        registry, manager = _build_stack()
        _inject_sse_stream(registry, _tool_call_sse_lines("get_weather", ['{"city": ', '"Paris"}']))

        toolset = registry.parse_tools("main", LLM_PARAMS)
        collected = []
        async for chunk in registry.stream_model_call("main", MESSAGES, **LLM_PARAMS):
            if chunk.delta_tool_calls:
                collected.extend(chunk.delta_tool_calls)

        assert collected, "expected assembled tool calls from the stream"
        result = await manager.are_tool_calls_safe(collected, toolset)
        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_streamed_undeclared_call_is_blocked(self):
        registry, manager = _build_stack()
        _inject_sse_stream(registry, _tool_call_sse_lines("rm_rf", ["{}"]))

        toolset = registry.parse_tools("main", LLM_PARAMS)
        collected = []
        async for chunk in registry.stream_model_call("main", MESSAGES, **LLM_PARAMS):
            if chunk.delta_tool_calls:
                collected.extend(chunk.delta_tool_calls)

        assert collected, "expected assembled tool calls from the stream"
        result = await manager.are_tool_calls_safe(collected, toolset)
        assert result.is_safe is False
        assert result.reason is not None
        assert "rm_rf" in result.reason
