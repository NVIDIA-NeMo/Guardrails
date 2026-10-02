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

"""Integration tests for the per-tool CEL rails wired into IORails.

The IORails-level companion to tests/test_cel_actions.py, mirroring
test_per_tool_regex_rails_iorails.py and reusing test_tool_rails_iorails.py's transport-mocking
helpers.
"""

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.iorails import INTERNAL_ERROR_MESSAGE, REFUSAL_MESSAGE
from nemoguardrails.imports import check_optional_dependency
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_tool_rails_iorails import (
    _inject_forbidden_transport,
    _inject_json_response,
    _inject_sse_stream,
    _sse,
    _stream_violation_chunks,
    _text_payload,
    _tool_call_payload,
    _tool_call_sse_lines,
)

pytestmark = pytest.mark.skipif(not check_optional_dependency("cel"), reason="requires the 'cel' extra (Python 3.11+)")

RUN_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

MAIN_MODEL = {
    "type": "main",
    "engine": "nim",
    "model": "meta/llama-3.3-70b-instruct",
    "parameters": {"tools": [RUN_SHELL_TOOL]},
}


def _tool_output_config(*expressions: str) -> dict:
    return {
        "models": [MAIN_MODEL],
        "rails": {
            "config": {"cel": {"tool_output": {"run_shell": {"expressions": list(expressions)}}}},
            "tool_output": {"per_tool": {"run_shell": ["cel check tool output"]}},
        },
    }


TOOL_OUTPUT_CONFIG = _tool_output_config(r'args.command.matches("\\brm\\s+-rf")')
# Fails on every call: `flags` is never sent, and the expression has no has() guard.
TOOL_OUTPUT_FAILING_CONFIG = _tool_output_config('args.flags.exists(f, f == "--force")')

TOOL_INPUT_CONFIG = {
    "models": [MAIN_MODEL],
    "rails": {
        "config": {"cel": {"tool_input": {"run_shell": {"expressions": ['result.contains("BEGIN RSA PRIVATE KEY")']}}}},
        "tool_input": {"per_tool": {"run_shell": ["cel check tool input"]}},
    },
}

MESSAGES = [{"role": "user", "content": "clean up the temp directory"}]


def _tool_conversation(result_content: str) -> list:
    return [
        {"role": "user", "content": "show me the key"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_shell", "arguments": '{"command": "cat key.pem"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "run_shell", "content": result_content},
    ]


def _text_sse_lines(text: str) -> list:
    chunks = [
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]},
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return [_sse(chunk) for chunk in chunks] + [b"data: [DONE]\n\n"]


async def _collect(stream) -> list:
    return [chunk async for chunk in stream]


@pytest_asyncio.fixture
async def call_iorails():
    async with started_iorails(TOOL_OUTPUT_CONFIG) as engine:
        yield engine


@pytest_asyncio.fixture
async def failing_call_iorails():
    async with started_iorails(TOOL_OUTPUT_FAILING_CONFIG) as engine:
        yield engine


@pytest_asyncio.fixture
async def result_iorails():
    async with started_iorails(TOOL_INPUT_CONFIG) as engine:
        yield engine


class TestNonStreamingPerToolCallCel:
    @pytest.mark.asyncio
    async def test_matching_expression_blocked(self, call_iorails):
        _inject_json_response(call_iorails, _tool_call_payload("run_shell", '{"command": "rm -rf /tmp"}'))
        result = await call_iorails.generate_async(messages=MESSAGES)
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}

    @pytest.mark.asyncio
    async def test_non_matching_expression_passes(self, call_iorails):
        _inject_json_response(call_iorails, _tool_call_payload("run_shell", '{"command": "ls /tmp"}'))
        result = await call_iorails.generate_async(messages=MESSAGES)
        assert result["tool_calls"][0]["function"]["name"] == "run_shell"

    @pytest.mark.asyncio
    async def test_schema_violation_blocks_before_evaluation(self, call_iorails):
        _inject_json_response(call_iorails, _tool_call_payload("run_shell", '{"command": 123}'))
        result = await call_iorails.generate_async(messages=MESSAGES)
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}

    @pytest.mark.asyncio
    async def test_failing_expression_fails_closed_as_internal_error(self, failing_call_iorails):
        _inject_json_response(failing_call_iorails, _tool_call_payload("run_shell", '{"command": "ls"}'))
        result = await failing_call_iorails.generate_async(messages=MESSAGES)
        assert result == {"role": "assistant", "content": INTERNAL_ERROR_MESSAGE}


class TestStreamingPerToolCallCel:
    @pytest.mark.asyncio
    async def test_matching_expression_blocks_stream(self, call_iorails):
        _inject_sse_stream(call_iorails, _tool_call_sse_lines("run_shell", ['{"command": "rm -rf /tmp"}']))
        chunks = await _collect(call_iorails.stream_async(MESSAGES))
        violations = _stream_violation_chunks(chunks)
        assert len(violations) == 1
        assert violations[0]["error"]["param"] == "tool_output_rails"

    @pytest.mark.asyncio
    async def test_non_matching_expression_streams_through(self, call_iorails):
        _inject_sse_stream(call_iorails, _tool_call_sse_lines("run_shell", ['{"command": "ls /tmp"}']))
        chunks = await _collect(call_iorails.stream_async(MESSAGES))
        assert _stream_violation_chunks(chunks) == []


class TestNonStreamingPerToolResultCel:
    @pytest.mark.asyncio
    async def test_matching_expression_blocked_before_generation(self, result_iorails):
        forbidden_post = _inject_forbidden_transport(result_iorails)
        result = await result_iorails.generate_async(messages=_tool_conversation("-----BEGIN RSA PRIVATE KEY-----"))
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        forbidden_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_matching_expression_passes(self, result_iorails):
        _inject_json_response(result_iorails, _text_payload("The file has no key."))
        result = await result_iorails.generate_async(messages=_tool_conversation("no keys here"))
        assert result == {"role": "assistant", "content": "The file has no key."}


class TestStreamingPerToolResultCel:
    @pytest.mark.asyncio
    async def test_matching_expression_blocks_stream_before_generation(self, result_iorails):
        forbidden_post = _inject_forbidden_transport(result_iorails)
        chunks = await _collect(result_iorails.stream_async(_tool_conversation("-----BEGIN RSA PRIVATE KEY-----")))
        violations = _stream_violation_chunks(chunks)
        assert len(violations) == 1
        assert violations[0]["error"]["param"] == "tool_input_rails"
        forbidden_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_matching_expression_streams_through(self, result_iorails):
        _inject_sse_stream(result_iorails, _text_sse_lines("The file has no key."))
        chunks = await _collect(result_iorails.stream_async(_tool_conversation("no keys here")))
        assert _stream_violation_chunks(chunks) == []
        assert "The file has no key." in "".join(chunk for chunk in chunks if isinstance(chunk, str))
