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

"""Integration tests for the per-tool regex rails wired into IORails.

The IORails-level companion to test_per_tool_regex_rails.py (which stops at the
RailsManager surface), same relationship test_tool_rails_iorails.py has to
test_tool_rails_e2e.py. Reuses that module's transport-mocking helpers rather than
duplicating them.
"""

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_tool_rails_iorails import (
    _inject_forbidden_transport,
    _inject_json_response,
    _inject_sse_stream,
    _stream_violation_chunks,
    _text_payload,
    _tool_call_payload,
    _tool_call_sse_lines,
)

BASE_CONFIG = {"models": [{"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"}]}

# @tool_output_validation blocks a call to an undeclared tool, so run_sql is declared here
# with a permissive schema -- these tests exercise the per-tool regex check, not schema
# validation. Declared on the model (models[].parameters.tools) rather than per-request,
# matching CONFIG_TOOLS_CONFIG in test_tool_rails_iorails.py.
RUN_SQL_TOOL = {
    "type": "function",
    "function": {"name": "run_sql", "parameters": {"type": "object", "additionalProperties": True}},
}

TOOL_OUTPUT_PATTERN_CONFIG = {
    "models": [
        {
            "type": "main",
            "engine": "nim",
            "model": "meta/llama-3.3-70b-instruct",
            "parameters": {"tools": [RUN_SQL_TOOL]},
        }
    ],
    "rails": {
        "config": {"regex_detection": {"tool_output": {"run_sql": {"patterns": [r"DROP\s+TABLE"]}}}},
        "tool_output": {"per_tool": {"run_sql": ["regex check tool output"]}},
    },
}

STRICT_RUN_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

TOOL_OUTPUT_SCHEMA_CONFIG = {
    "models": [
        {
            "type": "main",
            "engine": "nim",
            "model": "meta/llama-3.3-70b-instruct",
            "parameters": {"tools": [STRICT_RUN_SQL_TOOL]},
        }
    ],
    "rails": {
        "config": {"regex_detection": {"tool_output": {"run_sql": {"patterns": [r"DROP\s+TABLE"]}}}},
        "tool_output": {"per_tool": {"run_sql": ["regex check tool output"]}},
    },
}

TOOL_INPUT_PATTERN_CONFIG = {
    **BASE_CONFIG,
    "rails": {
        "config": {"regex_detection": {"tool_input": {"run_sql": {"patterns": [r"ssn:\s*\d{3}-\d{2}-\d{4}"]}}}},
        "tool_input": {"per_tool": {"run_sql": ["regex check tool input"]}},
    },
}

MESSAGES = [{"role": "user", "content": "run a query"}]


async def _collect(stream) -> list:
    return [chunk async for chunk in stream]


@pytest_asyncio.fixture
async def call_pattern_iorails():
    async with started_iorails(TOOL_OUTPUT_PATTERN_CONFIG) as engine:
        yield engine


@pytest_asyncio.fixture
async def schema_pattern_iorails():
    async with started_iorails(TOOL_OUTPUT_SCHEMA_CONFIG) as engine:
        yield engine


@pytest_asyncio.fixture
async def result_pattern_iorails():
    async with started_iorails(TOOL_INPUT_PATTERN_CONFIG) as engine:
        yield engine


class TestNonStreamingPerToolCallRegex:
    @pytest.mark.asyncio
    async def test_matching_pattern_blocked(self, call_pattern_iorails):
        _inject_json_response(call_pattern_iorails, _tool_call_payload("run_sql", '{"query": "DROP TABLE users"}'))
        result = await call_pattern_iorails.generate_async(messages=MESSAGES)
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}

    @pytest.mark.asyncio
    async def test_non_matching_pattern_passes(self, call_pattern_iorails):
        _inject_json_response(call_pattern_iorails, _tool_call_payload("run_sql", '{"query": "SELECT 1"}'))
        result = await call_pattern_iorails.generate_async(messages=MESSAGES)
        assert result["tool_calls"][0]["function"]["name"] == "run_sql"

    @pytest.mark.asyncio
    async def test_tool_not_listed_in_per_tool_passes(self, call_pattern_iorails):
        # "other_tool" has no per_tool entry, so its arguments are never checked.
        _inject_json_response(call_pattern_iorails, _tool_call_payload("other_tool", '{"query": "DROP TABLE users"}'))
        result = await call_pattern_iorails.generate_async(messages=MESSAGES)
        assert result["tool_calls"][0]["function"]["name"] == "other_tool"

    @pytest.mark.asyncio
    async def test_schema_violation_blocks_before_regex_check(self, schema_pattern_iorails):
        """@tool_output_validation blocks arguments that violate the declared schema,
        even when the regex pattern itself would not have matched."""
        _inject_json_response(schema_pattern_iorails, _tool_call_payload("run_sql", '{"query": 123}'))
        result = await schema_pattern_iorails.generate_async(messages=MESSAGES)
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}


class TestStreamingPerToolCallRegex:
    @pytest.mark.asyncio
    async def test_matching_pattern_blocks_stream(self, call_pattern_iorails):
        _inject_sse_stream(call_pattern_iorails, _tool_call_sse_lines("run_sql", ['{"query": "DROP TABLE users"}']))
        chunks = await _collect(call_pattern_iorails.stream_async(MESSAGES))
        violations = _stream_violation_chunks(chunks)
        assert len(violations) == 1
        assert violations[0]["error"]["param"] == "tool_output_rails"

    @pytest.mark.asyncio
    async def test_non_matching_pattern_streams_through(self, call_pattern_iorails):
        _inject_sse_stream(call_pattern_iorails, _tool_call_sse_lines("run_sql", ['{"query": "SELECT 1"}']))
        chunks = await _collect(call_pattern_iorails.stream_async(MESSAGES))
        assert _stream_violation_chunks(chunks) == []


class TestNonStreamingPerToolResultRegex:
    def _tool_conversation(self, content: str) -> list:
        return [
            {"role": "user", "content": "run a query"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "run_sql", "content": content},
        ]

    @pytest.mark.asyncio
    async def test_matching_pattern_blocked_before_generation(self, result_pattern_iorails):
        forbidden_post = _inject_forbidden_transport(result_pattern_iorails)
        result = await result_pattern_iorails.generate_async(messages=self._tool_conversation("ssn: 123-45-6789"))
        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        forbidden_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_matching_pattern_passes(self, result_pattern_iorails):
        _inject_json_response(result_pattern_iorails, _text_payload("no sensitive data found"))
        result = await result_pattern_iorails.generate_async(messages=self._tool_conversation("no sensitive data"))
        assert result == {"role": "assistant", "content": "no sensitive data found"}
