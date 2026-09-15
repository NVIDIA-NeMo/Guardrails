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

"""Integration tests for the tool safety check rail wired into IORails.

The IORails-level companion to test_tool_safety_check_actions.py (which stops at the
action surface), same relationship test_per_tool_regex_rails_iorails.py has to
test_per_tool_regex_rails.py. Reuses test_tool_rails_iorails.py's payload builders, but
defines its own transport-injection helpers since those hardcode the "main" engine and
this rail also needs to mock a second (judge) model engine.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from tests.guardrails.async_helpers import started_iorails
from tests.guardrails.test_tool_rails_iorails import _tool_call_payload

RUN_SQL_TOOL = {
    "type": "function",
    "function": {"name": "run_sql", "parameters": {"type": "object", "additionalProperties": True}},
}

_OUTPUT_TASK = "tool_safety_check_output $model=llama_guard $variant=run_sql_check"
_INPUT_TASK = "tool_safety_check_input $model=llama_guard $variant=run_sql_check"

JUDGE_MODEL = {"type": "llama_guard", "engine": "nim", "model": "meta/llama-guard-3-8b"}

# @tool_output_validation blocks a call to an undeclared tool, so run_sql is declared here
# with a permissive schema, matching CONFIG_TOOLS_CONFIG in test_tool_rails_iorails.py.
TOOL_OUTPUT_CONFIG = {
    "models": [
        {
            "type": "main",
            "engine": "nim",
            "model": "meta/llama-3.3-70b-instruct",
            "parameters": {"tools": [RUN_SQL_TOOL]},
        },
        JUDGE_MODEL,
    ],
    "rails": {
        "tool_output": {"per_tool": {"run_sql": ["tool safety check output $model=llama_guard $variant=run_sql_check"]}}
    },
    "prompts": [
        {
            "task": _OUTPUT_TASK,
            "content": "Tool: {{ tool_name }}\nArguments: {{ tool_call_arguments }}\nRespond safe or unsafe: <reason>.",
        }
    ],
}

# No tools declared on the model, so run_sql is unresolvable -> @tool_output_validation
# blocks before the judge model is ever called.
TOOL_OUTPUT_UNDECLARED_CONFIG = {
    "models": [
        {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"},
        JUDGE_MODEL,
    ],
    "rails": {
        "tool_output": {"per_tool": {"run_sql": ["tool safety check output $model=llama_guard $variant=run_sql_check"]}}
    },
    "prompts": [
        {
            "task": _OUTPUT_TASK,
            "content": "Tool: {{ tool_name }}\nArguments: {{ tool_call_arguments }}\nRespond safe or unsafe: <reason>.",
        }
    ],
}

TOOL_INPUT_CONFIG = {
    "models": [
        {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"},
        JUDGE_MODEL,
    ],
    "rails": {
        "tool_input": {"per_tool": {"run_sql": ["tool safety check input $model=llama_guard $variant=run_sql_check"]}}
    },
    "prompts": [
        {
            "task": _INPUT_TASK,
            "content": "Tool: {{ tool_name }}\nResult: {{ tool_result_content }}\nRespond safe or unsafe: <reason>.",
        }
    ],
}

MESSAGES = [{"role": "user", "content": "run a query"}]


def _tool_result_conversation(content: str) -> list:
    return [
        {"role": "user", "content": "run a query"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "run_sql", "content": content},
    ]


def _judge_payload(text: str) -> dict:
    return {
        "id": "chatcmpl-judge",
        "model": "meta/llama-guard-3-8b",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


def _inject_json_response(iorails: IORails, model_type: str, payload: dict) -> None:
    """Wire *model_type*'s engine aiohttp client to return *payload* as JSON."""
    mock_response = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=payload)
    mock_client = AsyncMock()
    mock_client.post = MagicMock(return_value=mock_response)
    mock_client.closed = False
    engine = iorails.engine_registry._get_engine(model_type, ModelEngine)
    engine._client = mock_client
    engine._running = True


def _inject_forbidden_transport(iorails: IORails, model_type: str) -> MagicMock:
    """Wire *model_type*'s engine with a transport whose ``post`` must never be called."""
    mock_client = AsyncMock()
    mock_client.post = MagicMock(side_effect=AssertionError(f"{model_type} model must not be called"))
    mock_client.closed = False
    engine = iorails.engine_registry._get_engine(model_type, ModelEngine)
    engine._client = mock_client
    engine._running = True
    return mock_client.post


@pytest_asyncio.fixture
async def output_iorails():
    async with started_iorails(TOOL_OUTPUT_CONFIG) as engine:
        yield engine


@pytest_asyncio.fixture
async def output_undeclared_iorails():
    async with started_iorails(TOOL_OUTPUT_UNDECLARED_CONFIG) as engine:
        yield engine


@pytest_asyncio.fixture
async def input_iorails():
    async with started_iorails(TOOL_INPUT_CONFIG) as engine:
        yield engine


class TestToolSafetyCheckOutput:
    @pytest.mark.asyncio
    async def test_unsafe_judgment_blocks(self, output_iorails):
        _inject_json_response(output_iorails, "main", _tool_call_payload("run_sql", '{"query": "DROP TABLE users"}'))
        _inject_json_response(output_iorails, "llama_guard", _judge_payload("unsafe: drops a table"))

        result = await output_iorails.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}

    @pytest.mark.asyncio
    async def test_safe_judgment_passes(self, output_iorails):
        _inject_json_response(output_iorails, "main", _tool_call_payload("run_sql", '{"query": "SELECT 1"}'))
        _inject_json_response(output_iorails, "llama_guard", _judge_payload("safe"))

        result = await output_iorails.generate_async(messages=MESSAGES)

        assert result["tool_calls"][0]["function"]["name"] == "run_sql"

    @pytest.mark.asyncio
    async def test_undeclared_tool_blocks_before_judge_call(self, output_undeclared_iorails):
        """@tool_output_validation blocks before the judge model is ever called."""
        _inject_json_response(output_undeclared_iorails, "main", _tool_call_payload("run_sql", '{"query": "SELECT 1"}'))
        forbidden_post = _inject_forbidden_transport(output_undeclared_iorails, "llama_guard")

        result = await output_undeclared_iorails.generate_async(messages=MESSAGES)

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        forbidden_post.assert_not_called()


class TestToolSafetyCheckInput:
    @pytest.mark.asyncio
    async def test_unsafe_judgment_blocks_before_generation(self, input_iorails):
        forbidden_post = _inject_forbidden_transport(input_iorails, "main")
        _inject_json_response(input_iorails, "llama_guard", _judge_payload("unsafe: contains an ssn"))

        result = await input_iorails.generate_async(messages=_tool_result_conversation("ssn: 123-45-6789"))

        assert result == {"role": "assistant", "content": REFUSAL_MESSAGE}
        forbidden_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_safe_judgment_passes(self, input_iorails):
        _inject_json_response(input_iorails, "llama_guard", _judge_payload("safe"))
        _inject_json_response(
            input_iorails,
            "main",
            {
                "id": "chatcmpl-1",
                "model": "meta/llama-3.3-70b-instruct",
                "choices": [
                    {"message": {"role": "assistant", "content": "no sensitive data found"}, "finish_reason": "stop"}
                ],
            },
        )

        result = await input_iorails.generate_async(messages=_tool_result_conversation("no sensitive data"))

        assert result == {"role": "assistant", "content": "no sensitive data found"}
