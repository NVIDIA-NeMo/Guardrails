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

"""Real-process integration tests for the tool_safety_check rail against the actual server.

Same shared harness as test_per_tool_regex_rails_server.py (server_integration_helpers.py).
Both the main model and the judge model point at the same mock tool LLM server instance,
distinguished only by their configured `model` name. The mock's judge mode (keyed on
request.model, enabled here via MOCK_TOOL_LLM_JUDGE_MODEL) returns deterministic safe/unsafe
text driven by a trigger substring in the tool call's arguments or the tool result's content,
so a test drives block/allow purely by varying the message it sends, same as the regex
server tests do.
"""

from pathlib import Path
from typing import Iterator

import httpx
import pytest
import yaml

from tests.guardrails.server_integration_helpers import Servers, spawn_servers

JUDGE_MODEL_NAME = "mock-judge-model"
UNSAFE_TRIGGER = "UNSAFE_TRIGGER"


def _write_config(config_dir: Path, config_id: str, mock_port: int) -> None:
    mock_base_url = f"http://127.0.0.1:{mock_port}/v1"
    config = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "model": "mock-tool-model",
                "parameters": {"base_url": mock_base_url, "api_key": "unused"},
            },
            {
                "type": "judge",
                "engine": "openai",
                "model": JUDGE_MODEL_NAME,
                "parameters": {"base_url": mock_base_url, "api_key": "unused"},
            },
        ],
        # See regex per-tool server test for why this is required for tool-bearing requests.
        "passthrough": True,
        "rails": {
            "tool_output": {
                "per_tool": {
                    "run_sql": ["tool safety check output $model=judge $variant=run_sql_check"],
                    "scoped_tool": ["tool safety check output $model=judge $variant=scoped_check $argument=query"],
                }
            },
            "tool_input": {"per_tool": {"run_sql": ["tool safety check input $model=judge $variant=run_sql_check"]}},
        },
        "prompts": [
            {
                "task": "tool_safety_check_output $model=judge $variant=run_sql_check",
                "content": "Tool: {{ tool_name }}\nArguments: {{ tool_call_arguments }}\nRespond safe or unsafe: <reason>.",
            },
            {
                "task": "tool_safety_check_output $model=judge $variant=scoped_check",
                "content": "Tool: {{ tool_name }}\nArguments: {{ tool_call_arguments }}\nRespond safe or unsafe: <reason>.",
            },
            {
                "task": "tool_safety_check_input $model=judge $variant=run_sql_check",
                "content": "Tool: {{ tool_name }}\nResult: {{ tool_result_content }}\nRespond safe or unsafe: <reason>.",
            },
        ],
    }
    (config_dir / config_id).mkdir(parents=True)
    (config_dir / config_id / "config.yml").write_text(yaml.safe_dump(config))


@pytest.fixture(scope="module")
def servers(tmp_path_factory) -> Iterator[Servers]:
    mock_env = {
        "MOCK_TOOL_LLM_JUDGE_MODEL": JUDGE_MODEL_NAME,
        "MOCK_TOOL_LLM_JUDGE_UNSAFE_TRIGGER": UNSAFE_TRIGGER,
    }
    yield from spawn_servers(tmp_path_factory, "tool_safety_check", _write_config, mock_env=mock_env)


# @tool_output_validation blocks a call to an undeclared tool, so both tools are declared
# here, matching DECLARED_TOOLS in test_per_tool_regex_rails_server.py. scoped_tool declares
# query/notes explicitly so $argument=query has a real schema to scope against.
DECLARED_TOOLS = [
    {
        "type": "function",
        "function": {"name": "run_sql", "parameters": {"type": "object", "additionalProperties": True}},
    },
    {
        "type": "function",
        "function": {
            "name": "scoped_tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "notes": {"type": "string"}},
                "additionalProperties": True,
            },
        },
    },
]


def _last_prompt(servers: Servers, model: str) -> str:
    """The rendered prompt text the mock's judge model most recently received."""
    response = httpx.get(f"{servers.mock_url}/debug/last-request/{model}", timeout=10)
    response.raise_for_status()
    messages = response.json()["messages"]
    return next(message["content"] for message in reversed(messages) if message["role"] == "user")


def _chat(servers: Servers, content: str, *, tool_result: bool = False) -> dict:
    if tool_result:
        messages = [
            {"role": "user", "content": "run a query"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": content},
        ]
    else:
        messages = [{"role": "user", "content": content}]

    response = httpx.post(
        f"{servers.guardrails_url}/v1/chat/completions",
        json={"model": servers.config_id, "messages": messages, "tools": DECLARED_TOOLS},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


class TestToolSafetyCheckOutputAgainstRealServer:
    def test_unsafe_judgment_blocks(self, servers):
        message = _chat(servers, UNSAFE_TRIGGER)
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_safe_judgment_passes(self, servers):
        message = _chat(servers, "SELECT 1")
        assert message["tool_calls"][0]["function"]["name"] == "run_sql"

    def test_prompt_renders_tool_name_and_arguments(self, servers):
        """The judge receives the exact rendered prompt, not just a trigger word: proves
        {{ tool_name }} and {{ tool_call_arguments }} are substituted correctly, matching the
        run_sql_check prompt template declared in _write_config."""
        _chat(servers, "SELECT 1")

        prompt = _last_prompt(servers, JUDGE_MODEL_NAME)

        assert prompt == 'Tool: run_sql\nArguments: {"query": "SELECT 1"}\nRespond safe or unsafe: <reason>.'


class TestToolSafetyCheckInputAgainstRealServer:
    def test_unsafe_judgment_blocks_before_generation(self, servers):
        message = _chat(servers, UNSAFE_TRIGGER, tool_result=True)
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_safe_judgment_passes(self, servers):
        message = _chat(servers, "no sensitive data", tool_result=True)
        assert message["content"] == "ok"

    def test_prompt_renders_tool_name_and_result(self, servers):
        _chat(servers, "no sensitive data", tool_result=True)

        prompt = _last_prompt(servers, JUDGE_MODEL_NAME)

        assert prompt == "Tool: run_sql\nResult: no sensitive data\nRespond safe or unsafe: <reason>."


class TestToolSafetyCheckArgumentScopingAgainstRealServer:
    """scoped_tool's flow is `tool safety check output $model=judge $variant=scoped_check
    $argument=query`, so only the `query` argument reaches the judge; a trigger elsewhere in
    the same call's arguments must not block, and must never even reach the prompt."""

    def test_trigger_outside_scoped_argument_passes(self, servers):
        message = _chat(servers, f'scoped_tool:{{"query": "SELECT 1", "notes": "{UNSAFE_TRIGGER}"}}')

        assert message["tool_calls"][0]["function"]["name"] == "scoped_tool"
        prompt = _last_prompt(servers, JUDGE_MODEL_NAME)
        assert prompt == 'Tool: scoped_tool\nArguments: {"query": "SELECT 1"}\nRespond safe or unsafe: <reason>.'

    def test_trigger_inside_scoped_argument_blocks(self, servers):
        message = _chat(servers, f'scoped_tool:{{"query": "{UNSAFE_TRIGGER}", "notes": "irrelevant"}}')

        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."
