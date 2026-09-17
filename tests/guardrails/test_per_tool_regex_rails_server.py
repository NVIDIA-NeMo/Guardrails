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

"""Real-process integration tests for per-tool regex rails against the actual server.

Unlike every other server test in this repo (in-process fastapi.testclient.TestClient),
this drives a real `nemoguardrails server` instance and the mock tool LLM server it
points at over real HTTP, via the shared harness in server_integration_helpers.py.
"""

from pathlib import Path
from typing import Iterator

import httpx
import pytest
import yaml

from tests.guardrails.server_integration_helpers import Servers, spawn_servers


def _write_config(config_dir: Path, config_id: str, mock_port: int) -> None:
    config = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "model": "mock-tool-model",
                "parameters": {"base_url": f"http://127.0.0.1:{mock_port}/v1", "api_key": "unused"},
            }
        ],
        # The server's /v1/chat/completions gates a request-level "tools" field on
        # passthrough=True (nemoguardrails/server/api.py) regardless of which engine is
        # active -- a legacy-LLMRails-only flag that IORails itself never reads for tool
        # handling (see Guardrails.passthrough_fn, which raises for IORails). Harmless
        # here, but worth a separate issue: the gate isn't engine-aware.
        "passthrough": True,
        "rails": {
            "config": {
                "regex_detection": {
                    "tool_output": {
                        "run_sql": {"patterns": [r"DROP\s+TABLE"]},
                        "other_tool": {"patterns": [r"SECRET"]},
                        "scoped_tool": {"patterns": [r"DROP\s+TABLE"]},
                    },
                    "tool_input": {"run_sql": {"patterns": [r"ssn:\s*\d{3}-\d{2}-\d{4}"]}},
                }
            },
            "tool_output": {
                "per_tool": {
                    "run_sql": ["regex check tool output"],
                    "other_tool": ["regex check tool output"],
                    "scoped_tool": ["regex check tool output $argument=query"],
                }
            },
            "tool_input": {"per_tool": {"run_sql": ["regex check tool input"]}},
        },
    }
    (config_dir / config_id).mkdir(parents=True)
    (config_dir / config_id / "config.yml").write_text(yaml.safe_dump(config))


@pytest.fixture(scope="module")
def servers(tmp_path_factory) -> Iterator[Servers]:
    yield from spawn_servers(tmp_path_factory, "per_tool_regex", _write_config)


# @tool_output_validation blocks a call to an undeclared tool, so every tool these tests
# call is declared here, per request, matching how a real client would declare tools.
# run_sql/other_tool use a fully permissive schema; scoped_tool requires `query` to be
# a string so a real schema violation can be exercised too.
DECLARED_TOOLS = [
    {
        "type": "function",
        "function": {"name": "run_sql", "parameters": {"type": "object", "additionalProperties": True}},
    },
    {
        "type": "function",
        "function": {"name": "other_tool", "parameters": {"type": "object", "additionalProperties": True}},
    },
    {
        "type": "function",
        "function": {
            "name": "scoped_tool",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": True,
            },
        },
    },
]


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


class TestPerToolCallRegexAgainstRealServer:
    def test_matching_pattern_blocked(self, servers):
        message = _chat(servers, "DROP TABLE users")
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_non_matching_pattern_passes(self, servers):
        message = _chat(servers, "SELECT 1")
        assert message["tool_calls"][0]["function"]["name"] == "run_sql"

    def test_multiple_tool_calls_all_safe_pass(self, servers):
        """The fan-out loop evaluates every tool call, not just the first."""
        message = _chat(servers, "SELECT 1||SELECT 2")
        assert len(message["tool_calls"]) == 2
        assert [tc["function"]["arguments"] for tc in message["tool_calls"]] == [
            '{"query": "SELECT 1"}',
            '{"query": "SELECT 2"}',
        ]

    def test_multiple_tool_calls_second_one_blocks(self, servers):
        """A match on any tool call in the batch blocks the whole response, not just that call."""
        message = _chat(servers, "SELECT 1||DROP TABLE users")
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_multiple_tool_calls_first_one_blocks(self, servers):
        """Order doesn't matter: a match anywhere in the batch blocks."""
        message = _chat(servers, "DROP TABLE users||SELECT 1")
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_second_tools_pattern_does_not_cross_match_first_tools_content(self, servers):
        """run_sql and other_tool each have their own configured patterns (DROP TABLE
        vs SECRET). run_sql's pattern matching other_tool's content must not leak across
        -- other_tool's call only blocks on its own pattern, not run_sql's."""
        message = _chat(servers, "run_sql:SELECT 1||other_tool:DROP TABLE users")
        assert len(message["tool_calls"]) == 2
        names = [tc["function"]["name"] for tc in message["tool_calls"]]
        assert names == ["run_sql", "other_tool"]

    def test_second_tool_blocks_on_its_own_pattern(self, servers):
        """other_tool's own configured pattern (SECRET) still blocks when it matches,
        proving its check genuinely runs rather than being silently skipped."""
        message = _chat(servers, "run_sql:SELECT 1||other_tool:the SECRET is out")
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."


class TestArgumentScopingAgainstRealServer:
    """scoped_tool's flow is `regex check tool output $argument=query`, so only the
    `query` argument is checked; a match elsewhere in the same call's arguments must
    not block."""

    def test_match_outside_scoped_argument_passes(self, servers):
        message = _chat(servers, 'scoped_tool:{"query": "SELECT 1", "notes": "DROP TABLE users"}')
        assert message["tool_calls"][0]["function"]["name"] == "scoped_tool"

    def test_match_inside_scoped_argument_blocks(self, servers):
        message = _chat(servers, 'scoped_tool:{"query": "DROP TABLE users", "notes": "irrelevant"}')
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."


class TestSchemaValidationAgainstRealServer:
    def test_schema_violation_blocks_before_regex_check(self, servers):
        """@tool_output_validation blocks arguments that violate the declared schema
        (scoped_tool's `query` must be a string), even when the regex pattern itself
        would not have matched."""
        message = _chat(servers, 'scoped_tool:{"query": 123, "notes": "irrelevant"}')
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."


class TestPerToolResultRegexAgainstRealServer:
    def test_matching_pattern_blocked(self, servers):
        message = _chat(servers, "ssn: 123-45-6789", tool_result=True)
        assert "tool_calls" not in message
        assert message["content"] == "I'm sorry, I can't respond to that."

    def test_non_matching_pattern_passes(self, servers):
        message = _chat(servers, "no sensitive data", tool_result=True)
        assert message["content"] == "ok"
