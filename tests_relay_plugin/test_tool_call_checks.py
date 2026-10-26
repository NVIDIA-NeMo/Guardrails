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

"""Managed tool calls that the NeMo Relay plugin checks, in Relay's Python plugin host."""

import asyncio

import pytest
from nemo_relay import Json, ToolExecutionResult, tools


class RecordingTool:
    """A tool that records the arguments of each run."""

    def __init__(self) -> None:
        self.runs: list[Json] = []

    def __call__(self, args: Json) -> ToolExecutionResult:
        """Records the run and returns a fixed result."""
        self.runs.append(args)
        return ToolExecutionResult({"ran": True})


def call_tool(name: str, args: Json, tool: RecordingTool) -> ToolExecutionResult:
    """Runs a managed tool call through Relay's guardrails, as Relay's SDK and integrations do."""

    async def call() -> ToolExecutionResult:
        """Relay starts the call on the running event loop, so it is made inside one."""
        return await tools.execute(name, args, tool)

    return asyncio.run(call())


def test_example_config_allows_a_search(relay_plugin, example_hooks_server):
    """Guardrails' NeMo Relay example allows a search with a query, and the tool runs and returns its result."""
    relay_plugin({"endpoint": example_hooks_server})
    tool = RecordingTool()

    result = call_tool("search", {"query": "weather"}, tool)

    assert result.result == {"ran": True}
    assert tool.runs == [{"query": "weather"}]


def test_example_config_rejects_an_empty_search(relay_plugin, example_hooks_server):
    """Guardrails' NeMo Relay example rejects a search with an empty query by its regex rail, and the tool never runs."""
    relay_plugin({"endpoint": example_hooks_server})
    tool = RecordingTool()

    with pytest.raises(RuntimeError) as error:
        call_tool("search", {"query": ""}, tool)

    assert str(error.value) == "guardrail rejected: NeMo Guardrails blocked tool call 'search': regex check tool call"
    assert tool.runs == []


def test_unreachable_server_blocks_the_call(relay_plugin, refused_endpoint):
    """A check that can't reach the hooks server blocks the call with an internal error, and the tool never runs."""
    relay_plugin({"endpoint": refused_endpoint})
    tool = RecordingTool()

    with pytest.raises(RuntimeError) as error:
        call_tool("search", {"query": "weather"}, tool)

    message = str(error.value)
    assert message.startswith(
        "internal error: NeMo Guardrails check failed for tool call 'search': "
        f"error sending request for url ({refused_endpoint}/v1/hooks/nemo-relay)"
    ), message
    assert "Connection refused" in message, message
    assert tool.runs == []
