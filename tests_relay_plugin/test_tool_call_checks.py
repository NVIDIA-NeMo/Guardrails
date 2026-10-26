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
from mock_judge import SERVER_ERROR_MESSAGE, UNSAFE_MARKER, JudgeFailure
from nemo_relay import Json, ToolExecutionResult, tools

# Long enough that a check reaches the judge well before it, which the timeout case then confirms.
CHECK_TIMEOUT_MILLIS = 1000
JUDGE_ARRIVAL_TIMEOUT_SECONDS = 10


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


@pytest.mark.parametrize(
    ("name", "args"),
    [
        pytest.param("search", {"query": "weather"}, id="rail-passes"),
        pytest.param("send_email", {"to": "ops@example.com", "subject": "Status"}, id="no-rail"),
    ],
)
def test_allowed_call_runs_the_tool(relay_plugin, policy_hooks_server, name, args):
    """A call the tool policy allows, and any rail on its tool passes, runs the tool with its arguments."""
    relay_plugin({"endpoint": policy_hooks_server})
    tool = RecordingTool()

    result = call_tool(name, args, tool)

    assert result.result == {"ran": True}
    assert tool.runs == [args]


@pytest.mark.parametrize(
    ("name", "args", "reason"),
    [
        pytest.param(
            "delete_file",
            {"path": "notes.txt"},
            "tool call 'delete_file' is not an allowed tool",
            id="unlisted-tool",
        ),
        pytest.param(
            "run_shell",
            {"command": "ls"},
            "tool call 'run_shell' is a blocked tool",
            id="blocked-tool",
        ),
        pytest.param(
            "send_email",
            {"to": "ops@example.com", "bcc": "audit@example.com"},
            "tool call 'send_email' passes argument 'bcc', which is blocked",
            id="blocked-argument",
        ),
        pytest.param(
            "search",
            {"query": "weather", "limit": 5},
            "tool call 'search' passes 1 argument its allowed_tools entry doesn't list",
            id="unlisted-argument",
        ),
        pytest.param(
            "search",
            ["weather"],
            "tool call 'search' has arguments that aren't a JSON object",
            id="arguments-not-an-object",
        ),
        pytest.param(
            "search",
            {"query": "admin password"},
            "regex check tool call",
            id="regex-rail",
        ),
    ],
)
def test_denied_call_is_rejected_with_the_server_reason(relay_plugin, policy_hooks_server, name, args, reason):
    """A call the tool policy or a rail denies is rejected with the server's reason, and the tool never runs."""
    relay_plugin({"endpoint": policy_hooks_server})
    tool = RecordingTool()

    with pytest.raises(RuntimeError) as error:
        call_tool(name, args, tool)

    assert str(error.value) == f"guardrail rejected: NeMo Guardrails blocked tool call '{name}': {reason}"
    assert tool.runs == []


def test_judge_safe_verdict_runs_the_tool(relay_plugin, judge_hooks_server, judge):
    """A call the judge finds safe runs the tool, after one judge request whose prompt holds the call's arguments."""
    relay_plugin({"endpoint": judge_hooks_server})
    tool = RecordingTool()

    call_tool("fetch_url", {"url": "https://docs.example/guide"}, tool)

    assert tool.runs == [{"url": "https://docs.example/guide"}]
    assert len(judge.prompts) == 1
    assert "https://docs.example/guide" in judge.prompts[0]


def test_judge_unsafe_verdict_rejects_the_call(relay_plugin, judge_hooks_server, judge):
    """A call the judge finds unsafe is rejected with the judge rail's name, and the tool never runs."""
    relay_plugin({"endpoint": judge_hooks_server})
    tool = RecordingTool()

    with pytest.raises(RuntimeError) as error:
        call_tool("fetch_url", {"url": f"https://{UNSAFE_MARKER}/upload"}, tool)

    assert str(error.value) == (
        "guardrail rejected: NeMo Guardrails blocked tool call 'fetch_url': tool safety check tool call"
    )
    assert tool.runs == []
    assert len(judge.prompts) == 1


@pytest.mark.parametrize(
    ("failure", "detail"),
    [
        pytest.param(
            JudgeFailure.MALFORMED_VERDICT,
            "rail 'tool safety check tool call' failed: tool safety check tool call error",
            id="malformed-verdict",
        ),
        pytest.param(
            JudgeFailure.SERVER_ERROR,
            f"a tool-call rail failed: {SERVER_ERROR_MESSAGE}",
            id="server-error",
        ),
    ],
)
def test_failed_judge_blocks_the_call(relay_plugin, judge_hooks_server, judge, failure, detail):
    """A judge that gives no verdict fails the check with the server's 500 rail_failed, and the tool never runs."""
    relay_plugin({"endpoint": judge_hooks_server})
    judge.failure = failure
    tool = RecordingTool()

    with pytest.raises(RuntimeError) as error:
        call_tool("fetch_url", {"url": "https://docs.example/status"}, tool)

    assert str(error.value) == (
        "internal error: NeMo Guardrails check failed for tool call 'fetch_url': "
        f"server returned HTTP 500 Internal Server Error (rail_failed): {detail} "
        "(a rail failed instead of deciding, such as a judge model that can't be reached)"
    )
    assert tool.runs == []


def test_check_timeout_blocks_the_call(relay_plugin, judge_hooks_server, judge):
    """A check that outlasts timeout_millis, held by a judge that doesn't answer, blocks the call as a failed check."""
    relay_plugin({"endpoint": judge_hooks_server, "timeout_millis": CHECK_TIMEOUT_MILLIS})
    judge.failure = JudgeFailure.HANG
    tool = RecordingTool()

    with pytest.raises(RuntimeError) as error:
        call_tool("fetch_url", {"url": "https://docs.example/slow"}, tool)

    assert str(error.value) == (
        "internal error: NeMo Guardrails check failed for tool call 'fetch_url': "
        f"request timed out after {CHECK_TIMEOUT_MILLIS} ms"
    )
    assert tool.runs == []
    assert judge.hanging.wait(JUDGE_ARRIVAL_TIMEOUT_SECONDS), "the check never reached the judge"


def test_repeated_call_reuses_the_decision(relay_plugin, judge_hooks_server, judge):
    """The same call made twice within the plugin's cache lifetime asks the judge once, and runs the tool both times."""
    relay_plugin({"endpoint": judge_hooks_server})
    tool = RecordingTool()

    call_tool("fetch_url", {"url": "https://docs.example/twice"}, tool)
    call_tool("fetch_url", {"url": "https://docs.example/twice"}, tool)

    assert tool.runs == [{"url": "https://docs.example/twice"}] * 2
    assert len(judge.prompts) == 1


def test_failed_check_is_retried_on_the_next_call(relay_plugin, judge_hooks_server, judge):
    """A failed check isn't cached: the same call made again asks the judge again, and runs once the judge answers."""
    relay_plugin({"endpoint": judge_hooks_server})
    judge.failure = JudgeFailure.SERVER_ERROR
    tool = RecordingTool()

    with pytest.raises(RuntimeError):
        call_tool("fetch_url", {"url": "https://docs.example/retry"}, tool)
    judge.failure = None
    call_tool("fetch_url", {"url": "https://docs.example/retry"}, tool)

    assert tool.runs == [{"url": "https://docs.example/retry"}]
    assert len(judge.prompts) == 2


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
