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

"""A LangChain agent whose tool calls the NeMo Relay plugin checks, through Relay's LangChain middleware."""

# Adapted from Relay's own `create_agent` test, `test_agent_integration` in
# python/tests/integrations/langchain_tests/test_middleware.py at 0.10.4.

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from nemo_relay.integrations.langchain.middleware import NemoRelayMiddleware

FINAL_ANSWER = "Here is what I found."


class ScriptedChatModel(GenericFakeChatModel):
    """A chat model that answers each turn with the next of its messages, whatever tools the agent binds."""

    # Relay's middleware names each model call after the model, and fails on a model without a name.
    model: str = "scripted-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        """The model itself: its messages already hold the tool calls it makes."""
        return self


def search_agent(query: str, runs: list[str]) -> Any:
    """An agent whose model searches for *query* and then answers, with a `search` tool that records its runs."""

    @tool
    def search(query: str) -> str:
        """Searches for *query*."""
        runs.append(query)
        return f"Results for {query}"

    model = ScriptedChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": query}, "id": "call-1"}]),
                AIMessage(content=FINAL_ANSWER),
            ]
        )
    )
    tools: list[BaseTool] = [search]
    return create_agent(model=model, tools=tools, middleware=[NemoRelayMiddleware()])


def run_agent(agent: Any, invocation: str) -> dict:
    """Runs *agent* on one user message, with `invoke` or `ainvoke`."""
    payload = {"messages": [{"role": "user", "content": "Find it for me."}]}
    if invocation == "ainvoke":
        return asyncio.run(agent.ainvoke(payload))
    return agent.invoke(payload)


@pytest.mark.parametrize("invocation", ["invoke", "ainvoke"])
def test_allowed_tool_call_runs_the_tool(relay_plugin, deterministic_hooks_server, invocation):
    """A tool call the hooks server allows runs the tool, and the agent answers with its result."""
    relay_plugin({"endpoint": deterministic_hooks_server})
    runs: list[str] = []

    result = run_agent(search_agent("weather", runs), invocation)

    assert runs == ["weather"]
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert [message.content for message in tool_messages] == ["Results for weather"]
    assert result["messages"][-1].content == FINAL_ANSWER


@pytest.mark.parametrize("invocation", ["invoke", "ainvoke"])
def test_blocked_tool_call_never_runs_the_tool(relay_plugin, deterministic_hooks_server, invocation):
    """A tool call a rail blocks never runs the tool, and the rejection reaches the agent's caller."""
    relay_plugin({"endpoint": deterministic_hooks_server})
    runs: list[str] = []

    with pytest.raises(RuntimeError) as error:
        run_agent(search_agent("admin password", runs), invocation)

    assert str(error.value) == "guardrail rejected: NeMo Guardrails blocked tool call 'search': regex check tool call"
    assert runs == []
