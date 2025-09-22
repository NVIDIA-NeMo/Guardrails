# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Optional

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input, Output

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails


def has_nvidia_ai_endpoints():
    """Check if NVIDIA AI Endpoints package is installed."""
    try:
        import langchain_nvidia_ai_endpoints

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not has_nvidia_ai_endpoints(),
    reason="langchain-nvidia-ai-endpoints package not installed",
)
def test_runnable_binding_treated_as_llm():
    """Test that RunnableBinding with LLM tools is treated as an LLM, not passthrough_runnable."""
    from langchain_core.tools import tool
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's sunny in {city}!"

    config = RailsConfig.from_content(config={"models": []})
    guardrails = RunnableRails(config=config, passthrough=True)

    llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
    llm_with_tools = llm.bind_tools([get_weather])

    piped = guardrails | llm_with_tools

    assert piped.llm is llm_with_tools
    assert piped.passthrough_runnable is None


def test_tool_calls_preservation():
    """Test that tool calls are preserved in RunnableRails output."""
    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's sunny in {city}!"

    class MockLLMWithTools:
        def __init__(self):
            pass

        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="I'll check the weather for you.",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "San Francisco"},
                        "id": "call_123",
                        "type": "tool_call",
                    }
                ],
            )

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    config = RailsConfig.from_content(config={"models": []})
    llm_with_tools = MockLLMWithTools()
    rails = RunnableRails(config, llm=llm_with_tools)

    prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
    chain = prompt | rails

    result = chain.invoke({"input": "What's the weather?"})

    assert isinstance(result, AIMessage)
    assert result.content == "I'll check the weather for you."
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "get_weather"
    assert result.tool_calls[0]["args"]["city"] == "San Francisco"


def test_tool_calls_preservation_base_message_input():
    """Test tool calls preservation with BaseMessage input."""

    class MockLLMWithTools:
        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="Weather check",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "NYC"},
                        "id": "call_456",
                        "type": "tool_call",
                    }
                ],
            )

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=MockLLMWithTools())

    result = rails.invoke(HumanMessage(content="Weather?"))

    assert isinstance(result, AIMessage)
    assert result.tool_calls is not None
    assert result.tool_calls[0]["name"] == "get_weather"


def test_tool_calls_preservation_dict_input():
    """Test tool calls preservation with dict input containing BaseMessage list."""

    class MockLLMWithTools:
        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="Tool response",
                tool_calls=[
                    {
                        "name": "test_tool",
                        "args": {},
                        "id": "call_789",
                        "type": "tool_call",
                    }
                ],
            )

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=MockLLMWithTools())

    result = rails.invoke({"input": [HumanMessage(content="Test")]})

    assert isinstance(result, dict)
    assert "output" in result
    assert isinstance(result["output"], AIMessage)
    assert result["output"].tool_calls is not None
    assert result["output"].tool_calls[0]["name"] == "test_tool"


@pytest.mark.skipif(
    not has_nvidia_ai_endpoints(),
    reason="langchain-nvidia-ai-endpoints package not installed",
)
def test_runnable_binding_treated_as_llm():
    """Test that RunnableBinding with LLM tools is treated as an LLM, not passthrough_runnable."""
    from langchain_core.tools import tool
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's sunny in {city}!"

    config = RailsConfig.from_content(config={"models": []})
    guardrails = RunnableRails(config=config, passthrough=True)

    llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
    llm_with_tools = llm.bind_tools([get_weather])

    piped = guardrails | llm_with_tools

    assert piped.llm is llm_with_tools
    assert piped.passthrough_runnable is None
