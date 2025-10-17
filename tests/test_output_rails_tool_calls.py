# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Integration tests for tool calls with output rails."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails


def test_output_rails_skip_for_tool_calls():
    """Test that output rails are skipped when tool calls are present."""

    class MockLLMWithToolResponse:
        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "process_data",
                        "args": {"data": "test"},
                        "id": "call_process",
                        "type": "tool_call",
                    }
                ],
            )

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    # Config with aggressive output rails that would block empty content
    config = RailsConfig.from_content(
        """
        define flow strict_output_check
          if $bot_message == ""
            bot refuse to respond
            stop

        define flow add_prefix
          $bot_message = "PREFIX: " + $bot_message
        """,
        """
        rails:
          output:
            flows:
              - strict_output_check
              - add_prefix
        """,
    )

    rails = RunnableRails(config, llm=MockLLMWithToolResponse())
    result = rails.invoke(HumanMessage(content="Process this"))

    # Tool calls should bypass output rails entirely
    assert result.tool_calls is not None
    assert result.tool_calls[0]["name"] == "process_data"
    assert result.content == ""  # Should stay empty, not modified by rails
    assert "I'm sorry, I can't respond to that." not in result.content
    assert "PREFIX:" not in result.content  # Rails should not have run


def test_text_responses_still_use_output_rails():
    """Test that regular text responses still go through output rails."""

    class MockLLMTextResponse:
        def invoke(self, messages, **kwargs):
            return AIMessage(content="Hello there")

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    # Same config as above test
    config = RailsConfig.from_content(
        """
        define flow add_prefix
          $bot_message = "PREFIX: " + $bot_message
        """,
        """
        rails:
          output:
            flows:
              - add_prefix
        """,
    )

    rails = RunnableRails(config, llm=MockLLMTextResponse())
    result = rails.invoke(HumanMessage(content="Say hello"))

    assert "PREFIX: Hello there" in result.content
    assert result.tool_calls is None or result.tool_calls == []


def test_complex_chain_with_tool_calls():
    """Test tool calls work in complex LangChain scenarios."""

    class MockPatientIntakeLLM:
        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "print_gathered_patient_info",
                        "args": {
                            "patient_name": "John Doe",
                            "patient_dob": "01/01/1990",
                        },
                        "id": "call_intake",
                        "type": "tool_call",
                    }
                ],
            )

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    system_prompt = """
    You are a specialized assistant for handling patient intake.
    After gathering all information, use the print_gathered_patient_info tool.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )

    config = RailsConfig.from_content(
        colang_content="",
        yaml_content="""
        models: []
        rails:
          output:
            flows:
              - self check output

        prompts:
          - task: self_check_output
            content: |
              Instructions: {instructions}
              Output: {output}

              Check if the output is appropriate and safe.
        """,
    )

    guardrails = RunnableRails(
        config=config, llm=MockPatientIntakeLLM(), passthrough=True
    )

    chain = prompt | guardrails

    result = chain.invoke(
        {
            "messages": [
                ("user", "Hi!"),
                ("assistant", "Welcome! What's your name?"),
                ("user", "My name is John Doe."),
                ("assistant", "What's your date of birth?"),
                ("user", "My date of birth is 01/01/1990."),
            ]
        }
    )

    assert isinstance(result, AIMessage)
    assert result.tool_calls is not None
    assert result.tool_calls[0]["name"] == "print_gathered_patient_info"
    assert result.tool_calls[0]["args"]["patient_name"] == "John Doe"
    assert result.content == ""
    assert "I'm sorry, I can't respond to that." not in result.content


def test_self_check_output_rail_bypassed():
    """Test that self_check_output rail is bypassed for tool calls."""

    class MockLLMToolCallsWithSelfCheck:
        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sensitive_operation",
                        "args": {"action": "process"},
                        "id": "call_sensitive",
                        "type": "tool_call",
                    }
                ],
            )

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    config = RailsConfig.from_content(
        colang_content="",
        yaml_content="""
        models: []
        rails:
          output:
            flows:
              - self check output

        prompts:
          - task: self_check_output
            content: |
              Instructions: {instructions}
              Output: {output}

              Check if the output is appropriate and safe.
        """,
    )

    rails = RunnableRails(config, llm=MockLLMToolCallsWithSelfCheck())
    result = rails.invoke(HumanMessage(content="Perform sensitive operation"))

    assert result.tool_calls is not None
    assert result.tool_calls[0]["name"] == "sensitive_operation"
    assert "I'm sorry, I can't respond to that." not in result.content


def test_backward_compatibility_text_blocking():
    """Test that text-based blocking still works for non-tool responses."""

    class MockLLMProblematicText:
        def invoke(self, messages, **kwargs):
            return AIMessage(content="This response should be blocked by output rails")

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    config = RailsConfig.from_content(
        """
        define flow block_problematic
          if "should be blocked" in $bot_message
            bot refuse to respond
            stop
        """,
        """
        rails:
          output:
            flows:
              - block_problematic
        """,
    )

    rails = RunnableRails(config, llm=MockLLMProblematicText())
    result = rails.invoke(HumanMessage(content="Say something bad"))

    assert "I'm sorry, I can't respond to that." in result.content
    assert result.tool_calls is None or result.tool_calls == []


def test_mixed_tool_calls_and_content():
    """Test responses that have both content and tool calls."""

    class MockLLMWithBoth:
        def invoke(self, messages, **kwargs):
            return AIMessage(
                content="I'll gather the information for you.",
                tool_calls=[
                    {
                        "name": "gather_info",
                        "args": {"user_id": "123"},
                        "id": "call_gather",
                        "type": "tool_call",
                    }
                ],
            )

        async def ainvoke(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)

    config = RailsConfig.from_content(
        """
        define flow add_timestamp
          $bot_message = $bot_message + " [" + $current_time + "]"
        """,
        """
        rails:
          output:
            flows:
              - add_timestamp
        """,
    )

    rails = RunnableRails(config, llm=MockLLMWithBoth())
    result = rails.invoke(HumanMessage(content="Gather my info"))

    assert result.tool_calls is not None
    assert result.tool_calls[0]["name"] == "gather_info"
