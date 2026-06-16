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

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.types import LLMResponse, ToolCall, ToolCallFunction
from tests.utils import FakeLLMModel


@pytest.mark.asyncio
async def test_output_rails_skip_for_tool_calls():
    fake_llm = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_process",
                        type="function",
                        function=ToolCallFunction(name="process_data", arguments={"data": "test"}),
                    )
                ],
            )
        ]
    )

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
        passthrough: true
        rails:
          output:
            flows:
              - strict_output_check
              - add_prefix
        """,
    )

    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Process this"}])

    assert result["tool_calls"] is not None
    assert result["tool_calls"][0]["function"]["name"] == "process_data"
    assert result["content"] == ""
    assert "I'm sorry, I can't respond to that." not in result["content"]
    assert "PREFIX:" not in result["content"]


@pytest.mark.asyncio
async def test_text_responses_still_use_output_rails():
    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="Hello there")])

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

    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Say hello"}])

    assert "PREFIX: Hello there" in result["content"]
    assert result.get("tool_calls") is None or result.get("tool_calls") == []


@pytest.mark.asyncio
async def test_complex_chain_with_tool_calls():
    fake_llm = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_intake",
                        type="function",
                        function=ToolCallFunction(
                            name="print_gathered_patient_info",
                            arguments={
                                "patient_name": "John Doe",
                                "patient_dob": "01/01/1990",
                            },
                        ),
                    )
                ],
            )
        ]
    )

    config = RailsConfig.from_content(
        colang_content="",
        yaml_content="""
        models: []
        passthrough: true
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

    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "My date of birth is 01/01/1990."}])

    assert result["tool_calls"] is not None
    assert result["tool_calls"][0]["function"]["name"] == "print_gathered_patient_info"
    assert result["tool_calls"][0]["function"]["arguments"]["patient_name"] == "John Doe"
    assert result["content"] == ""
    assert "I'm sorry, I can't respond to that." not in result["content"]


@pytest.mark.asyncio
async def test_self_check_output_rail_bypassed():
    fake_llm = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_sensitive",
                        type="function",
                        function=ToolCallFunction(name="sensitive_operation", arguments={"action": "process"}),
                    )
                ],
            )
        ]
    )

    config = RailsConfig.from_content(
        colang_content="",
        yaml_content="""
        models: []
        passthrough: true
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

    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Perform sensitive operation"}])

    assert result["tool_calls"] is not None
    assert result["tool_calls"][0]["function"]["name"] == "sensitive_operation"
    assert "I'm sorry, I can't respond to that." not in result["content"]


@pytest.mark.asyncio
async def test_backward_compatibility_text_blocking():
    fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="This response should be blocked by output rails")])

    config = RailsConfig.from_content(
        """
        define flow block_problematic
          if "should be blocked" in $bot_message
            bot refuse to respond
            stop
        """,
        """
        passthrough: true
        rails:
          output:
            flows:
              - block_problematic
        """,
    )

    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Say something bad"}])

    assert "I'm sorry, I can't respond to that." in result["content"]
    assert result.get("tool_calls") is None or result.get("tool_calls") == []


@pytest.mark.asyncio
async def test_mixed_tool_calls_and_content():
    fake_llm = FakeLLMModel(
        llm_responses=[
            LLMResponse(
                content="I'll gather the information for you.",
                tool_calls=[
                    ToolCall(
                        id="call_gather",
                        type="function",
                        function=ToolCallFunction(name="gather_info", arguments={"user_id": "123"}),
                    )
                ],
            )
        ]
    )

    config = RailsConfig.from_content(
        """
        define flow add_timestamp
          $bot_message = $bot_message + " [" + $current_time + "]"
        """,
        """
        passthrough: true
        rails:
          output:
            flows:
              - add_timestamp
        """,
    )

    rails = LLMRails(config, llm=fake_llm)
    result = await rails.generate_async(messages=[{"role": "user", "content": "Gather my info"}])

    assert result["tool_calls"] is not None
    assert result["tool_calls"][0]["function"]["name"] == "gather_info"
