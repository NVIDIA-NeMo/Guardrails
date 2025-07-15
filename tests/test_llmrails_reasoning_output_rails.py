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

"""Tests for LLM Rails reasoning output configuration and behavior.

This module contains tests that verify the behavior of LLM Rails when handling
reasoning traces in the output, including configuration options and guardrail
behavior.
"""

from typing import Any, Dict, NamedTuple

import pytest

from nemoguardrails import RailsConfig
from tests.utils import TestChat


class ReasoningTraceTestCase(NamedTuple):
    """Test case for reasoning trace configuration.

    Attributes:
        description: description of the test case
        remove_reasoning_traces: Whether to remove reasoning traces in the model config
        apply_to_reasoning_traces: Whether to apply output rails to reasoning traces
        expected_think_tag: Whether the think tag should be present in the response
        expected_error_message: Whether the error message should be present in the response
    """

    description: str
    remove_reasoning_traces: bool
    apply_to_reasoning_traces: bool
    expected_think_tag: bool
    expected_error_message: bool


async def check_sensitive_info(context: Dict[str, Any]) -> bool:
    """Check if the response contains sensitive information."""
    response = context.get("bot_message", "")
    prompt = context.get("user_message", "")
    input_text = response or prompt
    return "credit card" in input_text.lower() or any(
        c.isdigit() for c in input_text if c.isdigit() or c == "-"
    )


async def check_think_tag_present(context: Dict[str, Any]) -> bool:
    """Check if the think tag is present in the bot's response."""
    response = context.get("bot_message", "")
    return "<think>" in response


@pytest.fixture
def base_config() -> RailsConfig:
    """Creates a base RailsConfig with common test configuration."""
    return RailsConfig.from_content(
        colang_content="""
        define flow check think tag
            $not_allowed = execute check_think_tag_present
            if $not_allowed
                bot informs tag not allowed
                stop

        define bot informs tag not allowed
            "think tag is not allowed it must be removed"
        """,
        yaml_content="""
        models:
          - type: main
            engine: fake
            model: fake
        colang_version: "1.0"
        rails:
          output:
            flows:
              - check think tag
        """,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        ReasoningTraceTestCase(
            description="Remove reasoning traces and show error when guardrail is enabled",
            remove_reasoning_traces=True,
            apply_to_reasoning_traces=True,
            expected_think_tag=True,
            expected_error_message=True,
        ),
        ReasoningTraceTestCase(
            description="Preserve reasoning traces and hide error when guardrail is disabled",
            remove_reasoning_traces=True,
            apply_to_reasoning_traces=False,
            expected_think_tag=True,
            expected_error_message=False,
        ),
        ReasoningTraceTestCase(
            description="Preserve reasoning traces and show error when guardrail is enabled",
            remove_reasoning_traces=False,
            apply_to_reasoning_traces=True,
            expected_think_tag=True,
            expected_error_message=True,
        ),
        ReasoningTraceTestCase(
            description="Remove reasoning traces and show error when both flags are disabled",
            remove_reasoning_traces=False,
            apply_to_reasoning_traces=False,
            expected_think_tag=True,
            expected_error_message=True,
        ),
    ],
    ids=lambda tc: tc.description,
)
async def test_output_rails_reasoning_traces_configuration(
    base_config: RailsConfig,
    test_case: ReasoningTraceTestCase,
) -> None:
    """Test output rails with different reasoning traces configurations.

    The test verifies the following behaviors based on configuration:

    1. When remove_reasoning_traces=True:
       - The model is configured to remove reasoning traces
       - However, the actual removal depends on apply_to_reasoning_traces

    2. When apply_to_reasoning_traces=True:
       - The output rail will check for and report think tags
       - Because we expect the think tag to be present as output rails explicitly requires it

    3. When apply_to_reasoning_traces=False:
       - The output rails will check for think tags
       - No error message will be shown because it is not there to get blocked

    """
    base_config.models[
        0
    ].reasoning_config.remove_reasoning_traces = test_case.remove_reasoning_traces
    base_config.rails.output.apply_to_reasoning_traces = (
        test_case.apply_to_reasoning_traces
    )

    chat = TestChat(
        base_config,
        llm_completions=[
            "<think> I should think more </think> Your kindness is appreciated"
        ],
    )

    chat.app.runtime.register_action(check_think_tag_present)

    messages = [{"role": "user", "content": "you are nice"}]
    response = await chat.app.generate_async(messages=messages)

    if test_case.expected_think_tag:
        assert (
            "<think>" in response["content"]
        ), "Think tag should be present in response"
    else:
        assert (
            "<think>" not in response["content"]
        ), "Think tag should not be present in response"

    if test_case.expected_error_message:
        assert (
            "think tag is not allowed" in response["content"]
        ), "Error message should be present"
    else:
        assert (
            "think tag is not allowed" not in response["content"]
        ), "Error message should not be present"


@pytest.mark.asyncio
async def test_output_rails_preserves_reasoning_traces() -> None:
    """Test that output rails preserve reasoning traces when configured to do so."""
    config = RailsConfig.from_content(
        colang_content="""
        define flow check sensitive info
            $not_allowed = execute check_sensitive_info
            if $not_allowed
                bot provide sanitized response
                stop
        define bot provide sanitized response
            "I cannot share sensitive information."
        """,
        yaml_content="""
        models:
          - type: main
            engine: fake
            model: fake
            reasoning_config:
              remove_reasoning_traces: True
        colang_version: "1.0"
        rails:
          output:
            flows:
              - check sensitive info
            apply_to_reasoning_traces: True
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            '<think> I should not share sensitive info </think>\n  "Here is my credit card: 1234-5678-9012-3456"',
        ],
    )

    chat.app.runtime.register_action(check_sensitive_info)

    messages = [{"role": "user", "content": "What's your credit card number?"}]
    response = await chat.app.generate_async(messages=messages)

    assert "<think>" in response["content"], "Reasoning traces should be preserved"
    assert (
        "I should not share sensitive info" in response["content"]
    ), "Reasoning content should be preserved"
    assert (
        "credit card" not in response["content"].lower()
    ), "Sensitive information should be removed"


@pytest.mark.asyncio
async def test_output_rails_without_reasoning_traces() -> None:
    """Test that output rails properly handle responses when reasoning traces are disabled."""
    config = RailsConfig.from_content(
        colang_content="""
        define flow check sensitive info
            $not_allowed = execute check_sensitive_info
            if $not_allowed
                bot provide sanitized response
                stop
        define flow check think tag
            $not_allowed = execute check_think_tag_present
            if $not_allowed
                bot says tag not allowed
                stop

        define bot says tag not allowed
            "<think> tag is not allowed it must be removed"

        define bot provide sanitized response
            "I cannot share sensitive information."
        """,
        yaml_content="""
        models:
          - type: main
            engine: fake
            model: fake
            reasoning_config:
              remove_reasoning_traces: True
        colang_version: "1.0"
        rails:
          input:
            flows:
              - check sensitive info
          output:
            flows:
              - check sensitive info
              - check think tag
            apply_to_reasoning_traces: false
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "<think> I should think more </think> Your credit card number is 1234-5678-9012-3456",
        ],
    )

    chat.app.runtime.register_action(check_sensitive_info)
    chat.app.runtime.register_action(check_think_tag_present)

    # case 1: Sensitive information is blocked by input rail
    messages = [{"role": "user", "content": "What's your credit card number?"}]
    response = await chat.app.generate_async(messages=messages)

    assert "<think>" not in response["content"], "Think tag should not be present"
    assert (
        "I should not share sensitive info" not in response["content"]
    ), "Reasoning content should not be present"
    assert (
        response["content"] == "I cannot share sensitive information."
    ), "Should return sanitized response"

    # case 2: Think tag is preserved but content is sanitized
    messages = [{"role": "user", "content": "Tell me some numbers"}]
    response = await chat.app.generate_async(messages=messages)

    assert "<think>" in response["content"], "Think tag should be present"
    assert (
        "I should not share sensitive info" not in response["content"]
    ), "Reasoning content should not be present"
    assert (
        response["content"]
        == "<think> I should think more </think>I cannot share sensitive information."
    ), "Should preserve think tag but sanitize content"
