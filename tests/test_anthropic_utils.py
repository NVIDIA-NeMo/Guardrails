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

"""Tests for Anthropic ↔ NeMo ChatMessage conversion utilities.

Covers thinking/redacted_thinking block preservation through the round-trip.
"""

from nemoguardrails.llm.anthropic_utils import (
    anthropic_to_nemo_messages,
    nemo_to_anthropic_messages,
)
from nemoguardrails.types import ChatMessage, Role, ToolCall, ToolCallFunction

THINKING_BLOCK = {
    "type": "thinking",
    "thinking": "Let me reason about this...",
    "signature": "abc123sig",
}

REDACTED_THINKING_BLOCK = {
    "type": "redacted_thinking",
    "data": "opaque-redacted-data",
}


def test_anthropic_to_nemo_preserves_thinking_blocks():
    messages = [
        {
            "role": "assistant",
            "content": [
                THINKING_BLOCK,
                {"type": "text", "text": "The answer is 42."},
            ],
        }
    ]
    result = anthropic_to_nemo_messages(None, messages)

    assert len(result) == 1
    msg = result[0]
    assert msg.role == Role.ASSISTANT
    assert msg.content == "The answer is 42."
    assert msg.reasoning == [THINKING_BLOCK]


def test_anthropic_to_nemo_preserves_redacted_thinking():
    messages = [
        {
            "role": "assistant",
            "content": [
                REDACTED_THINKING_BLOCK,
                {"type": "text", "text": "Here is my answer."},
            ],
        }
    ]
    result = anthropic_to_nemo_messages(None, messages)

    msg = result[0]
    assert msg.reasoning == [REDACTED_THINKING_BLOCK]


def test_anthropic_to_nemo_preserves_mixed_thinking():
    messages = [
        {
            "role": "assistant",
            "content": [
                THINKING_BLOCK,
                REDACTED_THINKING_BLOCK,
                {"type": "text", "text": "Response."},
            ],
        }
    ]
    result = anthropic_to_nemo_messages(None, messages)

    msg = result[0]
    assert msg.reasoning == [THINKING_BLOCK, REDACTED_THINKING_BLOCK]


def test_anthropic_to_nemo_no_thinking_yields_none():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "No thinking here."}],
        }
    ]
    result = anthropic_to_nemo_messages(None, messages)
    assert result[0].reasoning is None


def test_nemo_to_anthropic_reconstructs_thinking_blocks():
    msg = ChatMessage(
        role=Role.ASSISTANT,
        content="The answer is 42.",
        reasoning=[THINKING_BLOCK],
    )
    system, messages = nemo_to_anthropic_messages([msg])

    assert system is None
    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[0] == THINKING_BLOCK
    assert content[1] == {"type": "text", "text": "The answer is 42."}


def test_nemo_to_anthropic_reconstructs_redacted_thinking():
    msg = ChatMessage(
        role=Role.ASSISTANT,
        content="Answer.",
        reasoning=[REDACTED_THINKING_BLOCK],
    )
    _, messages = nemo_to_anthropic_messages([msg])

    content = messages[0]["content"]
    assert content[0] == REDACTED_THINKING_BLOCK
    assert content[1] == {"type": "text", "text": "Answer."}


def test_nemo_to_anthropic_string_reasoning_creates_thinking_block():
    msg = ChatMessage(
        role=Role.ASSISTANT,
        content="Result.",
        reasoning="I thought about it.",
    )
    _, messages = nemo_to_anthropic_messages([msg])

    content = messages[0]["content"]
    assert content[0] == {"type": "thinking", "thinking": "I thought about it."}
    assert content[1] == {"type": "text", "text": "Result."}


def test_nemo_to_anthropic_no_reasoning_no_thinking_block():
    msg = ChatMessage(role=Role.ASSISTANT, content="Plain response.")
    _, messages = nemo_to_anthropic_messages([msg])

    content = messages[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"


def test_thinking_block_round_trip():
    original_messages = [
        {
            "role": "assistant",
            "content": [
                THINKING_BLOCK,
                REDACTED_THINKING_BLOCK,
                {"type": "text", "text": "My answer."},
            ],
        }
    ]

    nemo_msgs = anthropic_to_nemo_messages(None, original_messages)
    _, reconstructed = nemo_to_anthropic_messages(nemo_msgs)

    content = reconstructed[0]["content"]
    assert content[0] == THINKING_BLOCK
    assert content[1] == REDACTED_THINKING_BLOCK
    assert content[2] == {"type": "text", "text": "My answer."}


def test_thinking_blocks_before_tool_use_in_output():
    msg = ChatMessage(
        role=Role.ASSISTANT,
        content="",
        reasoning=[THINKING_BLOCK],
        tool_calls=[
            ToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(
                    name="get_weather",
                    arguments={"city": "London"},
                ),
            )
        ],
    )
    _, messages = nemo_to_anthropic_messages([msg])

    content = messages[0]["content"]
    assert content[0] == THINKING_BLOCK
    assert content[-1]["type"] == "tool_use"
    assert content[-1]["name"] == "get_weather"
