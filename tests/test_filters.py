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

import textwrap
from typing import List, Tuple, Union

import pytest

from nemoguardrails.llm.filters import (
    ReasoningExtractionResult,
    extract_and_strip_trace,
    find_reasoning_tokens_position,
    first_turns,
    last_turns,
    to_chat_messages,
    user_assistant_sequence,
)


def test_first_turns():
    colang_history = textwrap.dedent(
        """
        user "Hi, how are you today?"
          express greeting
        bot express greeting
          "Greetings! I am the official NVIDIA Benefits Ambassador AI bot and I'm here to assist you."
        user "What can you help me with?"
          ask capabilities
        bot inform capabilities
          "As an AI, I can provide you with a wide range of services, such as ..."
        """
    ).strip()

    output_1_turn = textwrap.dedent(
        """
        user "Hi, how are you today?"
          express greeting
        bot express greeting
          "Greetings! I am the official NVIDIA Benefits Ambassador AI bot and I'm here to assist you."
        """
    ).strip()

    assert first_turns(colang_history, 1) == output_1_turn
    assert first_turns(colang_history, 2) == colang_history
    assert first_turns(colang_history, 3) == colang_history


def test_last_turns():
    colang_history = textwrap.dedent(
        """
        user "Hi, how are you today?"
          express greeting
        bot express greeting
          "Greetings! I am the official NVIDIA Benefits Ambassador AI bot and I'm here to assist you."
        user "What can you help me with?"
          ask capabilities
        bot inform capabilities
          "As an AI, I can provide you with a wide range of services, such as ..."
        """
    ).strip()

    output_1_turn = textwrap.dedent(
        """
        user "What can you help me with?"
          ask capabilities
        bot inform capabilities
          "As an AI, I can provide you with a wide range of services, such as ..."
        """
    ).strip()

    assert last_turns(colang_history, 1) == output_1_turn
    assert last_turns(colang_history, 2) == colang_history
    assert last_turns(colang_history, 3) == colang_history

    colang_history = textwrap.dedent(
        """
        user "Hi, how are you today?"
          express greeting
        """
    ).strip()

    assert last_turns(colang_history, 1) == colang_history
    assert last_turns(colang_history, 2) == colang_history


def _build_test_string(parts: List[Union[str, Tuple[str, int]]]) -> str:
    """Builds a test string from a list of parts.

    Each part can be a literal string or a (character, count) tuple.
    Example: [("a", 3), "[START]", ("b", 5)] -> "aaa[START]bbbbb"
    """
    result = []
    for part in parts:
        if isinstance(part, str):
            result.append(part)
        elif isinstance(part, tuple) and len(part) == 2:
            char, count = part
            result.append(char * count)
        else:
            raise TypeError(f"Invalid part type in _build_test_string: {part}")
    return "".join(result)


@pytest.mark.parametrize(
    "response, start_token, end_token, expected",
    [
        (
            _build_test_string(
                [
                    ("a", 5),
                    "[START]",
                    ("b", 10),
                    "[END]",
                    ("c", 5),
                ]
            ),
            "[START]",
            "[END]",
            (5, 22),  # 5 a's + 7 START + 10 b = 22
        ),
        # multiple reasoning sections
        (
            _build_test_string(
                [
                    ("a", 3),
                    "[START]",
                    ("b", 4),
                    "[END]",
                    ("c", 3),
                    "[START]",
                    ("d", 4),
                    "[END]",
                    ("e", 3),
                ]
            ),
            "[START]",
            "[END]",
            (
                3,
                33,
            ),
        ),
        (
            _build_test_string(
                [
                    ("a", 2),
                    "[START]",
                    ("b", 2),
                    "[START]",
                    ("c", 2),
                    "[END]",
                    ("d", 2),
                    "[END]",
                    ("e", 2),
                ]
            ),
            "[START]",
            "[END]",
            (
                2,
                27,
            ),
        ),
        (
            _build_test_string([("a", 10)]),
            "[START]",
            "[END]",
            (0, -1),  # no tokens found, start_index is 0
        ),
        (
            _build_test_string(
                [
                    ("a", 5),
                    "[START]",
                    ("b", 5),
                ]
            ),
            "[START]",
            "[END]",
            (5, -1),  # [START] at pos 5, no end token
        ),
        (
            _build_test_string(
                [
                    ("a", 5),
                    "[END]",
                    ("b", 5),
                ]
            ),
            "[START]",
            "[END]",
            (0, 5),  # no start token so 0, end at pos 5
        ),
        (
            "",
            "[START]",
            "[END]",
            (0, -1),  # empty string, start_index is 0
        ),
    ],
)
def test_find_token_positions_for_removal(response, start_token, end_token, expected):
    """Test finding token positions for removal.

    Test cases use _build_test_string for clarity and mathematical obviousness.
    """
    assert find_reasoning_tokens_position(response, start_token, end_token) == expected


@pytest.mark.parametrize(
    "response, start_token, end_token, expected_text, expected_trace",
    [
        (
            "This is an example [START]hidden reasoning[END] of a response.",
            "[START]",
            "[END]",
            "This is an example  of a response.",
            "[START]hidden reasoning[END]",
        ),
        (
            "Before [START]first[END] middle [START]second[END] after.",
            "[START]",
            "[END]",
            "Before  after.",
            "[START]first[END] middle [START]second[END]",
        ),
        (
            "Text [START] first [START] nested [END] second [END] more text.",
            "[START]",
            "[END]",
            "Text  more text.",
            "[START] first [START] nested [END] second [END]",
        ),
        (
            "No tokens here",
            "[START]",
            "[END]",
            "No tokens here",
            None,
        ),
        (
            "Only [START] start token",
            "[START]",
            "[END]",
            "Only [START] start token",
            None,
        ),
        (
            "Only end token [END]",
            "[START]",
            "[END]",
            "",
            "Only end token [END]",
        ),
        (
            "",
            "[START]",
            "[END]",
            "",
            None,
        ),
        # End token before start token (tests the final return path)
        (
            "some [END] text [START]",
            "[START]",
            "[END]",
            "some [END] text [START]",
            None,
        ),
        # Original test cases adapted
        (
            "[END] Out of order [START] tokens [END] example.",
            "[START]",
            "[END]",
            "[END] Out of order  example.",
            "[START] tokens [END]",
        ),
        (
            "[START] nested [START] tokens [END] out of [END] order.",
            "[START]",
            "[END]",
            " order.",
            "[START] nested [START] tokens [END] out of [END]",
        ),
        (
            "[END] [START] [START] example [END] text.",
            "[START]",
            "[END]",
            "[END]  text.",
            "[START] [START] example [END]",
        ),
        (
            "example text.",
            "[START]",
            "[END]",
            "example text.",
            None,
        ),
    ],
)
def test_extract_and_strip_trace(
    response, start_token, end_token, expected_text, expected_trace
):
    """Tests the extraction and stripping of reasoning traces."""
    result = extract_and_strip_trace(response, start_token, end_token)
    assert result.text == expected_text
    assert result.reasoning_trace == expected_trace


class TestToChatMessages:
    def test_to_chat_messages_with_text_only(self):
        """Test to_chat_messages with text-only messages."""
        events = [
            {"type": "UserMessage", "text": "Hello, how are you?"},
            {"type": "StartUtteranceBotAction", "script": "I'm doing well, thank you!"},
            {"type": "UserMessage", "text": "Great to hear."},
        ]

        result = to_chat_messages(events)

        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, how are you?"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "I'm doing well, thank you!"
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "Great to hear."

    def test_to_chat_messages_with_multimodal_content(self):
        """Test to_chat_messages with multimodal content."""
        multimodal_message = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]

        events = [
            {"type": "UserMessage", "text": multimodal_message},
            {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
        ]

        result = to_chat_messages(events)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == multimodal_message
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "I see a cat in the image."

    def test_to_chat_messages_with_empty_events(self):
        """Test to_chat_messages with empty events."""
        events = []
        result = to_chat_messages(events)
        assert result == []


class TestUserAssistantSequence:
    def test_user_assistant_sequence_with_text_only(self):
        """Test user_assistant_sequence with text-only messages."""
        events = [
            {"type": "UserMessage", "text": "Hello, how are you?"},
            {"type": "StartUtteranceBotAction", "script": "I'm doing well, thank you!"},
            {"type": "UserMessage", "text": "Great to hear."},
        ]

        result = user_assistant_sequence(events)

        assert result == (
            "User: Hello, how are you?\n"
            "Assistant: I'm doing well, thank you!\n"
            "User: Great to hear."
        )

    def test_user_assistant_sequence_with_multimodal_content(self):
        """Test user_assistant_sequence with multimodal content."""
        multimodal_message = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]

        events = [
            {"type": "UserMessage", "text": multimodal_message},
            {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
        ]

        result = user_assistant_sequence(events)

        assert result == (
            "User: What's in this image? [+ image]\n"
            "Assistant: I see a cat in the image."
        )

    def test_user_assistant_sequence_with_empty_events(self):
        """Test user_assistant_sequence with empty events."""
        events = []
        result = user_assistant_sequence(events)
        assert result == ""

    def test_user_assistant_sequence_with_multiple_text_parts(self):
        """Test user_assistant_sequence with multiple text parts."""
        multimodal_message = [
            {"type": "text", "text": "Hello!"},
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]

        events = [
            {"type": "UserMessage", "text": multimodal_message},
            {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
        ]

        result = user_assistant_sequence(events)

        assert result == (
            "User: Hello! What's in this image? [+ image]\n"
            "Assistant: I see a cat in the image."
        )

    def test_user_assistant_sequence_with_image_only(self):
        """Test user_assistant_sequence with image only."""
        multimodal_message = [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]

        events = [
            {"type": "UserMessage", "text": multimodal_message},
            {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
        ]

        result = user_assistant_sequence(events)

        assert result == ("User:  [+ image]\nAssistant: I see a cat in the image.")
