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

import textwrap
from typing import List, Tuple, Union

import pytest

from nemoguardrails.llm.filters import (
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
