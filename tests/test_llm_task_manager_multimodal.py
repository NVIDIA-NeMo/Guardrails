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

"""Tests for the LLMTaskManager with multimodal content."""

import random
import string

import jinja2
import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.llm.taskmanager import LLMTaskManager


def test_history_integration_with_filters():
    """Test the integration of filters with history processing."""
    from nemoguardrails.llm.filters import to_chat_messages, user_assistant_sequence

    # Create events with multimodal content
    multimodal_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]

    events = [
        {"type": "UserMessage", "text": multimodal_message},
        {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
    ]

    # Mock a template rendering environment similar to what the task manager would use

    env = jinja2.Environment()
    env.filters["user_assistant_sequence"] = user_assistant_sequence
    env.filters["to_chat_messages"] = to_chat_messages

    # Test with user_assistant_sequence filter
    template1 = env.from_string("{{ events | user_assistant_sequence }}")
    result1 = template1.render(events=events)

    # Test with to_chat_messages filter
    template2 = env.from_string("{{ events | to_chat_messages | tojson }}")
    result2 = template2.render(events=events)

    # Verify the multimodal content is correctly formatted by our filters
    assert "User: What's in this image? [+ image]" in result1
    assert "Assistant: I see a cat in the image." in result1

    # Verify that to_chat_messages preserves the structure
    import json

    chat_messages = json.loads(result2)
    assert len(chat_messages) == 2
    assert chat_messages[0]["role"] == "user"
    assert isinstance(chat_messages[0]["content"], list)
    assert chat_messages[0]["content"][0]["type"] == "text"
    assert chat_messages[0]["content"][1]["type"] == "image_url"


def test_user_assistant_sequence_with_multimodal():
    """Test that the user_assistant_sequence filter correctly formats multimodal content."""
    from nemoguardrails.llm.filters import user_assistant_sequence

    # Create events with multimodal content
    multimodal_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]

    events = [
        {"type": "UserMessage", "text": "Hello!"},
        {"type": "StartUtteranceBotAction", "script": "Hi there! How can I help you?"},
        {"type": "UserMessage", "text": multimodal_message},
        {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
    ]

    # Apply the user_assistant_sequence filter
    formatted_history = user_assistant_sequence(events)

    # Verify the multimodal content is correctly formatted
    assert "User: Hello!" in formatted_history
    assert "Assistant: Hi there! How can I help you?" in formatted_history
    assert "User: What's in this image? [+ image]" in formatted_history
    assert "Assistant: I see a cat in the image." in formatted_history


def test_to_chat_messages_multimodal_integration():
    """Test integration of to_chat_messages with multimodal content."""
    from nemoguardrails.llm.filters import to_chat_messages

    # Create events with multimodal content
    multimodal_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
    ]

    events = [
        {"type": "UserMessage", "text": multimodal_message},
        {"type": "StartUtteranceBotAction", "script": "I see a cat in the image."},
    ]

    # Apply the to_chat_messages filter
    chat_messages = to_chat_messages(events)

    # Verify that the structure is preserved correctly
    assert len(chat_messages) == 2
    assert chat_messages[0]["role"] == "user"
    assert isinstance(chat_messages[0]["content"], list)
    assert len(chat_messages[0]["content"]) == 2
    assert chat_messages[0]["content"][0]["type"] == "text"
    assert chat_messages[0]["content"][0]["text"] == "What's in this image?"
    assert chat_messages[0]["content"][1]["type"] == "image_url"
    assert (
        chat_messages[0]["content"][1]["image_url"]["url"]
        == "https://example.com/image.jpg"
    )

    assert chat_messages[1]["role"] == "assistant"
    assert chat_messages[1]["content"] == "I see a cat in the image."


@pytest.fixture
def config():
    return RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-4-vision-preview",
                }
            ]
        }
    )


@pytest.fixture
def task_manager(config):
    return LLMTaskManager(config)


def get_long_base64_str(length: int):
    # Set seed for deterministic results
    random.seed(13)

    # a dummy base64 string with valid alphanumeric characters
    # https://www.rfc-editor.org/rfc/rfc4648
    base64_chars = string.ascii_letters + string.digits + "+/"
    long_base64 = "".join(random.choices(base64_chars, k=length))

    return long_base64


@pytest.mark.parametrize("image_type", ["jpeg", "png"])
def test_message_length_with_base64_image(task_manager, image_type):
    """Test that base64 images don't count their full length in message length calculation."""
    # Create a dummy base64 string

    long_base64 = get_long_base64_str(100000)

    # Create a multimodal message with base64 image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is a test message",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{long_base64}"
                    },
                },
            ],
        }
    ]

    length = task_manager._get_messages_text_length(messages)

    # the length should only include the text and a placeholder for the image
    # NOT the full base64 string
    expected_length = len("This is a test message\n[IMAGE_CONTENT]\n")

    assert length == expected_length, (
        f"Expected length ~{expected_length}, got {length} "
        f"(Should not include full base64 length of {len(long_base64)})"
    )

    # length is much shorter than the actual base64 data
    assert length < len(
        long_base64
    ), "Length should be much shorter than the actual base64 data"


def test_regular_url_length(task_manager):
    """Test that regular image URLs count their placeholder"""
    regular_url = "https://example.com/image.jpg"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is a test",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": regular_url},
                },
            ],
        }
    ]

    length = task_manager._get_messages_text_length(messages)

    image_placeholder = "[IMAGE_CONTENT]\n"

    # the len should include the text and the full url
    expected_length = len("This is a test\n" + image_placeholder)

    assert length == expected_length, (
        f"Expected length {expected_length}, got {length} "
        f"(Should include full URL length of {len(regular_url)})"
    )


def test_base64_embedded_in_string(task_manager):
    """Test handling of base64 data embedded within a string."""
    long_base64 = get_long_base64_str(100000)

    messages = [
        {
            "role": "system",
            "content": f"System message with embedded image: data:image/jpeg;base64,{long_base64}",
        }
    ]

    length = task_manager._get_messages_text_length(messages)
    expected_length = len("System message with embedded image: [IMAGE_CONTENT]\n")

    assert length == expected_length, (
        f"Expected length {expected_length}, got {length}. "
        f"Base64 string should be replaced with placeholder."
    )


def test_multiple_base64_images(task_manager):
    """Test handling of multiple base64 images in a single message."""

    long_base64 = get_long_base64_str(50000)

    # openai supports multiple images in a single message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are two images:",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{long_base64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{long_base64}"},
                },
            ],
        }
    ]

    length = task_manager._get_messages_text_length(messages)
    expected_length = len("Here are two images:\n[IMAGE_CONTENT]\n[IMAGE_CONTENT]\n")

    assert length == expected_length, (
        f"Expected length {expected_length}, got {length}. "
        f"Both base64 strings should be replaced with placeholders."
    )


def test_multiple_base64_embedded_in_string(task_manager):
    """Test handling of multiple base64 images embedded in a string."""

    base64_segment = get_long_base64_str(10000)

    # openai supports multiple images in a single message
    content_string = (
        f"First image: data:image/jpeg;base64,{base64_segment} "
        f"Second image: data:image/png;base64,{base64_segment}"
    )

    messages = [
        {
            "role": "user",
            "content": content_string,
        }
    ]

    length = task_manager._get_messages_text_length(messages)
    expected_length = len(
        "First image: [IMAGE_CONTENT] Second image: [IMAGE_CONTENT]\n"
    )

    assert length == expected_length, (
        f"Expected length {expected_length}, got {length}. "
        f"Both embedded base64 imags should be replaced with placeholders."
    )


def test_multiple_message_types(task_manager):
    """Test handling of multiple message types with base64 images."""

    long_base64 = get_long_base64_str(50000)

    messages = [
        {
            "role": "system",
            "content": "System message",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "User message with image",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{long_base64}"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "Assistant response",
        },
        {
            "role": "user",
            "content": f"User with embedded image: data:image/jpeg;base64,{long_base64}",
        },
    ]

    length = task_manager._get_messages_text_length(messages)
    expected_length = len(
        "System message\n"
        "User message with image\n"
        "[IMAGE_CONTENT]\n"
        "Assistant response\n"
        "User with embedded image: [IMAGE_CONTENT]\n"
    )

    assert length == expected_length, (
        f"Expected length {expected_length}, got {length}. "
        f"Base64 images should be replaced with placeholders in all message types."
    )
