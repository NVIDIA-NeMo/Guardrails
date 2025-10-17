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

import pytest

from nemoguardrails import LLMRails, RailsConfig
from tests.utils import FakeLLM, TestChat


@pytest.mark.asyncio
async def test_system_message_conversion_v1():
    """Test that system messages are correctly converted to SystemMessage events in Colang 1.0."""

    config = RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                }
            ],
            "colang_version": "1.0",
        }
    )

    llm = FakeLLM(responses=["Hello!"])
    llm_rails = LLMRails(config=config, llm=llm)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    events = llm_rails._get_events_for_messages(messages, None)

    system_messages = [event for event in events if event["type"] == "SystemMessage"]
    assert len(system_messages) == 1
    assert system_messages[0]["content"] == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_system_message_conversion_v2x():
    """Test that system messages are correctly converted to SystemMessage events in Colang 2.x."""

    config = RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                }
            ],
            "colang_version": "2.x",
        }
    )

    llm = FakeLLM(responses=["Hello!"])
    llm_rails = LLMRails(config=config, llm=llm)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    events = llm_rails._get_events_for_messages(messages, None)

    system_messages = [event for event in events if event["type"] == "SystemMessage"]
    assert len(system_messages) == 1
    assert system_messages[0]["content"] == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_system_message_conversion_multiple():
    """Test that multiple system messages are correctly converted to SystemMessage events."""

    config = RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                }
            ],
        }
    )

    llm = FakeLLM(responses=["Hello!"])
    llm_rails = LLMRails(config=config, llm=llm)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "Please provide detailed thinking."},
        {"role": "user", "content": "Hello!"},
    ]

    events = llm_rails._get_events_for_messages(messages, None)

    system_messages = [event for event in events if event["type"] == "SystemMessage"]
    assert len(system_messages) == 2
    assert system_messages[0]["content"] == "You are a helpful assistant."
    assert system_messages[1]["content"] == "Please provide detailed thinking."


@pytest.mark.asyncio
async def test_system_message_end_to_end():
    """Test that system messages are correctly processed in an end-to-end scenario."""
    config = RailsConfig.parse_object(
        {
            "models": [
                {
                    "type": "main",
                    "engine": "fake",
                    "model": "fake",
                }
            ],
        }
    )

    llm = FakeLLM(responses=["Hello there!"])
    llm_rails = LLMRails(config=config, llm=llm)

    response = await llm_rails.generate_async(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
        ]
    )

    assert response["role"] == "assistant"
    assert response["content"] == "Hello there!"
