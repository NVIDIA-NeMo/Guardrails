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

"""Tests for the explicitly enabled output rails streaming functionality."""

import asyncio
import json
from json.decoder import JSONDecodeError
from typing import AsyncIterator

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.streaming import StreamingHandler
from tests.utils import TestChat


@pytest.fixture
def output_rails_streaming_config():
    """Config for testing output rails with streaming explicitly enabled"""

    return RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "flows": {"self check output"},
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                        "stream_first": False,
                    },
                }
            },
            "streaming": False,
            "prompts": [{"task": "self_check_output", "content": "a test template"}],
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot tell joke
        """,
    )


@pytest.fixture
def output_rails_streaming_config_default():
    """Config for testing output rails with default streaming settings"""

    return RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "flows": {"self check output"},
                }
            },
            "streaming": True,
            "prompts": [{"task": "self_check_output", "content": "a test template"}],
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot tell joke
        """,
    )


@pytest.mark.asyncio
async def test_stream_async_streaming_enabled(output_rails_streaming_config):
    """Tests if stream_async returns does not return StreamingHandler instance when streaming is enabled"""

    llmrails = LLMRails(output_rails_streaming_config)

    result = llmrails.stream_async(prompt="test")
    assert not isinstance(
        result, StreamingHandler
    ), "Did not expect StreamingHandler instance when streaming is enabled"


@action(is_system_action=True, output_mapping=lambda result: not result)
def self_check_output(**params):
    """A dummy self check action that checks if the bot message contains the BLOCK keyword."""

    if params.get("context", {}).get("bot_message"):
        bot_message_chunk = params.get("context", {}).get("bot_message")
        print(f"bot_message_chunk: {bot_message_chunk}")
        if "BLOCK" in bot_message_chunk:
            return False

    return True


async def run_self_check_test(config, llm_completions):
    """Helper function to run the self check test with the given config, llm completions"""

    chat = TestChat(
        config,
        llm_completions=llm_completions,
        streaming=True,
    )
    chat.app.register_action(self_check_output)
    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}],
    ):
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_streaming_output_rails_blocked_explicit(output_rails_streaming_config):
    """Tests if explicitly enabled output rails streaming blocks content with BLOCK keyword"""

    # text with a BLOCK keyword
    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a [BLOCK] joke that should be blocked."',
    ]

    chunks = await run_self_check_test(output_rails_streaming_config, llm_completions)

    expected_error = {
        "error": {
            "message": "Blocked by self check output rails.",
            "type": "guardrails_violation",
            "param": "self check output",
            "code": "content_blocked",
        }
    }

    error_chunks = [
        json.loads(chunk) for chunk in chunks if chunk.startswith('{"error":')
    ]
    assert len(error_chunks) > 0
    assert expected_error in error_chunks

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_streaming_output_rails_blocked_default_config(
    output_rails_streaming_config_default,
):
    """Tests that stream_async raises an error with default config (output rails without explicit streaming config)"""

    llmrails = LLMRails(output_rails_streaming_config_default)

    with pytest.raises(ValueError) as exc_info:
        async for chunk in llmrails.stream_async(
            messages=[{"role": "user", "content": "Hi!"}]
        ):
            pass

    assert str(exc_info.value) == (
        "stream_async() cannot be used when output rails are configured but "
        "rails.output.streaming.enabled is False. Either set "
        "rails.output.streaming.enabled to True in your configuration, or use "
        "generate_async() instead of stream_async()."
    )

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_streaming_output_rails_blocked_at_start(output_rails_streaming_config):
    """Tests blocking with BLOCK at the very beginning of the response"""

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "[BLOCK] This should be blocked immediately at the start."',
    ]

    chunks = await run_self_check_test(output_rails_streaming_config, llm_completions)

    expected_error = {
        "error": {
            "message": "Blocked by self check output rails.",
            "type": "guardrails_violation",
            "param": "self check output",
            "code": "content_blocked",
        }
    }

    assert len(chunks) == 1
    assert json.loads(chunks[0]) == expected_error

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


async def simple_token_generator() -> AsyncIterator[str]:
    """Simple generator that yields tokens."""
    tokens = ["Hello", " ", "world", "!"]
    for token in tokens:
        yield token


async def offensive_token_generator() -> AsyncIterator[str]:
    """Generator that yields potentially offensive content."""

    tokens = ["This", " ", "is", " ", "offensive", " ", "content", " ", "idiot", "!"]
    for token in tokens:
        yield token


@pytest.mark.asyncio
async def test_external_generator_without_output_rails():
    """Test that external generator works without output rails."""
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {},
            "streaming": True,
        }
    )

    rails = LLMRails(config)

    tokens = []
    async for token in rails.stream_async(generator=simple_token_generator()):
        tokens.append(token)

    assert tokens == ["Hello", " ", "world", "!"]
    assert "".join(tokens) == "Hello world!"


@pytest.mark.asyncio
async def test_external_generator_with_output_rails_allowed():
    """Test that external generator works with output rails that allow content."""
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "flows": ["self check output"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                        "stream_first": False,
                    },
                }
            },
            "streaming": True,
            "prompts": [
                {"task": "self_check_output", "content": "Check: {{ bot_response }}"}
            ],
        },
        colang_content="""
        define flow self check output
          execute self_check_output
        """,
    )

    rails = LLMRails(config)

    @action(name="self_check_output")
    async def self_check_output(**kwargs):
        return True

    rails.register_action(self_check_output, "self_check_output")

    tokens = []
    async for token in rails.stream_async(
        generator=simple_token_generator(),
        messages=[{"role": "user", "content": "Hello"}],
    ):
        tokens.append(token)

    assert tokens == ["Hello", " ", "world", "!"]


@pytest.mark.asyncio
async def test_external_generator_with_output_rails_blocked():
    """Test that external generator content can be blocked by output rails."""
    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "flows": ["self check output"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 6,
                        "context_size": 2,
                        "stream_first": False,
                    },
                }
            },
            "streaming": True,
            "prompts": [
                {"task": "self_check_output", "content": "Check: {{ bot_response }}"}
            ],
        },
        colang_content="""
        define flow self check output
          execute self_check_output
        """,
    )

    rails = LLMRails(config)

    @action(name="self_check_output")
    async def self_check_output(**kwargs):
        bot_message = kwargs.get(
            "bot_message", kwargs.get("context", {}).get("bot_message", "")
        )
        # block if message contains "offensive" or "idiot"
        if "offensive" in bot_message.lower() or "idiot" in bot_message.lower():
            return False
        return True

    rails.register_action(self_check_output, "self_check_output")

    tokens = []
    error_received = False

    async for token in rails.stream_async(
        generator=offensive_token_generator(),
        messages=[{"role": "user", "content": "Generate something"}],
    ):
        if isinstance(token, str) and token.startswith('{"error"'):
            error_received = True
            break
        tokens.append(token)

    assert error_received, "Expected to receive an error JSON when content is blocked"
    assert len(tokens) == 0


@pytest.mark.asyncio
async def test_external_generator_with_custom_llm():
    """Test using external generator as a custom LLM replacement."""

    async def custom_llm_generator(messages):
        """Simulate a custom LLM that generates based on input."""

        user_message = messages[-1]["content"] if messages else ""

        if "weather" in user_message.lower():
            response = "The weather is sunny today!"
        elif "name" in user_message.lower():
            response = "I am an AI assistant."
        else:
            response = "I can help you with that."

        for token in response.split(" "):
            yield token + " "

    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {},
            "streaming": True,
        }
    )

    rails = LLMRails(config)

    messages = [{"role": "user", "content": "What's the weather?"}]
    tokens = []

    async for token in rails.stream_async(
        generator=custom_llm_generator(messages), messages=messages
    ):
        tokens.append(token)

    result = "".join(tokens).strip()
    assert result == "The weather is sunny today!"


@pytest.mark.asyncio
async def test_external_generator_empty_stream():
    """Test that empty generator streams work correctly."""

    async def empty_generator():
        if False:
            yield

    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {},
            "streaming": True,
        }
    )

    rails = LLMRails(config)

    tokens = []
    async for token in rails.stream_async(generator=empty_generator()):
        tokens.append(token)

    assert tokens == []


@pytest.mark.asyncio
async def test_external_generator_single_chunk():
    """Test generator that yields a single large chunk."""

    async def single_chunk_generator():
        yield "This is a complete response in a single chunk."

    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "flows": ["self check output"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 10,
                        "context_size": 5,
                        "stream_first": True,
                    },
                }
            },
            "streaming": True,
            "prompts": [
                {"task": "self_check_output", "content": "Check: {{ bot_response }}"}
            ],
        },
        colang_content="""
        define flow self check output
          execute self_check_output
        """,
    )

    rails = LLMRails(config)

    @action(name="self_check_output")
    async def self_check_output(**kwargs):
        return True

    rails.register_action(self_check_output, "self_check_output")

    tokens = []
    async for token in rails.stream_async(generator=single_chunk_generator()):
        tokens.append(token)

    assert "".join(tokens) == "This is a complete response in a single chunk."
