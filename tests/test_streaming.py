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

import asyncio
import json
import math

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.streaming import StreamingHandler
from tests.utils import TestChat


@pytest.fixture
def chat_1():
    config: RailsConfig = RailsConfig.from_content(
        config={"models": [], "streaming": True}
    )
    return TestChat(
        config,
        llm_completions=[
            "Hello there! How are you?",
        ],
        streaming=True,
    )


@pytest.mark.asyncio
async def test_streaming_generate_async_api(chat_1):
    streaming_handler = StreamingHandler()

    chunks = []

    async def process_tokens():
        async for chunk in streaming_handler:
            chunks.append(chunk)

            # Or do something else with the token

    asyncio.create_task(process_tokens())

    response = await chat_1.app.generate_async(
        messages=[{"role": "user", "content": "Hi!"}],
        streaming_handler=streaming_handler,
    )

    assert chunks == ["Hello ", "there! ", "How ", "are ", "you?"]
    assert response == {"content": "Hello there! How are you?", "role": "assistant"}


@pytest.mark.asyncio
async def test_stream_async_api(chat_1):
    """Test the simplified stream_async interface"""

    chunks = []
    async for chunk in chat_1.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}],
    ):
        chunks.append(chunk)

    assert chunks == ["Hello ", "there! ", "How ", "are ", "you?"]


@pytest.mark.asyncio
async def test_streaming_predefined_messages():
    """Predefined messages should be streamed as a single chunk."""
    config: RailsConfig = RailsConfig.from_content(
        config={"models": [], "streaming": True},
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot express greeting

        define bot express greeting
          "Hello there!"
        """,
    )
    chat = TestChat(
        config,
        llm_completions=["  express greeting"],
        streaming=True,
    )

    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}],
    ):
        chunks.append(chunk)

    assert chunks == ["Hello there!"]
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_streaming_dynamic_bot_message():
    """Predefined messages should be streamed as a single chunk."""
    config: RailsConfig = RailsConfig.from_content(
        config={"models": [], "streaming": True},
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot express greeting
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            "express greeting",
            '  "Hello there! How are you today?"',
        ],
        streaming=True,
    )

    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}],
    ):
        chunks.append(chunk)

    assert chunks == ["Hello ", "there! ", "How ", "are ", "you ", "today?"]
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_streaming_single_llm_call():
    """Predefined messages should be streamed as a single chunk."""
    config: RailsConfig = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {"dialog": {"single_call": {"enabled": True}}},
            "streaming": True,
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot express greeting
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            '  express greeting\nbot express greeting\n  "Hi, how are you doing?"'
        ],
        streaming=True,
    )

    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}],
    ):
        chunks.append(chunk)

    assert chunks == ["Hi, ", "how ", "are ", "you ", "doing?"]
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_streaming_single_llm_call_with_message_override():
    """Predefined messages should be streamed as a single chunk."""
    config: RailsConfig = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {"dialog": {"single_call": {"enabled": True}}},
            "streaming": True,
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot express greeting

        define bot express greeting
          "Hey! Welcome back!"
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            '  express greeting\nbot express greeting\n  "Hi, how are you doing?"'
        ],
        streaming=True,
    )

    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}],
    ):
        chunks.append(chunk)

    assert chunks == ["Hey! Welcome back!"]

    # Wait for proper cleanup, otherwise we get a Runtime Error
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_streaming_single_llm_call_with_next_step_override_and_dynamic_message():
    """Predefined messages should be streamed as a single chunk."""
    config: RailsConfig = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {"dialog": {"single_call": {"enabled": True}}},
            "streaming": True,
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot tell joke
        """,
    )
    chat = TestChat(
        config,
        llm_completions=[
            '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
            '  "This is a funny joke."',
        ],
        streaming=True,
    )

    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}],
    ):
        chunks.append(chunk)

    assert chunks == ["This ", "is ", "a ", "funny ", "joke."]

    # Wait for proper cleanup, otherwise we get a Runtime Error
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.fixture
def output_rails_streaming_config():
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
    """Helper function to run the self check test with the given config, llm completions and expected chunks."""

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
async def test_streaming_output_rails_allowed(output_rails_streaming_config):
    """Checks if the streaming output rails allow the completions without any blocking."""

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a funny joke but you should not laught at it because you will be cursed!."',
    ]
    # when we do not yield tokens
    expected_chunks = [
        "This is a funny ",
        "joke but ",
        "you should ",
        "not laught ",
        "at it ",
        "because you ",
        "will be ",
        "cursed!.",
    ]

    expected_tokens = [
        "This",
        " is",
        " a",
        " funny",
        "joke",
        " but",
        "you",
        " should",
        "not",
        " laught",
        "at",
        " it",
        "because",
        " you",
        "will",
        " be",
        "cursed!.",
    ]
    tokens = await run_self_check_test(output_rails_streaming_config, llm_completions)
    assert tokens == expected_tokens
    # number of buffered chunks should be equal to the number of actions
    # we are apply #calculate_number_of_actions of time the output rails
    # FIXME: nice but stupid
    assert len(expected_chunks) == _calculate_number_of_actions(
        len(llm_completions[1].lstrip().split(" ")), 4, 2
    )
    # Wait for proper cleanup, otherwise we get a Runtime Error
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_streaming_output_rails_blocked(output_rails_streaming_config):
    """This test checks if the streaming output rails block the completions when a BLOCK keyword is present.
    It verifies that the chunks contain the stop data when the BLOCK keyword is detected.
    """
    # when BLOCK is in the bot message, it gets blocked by output rails
    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a funny joke but you should laught at it because [BLOCK] you will be cursed!."',
    ]
    expected_chunks = [" {DATA: STOP}"]
    chunks = await run_self_check_test(output_rails_streaming_config, llm_completions)
    expected_output = (
        '{"event": "ABORT", "data": {"reason": "Blocked by self check output rails."}}'
    )
    parsed_output = json.loads(expected_output)

    assert parsed_output in [
        json.loads(chunk) for chunk in chunks if chunk.startswith('{"event": "ABORT"')
    ]
    # Wait for proper cleanup, otherwise we get a Runtime Error
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_streaming_output_rails_blocked_at_first_call(
    output_rails_streaming_config,
):
    """This test checks if the streaming output rails block the completions when a BLOCK keyword is present at the first call.
    It verifies that the first chunk is the stop data and that there is only one chunk.
    """
    # when BLOCK is in the bot message, it gets blocked by output rails
    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "[BLOCK] This is a funny joke but you should laught at it because [BLOCK] you will be cursed!."',
    ]
    chunks = await run_self_check_test(output_rails_streaming_config, llm_completions)

    expected_output = {
        "event": "ABORT",
        "data": {"reason": "Blocked by self check output rails."},
    }

    # Parse the JSON string into a dictionary
    parsed_output = json.loads(chunks[0])

    assert expected_output == parsed_output
    assert len(chunks) == 1
    # Wait for proper cleanup, otherwise we get a Runtime Error
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


def _calculate_number_of_actions(input_length, chunk_size, context_size):
    if chunk_size <= context_size:
        raise ValueError("chunk_size must be greater than context_size.")
    if input_length <= chunk_size:
        return 1
    return math.ceil((input_length - context_size) / (chunk_size - context_size))


@pytest.mark.asyncio
async def test_streaming_error_handling():
    """Test that errors during streaming are properly formatted and returned."""
    # Create a config with an invalid model to trigger an error
    config: RailsConfig = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "non-existent-model",
                }
            ],
            "streaming": True,
        }
    )

    # Create a mock chat with an error response
    chat = TestChat(
        config,
        llm_completions=["Error"],  # This isn't going to be used due to the error
        streaming=True,
        llm_exception=Exception(
            "Error code: 404 - {'error': {'message': 'The model `non-existent-model` does not exist or you do not have access to it.', 'type': 'invalid_request_error', 'param': None, 'code': 'model_not_found'}}"
        ),
    )

    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}],
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    error_chunk = chunks[0]

    # Verify the error chunk is a valid json
    error_data = json.loads(error_chunk)
    assert "error" in error_data
    assert "message" in error_data["error"]
    assert (
        "The model `non-existent-model` does not exist"
        in error_data["error"]["message"]
    )
    assert error_data["error"]["type"] == "invalid_request_error"
    assert error_data["error"]["code"] == "model_not_found"

    # Wait for proper cleanup, otherwise we get a Runtime Error
    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
