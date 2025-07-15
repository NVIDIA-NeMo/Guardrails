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

"""Tests for the explicitly enabled output rails streaming functionality."""

import asyncio
import json
import math
from json.decoder import JSONDecodeError

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
async def test_stream_async_streaming_disabled(output_rails_streaming_config_default):
    """Tests if stream_async returns a StreamingHandler instance when streaming is disabled"""

    llmrails = LLMRails(output_rails_streaming_config_default)

    result = llmrails.stream_async(prompt="test")
    assert isinstance(
        result, StreamingHandler
    ), "Expected StreamingHandler instance when streaming is disabled"


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
    """Tests if output rails streaming default config do not block content with BLOCK keyword"""

    # text with a BLOCK keyword
    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a [BLOCK] joke that should be blocked."',
    ]

    chunks = await run_self_check_test(
        output_rails_streaming_config_default, llm_completions
    )

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
    assert len(error_chunks) == 0
    assert expected_error not in error_chunks

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


@pytest.mark.asyncio
async def test_streaming_output_rails_default_config_not_blocked_at_start(
    output_rails_streaming_config_default,
):
    """Tests blocking with BLOCK at the very beginning of the response does not return abort sse"""

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "[BLOCK] This should be blocked immediately at the start."',
    ]

    chunks = await run_self_check_test(
        output_rails_streaming_config_default, llm_completions
    )

    with pytest.raises(JSONDecodeError):
        json.loads(chunks[0])

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
