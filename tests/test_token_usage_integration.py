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

"""Integration tests for token usage tracking with streaming LLMs.

Note about token usage testing:
- In production, `stream_usage=True` is passed to ALL providers when streaming is enabled
- providers that don't support this parameter will simply ignore it
- for testing purposes, we simulate expected behavior based on known provider capabilities
- the _TEST_PROVIDERS_WITH_TOKEN_USAGE_SUPPORT list in nemoguardrails.llm.types defines
  which providers are known to support token usage reporting during streaming
- test cases verify both supported and unsupported provider behavior
"""

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.context import llm_stats_var
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse
from tests.utils import TestChat


@pytest.fixture
def streaming_config():
    # using 'openai' engine which is known to support token usage reporting.
    # in tests, the FakeLLM will simulate returning token usage data for this provider.
    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-4",
                }
            ],
            "streaming": True,
        },
        colang_content="""
        define user express greeting
          "hello"

        define flow
          user express greeting
          bot express greeting

        define bot express greeting
          "Hello there!"
        """,
    )
    return config


@pytest.fixture
def llm_calls_option():
    return GenerationOptions(log={"llm_calls": True})


@pytest.mark.asyncio
async def test_token_usage_integration_with_streaming(streaming_config, llm_calls_option):
    """Integration test for token usage tracking with streaming enabled using GenerationOptions."""

    # token usage data that the FakeLLM will return
    token_usage_data = [{"total_tokens": 15, "prompt_tokens": 8, "completion_tokens": 7}]

    chat = TestChat(
        streaming_config,
        llm_completions=["  express greeting"],
        streaming=True,
        token_usage=token_usage_data,
    )

    result = await chat.app.generate_async(messages=[{"role": "user", "content": "hello"}], options=llm_calls_option)

    assert isinstance(result, GenerationResponse)
    assert result.response[0]["content"] == "Hello there!"

    assert result.log is not None
    assert result.log.llm_calls is not None
    assert len(result.log.llm_calls) > 0

    llm_call = result.log.llm_calls[0]
    assert llm_call.total_tokens == 15
    assert llm_call.prompt_tokens == 8
    assert llm_call.completion_tokens == 7


@pytest.mark.asyncio
async def test_token_usage_integration_streaming_api(streaming_config, llm_calls_option):
    """Integration test for token usage tracking with streaming using GenerationOptions."""

    token_usage_data = [{"total_tokens": 25, "prompt_tokens": 12, "completion_tokens": 13}]

    chat = TestChat(
        streaming_config,
        llm_completions=["  express greeting"],
        streaming=True,
        token_usage=token_usage_data,
    )

    result = await chat.app.generate_async(messages=[{"role": "user", "content": "Hi!"}], options=llm_calls_option)

    assert result.response[0]["content"] == "Hello there!"

    assert result.log is not None
    assert result.log.llm_calls is not None
    assert len(result.log.llm_calls) > 0

    llm_call = result.log.llm_calls[0]
    assert llm_call.total_tokens == 25
    assert llm_call.prompt_tokens == 12
    assert llm_call.completion_tokens == 13


@pytest.mark.asyncio
async def test_token_usage_integration_actual_streaming(llm_calls_option):
    """Test that verifies actual streaming works with token usage tracking."""

    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-4",
                }
            ],
            "streaming": True,
        },
        colang_content="""
        define user ask question
          "what is AI?"

        define flow
          user ask question
          bot respond about ai

        define bot respond about ai
          "AI stands for Artificial Intelligence"
        """,
    )

    token_usage_data = [{"total_tokens": 30, "prompt_tokens": 15, "completion_tokens": 15}]

    chat = TestChat(
        config,
        llm_completions=["  ask question"],
        streaming=True,
        token_usage=token_usage_data,
    )

    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "what is AI?"}],
    ):
        chunks.append(chunk)

    expected_chunks = ["AI stands for Artificial Intelligence"]
    assert chunks == expected_chunks

    # now verify that even in streaming mode, if we use generate_async with options
    # we can get the token usage information

    chat.llm.i = 0  # reset counter to run the same scenario again

    result = await chat.app.generate_async(
        messages=[{"role": "user", "content": "what is AI?"}], options=llm_calls_option
    )

    assert result.log is not None
    assert result.log.llm_calls is not None
    assert len(result.log.llm_calls) > 0

    llm_call = result.log.llm_calls[0]
    assert llm_call.total_tokens == 30
    assert llm_call.prompt_tokens == 15
    assert llm_call.completion_tokens == 15


@pytest.mark.asyncio
async def test_token_usage_integration_multiple_calls(llm_calls_option):
    """Integration test for token usage tracking across multiple LLM calls using GenerationOptions."""

    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "azure_openai",
                    "model": "gpt-4",
                }
            ],
            "streaming": True,
        },
        colang_content="""
        define user ask question
          "what is 2+2?"

        define flow
          user ask question
          execute math_calculation
          bot provide answer
        """,
    )

    # token usage for two LLM calls (intent generation + response generation)
    token_usage_data = [
        {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4},
        {"total_tokens": 20, "prompt_tokens": 12, "completion_tokens": 8},
    ]

    async def math_calculation():
        return "2 + 2 = 4"

    chat = TestChat(
        config,
        llm_completions=[
            "  ask question",  # intent generation
            "The answer is 4",  # bot message generation
        ],
        streaming=True,
        token_usage=token_usage_data,
    )

    chat.app.register_action(math_calculation)

    result = await chat.app.generate_async(
        messages=[{"role": "user", "content": "what is 2+2?"}], options=llm_calls_option
    )

    assert isinstance(result, GenerationResponse)
    assert result.response[0]["content"] == "The answer is 4"

    assert result.log is not None
    assert result.log.llm_calls is not None
    assert len(result.log.llm_calls) == 2

    # verify accumllated token usage across multiple calls
    total_tokens = sum(call.total_tokens for call in result.log.llm_calls)
    total_prompt_tokens = sum(call.prompt_tokens for call in result.log.llm_calls)
    total_completion_tokens = sum(call.completion_tokens for call in result.log.llm_calls)

    assert total_tokens == 30  # 10 + 20
    assert total_prompt_tokens == 18  # 6 + 12
    assert total_completion_tokens == 12  # 4 + 8


@pytest.mark.asyncio
async def test_token_usage_not_tracked_without_streaming(llm_calls_option):
    """Integration test verifying token usage is NOT tracked when streaming is disabled."""

    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-4",
                }
            ],
            "streaming": False,
        }
    )

    token_usage_data = [{"total_tokens": 15, "prompt_tokens": 8, "completion_tokens": 7}]

    chat = TestChat(
        config,
        llm_completions=["Hello there!"],
        streaming=False,
        token_usage=token_usage_data,
    )

    result = await chat.app.generate_async(messages=[{"role": "user", "content": "Hi!"}], options=llm_calls_option)

    assert isinstance(result, GenerationResponse)
    assert result.response[0]["content"] == "Hello there!"

    assert result.log is not None
    assert result.log.llm_calls is not None
    assert len(result.log.llm_calls) > 0

    llm_call = result.log.llm_calls[0]
    assert llm_call.total_tokens == 0
    assert llm_call.prompt_tokens == 0
    assert llm_call.completion_tokens == 0


@pytest.mark.asyncio
async def test_token_usage_not_set_for_unsupported_provider():
    """Integration test verifying token usage is NOT tracked for unsupported providers.

    Even though stream_usage=True is passed to all providers,
    providers that don't support it won't return token usage data.
    This test simulates that behavior using an 'unsupported' provider.
    """

    config = RailsConfig.from_content(
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "unsupported",
                    "model": "some-model",
                }
            ],
            "streaming": True,
        }
    )

    token_usage_data = [{"total_tokens": 15, "prompt_tokens": 8, "completion_tokens": 7}]

    chat = TestChat(
        config,
        llm_completions=["Hello there!"],
        streaming=True,
        token_usage=token_usage_data,
    )

    result = await chat.app.generate_async(messages=[{"role": "user", "content": "Hi!"}])

    assert result["content"] == "Hello there!"

    llm_stats = llm_stats_var.get()

    assert llm_stats is not None
    assert llm_stats.get_stat("total_tokens") == 0
    assert llm_stats.get_stat("total_calls") == 1
