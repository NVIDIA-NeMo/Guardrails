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

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from nemoguardrails.actions.llm.utils import _log_prompt, _update_token_stats
from nemoguardrails.context import llm_call_info_var, llm_stats_var
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.stats import LLMStats


@pytest.mark.asyncio
async def test_token_usage_tracking_with_usage_metadata():
    """Test that token usage is tracked when usage_metadata is available."""

    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    explain_info = ExplainInfo()
    explain_info_var.set(explain_info)

    handler = LoggingCallbackHandler()

    # simulate the LLM response with usage metadata
    ai_message = AIMessage(
        content="Hello! How can I help you?",
        usage_metadata={"input_tokens": 10, "output_tokens": 6, "total_tokens": 16},
    )

    _update_token_stats(response)

    assert llm_call_info.total_tokens == 16
    assert llm_call_info.prompt_tokens == 10
    assert llm_call_info.completion_tokens == 6

    assert llm_stats.get_stat("total_tokens") == 16
    assert llm_stats.get_stat("total_prompt_tokens") == 10
    assert llm_stats.get_stat("total_completion_tokens") == 6


@pytest.mark.asyncio
async def test_token_usage_tracking_with_response_metadata_fallback():
    """Test token usage tracking with response_metadata format."""
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    response = MagicMock()
    response.usage_metadata = None
    response.response_metadata = {
        "token_usage": {
            "total_tokens": 20,
            "prompt_tokens": 12,
            "completion_tokens": 8,
        }
    }

    _update_token_stats(response)

    assert llm_call_info.total_tokens == 20
    assert llm_call_info.prompt_tokens == 12
    assert llm_call_info.completion_tokens == 8

    assert llm_stats.get_stat("total_tokens") == 20
    assert llm_stats.get_stat("total_prompt_tokens") == 12
    assert llm_stats.get_stat("total_completion_tokens") == 8


@pytest.mark.asyncio
async def test_no_token_usage_tracking_without_metadata():
    """Test that no token usage is tracked when metadata is not available."""
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    response = AIMessage(content="Hello! How can I help you?")

    _update_token_stats(response)

    # simulate LLM response without usage metadata
    ai_message = AIMessage(content="Hello! How can I help you?")
    chat_generation = ChatGeneration(message=ai_message)
    llm_result = LLMResult(generations=[[chat_generation]])

    await handler.on_llm_end(llm_result, run_id=uuid4())

    assert llm_call_info.total_tokens is None or llm_call_info.total_tokens == 0
    assert llm_call_info.prompt_tokens is None or llm_call_info.prompt_tokens == 0
    assert llm_call_info.completion_tokens is None or llm_call_info.completion_tokens == 0


@pytest.mark.asyncio
async def test_log_prompt_with_string():
    """Test that string prompts are logged correctly."""
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    _log_prompt("Hello, how are you?")

    assert llm_call_info.prompt == "Hello, how are you?"


@pytest.mark.asyncio
async def test_log_prompt_with_message_list():
    """Test that message list prompts are logged correctly."""
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    _log_prompt(messages)

    assert llm_call_info.prompt is not None
    assert "[cyan]System[/]" in llm_call_info.prompt
    assert "[cyan]User[/]" in llm_call_info.prompt
    assert "[cyan]Bot[/]" in llm_call_info.prompt
    assert "You are a helpful assistant." in llm_call_info.prompt
    assert "Hello" in llm_call_info.prompt
    assert "Hi there" in llm_call_info.prompt


@pytest.mark.asyncio
async def test_log_prompt_with_tool_message():
    """Test that tool messages are labeled correctly."""
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    messages = [
        {"role": "user", "content": "Hello"},
        {"type": "tool", "content": "Tool result"},
    ]

    _log_prompt(messages)

    assert llm_call_info.prompt is not None
    assert "[cyan]Tool[/]" in llm_call_info.prompt
