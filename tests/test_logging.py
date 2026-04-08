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

import asyncio

import pytest

from nemoguardrails.actions.llm.utils import _log_prompt, _update_token_stats
from nemoguardrails.context import explain_info_var, llm_call_info_var, llm_stats_var
from nemoguardrails.logging.explain import ExplainInfo, LLMCallInfo
from nemoguardrails.logging.llm_tracker import track_llm_call
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.types import LLMResponse, UsageInfo


@pytest.mark.asyncio
async def test_token_usage_tracking_with_usage():
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    response = LLMResponse(
        content="Hello! How can I help you?",
        usage=UsageInfo(input_tokens=10, output_tokens=6, total_tokens=16),
    )

    _update_token_stats(response)

    assert llm_call_info.total_tokens == 16
    assert llm_call_info.prompt_tokens == 10
    assert llm_call_info.completion_tokens == 6

    assert llm_stats.get_stat("total_tokens") == 16
    assert llm_stats.get_stat("total_prompt_tokens") == 10
    assert llm_stats.get_stat("total_completion_tokens") == 6


@pytest.mark.asyncio
async def test_no_token_usage_tracking_without_usage():
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    llm_stats = LLMStats()
    llm_stats_var.set(llm_stats)

    response = LLMResponse(content="Hello! How can I help you?")

    _update_token_stats(response)

    assert llm_call_info.total_tokens == 0
    assert llm_call_info.prompt_tokens == 0
    assert llm_call_info.completion_tokens == 0


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


class TestTrackLlmCallDecorator:
    @pytest.mark.asyncio
    async def test_tracks_timing_and_appends_to_processing_log(self):
        llm_call_info_var.set(None)
        llm_stats_var.set(None)
        processing_log_var.set([])

        @track_llm_call
        async def mock_llm_call():
            await asyncio.sleep(0.02)
            return "response"

        result = await mock_llm_call()

        assert result == "response"

        llm_call_info = llm_call_info_var.get()
        assert llm_call_info is not None
        assert llm_call_info.started_at is not None
        assert llm_call_info.finished_at is not None
        assert llm_call_info.duration > 0

        llm_stats = llm_stats_var.get()
        assert llm_stats.get_stat("total_calls") == 1

        processing_log = processing_log_var.get()
        assert len(processing_log) == 1
        assert processing_log[0]["type"] == "llm_call_info"

    @pytest.mark.asyncio
    async def test_appends_to_explain_info_when_present(self):
        llm_call_info_var.set(None)
        llm_stats_var.set(None)

        explain_info = ExplainInfo()
        explain_info_var.set(explain_info)

        @track_llm_call
        async def mock_llm_call():
            return "response"

        await mock_llm_call()

        assert len(explain_info.llm_calls) == 1
        assert explain_info.llm_calls[0].started_at is not None

    @pytest.mark.asyncio
    async def test_increments_total_time_stat(self):
        llm_call_info_var.set(None)
        llm_stats_var.set(None)

        @track_llm_call
        async def mock_llm_call():
            await asyncio.sleep(0.02)
            return "response"

        await mock_llm_call()

        llm_stats = llm_stats_var.get()
        assert llm_stats.get_stat("total_time") > 0
