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

from nemoguardrails.context import explain_info_var, llm_call_info_var, llm_stats_var
from nemoguardrails.llm.call import _log_prompt, _update_token_stats
from nemoguardrails.logging.explain import ExplainInfo, LLMCallInfo
from nemoguardrails.logging.llm_tracker import track_llm_call
from nemoguardrails.logging.processing_log import compute_generation_log, processing_log_var
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.types import ChatMessage, LLMResponse, UsageInfo


def test_compute_generation_log_includes_tool_rails():
    generation_log = compute_generation_log(
        [
            {"type": "step", "flow_id": "process bot tool call", "timestamp": 0.0, "next_steps": []},
            {"type": "event", "timestamp": 1.0, "data": {"type": "StartToolOutputRail", "flow_id": "check tool call"}},
            {"type": "event", "timestamp": 1.25, "data": {"type": "ToolOutputRailFinished"}},
            {"type": "step", "flow_id": "process user tool messages", "timestamp": 2.0, "next_steps": []},
            {"type": "event", "timestamp": 3.0, "data": {"type": "StartToolInputRail", "flow_id": "check tool result"}},
            {"type": "event", "timestamp": 3.5, "data": {"type": "ToolInputRailFinished"}},
        ]
    )

    activated_rails = generation_log.activated_rails

    assert [rail.type for rail in activated_rails] == ["tool_output", "tool_input"]
    assert [rail.name for rail in activated_rails] == ["check tool call", "check tool result"]
    assert [rail.duration for rail in activated_rails] == [0.25, 0.5]


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


class TestLLMCallInfoVarLifetime:
    """Pin the lifetime contract of llm_call_info_var.

    Unlike IORails' ``_rail_llm_call_var``, which is cleared on entry and drained with a
    read-and-clear, ``llm_call_info_var`` is never reset by production code. Consumers must
    therefore set a fresh ``LLMCallInfo`` before each call and read it immediately after,
    in the same task. These tests document that contract so a consumer built on top of it
    cannot silently attribute a stale call to a rail that made none.
    """

    @pytest.mark.asyncio
    async def test_var_is_not_cleared_after_a_call_completes(self):
        """A completed call leaves its LLMCallInfo in the context."""
        llm_call_info_var.set(None)

        @track_llm_call
        async def mock_llm_call():
            return "response"

        await mock_llm_call()

        assert llm_call_info_var.get() is not None

    @pytest.mark.asyncio
    async def test_consecutive_calls_reuse_the_same_info_object(self):
        """Without a fresh set between calls, the second call mutates the first's record."""
        llm_call_info_var.set(None)

        @track_llm_call
        async def mock_llm_call():
            return "response"

        await mock_llm_call()
        first = llm_call_info_var.get()

        await mock_llm_call()
        second = llm_call_info_var.get()

        assert second is first

    @pytest.mark.asyncio
    async def test_a_stale_task_label_survives_into_an_unlabelled_call(self):
        """An unlabelled call inherits the previous call's task, misattributing it."""
        llm_call_info_var.set(LLMCallInfo(task="content_safety_check_input"))

        @track_llm_call
        async def mock_llm_call():
            return "response"

        await mock_llm_call()

        assert llm_call_info_var.get().task == "content_safety_check_input"

    @pytest.mark.asyncio
    async def test_clearing_before_a_call_prevents_misattribution(self):
        """Clearing on entry is what makes a per-call read trustworthy."""
        llm_call_info_var.set(LLMCallInfo(task="content_safety_check_input"))

        llm_call_info_var.set(None)

        @track_llm_call
        async def mock_llm_call():
            return "response"

        await mock_llm_call()

        assert llm_call_info_var.get().task is None


@pytest.mark.asyncio
async def test_log_prompt_with_chat_message_list():
    """Test that ChatMessage prompts are logged with the same labels as dict prompts."""
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    messages = [
        ChatMessage.from_dict({"role": "system", "content": "You are a helpful assistant."}),
        ChatMessage.from_dict({"role": "user", "content": "Hello"}),
        ChatMessage.from_dict({"role": "assistant", "content": "Hi there"}),
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
async def test_log_prompt_renders_chat_messages_identically_to_dicts():
    """Test that the two accepted prompt shapes produce the same logged text."""
    dict_messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
    ]

    llm_call_info_var.set(LLMCallInfo())
    _log_prompt(dict_messages)
    from_dicts = llm_call_info_var.get().prompt

    llm_call_info_var.set(LLMCallInfo())
    _log_prompt([ChatMessage.from_dict(m) for m in dict_messages])
    from_chat_messages = llm_call_info_var.get().prompt

    assert from_chat_messages == from_dicts


@pytest.mark.asyncio
async def test_log_prompt_omits_non_textual_chat_message_content():
    """Test that a multimodal ChatMessage contributes its label but no content text."""
    llm_call_info = LLMCallInfo()
    llm_call_info_var.set(llm_call_info)

    message = ChatMessage.from_dict({"role": "user", "content": "placeholder"})
    object.__setattr__(message, "content", [{"type": "text", "text": "ignored"}])

    _log_prompt([message])

    assert llm_call_info.prompt is not None
    assert "[cyan]User[/]" in llm_call_info.prompt
    assert "ignored" not in llm_call_info.prompt


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
