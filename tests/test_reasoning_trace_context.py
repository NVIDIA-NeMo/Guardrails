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

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.llm.utils import get_and_clear_reasoning_trace_contextvar
from nemoguardrails.context import reasoning_trace_var
from nemoguardrails.rails.llm.llmrails import GenerationOptions, GenerationResponse
from tests.utils import TestChat


def test_get_and_clear_reasoning_trace_contextvar():
    """Test that it correctly gets and clears the trace."""
    reasoning_trace_var.set("<think> oh COT again </think>")

    result = get_and_clear_reasoning_trace_contextvar()

    assert result == "<think> oh COT again </think>"
    assert reasoning_trace_var.get() is None


def test_get_and_clear_reasoning_trace_contextvar_empty():
    """Test that it returns None when no trace exists."""
    reasoning_trace_var.set(None)

    result = get_and_clear_reasoning_trace_contextvar()

    assert result is None


@pytest.mark.asyncio
async def test_generate_async_trace_with_messages_and_options():
    """Test generate_async prepends reasoning trace when using generation options and messages."""
    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hi"
            "hello"

        define bot express greeting
            "Hello! How can I assist you today?"

        define flow
            user express greeting
            bot express greeting
        """,
        yaml_content="""
        models: []
        rails:
          output:
            apply_to_reasoning_traces: true
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "user express greeting",
            "bot express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    reasoning_trace_var.set("<think> yet another COT </think>")

    options = GenerationOptions()
    result = await chat.app.generate_async(
        messages=[{"role": "user", "content": "hi"}], options=options
    )

    assert isinstance(result, GenerationResponse)
    assert isinstance(result.response, list)
    assert len(result.response) == 1
    assert (
        result.response[0]["content"]
        == "<think> yet another COT </think>Hello! How can I assist you today?"
    )
    assert reasoning_trace_var.get() is None


@pytest.mark.asyncio
async def test_generate_async_trace_with_prompt_and_options():
    """Test generate_async prepends reasoning trace using prompt and options"""
    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hi"
            "hello"

        define bot express greeting
            "Hello! How can I assist you today?"

        define flow
            user express greeting
            bot express greeting
        """,
        yaml_content="""
        models: []
        rails:
          output:
            apply_to_reasoning_traces: true
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "user express greeting",
            "bot express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    reasoning_trace_var.set("<think> yet another COT </think>")

    options = GenerationOptions()
    result = await chat.app.generate_async(options=options, prompt="test prompt")

    assert isinstance(result, GenerationResponse)
    assert isinstance(result.response, str)
    assert (
        result.response
        == "<think> yet another COT </think>Hello! How can I assist you today?"
    )
    assert reasoning_trace_var.get() is None


@pytest.mark.asyncio
async def test_generate_async_trace_messages_only():
    """Test generate_async prepends reasoning trace when using only messages."""
    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hi"
            "hello"

        define bot express greeting
            "Hello! How can I assist you today?"

        define flow
            user express greeting
            bot express greeting
        """,
        yaml_content="""
        models: []
        rails:
          output:
            apply_to_reasoning_traces: true
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "user express greeting",
            "bot express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    reasoning_trace_var.set("<think> yet another COT </think>")

    result = await chat.app.generate_async(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(result, dict)
    assert result.get("role") == "assistant"
    assert (
        result.get("content")
        == "<think> yet another COT </think>Hello! How can I assist you today?"
    )
    assert reasoning_trace_var.get() is None


@pytest.mark.asyncio
async def test_generate_async_trace_with_prompt_only():
    """Test generate_async prepends reasoning trace when using prompt."""
    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hi"
            "hello"

        define bot express greeting
            "Hello! How can I assist you today?"

        define flow
            user express greeting
            bot express greeting
        """,
        yaml_content="""
        models: []
        rails:
          output:
            apply_to_reasoning_traces: true
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "user express greeting",
            "bot express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    reasoning_trace_var.set("<think> yet another COT </think>")

    result = await chat.app.generate_async(prompt="hi")

    assert (
        result == "<think> yet another COT </think>Hello! How can I assist you today?"
    )
    assert reasoning_trace_var.get() is None
